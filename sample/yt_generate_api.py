"""
yt_generate_api.py — Phiên bản generate có thể gọi như module Python
Đặt file này vào: SignSAM/sample/yt_generate_api.py

Hai cách dùng:
─────────────────────────────────────────────────────────────────────────────
1. Import trong app.py (FastAPI):
       from sample.yt_generate_api import load_model, generate_keypoints
       model_bundle = load_model(model_path, args_path)
       result = generate_keypoints(model_bundle, "xin chào", motion_length=5.0)
       # result là dict JSON-serializable

2. CLI trực tiếp (giữ nguyên workflow Kaggle):
       python -m sample.yt_generate_api \
           --model_path /kaggle/working/model000027000.pt \
           --text_prompt "xin chào" \
           --motion_length 6.0 \
           --timestep_respacing ddim100
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from numpy import array, argmin, full, inf, zeros
from math import isinf
from tqdm import tqdm

# ── NumPy / DTW helpers (copied from original, zero changes) ──────────────────

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1; j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i); q.insert(0, j)
    return array(p), array(q)


def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    assert len(x) and len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    for i in range(r):
        for j in range(c):
            if isinf(w) or (max(0, i - w) <= j <= min(c, i + w)):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j] * s, D0[i, min(j + k, c)] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


# ── Model bundle ──────────────────────────────────────────────────────────────

@dataclass
class ModelBundle:
    """Giữ tất cả state cần thiết để generate, tránh reload."""
    cfg_model:  object
    diffusion:  object
    data:       object
    device:     torch.device
    args:       object
    fps:        int = 25


# ── Unwrap DataParallel ───────────────────────────────────────────────────────

def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


# ── Load model (gọi một lần, giữ trong memory) ───────────────────────────────

def load_model(
    model_path: str,
    args_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    device_id: int = 0,
) -> ModelBundle:
    """
    Load model + diffusion vào device.
    
    Parameters
    ----------
    model_path  : đường dẫn đến model000027000.pt
    args_path   : đường dẫn đến args.json (mặc định: cùng thư mục model)
    dataset_dir : ghi đè DATASET_DIR nếu cần
    device_id   : GPU index (-1 = CPU)
    
    Returns
    -------
    ModelBundle — object chứa model đã load, dùng cho generate_keypoints()
    """
    from types import SimpleNamespace

    from utils import dist_util
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip
    from data_loaders.get_data import get_dataset_loader
    from model.cfg_sampler import ClassifierFreeSampleModel

    t0 = time.time()

    # ── device ──
    num_gpus = torch.cuda.device_count()
    if device_id >= 0 and num_gpus > 0:
        device = torch.device(f"cuda:{device_id}")
        dist_util.setup_dist(device_id)
    else:
        device = torch.device("cpu")
        dist_util.setup_dist(-1)

    # ── args ──
    if args_path is None:
        args_path = os.path.join(os.path.dirname(model_path), "args.json")
    with open(args_path) as f:
        args_dict = json.load(f)

    args_dict.update({
        "device":             device_id if device_id >= 0 else -1,
        "dataset":            "youtube_sign",
        "batch_size":         1,
        "num_samples":        1,
        "num_repetitions":    1,
        "guidance_param":     2.5,
        "motion_length":      6.0,
        "unconstrained":      False,
        "input_text":         "",
        "text_prompt":        "",
        "action_file":        "",
        "action_name":        "",
        "output_dir":         "",
        "seed":               10,
        "timestep_respacing": "ddim100",
        "skip_timesteps":     0,
    })
    args = SimpleNamespace(**args_dict)

    # ── dataset (text_only — không load full data, chỉ cần mean/std/tokenizer) ──
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=1,
        num_frames=150,
        split="test",
        hml_mode="text_only",
    )

    # ── model + diffusion ──
    model, diffusion = create_model_and_diffusion(args, data)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    clean_sd   = {k.replace("module.", ""): v for k, v in state_dict.items()}
    load_model_wo_clip(model, clean_sd)

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    cfg_model = ClassifierFreeSampleModel(unwrap_model(model))

    print(f"✅ Model loaded in {time.time()-t0:.1f}s on {device}")
    return ModelBundle(cfg_model=cfg_model, diffusion=diffusion, data=data, device=device, args=args)


# ── Generate ──────────────────────────────────────────────────────────────────

def generate_keypoints(
    bundle: ModelBundle,
    text: str,
    motion_length: float = 6.0,
    guidance: float = 2.5,
    seed: int = 10,
    ddim_steps: int = 100,
    show_progress: bool = False,
) -> dict:
    """
    Chạy DDIM sampling và trả về dict JSON-serializable.

    Parameters
    ----------
    bundle        : ModelBundle từ load_model()
    text          : văn bản tiếng Việt
    motion_length : số giây (1-20)
    guidance      : CFG scale (1.0 = không dùng guidance)
    seed          : random seed
    ddim_steps    : số bước DDIM
    show_progress : hiện tqdm progress bar

    Returns
    -------
    dict với keys: text, fps, total_frames, num_joints, keypoints
    """
    from utils.fixseed import fixseed
    from data_loaders.tensors import collate

    t0      = time.time()
    fps     = bundle.fps
    n_frames = min(500, int(motion_length * fps))
    text_clean = text.strip().lower()

    fixseed(seed)

    # Build model_kwargs
    collate_args = [{"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}]
    collate_args[0]["text"] = text_clean
    _, model_kwargs = collate(collate_args)

    # Move tensors to device
    dev = bundle.device
    for k, v in model_kwargs["y"].items():
        if isinstance(v, torch.Tensor):
            model_kwargs["y"][k] = v.to(dev)

    model_kwargs["y"]["scale"]      = torch.ones(1, device=dev) * guidance
    model_kwargs["y"]["text_embed"] = bundle.cfg_model.encode_text(model_kwargs["y"]["text"])

    # DDIM sample
    sample = bundle.diffusion.ddim_sample_loop(
        bundle.cfg_model,
        (1, bundle.cfg_model.in_channels, 1, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=show_progress,
        noise=None,
    )

    # Inverse normalise
    sample = bundle.data.dataset.t2m_dataset.inv_transform(
        sample.cpu().permute(0, 2, 3, 1)
    ).float()
    motion = sample.squeeze().cpu().numpy()  # (T, 231)

    T = motion.shape[0]
    keypoints = np.reshape(motion, (T, 77, 3)).tolist()

    elapsed = round(time.time() - t0, 2)

    return {
        "text":         text,
        "fps":          fps,
        "total_frames": T,
        "num_joints":   77,
        "keypoints":    keypoints,  # list[T][77][3]
        "elapsed_sec":  elapsed,
    }


# ── Standalone CLI entry point ────────────────────────────────────────────────

def _cli_main():
    """Giữ nguyên workflow Kaggle — chạy như script, lưu JSON ra file."""
    from utils.parser_util import generate_args
    from utils.fixseed import fixseed

    args = generate_args()
    assert args.dataset == "youtube_sign", \
        f"Script thiết kế cho youtube_sign, nhận được '{args.dataset}'"

    fixseed(args.seed)

    # Xác định output dir
    out_path = args.output_dir
    if not out_path:
        name  = os.path.basename(os.path.dirname(args.model_path))
        niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
        out_path = os.path.join(
            os.path.dirname(args.model_path),
            f"samples_{name}_{niter}_seed{args.seed}",
        )
        if args.text_prompt:
            suffix = args.text_prompt.replace(" ", "_").replace(".", "")[:50]
            out_path += f"_{suffix}"

    os.makedirs(out_path, exist_ok=True)

    # Load model
    device_id = 0 if torch.cuda.is_available() else -1
    bundle = load_model(args.model_path, device_id=device_id)

    # Text
    texts = []
    if args.text_prompt:
        texts = [args.text_prompt]
    elif args.input_text:
        with open(args.input_text) as f:
            texts = [l.strip() for l in f if l.strip()]
    else:
        raise ValueError("Cần --text_prompt hoặc --input_text")

    # Generate
    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] '{text}'")
        result = generate_keypoints(
            bundle,
            text,
            motion_length=args.motion_length,
            guidance=args.guidance_param,
            seed=args.seed,
            ddim_steps=int(getattr(args, "timestep_respacing", "ddim100").replace("ddim", "")),
            show_progress=True,
        )

        save_path = os.path.join(out_path, f"sample{i:02d}_rep00.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        print(f"  ✅ Saved {result['total_frames']} frames → {save_path} ({result['elapsed_sec']}s)")

    print(f"\n[Done] {out_path}")


if __name__ == "__main__":
    _cli_main()