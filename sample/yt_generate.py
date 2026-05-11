# YouTube Sign Language Generate Script
# Adapted for 77-joint MediaPipe format (33 pose + 21 left hand + 21 right hand + 2 face)
#
# Usage:
#   python -m sample.yt_generate \
#       --model_path <path_to_model.pt> \
#       --num_repetitions 1 \
#       --guidance_param 5.5 \
#       --motion_length 5 \
#       --num_samples 10
#
# Or with text prompt:
#   python -m sample.yt_generate \
#       --model_path <path_to_model.pt> \
#       --guidance_param 5.5 \
#       --motion_length 5 \
#       --text_prompt "a person signing hello"

from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
import shutil
from data_loaders.tensors import collate
import math
import cv2
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
import torch.nn as nn
from model.cfg_sampler import ClassifierFreeSampleModel

from numpy import zeros, array, argmin, inf, full
from math import isinf
from tqdm import tqdm


def unwrap_model(model):
    """Unwrap DataParallel to get the raw model, works for both 1 and multi-GPU."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


# ==============================================================================
# Constants for 77-joint MediaPipe layout
# ==============================================================================
NUM_JOINTS = 77
DIM_POSE = NUM_JOINTS * 3  # 231

# Joint index ranges
POSE_START, POSE_END = 0, 33       # MediaPipe Pose: 33 landmarks
LHAND_START, LHAND_END = 33, 54    # Left hand: 21 landmarks
RHAND_START, RHAND_END = 54, 75    # Right hand: 21 landmarks
FACE_START, FACE_END = 75, 77      # Extra face landmarks: 2

# MediaPipe Pose landmark names (indices 0-32)
# 0:  nose               11: left_shoulder     23: left_hip
# 1:  left_eye_inner     12: right_shoulder    24: right_hip
# 2:  left_eye           13: left_elbow        25: left_knee
# 3:  left_eye_outer     14: right_elbow       26: right_knee
# 4:  right_eye_inner    15: left_wrist        27: left_ankle
# 5:  right_eye          16: right_wrist       28: right_ankle
# 6:  right_eye_outer    17: left_pinky        29: left_heel
# 7:  left_ear           18: right_pinky       30: right_heel
# 8:  right_ear          19: left_index        31: left_foot_index
# 9:  mouth_left         20: right_index       32: right_foot_index
# 10: mouth_right        21: left_thumb
#                        22: right_thumb

# MediaPipe Hand landmark names (0-20, applied to both left and right hand)
# 0: wrist        5: index_mcp    9:  middle_mcp   13: ring_mcp    17: pinky_mcp
# 1: thumb_cmc    6: index_pip    10: middle_pip   14: ring_pip    18: pinky_pip
# 2: thumb_mcp    7: index_dip    11: middle_dip   15: ring_dip    19: pinky_dip
# 3: thumb_ip     8: index_tip    12: middle_tip   16: ring_tip    20: pinky_tip
# 4: thumb_tip


# ==============================================================================
# Color palette
# ==============================================================================
COLOR_FACE      = (0, 180, 0)       # Green
COLOR_MOUTH     = (0, 220, 100)     # Light green
COLOR_TORSO     = (80, 80, 80)      # Dark gray
COLOR_LEFT_ARM  = (255, 128, 0)     # Orange
COLOR_RIGHT_ARM = (0, 128, 255)     # Blue
COLOR_LEFT_LEG  = (180, 100, 50)    # Brown
COLOR_RIGHT_LEG = (50, 100, 180)    # Steel blue
COLOR_EXTRA     = (150, 150, 150)   # Light gray

# Hand finger colors (thumb, index, middle, ring, pinky)
FINGER_COLORS_LEFT = [
    (255, 80, 80),    # Thumb  - Red
    (255, 165, 0),    # Index  - Orange
    (255, 255, 0),    # Middle - Yellow
    (0, 255, 128),    # Ring   - Green
    (0, 200, 255),    # Pinky  - Cyan
]
FINGER_COLORS_RIGHT = [
    (180, 0, 0),      # Thumb  - Dark red
    (200, 120, 0),    # Index  - Dark orange
    (200, 200, 0),    # Middle - Dark yellow
    (0, 180, 80),     # Ring   - Dark green
    (0, 150, 200),    # Pinky  - Dark cyan
]
COLOR_LHAND_PALM = (255, 100, 50)   # Orange-red
COLOR_RHAND_PALM = (50, 100, 255)   # Blue


# ==============================================================================
# Skeleton structure for MediaPipe 77 joints
# ==============================================================================
def getSkeletalModelStructure():
    """
    Returns skeleton connectivity as list of (start_joint, end_joint, color).
    Designed for 77-joint MediaPipe Holistic layout:
      [0-32]  = Pose (33 landmarks)
      [33-53] = Left Hand (21 landmarks)
      [54-74] = Right Hand (21 landmarks)
      [75-76] = Extra face reference (2 landmarks)
    """
    bones = []

    # ---- FACE connections (Pose joints 0-10) ----
    # Nose → left eye
    bones.append((0, 1, COLOR_FACE))     # nose → left_eye_inner
    bones.append((1, 2, COLOR_FACE))     # left_eye_inner → left_eye
    bones.append((2, 3, COLOR_FACE))     # left_eye → left_eye_outer
    bones.append((3, 7, COLOR_FACE))     # left_eye_outer → left_ear

    # Nose → right eye
    bones.append((0, 4, COLOR_FACE))     # nose → right_eye_inner
    bones.append((4, 5, COLOR_FACE))     # right_eye_inner → right_eye
    bones.append((5, 6, COLOR_FACE))     # right_eye → right_eye_outer
    bones.append((6, 8, COLOR_FACE))     # right_eye_outer → right_ear

    # Mouth
    bones.append((9, 10, COLOR_MOUTH))   # mouth_left → mouth_right

    # ---- TORSO connections ----
    bones.append((11, 12, COLOR_TORSO))  # left_shoulder → right_shoulder
    bones.append((11, 23, COLOR_TORSO))  # left_shoulder → left_hip
    bones.append((12, 24, COLOR_TORSO))  # right_shoulder → right_hip
    bones.append((23, 24, COLOR_TORSO))  # left_hip → right_hip

    # Nose to mid-shoulder (visual reference)
    # We draw nose → left_shoulder and nose → right_shoulder for neck approximation
    bones.append((0, 11, COLOR_TORSO))
    bones.append((0, 12, COLOR_TORSO))

    # ---- LEFT ARM ----
    bones.append((11, 13, COLOR_LEFT_ARM))   # left_shoulder → left_elbow
    bones.append((13, 15, COLOR_LEFT_ARM))   # left_elbow → left_wrist

    # ---- RIGHT ARM ----
    bones.append((12, 14, COLOR_RIGHT_ARM))  # right_shoulder → right_elbow
    bones.append((14, 16, COLOR_RIGHT_ARM))  # right_elbow → right_wrist

    # ---- LEGS ----
    bones.append((23, 25, COLOR_LEFT_LEG))   # left_hip → left_knee
    bones.append((25, 27, COLOR_LEFT_LEG))   # left_knee → left_ankle
    bones.append((27, 29, COLOR_LEFT_LEG))   # left_ankle → left_heel
    bones.append((27, 31, COLOR_LEFT_LEG))   # left_ankle → left_foot_index
    bones.append((29, 31, COLOR_LEFT_LEG))   # left_heel → left_foot_index

    bones.append((24, 26, COLOR_RIGHT_LEG))  # right_hip → right_knee
    bones.append((26, 28, COLOR_RIGHT_LEG))  # right_knee → right_ankle
    bones.append((28, 30, COLOR_RIGHT_LEG))  # right_ankle → right_heel
    bones.append((28, 32, COLOR_RIGHT_LEG))  # right_ankle → right_foot_index
    bones.append((30, 32, COLOR_RIGHT_LEG))  # right_heel → right_foot_index

    # ---- Wrist-to-Hand bridge ----
    # Connect pose wrist to hand wrist (may overlap spatially)
    bones.append((15, LHAND_START + 0, COLOR_LHAND_PALM))   # left_wrist(pose) → left_hand_wrist
    bones.append((16, RHAND_START + 0, COLOR_RHAND_PALM))   # right_wrist(pose) → right_hand_wrist

    # ---- LEFT HAND (joints 33-53) ----
    lh = LHAND_START  # 33
    # Palm connections (wrist to each finger MCP)
    bones.append((lh + 0, lh + 1, COLOR_LHAND_PALM))   # wrist → thumb_cmc
    bones.append((lh + 0, lh + 5, COLOR_LHAND_PALM))   # wrist → index_mcp
    bones.append((lh + 0, lh + 9, COLOR_LHAND_PALM))   # wrist → middle_mcp
    bones.append((lh + 0, lh + 13, COLOR_LHAND_PALM))  # wrist → ring_mcp
    bones.append((lh + 0, lh + 17, COLOR_LHAND_PALM))  # wrist → pinky_mcp
    # Palm cross-links
    bones.append((lh + 5, lh + 9, COLOR_LHAND_PALM))   # index_mcp → middle_mcp
    bones.append((lh + 9, lh + 13, COLOR_LHAND_PALM))  # middle_mcp → ring_mcp
    bones.append((lh + 13, lh + 17, COLOR_LHAND_PALM)) # ring_mcp → pinky_mcp

    # Thumb: cmc → mcp → ip → tip
    for i, idx in enumerate([1, 2, 3]):
        bones.append((lh + idx, lh + idx + 1, FINGER_COLORS_LEFT[0]))
    # Index: mcp → pip → dip → tip
    for i, idx in enumerate([5, 6, 7]):
        bones.append((lh + idx, lh + idx + 1, FINGER_COLORS_LEFT[1]))
    # Middle: mcp → pip → dip → tip
    for i, idx in enumerate([9, 10, 11]):
        bones.append((lh + idx, lh + idx + 1, FINGER_COLORS_LEFT[2]))
    # Ring: mcp → pip → dip → tip
    for i, idx in enumerate([13, 14, 15]):
        bones.append((lh + idx, lh + idx + 1, FINGER_COLORS_LEFT[3]))
    # Pinky: mcp → pip → dip → tip
    for i, idx in enumerate([17, 18, 19]):
        bones.append((lh + idx, lh + idx + 1, FINGER_COLORS_LEFT[4]))

    # ---- RIGHT HAND (joints 54-74) ----
    rh = RHAND_START  # 54
    # Palm connections
    bones.append((rh + 0, rh + 1, COLOR_RHAND_PALM))
    bones.append((rh + 0, rh + 5, COLOR_RHAND_PALM))
    bones.append((rh + 0, rh + 9, COLOR_RHAND_PALM))
    bones.append((rh + 0, rh + 13, COLOR_RHAND_PALM))
    bones.append((rh + 0, rh + 17, COLOR_RHAND_PALM))
    # Palm cross-links
    bones.append((rh + 5, rh + 9, COLOR_RHAND_PALM))
    bones.append((rh + 9, rh + 13, COLOR_RHAND_PALM))
    bones.append((rh + 13, rh + 17, COLOR_RHAND_PALM))

    # Thumb
    for i, idx in enumerate([1, 2, 3]):
        bones.append((rh + idx, rh + idx + 1, FINGER_COLORS_RIGHT[0]))
    # Index
    for i, idx in enumerate([5, 6, 7]):
        bones.append((rh + idx, rh + idx + 1, FINGER_COLORS_RIGHT[1]))
    # Middle
    for i, idx in enumerate([9, 10, 11]):
        bones.append((rh + idx, rh + idx + 1, FINGER_COLORS_RIGHT[2]))
    # Ring
    for i, idx in enumerate([13, 14, 15]):
        bones.append((rh + idx, rh + idx + 1, FINGER_COLORS_RIGHT[3]))
    # Pinky
    for i, idx in enumerate([17, 18, 19]):
        bones.append((rh + idx, rh + idx + 1, FINGER_COLORS_RIGHT[4]))

    # ---- EXTRA FACE (joints 75-76) ----
    # Connect extra face landmarks to nose (joint 0), if present
    bones.append((0, 75, COLOR_EXTRA))
    bones.append((0, 76, COLOR_EXTRA))

    return bones


# ==============================================================================
# Utility functions
# ==============================================================================

def process_sentence(text):
    text = text.lower()
    return text.strip()


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


# ==============================================================================
# DTW functions
# ==============================================================================

def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size
    :param float s: weight applied on off-diagonal moves
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
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
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def avg_frames(frames):
    frames_sum = np.zeros_like(frames[0])
    for frame in frames:
        frames_sum += frame
    avg_frame = frames_sum / len(frames)
    return avg_frame


def alter_DTW_timing(pred_seq, ref_seq, transform_pred=True):
    """Apply DTW to the produced sequence for visual comparison with reference."""
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    d, cost_matrix, acc_cost_matrix, path = dtw(ref_seq, pred_seq, dist=euclidean_norm)
    d = d / acc_cost_matrix.shape[0]

    new_pred_seq = np.zeros_like(ref_seq)

    if transform_pred:
        j = 0
        skips = 0
        squeeze_frames = []
        for (i, pred_num) in enumerate(path[0]):
            if i == len(path[0]) - 1:
                break
            if path[1][i] == path[1][i + 1]:
                skips += 1
            if path[0][i] == path[0][i + 1]:
                squeeze_frames.append(pred_seq[i - skips])
                j += 1
            elif path[0][i] == path[0][i - 1]:
                new_pred_seq[pred_num] = avg_frames(squeeze_frames)
                squeeze_frames = []
            else:
                new_pred_seq[pred_num] = pred_seq[i - skips]

    return new_pred_seq, ref_seq, d


# ==============================================================================
# Visualization functions
# ==============================================================================

def draw_line(im, joint1, joint2, c=(0, 0, 255), t=1, width=3):
    """Draw an elliptical line between two 2D joint positions."""
    thresh = -100
    if joint1[0] > thresh and joint1[1] > thresh and joint2[0] > thresh and joint2[1] > thresh:
        center = (int((joint1[0] + joint2[0]) / 2), int((joint1[1] + joint2[1]) / 2))
        length = int(math.sqrt(((joint1[0] - joint2[0]) ** 2) + ((joint1[1] - joint2[1]) ** 2)) / 2)
        angle = math.degrees(math.atan2((joint1[0] - joint2[0]), (joint1[1] - joint2[1])))
        cv2.ellipse(im, center, (width, length), -angle, 0.0, 360.0, c, -1)


def draw_joint_dot(im, joint, c=(0, 0, 0), radius=2):
    """Draw a small circle at a joint position."""
    thresh = -100
    if joint[0] > thresh and joint[1] > thresh:
        cv2.circle(im, (int(joint[0]), int(joint[1])), radius, c, -1)


def put_text_unicode(frame, text, position, color=(0, 0, 255), font_size=20):
    """
    Render Unicode text (Vietnamese) on an OpenCV frame.
    Uses PIL if available, falls back to OpenCV putText.
    """
    if HAS_PIL:
        # Convert BGR (OpenCV) → RGB (PIL)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            # Try common fonts that support Vietnamese
            for font_name in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
            ]:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except (IOError, OSError):
                    continue
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # PIL uses RGB color order
        rgb_color = (color[2], color[1], color[0]) if len(color) == 3 else color
        draw.text(position, text, font=font, fill=rgb_color)
        # Convert RGB (PIL) → BGR (OpenCV)
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        # Fallback: OpenCV (no Unicode support)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_size / 30.0, color, 1)
    return frame


def draw_frame_2D(frame, joints_2d, canvas_size=650):
    """
    Draw a full skeleton frame with 77 MediaPipe joints.
    Auto-scales skeleton to fit ~70% of the canvas.
    joints_2d: (NUM_JOINTS, 2) array of 2D positions
    """
    # Separator line
    draw_line(frame, [1, canvas_size], [1, 1], c=(0, 0, 0), t=1, width=1)

    # --- Auto-scale: normalize joints to [0,1] then scale to canvas ---
    valid_mask = np.all(np.abs(joints_2d) > 1e-6, axis=1)  # filter zero joints
    if valid_mask.sum() < 2:
        return  # skip if too few valid joints

    valid_joints = joints_2d[valid_mask]
    j_min = valid_joints.min(axis=0)
    j_max = valid_joints.max(axis=0)
    j_range = j_max - j_min
    j_range[j_range < 1e-6] = 1.0  # prevent division by zero

    # Normalize to [0, 1]
    joints_norm = (joints_2d - j_min) / j_range

    # Scale to fit 70% of canvas, centered
    margin = canvas_size * 0.15
    usable = canvas_size - 2 * margin
    # Maintain aspect ratio
    scale = usable / max(j_range[0], j_range[1]) * max(j_range[0], j_range[1]) / j_range
    scale_uniform = min(usable / j_range[0], usable / j_range[1])
    joints = joints_2d * scale_uniform
    # Center
    scaled_min = valid_joints.min(axis=0) * scale_uniform
    scaled_max = valid_joints.max(axis=0) * scale_uniform
    offset_x = (canvas_size - (scaled_max[0] - scaled_min[0])) / 2 - scaled_min[0]
    offset_y = (canvas_size - (scaled_max[1] - scaled_min[1])) / 2 - scaled_min[1]
    # Shift up a bit to leave room for text at bottom
    offset_y -= 30
    joints = joints + np.array([offset_x, offset_y])

    # Get skeleton bones
    skeleton = getSkeletalModelStructure()

    # --- Draw all bones (thicker for pose, thinner for hands) ---
    for (start_idx, end_idx, color) in skeleton:
        if start_idx < len(joints) and end_idx < len(joints):
            # Pose bones: width 2, Hand bones: width 1
            bone_width = 2 if (start_idx < LHAND_START and end_idx < LHAND_START) else 1
            draw_line(frame,
                      [joints[start_idx][0], joints[start_idx][1]],
                      [joints[end_idx][0], joints[end_idx][1]],
                      c=color, t=1, width=bone_width)

    # --- Draw joint dots ---
    # Pose joints (bigger dots for important joints)
    key_pose_joints = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # nose, shoulders, elbows, wrists, hips
    for idx in key_pose_joints:
        if idx < len(joints):
            draw_joint_dot(frame, joints[idx], c=(0, 0, 0), radius=4)

    # All hand joints (small dots)
    for idx in range(LHAND_START, LHAND_END):
        if idx < len(joints):
            draw_joint_dot(frame, joints[idx], c=(200, 50, 0), radius=2)
    for idx in range(RHAND_START, RHAND_END):
        if idx < len(joints):
            draw_joint_dot(frame, joints[idx], c=(0, 50, 200), radius=2)

    # Hand fingertips (larger dots)
    fingertip_offsets = [4, 8, 12, 16, 20]
    for tip_off in fingertip_offsets:
        lh_idx = LHAND_START + tip_off
        rh_idx = RHAND_START + tip_off
        if lh_idx < len(joints):
            draw_joint_dot(frame, joints[lh_idx], c=(255, 0, 0), radius=3)
        if rh_idx < len(joints):
            draw_joint_dot(frame, joints[rh_idx], c=(0, 0, 255), radius=3)

    # Face joints
    for idx in range(FACE_START, FACE_END):
        if idx < len(joints):
            draw_joint_dot(frame, joints[idx], c=(0, 200, 0), radius=2)


def plot_video(joints, file_path, video_name, references=None, skip_frames=1, sequence_ID=None):
    """
    Generate an MP4 video visualizing the predicted skeleton (and optionally ground truth).

    :param joints: (T, 231) array — predicted motion
    :param file_path: output directory
    :param video_name: output file name
    :param references: (T, 231) array — ground truth motion (optional)
    :param skip_frames: frame skip factor
    :param sequence_ID: caption text to overlay
    """
    FPS = (25 // skip_frames)
    video_file = os.path.join(file_path, "{}.mp4".format(video_name.split(".")[0]))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if references is None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (650, 650), True)
    else:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (1300, 650), True)

    for j_idx, frame_joints in enumerate(joints):
        # ---- Predicted frame ----
        frame = np.ones((650, 650, 3), np.uint8) * 255

        # Reshape flat vector (231,) → (77, 3) → take (x, y) for 2D
        frame_joints_2d = np.reshape(frame_joints, (NUM_JOINTS, 3))[:, :2]
        draw_frame_2D(frame, frame_joints_2d)

        # Overlay text (with Unicode support for Vietnamese)
        if sequence_ID is not None:
            seq_text = "Predicted : " + str(sequence_ID).split("/")[-1]
            if len(seq_text) > 80:
                seq_text = seq_text[:77] + "..."
            frame = put_text_unicode(frame, seq_text, (20, 610), color=(220, 30, 30), font_size=18)

        # ---- Ground truth frame ----
        if references is not None and j_idx < len(references):
            ref_joints = references[j_idx]
            ref_frame = np.ones((650, 650, 3), np.uint8) * 255

            ref_joints_2d = np.reshape(ref_joints, (NUM_JOINTS, 3))[:, :2]
            draw_frame_2D(ref_frame, ref_joints_2d)

            ref_frame = put_text_unicode(ref_frame, "Ground Truth", (250, 610), color=(0, 0, 0), font_size=22)

            frame = np.concatenate((frame, ref_frame), axis=1)

        video.write(frame)

    video.release()
    print(f"  Video saved: {video_file}")


# ==============================================================================
# Template helpers
# ==============================================================================

def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}\" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}\" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames, split):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split=split,
                              hml_mode='text_only')
    return data


# ==============================================================================
# Training-time evaluation entry point (called from training_loop.py)
# ==============================================================================

def main(args, data, model, diffusion, epoch):
    """
    Evaluate DTW on validation set and save sample videos.
    Called from training_loop.py during training.
    Works with both single-GPU and DataParallel (multi-GPU) models.
    """
    model.eval()

    save_dtw = f"{args.save_dir}/eval_epoch_{epoch}"
    os.makedirs(save_dtw, exist_ok=True)

    fixseed(args.seed)

    # Unwrap DataParallel before wrapping with CFG sampler
    raw_model = unwrap_model(model)
    cfg_model = ClassifierFreeSampleModel(raw_model)

    all_dtws = []

    for k, (ground_truth, model_kwargs) in enumerate(data):

        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * 2.5

        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(
            cfg_model,
            (args.batch_size, cfg_model.in_channels, 1, ground_truth.shape[-1]),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            noise=None,
        )

        # Inverse Z-normalization
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        sample = sample.squeeze(1)

        texts = model_kwargs['y']['text']

        ground_truth = data.dataset.t2m_dataset.inv_transform(ground_truth.cpu().permute(0, 2, 3, 1)).float()
        ground_truth = ground_truth.squeeze(1)

        lengths = model_kwargs['y']['lengths'].to(dist_util.dev())
        texts = model_kwargs['y']['text']
        model_kwargs['y']['text_embed'] = cfg_model.encode_text(model_kwargs['y']['text'])

        batch_dtw = []
        for i, (caption, motion, gt_len, gt_motion) in enumerate(zip(texts, sample, lengths, ground_truth)):
            hyp_l = gt_len
            ref_l = gt_len
            motion = motion[:hyp_l]
            gt_motion = gt_motion[:ref_l]

            motion = motion.cpu().numpy()
            gt_motion = gt_motion.cpu().numpy()
            _, _, dis_dtw = alter_DTW_timing(motion, gt_motion, transform_pred=False)
            batch_dtw.append(dis_dtw)

            if i == 0:
                appr_dtw = round(dis_dtw, 3)
                measure_info = f"[DTW]={appr_dtw}"
                plot_video(joints=motion,
                           file_path=save_dtw,
                           video_name=f"sample_{k}_{i}",
                           references=gt_motion,
                           skip_frames=1,
                           sequence_ID=f"{measure_info}-{caption}")

        m_dtw = np.array(batch_dtw).mean()
        print(f"DTW BATCH | {m_dtw}")
        all_dtws.append(m_dtw)

    dtw_final = np.array(all_dtws).mean()
    return dtw_final, 0


# ==============================================================================
# Standalone generation entry point
# ==============================================================================

def done_main():
    """
    Standalone generation: load model, generate samples, visualize and save.
    Uses DDIM sampling for faster generation.
    Supports --timestep_respacing (e.g. 'ddim100') and --skip_timesteps.
    """
    import time
    split = "test"

    args = generate_args()

    assert args.dataset == 'youtube_sign', \
        f"This script is designed for youtube_sign dataset, got '{args.dataset}'"

    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 500
    fps = 25
    n_frames = max_frames = min(max_frames, int(args.motion_length * fps))

    # ============ GPU Detection ============
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"  YOUTUBE SIGN DDIM GENERATOR")
    print(f"{'='*60}")
    if num_gpus > 0:
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} | {gpu_mem:.1f} GB")
    else:
        print("  [WARNING] No GPU detected! Running on CPU.")
    print(f"  Frames: {n_frames} | FPS: {fps}")
    respacing = getattr(args, 'timestep_respacing', '')
    skip = getattr(args, 'skip_timesteps', 0)
    print(f"  Sampling: DDIM | Respacing: {respacing if respacing else 'full (1000 steps)'} | Skip: {skip}")
    print(f"  Guidance: {args.guidance_param}")
    print(f"{'='*60}\n")

    if num_gpus > 0:
        args.device = 0
    else:
        args.device = -1

    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)

    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')[:50]
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')[:50]

    # Load text prompts
    if args.text_prompt != '':
        raw_texts = [args.text_prompt]
        texts = [process_sentence(text) for text in raw_texts]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    args.batch_size = args.num_samples

    print('Loading dataset...')
    if is_using_data:
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size,
                                  num_frames=n_frames, split=split)
    else:
        data = load_dataset(args, max_frames, n_frames, split=split)

    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu', weights_only=False)
    # Strip 'module.' prefix if saved from DataParallel training
    cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    load_model_wo_clip(model, cleaned_state_dict)

    # Optionally use multi-GPU for inference
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"*** Using {num_gpus} GPUs for inference (DataParallel)")
        model = nn.DataParallel(model)

    model.to(dist_util.dev())
    model.eval()
    model.requires_grad_(False)

    # Wrap with CFG sampler (unwrap DataParallel first if needed)
    raw_model = unwrap_model(model)
    if args.guidance_param != 1:
        print(f"*** Using Classifier-Free Guidance with scale={args.guidance_param}")
        cfg_model = ClassifierFreeSampleModel(raw_model)
    else:
        print(f"*** No guidance (param={args.guidance_param})")
        cfg_model = raw_model

    all_gt_motions = []

    if is_using_data:
        iterator = iter(data)
        ground_truth, model_kwargs = next(iterator)
        n_frames = ground_truth.shape[-1]

        ground_truth = data.dataset.t2m_dataset.inv_transform(
            ground_truth.cpu().permute(0, 2, 3, 1)).float()
        ground_truth = ground_truth.squeeze(1)

        all_gt_motions.append(ground_truth)
        gt_lengths = model_kwargs["y"]["lengths"].cpu().numpy()
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)

    all_motions = []
    all_lengths = []
    all_text = []

    for rep_i in range(args.num_repetitions):
        t_start = time.time()
        print(f'### Sampling [repetition #{rep_i}]')

        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        model_kwargs['y']['text_embed'] = cfg_model.encode_text(model_kwargs['y']['text'])
        sample_fn = diffusion.ddim_sample_loop

        sample = sample_fn(
            cfg_model,
            (args.batch_size, cfg_model.in_channels, 1, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=getattr(args, 'skip_timesteps', 0),
            init_image=None,
            progress=True,
            noise=None,
        )

        sample = data.dataset.t2m_dataset.inv_transform(
            sample.cpu().permute(0, 2, 3, 1)).float()
        sample = sample.squeeze(1)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text'
            all_text += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        t_elapsed = time.time() - t_start
        print(f'### Repetition #{rep_i} done in {t_elapsed:.1f}s')

    all_motions = np.concatenate(all_motions, axis=0)[:total_num_samples]

    try:
        all_gt_motions = np.concatenate(all_gt_motions, axis=0)[:total_num_samples]
    except:
        all_gt_motions = None

    all_text = all_text[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    print(f"Saving visualizations to [{out_path}]...")

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i * args.batch_size + sample_i]
            motion = all_motions[rep_i * args.batch_size + sample_i]
            save_file = sample_file_template.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)

            if all_gt_motions is not None:
                gt_motion = all_gt_motions[sample_i]
                gt_len = gt_lengths[sample_i]
                gt_motion = gt_motion[:gt_len]

                _, _, d = alter_DTW_timing(motion / 3, gt_motion / 3, transform_pred=False)
                print(f"  DTW of sample {sample_i} rep {rep_i} = {d:.4f}")

                motion = motion[:gt_len]
                plot_video(joints=motion,
                           file_path=out_path,
                           video_name=save_file,
                           references=gt_motion,
                           skip_frames=1,
                           sequence_ID=f"DTW={d:.2f} | {caption}")
            else:
                plot_video(joints=motion,
                           file_path=out_path,
                           video_name=save_file,
                           references=None,
                           skip_frames=1,
                           sequence_ID=caption)

            print(f"  [{caption} | {animation_save_path}]")

    abs_path = os.path.abspath(out_path)
    print(f'\n[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    done_main()
