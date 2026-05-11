import json

with open(r'd:\52200034\NCKH\2025\code\testthang3\crawl\fork-of-crawlytb (2).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        keywords = ['mediapipe', 'landmark', 'holistic', 'pose', 'hand', 'joint', 'npy', 'POSE', 'HAND', 'concat']
        if any(k in src for k in keywords):
            print(f'=== CELL {i} ===')
            print(src[:3000])
            print()
