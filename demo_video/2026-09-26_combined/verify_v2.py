import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'exports'
REVIEW = Path(__file__).resolve().parent / 'review_v2'
FINAL = OUT / 'Full_demo_combined_v2.mp4'

def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)
    ]))

info = probe(FINAL)
assert len(info['streams']) == 1
v = info['streams'][0]
assert (v['width'], v['height'], v['r_frame_rate'], int(v['nb_frames'])) == (1920, 1080, '60/1', 10774)
assert abs(float(v['duration']) - 10774/60) < 0.001
task = probe(ROOT / 'outputs/demo_edits/T2_T4_20260926_combined_v7/Task_demos_v7.mp4')
assert int(task['streams'][0]['nb_frames']) == 3668
old_timeline = json.loads((ROOT / 'outputs/demo_edits/T4_20260926_v6/timeline.json').read_text())
new_timeline = json.loads((ROOT / 'outputs/demo_edits/T4_20260926_v7/timeline.json').read_text())
assert old_timeline == new_timeline

cap = cv2.VideoCapture(str(FINAL))
def get(c, frame):
    c.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = c.read()
    assert ok, frame
    return cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)

manifest = json.loads((OUT / 'sources_v2.json').read_text())
checks = []
sheet = Image.new('RGB', (1280, 1170))
for i, part in enumerate(manifest['segments']):
    start = round(part['start_seconds'] * 60)
    source = cv2.VideoCapture(part['source'])
    expected = get(source, 0)
    actual = get(cap, start)
    error = float(np.abs(expected.astype(float) - actual.astype(float)).mean())
    assert error < 8, (part['name'], error)
    checks.append(dict(segment=part['name'], start_frame=start, image_mae=error))
    x, y = i % 2 * 640, i // 2 * 390
    sheet.paste(Image.fromarray(cv2.cvtColor(actual, cv2.COLOR_BGR2RGB)), (x, y+30))
    ImageDraw.Draw(sheet).text((x+10, y+8), f"{part['name']} @ {start/60:.3f}s", fill='white')
    source.release()
last = get(cap, 10773)
end_start = get(cap, 10654)
assert float(np.abs(last.astype(float)-end_start.astype(float)).mean()) < 1
cap.release()
sheet.save(REVIEW / 'final_sequence.jpg')
subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-xerror', '-threads', '4',
                '-i', str(FINAL), '-map', '0:v:0', '-f', 'null', '-'], check=True)
result = dict(full_decode_passed=True, timeline_unchanged=True, total_frames=10774,
              duration_seconds=10774/60, end_card_frames=120, end_card_seconds=2,
              ordered_segment_checks=checks, size_bytes=FINAL.stat().st_size)
(OUT / 'verification_v2.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
