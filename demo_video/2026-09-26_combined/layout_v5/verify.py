"""Verify export geometry/timing, reference previews, and the untouched tail."""
from pathlib import Path
import hashlib
import json
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

from render import BASE, SOURCE, OUTPUT, SECTIONS


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]))


def frame(cap, fid):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, img = cap.read()
    assert ok, fid
    return img


def hashes(path):
    return [line.rsplit(',', 1)[-1].strip() for line in path.read_text().splitlines()
            if line and not line.startswith('#')]


info = probe(OUTPUT)
s = info['streams'][0]
assert (s['width'], s['height'], s['r_frame_rate'], int(s['nb_frames'])) == (1920, 1080, '60/1', 10774)
assert len(info['streams']) == 1
assert abs(float(info['format']['duration']) - 10774 / 60) < .002
print('Resolution, frame count and duration verified.', flush=True)

cv2.setNumThreads(2)
cap = cv2.VideoCapture(str(OUTPUT), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
source_cap = cv2.VideoCapture(str(SOURCE), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
samples = []
sheet = Image.new('RGB', (1280, 1128), '#0b111a')
draw = ImageDraw.Draw(sheet)
for row, (name, start, count, _) in enumerate(SECTIONS):
    expected = cv2.imread(str(BASE / f'{name}_after.png'))
    actual = frame(cap, start + 180)
    mae = float(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).mean())
    assert mae < 4, (name, mae)
    samples.append(dict(section=name, frame=start + 180, preview_mae=mae))
    for col, fid in enumerate([start, start + count - 1]):
        img = frame(cap, fid)
        cv2.imwrite(str(BASE / f'verified_{fid:05d}.jpg'), img)
        thumb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((640, 360))
        sheet.paste(thumb, (col * 640, row * 376 + 16))
        draw.text((col * 640 + 8, row * 376), f'{name} | frame {fid} | {fid / 60:.3f}s', fill='white')
for fid in [0, 299, 3318, 7079, 7080, 10653, 10654, 10773]:
    actual = frame(cap, fid)
    expected = frame(source_cap, fid)
    mae = float(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).mean())
    assert mae < 4 if fid < 3318 else mae == 0, (fid, mae)
    samples.append(dict(frame=fid, original_mae=mae))
sheet.save(BASE / 'verified_contact_sheet.jpg', quality=94)
cap.release()
source_cap.release()
print('Reference previews and boundary samples verified.', flush=True)

for label, path, seek in [('output', OUTPUT, []), ('source_tail', SOURCE, ['-ss', '55.3'])]:
    with (BASE / f'{label}_decode.log').open('w') as log:
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-y', '-threads', '4',
                        *seek, '-i', str(path), '-map', '0:v:0', '-f', 'framemd5',
                        str(BASE / f'{label}.framemd5')], stderr=log, check=True)
    print(f'{label}: complete decode passed.', flush=True)
out_hashes = hashes(BASE / 'output.framemd5')
tail_hashes = hashes(BASE / 'source_tail.framemd5')
assert len(out_hashes) == 10774
assert len(tail_hashes) == 7456
assert out_hashes[3318:] == tail_hashes
receipt = dict(output=str(OUTPUT), resolution=[1920, 1080], fps=60, frames=len(out_hashes),
               duration_seconds=10774 / 60, bytes=OUTPUT.stat().st_size,
               full_decode='passed', untouched_tail_identical_frames=len(tail_hashes),
               preview_and_boundary_checks=samples,
               sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
(BASE / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
