"""Check full-height footage, every HUD frame, and untouched sections against v7."""
import hashlib
import json
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

from render import BASE, SOURCE, OUTPUT, v7
TASKS, states = v7.TASKS, v7.states


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def hashes(path):
    return [line.rsplit(',', 1)[-1].strip() for line in path.read_text().splitlines()
            if line and not line.startswith('#')]


def get_frame(cap, fid):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, img = cap.read()
    assert ok, fid
    return img


info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(OUTPUT)]))
s = info['streams'][0]
assert (s['width'], s['height'], s['r_frame_rate'], int(s['nb_frames'])) == (1920, 1080, '60/1', 10774)
assert len(info['streams']) == 1
assert abs(float(info['format']['duration']) - 10774 / 60) < .002
assert sha(SOURCE) == json.loads((BASE.parent / 'style_v7/verification.json').read_text())['sha256']
print('Export metadata and v7 source identity verified.', flush=True)

cv2.setNumThreads(2)
cap = cv2.VideoCapture(str(OUTPUT), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
src = cv2.VideoCapture(str(SOURCE), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
samples = []
for task, t in [('t2', 57), ('t2', 90), ('t2', 110), ('t4', 119), ('t4', 150), ('t4', 172)]:
    frame = get_frame(cap, round(t * 60))
    ref = cv2.imread(str(BASE / f'{task}_after_{t}.png'))
    mae = float(np.abs(frame.astype(np.int16) - ref.astype(np.int16)).mean())
    assert mae < 4, (task, t, mae)
    samples.append(dict(task=task, time_seconds=t, preview_mae=mae))
for fid in [1412, 1592, 2717]:
    actual, expected = get_frame(cap, fid), get_frame(src, fid)
    mae = float(np.abs(actual[74:].astype(np.int16) - expected[74:].astype(np.int16)).mean())
    assert mae < 4, (fid, mae)
    samples.append(dict(frame=fid, gate_picture_mae=mae))
for fid in [10654, 10773]:
    actual, expected = get_frame(cap, fid), get_frame(src, fid)
    mae = float(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).mean())
    assert mae < 4
    samples.append(dict(frame=fid, outro_mae=mae))

hud_checks = []
for task, conf in TASKS.items():
    timeline = states(task)
    cap.set(cv2.CAP_PROP_POS_FRAMES, conf['start'])
    cache, errors = {}, []
    for i, (speed, raw) in enumerate(timeline):
        ok, actual = cap.read()
        assert ok, (task, i)
        key = hashlib.sha256((str(speed) + raw + 'True').encode()).hexdigest()[:16]
        if key not in cache:
            hud_path = BASE / f'{task}_hud' / f'{key}.png'
            rgba = cv2.imread(str(hud_path), cv2.IMREAD_UNCHANGED)
            assert not rgba[:, :1428, 3].any(), 'Left viewport must contain no text or overlays'
            hud = rgba[:, :, :3]
            cache[key] = (hud[134:161:2, 1815:1890:2].astype(np.int16), hud[806:1078:3, 1445:1905:3].astype(np.int16))
        badge, todo = cache[key]
        badge_error = float(np.abs(actual[134:161:2, 1815:1890:2].astype(np.int16) - badge).mean())
        todo_error = float(np.abs(actual[806:1078:3, 1445:1905:3].astype(np.int16) - todo).mean())
        errors.append((badge_error, todo_error))
        assert badge_error < 3 and todo_error < 3, (task, i, speed, badge_error, todo_error)
    maxima = np.max(errors, axis=0).tolist()
    hud_checks.append(dict(task=task, checked_frames=len(timeline), max_speed_badge_mae=maxima[0], max_todo_mae=maxima[1]))
    print(f'{task}: all {len(timeline)} speed badges and to-do overlays checked.', flush=True)

sheet = Image.new('RGB', (1280, 1128), '#0b111a')
d = ImageDraw.Draw(sheet)
for row, pair in enumerate([(1412, 2717), (3318, 7079), (7080, 10653)]):
    for col, fid in enumerate(pair):
        img = get_frame(cap, fid)
        cv2.imwrite(str(BASE / f'verified_{fid:05d}.jpg'), img)
        sheet.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((640, 360)), (col * 640, row * 376 + 16))
        d.text((col * 640 + 8, row * 376), f'Frame {fid} | {fid / 60:.3f}s', fill='white')
sheet.save(BASE / 'verified_contact_sheet.jpg', quality=94)
cap.release()
src.release()

with (BASE / 'full_decode.log').open('w') as log:
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-y', '-threads', '4', '-i', str(OUTPUT),
                    '-map', '0:v:0', '-f', 'framemd5', str(BASE / 'output.framemd5')], stderr=log, check=True)
new = hashes(BASE / 'output.framemd5')
old = hashes(BASE.parent / 'style_v7/output.framemd5')
assert len(new) == 10774
assert new[:3318] == old[:3318]
assert new[10654:] == old[10654:]
receipt = dict(output=str(OUTPUT), source=str(SOURCE), resolution=[1920, 1080], fps=60, frames=10774,
               duration_seconds=10774 / 60, full_decode='passed', unchanged_identical_frames=3438,
               hud_checks=hud_checks, sample_checks=samples, bytes=OUTPUT.stat().st_size, sha256=sha(OUTPUT))
(BASE / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
