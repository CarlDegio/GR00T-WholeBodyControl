"""Export only the approved five-second champagne title sequence."""
from pathlib import Path
import json
import math
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
COVER = BASE.parent / '2026-09-26_cover'
SOURCE = BASE / 'exports/Full_demo_combined_v4_redacted.mp4'
OUTPUT = BASE / 'exports/Intro_5s_B_champagne_bold.mp4'
PLAN = json.loads((COVER / 'edit/selection.json').read_text())
CLIPS = {c['id']: c for c in PLAN['clips']}
TRACKS = json.loads((BASE / 'privacy_redaction_v4/mosaics/source_tracks.json').read_text())
W, H, FPS, N = 1920, 1080, 60, 300
COLOR = (242, 206, 131)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
cv2.setNumThreads(2)
assert not OUTPUT.exists(), OUTPUT

title = Image.new('RGBA', (W, H))
draw = ImageDraw.Draw(title)
font = ImageFont.truetype(FONT, 66)
draw.text((W/2, 503), PLAN['title'][0], font=font, fill='white', anchor='mm')
split = round(W/2 - draw.textlength(PLAN['title'][0], font=font)/2 + draw.textlength('Beyond Arrival: ', font=font))
pixels = np.array(title)
pixels[:, split:, :3] = COLOR
title = Image.fromarray(pixels)
ImageDraw.Draw(title).text((W/2, 594), PLAN['title'][1], font=ImageFont.truetype(FONT, 62), fill='white', anchor='mm')
yy, xx = np.mgrid[0:H, 0:W]
shade = np.exp(-((yy-H/2)/210)**4) * np.exp(-((xx-W/2)/1100)**4) * .58

original = cv2.VideoCapture(str(SOURCE), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
caps = {}
for cid in sum(PLAN['grid16'], []):
    cap = cv2.VideoCapture(str(COVER / CLIPS[cid]['path']), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    assert cap.isOpened(), cid
    cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
    caps[cid] = cap

cmd = ['ffmpeg', '-hide_banner', '-v', 'warning', '-nostdin', '-n',
       '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', 'pipe:0',
       '-an', '-map_metadata', '-1', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
       '-threads', '6', '-pix_fmt', 'yuv420p', '-video_track_timescale', '15360',
       '-movflags', '+faststart', str(OUTPUT)]
samples = [0, 90, 120, 150, 180, 240, 299]
expected = {}
canvas = np.zeros((1440, 2560, 3), np.uint8)
with (HERE / 'intro_encode.log').open('w') as log:
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
    for frame_id in range(N):
        if frame_id < 120:
            ok, frame = original.read()
            assert ok, frame_id
        else:
            for row, ids in enumerate(PLAN['grid16']):
                for col, cid in enumerate(ids):
                    ok, im = caps[cid].read()
                    assert ok, (cid, frame_id)
                    canvas[row*360:(row+1)*360, col*640:(col+1)*640] = cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA)
            for row in range(1, 4):
                canvas[row*360-1:row*360+1, :] = (14, 17, 20)
            for col in range(1, 4):
                canvas[:, col*640-1:col*640+1] = (14, 17, 20)
            frame = cv2.warpAffine(canvas, np.float32([[.75, 0, 0], [0, .75, 0]]), (W, H), flags=cv2.INTER_LINEAR)
            a = min(1, (frame_id-120)/FPS)
            smooth = a*a*(3-2*a)
            frame = (frame.astype(np.float32)*(1-smooth*(shade[:, :, None]+.08))).clip(0, 255).astype(np.uint8)
            # Full tracked rectangles are applied underneath the new glyphs.
            pix = cv2.resize(cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_NEAREST)
            for row, ids in enumerate(PLAN['grid16']):
                for col, cid in enumerate(ids):
                    for track in TRACKS.get(str(cid), []):
                        box = track[frame_id]
                        if not box:
                            continue
                        x1, y1, x2, y2 = box
                        x1, y1 = max(0, math.floor(col*480+x1)-1), max(0, math.floor(row*270+y1)-1)
                        x2, y2 = min(W, math.ceil(col*480+x2)+1), min(H, math.ceil(row*270+y2)+1)
                        frame[y1:y2, x1:x2] = pix[y1:y2, x1:x2]
            layer = np.array(title.filter(ImageFilter.GaussianBlur(18*(1-smooth))) if smooth < 1 else title)
            alpha = layer[:, :, 3:4].astype(np.float32)/255*smooth
            frame = (frame*(1-alpha)+layer[:, :, :3][:, :, ::-1]*alpha).clip(0, 255).astype(np.uint8)
        if frame_id in samples:
            expected[frame_id] = frame.copy()
        enc.stdin.write(frame.tobytes())
        if frame_id % 60 == 0:
            print(f'Rendered {frame_id}/{N}', flush=True)
    enc.stdin.close()
    assert enc.wait() == 0
original.release()
for cap in caps.values():
    cap.release()

probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(OUTPUT)]))
s = probe['streams'][0]
assert len(probe['streams']) == 1
assert (int(s['nb_frames']), s['width'], s['height'], s['r_frame_rate']) == (N, W, H, '60/1')
assert abs(float(probe['format']['duration'])-5) < .001
subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-threads', '2', '-i', str(OUTPUT), '-f', 'null', '-'], check=True)
cap = cv2.VideoCapture(str(OUTPUT))
checks = []
sheet = Image.new('RGB', (1440, 4*435), '#15191E')
sheet_draw = ImageDraw.Draw(sheet)
label_font = ImageFont.truetype(FONT, 20)
for i, fid in enumerate(samples):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, frame = cap.read()
    assert ok, fid
    mae = float(np.abs(frame.astype(np.int16)-expected[fid].astype(np.int16)).mean())
    assert mae < 5, (fid, mae)
    checks.append({'frame': fid, 'encoding_mae': mae})
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    x, y = i % 2 * 720, i // 2 * 435
    sheet.paste(im.resize((720, 405)), (x, y+30))
    sheet_draw.text((x+8, y+3), f'{fid/FPS:.3f} s', font=label_font, fill='white')
    if fid == 240:
        im.save(HERE / 'Intro_B_champagne_encoded.png')
        preview = np.array(Image.open(HERE / 'B_champagne.png').convert('RGB'))
        preview_mae = float(np.abs(preview.astype(np.int16)-np.array(im).astype(np.int16)).mean())
        assert preview_mae < 5, preview_mae
cap.release()
sheet.save(HERE / 'Intro_B_champagne_timeline.jpg', quality=92)
receipt = {'output': str(OUTPUT), 'duration_seconds': 5, 'frames': N, 'resolution': [W, H], 'fps': FPS,
           'color': '#F2CE83', 'font': FONT, 'font_sizes': [66, 62], 'full_decode': 'passed',
           'approved_preview_mae_at_4s': preview_mae, 'sample_checks': checks,
           'scope': 'standalone intro only; no combined demo modified'}
(HERE / 'intro_verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2), flush=True)
