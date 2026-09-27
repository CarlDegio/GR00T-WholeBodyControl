"""Select two continuous seconds per cover clip and compose the 25-tile outro."""
from pathlib import Path
import json
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
COVER = BASE.parent / '2026-09-26_cover'
OUT = BASE / 'exports'
REVIEW = BASE / 'review_v3'
PLAN = json.loads((COVER / 'edit/selection.json').read_text())
FPS, W, H = 60, 1920, 1080
SELECTION = {
    1: (1.5, '导航迈步和接近柜子'),
    2: (1.5, '行走接近冰箱'),
    3: (1.0, '连续迈步接近桌子'),
    4: (2.0, '持袋转向并行走'),
    5: (1.5, '环视转身动作'),
    6: (3.0, '将抓住的药瓶移入篮子'),
    7: (3.0, '持药瓶移到篮子并释放'),
    8: (3.0, '药瓶从桌面移到篮子'),
    9: (1.0, '连续向药瓶桌接近'),
    10: (0.0, '桌前最后接近'),
    11: (2.5, '抓住袋子并起身'),
    12: (0.75, '双手抬起纸箱'),
    13: (2.5, '接近纸箱和袋子'),
    14: (1.0, '袋前最后两步对齐'),
    15: (1.5, '转向冰箱旁的袋子'),
    16: (3.0, '抓碗并抬离桌面'),
    17: (0.5, '持碗倾倒'),
    18: (3.0, '持碗转向垃圾桶'),
    19: (2.0, '向碗所在桌子连续接近'),
    20: (1.5, '桌沿最后接近与对齐'),
    21: (2.0, '踩踏打开桶盖并投放'),
    22: (2.0, '近景踩踏打开桶盖'),
    23: (2.0, '正向接近桶踏板'),
    24: (2.0, '向垃圾桶连续迈步'),
    25: (2.0, '持袋接近并对齐垃圾桶'),
}

cv2.setNumThreads(2)
clips = {c['id']: c for c in PLAN['clips']}
caps = {}
records = []
for row, ids in enumerate(PLAN['grid25']):
    for col, cid in enumerate(ids):
        clip = clips[cid]
        start, reason = SELECTION[cid]
        cap = cv2.VideoCapture(str(COVER / clip['path']), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_N_THREADS, 1])
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 300
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * FPS))
        caps[cid] = cap
        records.append(dict(id=cid, path=str(COVER / clip['path']), row=row, col=col,
                            start_seconds=start, end_seconds=start+2, frames=120,
                            source_start_seconds=clip['start']+start*clip['speed'],
                            source_end_seconds=clip['start']+(start+2)*clip['speed'],
                            speed_unchanged=clip['speed'], selection_reason=reason))

layer = Image.new('RGBA', (W, H))
draw = ImageDraw.Draw(layer)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 84)
draw.text((W/2, H/2), 'Thank you for watching!', font=font, anchor='mm',
          fill=(255,255,255,255), stroke_width=2, stroke_fill=(8,14,22,180))
rgba = np.asarray(layer)
alpha = rgba[:,:,3:4].astype(np.float32)/255
yy = np.arange(H, dtype=np.float32)[:,None,None]
shade = 0.12 + 0.43*np.exp(-((yy-H/2)/155)**4)
cmd = ['ffmpeg','-hide_banner','-v','error','-nostdin','-n','-f','rawvideo',
       '-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0',
       '-an','-c:v','libx264','-preset','fast','-crf','18','-threads','4',
       '-pix_fmt','yuv420p','-video_track_timescale','15360','-movflags','+faststart',
       str(OUT/'thank_you_25grid_2s.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for frame in range(120):
    canvas = np.zeros((H,W,3), dtype=np.uint8)
    for rec in records:
        ok, image = caps[rec['id']].read()
        assert ok, (rec['id'],frame)
        x, y = rec['col']*384, rec['row']*216
        canvas[y:y+216,x:x+384] = cv2.resize(image,(384,216),interpolation=cv2.INTER_AREA)
    for x in (384,768,1152,1536): canvas[:,x-1:x+1] = (20,17,14)
    for y in (216,432,648,864): canvas[y-1:y+1,:] = (20,17,14)
    if frame == 60:
        Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)).save(REVIEW/'background_midpoint.jpg')
    canvas = canvas.astype(np.float32)*(1-shade)
    canvas = (canvas*(1-alpha)+rgba[:,:,:3][:,:,::-1]*alpha).clip(0,255).astype(np.uint8)
    proc.stdin.write(canvas.tobytes())
    if frame in (0,60,119):
        Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)).save(REVIEW/f'outro_{frame:03d}.jpg')
proc.stdin.close()
assert proc.wait() == 0
for cap in caps.values(): cap.release()
(OUT/'outro_selection_v3.json').write_text(json.dumps(dict(
    duration_seconds=2, fps=60, frames=120, grid=PLAN['grid25'], clips=records,
    title='Thank you for watching!',
),ensure_ascii=False,indent=2)+'\n')
print('Rendered all 25 selected continuous two-second clips.',flush=True)
