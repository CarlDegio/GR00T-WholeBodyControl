"""Render title design options from the existing cover composition at 4 seconds."""
from pathlib import Path
import json
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
COVER = BASE.parent / '2026-09-26_cover'
PLAN = json.loads((COVER / 'edit/selection.json').read_text())
CLIPS = {c['id']: c for c in PLAN['clips']}
W, H, FRAME = 1920, 1080, 240
cv2.setNumThreads(2)
canvas = np.zeros((1440, 2560, 3), dtype=np.uint8)
for row, ids in enumerate(PLAN['grid16']):
    for col, cid in enumerate(ids):
        cap = cv2.VideoCapture(str(COVER / CLIPS[cid]['path']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
        ok, frame = cap.read()
        cap.release()
        assert ok, cid
        canvas[row*360:(row+1)*360, col*640:(col+1)*640] = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
for row in range(1, 4):
    canvas[row*360-1:row*360+1, :] = (14, 17, 20)
for col in range(1, 4):
    canvas[:, col*640-1:col*640+1] = (14, 17, 20)
background = cv2.warpAffine(canvas, np.float32([[.75, 0, 0], [0, .75, 0]]), (W, H), flags=cv2.INTER_LINEAR)
yy, xx = np.mgrid[0:H, 0:W]
shade = np.exp(-((yy-H/2)/210)**4) * np.exp(-((xx-W/2)/1100)**4) * .58
background = (background.astype(np.float32) * (1-shade[:, :, None]-.08)).clip(0, 255).astype(np.uint8)

# Apply full source privacy rectangles before drawing the new title. Existing
# final-video masks have holes for the old glyphs and cannot be reused as-is.
tracks = json.loads((BASE / 'privacy_redaction_v4/mosaics/source_tracks.json').read_text())
pixelated = cv2.resize(cv2.resize(background, (48, 27), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_NEAREST)
for row, ids in enumerate(PLAN['grid16']):
    for col, cid in enumerate(ids):
        for track in tracks.get(str(cid), []):
            box = track[FRAME]
            if not box:
                continue
            x1, y1, x2, y2 = box
            x1, y1 = max(0, math.floor(col*480+x1)-1), max(0, math.floor(row*270+y1)-1)
            x2, y2 = min(W, math.ceil(col*480+x2)+1), min(H, math.ceil(row*270+y2)+1)
            background[y1:y2, x1:x2] = pixelated[y1:y2, x1:x2]

font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
first_font = ImageFont.truetype(font_path, 66)
second_font = ImageFont.truetype(font_path, 62)
options = [('A_cyan', '#64D8F2'), ('B_champagne', '#F2CE83'), ('C_mint', '#7FE0BE'), ('D_lavender', '#C0AEF2')]
prefix, emphasis = 'Beyond Arrival: ', 'Geometric Alignment'
manifest = {'source_video': str(BASE / 'exports/Full_demo_combined_v4_redacted.mp4'), 'time_seconds': 4, 'font': font_path, 'font_sizes': [66, 62], 'options': []}
for name, color in options:
    image = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    # Color the full-line glyph mask to keep the original exact kerning/alignment.
    layer = Image.new('RGBA', (W, H))
    ld = ImageDraw.Draw(layer)
    ld.text((W/2, 503), PLAN['title'][0], font=first_font, fill='white', anchor='mm')
    pixels = np.array(layer)
    full_width = draw.textlength(PLAN['title'][0], font=first_font)
    split = round(W/2-full_width/2 + draw.textlength(prefix, font=first_font))
    rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    pixels[:, split:, :3] = rgb
    image = Image.alpha_composite(image.convert('RGBA'), Image.fromarray(pixels))
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((W/2, 594), PLAN['title'][1], font=second_font, anchor='mm')
    assert box[0] > 75 and box[2] < W-75, box
    draw.text((W/2, 594), PLAN['title'][1], font=second_font, fill='white', anchor='mm')
    path = OUT / f'{name}.png'
    image.convert('RGB').save(path)
    manifest['options'].append({'id': name, 'color': color, 'path': str(path)})
    print(name, color, image.size, 'second-line bbox:', box)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
