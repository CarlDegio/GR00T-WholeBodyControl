"""Recompose only the requested sections from the already redacted v4 video."""
from pathlib import Path
import json
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / 'exports/Full_demo_combined_v4_redacted.mp4'
OUTPUT = BASE.parent / 'exports/Full_demo_combined_v5_redacted.mp4'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
ACC = '#70DFC5'
BG = '#0b111a'
TASKS = ['Medicine Bottle into Basket', 'Bag Placement and Box Transfer',
         'Pick Up Bowl and Empty Contents', 'Foot-Operated Waste Disposal']
SECTIONS = [('manipulation', 300, 1112, 'Whole Body Manipulation'),
            ('gate', 1412, 1306, 'GATE'),
            ('ovmm', 2718, 600, 'GATE vs Official RL on OVMM')]


def run(args):
    subprocess.run(args, check=True)


def overlay(name, title):
    im = Image.new('RGBA', (1920, 1080))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1919, 71), fill=BG)
    d.rectangle((24, 20, 29, 52), fill=ACC)
    d.text((46, 7), title, font=ImageFont.truetype(FONT, 34), fill='#f0f4f8')
    if name != 'ovmm':
        for i, text in enumerate(TASKS):
            x, y = (i % 2) * 960, 72 + (i // 2) * 504
            d.rectangle((x, y, x + 959, y + 51), fill=(11, 17, 26, 185))
            d.rectangle((x + 20, y + 15, x + 23, y + 37), fill=ACC)
            d.text((x + 36, y + 6), text, font=ImageFont.truetype(FONT, 25), fill=ACC)
    else:
        for i, text in enumerate(['Board game', 'Vase', 'Electric kettle', 'Backpack']):
            for row in range(2):
                x, y = i * 480, 72 + row * 504
                color = ACC if row == 0 else '#FF9A9E'
                status = 'GATE  |  Success' if row == 0 else 'Official RL  |  Failure'
                d.rectangle((x, y, x + 479, y + 77), fill=(11, 17, 26, 192))
                d.text((x + 20, y + 8), text, font=ImageFont.truetype(BOLD, 25), fill='#f0f4f8')
                d.text((x + 20, y + 43), status, font=ImageFont.truetype(BOLD, 22), fill=color)
                d.rectangle((x, y, x + 479, y + 2), fill=color)
    path = BASE / f'{name}_overlay.png'
    im.save(path)
    return path


def filters(name):
    if name == 'ovmm':
        # v4 is a 0.75x resize of the 2560x1440 OVMM composition.
        # exact=1 preserves the odd y coordinates of the square video windows.
        boxes = [(48 + col * 468, y, 420, 420) for y in (145, 619) for col in range(4)]
        size, crop = '504:504', '480:504:12:0'
        positions = [(col * 480, row * 504) for row in range(2) for col in range(4)]
    else:
        boxes = [(x, y, 804, 452) for y in (112, 616) for x in (114, 1002)]
        # Bottom-anchored crop retains all existing playback-speed badges.
        size, crop = '960:540', '960:504:0:36'
        positions = [(0, 0), (960, 0), (0, 504), (960, 504)]
    n = len(boxes)
    f = [f'[0:v]setpts=PTS-STARTPTS,split={n}' + ''.join(f'[s{i}]' for i in range(n))]
    for i, (x, y, w, h) in enumerate(boxes):
        f.append(f'[s{i}]crop={w}:{h}:{x}:{y}:exact=1,scale={size}:flags=lanczos,crop={crop},setsar=1[v{i}]')
    layout = '|'.join(f'{x}_{y}' for x, y in positions)
    f.append(''.join(f'[v{i}]' for i in range(n)) +
             f'xstack=inputs={n}:layout={layout},pad=1920:1080:0:72:color={BG.replace("#", "0x")}[grid]')
    f.append('[grid][1:v]overlay=0:0:format=auto,format=yuv420p[out]')
    return ';\n'.join(f)


def render_section(name, start, frames, title, preview=False):
    asset = overlay(name, title)
    filt = BASE / f'{name}_filters.txt'
    filt.write_text(filters(name))
    offset = 3 if preview else 0
    dest = BASE / (f'{name}_after.png' if preview else f'{name}.mp4')
    cmd = ['ffmpeg', '-v', 'warning', '-nostdin', '-y', '-threads', '2',
           '-ss', f'{start / 60 + offset:.12f}', '-i', str(SOURCE), '-i', str(asset),
           '-filter_complex_threads', '2', '-filter_complex_script', str(filt),
           '-map', '[out]', '-an', '-frames:v', '1' if preview else str(frames)]
    if not preview:
        cmd += ['-r', '60', '-c:v', 'libx264', '-preset', 'fast', '-crf', '17',
                '-threads', '6', '-pix_fmt', 'yuv420p', '-video_track_timescale', '15360',
                '-map_metadata', '-1', '-movflags', '+faststart', '-progress', str(BASE / f'{name}_progress.txt')]
    run(cmd + [str(dest)])
    print(f'Finished {dest.name}', flush=True)


def render_video():
    cover = BASE / 'cover.mp4'
    run(['ffmpeg', '-v', 'warning', '-nostdin', '-y', '-threads', '2', '-i', str(SOURCE),
         '-map', '0:v:0', '-frames:v', '300', '-an', '-c:v', 'libx264', '-preset', 'fast',
         '-crf', '17', '-threads', '6', '-pix_fmt', 'yuv420p', '-video_track_timescale', '15360', str(cover)])
    for section in SECTIONS:
        render_section(*section)
    # v4 was assembled with an independently encoded part starting at frame 3318.
    # This exact keyframe permits lossless packet copying of the remaining video.
    tail = BASE / 'tail.mp4'
    run(['ffmpeg', '-v', 'warning', '-nostdin', '-y', '-ss', '55.3', '-i', str(SOURCE),
         '-map', '0:v:0', '-an', '-c', 'copy', '-map_metadata', '-1', str(tail)])
    parts = [('cover', 300), ('manipulation', 1112), ('gate', 1306), ('ovmm', 600), ('tail', 7456)]
    for name, frames in parts:
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                                                 '-of', 'json', str(BASE / f'{name}.mp4')]))['streams'][0]
        assert int(info['nb_frames']) == frames, (name, info['nb_frames'], frames)
        assert (info['width'], info['height'], info['r_frame_rate']) == (1920, 1080, '60/1')
    concat = BASE / 'concat.txt'
    concat.write_text(''.join(f"file '{BASE / (name + '.mp4')}'\nduration {frames / 60:.12f}\n" for name, frames in parts))
    run(['ffmpeg', '-v', 'warning', '-nostdin', '-n', '-f', 'concat', '-safe', '0', '-i', str(concat),
         '-map', '0:v:0', '-c', 'copy', '-an', '-map_metadata', '-1', '-movflags', '+faststart', str(OUTPUT)])
    manifest = dict(source=str(SOURCE), output=str(OUTPUT), fps=60, frames=10774,
                    header_height=72, accent=ACC, privacy='All video pixels sourced from redacted v4',
                    sections=[dict(name=n, start_frame=s, frames=f, title=t) for n, s, f, t in SECTIONS],
                    unchanged_tail_start_frame=3318)
    (BASE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(OUTPUT, flush=True)


if __name__ == '__main__':
    if '--preview' in sys.argv:
        for section in SECTIONS:
            render_section(*section, preview=True)
    else:
        render_video()
