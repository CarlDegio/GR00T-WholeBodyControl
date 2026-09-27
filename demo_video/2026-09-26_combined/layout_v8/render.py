"""Full-height long-task footage, with every annotation in the right sidebar."""
from pathlib import Path
import importlib.util
import json
import sys
import subprocess

from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
PREVIOUS = BASE.parent / 'style_v7'
spec = importlib.util.spec_from_file_location('v7_render', PREVIOUS / 'render.py')
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)
OUTPUT = BASE.parent / 'exports/Full_demo_combined_v8_redacted.mp4'
SOURCE = BASE.parent / 'exports/Full_demo_combined_v7_redacted.mp4'
old_hud = v7.hud


def hud(task, speed, raw, single_labels=True):
    # Reuse the verified task-list layout, then replace the header and sidebar top.
    im = old_hud(task, speed, raw, single_labels=True)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1427, 1079), fill=(0, 0, 0, 0))
    d.rectangle((1428, 0, 1919, 131), fill=v7.BG)
    d.line((1428, 0, 1428, 1079), fill='#263540', width=2)
    d.rounded_rectangle((1460, 20, 1900, 110), radius=12, fill='#172f30', outline='#365754')
    lines = (['Long-Horizon Task in an', 'Open Environment'] if task == 't2'
             else ['Complex Multi-Step', 'Manipulation Task'])
    for i, line in enumerate(lines):
        d.text((1478, 24 + i * 36), line, font=v7.font(28), fill=v7.ACC)
    d.rounded_rectangle((1814, 133, 1890, 162), radius=8, fill='#18242f', outline='#30414e')
    d.text((1852, 147), f'{speed}×', anchor='mm', font=v7.font(21), fill=v7.FG)
    return im


def filters(task):
    # Crop below v6's baked-in prompt, then fill the entire left viewport at
    # the original aspect ratio. Reuse redacted v6 pixels and v7's exact grade.
    return (f"[0:v]setpts=PTS-STARTPTS,{v7.TASKS[task]['grade']},split=3[ms][hs][cs];"
            '[ms]crop=1428:978:0:102,scale=-2:1080:flags=lanczos,crop=1428:1080:74:0,'
            'pad=1920:1080:0:0:color=0x0b111a[main];'
            '[hs]crop=348:220:1488:64,scale=420:266:flags=lanczos[head];'
            '[cs]crop=348:220:1488:344,scale=420:266:flags=lanczos[chest];'
            '[main][head]overlay=1470:168[mid];[mid][chest]overlay=1470:476[video];'
            '[video][1:v]overlay=0:0:format=auto,format=yuv420p[out]')


v7.BASE = BASE
v7.hud = hud
v7.filters = filters


def assemble():
    parts = [(BASE.parent / 'exports/Intro_5s_B_champagne_bold.mp4', 300),
             (BASE.parent / 'layout_v5/manipulation.mp4', 1112), (PREVIOUS / 'gate.mp4', 1306),
             (BASE.parent / 'layout_v5/ovmm.mp4', 600), (BASE / 't2.mp4', 3762),
             (BASE / 't4.mp4', 3574), (PREVIOUS / 'outro.mp4', 120)]
    for path, count in parts:
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))['streams'][0]
        assert int(info['nb_frames']) == count
    concat = BASE / 'concat.txt'
    concat.write_text(''.join(f"file '{p}'\nduration {n / 60:.12f}\n" for p, n in parts))
    assert not OUTPUT.exists()
    v7.run(['-f', 'concat', '-safe', '0', '-i', str(concat), '-map', '0:v:0', '-c', 'copy', '-an', '-map_metadata', '-1', '-movflags', '+faststart', str(OUTPUT)])
    (BASE / 'manifest.json').write_text(json.dumps(dict(output=str(OUTPUT), previous=str(SOURCE),
        redacted_footage_source=str(v7.SOURCE), frames=10774, left_viewport=[0, 0, 1428, 1080],
        sidebar=[1428, 0, 492, 1080], titles=v7.LABELS, all_annotations='right sidebar',
        original_task_prompt_removed=True, color_grade='unchanged from v7'), indent=2))
    print(OUTPUT, flush=True)


if __name__ == '__main__':
    if '--preview' in sys.argv:
        for task, t in [('t2', 57), ('t2', 90), ('t2', 110), ('t4', 119), ('t4', 150), ('t4', 172)]:
            v7.task_render(task, t, True)
    else:
        v7.task_render('t2', single_labels=True)
        v7.task_render('t4', single_labels=True)
        assemble()
