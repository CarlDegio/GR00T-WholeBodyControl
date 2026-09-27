"""Restyle redacted long-task footage and expand the GATE section title."""
from pathlib import Path
import bisect
import hashlib
import json
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
SOURCE = BASE.parent / 'exports/Full_demo_combined_v6_redacted.mp4'
OUTPUT = BASE.parent / 'exports/Full_demo_combined_v7_redacted.mp4'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BG, PANEL, ACC, FG, MUTED = '#0b111a', '#101923', '#70DFC5', '#f0f4f8', '#a4b1c0'
LABELS = ['Long-Horizon Task in an Open Environment', 'Complex Multi-Step Manipulation Task']
TITLE = 'Geometry-Aligned Task Execution bridge'
TASKS = {
    't2': dict(start=3318, frames=3762, title='Task: Bag Placement & Box Transport',
               edit='T2_20260926_v8', session='full_vln_T2_visible_01_c84114bf87cc', epoch=1790333745.0,
               grade='eq=contrast=1.025:saturation=1.05:gamma=1.012,colorbalance=gm=-0.009:bm=0.012'),
    't4': dict(start=7080, frames=3574, title='Task: Pick Up the Bag & Dispose of It',
               edit='T4_20260926_v7', session='full_vln_T4_visible_01_0fce0b283641', epoch=1790342419.0,
               grade='eq=contrast=1.025:saturation=0.97:gamma=1.012,colorbalance=rm=-0.012:bm=0.020:bh=0.020'),
}


def font(n):
    return ImageFont.truetype(FONT, n)


def run(args):
    subprocess.run(['ffmpeg', '-v', 'warning', '-nostdin', '-y', *args], check=True)


def encode_args(frames):
    return ['-frames:v', str(frames), '-an', '-r', '60', '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '17', '-threads', '6', '-pix_fmt', 'yuv420p', '-video_track_timescale', '15360',
            '-map_metadata', '-1', '-movflags', '+faststart']


def todo_label(task, text):
    if task == 't2':
        if 'Align with' in text: return 'Align with cardboard box'
        if 'pick up the bag' in text: return 'Approach box; bag into box; carry box to open area and place it on the ground'
        if 'black cabinet' in text: return 'Move to black cabinet'
        if 'white rubbish bin' in text: return 'Turn left → white rubbish bin'
        if 'refrigerator' in text: return 'Move to refrigerator'
        if 'cardboard box' in text: return 'Turn left → cardboard box'
        if 'open area' in text: return 'Move into open area'
    else:
        text = text.split('Result:')[0]
        for key, target, label in [
            ('MOVE_TO', 'white gift bag', 'Move to white gift bag'),
            ('ALIGN', 'white gift bag', 'Align to pick up the bag'),
            ('MANIPULATE', 'Pick up', 'Pick up white gift bag'),
            ('MOVE_TO', 'green rubbish bin', 'Carry bag to green bin'),
            ('ALIGN', 'green rubbish bin', 'Align with green bin pedal'),
            ('MANIPULATE', 'pedal', 'Step on pedal; open bin; throw the bag in'),
        ]:
            if key in text and target in text: return label
    return text.split('Result:')[0].strip('- [x]')


def wrap(text, width, size):
    f, lines, line = font(size), [], ''
    for word in text.split():
        candidate = (line + ' ' + word).strip()
        if line and f.getlength(candidate) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line: lines.append(line)
    return lines


def hud(task, speed, raw, single_labels=False):
    im = Image.new('RGBA', (1920, 1080))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1919, 131), fill=BG)
    d.rectangle((24, 20, 29, 52), fill=ACC)
    d.text((46, 7), TASKS[task]['title'], font=font(34), fill=FG)
    selected = [LABELS[0 if task == 't2' else 1]] if single_labels else LABELS
    x = 46
    for label in selected:
        w = round(font(23).getlength(label)) + 32
        d.rounded_rectangle((x, 78, x + w, 115), radius=12, fill='#172f30', outline='#365754')
        d.text((x + 16, 80), label, font=font(23), fill=ACC)
        x += w + 12
    d.rounded_rectangle((1315, 19, 1404, 57), radius=10, fill='#18242f', outline='#30414e')
    d.text((1359, 37), f'{speed}×', anchor='mm', font=font(25), fill=FG)
    d.rectangle((1428, 132, 1919, 1079), fill=BG)
    d.line((1428, 132, 1428, 1079), fill='#263540', width=2)
    for title, top, picture_y in [('Head camera', 133, 168), ('Chest camera', 441, 476)]:
        d.text((1470, top), title, font=font(21), fill=ACC)
        d.rectangle((1469, picture_y - 1, 1890, picture_y + 266), outline='#30414e')
        # Transparent windows expose the repositioned, already redacted feeds.
        d.rectangle((1470, picture_y, 1889, picture_y + 265), fill=(0, 0, 0, 0))
    d.line((1460, 761, 1900, 761), fill='#263540', width=1)
    d.text((1470, 769), 'To-do list', font=font(24), fill=ACC)
    if not raw:
        d.text((1470, 817), 'Awaiting LA plan', font=font(20), fill=MUTED)
        return im
    lines = raw.splitlines()
    active = next((i for i, line in enumerate(lines) if '[ ]' in line), -1)
    rows = [(todo_label(task, line), '[x]' in line.lower(), i == active) for i, line in enumerate(lines)]
    size = 20
    while sum(len(wrap(label, 386, size)) * (size + 5) + 7 for label, _, _ in rows) > 262:
        size -= 1
        assert size >= 17
    y = 815
    for label, done, chosen in rows:
        content = wrap(label, 386, size)
        h = len(content) * (size + 5)
        color = ACC if done else (FG if chosen else MUTED)
        if chosen:
            d.rounded_rectangle((1455, y - 1, 1900, y + h + 2), radius=7, fill='#1c3038')
        d.text((1469, y), '✓' if done else ('›' if chosen else '○'), font=font(size), fill=color)
        for j, line in enumerate(content):
            d.text((1497, y + j * (size + 5)), line, font=font(size), fill=color)
        y += h + 7
    assert y <= 1078, y
    return im


def states(task):
    conf = TASKS[task]
    timeline = json.loads((ROOT / 'outputs/demo_edits' / conf['edit'] / 'timeline.json').read_text())
    events = [json.loads(line) for line in (ROOT / 'outputs/experiments' / conf['session'] / 'events.jsonl').read_text().splitlines()]
    todos = sorted((e['wall_time_ns'] / 1e9 - conf['epoch'], e['todo_list']) for e in events if e.get('code') == 'TODO_UPDATED')
    times = [t for t, _ in todos]
    result = []
    for seg in timeline['segments']:
        for j in range(seg['frames']):
            t = seg['start'] + (seg['end'] - seg['start']) * j / seg['frames']
            k = bisect.bisect_right(times, t + 1e-5) - 1
            raw = todos[k][1] if k >= 0 else ''
            speed = seg.get('speed', 4 if task == 't2' else 3)
            if seg['kind'] == 'outro': raw = raw.replace('[ ]', '[x]')
            result.extend([(speed, raw)] * 2)
    assert len(result) == conf['frames']
    return result


def assets(task, single_labels=False):
    folder = BASE / f'{task}_hud'
    folder.mkdir(exist_ok=True)
    sequence, last = [], None
    for speed, raw in states(task):
        key = hashlib.sha256((str(speed) + raw + str(single_labels)).encode()).hexdigest()[:16]
        if key == last:
            sequence[-1]['frames'] += 1
        else:
            path = folder / f'{key}.png'
            if not path.exists(): hud(task, speed, raw, single_labels).save(path)
            sequence.append(dict(path=str(path), frames=1, speed=speed, raw=raw))
            last = key
    concat = BASE / f'{task}_hud.txt'
    concat.write_text(''.join(f"file '{r['path']}'\noption framerate 60\nduration {r['frames'] / 60:.12f}\n" for r in sequence)
                      + f"file '{sequence[-1]['path']}'\noption framerate 60\n")
    (BASE / f'{task}_states.json').write_text(json.dumps(sequence, indent=2))
    return concat


def filters(task):
    return (f"[0:v]setpts=PTS-STARTPTS,{TASKS[task]['grade']},split=3[main][hs][cs];"
            '[hs]crop=348:220:1488:64,scale=420:266:flags=lanczos[head];'
            '[cs]crop=348:220:1488:344,scale=420:266:flags=lanczos[chest];'
            '[main][head]overlay=1470:168[mid];[mid][chest]overlay=1470:476[video];'
            '[video][1:v]overlay=0:0:format=auto,format=yuv420p[out]')


def task_render(task, preview_time=None, single_labels=False):
    conf = TASKS[task]
    if preview_time is not None:
        fid = round((preview_time - conf['start'] / 60) * 60)
        speed, raw = states(task)[fid]
        asset = BASE / f'{task}_preview_hud.png'
        hud(task, speed, raw, single_labels).save(asset)
        cmd = ['-threads', '2', '-ss', str(preview_time), '-i', str(SOURCE), '-i', str(asset)]
        output_args = ['-frames:v', '1', str(BASE / f'{task}_after_{preview_time:g}.png')]
    else:
        assets(task, single_labels)
        cmd = ['-threads', '2', '-ss', f"{conf['start'] / 60:.12f}", '-i', str(SOURCE),
               '-threads', '2', '-f', 'image2pipe', '-framerate', '60', '-vcodec', 'png', '-i', 'pipe:0']
        output_args = encode_args(conf['frames']) + ['-progress', str(BASE / f'{task}_progress.txt'), str(BASE / f'{task}.mp4')]
    command = cmd + ['-filter_complex_threads', '2', '-filter_complex', filters(task), '-map', '[out]'] + output_args
    if preview_time is not None:
        run(command)
    else:
        # ffmpeg 4.4 lacks concat's per-file option directive. A PNG pipe with
        # an explicit 60 Hz clock keeps every HUD transition frame-exact.
        proc = subprocess.Popen(['ffmpeg', '-v', 'warning', '-nostdin', '-y', *command], stdin=subprocess.PIPE)
        sequence = json.loads((BASE / f'{task}_states.json').read_text())
        try:
            for row in sequence:
                data = Path(row['path']).read_bytes()
                for _ in range(row['frames']):
                    proc.stdin.write(data)
        finally:
            proc.stdin.close()
        assert proc.wait() == 0, task
    print('Finished', task, preview_time, flush=True)


def gate_render(preview=False):
    im = Image.new('RGBA', (1920, 1080))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1919, 71), fill=BG)
    d.rectangle((24, 20, 29, 52), fill=ACC)
    d.text((46, 7), TITLE, font=font(34), fill=FG)
    asset = BASE / 'gate_title.png'
    im.save(asset)
    cmd = ['-threads', '2', '-ss', f'{1412 / 60 + (3 if preview else 0):.12f}', '-i', str(SOURCE), '-i', str(asset),
           '-filter_complex_threads', '2', '-filter_complex', '[0:v][1:v]overlay=0:0:format=auto,format=yuv420p[out]', '-map', '[out]']
    args = ['-frames:v', '1', str(BASE / 'gate_after.png')] if preview else encode_args(1306) + [str(BASE / 'gate.mp4')]
    run(cmd + args)


def assemble():
    # Unchanged sections come from their verified v6 source components.
    parts = [(BASE.parent / 'exports/Intro_5s_B_champagne_bold.mp4', 300),
             (BASE.parent / 'layout_v5/manipulation.mp4', 1112), (BASE / 'gate.mp4', 1306),
             (BASE.parent / 'layout_v5/ovmm.mp4', 600), (BASE / 't2.mp4', 3762), (BASE / 't4.mp4', 3574)]
    # The outro is only two seconds; encode its exact source-frame range.
    outro = BASE / 'outro.mp4'
    run(['-threads', '2', '-ss', f'{10654 / 60:.12f}', '-i', str(SOURCE), '-map', '0:v:0', *encode_args(120), str(outro)])
    parts.append((outro, 120))
    for path, count in parts:
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))['streams'][0]
        assert int(info['nb_frames']) == count, (path, info['nb_frames'], count)
    concat = BASE / 'concat.txt'
    concat.write_text(''.join(f"file '{p}'\nduration {n / 60:.12f}\n" for p, n in parts))
    assert not OUTPUT.exists()
    run(['-f', 'concat', '-safe', '0', '-i', str(concat), '-map', '0:v:0', '-c', 'copy', '-an', '-map_metadata', '-1', '-movflags', '+faststart', str(OUTPUT)])
    (BASE / 'manifest.json').write_text(json.dumps(dict(source=str(SOURCE), output=str(OUTPUT), title=TITLE, labels=LABELS,
                                                       single_labels='--single-labels' in sys.argv, tasks=TASKS, total_frames=10774), indent=2))
    print(OUTPUT, flush=True)


if __name__ == '__main__':
    single = '--single-labels' in sys.argv
    if '--preview' in sys.argv:
        gate_render(preview=True)
        for task, t in [('t2', 57), ('t2', 90), ('t2', 110), ('t4', 119), ('t4', 150), ('t4', 172)]:
            task_render(task, t, single)
    else:
        if not (BASE / 'gate.mp4').exists():
            gate_render()
        task_render('t2', single_labels=single)
        task_render('t4', single_labels=single)
        assemble()
