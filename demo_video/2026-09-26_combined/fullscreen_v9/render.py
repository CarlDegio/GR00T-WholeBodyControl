"""Full-screen handheld footage with floating annotations and quiet piano music."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v8_render', BASE.parent / 'layout_v8/render.py')
v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v8)
v7 = v8.v7
v7.BASE = BASE
SOURCE = BASE.parent / 'exports/Full_demo_combined_v8_redacted.mp4'
OUTPUT = BASE.parent / 'exports/Full_demo_combined_v9_redacted_music.mp4'
W, H, N = 1920, 1080, 10774
DURATION = N / 60


def hud(task, speed, raw):
    im = v8.hud(task, speed, raw, True)
    pixels = np.array(im)
    region = Image.new('L', (W,H))
    d = ImageDraw.Draw(region)
    # Independent cards; clear gaps reveal the original full-width footage.
    for rect in [(1460,20,1900,110), (1450,128,1908,438), (1450,440,1908,748), (1450,762,1908,1078)]:
        d.rounded_rectangle(rect, radius=12, fill=255)
    mask = np.array(region)
    background = np.all(pixels[:,:,:3] == [11,17,26], axis=2)
    pixels[:,:,3][background] = 226
    pixels[:,:,3] = np.minimum(pixels[:,:,3], mask)
    return Image.fromarray(pixels)


def filters(task):
    grade = v7.TASKS[task]['grade']
    return (f'[0:v]setpts=PTS-STARTPTS,{grade},split=2[hs][cs];'
            f'[2:v]setpts=PTS-STARTPTS,fps=60,scale=1920:1080:flags=lanczos,{grade}[main];'
            '[hs]crop=348:220:1488:64,scale=420:266:flags=lanczos[head];'
            '[cs]crop=348:220:1488:344,scale=420:266:flags=lanczos[chest];'
            '[main][head]overlay=1470:168[mid];[mid][chest]overlay=1470:476[video];'
            '[video][1:v]overlay=0:0:format=auto,format=yuv420p[out]')


def render_task(task, t=None):
    conf = v7.TASKS[task]
    if t is not None:
        frame = round((t - conf['start']/60)*60)
        speed, raw = v7.states(task)[frame]
        asset = BASE / f'{task}_preview_hud.png'
        hud(task, speed, raw).save(asset)
        inputs = ['-threads','2','-ss',str(t),'-i',str(v7.SOURCE),'-i',str(asset),
                  '-threads','2','-ss',f"{frame/60:.12f}",'-i',str(BASE/f'{task}_main.mp4')]
        dest = BASE / f'{task}_after_{t:g}.png'
        v7.run(inputs + ['-filter_complex_threads','2','-filter_complex',filters(task),'-map','[out]','-frames:v','1',str(dest)])
        return
    sequence = json.loads((BASE.parent/'style_v7'/f'{task}_states.json').read_text())
    for index,row in enumerate(sequence):
        p = BASE / f'{task}_hud_{index:02d}.png'
        hud(task,row['speed'],row['raw']).save(p)
        row['path'] = str(p)
    (BASE / f'{task}_states.json').write_text(json.dumps(sequence,indent=2))
    cmd = ['ffmpeg','-v','warning','-nostdin','-y','-threads','2','-ss',f"{conf['start']/60:.12f}",'-i',str(v7.SOURCE),
           '-threads','2','-f','image2pipe','-framerate','60','-vcodec','png','-i','pipe:0',
           '-threads','2','-i',str(BASE/f'{task}_main.mp4'),'-filter_complex_threads','2','-filter_complex',filters(task),'-map','[out]',
           *v7.encode_args(conf['frames']),'-progress',str(BASE/f'{task}_progress.txt'),str(BASE/f'{task}.mp4')]
    proc = subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for row in sequence:
            data = Path(row['path']).read_bytes()
            for _ in range(row['frames']): proc.stdin.write(data)
    finally:
        proc.stdin.close()
    assert proc.wait() == 0
    print(task,'full-screen render finished',flush=True)


def music():
    source = BASE/'assets/003_Vaporware.mp3'
    assert source.stat().st_size == 6604773, 'Music download is incomplete'
    filt = (f'[0:a]asplit=2[a][b];[a][b]acrossfade=d=8:c1=tri:c2=tri,atrim=duration={DURATION:.12f},'
            f'loudnorm=I=-24:LRA=7:TP=-3,afade=t=in:st=0:d=2,afade=t=out:st={DURATION-4:.12f}:d=4,aresample=48000[out]')
    v7.run(['-i',str(source),'-filter_complex',filt,'-map','[out]','-c:a','aac','-b:a','192k','-ac','2',str(BASE/'music.m4a')])
    credit = ('Music: "Calm Piano 1 (Vaporware)" / "Vaporware"\n'
              'Artist: The Cynic Project / cynicmusic.com / pixelsphere.org\n'
              'Source: https://opengameart.org/content/calm-piano-1-vaporware\n'
              'License: CC0 1.0 — https://creativecommons.org/publicdomain/zero/1.0/\n'
              'Edit: crossfaded repeat, trimmed to video, normalized to -24 LUFS, 2s fade-in / 4s fade-out.\n')
    (BASE/'music_credit.txt').write_text(credit)
    (OUTPUT.with_suffix('.music.txt')).write_text(credit)


def assemble():
    parts = [(BASE.parent/'exports/Intro_5s_B_champagne_bold.mp4',300),
             (BASE.parent/'layout_v5/manipulation.mp4',1112),(BASE.parent/'style_v7/gate.mp4',1306),
             (BASE.parent/'layout_v5/ovmm.mp4',600),(BASE/'t2.mp4',3762),(BASE/'t4.mp4',3574),
             (BASE.parent/'style_v7/outro.mp4',120)]
    for path,count in parts:
        s=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))['streams'][0]
        assert int(s['nb_frames'])==count
    concat=BASE/'concat.txt'
    concat.write_text(''.join(f"file '{p}'\nduration {n/60:.12f}\n" for p,n in parts))
    assert not OUTPUT.exists()
    v7.run(['-f','concat','-safe','0','-i',str(concat),'-i',str(BASE/'music.m4a'),'-map','0:v:0','-map','1:a:0',
            '-c','copy','-t',f'{DURATION:.12f}','-map_metadata','-1','-metadata',
            'comment=Music: Vaporware by The Cynic Project (cynicmusic.com), CC0. https://opengameart.org/content/calm-piano-1-vaporware',
            '-movflags','+faststart',str(OUTPUT)])
    (BASE/'manifest.json').write_text(json.dumps(dict(output=str(OUTPUT),previous=str(SOURCE),frames=N,
        main_viewport=[0,0,1920,1080],annotations='Independent floating cards over the full-screen main footage',
        original_task_prompt_removed=True,music='Vaporware — The Cynic Project',music_license='CC0',
        music_source='https://opengameart.org/content/calm-piano-1-vaporware',music_target_lufs=-24),indent=2))
    print(OUTPUT,flush=True)


if __name__=='__main__':
    if '--preview' in sys.argv:
        for task,t in [('t2',57),('t2',90),('t2',110),('t4',119),('t4',150),('t4',172)]:render_task(task,t)
    elif '--music' in sys.argv:music()
    elif '--assemble' in sys.argv:assemble()
    else:
        render_task('t2')
        render_task('t4')
        music()
        assemble()
