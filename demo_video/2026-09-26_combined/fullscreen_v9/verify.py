"""Verify full-canvas footage, floating HUD, unchanged sections and audible music."""
import hashlib
import json
import re
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw
from render import BASE, SOURCE, OUTPUT, DURATION, v7


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def hashes(p):
    return [s.rsplit(',',1)[-1].strip() for s in p.read_text().splitlines() if s and not s.startswith('#')]


def frame(c,n):
    c.set(cv2.CAP_PROP_POS_FRAMES,n)
    ok,f=c.read()
    assert ok,n
    return f


j=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(OUTPUT)]))
vs=next(s for s in j['streams'] if s['codec_type']=='video')
au=next(s for s in j['streams'] if s['codec_type']=='audio')
assert len(j['streams'])==2
assert (vs['width'],vs['height'],vs['r_frame_rate'],int(vs['nb_frames']))==(1920,1080,'60/1',10774)
assert (au['codec_name'],int(au['sample_rate']),au['channels'])==('aac',48000,2)
assert abs(float(j['format']['duration'])-DURATION)<.04
assert abs(float(au['duration'])-DURATION)<.04
assert sha(SOURCE)==json.loads((BASE.parent/'layout_v8/verification.json').read_text())['sha256']
cv2.setNumThreads(2)
c=cv2.VideoCapture(str(OUTPUT),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
samples=[]
for task,t in [('t2',57),('t2',90),('t2',110),('t4',119),('t4',150),('t4',172)]:
    actual=frame(c,round(t*60))
    expected=cv2.imread(str(BASE/f'{task}_after_{t}.png'))
    mae=float(np.abs(actual.astype(np.int16)-expected.astype(np.int16)).mean())
    assert mae<4,(task,t,mae)
    # The outer right strip and inter-card gaps must contain moving scene pixels.
    right_mae=float(np.abs(actual[:,1910:].astype(np.int16)-expected[:,1910:].astype(np.int16)).mean())
    assert right_mae<4
    samples.append(dict(task=task,time_seconds=t,preview_mae=mae,right_edge_mae=right_mae))
print('Full-width layout and reference previews verified.',flush=True)

checks=[]
for task,conf in v7.TASKS.items():
    rows=json.loads((BASE/f'{task}_states.json').read_text())
    c.set(cv2.CAP_PROP_POS_FRAMES,conf['start'])
    count=0
    maximum=0
    for row in rows:
        hud=cv2.imread(row['path'],cv2.IMREAD_UNCHANGED)
        assert not hud[:,:1450,3].any()
        assert not hud[:,1910:,3].any()
        assert not hud[112:126,:,3].any()
        # Compare opaque text/strokes only; card backgrounds intentionally transmit video.
        roi=hud[::3,::3]
        mask=(roi[:,:,3]==255)
        # Compare luminance: 4:2:0 chroma at thin glyph edges mixes with the
        # translucent card background even when the glyph alpha is opaque.
        # Luminance preserves the actual text/state geometry frame by frame.
        expected=cv2.cvtColor(hud[:,:,:3],cv2.COLOR_BGR2GRAY)[::3,::3][mask].astype(np.int16)
        assert len(expected)>5000
        for _ in range(row['frames']):
            ok,actual=c.read();assert ok
            error=float(np.abs(cv2.cvtColor(actual,cv2.COLOR_BGR2GRAY)[::3,::3][mask].astype(np.int16)-expected).mean())
            assert error<4,(task,count,error)
            maximum=max(maximum,error)
            count+=1
    assert count==conf['frames']
    checks.append(dict(task=task,frames=count,max_opaque_hud_luma_mae=maximum))
    print(task,'all dynamic overlays verified:',count,flush=True)

sheet=Image.new('RGB',(1280,752),'#0b111a');d=ImageDraw.Draw(sheet)
for row,pair in enumerate([(3318,7079),(7080,10653)]):
    for col,fid in enumerate(pair):
        im=frame(c,fid)
        cv2.imwrite(str(BASE/f'verified_{fid:05d}.jpg'),im)
        sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((640,360)),(col*640,row*376+16))
        d.text((col*640+8,row*376),f'{fid/60:.3f}s',fill='white')
sheet.save(BASE/'verified_contact_sheet.jpg',quality=94)
c.release()
with (BASE/'video_decode.log').open('w') as log:
    subprocess.run(['ffmpeg','-v','error','-xerror','-nostdin','-y','-threads','4','-i',str(OUTPUT),'-map','0:v:0','-f','framemd5',str(BASE/'output.framemd5')],stderr=log,check=True)
new=hashes(BASE/'output.framemd5');old=hashes(BASE.parent/'layout_v8/output.framemd5')
assert len(new)==10774
assert new[:3318]==old[:3318]
assert new[10654:]==old[10654:]
a=subprocess.run(['ffmpeg','-hide_banner','-nostdin','-xerror','-i',str(OUTPUT),'-vn','-af','loudnorm=I=-24:LRA=7:TP=-3:print_format=json','-f','null','-'],capture_output=True,text=True,check=True)
(BASE/'audio_decode.log').write_text(a.stderr)
measurement=json.loads(re.findall(r'\{\s*"input_i"[\s\S]*?\}',a.stderr)[-1])
assert -28<float(measurement['input_i'])<-21,measurement
assert float(measurement['input_tp'])<-.5,measurement
receipt=dict(output=str(OUTPUT),frames=10774,resolution=[1920,1080],fps=60,duration_seconds=DURATION,
             full_video_decode='passed',full_audio_decode='passed',audio_codec='AAC stereo 48 kHz',
             audio_loudness_lufs=float(measurement['input_i']),audio_true_peak_db=float(measurement['input_tp']),
             unchanged_video_frames=3438,main_footage='full 1920x1080 canvas; floating overlays',
             hud_checks=checks,sample_checks=samples,bytes=OUTPUT.stat().st_size,sha256=sha(OUTPUT))
(BASE/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
