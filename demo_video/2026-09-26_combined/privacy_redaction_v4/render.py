"""Apply frame-accurate privacy masks to the approved final edit without changing its timeline."""
from pathlib import Path
import json,subprocess,cv2,numpy as np,time
BASE=Path(__file__).resolve().parent
SOURCE=BASE.parent/'exports/Full_demo_combined_v3.mp4'
OUTPUT=BASE.parent/'exports/Full_demo_combined_v4_redacted.mp4'
PARTS=['mosaics','preview','t2_main_faces','t2_codes','t2_chest_faces','t2_inset_codes','t4']
FPS,W,H,N=60,1920,1080,10774

def merge():
 frames={};summ=[]
 for name in PARTS:
  p=BASE/name/'masks.json';data=json.loads(p.read_text());d=data['frames']
  for k,boxes in d.items():
   f=int(k);assert 0<=f<N,(name,f)
   for box in boxes:
    assert len(box)==4,(name,box)
    x1,y1,x2,y2=box;assert all(isinstance(v,int) for v in box),(name,box)
    assert 0<=x1<x2<=W and 0<=y1<y2<=H,(name,box)
    frames.setdefault(str(f),[]).append(box)
  summ.append(dict(part=name,frame_count=len(d),targets=data.get('targets',[])))
 merged=dict(source=str(SOURCE),fps=FPS,width=W,height=H,frame_count=N,frames=frames,parts=summ)
 (BASE/'merged_masks.json').write_text(json.dumps(merged,separators=(',',':')))
 return merged

def redact(frame,boxes):
 if not boxes:return frame
 # Precompute a coarse mosaic from the whole frame. This also handles 1px mask strips
 # cut around the outro's existing lettering without retaining detail in thin strips.
 tiny=cv2.resize(frame,(48,27),interpolation=cv2.INTER_AREA)
 pix=cv2.resize(tiny,(W,H),interpolation=cv2.INTER_NEAREST)
 for x1,y1,x2,y2 in boxes:frame[y1:y2,x1:x2]=pix[y1:y2,x1:x2]
 return frame

def main():
 data=merge();frames=data['frames'];cv2.setNumThreads(2)
 cap=cv2.VideoCapture(str(SOURCE),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);assert cap.isOpened()
 assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==N
 assert not OUTPUT.exists(),OUTPUT
 cmd=['ffmpeg','-hide_banner','-v','warning','-nostdin','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-map_metadata','-1','-c:v','libx264','-preset','fast','-crf','17','-threads','8','-pix_fmt','yuv420p','-video_track_timescale','15360','-movflags','+faststart',str(OUTPUT)]
 start=time.monotonic();log=(BASE/'encode.log').open('w');p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
 for f in range(N):
  ok,im=cap.read();assert ok,f
  out=redact(im,frames.get(str(f),[]));p.stdin.write(out.tobytes())
  if f%600==0:print(f'{f}/{N} frames, elapsed {time.monotonic()-start:.1f}s',flush=True)
 p.stdin.close();assert p.wait()==0;cap.release();log.close()
 print('EXPORTED',OUTPUT,flush=True)
if __name__=='__main__':main()
