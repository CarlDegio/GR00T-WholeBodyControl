from render import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,required=True);ap.add_argument('--end',type=int,required=True);ap.add_argument('--parts',nargs='+',required=True);ap.add_argument('--name',required=True)
a=ap.parse_args();frames={}
for name in a.parts:
 data=json.loads((BASE/name/'masks.json').read_text())
 for key,boxes in data['frames'].items():
  fid=int(key)
  if not a.start<=fid<a.end:continue
  for b in boxes:
   assert len(b)==4 and all(isinstance(x,int) for x in b),(name,b)
   x1,y1,x2,y2=b;assert 0<=x1<x2<=W and 0<=y1<y2<=H,(name,b)
  frames.setdefault(key,[]).extend(boxes)
cv2.setNumThreads(2)
D=BASE/'parts';D.mkdir(exist_ok=True);out=D/f'{a.name}.mp4';assert not out.exists(),out
cap=cv2.VideoCapture(str(SOURCE),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);assert cap.isOpened();cap.set(cv2.CAP_PROP_POS_FRAMES,a.start)
cmd=['ffmpeg','-hide_banner','-v','warning','-nostdin','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-map_metadata','-1','-c:v','libx264','-preset','fast','-crf','17','-threads','6','-pix_fmt','yuv420p','-video_track_timescale','15360','-movflags','+faststart',str(out)]
log=(D/f'{a.name}.log').open('w');p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log);start=time.monotonic()
for fid in range(a.start,a.end):
 ok,im=cap.read();assert ok,fid
 p.stdin.write(redact(im,frames.get(str(fid),[])).tobytes())
 if (fid-a.start)%600==0: print(a.name,f'{fid-a.start}/{a.end-a.start}',f'{time.monotonic()-start:.1f}s',flush=True)
p.stdin.close();assert p.wait()==0;cap.release();log.close()
(D/f'{a.name}_masks.json').write_text(json.dumps(dict(start=a.start,end=a.end,parts=a.parts,frames=frames),separators=(',',':')))
print('FINISHED',out,flush=True)
