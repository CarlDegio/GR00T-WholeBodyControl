from pathlib import Path
import json,math
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
cv2.setNumThreads(2)
P=Path(__file__).resolve().parent;ROOT=P.parents[3];COVER=ROOT/'2026-09-26_cover'
# ROOT resolves demo_video from nested output.
COVER=Path('/home/user/Project/GR00T-WholeBodyControl/demo_video/2026-09-26_cover')
plan=json.load(open(COVER/'edit/selection.json'));clips={c['id']:c for c in plan['clips']}
tracks={};initial={11:[[144,218,190,244]],13:[[207,210,243,239]],14:[[219,187,255,218]],21:[[13,49,36,70],[449,56,479,79]]}
source_frames={}
for cid in [11,12,13,14,21]:
 cap=cv2.VideoCapture(str(COVER/clips[cid]['path']));arr=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  arr.append(cv2.resize(f,(480,270)))
 source_frames[cid]=arr
 if cid==12:continue
 alltracks=[]
 for box in initial[cid]:
  x1,y1,x2,y2=box;gray0=cv2.cvtColor(arr[0],cv2.COLOR_BGR2GRAY);template=gray0[y1:y2,x1:x2];w=x2-x1;h=y2-y1;prev=np.array([x1,y1]);tr=[]
  for n,f in enumerate(arr):
   gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
   if n:
    xa=max(0,int(prev[0])-12);ya=max(0,int(prev[1])-12);xb=min(480,int(prev[0])+w+12);yb=min(270,int(prev[1])+h+12)
    scores=cv2.matchTemplate(gray[ya:yb,xa:xb],template,cv2.TM_CCOEFF_NORMED);_,score,_,loc=cv2.minMaxLoc(scores)
    if score>.45:prev=np.array([xa+loc[0],ya+loc[1]])
   x,y=prev;tr.append([int(x)-3,int(y)-3,int(x+w)+3,int(y+h)+3])
  alltracks.append(tr)
 tracks[cid]=alltracks
# Manually placed keyframes on 480x270 source frames; pair of adjacent labels, not the whole box.
keys=[(0,[156,176,211,213]),(.5,[156,176,211,213]),(.75,[156,170,211,207]),(1,[157,159,212,200]),(1.25,[157,145,211,187]),(1.5,[170,135,222,180]),(1.625,[188,133,237,180]),(1.75,[210,127,250,186]),(1.875,[270,139,281,170]),(2,[0,0,0,0]),(2.05,[215,110,239,155]),(2.125,[229,94,276,153]),(2.25,[256,91,315,143]),(2.375,[282,90,349,145]),(2.5,[291,88,359,143]),(2.75,[287,97,357,149]),(3,[294,102,367,153]),(3.25,[310,123,379,177]),(3.5,[309,161,374,211]),(3.75,[309,177,376,229]),(4,[285,196,346,246]),(4.25,[287,207,351,263]),(4.5,[286,210,350,268]),(299/60,[286,210,350,268])]
# Source clip12 turns its code away between roughly 1.80 and 2.05s; boundary covered conservatively.
def code12(t):
 if 1.85<=t<2.0166667:return []
 if 1.8<=t<1.85:return [237,139,259,177]
 if 2.0166667<=t<2.05:return [207,104,239,160]
 for (t0,a),(t1,b) in zip(keys,keys[1:]):
  if t0<=t<=t1:
   u=(t-t0)/(t1-t0);return ((1-u)*np.array(a)+u*np.array(b)).tolist()
 return keys[-1][1]
tracks[12]=[[code12(n/60) for n in range(300)]]
for b in tracks[12][0]:
 if b:
  b[0]-=2; b[1]-=2; b[2]+=2; b[3]+=8
# Exact title glyph footprint; preserve text, cover revealed code pixels around it.
W,H=1920,1080
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
cover_title=Image.new('L',(W,H));d=ImageDraw.Draw(cover_title)
for i,line in enumerate(plan['title']):d.text((W/2,503+i*91),line,font=ImageFont.truetype(bold if i==0 else font,66 if i==0 else 62),fill=255,anchor='mm')
out_title=Image.new('L',(W,H));d=ImageDraw.Draw(out_title);d.text((W/2,H/2),'Thank you for watching!',font=ImageFont.truetype(bold,84),fill=255,anchor='mm',stroke_width=2,stroke_fill=255)
cover_protect=np.asarray(cover_title)>20;out_protect=np.asarray(out_title)>20
# Strip decomposition of rectangles with opaque title glyphs removed; each strip stays tiny/local.
def subtract_text(rect,protect):
 x1,y1,x2,y2=rect
 if not protect[y1:y2,x1:x2].any():return [rect]
 result=[];active={}
 for y in range(y1,y2):
  row=~protect[y,x1:x2];edges=np.flatnonzero(np.diff(np.r_[False,row,False]));runs={(x1+int(a),x1+int(b)) for a,b in zip(edges[::2],edges[1::2])}
  for run in list(active):
   if run not in runs:result.append([run[0],active.pop(run),run[1],y])
  for run in runs:active.setdefault(run,y)
 for run,ys in active.items():result.append([run[0],ys,run[1],y2])
 return result
frames={};raw={};targetrecords=[]
def finish(b,protect):
 b=[max(0,math.floor(b[0])-1),max(0,math.floor(b[1])-1),min(W,math.ceil(b[2])+1),min(H,math.ceil(b[3])+1)]
 if b[2]<=b[0] or b[3]<=b[1]:return []
 return subtract_text(b,protect) if protect is not None else [b]
for n in range(300):
 u=min(1,max(0,n/60-1));ease=u*u*(3-2*u);scale=4/(2+2*ease);rects=[]
 for row,ids in enumerate(plan['grid16']):
  for col,cid in enumerate(ids):
   if cid not in tracks:continue
   for tr in tracks[cid]:
    b=tr[n]
    if not b:continue
    x1,y1,x2,y2=b;b=[(col*480+x1-960)*scale+960,(row*270+y1-540)*scale+540,(col*480+x2-960)*scale+960,(row*270+y2-540)*scale+540]
    rects+=finish(b,cover_protect if n>=180 else None)
 frames[str(n)]=rects
out=json.load(open(COVER.parent/'2026-09-26_combined/exports/outro_selection_v3.json'))
for j in range(120):
 rects=[]
 for rec in out['clips']:
  cid=rec['id']
  if cid not in tracks:continue
  n=round(rec['start_seconds']*60)+j
  for tr in tracks[cid]:
   b=tr[n]
   if not b:continue
   x1,y1,x2,y2=b;b=[rec['col']*384+x1*.8,rec['row']*216+y1*.8,rec['col']*384+x2*.8,rec['row']*216+y2*.8]
   rects+=finish(b,out_protect)
 frames[str(10654+j)]=rects
result={'frames':frames,'targets':['C01: clip11 foreground square-code pair','C03: clip21 left/right background square-code areas','C04/O01: clip13 foreground code area','C05/O03: clip12 moving box code area, dense manual turning keyframes','O02: clip11 foreground','O04: clip14 foreground'],'notes':['1920x1080 xyxy integer exclusive; global zero-based frames 0–299 and10654–10773.','Source coordinates 480x270. Exact original smoothstep projection for cover zoom.','Static boxes tracked frame by frame with bounded template matching and margin; clip12 manually keyed at up to0.125s during turn.','Only visible code label areas, not whole windows. Adjacent circular packaging label included as compact conservative margin, not asserted a QR.','Opaque title glyphs preserved using original font/layout strip subtraction; no face masks in this file.']}
(P/'masks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(P/'source_tracks.json').write_text(json.dumps(tracks))
# Source12 dense review sheet and full-frame mask samples.
sh=Image.new('RGB',(1440,1800))
for i,n in enumerate([45,60,75,90,98,105,109,112,120,122,127,135,142,150,165,180,195,210]):
 f=source_frames[12][n].copy();b=tracks[12][0][n]
 if b:cv2.rectangle(f,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(0,0,255),2)
 im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));x=i%3*480;y=i//3*300;sh.paste(im,(x,y+30));ImageDraw.Draw(sh).text((x+3,y+3),f'clip12 n={n} t={n/60:.3f}',fill='white')
sh.save(P/'clip12_mask_review.jpg',quality=98)
video=COVER.parent/'2026-09-26_combined/exports/Full_demo_combined_v3.mp4';cap=cv2.VideoCapture(str(video));sample=[0,60,75,90,105,119,135,165,210,299,10654,10684,10714,10773];sheet=Image.new('RGB',(1920,540*7))
for i,n in enumerate(sample):
 cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,f=cap.read();assert ok
 for x1,y1,x2,y2 in frames[str(n)]:cv2.rectangle(f,(x1,y1),(x2-1,y2-1),(0,0,255),1)
 im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));im.save(P/f'check_{n:05d}.jpg',quality=98);im=im.resize((960,540));ImageDraw.Draw(im).text((4,4),f'frame {n}',fill='yellow');sheet.paste(im,(i%2*960,i//2*540))
sheet.save(P/'full_mask_review.jpg',quality=95)
print('frames',len(frames),'masks',sum(map(len,frames.values())))
