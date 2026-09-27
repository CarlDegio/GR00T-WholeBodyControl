from pathlib import Path
import cv2,json,numpy as np,math
from PIL import Image,ImageDraw
P=Path(__file__).parent;B=P.parent;cv2.setNumThreads(2)
d=json.loads((P/'input_per_target.json').read_text());cap=cv2.VideoCapture(str(B.parent/'exports/Full_demo_combined_v3.mp4'));cap.set(1,3693);frames={}
for n in range(3693,4831):
 ok,im=cap.read();assert ok;frames[n]=im[:580,:1426].copy()
# Real-frame exemplars from independently checked correctly positioned boxes.
groups=[('main_white',3693,3779,[3705,3713,3745,3773]),('main_white',4005,4073,[4005,4008,4011,4017,4023,4045,4057,4063,4068]),('main_white_b',4086,4110,[4093,4099,4107]),('main_gray_early',4065,4090,[4069,4075,4081,4083]),('main_gray_early_b',4090,4114,[4095,4097,4101]),('main_gray',4335,4361,[4335,4341,4345,4353,4357,4359]),('main_gray_c',4369,4390,[4373,4375,4377,4383]),('main_white_revisit',4353,4379,[4357,4359,4363,4367,4373])]
records=[]
for name,lo,hi,ns in groups:
 bank=[]
 for n in ns:
  box=d[name].get(str(n));
  if not box:continue
  x1,y1,x2,y2=box;pad=15
  # Core manual face box, excluding safety padding.
  x1+=pad;y1+=pad;x2-=pad;y2-=pad
  if x2-x1<15 or y2-y1<20:continue
  bank.append((n,frames[n][y1:y2,x1:x2].copy()))
 for n in range(lo,hi+1):
  old=d[name].get(str(n))
  if not old:continue
  x1,y1,x2,y2=old;cx=(x1+x2)/2;cy=(y1+y2)/2;best=None
  for bn,patch in bank:
   h,w=patch.shape[:2];xx1=max(0,int(cx-w/2-230));xx2=min(1426,int(cx+w/2+230));yy1=max(100,int(cy-h/2-115));yy2=min(580,int(cy+h/2+115))
   if xx2-xx1<w or yy2-yy1<h:continue
   sc=cv2.matchTemplate(frames[n][yy1:yy2,xx1:xx2],patch,cv2.TM_CCOEFF_NORMED);_,score,_,loc=cv2.minMaxLoc(sc)
   val=score-abs(n-bn)*.0003
   if best is None or val>best[0]:best=(val,xx1+loc[0],yy1+loc[1],w,h,bn,score)
  if best:
   val,x,y,w,h,bn,score=best
   b=[max(0,x-16),max(100,y-16),min(1426,x+w+16),min(580,y+h+16)]
   d[name][str(n)]=b;records.append(dict(n=n,name=name,box=b,old=old,score=score,bank=bn))
# Render all refined frames, including original top-region context plus enlarged target box context.
(P/'match_records.json').write_text(json.dumps(records,indent=2));(P/'per_target.json').write_text(json.dumps(d))
res={}
for name,ds in d.items():
 for n,b in ds.items():res.setdefault(n,[]).append(b)
(P/'masks.json').write_text(json.dumps(dict(frames=res,targets=list(d),notes=['Independent main-view full-top-area review; real-frame template corrections in fast pans; title remains covered by existing source overlay.'])))
for a in range(0,len(records),24):
 sh=Image.new('RGB',(1800,1320));dr=ImageDraw.Draw(sh)
 for i,r in enumerate(records[a:a+24]):
  n=r['n'];b=r['box'];im=frames[n].copy();x1,y1,x2,y2=b;cv2.rectangle(im,(x1,y1),(x2-1,y2-1),(0,0,255),2)
  cx=(x1+x2)//2;cy=(y1+y2)//2;xx=max(0,min(1126,cx-150));yy=max(0,min(380,cy-100));crop=im[yy:yy+200,xx:xx+300]
  x=i%6*300;y=i//6*330;sh.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),(x,y+25));dr.text((x+3,y+3),f'{n} {r["name"]} {r["score"]:.2f}',fill='white')
  context=cv2.resize(im,(300,122));sh.paste(Image.fromarray(cv2.cvtColor(context,cv2.COLOR_BGR2RGB)),(x,y+208))
 sh.save(P/f'refined_{a//24:02d}.jpg',quality=97)
print('matched',len(records),'pages',math.ceil(len(records)/24))
