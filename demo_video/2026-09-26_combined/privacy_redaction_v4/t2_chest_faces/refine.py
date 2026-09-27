from pathlib import Path
import runpy,json,math
import cv2,numpy as np
from PIL import Image,ImageDraw
cv2.setNumThreads(2)
P=Path(__file__).resolve().parent;BASE=P.parent;video=BASE.parent/'exports/Full_demo_combined_v3.mp4';scope={};exec((BASE/'t2/anchors.py').read_text().split('# Post-review corrections.')[0],scope);tracks=[t for t in scope['tracks'] if t['name'].startswith('chest_')]
cap=cv2.VideoCapture(str(video));frames={};result={};details=[]
for tr in tracks:
 anchors=tr['anchors'];lo=math.floor(anchors[0][0]*60);hi=math.ceil(anchors[-1][0]*60);cap.set(cv2.CAP_PROP_POS_FRAMES,lo)
 for n in range(lo,hi+1):
  ok,f=cap.read();assert ok;frames[n]=f[343:565,1488:1836].copy()
 bank=[]
 for t,cx,cy,w,h in anchors:
  n=round(t*60);x=int(cx-1488-w/2);y=int(cy-343-h/2)
  if x<0 or x+w>348 or y<0 or y+h>222:continue
  patch=frames[n][y:int(y+h),x:int(x+w)];bank.append((n,patch,w,h))
 for n in range(lo,hi+1):
  t=n/60;ax=np.array(anchors);pred=np.array([np.interp(t,ax[:,0],ax[:,i]) for i in range(1,5)]);pcx,pcy,pw,ph=pred;px=pcx-1488;py=pcy-343
  # Search around the coarse annotation, using several nearby real face exemplars.
  best=None;im=frames[n];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  for bn,patch,w,h in bank:
   if abs(bn-n)>36:continue
   x1=max(0,int(px-w/2)-105);x2=min(348,int(px+w/2)+105);y1=max(0,int(py-h/2)-50);y2=min(222,int(py+h/2)+50)
   if x2-x1<w or y2-y1<h:continue
   templ=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY);score=cv2.matchTemplate(gray[y1:y2,x1:x2],templ,cv2.TM_CCOEFF_NORMED);_,val,_,loc=cv2.minMaxLoc(score)
   # Prefer very close exemplars only modestly: rapid pan is the real motion.
   quality=val-abs(bn-n)*.0003
   if best is None or quality>best[0]:best=(quality,x1+loc[0]+w/2,y1+loc[1]+h/2,w,h,bn,val)
  if best and best[0]>.55:
   _,cx,cy,w,h,bn,score=best
  else:cx,cy,w,h=px,py,pw,ph;bn=None;score=0
  # Coarse out-of-view periods cannot pull masks onto unrelated interior objects.
  if px<-18 or px>366:cx=px;cy=py;score=0
  pad=12;x1=max(1488,math.floor(cx+1488-w/2-pad));y1=max(343,math.floor(cy+343-h/2-pad));x2=min(1836,math.ceil(cx+1488+w/2+pad));y2=min(565,math.ceil(cy+343+h/2+pad))
  boxes=[] if x2<=x1 or y2<=y1 else [[x1,y1,x2,y2]];result[str(n)]=boxes;details.append(dict(n=n,name=tr['name'],score=float(score),bank=bn,box=boxes,pred=pred.tolist()))
# Manual corrections from inspection of every actual frame: cropped faces at viewport edges.
edge_spans=[
(4092,4092,[1488,400,1512,459]),
(4232,4235,[1488,410,1517,466]),
(4236,4237,[1488,419,1500,480]),
(4256,4262,[1488,410,1520,472]),
(4263,4265,[1488,414,1512,474]),
(4434,4435,[1488,419,1523,473]),
(4468,4469,[1488,386,1517,456]),
(4472,4475,[1488,386,1522,455]),
(4612,4613,[1488,414,1522,464]),
(4626,4627,[1488,371,1520,427]),
(4634,4635,[1488,403,1520,462]),
(4656,4665,[1488,381,1522,436]),
(4682,4683,[1488,403,1522,481]),
(4690,4691,[1488,385,1524,457]),
(4742,4745,[1488,395,1525,463]),
(5146,5147,[1806,368,1836,418]),
(5158,5159,[1806,383,1836,434]),
]
absent_spans=[(4059,4059),(4093,4094),(4192,4193),(4238,4247),(4266,4266),(4433,4433),(4470,4471),(4482,4485),(4611,4611),(4628,4633),(4684,4689),(4746,4749),(5145,5145),(5160,5160)]
for lo,hi,box in edge_spans:
 for n in range(lo,hi+1):result[str(n)]=[box]
for lo,hi in absent_spans:
 for n in range(lo,hi+1):result[str(n)]=[]
for d in details:
 d['box']=result[str(d['n'])]
 d['manual_edge']=any(lo<=d['n']<=hi for lo,hi,_ in edge_spans)
 d['verified_absent']=any(lo<=d['n']<=hi for lo,hi in absent_spans)
(P/'masks.json').write_text(json.dumps({'frames':result,'targets':[t['name'] for t in tracks],'notes':['Real-frame local template matching; every one of 319 frames visually inspected; manually corrected partial faces at viewport edges. Final 1920x1080 exclusive xyxy.','CHEST viewport1488,343,1836,565; no QR masks.']},ensure_ascii=False,indent=2));(P/'details.json').write_text(json.dumps(details,indent=2))
# Every actual video frame is shown; no temporal sampling.
ns=sorted(frames)
for start in range(0,len(ns),20):
 sh=Image.new('RGB',(1740,1230))
 for k,n in enumerate(ns[start:start+20]):
  f=frames[n].copy()
  for x1,y1,x2,y2 in result[str(n)]:cv2.rectangle(f,(x1-1488,y1-343),(x2-1489,y2-344),(0,0,255),1)
  im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)).resize((435,278));x=k%4*435;y=k//4*246
  # Crop bottom irrelevant lower viewport to keep face large.
  sh.paste(im.crop((0,0,435,222)),(x,y+24));ImageDraw.Draw(sh).text((x+3,y+3),f'{n} / {n/60:.3f}s',fill='white')
 sh.save(P/f'qc_{start//20:02d}.jpg',quality=97)
print('frames',len(result),'pages',math.ceil(len(ns)/20),'low',[(x['n'],round(x['score'],2)) for x in details if x['score']<.55])
