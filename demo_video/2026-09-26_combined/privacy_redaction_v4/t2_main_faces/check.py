from pathlib import Path
import cv2,json,numpy as np
from PIL import Image,ImageDraw
P=Path(__file__).parent; B=P.parent;cv2.setNumThreads(2)
d=json.loads((B/'t2/per_target.json').read_text()); (P/'input_per_target.json').write_text(json.dumps(d));(P/'masks.json').write_text((B/'t2/masks.json').read_text())
ns=set(range(3693,3781,2))|set(range(4005,4111,2))|set(range(4335,4381,2));ns|=set(range(3693,4831,15));ns=sorted(ns)
cap=cv2.VideoCapture(str(B.parent/'exports/Full_demo_combined_v3.mp4'));cap.set(1,min(ns));ims={}
for n in range(min(ns),max(ns)+1):
 ok,im=cap.read();assert ok
 if n not in ns:continue
 for name,boxes in d.items():
  if str(n) in boxes:
   x1,y1,x2,y2=boxes[str(n)];cv2.rectangle(im,(x1,y1),(x2-1,y2-1),(0,0,255),2)
 ims[n]=im[:550,:1426].copy()
for a in range(0,len(ns),12):
 sh=Image.new('RGB',(2139,1200));dr=ImageDraw.Draw(sh)
 for i,n in enumerate(ns[a:a+12]):
  x=i%3*713;y=i//3*300;sh.paste(Image.fromarray(cv2.cvtColor(ims[n],cv2.COLOR_BGR2RGB)).resize((713,275)),(x,y+25));dr.text((x+3,y+3),f'{n} / {n/60:.3f}s',fill='white')
 sh.save(P/f'top_{a//12:02d}.jpg',quality=97)
print(len(ns))
