from pathlib import Path
from PIL import Image,ImageDraw
import cv2,json
D=Path(__file__).parent
src=json.loads((D/'per_target.json').read_text());targets={k:v for k,v in src.items() if k.startswith(('head_qr','qr_chest'))}
(D/'per_target.json').write_text(json.dumps(targets,separators=(',',':')))
c=cv2.VideoCapture(str(D.parents[1]/'exports/Full_demo_combined_v3.mp4'))
for lab in ['head','chest']:
 starts=[(4430,4515),(4571,4587),(4604,4751)] if lab=='head' else [(4338,4392),(4412,4434),(4492,4575)]
 ids=set()
 for a,b in starts:ids.update(range(a,b,3));ids.update(range(a,a+8));ids.update(range(b-8,b))
 if lab=='chest':ids.update(range(4377,4389));ids.update(range(4415,4433))
 else:ids.update(range(4571,4587))
 ids=sorted(ids);x,y,r,b=(1488,64,1836,285) if lab=='head' else (1488,343,1836,565)
 for off in range(0,len(ids),16):
  sh=Image.new('RGB',(1392,992),'white');dr=ImageDraw.Draw(sh)
  for j,i in enumerate(ids[off:off+16]):
   c.set(1,i);ok,im=c.read();im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
   for k,v in targets.items():
    if (lab=='head')==k.startswith('head') and str(i) in v:d.rectangle(v[str(i)],outline='red',width=2)
   im=im.crop((x,y,r,b));xx=j%4*348;yy=j//4*248;sh.paste(im,(xx,yy+26));dr.text((xx+4,yy+5),f'{i} / {i/60:.4f}s',fill='black')
  sh.save(D/f'final_{lab}_{off//16:02}.jpg',quality=96)
 print(lab,len(ids))
