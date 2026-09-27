from pathlib import Path
import cv2,json
from PIL import Image,ImageDraw
P=Path(__file__).parent;d=json.loads((P/'per_target.json').read_text());cap=cv2.VideoCapture(str(P.parent.parent/'exports/Full_demo_combined_v3.mp4'))
ns=list(range(4470,4740,12))+list(range(4818,4831,2));sh=Image.new('RGB',(1800,math_h:=((len(ns)+5)//6)*240));dr=ImageDraw.Draw(sh)
for i,n in enumerate(ns):
 cap.set(1,n);ok,im=cap.read();assert ok
 b=d['main_gray_c'].get(str(n));xx=0 if b is None else max(0,(b[0]+b[2])//2-150)
 crop=im[:210,xx:xx+300].copy()
 for yy in [25,100]:cv2.line(crop,(0,yy),(299,yy),(0,255,255),1)
 x=i%6*300;y=i//6*240;sh.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),(x,y+25));dr.text((x+3,y+3),f'{n} x={xx}',fill='white')
sh.save(P/'topedge.jpg',quality=98)
