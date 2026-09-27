from pathlib import Path
from PIL import Image,ImageDraw
import cv2,json
D=Path(__file__).parent;targets=json.loads((D/'per_target.json').read_text());c=cv2.VideoCapture(str(D.parents[1]/'exports/Full_demo_combined_v3.mp4'))
sets={'densehead':list(range(4434,4496)), 'densechest':list(range(4376,4389))+list(range(4418,4430))+list(range(4560,4573))}
for lab,ids in sets.items():
 for off in range(0,len(ids),32):
  sh=Image.new('RGB',(1392,944),'white');dr=ImageDraw.Draw(sh)
  for j,i in enumerate(ids[off:off+32]):
   c.set(1,i);ok,f=c.read();im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
   for k,v in targets.items():
    if ('head' in lab)==k.startswith('head') and str(i) in v:d.rectangle(v[str(i)],outline='red',width=1)
   box=(1488,64,1836,154) if 'head' in lab else (1488,475,1836,565)
   im=im.crop(box);x=j%4*348;y=j//4*118;sh.paste(im,(x,y+28));dr.text((x+3,y+6),str(i),fill='black')
  sh.save(D/f'{lab}_{off//32}.jpg',quality=97)
