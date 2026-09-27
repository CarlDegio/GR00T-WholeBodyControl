from pathlib import Path
from PIL import Image,ImageDraw
import json,cv2
D=Path(__file__).parent;dat=json.loads((D/'masks.json').read_text());c=cv2.VideoCapture(str(D.parents[1]/'exports/Full_demo_combined_v3.mp4'))
sets={'final_switch':[i/60 for i in range(1299,1308)]}
for label,ts in sets.items():
 for st in range(0,len(ts),9):
  sh=Image.new('RGB',(1608,984),'white');d=ImageDraw.Draw(sh)
  for j,t in enumerate(ts[st:st+9]):
   idx=round(t*60);c.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,f=c.read()
   im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));dr=ImageDraw.Draw(im)
   for b in dat['frames'].get(str(idx),[]):dr.rectangle(b,outline='red',width=3)
   if label=='t4':box=(1002,616,1806,1068)
   elif label=='faces':
    if t<13:box=(114,616,918,1068)
    elif 23.533<t<23.9 or 33.75<t<34.3:box=(1002,616,1806,1068)
    else:box=(1002,112,1806,564)
   else:box=(1002,112,1806,564)
   im=im.crop(box).resize((536,301));x=j%3*536;y=j//3*328;sh.paste(im,(x,y+27));d.text((x+6,y+5),f'{t:.3f}s frame{idx}',fill='black')
  sh.save(D/f'check_{label}_{st//9:02}.jpg',quality=94)
