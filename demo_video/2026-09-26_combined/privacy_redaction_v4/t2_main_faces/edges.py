from pathlib import Path
import cv2
from PIL import Image,ImageDraw
P=Path(__file__).parent;cap=cv2.VideoCapture(str(P.parent.parent/'exports/Full_demo_combined_v3.mp4'))
items=[(n,0,100,360,360) for n in [3693,3694,3695]]+[(n,200,60,600,300) for n in [4012,4013]]+[(n,0,170,650,480) for n in [4064,4065,4066,4067]]+[(n,1080,170,1426,560) for n in range(4070,4086,2)]+[(n,0,120,200,340) for n in [4360,4361,4370,4371]]+[(n,1250,130,1426,350) for n in [4352,4353,4354,4355,4378,4379,4380,4381]]
for a in range(0,len(items),12):
 sh=Image.new('RGB',(1800,1320));dr=ImageDraw.Draw(sh)
 for i,(n,x1,y1,x2,y2) in enumerate(items[a:a+12]):
  cap.set(1,n);ok,im=cap.read();assert ok;im=im[y1:y2,x1:x2];scale=min(450/(x2-x1),300/(y2-y1));im=cv2.resize(im,None,fx=scale,fy=scale)
  x=i%4*450;y=i//4*440;sh.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(x,y+30));dr.text((x+5,y+5),f'{n}: xy {x1},{y1} scale {scale:.3f}',fill='white')
 sh.save(P/f'edges_{a//12}.jpg',quality=98)
