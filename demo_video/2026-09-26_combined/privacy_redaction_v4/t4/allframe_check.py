import cv2,json,pathlib,math
from PIL import Image,ImageDraw
out=pathlib.Path('demo_video/2026-09-26_combined/privacy_redaction_v4/t4');m=json.load(open(out/'masks.json'))['frames'];c=cv2.VideoCapture('demo_video/2026-09-26_combined/exports/Full_demo_combined_v3.mp4');c.set(1,8474);items=[]
for n in range(8474,10510):
 ok,f=c.read()
 if n%2 or str(n) not in m:continue
 for b in m[str(n)]:
  x1,y1,x2,y2=b;cx=(x1+x2)//2;cy=(y1+y2)//2;xx=max(0,cx-48);yy=max(0,cy-48)
  cr=Image.fromarray(cv2.cvtColor(f[yy:yy+96,xx:xx+96],cv2.COLOR_BGR2RGB));ImageDraw.Draw(cr).rectangle((x1-xx,y1-yy,x2-xx-1,y2-yy-1),outline='red',width=1);items.append((n,cr))
for page in range(math.ceil(len(items)/80)):
 im=Image.new('RGB',(960,928),'#222');d=ImageDraw.Draw(im)
 for j,(n,cr) in enumerate(items[page*80:(page+1)*80]):
  x=j%10*96;y=j//10*116;im.paste(cr,(x,y));d.text((x,y+96),str(n),fill='white')
 im.save(out/f'allframes_{page}.jpg',quality=95)
print('inspected samples ready',len(items))
