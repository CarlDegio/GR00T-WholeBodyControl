import cv2,json,pathlib,math
from PIL import Image,ImageDraw
out=pathlib.Path('demo_video/2026-09-26_combined/privacy_redaction_v4/t2_codes');tracks=json.load(open(out/'tracked.json'))['tracks'];c=cv2.VideoCapture('demo_video/2026-09-26_combined/exports/Full_demo_combined_v3.mp4')
for name,data in tracks.items():
 items=[]
 for key,b in sorted(data.items(),key=lambda x:int(x[0])):
  n=int(key)
  if n%30!=0:continue
  c.set(1,n);ok,f=c.read();x1,y1,x2,y2=b;cx=int((x1+x2)/2);cy=int((y1+y2)/2);xx=max(0,min(1660,cx-130));yy=max(0,min(900,cy-90));cr=Image.fromarray(cv2.cvtColor(f[yy:yy+180,xx:xx+260],cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(cr);d.rectangle((x1-xx-10,y1-yy-10,x2-xx+10,y2-yy+10),outline='red',width=2);items.append((n,cr,xx,yy))
 for page in range(math.ceil(len(items)/30)):
  im=Image.new('RGB',(1560,1040),'#222');d=ImageDraw.Draw(im)
  for j,(n,cr,xx,yy) in enumerate(items[page*30:(page+1)*30]):
   x=j%6*260;y=j//6*208;im.paste(cr,(x,y));d.text((x+4,y+181),f'{n/60:.2f} {n} ({xx},{yy})',fill='white')
  im.save(out/f'{name}_{page}.jpg',quality=95)
