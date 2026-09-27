import cv2,json,pathlib,math
from PIL import Image,ImageDraw
out=pathlib.Path('demo_video/2026-09-26_combined/privacy_redaction_v4/t2_codes');tracks=json.load(open(out/'chosen.json'))['tracks'];c=cv2.VideoCapture('demo_video/2026-09-26_combined/exports/Full_demo_combined_v3.mp4');c.set(1,4260);items=[]
for n in range(4260,7080):
 ok,f=c.read()
 if n%2:continue
 t=n/60;fast=(72.35<t<73.05 or 86.05<t<88 or 114.7<t<117)
 if not fast and n%16:continue
 for name,data in tracks.items():
  if str(n) not in data:continue
  b=data[str(n)];x1,y1,x2,y2=b;cx=int((x1+x2)/2);cy=int((y1+y2)/2);xx=max(0,min(1660,cx-130));yy=max(0,min(900,cy-90));cr=Image.fromarray(cv2.cvtColor(f[yy:yy+180,xx:xx+260],cv2.COLOR_BGR2RGB));dr=ImageDraw.Draw(cr);dr.rectangle((x1-xx-20,y1-yy-20,x2-xx+20,y2-yy+20),outline='red',width=2);items.append((n,cr,xx,yy,name))
for page in range(math.ceil(len(items)/30)):
 im=Image.new('RGB',(1560,1040),'#222');dr=ImageDraw.Draw(im)
 for j,(n,cr,xx,yy,name) in enumerate(items[page*30:(page+1)*30]):
  x=j%6*260;y=j//6*208;im.paste(cr,(x,y));dr.text((x+4,y+181),f'{n/60:.3f} {n} ({xx},{yy}) {name[0]}',fill='white')
 im.save(out/f'audit_{page}.jpg',quality=94)
print(len(items),math.ceil(len(items)/30))
