import cv2
from PIL import Image,ImageDraw
from pathlib import Path
out=Path('demo_video/2026-09-26_combined/privacy_redaction_v4/t2_codes');c=cv2.VideoCapture('demo_video/2026-09-26_combined/exports/Full_demo_combined_v3.mp4');ts=[72.4,72.5,72.6,72.7,72.8,72.9]+[86+i*.2 for i in range(11)]+[114.8+i*.2 for i in range(12)];ims=[]
for t in ts:
 n=round(t*60);c.set(1,n);ok,f=c.read()
 if t<80:xx,yy,w,h=250,250,800,600
 elif t<90:xx,yy,w,h=0,100,800,600
 else:xx,yy,w,h=750,480,1000,600
 f=cv2.resize(f[yy:yy+h,xx:xx+w],(500,300));im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
 for x in range((xx//100+1)*100,xx+w,100):d.line(((x-xx)*500/w,0,(x-xx)*500/w,300),fill='#888888');d.text(((x-xx)*500/w,2),str(x),fill='yellow')
 for y in range((yy//100+1)*100,yy+h,100):d.line((0,(y-yy)/2,500,(y-yy)/2),fill='#888888');d.text((2,(y-yy)/2),str(y),fill='yellow')
 ims.append((t,im))
for page in range((len(ims)+5)//6):
 im=Image.new('RGB',(1000,990),'#222');d=ImageDraw.Draw(im)
 for j,(t,cr) in enumerate(ims[page*6:(page+1)*6]):
  x=j%2*500;y=j//2*330;im.paste(cr,(x,y));d.text((x+5,y+304),f'{t:.2f}',fill='white')
 im.save(out/f'fast_{page}.jpg',quality=95)
