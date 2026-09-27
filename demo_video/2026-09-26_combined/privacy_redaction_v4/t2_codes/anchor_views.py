import cv2,json,pathlib
from PIL import Image,ImageDraw
out=pathlib.Path('demo_video/2026-09-26_combined/privacy_redaction_v4/t2_codes');d=json.load(open(out/'tracked.json'))['tracks'];c=cv2.VideoCapture('demo_video/2026-09-26_combined/exports/Full_demo_combined_v3.mp4')
items=[]
for t in range(71,118):
 n=t*60;c.set(1,n);ok,f=c.read()
 if t<81:
  b=d['foreground'][str(n)];cx=(b[0]+b[2])/2-40;cy=(b[1]+b[3])/2+120
 elif t<96:
  b=d['background'][str(n)] if t>=82 else d['background_back'][str(n)];cx=(b[0]+b[2])/2-35;cy=(b[1]+b[3])/2-20
 else:
  b=d['background'][str(n)];cx=(b[0]+b[2])/2+45;cy=(b[1]+b[3])/2+90
 if t>=115:cx=1140 if t==115 else 1035;cy=840
 xx=max(0,min(1560,int(cx)-180));yy=max(0,min(780,int(cy)-150));im=Image.fromarray(cv2.cvtColor(f[yy:yy+300,xx:xx+360],cv2.COLOR_BGR2RGB));dr=ImageDraw.Draw(im)
 for x in range((xx//50+1)*50,xx+360,50):dr.line((x-xx,0,x-xx,300),fill='#999999');dr.text((x-xx,2),str(x),fill='yellow')
 for y in range((yy//50+1)*50,yy+300,50):dr.line((0,y-yy,360,y-yy),fill='#999999');dr.text((2,y-yy),str(y),fill='yellow')
 items.append((t,im,xx,yy))
for page in range((len(items)+8)//9):
 im=Image.new('RGB',(1080,990),'#222');dr=ImageDraw.Draw(im)
 for j,(t,cr,xx,yy) in enumerate(items[page*9:(page+1)*9]):
  x=j%3*360;y=j//3*330;im.paste(cr,(x,y));dr.text((x+5,y+303),f'{t}sec crop({xx},{yy})',fill='white')
 im.save(out/f'anchors_{page}.jpg',quality=96)
