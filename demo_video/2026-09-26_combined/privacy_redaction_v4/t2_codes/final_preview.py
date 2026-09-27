import cv2,json,pathlib
from PIL import Image,ImageDraw
out=pathlib.Path('demo_video/2026-09-26_combined/privacy_redaction_v4/t2_codes');m=json.load(open(out/'masks.json'))['frames'];c=cv2.VideoCapture('demo_video/2026-09-26_combined/exports/Full_demo_combined_v3.mp4');ns=[4352,4354,4360,4362,4364,4368,4370,5238,5460,5640,5940,6240,6390,6398,6402,6480,6888,6908,6910,6916,6920,6936,6972,7020];im=Image.new('RGB',(1560,960),'#222');dr=ImageDraw.Draw(im)
for j,n in enumerate(ns):
 c.set(1,n);ok,f=c.read();bs=m.get(str(n),[])
 if not bs:continue
 b=bs[0];cx=(b[0]+b[2])//2;cy=(b[1]+b[3])//2;xx=max(0,min(1660,cx-130));yy=max(0,min(870,cy-105));cr=Image.fromarray(cv2.cvtColor(f[yy:yy+210,xx:xx+260],cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(cr)
 for b in bs:d.rectangle((b[0]-xx,b[1]-yy,b[2]-xx,b[3]-yy),outline='red',width=2)
 x=j%6*260;y=j//6*240;im.paste(cr,(x,y));dr.text((x+3,y+212),f'{n} {n/60:.3f}s',fill='white')
im.save(out/'final_validation.jpg',quality=95)
