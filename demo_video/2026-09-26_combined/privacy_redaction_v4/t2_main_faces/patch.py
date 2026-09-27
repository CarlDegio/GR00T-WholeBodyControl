from pathlib import Path
import json,cv2
from PIL import Image,ImageDraw
P=Path(__file__).parent;d=json.loads((P/'per_target.json').read_text())
for n in [3693,3694,3695]:d['main_white'].pop(str(n),None)
for n in [4012,4013]:d['main_white'][str(n)]=[325,100,420,150]
d['main_gray_early'].pop('4065',None)
for n in [4066,4067]:d['main_gray_early'][str(n)]=[388,255,435,327]
for n in range(4106,4115):d['main_gray_early_b'].pop(str(n),None)
for n in [4360,4361]:d['main_gray'][str(n)]=[0,212,40,293]
for n in [4370,4371]:d['main_gray_c'][str(n)]=[0,190,48,290]
d['main_white_revisit'].pop('4353',None)
for n in [4354,4355]:d['main_white_revisit'][str(n)]=[1390,212,1426,284]
for n in [4378,4379]:d['main_white_revisit'][str(n)]=[1360,164,1426,244]
for n,cx,cy in [(4070,1330,280),(4072,1352,278),(4074,1378,323),(4076,1343,344),(4078,1327,382),(4080,1232,419),(4082,1188,457),(4084,1181,451)]:
 for k in [n,n+1]:d['main_white'][str(k)]=[cx-39,cy-40,min(1426,cx+39),cy+40]
(P/'per_target.json').write_text(json.dumps(d));res={}
for name,ds in d.items():
 for n,b in ds.items():res.setdefault(n,[]).append(b)
(P/'masks.json').write_text(json.dumps(dict(frames=res,targets=list(d),notes=['Independent full top region review; 306 actual target frames in fast pans checked after template matching; manual partial-face corrections; original title overlay preserved.'])))
items=[(n,'main_white',0,400) for n in [4012,4013]]+[(n,'main_gray_early',280,480) for n in [4066,4067]]+[(n,'main_white',1080,580) for n in range(4070,4086)]+[(n,'main_gray',0,400) for n in [4360,4361]]+[(n,'main_gray_c',0,400) for n in [4370,4371]]+[(n,'main_white_revisit',1100,400) for n in [4354,4355,4378,4379]]
cap=cv2.VideoCapture(str(P.parent.parent/'exports/Full_demo_combined_v3.mp4'))
for a in range(0,len(items),12):
 sh=Image.new('RGB',(1800,1260));dr=ImageDraw.Draw(sh)
 for i,(n,name,xlo,yhi) in enumerate(items[a:a+12]):
  cap.set(1,n);ok,im=cap.read();assert ok
  x1,y1,x2,y2=d[name][str(n)];cv2.rectangle(im,(x1,y1),(x2-1,y2-1),(0,0,255),2)
  # Fit native 400x350 near target, with coordinate header.
  cx=(x1+x2)//2;cy=(y1+y2)//2;xx=max(0,min(1026,cx-200));yy=max(0,min(230,cy-175));crop=im[yy:yy+350,xx:xx+400]
  x=i%4*450;y=i//4*420;sh.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),(x,y+25));dr.text((x+4,y+4),f'{n} {name}: {xx},{yy}',fill='white')
 sh.save(P/f'patchcheck_{a//12}.jpg',quality=98)
print(len(res))
