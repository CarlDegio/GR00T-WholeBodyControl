import cv2, pathlib, math,json
from PIL import Image,ImageDraw
out=pathlib.Path('demo_video/2026-09-26_combined/privacy_redaction_v4/t4');c=cv2.VideoCapture('demo_video/2026-09-26_combined/exports/Full_demo_combined_v3.mp4')
groups={'a':(141.2,143.21,.1),'b':(151.3,152.31,1/30),'c':(154.55,155.36,1/30),'d':(155.9,156.26,1/30),'e':(156.6,157.41,1/30),'f':(157.4,159.41,.1),'g':(164.1,175.21,.3)}
manifest={}
for name,(a,b,step) in groups.items():
 ns=sorted(set(round((a+i*step)*60) for i in range(math.ceil((b-a)/step))));manifest[name]=ns
 for page in range(math.ceil(len(ns)/20)):
  im=Image.new('RGB',(1770,1000),'#222');dr=ImageDraw.Draw(im)
  for j,n in enumerate(ns[page*20:(page+1)*20]):
   c.set(cv2.CAP_PROP_POS_FRAMES,n);ok,f=c.read()
   crop=Image.fromarray(cv2.cvtColor(f[342:566,1486:1840],cv2.COLOR_BGR2RGB));crop.save(out/f'f{n}.png')
   x=j%5*354;y=j//5*250;im.paste(crop,(x,y));dr.text((x+4,y+225),f'{n} | {n/60:.3f}',fill='white')
  im.save(out/f'{name}_{page}.jpg',quality=95)
json.dump(manifest,open(out/'manifest.json','w'))
