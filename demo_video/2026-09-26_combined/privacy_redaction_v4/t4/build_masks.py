import json,math,pathlib
from PIL import Image,ImageDraw
import numpy as np
out=pathlib.Path('demo_video/2026-09-26_combined/privacy_redaction_v4/t4');frames={};targets=[]
def put(n,c,w=44,h=46):
 x,y=c;b=[1486+max(0,round(x-w/2)),342+max(0,round(y-h/2)),1486+min(354,round(x+w/2)),342+min(224,round(y+h/2))]
 if b[0]<b[2] and b[1]<b[3]:frames.setdefault(str(n),[]).append(b)
def exact(start,centers,w=44,h=46):
 for i,v in enumerate(centers):
  if v is None:continue
  cs=v if isinstance(v[0],tuple) else [v]
  n=(start+2*i)//2*2
  for k in [n,n+1]:
   for c in cs:put(k,c,w,h)
def interp(points,w=50,h=52):
 points=sorted(points)
 for n in range(points[0][0],points[-1][0]+2):
  x=np.interp(n,[p[0] for p in points],[p[1] for p in points]);y=np.interp(n,[p[0] for p in points],[p[2] for p in points]);put(n,(x,y),w,h)
interp([(8478,24,85),(8484,110,82),(8490,170,80),(8496,190,83),(8502,180,103),(8508,171,105),(8514,168,106),(8520,166,112),(8526,166,110),(8532,162,106),(8538,162,107),(8544,163,108),(8550,163,108),(8556,163,107),(8562,161,101),(8568,166,98),(8574,174,93),(8580,174,72),(8586,165,52),(8592,155,35)],54,54)
exact(9082,[(0,113),(27,98),(98,101),(161,103),[(50,100),(258,107)],[(109,101),(321,102)],(198,104),(279,105),(337,100),None,None,None,None,None,(355,52),(344,54),(296,57),(246,60),(198,59),(165,62),(185,66),(185,75),(185,77),(128,48),(59,60)])
exact(9277,[(35,92),(57,71),(69,76),(74,75),(76,75),(66,72),[(49,73),(234,100)],[(24,74),(207,99)],(10,72),(8,78),(8,73),(10,73),(10,66),(14,65),(18,64),(18,65),(18,62),(18,65),(12,66),(0,70)])
exact(9360,[(26,68),(118,74),(186,86),(255,88),(350,96)])
exact(9402,[(336,63),(347,70),(354,90),(354,92),(354,93),(354,93),(354,93),(337,69),(323,76),(321,93),(321,96),(329,93),(310,70),(247,67),(177,60),(75,75),(54,70),(36,64),(34,62),(35,67),(36,65),(37,65)])
interp([(9446,36,64),(9450,37,66),(9456,37,65),(9462,37,65),(9468,37,65),(9474,36,70),(9480,38,65),(9486,39,64),(9492,40,65),(9498,41,66),(9504,43,65),(9510,43,64),(9516,44,65),(9522,44,65),(9528,44,64),(9534,42,58),(9540,28,57),(9546,15,55),(9552,0,56)],44,46)
g=[(90,107),(70,83),(88,110),(77,83),(74,95),(49,96),(29,102),(63,108),(86,119),(95,116),(93,119),(113,103),(105,103),(89,126),(100,101),(92,92),(102,96),(107,111),(98,107),(91,112),(93,111),(83,107),(96,111),(95,119),(94,108),(88,97),(93,88),(99,129),(98,113),(67,126),(108,110),(111,116),(106,116),(84,114),(65,156),(102,104),(130,104)]
# From 164.4 onward, one manually inspected keyframe per .3 second.
interp([(9852,105,100)]+[(9864+i*18,*p) for i,p in enumerate(g)]+[(10512,130,104)],54,58)
# Keep longest confirmed interval through 175.1; omit already occluded later frames.
frames={n:b for n,b in frames.items() if int(n)<=10506}
# Corrections from inspection of every actual changing frame, including camera jolts.
for a,b,dx,dy in [(9906,9915,39,0),(9946,9949,-14,-3),(9958,9961,-15,-5),(10126,10135,0,8),(10176,10181,-20,-18),(10416,10421,0,-8),(10462,10465,21,0)]:
 for n in range(a,b+1):
  if str(n) in frames:
   frames[str(n)]=[[x1+dx,y1+dy,x2+dx,y2+dy] for x1,y1,x2,y2 in frames[str(n)]]
targets=[{'type':'face' ,'panel':'CHEST','description':'蹲下/坐地人员与桌后观察人员；仅授权确认的人脸片段，快速转向按30fps实际变化逐张定位。','requested_ranges_seconds':[[141.30,143.20],[151.35,152.30],[154.60,155.30],[155.95,156.20],[156.65,159.30],[164.20,175.10]]}]
notes=['源为Full_demo_combined_v3.mp4,1920x1080,60fps。坐标全屏xyxy exclusive，帧号0基。','快速扫描逐个实际变化帧(30fps)检查；60fps成片相邻偶数/奇数画面重复，使用同一遮罩。','稳定片段人工关键帧插值，矩形约44–54×46–58像素，覆盖头脸并含安全余量；脸出画的空帧不加遮罩。','不处理未确认的纸箱标签/二维码及119、124秒保守候选。']
json.dump({'frames':frames,'targets':targets,'notes':notes},open(out/'masks.json','w'),ensure_ascii=False,indent=2)
manifest=json.load(open(out/'manifest.json'))
for name,ns in manifest.items():
 for page in range(math.ceil(len(ns)/20)):
  im=Image.new('RGB',(1770,1000),'#222');d=ImageDraw.Draw(im)
  for j,n in enumerate(ns[page*20:(page+1)*20]):
   cr=Image.open(out/f'f{n}.png');dr=ImageDraw.Draw(cr)
   for x1,y1,x2,y2 in frames.get(str(n),[]):dr.rectangle((x1-1486,y1-342,x2-1486-1,y2-342-1),outline='red',width=2)
   x=j%5*354;y=j//5*250;im.paste(cr,(x,y));d.text((x+4,y+225),f'{n} | {n/60:.3f}',fill='white')
  im.save(out/f'verify_{name}_{page}.jpg',quality=95)
print(len(frames),min(map(int,frames)),max(map(int,frames)))
