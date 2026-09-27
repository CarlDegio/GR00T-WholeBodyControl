from pathlib import Path
from PIL import Image,ImageDraw
import cv2
D=Path(__file__).parent;parts=D.parents[1]/'parts'
samples=[(5,3),(6.5,3),(12.45,3),(14.375,2),(14.625,2),(21.68,2),(22.125,2),(23.55,4),(23.55,2),(24,2),(30,2),(33.95,4),(34.2,4),(35.58,2),(43.125,2),(45.28,2),(45.28,4)]
c=cv2.VideoCapture(str(parts/'prefix.mp4'))
for st in range(0,len(samples),9):
 sh=Image.new('RGB',(1608,984),'white');dr=ImageDraw.Draw(sh)
 for j,(t,panel) in enumerate(samples[st:st+9]):
  c.set(1,round(t*60));ok,f=c.read();assert ok
  x,y={2:(1002,112),3:(114,616),4:(1002,616)}[panel]
  im=Image.fromarray(cv2.cvtColor(f[y:y+452,x:x+804],cv2.COLOR_BGR2RGB)).resize((536,301));xx=j%3*536;yy=j//3*328;sh.paste(im,(xx,yy+27));dr.text((xx+5,yy+5),f'prefix global {t:.3f}s T{panel}',fill='black')
 sh.save(D/f'prefix_{st//9}.jpg',quality=97)
c=cv2.VideoCapture(str(parts/'tail.mp4'));ts=[141.4,142.5,151.5,154.7,156.1,157.,159.,165.,174.9]
sh=Image.new('RGB',(1392,810),'white');dr=ImageDraw.Draw(sh)
for j,t in enumerate(ts):
 c.set(1,round((t-118)*60));ok,f=c.read();assert ok
 im=Image.fromarray(cv2.cvtColor(f[343:565,1488:1836],cv2.COLOR_BGR2RGB)).resize((464,296));xx=j%3*464;yy=j//3*270;im=im.resize((464,244));sh.paste(im,(xx,yy+26));dr.text((xx+4,yy+5),f'tail global {t:.3f}s CHEST',fill='black')
sh.save(D/'tail_faces.jpg',quality=97)
for lab,t in [('prefix',.5),('tail',178.5)]:
 c=cv2.VideoCapture(str(parts/(lab+'.mp4')));c.set(1,round((t if lab=='prefix' else t-118)*60));ok,f=c.read();assert ok;cv2.imwrite(str(D/f'{lab}_mosaic.jpg'),f)
print('done')
