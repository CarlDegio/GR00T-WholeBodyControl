from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent
V=P.parents[1]/'exports/Full_demo_combined_v3.mp4'
cv2.setNumThreads(2)
cap=cv2.VideoCapture(str(V),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
def frame(t):
 f=P/'frames'/f'{round(t*60):05d}.jpg'; f.parent.mkdir(exist_ok=True)
 if f.exists(): return cv2.imread(str(f))
 cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*60)); ok,im=cap.read();assert ok
 cv2.imwrite(str(f),im,[cv2.IMWRITE_JPEG_QUALITY,96]); return im

def sheet(name,times,roi,scale,cols=2,rows=8):
 x1,y1,x2,y2=roi;w=round((x2-x1)*scale);h=round((y2-y1)*scale)
 for st in range(0,len(times),cols*rows):
  ts=times[st:st+cols*rows];im=Image.new('RGB',(w*cols,(h+24)*((len(ts)+cols-1)//cols)),(20,24,30));d=ImageDraw.Draw(im)
  for j,t in enumerate(ts):
   src=frame(float(t))[y1:y2,x1:x2].copy()
   for x in range((x1//100+1)*100,x2,100):
    cv2.line(src,(x-x1,0),(x-x1,y2-y1),(110,95,50),1);cv2.putText(src,str(x),(x-x1+2,14),0,.4,(80,230,230),1)
   for y in range((y1//100+1)*100,y2,100):
    cv2.line(src,(0,y-y1),(x2-x1,y-y1),(110,95,50),1);cv2.putText(src,str(y),(2,y-y1-2),0,.4,(80,230,230),1)
   src=cv2.resize(src,(w,h));x=(j%cols)*w;y=(j//cols)*(h+24)
   im.paste(Image.fromarray(cv2.cvtColor(src,cv2.COLOR_BGR2RGB)),(x,y+24));d.text((x+4,y+3),f'{t:.3f}s / frame {round(t*60)} | ROI {roi}',fill='white')
  im.save(P/f'{name}_{st//(cols*rows)}.jpg',quality=94)
if __name__=='__main__':
 sheet('white',list(np.arange(61.5,70.51,.5)),(350,100,1420,350),.65,2,6)
 sheet('gray',list(np.arange(69.5,81.01,.5)),(0,90,1150,380),.65,2,6)
 sheet('codes',list(np.arange(71,118,1)),(0,160,1420,1080),.5,3,3)
 sheet('chestface',list(np.r_[np.arange(67.5,68.31,.1),np.arange(69.8,71.21,.1),np.arange(73.85,74.86,.1),np.arange(76.75,79.26,.1),np.arange(85.65,86.11,.05)]),(1488,343,1836,565),1.3,4,5)
 sheet('headqr',list(np.arange(73.5,79.01,.25)),(1488,64,1836,285),1.3,4,4)
 sheet('chestqr',list(np.arange(72.3,76.31,.2)),(1488,343,1836,565),1.3,4,4)
