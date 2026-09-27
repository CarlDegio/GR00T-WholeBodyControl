import cv2
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).parent
c=cv2.VideoCapture(str(D.parents[1]/'exports/Full_demo_combined_v3.mp4'))
# Draw grid on native panel coordinates. Frame labels do not shift image coordinates.
def sheets(times,panel,name):
 pos={2:(1002,112,1806,564),3:(114,616,918,1068),4:(1002,616,1806,1068)}[panel]
 for st in range(0,len(times),4):
  sh=Image.new('RGB',(1608,984),'white');d=ImageDraw.Draw(sh)
  for j,t in enumerate(times[st:st+4]):
   c.set(cv2.CAP_PROP_POS_MSEC,t*1000);ret,f=c.read();im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)).crop(pos);dr=ImageDraw.Draw(im)
   for x in range(0,804,50):dr.line((x,0,x,452),fill=(220,160,160),width=1);dr.text((x+2,5),str(x),fill='red')
   for y in range(0,452,50):dr.line((0,y,804,y),fill=(220,160,160),width=1);dr.text((2,y+2),str(y),fill='red')
   x=j%2*804;y=j//2*492;sh.paste(im,(x,y+40));d.text((x+10,y+10),f'T{panel} full {t:.3f}',fill='black')
  sh.save(D/f'{name}_{st//4:02}.jpg',quality=95)
sheets([5,8,11,12.5]+[13+i*.25 for i in range(12)]+[15.77,17,18.5,20,21]+[21.25+i*.25 for i in range(10)]+[24,26,28,30,32,34,36,38,40,41]+[41.5+i*.25 for i in range(15)],[2][0],'main')
sheets([5,10,15,16.37,18,21,23.5,23.54,28,32,34,38,42,44,45.28],4,'t4')
sheets([5,5.25,5.5,6,6.5,6.75,7,7.25,8,10,12.45],3,'t3')
sheets([13.625,13.875,14.125,21.375,21.625,21.875,41.875,42.125,42.375],2,'rotation')
sheets([23.533333,23.6,23.7,23.8,23.9,24,33.8,34.2],4,'face_edge')
