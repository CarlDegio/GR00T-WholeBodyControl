from prepare import *
import json,sys
M=json.loads((P/'masks.json').read_text())['frames']
G=json.loads((P/'per_target.json').read_text())
def check(name,times,roi,scale,cols=2,rows=8):
 x1,y1,x2,y2=roi;w=round((x2-x1)*scale);h=round((y2-y1)*scale)
 for st in range(0,len(times),cols*rows):
  ts=times[st:st+cols*rows];im=Image.new('RGB',(w*cols,(h+24)*((len(ts)+cols-1)//cols)),(20,24,30));d=ImageDraw.Draw(im)
  for j,t in enumerate(ts):
   src=frame(float(t)).copy();fid=str(round(t*60))
   for k,b in enumerate(M.get(fid,[])):
    cv2.rectangle(src,(b[0],b[1]),(b[2]-1,b[3]-1),(20,40,255),2)
   src=cv2.resize(src[y1:y2,x1:x2],(w,h));x=(j%cols)*w;y=(j//cols)*(h+24)
   im.paste(Image.fromarray(cv2.cvtColor(src,cv2.COLOR_BGR2RGB)),(x,y+24));d.text((x+4,y+3),f'{t:.3f}s / frame {round(t*60)}',fill='white')
  im.save(P/f'{name}_{st//(cols*rows)}.jpg',quality=95)
if __name__=='__main__':
 check('qcwhite',list(np.arange(61.5,70.25,.25)),(0,100,1420,400),.6,2,8)
 check('qcgray',list(np.arange(69.5,80.6,.25)),(0,100,1420,400),.6,2,8)
 check('qcchest',list(np.r_[np.arange(67.65,68.21,.05),np.arange(69.9,71.11,.05),np.arange(73.9,74.76,.05),np.arange(76.85,79.16,.05),np.arange(85.75,86.01,.05)]),(1488,343,1836,565),1.2,4,5)
