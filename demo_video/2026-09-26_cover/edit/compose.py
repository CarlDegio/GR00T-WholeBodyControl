from pathlib import Path
import json,subprocess,math,sys
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
cv2.setNumThreads(2)
B=Path(__file__).resolve().parents[1];P=json.load(open(B/'edit/selection.json'));W,H=1920,1080;FPS=60
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
C={c['id']:c for c in P['clips']}
def reader(path):return cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
def title_assets():
 text=Image.new('RGBA',(W,H));d=ImageDraw.Draw(text)
 for i,line in enumerate(P['title']):
  font=ImageFont.truetype(BOLD if i==0 else FONT,66 if i==0 else 62)
  assert d.textbbox((0,0),line,font=font)[2]<W-150
  d.text((W/2,503+i*91),line,font=font,fill=(255,255,255,255),anchor='mm',stroke_width=0)
 return text
TITLE=title_assets()
yy,xx=np.mgrid[0:H,0:W];SHADE=np.exp(-((yy-H/2)/210)**4)*np.exp(-((xx-W/2)/1100)**4)*.58

def compose(n):
 grid=P[f'grid{n}'];side=len(grid);ids=sum(grid,[]);caps={i:reader(B/C[i]['path']) for i in ids}
 assert all(c.isOpened() for c in caps.values())
 cw,ch=640,360;canvas=np.zeros((side*ch,side*cw,3),np.uint8);start=3 if side==5 else 2
 output=B/'exports'/f'cover_{n}_5s.mp4'
 cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(output)]
 enc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 contact=Image.new('RGB',(960*3,540*2),'#121820');draw=ImageDraw.Draw(contact);sf=ImageFont.truetype(FONT,23)
 sample_frames=[0,59,90,119,150,239]
 for frame in range(300):
  for row in range(side):
   for col in range(side):
    id=grid[row][col];ok,im=caps[id].read()
    if not ok:raise RuntimeError(f'Short clip {id} at {frame}')
    im=cv2.resize(im,(cw,ch),interpolation=cv2.INTER_AREA)
    canvas[row*ch:(row+1)*ch,col*cw:(col+1)*cw]=im
  # Narrow 2px dividers scale continuously with the video wall.
  for row in range(1,side):canvas[row*ch-1:row*ch+1,:]=(14,17,20)
  for col in range(1,side):canvas[:,col*cw-1:col*cw+1]=(14,17,20)
  t=frame/FPS;u=min(1,max(0,t-1));ease=u*u*(3-2*u)
  visible=start+(side-start)*ease;zoom=W/(visible*cw)
  mat=np.float32([[zoom,0,W/2-zoom*canvas.shape[1]/2],[0,zoom,H/2-zoom*canvas.shape[0]/2]])
  image=cv2.warpAffine(canvas,mat,(W,H),flags=cv2.INTER_LINEAR)
  if t>=2:
   a=min(1,(t-2));smooth=a*a*(3-2*a)
   image=(image.astype(np.float32)*(1-smooth*(SHADE[:,:,None]+.08))).clip(0,255).astype(np.uint8)
   layer=TITLE.filter(ImageFilter.GaussianBlur(18*(1-smooth))) if smooth<1 else TITLE
   l=np.array(layer);alpha=l[:,:,3:4].astype(np.float32)/255*smooth
   image=(image*(1-alpha)+l[:,:,:3][:,:,::-1]*alpha).clip(0,255).astype(np.uint8)
  enc.stdin.write(image.tobytes())
  if frame in sample_frames:
   k=sample_frames.index(frame);thumb=Image.fromarray(cv2.cvtColor(cv2.resize(image,(960,540)),cv2.COLOR_BGR2RGB));contact.paste(thumb,(k%3*960,k//3*540));draw.rectangle((k%3*960,k//3*540,k%3*960+145,k//3*540+32),fill='#121820');draw.text((k%3*960+8,k//3*540+2),f'{t:.2f} s',font=sf,fill='white')
  if frame==239:Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).save(output.with_suffix('.jpg'),quality=95)
  if frame%60==0:print(f'{n} grid: {frame}/300',flush=True)
 enc.stdin.close();assert enc.wait()==0
 for c in caps.values():c.release()
 contact.save(B/'review'/f'cover_{n}_timeline.jpg',quality=92)
 print('DONE',output,flush=True)
if __name__=='__main__':
 for n in (list(map(int,sys.argv[1:])) or [25,16]):compose(n)
