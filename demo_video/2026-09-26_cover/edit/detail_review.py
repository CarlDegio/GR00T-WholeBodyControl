from pathlib import Path
import json,shutil,hashlib,subprocess
from PIL import Image,ImageDraw,ImageFont
import cv2
R=Path('/home/user/Project/GR00T-WholeBodyControl');B=R/'demo_video/2026-09-26_cover';L=R/'demo_video/2026-09-25/assets';m=json.load(open(L/'manifest.json'))
p=Path('/home/user/.codex/attachments/57436a0d-d21e-420f-a41b-0323452ecfe3/625179984fa27c292e5304d094bac643.mp4')
digest=hashlib.sha256()
with p.open('rb') as inp:
 for block in iter(lambda:inp.read(1048576),b''):digest.update(block)
h=digest.hexdigest();match=next((v for v in m if v['sha256']==h),None)
if not match:
 target=L/'originals'/('22_'+p.name);shutil.copy2(p,target);match=dict(order=22,original_name=p.name,source_path=str(p),local_path=str(target.relative_to(R)),sha256=h,task='T4',kind='full_demo',status='active');m.append(match);(L/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
def sheet(n,times,label,rotate=False):
 p=R/next(v for v in m if v['order']==n)['local_path'];c=cv2.VideoCapture(str(p));c.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
 im=Image.new('RGB',(1280,((len(times)+3)//4)*204+30),'#121820');d=ImageDraw.Draw(im);d.text((6,4),f'SOURCE {n} {label}',font=f,fill='white')
 for i,t in enumerate(times):
  c.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,fr=c.read()
  if not ok:continue
  if rotate:fr=cv2.rotate(fr,cv2.ROTATE_180)
  pic=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));pic.thumbnail((320,180));x=i%4*320;y=i//4*204+30;im.paste(pic,(x,y));d.text((x+5,y+181),f'{t:.2f}s',font=f,fill='white')
 im.save(B/'review'/f'detail_{label}.jpg');c.release()
sheet(22,list(range(0,361,24)),'t4_overview',True)
sheet(21,[40,42,44,46,48,50,106,108,110,112,114,116,246,250,254,258,262,266,270,334],'t2_nav',True)
sheet(3,[0,2,4,6,8,10,12,14,16,18,20,22,28,30,32,34,36,38,40,42],'t1_align',True)
sheet(13,list(range(0,20)),'t3_align')
for n in [1,21,22]:
 p=R/next(v for v in m if v['order']==n)['local_path'];subprocess.run(['ffmpeg','-v','error','-ss','3','-i',str(p),'-frames:v','1','-y',str(B/'review'/f'ffmpeg_{n}.jpg')],check=True)
sheet(22,[10,11,12,14,16,18,19,20,41,42,43,44,45,46,47,48,232,233,234,235,236,237,238,239],'t4_navigation',True)
sheet(22,[291,293,295,297,299,301,303,305,307,309,311,313,315,317,319,321],'t4_alignment',True)
sheet(21,list(range(392,421,2)),'t2_alignment',True)
sheet(3,list(range(30,46)),'t1_final_approach',True)
sheet(8,list(range(8,20))+list(range(45,57)),'t2_approaches')
