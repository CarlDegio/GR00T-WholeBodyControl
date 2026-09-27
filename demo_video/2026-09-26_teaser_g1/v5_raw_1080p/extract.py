from pathlib import Path
import json,bisect,subprocess
from PIL import Image,ImageDraw,ImageFont
import cv2,numpy as np
p=Path(__file__).resolve().parent
pts=[float(l.strip().split(',')[0]) for l in (p/'raw_frame_pts.csv').read_text().splitlines() if l.strip() and l.strip()[0].isdigit()]
assert len(pts)==21841,len(pts)
m=json.loads((p.parent/'frame_manifest.json').read_text());selected=[]
def extract(t,path):
 subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-ss',f'{max(0,t-0.000001):.6f}','-i',str(p/'source_raw_1080p.mp4'),'-frames:v','1',str(p/path)],check=True)
for r in m['frames']:
 i=r['index']
 if i==4:continue
 t=r['source_event_seconds'];j=0 if i==1 else bisect.bisect_left(pts,t)-1 if i in [2,5] else bisect.bisect_right(pts,t)
 selected.append(dict(index=i,frame_index_0based=j,pts=pts[j],event_seconds=t))
 extract(pts[j],f'source_frames/{i:02}.png')
j=bisect.bisect_left(pts,351);extract(pts[j],'background_0551.png');bgj=j
k=selected[-1]['frame_index_0based'];near=[]
sheet=Image.new('RGB',(7*265,455),'white');dr=ImageDraw.Draw(sheet)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
for idx,d in enumerate([-15,-10,-5,0,5,10,15]):
 j=k+d;fn=f'review/neighbor_{j}.png';extract(pts[j],fn)
 im=Image.open(p/fn);crop=im.crop((600,685,800,975));small=crop.resize((265,384))
 sheet.paste(small,(idx*265,65));dr.text((idx*265+5,5),f'{j}\n{pts[j]:.6f}s',font=font,fill='black')
 gray=cv2.cvtColor(np.array(im.crop((610,820,795,955))),cv2.COLOR_RGB2GRAY)
 near.append(dict(frame=j,pts=pts[j],path=fn,foot_sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())))
sheet.save(p/'review/feet_neighbors_raw.png')
(p/'selection.json').write_text(json.dumps(dict(frames=selected,background=dict(frame=bgj,pts=pts[bgj]),neighbors=near,source='source_raw_1080p.mp4',source_sha256='761be96372e386e45b94bea84ac186fb573553c25b77aeb18c9cd1817d63ac87'),indent=2))
print(json.dumps(dict(selected=selected,neighbors=near),indent=2))
