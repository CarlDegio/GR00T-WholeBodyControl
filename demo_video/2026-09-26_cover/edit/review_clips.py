from pathlib import Path
import json,cv2
from PIL import Image,ImageDraw,ImageFont
B=Path(__file__).resolve().parents[1];p=json.load(open(B/'edit/selection.json'));f=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
for page in range(5):
 im=Image.new('RGB',(1280,5*208),'#101820');d=ImageDraw.Draw(im)
 for row,c in enumerate(p['clips'][page*5:page*5+5]):
  cap=cv2.VideoCapture(str(B/c['path']),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);y=row*208
  d.text((5,y),f"{c['id']:02d} {c['label']} | {c['speed']:g}x | {c['start']:.1f}–{c['end']:.1f}s",font=f,fill='white')
  for i,t in enumerate([0,.99,2.49,4.8]):
   cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,a=cap.read();assert ok
   a=cv2.resize(a,(320,180));im.paste(Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)),(i*320,y+28))
  cap.release()
 im.save(B/'review'/f'selected_page_{page+1}.jpg',quality=92)
