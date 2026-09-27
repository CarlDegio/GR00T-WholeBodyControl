from pathlib import Path
import json,hashlib,shutil,subprocess,concurrent.futures
import cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/home/user/Project/GR00T-WholeBodyControl'); BASE=ROOT/'demo_video/2026-09-26_cover'; LIB=ROOT/'demo_video/2026-09-25/assets'
names='f0dabf866ca20803b45ba933a38efcc4 002c4f32b436ae321e360722c75c49bb 1d8c39be8a83fbd3fc087c89e3c8b72f 1a7eade8191d6cdb46d12b311e2c75d5 1290e0e88e03cbccca3fde45e30d94df a9f2847cafb7a3512d184a5dd56696c3 6b4eed38edcb04f97ff2106bef5d6170 0455e030e9e64ba93f00c6de59941520 3ad48658ddfb34ff34c04327b0d3bf3e 2d46b2a0b5a91ccc8918f6a1db8011d3 9b42be712b7e89005e0882831ac19ec8 3dc7bdef5151147c57aaab6b0001404c 534ddd95ab57564c721c921f05473fc2 9f6dcddef4bebc74593fbd9b6fe57495 56cdf1e16d1a2c9bc7c85374218af9f0 51b7ad6a252ea9a75a88e5d18cce7f10 42786dc4a4d2311caf07b3e61f6cc369 893096baefaf7788af75cf9c84c7f8a8 ca3b07a8079e7c334bab52ce06d5c4c9 0627289735acdd246711f065a6e7b19c b80a27efc60e04b9dd7ec8cbdb30158a_raw'.split()
folders='48e99a29-5398-41b9-a762-d5c42f5f7c74 9221ec3c-0e0f-4e8f-908f-1ccc45912fe9 e182cb22-fead-44fe-8e93-730a3965c2e6 5cfd1e84-9fdf-4562-8c68-0a8669f715ad 2269cdb2-05f2-49d8-8964-a50ebe4c3aba 31d94f78-903e-4358-8af1-13b4cff1b92e 72525392-4327-4384-8fb6-440bf564fbdb ba6fef98-279f-458c-9e68-a3e3d3206594 606c8456-db5c-4ed5-8b03-311b268abc4a b50d102f-1a76-43cc-ba5e-7432c36e3656 23373c51-196c-41b8-aee9-e9180079ea14 56d9282a-efd6-4bcd-a641-50b03ceda38f 7943103d-454d-45c3-a3e9-6eaade4cd332 8abb6eaf-3db4-491b-a280-b7e0956ab14e 9e81eb91-757f-4720-9730-3acf10f96086 c49acc0a-5dcd-4375-9670-5ca3411e404b aa7ada40-3e31-4add-bea2-10d79c1b715d 43bc020d-f871-4ec6-a4f9-14b729c63ad9 952042a2-ab33-4596-9df6-0b37c2658280 afd52446-fec7-4a90-8aef-5c421902edd7 61aa00e7-8652-4597-b25e-834a2103790e'.split()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
m=json.loads((LIB/'manifest.json').read_text()); report=[]
for name,folder in zip(names,folders):
 p=Path('/home/user/.codex/attachments')/folder/(name+'.mp4'); h=sha(p)
 match=next((v for v in m if v['sha256']==h),None)
 if match: status='existing_identical'
 else:
  n=max(v['order'] for v in m)+1; target=LIB/'originals'/f'{n:02d}_{p.name}';shutil.copy2(p,target)
  assert sha(target)==h
  match=dict(order=n,original_name=p.name,source_path=str(p),local_path=str(target.relative_to(ROOT)),sha256=h,status='active');m.append(match);status='imported'
 report.append(dict(attachment=str(p),order=match['order'],status=status,sha256=h))
(LIB/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n');(BASE/'edit/import_report.json').write_text(json.dumps(report,indent=2))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
def sheet(v):
 p=ROOT/v['local_path'];cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS); dur=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
 v.update(duration=dur,width=int(cap.get(3)),height=int(cap.get(4)),fps=fps)
 im=Image.new('RGB',(1280,4*204+35),'#11151a');d=ImageDraw.Draw(im);d.text((8,5),f"SOURCE {v['order']:02d} | {p.name} | {dur:.2f}s",font=font,fill='white')
 for i in range(16):
  t=i*(dur-.15)/15;cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,f=cap.read()
  if not ok:continue
  pic=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));pic.thumbnail((320,180));x=(i%4)*320;y=(i//4)*204+35;im.paste(pic,(x+(320-pic.width)//2,y));d.text((x+5,y+181),f'{t:.1f}s',font=font,fill='white')
 cap.release();im.save(BASE/'review'/f"source_{v['order']:02d}.jpg")
 return v
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex: enriched=list(ex.map(sheet,m))
(BASE/'edit/sources.json').write_text(json.dumps(enriched,indent=2));print(json.dumps(report,indent=2));print([(v['order'],round(v['duration'],2),v['width'],v['height']) for v in enriched])
