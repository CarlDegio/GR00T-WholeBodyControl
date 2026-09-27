from pathlib import Path
import json,hashlib,subprocess,zipfile
B=Path(__file__).resolve().parents[1]
p=json.load(open(B/'edit/selection.json'));old=json.load(open(B/'versions/v1_before_replace_16_with_17/selection.json'));v=json.load(open(B/'versions/v1_before_replace_16_with_17/verification.json'))
changes=[(r,c,old['grid16'][r][c],p['grid16'][r][c]) for r in range(4) for c in range(4) if old['grid16'][r][c]!=p['grid16'][r][c]]
assert changes==[(2,1,16,17)] and old['grid25']==p['grid25'] and old['clips']==p['clips']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
for item in v['files']:
 path=B/item['path'];h=sha(path)
 if item['path']!='exports/cover_16_5s.mp4':assert h==item['sha256'],item['path'];continue
 d=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]));s=d['streams'][0]
 assert s['width']==1920 and s['height']==1080 and s['r_frame_rate']=='60/1' and int(s['nb_frames'])==300 and abs(float(d['format']['duration'])-5)<.001
 result=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(path),'-f','null','-'],capture_output=True,text=True);assert result.returncode==0 and not result.stderr.strip(),result.stderr
 item['sha256']=h
v['curated16']=sum(p['grid16'],[]);v['revision']='Replace clip 16 with clip 17 in 16-grid only';v['revision_checks']={'grid_changes':changes,'unchanged_videos_hash_checked':26,'updated_video_full_decode':'pass'}
(B/'edit/verification.json').write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
paths=[B/c['path'] for c in p['clips']]+[B/'exports'/f'cover_{n}_5s.mp4' for n in (25,16)]
with zipfile.ZipFile(B/'demo_cover_delivery.zip','w',compression=zipfile.ZIP_STORED) as z:
 for path in [B/'README.md',B/'gallery.html',B/'selection.csv',B/'selected_25_contact_sheet.jpg']+paths+list((B/'exports').glob('*.jpg'))+list((B/'review/thumbs').glob('*.jpg'))+list((B/'edit').glob('*.json'))+list((B/'edit').glob('*.py')):z.write(path,str(path.relative_to(B)))
print(json.dumps({'status':'PASS','replacement':'16 -> 17','duration':5,'frames':300,'unchanged_videos':26,'zip_updated':True}))
