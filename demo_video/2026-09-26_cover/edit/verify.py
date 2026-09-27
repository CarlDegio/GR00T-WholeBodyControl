from pathlib import Path
import subprocess,json,hashlib,concurrent.futures,collections,zipfile
R=Path('/home/user/Project/GR00T-WholeBodyControl');B=R/'demo_video/2026-09-26_cover';p=json.load(open(B/'edit/selection.json'));manifest=json.load(open(R/'demo_video/2026-09-25/assets/manifest.json'))
ids={c['id'] for c in p['clips']};g25=sum(p['grid25'],[]);g16=sum(p['grid16'],[])
assert len(ids)==25 and len(g25)==len(set(g25))==25 and set(g25)==ids
assert len(g16)==len(set(g16))==16 and set(g16)<=ids and set(range(1,6))<=set(g16)
counts=collections.Counter(c['kind'] for c in p['clips']);assert counts==dict(navigation=4,scan=1,manipulation=10,alignment=10)
for task in ['T1','T2','T3','T4']:assert sum(c['task']==task and c['kind'] in ('manipulation','alignment') for c in p['clips'])==5
for c in p['clips']:assert abs((c['end']-c['start'])/c['speed']-5)<.001

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for s in manifest:assert sha(R/s['local_path'])==s['sha256'],s['order']
assert len({s['sha256'] for s in manifest})==len(manifest)==22

def check(path):
 d=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)]));v=next(v for v in d['streams'] if v['codec_type']=='video')
 assert v['codec_name']=='h264' and v['pix_fmt']=='yuv420p' and v['r_frame_rate']=='60/1'
 assert int(v['nb_frames'])==300 and abs(float(d['format']['duration'])-5)<.001
 assert not any(s['codec_type']=='audio' for s in d['streams'])
 assert (v['width'],v['height'])==((1920,1080) if path.parent.name=='exports' else (1280,720))
 result=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path),'-f','null','-'],capture_output=True,text=True)
 assert result.returncode==0 and not result.stderr.strip(),result.stderr
 return dict(path=str(path.relative_to(B)),duration=float(d['format']['duration']),frames=int(v['nb_frames']),fps=v['r_frame_rate'],width=v['width'],height=v['height'],full_decode='pass',sha256=sha(path))
paths=[B/c['path'] for c in p['clips']]+[B/'exports'/f'cover_{n}_5s.mp4' for n in (25,16)]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:checks=list(pool.map(check,paths))
result=dict(status='PASS',original_hashes_checked=len(manifest),attachment_imports=collections.Counter(e['status'] for e in json.load(open(B/'edit/import_report.json'))),selection_counts=counts,curated16=g16,files=checks)
(B/'edit/verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
with zipfile.ZipFile(B/'demo_cover_delivery.zip','w',compression=zipfile.ZIP_STORED) as z:
 for path in [B/'README.md',B/'gallery.html',B/'selection.csv',B/'selected_25_contact_sheet.jpg']+paths+list((B/'exports').glob('*.jpg'))+list((B/'review/thumbs').glob('*.jpg'))+list((B/'edit').glob('*.json'))+list((B/'edit').glob('*.py')):z.write(path,str(path.relative_to(B)))
print(json.dumps(dict(status='PASS',videos=len(checks),frames_each=300,duration_each=5,source_hashes=len(manifest),counts=counts,zip_bytes=(B/'demo_cover_delivery.zip').stat().st_size),indent=2))
