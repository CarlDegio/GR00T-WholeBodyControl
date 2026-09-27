from pathlib import Path
import json,subprocess,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'edit/intermediate'
plan=json.loads((ROOT/'edit/edit_decisions.json').read_text()); manifest=json.loads((ROOT/'assets/manifest.json').read_text())
def probe(p):
 return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(p)]))['streams'][0]
def frame(p,n,filter_tail=''):
 vf=f'select=eq(n\\,{n})'+(','+filter_tail if filter_tail else '')
 b=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(p),'-vf',vf,'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'])
 return np.frombuffer(b,dtype=np.uint8).astype(float)
report={'revision':'v2_full_continuous','clips':{},'exports':[]}
for name,c in plan['clips'].items():
 src=ROOT/'assets/originals'/Path(manifest[c['source']-1]['local_path']).name;out=WORK/f'{name}.mp4'
 source_meta=probe(src);out_meta=probe(out);ns=int(source_meta['nb_read_frames']);no=int(out_meta['nb_read_frames']);assert ns==no,(name,ns,no)
 errors={}
 for tag,n in [('first',0),('last',ns-1)]:
  expected=frame(src,n,'scale=832:468:flags=lanczos,setsar=1,eq=contrast=1.015:saturation=1.025,format=yuv420p');actual=frame(out,n)
  mae=float(np.abs(expected-actual).mean());assert mae<4,(name,tag,mae);errors[tag+'_frame_mean_absolute_pixel_error']=round(mae,3)
 report['clips'][name]={'source_frames':ns,'edited_frames':no,'all_frame_count_preserved':True,**errors}
 print('verified source coverage + first/last frames:',name,flush=True)
for name,a,b in [('t1_operations','t1_op_a','t1_op_b'),('other_operations','other_op_a','other_op_b')]:
 actual=int(probe(WORK/f'{name}.mp4')['nb_read_frames']);expected=sum(report['clips'][x]['edited_frames'] for x in [a,b]);assert actual==expected,(name,actual,expected)
 report['clips'][name]={'edited_frames':actual,'complete_take_concatenation':True}
for p in sorted((ROOT/'exports').glob('*_v2.mp4')):
 meta=probe(p);subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(p),'-f','null','-'],check=True)
 report['exports'].append({'file':p.name,**meta,'complete_decode':'passed'})
for row in manifest:
 p=ROOT/'assets/originals'/Path(row['local_path']).name;assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
report['original_hash_check']='All six matched manifest'
(ROOT/'edit/verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2),flush=True)
