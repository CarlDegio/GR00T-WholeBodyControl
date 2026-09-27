from pathlib import Path
import json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
plan=json.loads((ROOT/'edit/edit_decisions.json').read_text())
check=json.loads((ROOT/'edit/v4_change_check.json').read_text())
check['revision']='v4_t1_additional_tail_second';check['clips']={};check['exports']=[]
def probe(p):
 return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,duration,avg_frame_rate,width,height','-of','json',str(p)]))['streams'][0]
for name in ['t1_full','other_full']:
 c=plan['clips'][name];meta=probe(ROOT/'edit/intermediate'/f'{name}.mp4')
 assert int(meta['nb_read_frames'])==c['retained_frames']
 check['clips'][name]={'retained_frames':int(meta['nb_read_frames']),'removed_source_frames':c['source_frames']-int(meta['nb_read_frames']),'removed_source_seconds':c['trim_tail_source_seconds'],'output_seconds_removed':c['trim_tail_source_seconds']/2}
op=ROOT/'exports/01_operations_2x2_v2.mp4'
assert hashlib.sha256(op.read_bytes()).hexdigest()==check['operations_sha256_before']
check['operations_export_byte_unchanged']=True
for p in sorted((ROOT/'exports').glob('*_v4.mp4')):
 meta=probe(p);subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(p),'-f','null','-'],check=True)
 check['exports'].append({'file':p.name,**meta,'complete_decode':'passed'})
check['layout_render_settings']='Unchanged from v2; only alignment source duration and export version changed.'
(ROOT/'edit/v4_change_check.json').write_text(json.dumps(check,indent=2,ensure_ascii=False)+'\n')
(ROOT/'edit/verification.json').write_text(json.dumps(check,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(check,indent=2),flush=True)
