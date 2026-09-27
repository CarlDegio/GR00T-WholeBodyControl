from pathlib import Path
import subprocess,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'edit/intermediate';page=ROOT/'exports/01_operations_2x2_v7.mp4'
report=json.loads((ROOT/'edit/v7_change_check.json').read_text());report['revision']='v7_add_t2_operations_3x';report['t2_operations']=[]
rows=json.loads((ROOT/'assets/manifest.json').read_text())
def probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,avg_frame_rate,duration,nb_read_frames','-of','json',str(p)]))['streams'][0]
def frame(p,n,vf):
 b=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(p),'-vf',f'select=eq(n\\,{n}),{vf}','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'])
 return np.frombuffer(b,np.uint8).astype(float)
offset=0
for order,name in [(9,'t2_op_a'),(10,'t2_op_b')]:
 row=next(x for x in rows if x['order']==order);src=ROOT/'assets/originals'/Path(row['local_path']).name;clip=WORK/f'{name}.mp4'
 assert hashlib.sha256(src.read_bytes()).hexdigest()==row['sha256']
 sm=probe(src);cm=probe(clip);ns=int(sm['nb_read_frames']);sampled=[n for n in range(ns) if n%3!=1];count=len(sampled);assert int(cm['nb_read_frames'])==count;assert sampled[0]==0 and sampled[-1]==ns-1
 assert abs(float(cm['duration'])-float(sm['duration'])/3)<1/60
 checks=[]
 for label,n in [('first',0),('last',count-1)]:
  a=frame(src,sampled[n],'scale=832:468:flags=lanczos,setsar=1,eq=contrast=1.015:saturation=1.025,format=yuv420p');b=frame(clip,n,'format=rgb24');mae=float(abs(a-b).mean());assert mae<4,(name,label,mae)
  a=frame(clip,n,'scale=804:452:flags=lanczos,crop=804:398:0:0');b=frame(page,offset+n,'crop=804:398:1002:112');pm=float(abs(a-b).mean());assert pm<4,(name,label,pm)
  checks.append({'frame':label,'source_index':sampled[n],'page_index':offset+n,'source_to_clip_mean_pixel_error':round(mae,3),'clip_to_page_mean_pixel_error':round(pm,3)})
 report['t2_operations'].append({'clip':name,'source_frames':ns,'rendered_frames':count,'source_duration':float(sm['duration']),'rendered_duration':float(cm['duration']),'requested_speed':3,'full_time_range_preserved':True,'endpoint_checks':checks});offset+=count
assert int(probe(WORK/'t2_operations.mp4')['nb_read_frames'])==offset
for f,h in report['unchanged_file_hashes_before'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
report['alignment_page_and_t1_t4_operations_byte_unchanged']=True;report['exports']=[]
for p in sorted((ROOT/'exports').glob('*_v7.mp4')):
 m=probe(p);subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(p),'-f','null','-'],check=True);report['exports'].append({'file':p.name,**m,'complete_decode':'passed'})
for n in ['v7_change_check.json','verification.json']:(ROOT/'edit'/n).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
