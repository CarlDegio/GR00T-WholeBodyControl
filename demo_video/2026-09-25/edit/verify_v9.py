from pathlib import Path
import json,hashlib,subprocess,math,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'edit/intermediate';OUT=ROOT/'exports'
report=json.loads((ROOT/'edit/v9_change_check.json').read_text());report['revision']='v9_add_t3_slow_t4';report['clips']=[]
plan=json.loads((ROOT/'edit/edit_decisions.json').read_text());manifest=json.loads((ROOT/'assets/manifest.json').read_text())
def probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,avg_frame_rate,duration,nb_read_frames','-of','json',str(p)]))['streams'][0]
def frame(p,n,vf):
 b=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(p),'-vf',f'select=eq(n\\,{n}),{vf}','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'])
 return np.frombuffer(b,np.uint8).astype(float)
counts={};offsets={'t3_op_a':0,'other_op_a':0,'t3_full':0,'other_full':0}
for name in ['t3_op_a','t3_op_b','t3_full','other_op_a','other_op_b','other_full']:
 c=plan['clips'][name];row=next(r for r in manifest if r['order']==c['source']);src=ROOT/'assets/originals'/Path(row['local_path']).name;clip=WORK/f'{name}.mp4';meta=probe(clip)
 assert hashlib.sha256(src.read_bytes()).hexdigest()==row['sha256']
 ns=c.get('retained_frames',c['source_frames']);speed=c['playback_speed'];count=int(meta['nb_read_frames']);counts[name]=count
 expected=ns if speed==2 else (len([n for n in range(ns) if n%3!=1]) if speed==3 else math.ceil(ns*4/3))
 assert count==expected,(name,count,expected)
 assert abs(float(meta['duration'])-ns/(30*speed))<1/60,(name,meta['duration'],ns/(30*speed))
 if name=='t3_op_b':offsets[name]=counts['t3_op_a']
 if name=='other_op_b':offsets[name]=counts['other_op_a']
 section='02_alignment_operations_2x2_v9.mp4' if name.endswith('full') else '01_operations_2x2_v9.mp4'
 x,y=(114,616) if name.startswith('t3') else (1002,616)
 checks=[]
 for label,source_n,n in [('first',0,0),('last',ns-1,count-1)]:
  a=frame(src,source_n,'scale=832:468:flags=lanczos,setsar=1,eq=contrast=1.015:saturation=1.025,format=yuv420p');b=frame(clip,n,'format=rgb24');mae=float(abs(a-b).mean());assert mae<4,(name,label,mae)
  a=frame(clip,n,'scale=804:452:flags=lanczos,crop=804:398:0:0');b=frame(OUT/section,offsets[name]+n,f'crop=804:398:{x}:{y}');pm=float(abs(a-b).mean());assert pm<4,(name,label,pm)
  checks.append({'frame':label,'source_frame':source_n,'page_frame':offsets[name]+n,'source_to_clip_pixel_error':round(mae,3),'clip_to_page_pixel_error':round(pm,3)})
 report['clips'].append({'clip':name,'speed':speed,'retained_source_frames':ns,'output_frames':count,'duration':float(meta['duration']),'endpoint_checks':checks})
 print('verified speed, duration and endpoints:',name,flush=True)
for joined,a,b in [('t3_operations','t3_op_a','t3_op_b'),('other_operations','other_op_a','other_op_b')]:assert int(probe(WORK/f'{joined}.mp4')['nb_read_frames'])==counts[a]+counts[b]
for f,h in report['unchanged_file_hashes_before'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
report['t1_t2_intermediates_byte_unchanged']=True;report['exports']=[]
for p in sorted(OUT.glob('*_v9.mp4')):
 meta=probe(p);subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(p),'-f','null','-'],check=True);report['exports'].append({'file':p.name,**meta,'complete_decode':'passed'})
for n in ['v9_change_check.json','verification.json']:(ROOT/'edit'/n).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
