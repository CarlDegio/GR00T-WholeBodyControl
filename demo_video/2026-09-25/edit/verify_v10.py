from pathlib import Path
from PIL import Image
import numpy as np,json,hashlib,subprocess,re
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'edit/intermediate';OUT=ROOT/'exports'
report=json.loads((ROOT/'edit/v10_change_check.json').read_text());report['revision']='v10_english_hold_last_frame';report['freeze_checks']=[];report['exports']=[]
def probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,duration,nb_read_frames,avg_frame_rate','-of','json',str(p)]))['streams'][0]
def frame(p,n,vf='null',w=1920,h=1080):
 b=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(p),'-vf',f'select=eq(n\\,{n}),{vf}','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'])
 return np.frombuffer(b,np.uint8).reshape(h,w,3).astype(float)
positions=[(114,112),(1002,112),(114,616),(1002,616)]
for section,names in [('01_operations',['t1_operations','t2_operations','t3_operations','other_operations']),('02_alignment_operations',['t1_full','t2_full','t3_full','other_full'])]:
 p=OUT/f'{section}_2x2_v10.mp4';meta=probe(p);previous=probe(OUT/f'{section}_2x2_v9.mp4');assert meta['nb_read_frames']==previous['nb_read_frames']
 page_last=frame(p,int(meta['nb_read_frames'])-1)
 clip_meta=[probe(WORK/f'{name}.mp4') for name in names]
 shortest=min(range(4),key=lambda k:float(clip_meta[k]['duration']));mid_time=(float(clip_meta[shortest]['duration'])+max(float(m['duration']) for m in clip_meta))/2
 page_mid=frame(p,int(mid_time*60))
 for k,(name,(x,y),cm) in enumerate(zip(names,positions,clip_meta)):
  expected=frame(WORK/f'{name}.mp4',int(cm['nb_read_frames'])-1,'scale=804:452:flags=lanczos,crop=804:398:0:0',804,398)
  actual=page_last[y:y+398,x:x+804];mae=float(abs(expected-actual).mean());assert mae<4,(section,name,mae)
  row={'page':section,'task':k+1,'last_frame_matches_task_tail':True,'mean_pixel_error':round(mae,3)}
  if k==shortest:
   mid_actual=page_mid[y:y+398,x:x+804];mid_mae=float(abs(expected-mid_actual).mean());assert mid_mae<4,(section,name,mid_mae)
   tile_difference=float(abs(page_mid[y:y+452,x:x+804]-page_last[y:y+452,x:x+804]).mean());assert tile_difference<2,(section,'frozen tile',tile_difference)
   row.update({'holds_while_other_tasks_play':True,'checked_at_s':round(mid_time,3),'whole_tile_mean_pixel_change':round(tile_difference,3)})
  report['freeze_checks'].append(row)
 print('Verified frozen endpoints and duration:',section,flush=True)
for rel,h in report['unchanged_file_hashes_before'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
report['all_eight_task_intermediates_byte_unchanged']=True
script=(ROOT/'edit/render_pages.py').read_text();assert not re.search('[\u3400-\u9fff]',script);report['all_added_text_english']=True
for name in ['operations','alignment']:
 im=np.array(Image.open(WORK/f'{name}_background.png').convert('RGB'));assert np.all(im[12:56,1400:1872]==np.array([11,17,26]))
report['top_right_speed_summary_removed']=True
for p in sorted(OUT.glob('*_v10.mp4')):
 m=probe(p);subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(p),'-f','null','-'],check=True);report['exports'].append({'file':p.name,**m,'complete_decode':'passed'})
for n in ['v10_change_check.json','verification.json']:(ROOT/'edit'/n).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
