from pathlib import Path
from PIL import Image
import numpy as np,json,subprocess,math
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'edit/intermediate';OUT=ROOT/'exports'
plan=json.loads((ROOT/'edit/edit_decisions.json').read_text());manifest=json.loads((ROOT/'assets/manifest.json').read_text())
report=json.loads((ROOT/'edit/v11_change_check.json').read_text());report['clips']=[];report['freeze_checks']=[];report['exports']=[]
def probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,duration,nb_read_frames,avg_frame_rate','-of','json',str(p)]))['streams'][0]
def frame(p,n,vf='null',w=832,h=468):
 b=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(p),'-vf',f'select=eq(n\\,{n}),{vf}','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'])
 return np.frombuffer(b,np.uint8).reshape(h,w,3).astype(float)
for name,c in plan['clips'].items():
 p=WORK/f'{name}.mp4';m=probe(p);n=int(m['nb_read_frames']);retained=c.get('retained_frames',c['source_frames']);speed=c['playback_speed']
 expected=math.ceil(retained*2/speed);assert abs(n-expected)<=1,(name,n,expected)
 src=ROOT/'assets/originals'/Path(manifest[c['source']-1]['local_path']).name
 source_start=c.get('keep_frame_ranges',[[0,retained]])[0][0];source_end=c.get('keep_frame_ranges',[[0,retained]])[-1][1]-1
 errors=[]
 for a,b in [(source_start,0),(source_end,n-1)]:
  orig=frame(src,a,'scale=832:468:flags=lanczos,setsar=1,eq=contrast=1.015:saturation=1.025,format=yuv420p')
  actual=frame(p,b);mae=float(abs(orig-actual).mean());assert mae<4,(name,a,b,mae);errors.append(round(mae,3))
 report['clips'].append({'name':name,'speed':speed,'output_frames':n,'duration':m['duration'],'first_last_frame_errors':errors})
 print('Verified speed and retained endpoints:',name,flush=True)
positions=[(114,112),(1002,112),(114,616),(1002,616)]
for section,names in [('01_operations',['t1_operations','t2_operations','t3_operations','other_operations']),('02_alignment_operations',['t1_full','t2_full','t3_full','other_full'])]:
 p=OUT/f'{section}_2x2_v11.mp4';meta=probe(p);page_last=frame(p,int(meta['nb_read_frames'])-1,w=1920,h=1080)
 clip_meta=[probe(WORK/f'{name}.mp4') for name in names]
 shortest=min(range(4),key=lambda k:float(clip_meta[k]['duration']));mid_time=(float(clip_meta[shortest]['duration'])+max(float(m['duration']) for m in clip_meta))/2
 page_mid=frame(p,int(mid_time*60),w=1920,h=1080)
 for k,(name,(x,y),cm) in enumerate(zip(names,positions,clip_meta)):
  expected=frame(WORK/f'{name}.mp4',int(cm['nb_read_frames'])-1,'scale=804:452:flags=lanczos,crop=804:398:0:0',804,398)
  mae=float(abs(expected-page_last[y:y+398,x:x+804]).mean());assert mae<4,(section,name,mae)
  row={'page':section,'task':k+1,'last_frame_matches_task_tail':True,'mean_pixel_error':round(mae,3)}
  if k==shortest:
   diff=float(abs(page_mid[y:y+452,x:x+804]-page_last[y:y+452,x:x+804]).mean());assert diff<2,(section,diff)
   row.update({'holds_while_other_tasks_play':True,'checked_at_s':round(mid_time,3)})
  report['freeze_checks'].append(row)
 print('Verified last-frame holds:',section,flush=True)
script=(ROOT/'edit/render_pages.py').read_text();assert 'Take 01' not in script and "txt=f'{speed:g}×'" in script
report['lower_left_speed_only']=True
for name in ['operations','alignment']:
 im=np.array(Image.open(WORK/f'{name}_background.png').convert('RGB'));assert np.all(im[12:56,1400:1872]==np.array([11,17,26]))
report['top_right_speed_summary_removed']=True
for p in sorted(OUT.glob('*_v11.mp4')):
 m=probe(p);assert (m['width'],m['height'],m['avg_frame_rate'])==(1920,1080,'60/1')
 subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(p),'-f','null','-'],check=True)
 report['exports'].append({'file':p.name,**m,'complete_decode':'passed'})
for n in ['v11_change_check.json','verification.json']:(ROOT/'edit'/n).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print('All checks passed.',flush=True)
