from pathlib import Path
import subprocess,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'edit/v6_change_check.json').read_text());report['revision']='v6_replace_t2_requested_cuts'
manifest=json.loads((ROOT/'assets/manifest.json').read_text());row=next(r for r in manifest if r['order']==8)
src=ROOT/'assets/originals'/Path(row['local_path']).name;clip=ROOT/'edit/intermediate/t2_full.mp4';page=ROOT/'exports/02_alignment_operations_2x2_v6.mp4'
def probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,avg_frame_rate,duration,nb_read_frames','-of','json',str(p)]))['streams'][0]
def frame(p,n,vf):
 data=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(p),'-vf',f'select=eq(n\\,{n}),{vf}','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'])
 return np.frombuffer(data,np.uint8).astype(float)
assert hashlib.sha256(src.read_bytes()).hexdigest()==row['sha256']
source_meta=probe(src);clip_meta=probe(clip);assert source_meta['nb_read_frames']=='2701'; assert clip_meta['nb_read_frames']=='2161'
report['t2_source_frames']=2701;report['t2_edited_frames']=2161;report['t2_source_seconds_removed']=18;report['t2_cut_boundary_checks']=[]
for label,source_n,n in [('first_after_8s',240,0),('last_before_36s',1079,839),('first_after_46s',1380,840),('source_last_frame',2700,2160)]:
 a=frame(src,source_n,'scale=832:468:flags=lanczos,setsar=1,eq=contrast=1.015:saturation=1.025,format=yuv420p');b=frame(clip,n,'format=rgb24');mae=float(abs(a-b).mean());assert mae<4,(label,mae)
 a=frame(clip,n,'scale=804:452:flags=lanczos,crop=804:398:0:0');b=frame(page,n,'crop=804:398:1002:112');page_mae=float(abs(a-b).mean());assert page_mae<4,(label,page_mae)
 report['t2_cut_boundary_checks'].append({'frame':label,'source_frame':source_n,'edited_frame':n,'source_to_clip_mean_pixel_error':round(mae,3),'clip_to_page_mean_pixel_error':round(page_mae,3)})
for name,h in report['unchanged_file_hashes_before'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
report['t1_t4_clips_and_operations_page_byte_unchanged']=True
report['exports']=[]
for p in sorted((ROOT/'exports').glob('*_v6.mp4')):
 meta=probe(p);subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(p),'-f','null','-'],check=True);report['exports'].append({'file':p.name,**meta,'complete_decode':'passed'})
(ROOT/'edit/v6_change_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(ROOT/'edit/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
