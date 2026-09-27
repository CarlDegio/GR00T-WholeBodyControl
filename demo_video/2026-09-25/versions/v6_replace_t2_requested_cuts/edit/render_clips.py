from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
plan=json.loads((ROOT/'edit/edit_decisions.json').read_text())
manifest=json.loads((ROOT/'assets/manifest.json').read_text())
W,H=832,468
out=ROOT/'edit/intermediate';out.mkdir(exist_ok=True)
for name,c in plan['clips'].items():
 if len(sys.argv)>1 and name not in sys.argv[1:]: continue
 src=ROOT/'assets/originals'/Path(manifest[c['source']-1]['local_path']).name
 # Apply only explicit user-requested frame ranges or tail trims.
 if 'keep_frame_ranges' in c:
  expr='+'.join(f'gte(n,{a})*lt(n,{b})' for a,b in c['keep_frame_ranges'])
  trim=f"select='{expr}',"
 else:
  trim=f"trim=end_frame={c['retained_frames']}," if 'retained_frames' in c else '' 
 filters=trim+f'setpts=N/(60*TB),scale={W}:{H}:flags=lanczos,setsar=1,eq=contrast=1.015:saturation=1.025,format=yuv420p'
 cmd=['ffmpeg','-v','error','-threads','2','-i',str(src),'-vf',filters,'-an','-r','60','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-metadata:s:v:0','rotate=0','-movflags','+faststart','-y',str(out/f'{name}.mp4')]
 subprocess.run(cmd,check=True)
 print('rendered',name,flush=True)
