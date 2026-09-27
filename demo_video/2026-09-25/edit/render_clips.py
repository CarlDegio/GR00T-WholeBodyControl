from pathlib import Path
import json,subprocess,sys,math
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
 speed=c.get('playback_speed',2.0)
 assert speed in (1.5,2.0,2.5,3.0,3.5,4.0,5.0)
 sampling=''
 if speed>2:
  # Uniform temporal sampling after source cuts; preserve both retained endpoints.
  retained=c.get('retained_frames',c['source_frames'])
  count=math.ceil(retained*2/speed)
  final_regular=math.floor((count-1)*speed/2)
  expr=f'gt(ceil((n+1)*2/{speed}),ceil(n*2/{speed}))*lt(n,{final_regular})+eq(n,{retained-1})'
  sampling=f"select='{expr}',"
 timing='setpts=N/(45*TB),fps=60:round=up:start_time=0' if speed==1.5 else 'setpts=N/(60*TB)'
 filters=trim+sampling+timing+f',scale={W}:{H}:flags=lanczos,setsar=1,eq=contrast=1.015:saturation=1.025,format=yuv420p'
 cmd=['ffmpeg','-v','error','-threads','2','-i',str(src),'-vf',filters,'-an','-r','60','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-metadata:s:v:0','rotate=0','-movflags','+faststart','-y',str(out/f'{name}.mp4')]
 subprocess.run(cmd,check=True)
 print('rendered',name,flush=True)
