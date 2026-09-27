from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parents[1]
plan=json.loads((ROOT/'edit/edit_decisions.json').read_text())
manifest=json.loads((ROOT/'assets/manifest.json').read_text())
W,H=832,468
out=ROOT/'edit/intermediate';out.mkdir(exist_ok=True)
for name,c in plan['clips'].items():
 src=ROOT/'assets/originals'/Path(manifest[c['source']-1]['local_path']).name
 filters=[];parts=[]
 for k,(start,end) in enumerate(c['ranges']):
  filters.append(f'[0:v]trim=start={start}:end={end},setpts=(PTS-STARTPTS)/2,scale={W}:{H}:flags=lanczos,setsar=1,fps=30,eq=contrast=1.015:saturation=1.025[v{k}]')
  parts.append(f'[v{k}]')
 filters.append(''.join(parts)+f'concat=n={len(parts)}:v=1:a=0,format=yuv420p[out]')
 cmd=['ffmpeg','-v','error','-threads','2','-i',str(src),'-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-metadata:s:v:0','rotate=0','-movflags','+faststart','-y',str(out/f'{name}.mp4')]
 subprocess.run(cmd,check=True)
 print('rendered',name,flush=True)
