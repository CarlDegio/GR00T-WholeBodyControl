from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parents[1]
plan=json.loads((ROOT/'edit/edit_decisions.json').read_text())
manifest=json.loads((ROOT/'assets/manifest.json').read_text())
W,H=832,468
out=ROOT/'edit/intermediate';out.mkdir(exist_ok=True)
for name,c in plan['clips'].items():
 src=ROOT/'assets/originals'/Path(manifest[c['source']-1]['local_path']).name
 # Preserve every source frame, including the exact first and last frames.
 filters=f'setpts=N/(60*TB),scale={W}:{H}:flags=lanczos,setsar=1,eq=contrast=1.015:saturation=1.025,format=yuv420p'
 cmd=['ffmpeg','-v','error','-threads','2','-i',str(src),'-vf',filters,'-an','-r','60','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-metadata:s:v:0','rotate=0','-movflags','+faststart','-y',str(out/f'{name}.mp4')]
 subprocess.run(cmd,check=True)
 print('rendered',name,flush=True)
