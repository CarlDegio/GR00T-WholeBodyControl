from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import subprocess,json
ROOT=Path(__file__).resolve().parents[1]; WORK=ROOT/'edit/intermediate'; OUT=ROOT/'exports'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
BG='#0b111a'; PANEL='#131d2a'; MUTED='#8291a3'; FG='#f0f4f8'; ACC='#5dd9bf'
OTHER=4
POSITIONS={1:(114,112),2:(1002,112),3:(114,616),4:(1002,616)}
W,H=804,452

def font(size):return ImageFont.truetype(FONT,size)
def center(d,xy,text,f,fill):d.text(xy,text,font=f,fill=fill,anchor='mm')
def image_asset(path,section):
 active=[1,2,3,OTHER]
 im=Image.new('RGB',(1920,1080),BG);d=ImageDraw.Draw(im)
 d.rectangle((48,20,53,53),fill=ACC)
 title='Manipulation Demonstrations' if section=='operations' else 'Alignment + Manipulation'
 d.text((68,10),title,font=font(34),fill=FG)
 for task,(x,y) in POSITIONS.items():
  color=ACC if task in active else MUTED
  d.text((x,y-37),f'T{task}',font=ImageFont.truetype(BOLD,23),fill=color)
  name={1:'Medicine Bottle into Basket',2:'Bag Placement and Box Transfer',3:'Pick Up Bowl and Empty Contents',4:'Foot-Operated Waste Disposal'}.get(task,'') if task in active else ''
  d.text((x+50,y-41),name,font=font(23),fill=FG)
  d.rounded_rectangle((x-1,y-1,x+W+1,y+H+1),radius=5,fill=PANEL,outline='#253244',width=1)
 im.save(path)

def run(cmd):subprocess.run(cmd,check=True)
def duration(p):return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(p)]))
def join(names,dest):
 # Concatenate complete takes with no temporal overlap or hidden boundary frames.
 a,b=[WORK/f'{n}.mp4' for n in names]; da=duration(a)
 run(['ffmpeg','-v','error','-threads','2','-i',str(a),'-threads','2','-i',str(b),'-filter_complex_threads','1','-filter_complex','[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0,format=yuv420p[v]','-map','[v]','-an','-r','60','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-y',str(dest)])
 return da

def render(section,files,changes):
 tasks=[1,2,3,OTHER]
 bg=WORK/f'{section}_background.png';image_asset(bg,section)
 ds=[duration(p) for p in files]; total=round(max(ds)+.4,3)
 args=['ffmpeg','-v','error','-loop','1','-framerate','60','-i',str(bg)]
 for p in files:args+=['-threads','2','-i',str(p)]
 f=[]
 for k,dur in enumerate(ds):
  f.append(f'[{k+1}:v]scale={W}:{H}:flags=lanczos,setsar=1,format=rgba[v{k}]')
 last='0:v'
 for k,task in enumerate(tasks):
  x,y=POSITIONS[task];label=f'c{k}'
  f.append(f'[{last}][v{k}]overlay=x={x}:y={y}:eof_action=repeat:repeatlast=1[{label}]');last=label
 # Each lower-left badge contains only the task playback speed.
 for k,task in enumerate(tasks):
  x,y=POSITIONS[task]
  speed={1:2.5,2:5,3:4,4:1.5}[task] if section=='operations' else {1:3,2:3.5,3:3,4:2.5}[task]
  txt=f'{speed:g}×';out=f't{k}'
  f.append(f"[{last}]drawtext=fontfile={FONT}:text='{txt}':fontsize=20:fontcolor=white:box=1:boxcolor=0x0b111a@0.78:boxborderw=8:x={x+18}:y={y+H-38}[{out}]");last=out
 f.append(f'[{last}]format=yuv420p[final]')
 path=OUT/('01_operations_2x2_v12.mp4' if section=='operations' else '02_alignment_operations_2x2_v12.mp4')
 args+=['-filter_complex_threads','1','-filter_complex',';'.join(f),'-map','[final]','-t',str(total),'-an','-r','60','-c:v','libx264','-preset','fast','-crf','18','-threads','3','-movflags','+faststart','-y',str(path)]
 run(args)
 print(path.name,total,flush=True)
 return path

if __name__=='__main__':
 for prefix in ['t1','t2','t3','other']:
  join([f'{prefix}_op_a',f'{prefix}_op_b'],WORK/f'{prefix}_operations.mp4')
 t3_split=duration(WORK/'t3_op_a.mp4')
 t4_split=duration(WORK/'other_op_a.mp4')
 splits=[duration(WORK/'t1_op_a.mp4'),duration(WORK/'t2_op_a.mp4'),t3_split,t4_split]
 p1=render('operations',[WORK/'t1_operations.mp4',WORK/'t2_operations.mp4',WORK/'t3_operations.mp4',WORK/'other_operations.mp4'],splits)
 p2=render('alignment',[WORK/'t1_full.mp4',WORK/'t2_full.mp4',WORK/'t3_full.mp4',WORK/'other_full.mp4'],[None]*4)
 first_frames=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','csv=p=0',str(p1)]))
 (WORK/'playlist.txt').write_text(f"file '{p1}'\nduration {first_frames/60:.12f}\nfile '{p2}'\n")
 run(['ffmpeg','-v','error','-f','concat','-safe','0','-i',str(WORK/'playlist.txt'),'-c','copy','-movflags','+faststart','-y',str(OUT/'00_combined_preview_v12.mp4')])
 for p in [p1,p2]:
  run(['ffmpeg','-v','error','-ss','2','-i',str(p),'-frames:v','1','-y',str(p.with_suffix('.jpg'))])
