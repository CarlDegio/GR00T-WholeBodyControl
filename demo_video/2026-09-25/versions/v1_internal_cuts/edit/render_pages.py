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
 im=Image.new('RGB',(1920,1080),BG);d=ImageDraw.Draw(im)
 d.rectangle((48,20,53,53),fill=ACC)
 title='操作能力展示' if section=='operations' else '对齐与操作'
 d.text((68,10),title,font=font(34),fill=FG)
 d.text((1872,31),'WHOLE-BODY CONTROL   /   2× SPEED',font=font(19),fill=MUTED,anchor='rm')
 for task,(x,y) in POSITIONS.items():
  color=ACC if task in [1,OTHER] else MUTED
  d.text((x,y-37),f'T{task}',font=ImageFont.truetype(BOLD,23),fill=color)
  name='药瓶入篮' if task==1 else ('踩桶倒垃圾' if task==OTHER else '')
  d.text((x+50,y-41),name,font=font(23),fill=FG)
  d.rounded_rectangle((x-1,y-1,x+W+1,y+H+1),radius=5,fill=PANEL,outline='#253244',width=1)
  if task in [1,OTHER]:
   center(d,(x+W/2,y+H/2-15),'片段结束',font(30),FG)
   center(d,(x+W/2,y+H/2+32),'TWO DEMONSTRATIONS' if section=='operations' else 'ALIGNMENT + MANIPULATION',font(16),MUTED)
  else:
   center(d,(x+W/2,y+H/2-17),f'T{task}',ImageFont.truetype(BOLD,48),'#344357')
   center(d,(x+W/2,y+H/2+40),'素材待补充',font(24),MUTED)
 im.save(path)

def run(cmd):subprocess.run(cmd,check=True)
def duration(p):return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(p)]))
def join(names,dest):
 # Brief dissolve bridges independent takes. Within-take waiting cuts remain cuts.
 a,b=[WORK/f'{n}.mp4' for n in names]; da=duration(a)
 run(['ffmpeg','-v','error','-threads','2','-i',str(a),'-threads','2','-i',str(b),'-filter_complex_threads','1','-filter_complex',f'[0:v]settb=AVTB,setpts=PTS-STARTPTS[a];[1:v]settb=AVTB,setpts=PTS-STARTPTS[b];[a][b]xfade=transition=fade:duration=0.20:offset={da-.20:.6f},format=yuv420p[v]','-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-y',str(dest)])
 return da-.20

def render(section,files,changes):
 bg=WORK/f'{section}_background.png';image_asset(bg,section)
 ds=[duration(p) for p in files]; total=round(max(ds)+.4,3)
 args=['ffmpeg','-v','error','-loop','1','-framerate','30','-i',str(bg)]
 for p in files:args+=['-threads','2','-i',str(p)]
 f=[]
 for k,dur in enumerate(ds):
  f.append(f'[{k+1}:v]scale={W}:{H}:flags=lanczos,setsar=1,format=rgba,fade=t=out:st={max(0,dur-.22)}:d=0.22:alpha=1[v{k}]')
 last='0:v'
 for k,task in enumerate([1,OTHER]):
  x,y=POSITIONS[task];label=f'c{k}'
  f.append(f'[{last}][v{k}]overlay=x={x}:y={y}:eof_action=pass:repeatlast=0[{label}]');last=label
 # Caption badges sit in the footage's lower left, leaving robot heads clear.
 for k,task in enumerate([1,OTHER]):
  x,y=POSITIONS[task]; split=changes[k]
  labels=['操作 01 / 02','操作 02 / 02'] if section=='operations' else ['对齐','操作']
  for n,txt in enumerate(labels):
   start=0 if n==0 else split;end=split if n==0 else ds[k]-.22
   out=f't{k}_{n}'
   f.append(f"[{last}]drawtext=fontfile={FONT}:text='{txt}':fontsize=20:fontcolor=white:box=1:boxcolor=0x0b111a@0.78:boxborderw=8:x={x+18}:y={y+H-38}:enable='between(t,{start},{end})'[{out}]");last=out
 f.append(f'[{last}]format=yuv420p[final]')
 path=OUT/('01_operations_2x2.mp4' if section=='operations' else '02_alignment_operations_2x2.mp4')
 args+=['-filter_complex_threads','1','-filter_complex',';'.join(f),'-map','[final]','-t',str(total),'-an','-r','30','-c:v','libx264','-preset','fast','-crf','18','-threads','3','-movflags','+faststart','-y',str(path)]
 run(args)
 print(path.name,total,flush=True)
 return path

if __name__=='__main__':
 splits=[join(['t1_op_a','t1_op_b'],WORK/'t1_operations.mp4'),join(['other_op_a','other_op_b'],WORK/'other_operations.mp4')]
 p1=render('operations',[WORK/'t1_operations.mp4',WORK/'other_operations.mp4'],splits)
 p2=render('alignment',[WORK/'t1_full.mp4',WORK/'other_full.mp4'],[(31.8-7.6)/2,(31.1-19.1)/2])
 (WORK/'playlist.txt').write_text(f"file '{p1}'\nfile '{p2}'\n")
 run(['ffmpeg','-v','error','-f','concat','-safe','0','-i',str(WORK/'playlist.txt'),'-c','copy','-movflags','+faststart','-y',str(OUT/'00_combined_preview.mp4')])
 for p in [p1,p2]:
  run(['ffmpeg','-v','error','-ss','2','-i',str(p),'-frames:v','1','-y',str(p.with_suffix('.jpg'))])
