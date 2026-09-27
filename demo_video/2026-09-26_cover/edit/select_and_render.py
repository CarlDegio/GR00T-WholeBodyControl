from pathlib import Path
import json,subprocess,concurrent.futures,sys
R=Path('/home/user/Project/GR00T-WholeBodyControl');B=R/'demo_video/2026-09-26_cover';M=json.load(open(R/'demo_video/2026-09-25/assets/manifest.json'))
# Each range is a continuous physical action. End is exclusive; all outputs last 5 s.
data=[
(1,21,44.0,50.0,'T2','navigation','走向柜子'),
(2,21,334.1,340.6,'T2','navigation','走到冰箱旁'),
(3,22,41.4,46.9,'T4','navigation','导航到桌子'),
(4,22,232.3,237.3,'T4','navigation','持袋走向垃圾桶'),
(5,22,10.2,19.2,'T4','scan','180度环视'),
(6,1,3.0,13.0,'T1','manipulation','抓起药瓶并放入篮子'),
(7,2,12.0,22.0,'T1','manipulation','抓取药瓶并移入篮子'),
(8,3,51.0,61.0,'T1','manipulation','侧面视角抓取药瓶'),
(9,3,0.5,10.5,'T1','alignment','向药瓶桌接近'),
(10,3,10.5,17.5,'T1','alignment','桌前近距离接近对齐'),
(11,10,1.0,11.0,'T2','manipulation','下跪抓袋并起身'),
(12,14,31.0,41.0,'T2','manipulation','双手搬起纸箱'),
(13,8,8.5,18.5,'T2','alignment','向纸箱和袋子接近'),
(14,8,48.0,53.0,'T2','alignment','袋前最终接近'),
(15,21,393.0,400.5,'T2','alignment','冰箱旁袋子对齐'),
(16,11,1.5,11.5,'T3','manipulation','桌前抓碗并抬起'),
(17,12,24.0,34.0,'T3','manipulation','持碗转身倾倒'),
(18,13,27.0,37.0,'T3','manipulation','抓碗后转向垃圾桶'),
(19,13,2.0,7.0,'T3','alignment','接近碗所在桌子'),
(20,12,12.0,19.5,'T3','alignment','桌沿近距离接近'),
(21,5,0.0,9.5,'T4','manipulation','踩开桶盖并投放'),
(22,18,21.0,29.5,'T4','manipulation','近景踩踏开桶'),
(23,6,20.0,28.0,'T4','alignment','正向接近桶踏板'),
(24,20,6.0,14.0,'T4','alignment','远处向垃圾桶接近'),
(25,22,294.0,304.0,'T4','alignment','持袋接近并对齐垃圾桶')]
clips=[]
for id,src,a,b,task,kind,label in data:
 s=next(v for v in M if v['order']==src)
 clips.append(dict(id=id,source=src,source_path=s['local_path'],source_sha256=s['sha256'],start=a,end=b,speed=round((b-a)/5,3),duration=5,task=task,kind=kind,label=label,path=f'clips/{id:02d}_{task}_{kind}.mp4'))
# Central 3x3 is the opening; central 2x2 is the curated opening.
g25=[[9,1,13,3,19],[7,6,11,16,20],[2,12,5,21,4],[14,23,18,25,22],[10,8,15,24,17]]
g16=[[1,9,3,19],[13,6,11,2],[4,17,21,23],[12,5,7,18]]
plan=dict(title=['Beyond Arrival: Geometric Alignment','for Agentic Humanoid Loco-manipulation'],title_source='/home/user/Project/Overleaf/6a925da904325a8d60afac0b/root.tex',duration=5,fps=60,width=1920,height=1080,timing={'opening':[0,1],'zoom':[1,2],'full_grid':[2,5],'title_blur_reveal':[2,3]},grid25=g25,grid16=g16,clips=clips)
(B/'edit/selection.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
def render(c):
 p=B/c['path'];src=R/c['source_path']
 cmd=['ffmpeg','-v','error','-threads','2','-ss',str(c['start']),'-i',str(src),'-an','-vf',f"trim=duration={c['end']-c['start']},setpts=(PTS-STARTPTS)/{c['speed']},fps=60,scale=1280:720:flags=lanczos,setsar=1,format=yuv420p",'-t','5','-frames:v','300','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-metadata:s:v:0','rotate=0','-movflags','+faststart','-y',str(p)]
 subprocess.run(cmd,check=True)
 print('clip',c['id'],c['label'],c['speed'],flush=True)
if __name__=='__main__':
 requested=set(map(int,sys.argv[1:])) if len(sys.argv)>1 else None
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(render,[c for c in clips if requested is None or c['id'] in requested]))
