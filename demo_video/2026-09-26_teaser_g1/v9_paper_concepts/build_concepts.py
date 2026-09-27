"""Scientific teaser design studies. Real images are placed, never repainted.
Vector annotations and plots are exported to PDF/SVG; photos stay raster.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Arc
from PIL import Image

P=Path(__file__).resolve().parent
ROOT=P.parent
PAPER=Path('/home/user/Project/Overleaf/6a925da904325a8d60afac0b')
F=PAPER/'figures'
W,H=2200,900
INK='#192b3d'; MUTED='#566878'; LINE='#d5dfe4'; ORANGE='#c76720'; PALE='#fff2e5'; BLUE='#337aab'; TEAL='#178478'; GRAY='#8395a4'; BG='#f1f5f7'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
assets={}
texts=[]

def canvas():
 fig=plt.figure(figsize=(7,7*H/W),dpi=400,facecolor='white')
 ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
 return fig,ax

def text(ax,x,y,s,size=32,color=INK,weight='normal',ha='left',va='center',z=5):
 t=ax.text(x,y,s,fontsize=size*7*72/W,color=color,fontweight=weight,ha=ha,va=va,zorder=z,linespacing=1.3)
 texts.append(t)
 return t

def box(ax,x,y,w,h,fill='white',edge=LINE,r=0,lw=1.0,z=2,alpha=1):
 p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fill,edgecolor=edge,linewidth=lw,zorder=z,alpha=alpha)
 ax.add_patch(p);return p

def line(ax,pts,c=LINE,lw=1,style='-',z=3):
 ax.plot(*zip(*pts),color=c,linewidth=lw,linestyle=style,zorder=z)

def arrow(ax,a,b,c=ORANGE,lw=1.6,style='-|>',size=8,z=5):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=size,color=c,linewidth=lw,zorder=z,shrinkA=0,shrinkB=0))

def pic(ax,p,x,y,w,h=None,border=True):
 p=Path(p);im=np.array(Image.open(p).convert('RGB'));ih,iw=im.shape[:2]
 if h is None:h=w*ih/iw
 assert abs(w/h-iw/ih)<.01, f'Image stretched: {p}'
 assets[str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':[iw,ih]}
 ax.imshow(im,extent=(x,x+w,y+h,y),zorder=1,interpolation='lanczos',aspect='auto')
 if border:box(ax,x,y,w,h,'none',LINE,lw=.55)
 return h

def header(ax,title,subtitle=None):
 text(ax,0,38,title,48,weight='bold')
 if subtitle:text(ax,0,86,subtitle,29,color=MUTED)

def flow(ax,x,y,w,h,labels=('Navigate','GATE','Manipulate')):
 gap=35;bw=(w-gap*2)/3
 for i,(label,c,fill) in enumerate(zip(labels,[BLUE,ORANGE,TEAL],['#eaf3fa',PALE,'#e9f5f1'])):
  box(ax,x+i*(bw+gap),y,bw,h,fill,fill,r=8,lw=.5)
  text(ax,x+i*(bw+gap)+bw/2,y+h/2,label,32,c,'bold',ha='center')
  if i<2:arrow(ax,(x+(i+1)*bw+i*gap+5,y+h/2),(x+(i+1)*(bw+gap)-6,y+h/2),MUTED,lw=1,size=6)

def stats(ax,x,y,w=850,heading=True):
 if heading:text(ax,x,y,'Complete-task success',36,weight='bold')
 text(ax,x+w,y,'+67.5 pp',36,ORANGE,'bold',ha='right')
 start=x+340;maxw=w-485
 for k,(label,v,count,c) in enumerate([('Direct handoff',15.0,'6/40',GRAY),('Full agent + GATE',82.5,'33/40',ORANGE)]):
  yy=y+60+k*68
  text(ax,x,yy,label,29,color=c,weight='bold' if k else 'normal')
  box(ax,start,yy-15,maxw,30,BG,BG,r=3,lw=0)
  box(ax,start,yy-15,maxw*v/100,30,c,c,r=3,lw=0)
  text(ax,x+w,yy,f'{v:.1f}%',38,c,'bold',ha='right')
  text(ax,start+maxw+20,yy+29,count,23,MUTED)
 text(ax,x,y+177,'4 tasks · 10 trials per task and method',25,MUTED)

def pair(ax,before,after,x,y,w,label='Head-camera observations',labels=('After navigation','After GATE')):
 gap=46;iw=(w-gap)/2
 text(ax,x,y,label,36,weight='bold')
 text(ax,x,y+45,labels[0],29,GRAY,'bold')
 text(ax,x+iw+gap,y+45,labels[1],29,ORANGE,'bold')
 hh=pic(ax,before,x,y+76,iw);pic(ax,after,x+iw+gap,y+76,iw)
 arrow(ax,(x+iw+8,y+76+hh/2),(x+iw+gap-8,y+76+hh/2),ORANGE,lw=1.5,size=7)
 return y+76+hh

def robot_plan(ax,x,y,c,ang=0):
 # Footprint and heading, explicitly a schematic rather than measured pose.
 th=np.deg2rad(ang);pts=np.array([[-20,26],[20,26],[20,-20],[-20,-20]])
 rot=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
 pts=pts@rot.T+[x,y];ax.add_patch(Polygon(pts,fc=c,ec='white',lw=.6,zorder=4))
 p=np.array([0,-68])@rot.T+[x,y];arrow(ax,(x,y),tuple(p),c,lw=1.1,size=6)

def geometry(ax,x,y,w,h):
 box(ax,x,y,w,h,PALE,'#ebc7a9',r=10,lw=.7)
 text(ax,x+22,y+32,'GATE: task-conditioned geometry',34,ORANGE,'bold')
 # left: top view of target and two base configurations
 bx=x+180;by=y+102
 box(ax,bx-130,by-18,260,25,'#d6e3ec','#91a9ba',r=2,lw=.7)
 box(ax,bx-25,by-33,50,26,TEAL,TEAL,r=3,lw=0)
 line(ax,[(bx,by+10),(bx,by+162)],TEAL,.7,'--')
 robot_plan(ax,bx-98,by+160,GRAY,-33);robot_plan(ax,bx,by+140,ORANGE,0)
 arrow(ax,(bx-70,by+168),(bx-25,by+150),ORANGE,lw=1,size=6)
 arrow(ax,(bx+56,by+35),(bx+56,by+135),TEAL,lw=.8,style='<->',size=5)
 text(ax,bx+77,by+87,'distance',23,TEAL)
 text(ax,bx-98,by+209,'Arrival',25,GRAY,ha='center')
 text(ax,bx+20,by+209,'Aligned',25,ORANGE,ha='center')
 text(ax,x+390,y+101,'Position reference',29,weight='bold')
 text(ax,x+390,y+139,'Object / container',29,MUTED)
 text(ax,x+390,y+195,'Orientation reference',29,weight='bold')
 text(ax,x+390,y+233,'Visible structural edge',29,MUTED)
 text(ax,x+22,y+h-35,'Correct distance, lateral offset and heading',30,ORANGE,'bold')
 text(ax,x+w-17,y+h-10,'schematic',20,MUTED,ha='right')

def save(fig,name):
 fig.canvas.draw()
 renderer=fig.canvas.get_renderer();outside=[]
 for t in fig.axes[0].texts:
  b=t.get_window_extent(renderer);f=fig.bbox
  if b.x0<f.x0-1 or b.x1>f.x1+1 or b.y0<f.y0-1 or b.y1>f.y1+1:outside.append(t.get_text())
 assert not outside,(name,outside)
 fig.savefig(P/f'{name}.png',dpi=400,facecolor='white')
 fig.savefig(P/f'{name}_preview.jpg',dpi=200,facecolor='white',pil_kwargs={'quality':94,'subsampling':0})
 fig.savefig(P/f'{name}.pdf',facecolor='white')
 fig.savefig(P/f'{name}.svg',facecolor='white')
 plt.close(fig)
 print(name,'OK')

# A: scene + enlarged same-run handoff evidence + complete-agent outcome.
fig,ax=canvas()
header(ax,'GATE bridges arrival and successful whole-body manipulation',
       'A geometry-aware handoff for an agent that navigates, aligns and manipulates')
text(ax,0,127,'a  Long-horizon execution on G1',34,weight='bold')
pic(ax,ROOT/'v6_trajectories/teaser_v6_clean.png',0,157,1230)
box(ax,23,179,850,64,'white','white',r=7,alpha=.93,lw=0)
text(ax,44,211,'“Carry the bag to the bin and dispose of it.”',31,weight='bold')
# concise task-conditioned transition, placed in otherwise empty upper wall.
flow(ax,400,264,792,58)
end=pair(ax,ROOT/'v8_front_rim/head_5_annotated.png',ROOT/'v8_front_rim/head_6_annotated.png',1290,127,890,'b  From arrival view to aligned view')
box(ax,1290,554,890,71,PALE,PALE,r=8,lw=0)
text(ax,1735,590,'Distance  +  lateral offset  +  orientation',31,ORANGE,'bold',ha='center')
stats(ax,1290,679,890)
text(ax,0,881,'Real video frames; overlaid poses and paths illustrate the task sequence.',24,MUTED)
save(fig,'A_scene_and_evidence')

# B: geometric mechanism, paired real observations and separate local/system outcomes.
fig,ax=canvas()
header(ax,'Geometry turns a nearby arrival into a useful manipulation view',
       'Task semantics select the references; visual feedback adjusts the robot base')
text(ax,0,133,'a  Whole-body task execution',34,weight='bold')
pic(ax,ROOT/'v6_trajectories/teaser_v6_clean.png',0,169,1110)
box(ax,23,191,779,63,'white','white',r=7,alpha=.94,lw=0)
text(ax,44,222,'Navigate → align → manipulate',35,weight='bold')
# use a real pair from T1, showing geometry corrections beyond the bin example.
pair(ax,F/'gate_before_after/t1_before.png',F/'gate_before_after/t1_after.png',1150,133,500,'b  T1: align the view',labels=('Arrival','After GATE'))
# compact schematic at right of photographs
box(ax,1700,133,480,331,PALE,'#ebc7a9',r=8,lw=.6)
text(ax,1720,171,'GATE',42,ORANGE,'bold')
text(ax,1860,170,'Geometric feedback',27,ORANGE)
text(ax,1730,219,'Target + structural edge',27,weight='bold')
box(ax,1730,250,230,16,'#a6bbc8','#a6bbc8',r=0,lw=0)
box(ax,1820,239,50,24,TEAL,TEAL,r=3,lw=0)
line(ax,[(1845,268),(1845,393)],TEAL,.75,'--')
robot_plan(ax,1773,372,GRAY,-32)
robot_plan(ax,1845,359,ORANGE,0)
arrow(ax,(1796,391),(1832,387),ORANGE,lw=.8,size=5)
arrow(ax,(1900,281),(1900,361),TEAL,lw=.8,style='<->',size=5)
text(ax,1975,285,'Distance',25,TEAL)
text(ax,1975,337,'Offset',25,ORANGE)
text(ax,1975,389,'Heading',25,ORANGE)
text(ax,1720,438,'Closed-loop base adjustment',25,MUTED)
text(ax,2166,454,'schematic',19,MUTED,ha='right')
# Middle: two highly compact evidence units, with explicit comparison scope.
text(ax,1150,521,'c  Same-policy near-field success',34,weight='bold')
text(ax,1150,558,'Direct VLA → GATE + VLA',28,MUTED)
for i,(task,b,a,n1,n2) in enumerate([('T1',48.0,83.3,'24/50','45/54'),('T2',18.6,92.5,'8/43','49/53')]):
 x=1150+i*525
 box(ax,x,591,490,145,BG,BG,r=7,lw=0)
 text(ax,x+22,626,task,30,weight='bold')
 text(ax,x+103,626,f'{b:.1f}% → {a:.1f}%',39,ORANGE,'bold')
 text(ax,x+103,683,f'{n1} → {n2}',29,MUTED)
box(ax,1150,768,1030,92,PALE,PALE,r=8,lw=0)
text(ax,1175,803,'Full agent: 82.5%',44,ORANGE,'bold')
text(ax,1740,802,'33/40 complete tasks',30,weight='bold')
text(ax,1175,841,'Direct handoff: 15.0% (6/40) · 4 tasks, 10 trials each',26,MUTED)
text(ax,0,837,'Alignment changes the robot pose and observation',32,weight='bold')
text(ax,0,877,'No additional alignment-policy training or fiducial markers.',27,MUTED)
save(fig,'B_mechanism_and_policy')

# C: two alignment episodes along a long-horizon task, plus per-task outcomes.
fig,ax=canvas()
header(ax,'Align at every handoff, sustain the whole task',
       'One shared alignment skill across pickup, carrying and pedal-assisted disposal')
# main hero surrounded by evidence wings
hero_x=485;hero_w=1230;hero_y=157
pic(ax,ROOT/'v6_trajectories/teaser_v6_clean.png',hero_x,hero_y,hero_w)
box(ax,hero_x+19,hero_y+20,930,60,'white','white',r=7,alpha=.94,lw=0)
text(ax,hero_x+38,hero_y+50,'T4: pick up the bag → carry it → dispose of it',31,weight='bold')
# left wing bag reference from real same-paper run; right wing bin same-run sequence
text(ax,0,139,'1  Pickup example',34,ORANGE,'bold')
for yy,name,label in [(194,'t4_bag_before.png','Before'),(550,'t4_bag_after.png','After GATE')]:
 text(ax,0,yy-24,label,28,GRAY if label=='Before' else ORANGE,'bold')
 pic(ax,F/'gate_before_after'/name,0,yy,425)
text(ax,1775,139,'2  Bin alignment',34,ORANGE,'bold')
for yy,name,label in [(194,'head_5_annotated.png','Before'),(550,'head_6_annotated.png','After GATE')]:
 text(ax,1775,yy-24,label,28,GRAY if label=='Before' else ORANGE,'bold')
 pic(ax,ROOT/'v8_front_rim'/name,1775,yy,425)
# Photo callouts carefully restricted to empty upper wall. Counts belong to cohort, not hero trial.
box(ax,hero_x+190,hero_y+95,1015,105,'white','white',r=9,alpha=.94,lw=0)
text(ax,hero_x+209,hero_y+130,'Task success (%)',27,weight='bold')
text(ax,hero_x+209,hero_y+173,'Direct → full agent',24,MUTED)
for i,(task,b,a) in enumerate([('T1',60,90),('T2',0,90),('T3',0,80),('T4',0,70)]):
 x=hero_x+535+i*165
 text(ax,x,hero_y+130,task,29,weight='bold')
 text(ax,x,hero_y+173,f'{b}→{a}',35,ORANGE,'bold')
# small callout rings tied to the two schematic trajectories
for x,y,n in [(hero_x+540,hero_y+484,'1'),(hero_x+380,hero_y+578,'2')]:
 ax.add_patch(Circle((x,y),24,fc=ORANGE,ec='white',lw=1,zorder=5));text(ax,x,y,n,32,'white','bold',ha='center',z=6)
text(ax,hero_x+8,862,'Stage examples, not a synchronized sequence; paths are schematic.',24,MUTED)
text(ax,hero_x+8,886,'Success rates: 10 trials per task and method.',23,MUTED)
save(fig,'C_repeated_handoffs')

manifest={'paper':str(PAPER),'designs':['A_scene_and_evidence','B_mechanism_and_policy','C_repeated_handoffs'],
 'full_agent':{'successes':33,'trials':40,'percent':82.5},'direct_handoff':{'successes':6,'trials':40,'percent':15.0},
 'nearfield':{'T1':{'direct':[24,50],'gate':[45,54]},'T2':{'direct':[8,43],'gate':[49,53]}},
 'assets':assets,'notes':['All photos are existing real-camera images, placed without repainting.','The base scene is V6, the same scene as V8 before its head-camera inset layer. The current V8 head images are relocated and enlarged.','V8 detection overlays were computed offline with the recorded green rubbish bin prompt.','Trajectories and ghost placements are schematic and not measured odometry.','Direct VLA comparison bypasses both alignment and handoff readiness checks.','Near-field counts are separate cohorts, not matched starts.','Alignment photos show observations, not task completion.','C left bag pair is the paper example, not asserted to be the same trial as the central hero.']}
(P/'sources_and_claims.json').write_text(json.dumps(manifest,indent=2)+'\n')
