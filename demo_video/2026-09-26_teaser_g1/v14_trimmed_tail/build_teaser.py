"""Native editable PowerPoint teaser; existing camera photographs only."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,'/tmp/align-ppt-deps')
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR,MSO_AUTO_SIZE
from pptx.oxml.xmlchemy import OxmlElement
from PIL import Image,ImageFont

P=Path(__file__).resolve().parent;ROOT=P.parent
W,H=2700,1080
INK='203447';MUTED='627588';ORANGE='C56927';GRAY='879CAB';TEAL='218777';BLUE='347BAB';PALE='FFF0E2';BG='EDF2F5';LINE='D5DFE5';WHITE='FFFFFF'
prs=Presentation();prs.slide_width=Inches(W/100);prs.slide_height=Inches(H/100)
slide=prs.slides.add_slide(prs.slide_layouts[6]);slide.background.fill.solid();slide.background.fill.fore_color.rgb=RGBColor.from_string(WHITE)
text_checks=[];asset_checks=[]
def u(v):return Inches(v/100)
def c(v):return RGBColor.from_string(v)
def rect(name,x,y,w,h,fill=WHITE,stroke=None,r=False,linewidth=1):
 s=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if r else MSO_SHAPE.RECTANGLE,u(x),u(y),u(w),u(h));s.name=name
 s.fill.solid();s.fill.fore_color.rgb=c(fill)
 if stroke:s.line.color.rgb=c(stroke);s.line.width=Pt(linewidth)
 else:s.line.fill.background()
 if r:s.adjustments[0]=.08
 return s

def text(name,txt,x,y,w,h,size=40,color=INK,bold=False,align='left'):
 s=slide.shapes.add_textbox(u(x),u(y),u(w),u(h));s.name=name
 tf=s.text_frame;tf.clear();tf.word_wrap=False;tf.auto_size=MSO_AUTO_SIZE.NONE
 tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0;tf.vertical_anchor=MSO_ANCHOR.MIDDLE
 for i,ln in enumerate(txt.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=ln;p.alignment={'left':PP_ALIGN.LEFT,'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}[align]
  p.space_before=p.space_after=Pt(0);p.line_spacing=1.06
  for run in p.runs:run.font.name='Liberation Sans';run.font.size=Pt(size*.72);run.font.bold=bold;run.font.color.rgb=c(color)
 font=ImageFont.truetype('/usr/share/fonts/truetype/liberation2/LiberationSans-'+('Bold' if bold else 'Regular')+'.ttf',size*10)
 measured=max(font.getlength(ln)/10 for ln in txt.split('\n'))
 assert measured<=w-1,(name,measured,w)
 assert len(txt.split('\n'))*size*1.06<=h+3,(name,'height')
 text_checks.append(dict(name=name,text=txt,font_px=size,measured_width=round(measured,2),box_width=w))
 return s

def lines(name,pts,color=TEAL,width=3,arrow=False):
 for i,((x1,y1),(x2,y2)) in enumerate(zip(pts,pts[1:])):
  s=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,u(x1),u(y1),u(x2),u(y2));s.name=f'{name}_{i}'
  s.line.color.rgb=c(color);s.line.width=Pt(width*.72)
  if arrow and i==len(pts)-2:
   e=OxmlElement('a:tailEnd');e.set('type','triangle');e.set('w','sm');e.set('len','sm');s.line._get_or_add_ln().append(e)

def pic(name,path,x,y,w):
 path=Path(path);iw,ih=Image.open(path).size;h=w*ih/iw
 s=slide.shapes.add_picture(str(path),u(x),u(y),width=u(w),height=u(h));s.name=name
 asset_checks.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_size=[iw,ih],placement=[x,y,w,h]))
 return s

# The photographic scene occupies the entire left area with no sidebar.
pic('Real G1 task montage',ROOT/'v6_trajectories/teaser_v6_clean.png',0,0,1920)

# A gently left-bending native Bezier curve from the first to second robot.
# Endpoints reuse the existing photographic placement anchors (not odometry).
meta=json.loads((ROOT/'v5_raw_1080p/verification.json').read_text())
a=meta['placements']['1']['target_anchor'];next_start=meta['placements']['2']['target_anchor']
# Stop 28px before the next white path, along its incoming direction.
import math
distance=math.dist(a,next_start)
b=[next_start[j]+(a[j]-next_start[j])*28/distance for j in range(2)]
controls=[[a[0]+(b[0]-a[0])/3-52,a[1]+(b[1]-a[1])/3],
          [a[0]+2*(b[0]-a[0])/3-52,a[1]+2*(b[1]-a[1])/3]]
# Trim only the tail with de Casteljau subdivision, preserving the remaining arc.
original_start=list(a);trim_t=.18
def lerp(p,q):return [(1-trim_t)*p[j]+trim_t*q[j] for j in range(2)]
p01=lerp(a,controls[0]);p12=lerp(controls[0],controls[1]);p23=lerp(controls[1],b)
p012=lerp(p01,p12);p123=lerp(p12,p23)
a=lerp(p012,p123);controls=[p123,p23]
x0=min(a[0],b[0]);y0=min(a[1],b[1]);cw=abs(b[0]-a[0]);ch=abs(b[1]-a[1])
curve=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,u(x0),u(y0),u(cw),u(ch))
curve.name='Navigation curve robot 1 to robot 2 - editable Bezier'
sp=curve._element.spPr
geom=sp.find('{http://schemas.openxmlformats.org/drawingml/2006/main}prstGeom')
sp.remove(geom)
custom=OxmlElement('a:custGeom')
for name in ['avLst','gdLst','ahLst','cxnLst']:custom.append(OxmlElement('a:'+name))
rr=OxmlElement('a:rect')
for k,v in [('l','0'),('t','0'),('r','r'),('b','b')]:rr.set(k,v)
custom.append(rr)
paths=OxmlElement('a:pathLst');path=OxmlElement('a:path')
for k,v in [('w','100000'),('h','100000'),('fill','none'),('stroke','1')]:path.set(k,v)
def curvept(p):
 q=OxmlElement('a:pt');q.set('x',str(round((p[0]-x0)/cw*100000)));q.set('y',str(round((p[1]-y0)/ch*100000)));return q
move=OxmlElement('a:moveTo');move.append(curvept(a));path.append(move)
bezier=OxmlElement('a:cubicBezTo')
for p in [*controls,b]:bezier.append(curvept(p))
path.append(bezier);paths.append(path);custom.append(paths);sp.insert(1,custom)
curve.fill.background();curve.line.color.rgb=c('399BE0');curve.line.width=Pt(4*.72)
curve.line._get_or_add_ln().set('cap','rnd')
# Explicit native open arrowhead: reliable in PowerPoint and LibreOffice.
vx=b[0]-controls[-1][0];vy=b[1]-controls[-1][1];vl=math.hypot(vx,vy);vx/=vl;vy/=vl
size=16;half=8
wing1=[b[0]-vx*size-vy*half,b[1]-vy*size+vx*half]
wing2=[b[0]-vx*size+vy*half,b[1]-vy*size-vx*half]
lines('Navigation small arrowhead',[wing1,b,wing2],'399BE0',4)

(P/'navigation_curve.json').write_text(json.dumps({'start_robot':1,'end_robot':2,'start':a,'end':b,'controls':controls,'stroke_design_px':4,'color':'#399BE0','type':'native cubic Bezier','schematic':True,'next_white_path_start':next_start,'endpoint_gap_px':math.dist(b,next_start),'arrow':'small native open arrowhead','original_start':original_start,'tail_trim_parameter':trim_t,'tail_clearance_from_original_anchor_px':math.dist(a,original_start)},indent=2)+'\n')

for i,(f,label,col) in enumerate([('head_5_annotated.png','Before alignment','FFF3B0'),('head_6_annotated.png','After alignment','BAFFDD')]):
 x=20+i*430;y=12;w=402;h=w*.75
 rect(f'Inset {i+1} thin border',x-3,y-3,w+6,h+6,WHITE)
 pic(f'Head camera {label}',ROOT/'v8_front_rim'/f,x,y,w)
 # Caption is directly on the photo; native dark text outline, no caption band.
 cap=text(f'Inset {i+1} caption',label,x+12,y+h-53,w-24,49,38,col,True,'center')
 for para in cap.text_frame.paragraphs:
  for run in para.runs:
   ln=OxmlElement('a:ln');ln.set('w','19050')
   fill=OxmlElement('a:solidFill');rgb=OxmlElement('a:srgbClr');rgb.set('val','18242C');fill.append(rgb);ln.append(fill)
   run._r.get_or_add_rPr().insert(0,ln)
# Preserve the compact, editable skill bridge within the scene.
start=1170;gap=28;bw=214
for i,(label,color,fill) in enumerate([('Navigate',BLUE,'EBF3F9'),('GATE',ORANGE,PALE),('Manipulate',TEAL,'EAF5F1')]):
 x=start+i*(bw+gap)
 rect(f'Skill {label}',x,23,bw,65,fill,r=True)
 text(f'Skill label {label}',label,x+5,29,bw-10,52,37,color,True,'center')
 if i<2:lines('Skill transition '+str(i),[(x+bw+4,56),(x+bw+gap-4,56)],MUTED,2,True)

cohorts={
 'w/o GATE':[{'cohort':'Full task T1-T4','successes':6,'trials':40},{'cohort':'Near-field T1','successes':24,'trials':50},{'cohort':'Near-field T2','successes':8,'trials':43}],
 'w/ GATE':[{'cohort':'Full task T1-T4','successes':33,'trials':40},{'cohort':'Near-field T1','successes':45,'trials':54},{'cohort':'Near-field T2','successes':49,'trials':53}]
}
pooled={key:{'successes':sum(r['successes'] for r in rows),'trials':sum(r['trials'] for r in rows)} for key,rows in cohorts.items()}
for key,v in pooled.items():v['percent']=100*v['successes']/v['trials']
rx=1970;rw=690
text('Task success title','Task success',rx,38,rw,72,54,INK,True)
for i,(label,color) in enumerate([('w/o GATE',GRAY),('w/ GATE',ORANGE)]):
 v=pooled[label];y=166+i*185
 text(f'{label} label',label,rx,y,380,64,48,color,True)
 text(f'{label} percent',f"{v['percent']:.1f}%",rx+391,y-6,299,78,65,color,True,'right')
 rect(f'{label} 100 percent track',rx,y+80,rw,42,BG,r=True)
 rect(f'{label} success bar',rx,y+80,rw*v['percent']/100,42,color,r=True)
 text(f'{label} count',f"{v['successes']}/{v['trials']}",rx,y+128,rw,44,35,MUTED,False,'right')
# Right bottom: capabilities, with all checkboxes after the labels.
text('Agent capabilities title','Agent capabilities',rx,627,rw,74,51,INK,True)
for i,label in enumerate(['Semantic navigation','Geometric alignment','Whole-body skills','Long-horizon tasks']):
 y=735+i*81
 text(f'Capability {i+1} label',label,rx,y,580,63,44,INK,True)
 cx=rx+rw-57;cy=y+10
 rect(f'Capability {i+1} checkbox',cx,cy,45,45,WHITE,TEAL,linewidth=2.2)
 lines(f'Capability {i+1} tick',[(cx+8,cy+24),(cx+18,cy+34),(cx+37,cy+12)],TEAL,4.2)

notes='''GATE teaser V14 — requested pooled comparison and embedded camera panels.
No AI image generation. Photos are real recorded-camera assets; text, checks, connectors and bars are native editable PowerPoint objects. The two camera panels are wholly inside the scene. Captions are outlined colored text directly over the photos, with no white caption strip. Left sidebar removed; capability checkboxes are to the right of the words.

POOLING REQUESTED BY USER. Rates are ratios of summed successes to summed trial counts, NOT unweighted averages of percentages:
w/ GATE = (33 + 45 + 49)/(40 + 54 + 53) = 127/147 = 86.3946%.
w/o GATE = (6 + 24 + 8)/(40 + 50 + 43) = 38/133 = 28.5714%.
Includes complete-task T1-T4 and separate near-field T1/T2 cohorts. The full-task component is 33/40 versus 6/40; near-field components are T1 45/54 versus 24/50, T2 49/53 versus 8/43. This pooled rate combines protocols and should NOT be described as end-to-end agent success alone.
Source: current /home/user/Project/Overleaf/6a925da904325a8d60afac0b/supplementary.tex tab:main and tab:near_field_results (reviewed 2026-09-26). Main task counts also match figures/data/realworld_success_progress.json. No trial annotation or archive statistics changed.
The current manuscript describes these cohorts as not matched by initial configuration; an earlier subset of T2 GATE + VLA was selected for successful outcomes. Consequently the figure reports a descriptive comparison of the recorded cohorts, not an unbiased causal effect. These qualifications belong in the caption alongside the pooled protocol. In the complete-task comparison, direct handoff bypasses alignment and its associated readiness checks.

Photo provenance: V6 clean scene (same base as V8), V8 head_5_annotated.png and head_6_annotated.png. Existing overlaid poses and dashed trajectories are schematic. YOLOE/Hough bin overlays were computed offline with the actual recorded prompt green rubbish bin; the detected red segment is the front lip. Camera panels illustrate handoff observations, not by themselves physical task success.
Suggested caption sentence: Pooled success combines complete-task and near-field trials (127/147 with GATE; 38/133 without GATE). This is a descriptive comparison of recorded, unmatched cohorts; the near-field GATE cohort includes an earlier success-selected subset. Task-specific results and protocols are reported separately in the main text and appendix.
'''
slide.notes_slide.notes_text_frame.text=notes+'\nV14: Shortened the blue tail by removing the first 18% of the Bezier parameter interval, preserving the remaining curve and small endpoint arrow exactly. This clears the first robot. Tip remains 28px before the white path. Schematic path only.'
for shape in slide.shapes:
 sppr=shape._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}spPr')
 if sppr is not None:sppr.append(OxmlElement('a:effectLst'))
 for ref in shape._element.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}effectRef'):ref.set('idx','0')
out=P/'GATE_teaser_editable.pptx';prs.save(out)
r=Presentation(out);s=r.slides[0]
for sh in s.shapes:
 assert sh.left>=0 and sh.top>=0,(sh.name,'negative position')
 assert sh.left+sh.width<=r.slide_width+u(1) and sh.top+sh.height<=r.slide_height+u(1),(sh.name,'out of canvas')
assert len(r.slides)==1 and not any(x.has_table for x in s.shapes)
for a in asset_checks[1:]:
 x,y,w,h=a['placement'];assert x>=0 and y>=0 and x+w<=1920 and y+h<=1080
assert pooled['w/ GATE']['successes']==127 and pooled['w/o GATE']['trials']==133
report={'slide_size_px':[W,H],'shapes':len(s.shapes),'native_textboxes':sum(x.has_text_frame for x in s.shapes),'native_tables':0,'photographs':len(asset_checks),'assets':asset_checks,'text_checks':text_checks,'ai_generated':False,'pooled':pooled,'cohorts':cohorts,'checks':['native PowerPoint reopened','all shapes inside canvas','textbox dimensions checked','insets wholly inside main photograph','no caption strip','no left sidebar','capability checks right of labels']}
(P/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(out);print(json.dumps(pooled,indent=2))
