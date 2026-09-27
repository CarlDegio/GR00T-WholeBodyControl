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
W,H=2700,1050
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

# Main scene, unmodified original photographic composite; trajectories already included.
pic('Real G1 task montage',ROOT/'v6_trajectories/teaser_v6_clean.png',375,100,1615)
# Upper-left enlarged, side-by-side same-run camera observations.
for i,(f,label,col) in enumerate([('head_5_annotated.png','Before alignment',GRAY),('head_6_annotated.png','After alignment',ORANGE)]):
 x=395+i*452;y=16;w=410;h=w*.75
 rect(f'Inset {i+1} border',x-5,y-5,w+10,h+10,WHITE,LINE,linewidth=1)
 pic(f'Head camera {label}',ROOT/'v8_front_rim'/f,x,y,w)
 rect(f'Inset {i+1} caption band',x,y+h-56,w,56,WHITE)
 text(f'Inset {i+1} caption',label,x+12,y+h-54,w-24,49,37,col,True,'center')
# Native bridge labels above the right-hand side of the scene.
start=1300;gap=30;bw=205
for i,(label,color,fill) in enumerate([('Navigate',BLUE,'EBF3F9'),('GATE',ORANGE,PALE),('Manipulate',TEAL,'EAF5F1')]):
 x=start+i*(bw+gap)
 rect(f'Skill {label}',x,20,bw,65,fill,r=True)
 text(f'Skill label {label}',label,x+5,26,bw-10,52,37,color,True,'center')
 if i<2:lines('Skill transition '+str(i),[(x+bw+4,53),(x+bw+gap-4,53)],MUTED,2,True)

# Left: concise capabilities with native outlined squares and tick strokes.
text('GATE agent title','GATE\nagent',24,108,315,150,65,INK,True)
text('Capabilities heading','Capabilities',24,323,325,65,46,INK,True)
for i,(a,b) in enumerate([('Semantic','navigation'),('Geometric','alignment'),('Whole-body','skills'),('Long-horizon','tasks')]):
 y=422+i*131
 rect(f'Capability {i+1} checkbox',26,y+21,43,43,WHITE,TEAL,linewidth=2.1)
 lines(f'Capability {i+1} tick',[(34,y+42),(43,y+52),(62,y+31)],TEAL,4.1)
 text(f'Capability {i+1} label',a+'\n'+b,87,y,270,92,41,INK,True)

# Right top: complete-task success. Scale tracks to 100%; values native text.
rx=2036;rw=638
text('Agent results title','Agent success',rx,28,rw,71,52,INK,True)
for i,(label,val,count,color) in enumerate([('Direct handoff',15.0,'6/40',GRAY),('Full agent',82.5,'33/40',ORANGE)]):
 y=139+i*177
 text(f'{label} label',label,rx,y,392,58,42,color,True)
 text(f'{label} percent',f'{val:.1f}%',rx+399,y-5,239,69,58,color,True,'right')
 rect(f'{label} 100 percent track',rx,y+74,rw,39,BG,r=True)
 rect(f'{label} success bar',rx,y+74,rw*val/100,39,color,r=True)
 text(f'{label} count',count,rx,y+119,rw,44,33,MUTED,False,'right')
# Body table: T1/T2 x Direct VLA/GATE+VLA; headers are additional to 2x2 data.
text('Near-field results title','Near-field success',rx,553,rw,68,49,INK,True)
rows,cols=3,3
table_shape=slide.shapes.add_table(rows,cols,u(rx),u(639),u(rw),u(353));table_shape.name='Near-field success table - editable 2x2 data'
table=table_shape.table
for col,width in zip(table.columns,[92,273,273]):col.width=u(width)
for row,height in zip(table.rows,[77,138,138]):row.height=u(height)
table.first_row=False;table.first_col=False;table.horz_banding=False;table.vert_banding=False
contents=[['','Direct VLA','GATE + VLA'],['T1','48.0%\n24/50','83.3%\n45/54'],['T2','18.6%\n8/43','92.5%\n49/53']]
for r in range(rows):
 for cc in range(cols):
  cell=table.cell(r,cc);cell.text='';cell.margin_left=cell.margin_right=u(6);cell.margin_top=cell.margin_bottom=0;cell.vertical_anchor=MSO_ANCHOR.MIDDLE
  fill=WHITE if cc==0 else (BG if cc==1 else PALE)
  cell.fill.solid();cell.fill.fore_color.rgb=c(fill)
  # White separators keep the table clean and native-editable.
  for side in ['lnL','lnR','lnT','lnB']:
   ln=OxmlElement('a:'+side);ln.set('w','15240');solid=OxmlElement('a:solidFill');rgb=OxmlElement('a:srgbClr');rgb.set('val',WHITE);solid.append(rgb);ln.append(solid);cell._tc.get_or_add_tcPr().append(ln)
  for j,txt in enumerate(contents[r][cc].split('\n')):
   p=cell.text_frame.paragraphs[0] if j==0 else cell.text_frame.add_paragraph();p.text=txt;p.alignment=PP_ALIGN.CENTER;p.space_before=p.space_after=Pt(0)
   size=35 if r==0 else (48 if j==0 else 31)
   for run in p.runs:
    run.font.name='Liberation Sans';run.font.size=Pt(size*.72);run.font.bold=(j==0);run.font.color.rgb=c(INK if cc==0 else (MUTED if cc==1 else ORANGE))

notes='''GATE teaser, revision V10. User-approved fusion of concept A.
All photographs are existing real-camera assets. No generative synthesis.
Editable native elements: text, shapes, connectors/check marks, bars, and the 2x2 data table. The 2x2 data body has T1/T2 rows and Direct VLA/GATE + VLA columns, with extra headers.
Agent success: Direct handoff = 6/40 (15.0%), Full agent = 33/40 (82.5%); 4 tasks, 10 trials per task/method. Direct handoff bypasses both alignment and associated readiness checks; navigation and task policies are otherwise shared in this comparison.
Near-field success: T1 Direct VLA 24/50 (48.0%), GATE + VLA 45/54 (83.3%); T2 Direct VLA 8/43 (18.6%), GATE + VLA 49/53 (92.5%). These are separate cohorts without paired starting states and are not the same trials as the complete-agent comparison.
Source: /home/user/Project/Overleaf/6a925da904325a8d60afac0b/supplementary.tex, tab:main and tab:near_field_results; figures/data/realworld_success_progress.json.
Main scene uses the full V6 photographed composite, also the base of V8. Overlaid poses and dashed trajectories are schematic, not measured odometry. The two head-camera images are the V8 selected before/after observations, with offline YOLOE/Hough overlays from the actual recorded green rubbish bin prompt. The red line is a detected front lip. These observations illustrate handoff alignment, not by themselves manipulation success.
Capabilities: semantic navigation; geometric alignment; whole-body manipulation skills; long-horizon task execution. See current manuscript methods and G1 tasks.
A short caption should define the two result cohorts and note the schematic paths and offline visual overlays. All lengthy explanations are intentionally left out of the artwork as requested.
'''
slide.notes_slide.notes_text_frame.text=notes
# Suppress inherited theme shadows so the figure keeps a flat paper style.
for shape in slide.shapes:
 sppr=shape._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}spPr')
 if sppr is not None:
  for effect in list(sppr):
   if effect.tag in ['{http://schemas.openxmlformats.org/drawingml/2006/main}effectLst','{http://schemas.openxmlformats.org/drawingml/2006/main}effectDag']:sppr.remove(effect)
  sppr.append(OxmlElement('a:effectLst'))
out=P/'GATE_teaser_editable.pptx';prs.save(out)
# Native object bounds and reopened structure checks.
r=Presentation(out);s=r.slides[0]
for sh in s.shapes:
 assert sh.left>=0 and sh.top>=0,(sh.name,'negative position')
 assert sh.left+sh.width<=r.slide_width+u(1) and sh.top+sh.height<=r.slide_height+u(1),(sh.name,'out of canvas')
assert len(r.slides)==1
assert len([x for x in s.shapes if x.has_table])==1
assert s.shapes[-1].table.cell(2,2).text=='92.5%\n49/53'
report={'slide_size_px':[W,H],'shapes':len(s.shapes),'native_textboxes':sum(x.has_text_frame for x in s.shapes),'native_tables':1,'photographs':len(asset_checks),'assets':asset_checks,'text_checks':text_checks,'ai_generated':False,'checks':['native PowerPoint reopened','all shapes inside canvas','textbox widths and heights checked','2x2 data cells and headers present']}
(P/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(out);print('Native shapes:',len(s.shapes),'photos:',len(asset_checks),'tables:',1)
