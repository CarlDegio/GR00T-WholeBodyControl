"""Three native PowerPoint visual styles; preserved photographs and curve.
All new pills, symbols, charts and typography are editable vector objects.
"""
from pathlib import Path
from copy import deepcopy
from io import BytesIO
import sys,math,json,hashlib
sys.path.insert(0,'/tmp/align-ppt-deps')
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR,MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR,MSO_AUTO_SIZE
from pptx.oxml.xmlchemy import OxmlElement
from PIL import ImageFont
P=Path(__file__).resolve().parent;ROOT=P.parent
BASE=Presentation(ROOT/'v14_trimmed_tail/GATE_teaser_editable.pptx')
W,H=2700,1080
CAPS=['Semantic navigation','Geometric alignment','Whole-body skills','Long-horizon tasks']
COUNTS={'w/o GATE':(38,133),'w/ GATE':(127,147)}
VALUES={k:100*n/d for k,(n,d) in COUNTS.items()}
FONTFILES={('Ubuntu',False):'/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf',('Ubuntu',True):'/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',('Liberation Serif',False):'/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf',('Liberation Serif',True):'/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf',('Liberation Sans',False):'/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',('Liberation Sans',True):'/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'}
text_checks=[];groups=[]

def u(x):return Inches(x/100)
def rgb(x):return RGBColor.from_string(x)
def no_effect(s):
 for ref in s._element.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}effectRef'):ref.set('idx','0')
 sp=s._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}spPr')
 if sp is not None:sp.append(OxmlElement('a:effectLst'))

def rect(sl,name,x,y,w,h,fill,stroke=None,radius=0,lw=1):
 s=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,u(x),u(y),u(w),u(h));s.name=name
 if fill is None:s.fill.background()
 else:s.fill.solid();s.fill.fore_color.rgb=rgb(fill)
 if stroke:s.line.color.rgb=rgb(stroke);s.line.width=Pt(lw*.72)
 else:s.line.fill.background()
 if radius:s.adjustments[0]=min(.5,radius/min(w,h))
 no_effect(s);return s

def ellipse(sl,name,x,y,w,h,fill,stroke=None,lw=1):
 s=sl.shapes.add_shape(MSO_SHAPE.OVAL,u(x),u(y),u(w),u(h));s.name=name
 if fill is None:s.fill.background()
 else:s.fill.solid();s.fill.fore_color.rgb=rgb(fill)
 if stroke:s.line.color.rgb=rgb(stroke);s.line.width=Pt(lw*.72)
 else:s.line.fill.background()
 no_effect(s);return s

def text(sl,name,txt,x,y,w,h,size=40,color='26343D',bold=False,family='Ubuntu',align='left'):
 s=sl.shapes.add_textbox(u(x),u(y),u(w),u(h));s.name=name
 tf=s.text_frame;tf.clear();tf.word_wrap=False;tf.auto_size=MSO_AUTO_SIZE.NONE
 tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0;tf.vertical_anchor=MSO_ANCHOR.MIDDLE
 for j,ln in enumerate(txt.split('\n')):
  p=tf.paragraphs[0] if j==0 else tf.add_paragraph();p.text=ln;p.alignment={'left':PP_ALIGN.LEFT,'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}[align];p.space_before=p.space_after=Pt(0);p.line_spacing=1.08
  for run in p.runs:run.font.name=family;run.font.size=Pt(size*.72);run.font.bold=bold;run.font.color.rgb=rgb(color)
 ff=ImageFont.truetype(FONTFILES[(family,bold)],round(size*10))
 width=max(ff.getlength(ln)/10 for ln in txt.split('\n'))
 assert width<w-1,(name,width,w)
 assert size*1.08*len(txt.split('\n'))<=h+2,(name,'text height')
 text_checks.append({'slide':len(groups)//3+1,'name':name,'text':txt,'width':round(width,2),'box_width':w,'font':family,'size':size})
 no_effect(s);return s

def line(sl,name,pts,color,lw=2):
 shapes=[]
 for j,(a,b) in enumerate(zip(pts,pts[1:])):
  s=sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,u(a[0]),u(a[1]),u(b[0]),u(b[1]));s.name=f'{name} {j}'
  s.line.color.rgb=rgb(color);s.line.width=Pt(lw*.72);s.line._get_or_add_ln().set('cap','rnd');no_effect(s);shapes.append(s)
 return shapes

def group(sl,name,start):
 g=sl.shapes.add_group_shape(list(sl.shapes)[start:]);g.name=name;return g

def badge(sl,label,x,y,w,h,color,fill,style):
 st=len(sl.shapes)
 fg='FFFFFF' if style=='dark' else color
 body=color if style=='dark' else fill
 rect(sl,'Pill background',x,y,w,h,body,None if style=='dark' else color,radius=h/2,lw=1.6)
 # Small geometric marks reinforce each role, drawn as editable line shapes.
 if label=='Manipulate':
  # Text-only pill fits the narrow clear region above the open rubbish bin.
  text(sl,'Pill label',label,x+10,y+8,w-20,h-16,29,fg,True,family='Liberation Sans',align='center')
 else:
  cx=x+27;cy=y+h/2
  if label=='Navigate':
   line(sl,'Navigation glyph',[(cx+8,cy+9),(cx-8,cy-9)],fg,2.5)
   line(sl,'Navigation glyph tip',[(cx+3,cy-9),(cx-8,cy-9),(cx-8,cy+2)],fg,2.5)
  else:
   ellipse(sl,'Alignment glyph',cx-10,cy-10,20,20,None,fg,2)
   line(sl,'Alignment glyph horizontal',[(cx-15,cy),(cx+15,cy)],fg,1.8)
   line(sl,'Alignment glyph vertical',[(cx,cy-15),(cx,cy+15)],fg,1.8)
  text(sl,'Pill label',label,x+49,y+8,w-60,h-16,31,fg,True,family='Liberation Sans',align='center')
 g=group(sl,'MOVE ME — '+label,st)
 groups.append({'label':label,'box':[x,y,w,h],'native_group':True,'children':len(g.shapes)})
 return g

def checked(sl,x,y,color,bg=None,style='square'):
 if style=='circle':ellipse(sl,'Check circle',x,y,39,39,bg,None)
 else:rect(sl,'Check square',x,y,37,37,bg,color if bg is None else None,radius=8,lw=1.5)
 line(sl,'Check mark',[(x+9,y+19),(x+16,y+26),(x+29,y+11)],color,2.8)

def base_slide(prs,style):
 sl=prs.slides.add_slide(prs.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=rgb('FFFFFF')
 keep=('Real G1','Navigation curve','Navigation small arrowhead','Inset ','Head camera')
 for s in BASE.slides[0].shapes:
  if not s.name.startswith(keep):continue
  el=deepcopy(s._element)
  for blip in el.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
   key='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'
   rid=blip.get(key)
   if rid:
    blob=BASE.slides[0].part.related_part(rid).blob
    _,newrid=sl.part.get_or_add_image_part(BytesIO(blob));blip.set(key,newrid)
  sl.shapes._spTree.insert_element_before(el,'p:extLst')
 if style=='A':colors=[('377AA5','F4FAFD'),('B46B36','FFF7EC'),('2E8172','F0FAF6')];bs='light'
 elif style=='B':colors=[('416F91','E5F0F6'),('AC663B','F9E8D8'),('497D69','E8F2E8')];bs='light'
 else:colors=[('377AA5','377AA5'),('C58645','C58645'),('398B77','398B77')];bs='dark'
 for args,(color,fill) in zip([('Navigate',1108,840,218,62),('GATE',461,920,152,58),('Manipulate',18,582,190,60)],colors):badge(sl,*args,color,fill,bs)
 sl.notes_slide.notes_text_frame.text=BASE.slides[0].notes_slide.notes_text_frame.text+f'\nV15 option {style}: three skill pills are individually grouped native PowerPoint shapes. Drag the group to move a full badge; ungroup to edit its components. Right-side style is redesigned; all displayed statistics and capability wording are unchanged.'
 return sl

def a_style(sl):
 ink='223440';muted='7C8B94';accent='BC713D';base='94A6B1';track='EAF0F2'
 rect(sl,'Sidebar white',1920,0,780,1080,'FFFFFF')
 x=1982;w=648
 text(sl,'Task success','Task success',x,51,w,68,51,ink,False)
 for i,(label,col) in enumerate([('w/o GATE',base),('w/ GATE',accent)]):
  y=174+i*191;n,d=COUNTS[label]
  text(sl,label,label,x,y+9,260,52,36,ink,False)
  text(sl,label+' percentage',f'{VALUES[label]:.1f}%',x+308,y-7,340,95,80,col,True,align='right')
  rect(sl,label+' track',x,y+94,w,17,track,radius=8.5)
  rect(sl,label+' progress',x,y+94,w*VALUES[label]/100,17,col,radius=8.5)
  text(sl,label+' count',f'{n}/{d}',x,y+125,w,40,30,muted,False,align='right')
 line(sl,'Section divider',[(x,596),(x+w,596)],'DDE5E8',1.1)
 text(sl,'Agent capabilities','Agent capabilities',x,639,w,60,43,ink,False)
 for i,label in enumerate(CAPS):
  y=734+i*77
  text(sl,'Capability '+label,label,x,y,560,54,37,ink,False)
  checked(sl,x+w-41,y+8,'44877B',None)

def b_style(sl):
 ink='343A37';muted='848C84';accent='AD6940';base='789391';warm='F7F5EF'
 rect(sl,'Warm sidebar',1920,0,780,1080,warm)
 x=1978;w=664
 text(sl,'Task success','Task success',x,49,w,79,61,ink,False,family='Liberation Serif')
 for i,(label,col) in enumerate([('w/o GATE',base),('w/ GATE',accent)]):
  xx=x+i*345;n,d=COUNTS[label]
  rect(sl,label+' metric card',xx,172,319,337,'FFFFFF',None,radius=23)
  text(sl,label,label,xx+23,197,273,51,34,ink,False,align='center')
  text(sl,label+' percentage',f'{VALUES[label]:.1f}%',xx+17,274,285,110,80,col,False,family='Liberation Serif',align='center')
  text(sl,label+' count',f'{n}/{d}',xx+22,390,275,45,31,muted,False,align='center')
  rect(sl,label+' track',xx+28,467,263,12,'EEEFE8',radius=6)
  rect(sl,label+' progress',xx+28,467,263*VALUES[label]/100,12,col,radius=6)
 text(sl,'Agent capabilities','Agent capabilities',x,592,w,70,52,ink,False,family='Liberation Serif')
 rect(sl,'Capability card',x,692,w,339,'FFFFFF',None,radius=21)
 for i,label in enumerate(CAPS):
  yy=710+i*79
  text(sl,'Capability '+label,label,x+26,yy,536,56,36,ink,False)
  checked(sl,x+w-67,yy+9,'598071','EBF2EC')
  if i<3:line(sl,'Capability rule',[(x+27,yy+66),(x+w-27,yy+66)],'ECEEE8',.8)

def ring(sl,x,y,r,percent,color):
 st=len(sl.shapes)
 ellipse(sl,'Ring track',x-r,y-r,r*2,r*2,None,'304750',12)
 # Native open freeform arc, sampled densely; editable points after ungrouping.
 points=[]
 for i in range(201):
  a=math.radians(-90+360*percent/100*i/200);points.append((x+r*math.cos(a),y+r*math.sin(a)))
 fb=sl.shapes.build_freeform(u(points[0][0]),u(points[0][1]))
 fb.add_line_segments([(u(a),u(b)) for a,b in points[1:]],close=False)
 s=fb.convert_to_shape();s.name='Success ring arc';s.fill.background();s.line.color.rgb=rgb(color);s.line.width=Pt(12*.72);s.line._get_or_add_ln().set('cap','rnd');no_effect(s)
 group(sl,'Editable success ring',st)

def c_style(sl):
 ink='EFF5F4';muted='A1B2B8';mint='8CD7BC';base='9AB3C0'
 rect(sl,'Navy sidebar',1920,0,780,1080,'172E38')
 x=1980;w=660
 text(sl,'Task success','Task success',x,53,w,67,50,ink,False)
 for i,(label,col) in enumerate([('w/o GATE',base),('w/ GATE',mint)]):
  cx=2134+i*340;n,d=COUNTS[label]
  text(sl,label,label,cx-145,151,290,51,34,ink,False,align='center')
  ring(sl,cx,342,123,VALUES[label],col)
  text(sl,label+' percentage',f'{VALUES[label]:.1f}%',cx-117,293,234,82,63,col,True,align='center')
  text(sl,label+' count',f'{n}/{d}',cx-100,380,200,43,30,muted,False,align='center')
 line(sl,'Section divider',[(x,578),(x+w,578)],'38505A',1.2)
 text(sl,'Agent capabilities','Agent capabilities',x,623,w,61,43,ink,False)
 for i,label in enumerate(CAPS):
  yy=724+i*78
  rect(sl,'Capability row',x,yy,w,66,'203D48',None,radius=12)
  text(sl,'Capability '+label,label,x+20,yy+6,547,53,35,ink,False)
  checked(sl,x+w-60,yy+13,'8CD7BC','2C514E',style='circle')

styles=[('A','A_editorial_white',a_style),('B','B_warm_cards',b_style),('C','C_navy_rings',c_style)]
allprs=Presentation();allprs.slide_width=u(W);allprs.slide_height=u(H)
for key,name,draw in styles:
 sl=base_slide(allprs,key);draw(sl)
 # Standalone copy retains all native shapes and grouping.
 one=Presentation();one.slide_width=u(W);one.slide_height=u(H)
 dest=one.slides.add_slide(one.slide_layouts[6])
 for shape in sl.shapes:
  el=deepcopy(shape._element)
  for blip in el.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
   k='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed';rid=blip.get(k)
   if rid:
    _,nr=dest.part.get_or_add_image_part(BytesIO(sl.part.related_part(rid).blob));blip.set(k,nr)
  dest.shapes._spTree.insert_element_before(el,'p:extLst')
 dest.notes_slide.notes_text_frame.text=sl.notes_slide.notes_text_frame.text
 one.save(P/f'{name}.pptx')
allprs.save(P/'GATE_teaser_three_styles.pptx')
report={'variants':[x[1] for x in styles],'skill_groups':groups,'text_checks':text_checks,'slide_size':[W,H],'rates':{k:{'counts':COUNTS[k],'percent':v} for k,v in VALUES.items()},'photos_and_curve_preserved_from':str(ROOT/'v14_trimmed_tail/GATE_teaser_editable.pptx'),'checks':[]}
r=Presentation(P/'GATE_teaser_three_styles.pptx')
assert len(r.slides)==3
for sl in r.slides:
 pills=[s for s in sl.shapes if s.name.startswith('MOVE ME')]
 assert len(pills)==3 and all(s.shape_type==MSO_SHAPE_TYPE.GROUP for s in pills)
 for s in sl.shapes:
  assert s.left>=0 and s.top>=0 and s.left+s.width<=r.slide_width+u(1) and s.top+s.height<=r.slide_height+u(1),(s.name,'bounds')
 assert sum(s.shape_type==MSO_SHAPE_TYPE.PICTURE for s in sl.shapes)==3
report['checks']=['Three slides reopen successfully','Three separately movable native skill groups per slide','Original three photos retained per slide','All shape bounds and text dimensions checked']
(P/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('Saved three editable slides and standalone decks.')
