"""Vector-style annotations only; photographed pixels and poses come from V5."""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image,ImageDraw

P=Path(__file__).resolve().parent
SRC=P.parent/'v5_raw_1080p'
meta=json.loads((SRC/'verification.json').read_text())
anchors={int(k):v['target_anchor'] for k,v in meta['placements'].items()}
curves=[
 dict(name='2_to_3',start_id=2,end_id=3,segments=[
  [anchors[2],[970,778],[913,766],[883,758]],
  [[883,758],[869,754],[857,747],anchors[3]]],
  evidence='93.210–129.160s: short diagonal approach toward table; gently curved schematic'),
 dict(name='5_to_6',start_id=5,end_id=6,segments=[
  [anchors[5],[658,762],[580,797],[546,834]],
  [[546,834],[533,860],[621,926],[678,938]],
  [[678,938],[708,940],[686,903],anchors[6]]],
  evidence='287.567–331.820s; source 297s feet about (550,835), 307s about (700,948), 312s about (704,919). Approximate bends corrected to existing composite endpoints.')]
S=3;W,H=1920,1080
overlay=Image.new('RGBA',(W*S,H*S));d=ImageDraw.Draw(overlay)
WHITE=(255,255,255,248);SHADOW=(20,27,32,145)
def xy(p):return tuple(float(z)*S for z in p)
def stroke(points,width=4,color=WHITE,shadow=True):
 q=[xy(a) for a in points]
 if shadow:d.line(q,fill=SHADOW,width=round((width+2)*S),joint='curve')
 d.line(q,fill=color,width=round(width*S),joint='curve')
 r=width*S/2
 for x,y in [q[0],q[-1]]:d.ellipse((x-r,y-r,x+r,y+r),fill=color)
def sample(seg):
 p=np.array(seg,float);t=np.linspace(0,1,160)[:,None]
 return (1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t**2*p[2]+t**3*p[3]
def dashes(points,on=13,off=9,width=4):
 points=np.array(points);delta=np.diff(points,axis=0);dist=np.r_[0,np.cumsum(np.linalg.norm(delta,axis=1))]
 for a in np.arange(0,dist[-1],on+off):
  b=min(a+on,dist[-1]);s=np.r_[a,dist[(dist>a)&(dist<b)],b]
  q=np.column_stack([np.interp(s,dist,points[:,j]) for j in range(2)])
  stroke(q,width)
def arrow(tip,direction,size=14,width=4):
 tip=np.array(tip,float);v=np.array(direction,float);v/=np.linalg.norm(v);n=np.array([-v[1],v[0]])
 stroke([tip-v*size+n*size*.52,tip,tip-v*size-n*size*.52],width)
for path in curves:
 assert np.allclose(path['segments'][0][0],anchors[path['start_id']])
 assert np.allclose(path['segments'][-1][-1],anchors[path['end_id']])
 points=np.concatenate([sample(s) for s in path['segments']])
 dashes(points)
 p=path['segments'][0][0];x,y=xy(p);r=5*S
 d.ellipse((x-r-S,y-r-S,x+r+S,y+r+S),fill=SHADOW)
 d.ellipse((x-r,y-r,x+r,y+r),fill=WHITE)
 arrow(points[-1],points[-1]-points[-5],size=15)
# A small ground-plane elliptical arrow beside endpoint 6 indicates turning.
# Schematic screen-space direction only; no calibrated yaw or angle asserted.
center=(741,866);radii=(24,12)
angles=np.linspace(math.radians(-105),math.radians(150),100)
arc=np.column_stack([center[0]+radii[0]*np.cos(angles),center[1]+radii[1]*np.sin(angles)])
stroke(arc,width=3)
arrow(arc[-1],arc[-1]-arc[-3],size=9,width=3)
overlay=overlay.resize((W,H),Image.Resampling.LANCZOS)
overlay.save(P/'trajectory_overlay.png')
for kind in ['numbered','clean']:
 base=Image.open(SRC/f'teaser_v5_{kind}.png').convert('RGBA')
 base.alpha_composite(overlay);base.convert('RGB').save(P/f'teaser_v6_{kind}.png')
# SVG retains the editable curve geometry without rasterizing the photos.
paths=[]
for curve in curves:
 start=curve['segments'][0][0];s=f'M {start[0]:.3f},{start[1]:.3f}'
 for seg in curve['segments']:
  s+=' C '+' '.join(f'{q[0]:.3f},{q[1]:.3f}' for q in seg[1:])
 paths.append(f'<path d="{s}" fill="none" stroke="white" stroke-width="4" stroke-dasharray="13 9" stroke-linecap="round" marker-end="url(#arrow)"/>')
svg='<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080"><defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M 0 0 L 10 5 L 0 10" fill="none" stroke="white" stroke-width="1.5"/></marker></defs>'+''.join(paths)+'</svg>'
(P/'editable_paths.svg').write_text(svg)
(P/'trajectory_notes.json').write_text(json.dumps(dict(curves=curves,rotation_icon=dict(center=center,radii=radii,meaning='Schematic alignment turn, not measured yaw'),accuracy='Visually estimated from intermediate video frames; user-approved composite endpoint placements preserved exactly. Not measured odometry.',base='../v5_raw_1080p/teaser_v5_numbered.png'),indent=2))
print('Saved two curved dashed paths with exact V5 endpoint anchors; rotation symbol at endpoint 6.')
