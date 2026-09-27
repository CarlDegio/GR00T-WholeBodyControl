from pathlib import Path
import json, sys
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
# Reuse the same local offline model and actual RGB edge detector.
src=(P.parent/'v7_head_insets/detect_head.py').read_text()
exec(src.split('results=[]')[0])
P=Path(__file__).resolve().parent
from PIL import Image, ImageDraw
V=ROOT/'outputs/experiments/full_vln_T4_visible_01_0fce0b283641/demo/0fce0b283641-g1/video'
entries=[json.loads(l) for l in (V/'head.jsonl').read_text().splitlines()]
center=next(i for i,e in enumerate(entries) if e['sequence']==8583)
records=[];tiles=[]
with (V/'head.mjpeg').open('rb') as f:
 for delta in range(-8,9):
  e=entries[center+delta];f.seek(e['offset']);raw=f.read(e['length'])
  name=f"head6_{delta:+03d}"
  (P/f'{name}.jpg').write_bytes(raw)
  bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
  result=model.predict(source=bgr,device='cpu',imgsz=640,conf=.25,iou=.5,half=False,retina_masks=True,verbose=False,save=False)[0]
  k=int(result.boxes.conf.argmax());bbox=result.boxes.xyxy[k].cpu().numpy()
  mask=(ops.scale_masks(result.masks.data[:,None],rgb.shape[:2],padding=True,mode='nearest')[:,0].cpu().numpy()[k]>.5).astype(np.uint8)
  edge_mask=_rgb_mask_edge_intersection(rgb,_dilate_yaw_align_edge_mask(_largest_filled_component(mask)))
  lines=_rgb_edge_line_segments(edge_mask)
  ls=[dict(endpoint_a=list(map(float,l.endpoint_a)),endpoint_b=list(map(float,l.endpoint_b)),length_px=float(l.length)) for l in lines]
  # The front lip lies at about 40–50% of the detected full bin height.
  eligible=[]
  x1,y1,x2,y2=bbox
  for j,l in enumerate(ls):
   a,b=l['endpoint_a'],l['endpoint_b'];dx=abs(a[0]-b[0]);dy=abs(a[1]-b[1]);ym=(a[1]+b[1])/2
   if dx>30 and dy/dx<.13 and .34<(ym-y1)/(y2-y1)<.54:eligible.append(j)
  chosen=max(eligible,key=lambda j:ls[j]['length_px']) if eligible else None
  cv2.imwrite(str(P/f'{name}_mask.png'),mask*255)
  vis=rgb.copy();active=mask>0
  vis[active]=np.rint(vis[active]*.8+np.array([40,245,100])*.2).astype(np.uint8)
  contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(vis,contours,-1,(80,255,125),2,cv2.LINE_AA)
  cv2.rectangle(vis,(round(float(x1)),round(float(y1))),(round(float(x2)),round(float(y2))),(70,255,115),2,cv2.LINE_AA)
  if chosen is not None:
   l=ls[chosen];a=tuple(map(round,l['endpoint_a']));b=tuple(map(round,l['endpoint_b']))
   cv2.line(vis,a,b,(255,60,60),5,cv2.LINE_AA)
   for pt in [a,b]:cv2.circle(vis,pt,4,(255,255,255),-1,cv2.LINE_AA)
  Image.fromarray(vis).save(P/f'{name}_annotated.png')
  tile=Image.fromarray(vis).crop((260,0,465,250)).resize((287,350))
  ImageDraw.Draw(tile).text((5,320),f"delta {delta:+d}, line {chosen}",fill='white',stroke_width=2,stroke_fill='black');tiles.append(tile)
  records.append(dict(delta=delta,frame_index_entry=e,video_seconds=(e['source_timestamp_ns']-1790342419000000000)/1e9,bbox_xyxy=bbox.tolist(),confidence=float(result.boxes.conf[k]),hough_candidates=ls,selected_candidate=chosen))
  print(delta,chosen,ls[chosen] if chosen is not None else None,flush=True)
(P/'candidates.json').write_text(json.dumps(records,indent=2)+'\n')
sheet=Image.new('RGB',(287*6,350*3),(35,35,35))
for i,t in enumerate(tiles):sheet.paste(t,((i%6)*287,(i//6)*350))
sheet.save(P/'candidates.jpg')
