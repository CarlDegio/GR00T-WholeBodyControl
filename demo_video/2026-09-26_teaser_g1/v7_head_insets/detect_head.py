from pathlib import Path
import os,sys,json
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'tools/yoloe26m/config'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch,cv2,numpy as np
from ultralytics import YOLOE
from ultralytics.utils import ops
from gear_sonic.utils.inference.base_pose.servo import _largest_filled_component,_dilate_yaw_align_edge_mask,_rgb_mask_edge_intersection,_rgb_edge_line_segments
torch.set_num_threads(4)
prompt='green rubbish bin'
model=YOLOE(str(ROOT/'tools/yoloe26m/weights/yoloe-26m-seg.pt'))
model.set_classes([prompt],embeddings=model.get_text_pe([prompt]))
results=[]
for i in [5,6]:
 source=P/f'head_{i}_raw.jpg';bgr=cv2.imread(str(source));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
 result=model.predict(source=bgr,device='cpu',imgsz=640,conf=.25,iou=.5,half=False,retina_masks=True,verbose=False,save=False)[0]
 if result.boxes is None or result.masks is None or len(result.boxes)==0:raise RuntimeError(f'No YOLOE detection in head {i}')
 k=int(result.boxes.conf.argmax());bbox=result.boxes.xyxy[k].cpu().numpy();conf=float(result.boxes.conf[k])
 masks=ops.scale_masks(result.masks.data[:,None],rgb.shape[:2],padding=True,mode='nearest')[:,0].cpu().numpy()
 mask=(masks[k]>.5).astype(np.uint8)
 cv2.imwrite(str(P/f'head_{i}_yoloe_mask.png'),mask*255)
 edge_mask=_rgb_mask_edge_intersection(rgb,_dilate_yaw_align_edge_mask(_largest_filled_component(mask)))
 lines=_rgb_edge_line_segments(edge_mask)
 serialized=[dict(endpoint_a=list(map(float,l.endpoint_a)),endpoint_b=list(map(float,l.endpoint_b)),length_px=float(l.length)) for l in lines]
 line=serialized[0] if serialized else None
 cv2.imwrite(str(P/f'head_{i}_masked_edges.png'),edge_mask)
 vis=rgb.copy();active=mask>0
 vis[active]=np.rint(vis[active]*.80+np.array([40,245,100])*.20).astype(np.uint8)
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(vis,contours,-1,(80,255,125),2,cv2.LINE_AA)
 x1,y1,x2,y2=np.rint(bbox).astype(int);cv2.rectangle(vis,(x1,y1),(x2,y2),(70,255,115),2,cv2.LINE_AA)
 if i==6:
  if line is None:raise RuntimeError('No detected edge for head6')
  a=tuple(map(round,line['endpoint_a']));b=tuple(map(round,line['endpoint_b']))
  cv2.line(vis,a,b,(255,60,60),5,cv2.LINE_AA)
  cv2.circle(vis,a,4,(255,255,255),-1,cv2.LINE_AA);cv2.circle(vis,b,4,(255,255,255),-1,cv2.LINE_AA)
 cv2.imwrite(str(P/f'head_{i}_annotated.png'),cv2.cvtColor(vis,cv2.COLOR_RGB2BGR))
 results.append(dict(index=i,source=source.name,prompt=prompt,model='yoloe-26m-seg.pt',device='cpu',confidence=conf,bbox_xyxy=bbox.tolist(),mask_pixels=int(mask.sum()),hough_candidates=serialized,displayed_line=line if i==6 else None,line_selection='longest segment surviving the existing RGB/mask Hough filters; offline RGB only, no depth ranking',annotation_origin='offline inference; not recorded online overlay'))
 print(json.dumps(results[-1]),flush=True)
(P/'offline_detection.json').write_text(json.dumps(results,indent=2)+'\n')
