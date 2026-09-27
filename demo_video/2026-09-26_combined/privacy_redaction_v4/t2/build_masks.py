from pathlib import Path
import json,cv2,numpy as np
from anchors import tracks
P=Path(__file__).resolve().parent
cv2.setNumThreads(2)
V=P.parents[1]/'exports/Full_demo_combined_v3.mp4'
# Estimate per-frame movement in each manually anchored short interval; correct drift at both anchors.
cap=cv2.VideoCapture(str(V),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
start=min(round(t['anchors'][0][0]*60) for t in tracks);end=max(round(t['anchors'][-1][0]*60) for t in tracks)
cap.set(cv2.CAP_PROP_POS_FRAMES,start)
for tr in tracks:
 tr['fids']=np.array([round(p[0]*60) for p in tr['anchors']]);tr['pts']=np.array([p[1:] for p in tr['anchors']],float);tr['raw']={}
prev=None
for fid in range(start,end+1):
 ok,im=cap.read();assert ok
 gray=cv2.cvtColor(cv2.resize(im,(960,540)),cv2.COLOR_BGR2GRAY)
 for tr in tracks:
  fids=tr['fids'];pts=tr['pts']
  if fid<fids[0] or fid>fids[-1]:continue
  j=np.searchsorted(fids,fid,side='right')-1
  if fid==fids[j]:
   box=pts[j].copy()
  else:
   w=(fid-fids[j])/(fids[j+1]-fids[j]);box=pts[j]*(1-w)+pts[j+1]*w
   old=tr['raw'].get(fid-1)
   if old is not None and prev is not None:
    # Features on target plus small local context. Main tracking at half resolution.
    cx,cy,bw,bh=old/2
    x1=max(0,int(cx-bw*.65));x2=min(960,int(cx+bw*.65));y1=max(0,int(cy-bh*.65));y2=min(540,int(cy+bh*.65))
    c=tr['clip'];x1=max(x1,c[0]//2);x2=min(x2,c[2]//2);y1=max(y1,c[1]//2);y2=min(y2,c[3]//2)
    if x2-x1>6 and y2-y1>6:
     local=prev[y1:y2,x1:x2]
     feat=cv2.goodFeaturesToTrack(local,maxCorners=35,qualityLevel=.015,minDistance=3,blockSize=3)
     if feat is not None and len(feat)>=4:
      feat+=np.array([[[x1,y1]]],np.float32)
      nxt,st,err=cv2.calcOpticalFlowPyrLK(prev,gray,feat,None,winSize=(25,25),maxLevel=4,criteria=(3,25,.01))
      sel=(st[:,0]>0)&(err[:,0]<35)
      if sel.sum()>=4:
       delta=np.median((nxt-feat)[sel,0,:],axis=0)*2
       if np.linalg.norm(delta)<190:
        box[:2]=old[:2]+delta
  tr['raw'][fid]=box
 prev=gray
 if fid%300==0: print('tracked',fid,flush=True)
cap.release()
frames={};diagnostics=[];per_target={}
for tr in tracks:
 fids=tr['fids'];pts=tr['pts'];out={}
 for j in range(len(fids)-1):
  f0,f1=fids[j:j+2];raw=tr['raw']
  # Flow endpoint before manual reset is inferred from final prior frame, avoiding reset hiding drift.
  if f1-1 in raw: predicted=raw[f1-1][:2]
  else:predicted=pts[j,:2]
  target=pts[j+1,:2];drift=target-predicted
  diagnostics.append([tr['name'],int(f0),int(f1),float(np.linalg.norm(drift))])
  for fid in range(f0,f1):
   u=(fid-f0)/(f1-f0);b=raw[fid].copy();b[:2]+=u*drift
   # Very fast small-inset motions use dense manual anchors with a bounded margin.
   if tr['name'].startswith('chest'):
    b=pts[j]*(1-u)+pts[j+1]*u
   pad=tr['pad'];cx,cy,w,h=b;c=tr['clip']
   x1=max(c[0],int(np.floor(cx-w/2-pad)));y1=max(c[1],int(np.floor(cy-h/2-pad)))
   x2=min(c[2],int(np.ceil(cx+w/2+pad)));y2=min(c[3],int(np.ceil(cy+h/2+pad)))
   if x2>x1 and y2>y1:
    frames.setdefault(str(fid),[]).append([x1,y1,x2,y2]);out[str(fid)]=[x1,y1,x2,y2]
 per_target[tr['name']]=out
(P/'masks.json').write_text(json.dumps(dict(frames=frames,targets=[t['name'] for t in tracks],notes=['Manual anchor boxes with per-frame local optical motion and endpoint correction.']),separators=(',',':')))
(P/'per_target.json').write_text(json.dumps(per_target,separators=(',',':')))
(P/'flow_diagnostics.json').write_text(json.dumps(sorted(diagnostics,key=lambda x:-x[-1]),indent=2))
print('frames',len(frames),'targets',len(tracks))
