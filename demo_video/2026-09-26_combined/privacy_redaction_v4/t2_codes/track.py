import cv2,numpy as np,json,pathlib
out=pathlib.Path('demo_video/2026-09-26_combined/privacy_redaction_v4/t2_codes');c=cv2.VideoCapture('demo_video/2026-09-26_combined/exports/Full_demo_combined_v3.mp4');c.set(1,4260);gray={}
for n in range(4260,7080):
 ok,f=c.read()
 if n%2==0:gray[n]=cv2.cvtColor(cv2.resize(f,(960,540)),cv2.COLOR_BGR2GRAY)
print('decoded',flush=True)
def flow(prev,cur,rect):
 x1,y1,x2,y2=rect;mask=np.zeros_like(prev);a=max(0,int(x1)-3);b=max(0,int(y1)-3);d=min(960,int(x2)+3);e=min(540,int(y2)+3);mask[b:e,a:d]=255
 pts=cv2.goodFeaturesToTrack(prev,100,.008,3,mask=mask)
 if pts is None or len(pts)<4:return rect,0
 p,st,err=cv2.calcOpticalFlowPyrLK(prev,cur,pts,None,winSize=(25,25),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01))
 back,bs,be=cv2.calcOpticalFlowPyrLK(cur,prev,p,None,winSize=(25,25),maxLevel=3)
 good=(st.ravel()==1)&(bs.ravel()==1)&(np.linalg.norm(back-pts,axis=2).ravel()<1.5)
 if sum(good)<4:return rect,0
 u=pts[good].reshape(-1,2);v=p[good].reshape(-1,2);M,inl=cv2.estimateAffinePartial2D(u,v,method=cv2.RANSAC,ransacReprojThreshold=2)
 if M is None:return rect,0
 # Update center plus scale, preserving axis-aligned shape (enough margin for small rotations).
 center=np.array([(x1+x2)/2,(y1+y2)/2,1]);cc=M@center;sc=np.clip(np.hypot(M[0,0],M[0,1]),.92,1.08);w=(x2-x1)*sc;h=(y2-y1)*sc
 return np.array([cc[0]-w/2,cc[1]-h/2,cc[0]+w/2,cc[1]+h/2]),int(sum(good))
tracks={};problems=[]
def run(name,start,end,rect,anchors={}):
 r=np.array(rect,dtype=float)/2;res={};step=2 if end>start else -2;prev=gray[start]
 for n in range(start,end+step,step):
  cur=gray[n]
  if n in anchors:r=np.array(anchors[n],float)/2
  elif n!=start:
   r,quality=flow(prev,cur,r)
   if quality<4:problems.append((name,n,quality))
  res[n]=list(r*2);prev=cur
 tracks[name]=res
anchors=json.load(open(out/'anchors.json'))
a={name:{round(float(t)*60):v for t,v in d.items()} for name,d in anchors.items()}
run('foreground',4260,4818,[890,380,928,426],a['foreground'])
run('background_back',4920,4560,[336,307,399,350],a['background'])
run('background',4920,7078,[336,307,399,350],a['background'])
run('foreground_rev',4818,4260,tracks['foreground'][4818],a['foreground'])
run('background_rev',7078,4920,tracks['background'][7078],a['background'])
json.dump({'tracks':tracks,'problems':problems},open(out/'tracked.json','w'));print('done',len(problems),problems[:30],flush=True)
