import json,pathlib,math,numpy as np
out=pathlib.Path('demo_video/2026-09-26_combined/privacy_redaction_v4/t2_codes');d=json.load(open(out/'chosen.json'))['tracks'];fixf={4352:[475,565,529,626],4354:[345,575,405,648],4360:[19,655,106,740],4362:[0,670,37,741],4364:[0,668,50,739],4366:[0,668,45,739],4368:[0,668,43,740],4370:[3,665,85,741]}
for n,b in fixf.items():d['foreground'][str(n)]=b
fixb={5238:[0,249,77,297],6908:[1280,878,1440,972],6910:[1300,930,1430,1020],6916:[967,855,1015,1015],6918:[974,863,1020,1022]}
for n,b in fixb.items():d['background'][str(n)]=b
# The code panel briefly leaves the main picture during camera pans.
for n in [5164,5166,5168,5170,5224,5226,5228,5230,5232,5234,5236,6912,6914]:d['background'].pop(str(n),None)
anchors={6390:[791,458,852,564],6394:[831,456,888,553],6398:[901,446,963,553],6402:[892,452,960,564],6406:[916,450,987,555],6410:[920,448,985,555]}
ks=sorted(anchors)
for n in range(6390,6412,2):
 l=max(k for k in ks if k<=n);r=min(k for k in ks if k>=n);b=np.array(anchors[l]) if l==r else np.array(anchors[l])+(np.array(anchors[r])-np.array(anchors[l]))*(n-l)/(r-l);d['background'][str(n)]=list(map(float,b))
frames={}
for name,data in d.items():
 for k,b in data.items():
  n=int(k);x1,y1,x2,y2=b
  # Perspective compresses the two labels diagonally; extend the upper/left edge.
  if name=='background' and 96*60<=n<114.65*60 and n not in range(6390,6412):x1-=15;y1-=42
  if name=='background' and 115.333*60<=n<118*60:x1-=12;y1-=32
  margin=20
  box=[max(0,math.floor(x1-margin)),max(0,math.floor(y1-margin)),min(1428,math.ceil(x2+margin)),min(1080,math.ceil(y2+margin))]
  if box[2]<=box[0] or box[3]<=box[1]:continue
  for f in [n,n+1]:
   if box not in frames.setdefault(str(f),[]):frames[str(f)].append(box)
obj={'frames':frames,'targets':[{'type':'QR/code labels','position':'T2 main picture','time_range':[71,118],'description':'Two square labels together on the office foreground box (71–80.3s) and transported box (76–118s). Dense square code and adjacent circular-label square are covered.'}],'notes':['0-based global frame indices, full-screen integer xyxy exclusive. Main picture only; parent handles small-camera windows.','Manual anchors and local bidirectional optical flow on each actual-changing frame (30fps source duplicated into60fps). 323 crop checks, with every changing frame reviewed in rapid-pan problem intervals; remaining samples every 0.267s.','Explicit manual fixes for pan failures at72.53–72.83s,87.30s,106.5–106.83s and115.13–115.30s. Screenspace margins20px plus perspective-specific upper/left extension.','Brief code-offscreen intervals omitted. No decoding or identity inference performed.']}
json.dump(obj,open(out/'masks.json','w'),indent=2)
json.dump({'tracks':d},open(out/'final_tracks.json','w'))
print('frames',len(frames),'rects',sum(map(len,frames.values())),'range',min(map(int,frames)),max(map(int,frames)))
for n,bs in frames.items():
 assert 4260<=int(n)<7080
 for b in bs:assert all(type(v)==int for v in b) and 0<=b[0]<b[2]<=1428 and 0<=b[1]<b[3]<=1080
