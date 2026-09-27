import json,pathlib,numpy as np
out=pathlib.Path('demo_video/2026-09-26_combined/privacy_redaction_v4/t2_codes');d=json.load(open(out/'tracked.json'))['tracks'];a=json.load(open(out/'anchors.json'));combined={}
for name in ['foreground','background','background_back']:
 res={};anchors=sorted(round(float(t)*60) for t in a['foreground' if name=='foreground' else 'background'])
 for k,b in d[name].items():
  n=int(k)
  if name+'_rev' in d:
   prv=max([x for x in anchors if x<=n],default=anchors[0]);nxt=min([x for x in anchors if x>=n],default=anchors[-1]);
   if nxt-n<n-prv:b=d[name+'_rev'][k]
  res[k]=b
 combined[name]=res
json.dump({'tracks':combined},open(out/'chosen.json','w'))
