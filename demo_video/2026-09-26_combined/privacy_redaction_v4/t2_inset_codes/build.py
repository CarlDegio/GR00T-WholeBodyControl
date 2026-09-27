from pathlib import Path
import json
D=Path(__file__).parent
src=json.loads((D.parent/'t2/inset_code_seed.json').read_text());t={k:v for k,v in src.items() if k.startswith(('head_qr','qr_chest'))}
# Exact local boxes from native-resolution frame-by-frame review of sudden pan changes.
def setbox(k,ids,b):
 ox,oy=(1488,64) if k.startswith('head') else (1488,343)
 for i in ids:t[k][str(i)]=[b[0]+ox,b[1]+oy,b[2]+ox,b[3]+oy]
setbox('head_qr_1',[4438,4439],[118,3,176,54])
setbox('head_qr_1',[4450,4451],[143,0,201,35])
setbox('head_qr_1',[4456,4457],[158,0,219,36])
setbox('head_qr_1',[4462,4463],[121,0,178,38])
setbox('head_qr_1',[4480,4481],[10,0,81,57])
setbox('head_qr_1',range(4488,4494),[0,0,45,53])
for i in range(4494,4513):t['head_qr_1'].pop(str(i),None)
setbox('head_qr_flash',[4576,4577],[0,8,44,77])
setbox('head_qr_2',[4610,4611],[0,0,47,74])
t['head_qr_2']['4745']=t['head_qr_2']['4744'][:]
setbox('qr_chest_a',[4378,4379],[160,190,205,222])
setbox('qr_chest_b',[4420,4421],[0,195,43,222])
setbox('qr_chest_b',[4424,4425],[177,187,224,222])
setbox('qr_chest_b',[4426,4427],[248,186,299,222])
for i in range(4562,4566):t['qr_chest_c'][str(i)][0]-=25
# Small 4 px finishing margin within viewport, preserving root's tracker where it fits.
frames={}
for name,vals in t.items():
 top,bottom=(64,285) if name.startswith('head') else (343,565)
 for k,b in vals.items():
  b[:]=[max(1488,b[0]-4),max(top,b[1]-4),min(1836,b[2]+4),min(bottom,b[3]+4)]
  frames.setdefault(k,[]).append(b)
(D/'per_target.json').write_text(json.dumps(t,separators=(',',':')))
result={'frames':dict(sorted(frames.items(),key=lambda q:int(q[0]))),'targets':[{'id':k,'start_frame':min(map(int,v)),'end_frame_exclusive':max(map(int,v))+1} for k,v in t.items()],'notes':['T2 HEAD/CHEST 小窗箱码，基于root最新inset_code_seed.json，逐帧坐标为全局0基、xyxy exclusive。','HEAD首段、76.3秒闪现和CHEST73.0/73.65–73.8突跳区间按原片逐帧核对纠正，另保留4px小余量。','HEAD实际首段方码已于74.9秒离开，移除该时刻以后的空场边缘遮罩；79.083秒末尾边缘残留补1帧。','未扩大到保守候选，无主画面、CHEST脸等其他目标。']}
(D/'masks.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')))
print(len(frames),sum(map(len,frames.values())))
