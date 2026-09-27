from render import *
import hashlib
D=BASE/'parts';ranges=[('prefix',0,3318),('t2',3318,7080),('tail',7080,10774)]
checks=[]
for name,a,b in ranges:
 p=D/f'{name}.mp4';j=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)]));s=j['streams'][0]
 assert int(s['nb_frames'])==b-a,(name,s['nb_frames'],b-a)
 assert (s['width'],s['height'],s['r_frame_rate'])==(1920,1080,'60/1')
 checks.append(dict(part=name,frames=int(s['nb_frames']),duration=j['format']['duration']))
concat=D/'concat.txt';concat.write_text(''.join("file '"+str(D/(name+'.mp4'))+"'\n" for name,_,_ in ranges))
assert not OUTPUT.exists()
subprocess.run(['ffmpeg','-hide_banner','-v','warning','-nostdin','-n','-f','concat','-safe','0','-i',str(concat),'-map','0:v:0','-c','copy','-an','-map_metadata','-1','-movflags','+faststart',str(OUTPUT)],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(OUTPUT)]));s=probe['streams'][0]
assert int(s['nb_frames'])==N
assert (s['width'],s['height'],s['r_frame_rate'])==(1920,1080,'60/1')
assert abs(float(probe['format']['duration'])-N/60)<.002
with (BASE/'full_decode.log').open('w') as log:
 subprocess.run(['ffmpeg','-v','error','-xerror','-threads','4','-i',str(OUTPUT),'-map','0:v:0','-f','null','-'],stderr=log,check=True)
frames={}
for name,_,_ in ranges:
 d=json.loads((D/f'{name}_masks.json').read_text())
 for k,boxes in d['frames'].items():frames.setdefault(k,[]).extend(boxes)
merged=dict(source=str(SOURCE),output=str(OUTPUT),frames=frames,parts=checks)
(BASE/'merged_masks.json').write_text(json.dumps(merged,separators=(',',':')))
cv2.setNumThreads(2)
src=cv2.VideoCapture(str(SOURCE),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);out=cv2.VideoCapture(str(OUTPUT),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
# Include every segment boundary, clear targets, quick flashes, box turns and the final frame.
sample_times=[0,1.5,4.98,5,6.5,12.45,14.375,21.68,23.533333,24,33.95,43.125,45.283333,45.3,55.283333,55.3,61.95,62.15,66.78,67.75,68.05,69.95,72.45,72.65,73.0,74.0,74.9,76.3,78.8,79.3,80.3,85.85,87,92.3,108.3,112.3,117.983333,118,141.8,151.967,154.767,156.1,158,168.5,175.0,177.55,177.566667,179.4,(N-1)/60]
samples=[];shots=BASE/'final_check';shots.mkdir(exist_ok=True)
for t in sample_times:
 fid=round(t*60);src.set(cv2.CAP_PROP_POS_FRAMES,fid);out.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,a=src.read();ok2,b=out.read();assert ok and ok2
 boxes=frames.get(str(fid),[]);mask=np.zeros((H,W),np.uint8)
 for x1,y1,x2,y2 in boxes:mask[y1:y2,x1:x2]=1
 expected=redact(a.copy(),boxes)
 err=np.abs(expected.astype(np.int16)-b.astype(np.int16)).mean(axis=2)
 outside=float(err[mask==0].mean());inside=float(err[mask==1].mean()) if mask.any() else None
 assert outside<6,(fid,'outside frame differs',outside)
 if inside is not None:assert inside<10,(fid,'redaction mismatch',inside)
 samples.append(dict(frame=fid,seconds=fid/60,rectangles=len(boxes),expected_vs_encoded_outside_mae=outside,expected_vs_encoded_mask_mae=inside))
 cv2.imwrite(str(shots/f'{fid:05d}.jpg'),b,[cv2.IMWRITE_JPEG_QUALITY,93])
src.release();out.release()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
source_sha=sha(SOURCE);assert source_sha=='b8f7e123c329ef6694f36b02d2a1d7d8ecfeeecc95b2aefc5bf3f96a2b89cb52'
receipt=dict(output=str(OUTPUT),source_sha256=source_sha,output_sha256=sha(OUTPUT),frame_count=N,fps=60,resolution=[1920,1080],duration_seconds=N/60,bytes=OUTPUT.stat().st_size,full_decode='passed',timeline_parts=checks,sample_checks=samples,nonempty_mask_frames=sum(bool(b) for b in frames.values()),style='40px coarse mosaic inside tracked local masks; existing outro lettering preserved')
(BASE/'verification.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='sample_checks'},indent=2),flush=True)
