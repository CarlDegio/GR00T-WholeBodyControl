from render import *
D=BASE/'applied_preview';D.mkdir(exist_ok=True);frames={}
for name in ['mosaics','preview','t4','t2_chest_faces']:
 for fid,boxes in json.loads((BASE/name/'masks.json').read_text())['frames'].items():frames.setdefault(fid,[]).extend(boxes)
cv2.setNumThreads(2);cap=cv2.VideoCapture(str(SOURCE),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
for t in [0,4,5,6.5,23.55,24,33.95,45,67.95,74.0,78.25,85.85,142.5,151.967,158,168.5,177.567,179.4]:
 fid=round(t*60);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok
 boxes=frames.get(str(fid),[]);out=redact(im.copy(),boxes);cv2.imwrite(str(D/f'frame_{fid:05d}.jpg'),out,[cv2.IMWRITE_JPEG_QUALITY,95])
 if t>=55.3 and t<177.5:
  pair=np.concatenate([im[343:565,1488:1836],out[343:565,1488:1836]],axis=1)
  cv2.imwrite(str(D/f'chest_{fid:05d}.jpg'),cv2.resize(pair,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
cap.release()
print('Wrote',len(list(D.glob('*.jpg'))),'applied preview files')
