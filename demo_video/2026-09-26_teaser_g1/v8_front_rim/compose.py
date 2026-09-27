from pathlib import Path
import json, shutil
import numpy as np
from PIL import Image
P=Path(__file__).resolve().parent
old=P.parent/'v7_head_insets'
selected=next(r for r in json.loads((P/'candidates.json').read_text()) if r['delta']==3)
line=selected['hough_candidates'][selected['selected_candidate']]
for name in ['head_5_raw.jpg','head_5_annotated.png']:shutil.copyfile(old/name,P/name)
shutil.copyfile(P/'head6_+03.jpg',P/'head_6_raw.jpg')
shutil.copyfile(P/'head6_+03_annotated.png',P/'head_6_annotated.png')
det=json.loads((old/'offline_detection.json').read_text())
det[1].update(source='head_6_raw.jpg',confidence=selected['confidence'],bbox_xyxy=selected['bbox_xyxy'],hough_candidates=selected['hough_candidates'],displayed_line=line,line_selection='Detected near-horizontal front lip within 34–54% of bin bbox height; this frame also ranks it first by length.')
(P/'offline_detection.json').write_text(json.dumps(det,indent=2)+'\n')
code=(old/'compose_insets.py').read_text().replace('teaser_v7_head_insets.png','teaser_v8_front_rim.png')
exec(compile(code,str(P/'compose.py'),'exec'))
a=np.array(Image.open(old/'teaser_v7_head_insets.png'))
b=np.array(Image.open(P/'teaser_v8_front_rim.png'))
changed=np.any(a!=b,axis=2)
y,x=np.where(changed)
assert x.min()>=23 and x.max()<295 and y.min()>=247 and y.max()<451
assert np.array_equal(np.array(Image.open(old/'head_5_annotated.png')),np.array(Image.open(P/'head_5_annotated.png')))
base=json.loads((old/'head_frame_selection.json').read_text())[1]['frame_index_entry']['source_timestamp_ns']
selected.update(time_shift_seconds=(selected['frame_index_entry']['source_timestamp_ns']-base)/1e9,displayed_line=line,changed_pixels_bbox=[int(x.min()),int(y.min()),int(x.max()),int(y.max())],only_second_inset_changed=True)
(P/'selected_frame_and_verification.json').write_text(json.dumps(selected,indent=2)+'\n')
(P/'README.md').write_text(f'''# Teaser: front rim correction\n\nCompared the original head frame plus 8 neighboring recorded frames on each side (17 total). Selected 3 frames later: sequence {selected['frame_index_entry']['sequence']}, source-video time {selected['video_seconds']:.6f} s, offset +{selected['time_shift_seconds']:.6f} s from V7.\n\nSame offline YOLOE prompt `green rubbish bin` and existing RGB/mask Hough detector. The displayed segment {line['endpoint_a']} → {line['endpoint_b']} is a detected line on the front lip, not manually invented coordinates. It is also the longest candidate in this frame. Only the second inset changed; exact pixel comparison confirms the rest is identical to V7. No experiment annotations changed.\n''')
print(json.dumps(selected,indent=2))
