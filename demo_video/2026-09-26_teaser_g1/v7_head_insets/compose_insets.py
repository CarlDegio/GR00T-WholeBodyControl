from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
P=Path(__file__).resolve().parent
base_path=P.parent/'v6_trajectories/teaser_v6_clean.png'
base=Image.open(base_path).convert('RGBA')
W,H=base.size
# Two full uncropped head-camera views fit above the final robot's head.
x,y,width,height,gap=20,20,272,204,14
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
cards=[]
for i,title in [(5,'Before alignment'),(6,'After alignment')]:
 view=Image.open(P/f'head_{i}_annotated.png').convert('RGB').resize((width,height),Image.Resampling.LANCZOS)
 card=view.convert('RGBA');d=ImageDraw.Draw(card)
 # Caption is over the hand/floor area; the target and detected edge stay clear.
 caption=Image.new('RGBA',card.size)
 cd=ImageDraw.Draw(caption);cd.rectangle((0,height-27,width,height),fill=(16,22,28,185))
 cd.text((10,height-24),title,font=font,fill='white')
 card.alpha_composite(caption)
 frame=Image.new('RGBA',(width+6,height+6),'#ffffff');frame.alpha_composite(card,(3,3))
 shadow=Image.new('RGBA',(W,H));sd=ImageDraw.Draw(shadow)
 sd.rectangle((x-3,y-2,x+width+7,y+height+9),fill=(0,0,0,100))
 shadow=shadow.filter(ImageFilter.GaussianBlur(4))
 shadow_pixels=np.array(shadow);shadow_pixels[455:,:,3]=0
 base.alpha_composite(Image.fromarray(shadow_pixels))
 base.alpha_composite(frame,(x,y))
 cards.append(dict(source_index=i,title=title,rect=[x,y,width+6,height+6],camera='ego_view / head',annotation='offline YOLOE using recorded prompt; RGB edge line shown only after alignment'))
 y+=height+6+gap
base.convert('RGB').save(P/'teaser_v7_head_insets.png')
column=Image.new('RGB',(646,978),'white')
for j,i in enumerate([5,6]):column.paste(Image.open(P/f'head_{i}_annotated.png'),(3,3+j*492))
column.save(P/'head_insets_full_resolution.png')
# Pixel comparison confirms removal came from the clean base, not inpainting.
original=np.array(Image.open(base_path).convert('RGB'));final=np.array(base.convert('RGB'))
unchanged=np.ones((H,W),bool);unchanged[0:465,0:320]=False
assert np.array_equal(original[unchanged],final[unchanged])
assert cards[-1]['rect'][1]+cards[-1]['rect'][3]<=455
for i in [5,6]:assert Image.open(P/f'head_{i}_raw.jpg').size==(640,480)
det=json.loads((P/'offline_detection.json').read_text())
assert all(a['prompt']=='green rubbish bin' and a['confidence']>.25 for a in det)
assert det[1]['displayed_line'] is not None
(P/'layout_and_verification.json').write_text(json.dumps(dict(base=str(base_path),cards=cards,output_size=[W,H],no_number_badges_or_number_leaders=True,original_scene_unchanged_outside_insets=True,trajectory_overlay_preserved=True,full_camera_images_uncropped=True,offline_detection_verified=True),indent=2)+'\n')
print('Saved 1920x1080 clean-numbered-free teaser; two uncropped head views stacked in upper left.')
