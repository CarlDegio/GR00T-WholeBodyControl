"""1080p original-pixel composite with fixed user placements and alpha ghosts."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter

P=Path(__file__).resolve().parent
OLD=P.parent/'v3_pixel_cutouts'
cfg=json.loads((OLD/'placement_and_verification.json').read_text())
sel=json.loads((P/'selection.json').read_text())
W,H=1920,1080
sx,sy=W/568,H/320
outx,outy=W/1672,H/941
ALPHA=.55
reference=np.array(Image.open(P/'background_0551.png').convert('RGB'))
sift=cv2.SIFT_create(nfeatures=5000)
refvalid=np.full((H,W),255,np.uint8);refvalid[420:,:510]=0
kr,dr=sift.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY),refvalid)
# Narrow definite-foreground strokes preserve silver limbs and dark feet.
# Coordinates refer to the old frame system solely to initialize segmentation.
STROKES={
1:[([(353,142),(365,140),(376,143)],3), ([(365,170),(367,200)],8), ([(349,177),(345,199),(339,207),(322,216),(314,222)],3), ([(351,229),(344,249),(345,265),(348,278),(344,299),(344,317)],4), ([(365,239),(368,255),(372,270),(370,287),(369,303),(374,307)],4)],
2:[([(282,107),(289,105),(295,107)],3), ([(292,128),(294,149)],8), ([(279,130),(279,149),(273,156),(266,161),(256,167)],3), ([(278,174),(276,195),(280,205),(278,225),(278,237),(283,239)],3), ([(292,175),(293,193),(297,203),(296,224),(299,230)],3)],
3:[([(248,99),(253,97),(258,100)],3), ([(253,118),(254,138)],7), ([(243,121),(242,136),(235,143),(226,149),(220,151)],3), ([(242,165),(241,180),(244,191),(243,210),(242,219),(247,220)],3), ([(256,165),(257,180),(259,190),(259,211),(262,215)],3)],
5:[([(197,98),(204,99)],4), ([(194,120),(205,122),(201,138)],7), ([(189,118),(189,132),(180,138),(172,142)],3), ([(213,121),(214,133),(209,140),(200,147)],3), ([(190,158),(185,179),(186,190),(187,210),(190,214),(185,218)],3), ([(207,160),(210,183),(212,198),(217,211),(216,216)],3), ([(170,153),(172,161),(177,162)],4)],
6:[([(198,116),(205,116),(205,121)],4), ([(199,144),(202,165)],8), ([(212,143),(215,159),(218,168),(210,173),(188,180)],3), ([(189,160),(180,166),(171,168),(163,171)],3), ([(191,192),(191,211),(196,225),(196,239),(200,252),(190,259)],4), ([(208,195),(210,215),(214,232),(218,249),(223,262),(218,269)],4), ([(154,179),(158,197),(168,194)],7)]}

verification={};layers={};heads={};manifest={}
for i in [1,2,3,5,6]:
 source=P/f'source_frames/{i:02}.png'
 rgb=np.array(Image.open(source).convert('RGB'))
 assert rgb.shape==(H,W,3)
 poly=np.array(cfg['polygons'][str(i)],float)*[sx,sy]
 rough=np.zeros((H,W),np.uint8);cv2.fillPoly(rough,[np.rint(poly).astype(np.int32)],255)
 holes=[np.rint(np.array(h,float)*[sx,sy]).astype(np.int32) for h in cfg['holes'].get(str(i),[])]
 for hole in holes:cv2.fillPoly(rough,[hole],0)
 yy,xx=np.where(rough>0)
 x0=max(0,int(xx.min())-16);x1=min(W,int(xx.max())+17);y0=max(0,int(yy.min())-16);y1=min(H,int(yy.max())+17)
 roi=rough[y0:y1,x0:x1]
 seed=np.full(roi.shape,cv2.GC_BGD,np.uint8)
 seed[cv2.dilate(roi,np.ones((15,15),np.uint8))>0]=cv2.GC_PR_BGD
 seed[roi>0]=cv2.GC_PR_FGD
 seed[cv2.erode(roi,np.ones((15,15),np.uint8))>0]=cv2.GC_FGD
 for coords,width in STROKES[i]:
  pts=np.rint(np.array(coords,float)*[sx,sy]-[x0,y0]).astype(np.int32)
  cv2.polylines(seed,[pts],False,cv2.GC_FGD,round(width*sx))
 # The fixed room provides real background evidence to remove carpet/wall
 # fragments retained by coarse old low-resolution contour initialization.
 valid=np.full((H,W),255,np.uint8);valid[y0:y1,x0:x1]=0
 ks,ds=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),valid)
 matches=cv2.BFMatcher().knnMatch(dr,ds,k=2)
 good=[m for m,n in matches if m.distance<.7*n.distance]
 transform,inliers=cv2.findHomography(np.float32([kr[m.queryIdx].pt for m in good]),np.float32([ks[m.trainIdx].pt for m in good]),cv2.RANSAC,2)
 aligned=cv2.warpPerspective(reference,transform,(W,H))
 delta=np.max(np.abs(cv2.GaussianBlur(rgb,(3,3),0).astype(float)-cv2.GaussianBlur(aligned,(3,3),0).astype(float)),axis=2)
 same=cv2.erode((delta<12).astype(np.uint8),np.ones((3,3),np.uint8))
 seed[same[y0:y1,x0:x1]>0]=cv2.GC_BGD
 cv2.grabCut(rgb[y0:y1,x0:x1],seed,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
 mask=np.zeros((H,W),np.uint8)
 mask[y0:y1,x0:x1]=np.where((seed==cv2.GC_FGD)|(seed==cv2.GC_PR_FGD),255,0).astype(np.uint8)
 if i==6:
  # Native-resolution tracing of the rear ankle/sole: remove carpet mistakenly
  # retained by the old 568px polygon, while retaining the actual shoe pixels.
  keep=np.full((H,W),255,np.uint8);keep[820:,620:720]=0
  foot=np.array([(665,817),(696,817),(700,833),(704,842),(712,846),(719,854),(719,863),(714,868),(650,885),(638,883),(635,878),(639,872),(676,854),(679,850),(674,835)],np.int32)
  cv2.fillPoly(keep,[foot],255);mask[keep==0]=0
 # Keep fractional coverage confined to the edge; source RGB is never retouched.
 alpha=np.array(Image.fromarray(mask).filter(ImageFilter.GaussianBlur(.45)))
 rgba=np.dstack([rgb,alpha]);assert np.array_equal(rgba[:,:,:3],rgb)
 Image.fromarray(rgba).save(P/f'robot_{i:02}_fullcanvas.png')
 Image.fromarray(mask).save(P/f'mask_{i:02}.png')
 ys,xs=np.where(alpha>0);box=(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1))
 sprite=Image.fromarray(rgba).crop(box);sprite.save(P/f'robot_{i:02}.png')
 old=cfg['placements'][str(i)]
 anchor=np.array(old['source_anchor'])*[sx,sy]
 target=np.array(old['target_anchor'])*[outx,outy]
 # Preserve prior displayed dimensions, scaled only with the output canvas.
 scale=old['scale']*outx/sx
 size=(round(sprite.width*scale),round(sprite.height*scale))
 sprite=sprite.resize(size,Image.Resampling.LANCZOS)
 a=sprite.getchannel('A').point(lambda v:round(v*ALPHA));sprite.putalpha(a)
 pos=tuple(np.rint(target-(anchor-np.array(box[:2]))*scale).astype(int))
 layer=Image.new('RGBA',(W,H));layer.alpha_composite(sprite,pos);layers[i]=layer
 head=(np.array(cfg['polygons'][str(i)][:5],float)*[sx,sy]).mean(axis=0)
 heads[i]=target+(head-anchor)*scale
 verification[i]=dict(source_rgb_exact_match=True,foreground_pixels=int((mask>0).sum()))
 manifest[i]=dict(source=str(source.name),source_anchor=anchor.tolist(),target_anchor=target.tolist(),scale=float(scale),opacity=ALPHA,top_left=list(map(int,pos)))

bg=Image.open(P/'background_0551.png').convert('RGBA');assert bg.size==(W,H)
canvas=bg.copy()
for i in [3,5,2,6,1]:canvas.alpha_composite(layers[i])
canvas.convert('RGB').save(P/'teaser_v5_clean.png')
draw=ImageDraw.Draw(canvas)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',29)
labels={1:(1308,413),2:(977,283),3:(819,235),5:(527,211),6:(694,340)}
for i,(x,y) in labels.items():
 x=round(x*outx);y=round(y*outy);hx,hy=heads[i]
 draw.line([(x+22,y+42),(hx,hy-6)],fill='#fff0ae',width=2)
 draw.rounded_rectangle((x,y,x+45,y+46),radius=2,fill='#fff0ae')
 draw.text((x+12,y+4),str(i),font=font,fill='#171717')
canvas.convert('RGB').save(P/'teaser_v5_numbered.png')
(P/'verification.json').write_text(json.dumps(dict(cutouts=verification,placements=manifest,background_pts=sel['background'],excluded=[4],output_size=[W,H],method='Raw source RGB + segmentation alpha + whole-cutout resize/translation. No generative image synthesis.'),indent=2))
print('Saved 1080p composite, alpha=0.55, unchanged normalized placements. Source RGB verified.')
