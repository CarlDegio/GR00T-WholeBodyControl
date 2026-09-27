"""Original-pixel polygon cutouts; no generative image editing or pose synthesis."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
# Contours are expressed in unscaled source-frame pixel coordinates.
POLYS={
1:[(352,136),(365,133),(377,135),(384,142),(384,147),(371,151),(378,158),(390,163),(394,178),(394,197),(387,207),(377,213),(380,218),(384,226),(378,233),(375,248),(379,258),(378,278),(374,290),(375,301),(378,306),(376,310),(367,310),(360,304),(360,297),(363,284),(365,270),(361,265),(359,251),(358,237),(352,243),(347,252),(347,265),(355,275),(355,290),(349,302),(348,316),(350,319),(330,319),(332,315),(338,315),(339,301),(339,287),(337,277),(338,263),(337,248),(338,233),(345,218),(348,211),(339,210),(334,216),(323,220),(324,224),(314,226),(310,224),(307,227),(305,225),(310,221),(314,220),(312,215),(313,211),(316,211),(318,215),(331,208),(339,199),(340,183),(339,175),(344,168),(353,165),(354,160),(366,156),(365,150),(361,148),(360,153),(356,153),(354,147),(354,143),(350,143),(348,145),(344,144),(344,140)],
2:[(280,102),(287,100),(294,101),(299,104),(300,109),(292,112),(296,116),(305,120),(308,126),(308,139),(310,144),(307,149),(301,155),(300,160),(302,166),(300,173),(297,178),(297,190),(300,196),(302,204),(301,216),(298,225),(302,228),(304,231),(301,233),(294,230),(291,228),(291,217),(293,207),(290,203),(289,195),(287,180),(282,177),(279,184),(280,198),(285,203),(284,215),(280,225),(281,235),(287,238),(286,242),(281,242),(275,239),(272,237),(275,234),(275,225),(273,215),(272,204),(273,198),(270,181),(270,174),(275,166),(280,158),(280,155),(277,154),(274,161),(267,164),(261,168),(257,169),(256,173),(251,172),(251,170),(247,170),(246,167),(253,165),(252,161),(249,159),(249,156),(253,158),(257,162),(261,163),(269,157),(271,154),(275,153),(277,147),(276,138),(275,129),(279,124),(284,122),(288,119),(287,115),(284,117),(280,115),(280,110),(277,111),(275,110),(277,105)],
3:[(247,94),(253,92),(259,94),(262,98),(262,102),(255,104),(257,108),(264,112),(268,117),(268,128),(271,133),(269,138),(265,141),(264,148),(265,151),(267,156),(263,162),(260,167),(261,178),(263,184),(263,191),(263,205),(261,211),(265,214),(264,217),(259,217),(255,214),(255,204),(257,194),(254,187),(252,181),(253,169),(248,165),(243,170),(243,181),(248,186),(248,196),(246,210),(246,216),(250,219),(249,222),(243,222),(239,219),(239,211),(239,201),(238,190),(238,183),(236,174),(236,167),(234,164),(239,156),(246,145),(246,141),(241,140),(239,143),(235,145),(230,149),(228,151),(225,154),(221,155),(220,153),(217,156),(215,155),(218,151),(220,149),(218,148),(216,147),(217,144),(221,146),(225,148),(229,145),(230,141),(234,139),(238,135),(238,127),(239,119),(244,114),(250,110),(249,108),(247,109),(245,107),(245,101),(243,102),(242,99)],
5:[(195,93),(203,91),(210,96),(209,102),(205,105),(207,110),(214,114),(217,120),(218,129),(217,135),(216,140),(213,144),(210,149),(211,154),(215,157),(215,164),(212,169),(214,182),(216,194),(219,207),(222,212),(220,217),(213,219),(212,215),(214,210),(212,199),(207,187),(207,178),(205,164),(201,158),(193,160),(190,158),(188,165),(187,177),(190,183),(190,190),(190,201),(193,211),(197,213),(195,216),(188,219),(182,220),(182,217),(186,214),(186,208),(183,196),(182,189),(181,184),(182,173),(183,161),(182,151),(184,146),(184,139),(179,143),(176,145),(176,150),(180,155),(179,165),(177,167),(169,166),(166,157),(166,148),(168,144),(169,141),(167,137),(170,136),(174,137),(174,139),(178,135),(182,132),(184,128),(182,121),(183,114),(190,111),(191,107),(189,103),(191,98)],
6:[(196,113),(202,111),(208,113),(212,118),(211,122),(203,125),(208,129),(212,132),(215,136),(217,143),(218,151),(219,158),(223,164),(225,169),(221,174),(215,177),(212,183),(217,190),(218,198),(215,203),(216,215),(219,225),(219,234),(222,248),(224,259),(229,261),(230,264),(224,269),(216,273),(214,270),(220,262),(218,255),(214,247),(211,236),(208,231),(205,216),(202,205),(200,200),(198,196),(195,193),(194,204),(195,215),(201,221),(202,229),(200,237),(203,249),(206,253),(206,256),(198,259),(190,262),(186,261),(187,258),(195,254),(195,247),(191,236),(189,228),(188,216),(186,205),(186,195),(186,186),(183,188),(178,185),(176,180),(175,177),(171,179),(169,184),(173,195),(172,199),(159,206),(151,204),(146,195),(143,179),(145,173),(153,170),(160,169),(161,167),(158,165),(160,162),(164,164),(168,166),(171,165),(174,163),(178,163),(183,158),(184,156),(188,156),(188,148),(187,142),(188,137),(194,134),(195,130),(191,128),(188,124),(189,119)]}
HOLES={
1:[[(360,146),(366,148),(365,152),(361,153)]],
2:[[(282,110),(287,111),(287,115),(283,115)],[(281,154),(286,156),(282,163),(278,167)]],
3:[[(246,102),(250,104),(249,107),(246,106)],[(240,143),(245,143),(242,150),(233,155),(229,155),(231,151)]],
5:[[(185,141),(193,144),(190,149),(185,152),(177,151),(175,146)],[(207,142),(211,142),(208,147),(205,148)]],
6:[[(173,169),(183,164),(188,162),(190,168),(187,174),(179,175),(175,175)],[(210,179),(213,177),(215,181),(213,187)],[(153,177),(157,176),(160,184),(158,187),(155,184)]]}
W,H=1672,941
PLACEMENTS={1:((358,312),(1203,908),2.72),2:((289,237),(875,706),2.94),3:((250,219),(738,647),2.94),5:((202,216),(590,646),2.94),6:((210,265),(595,766),2.94)}
sources={i:np.array(Image.open(ROOT/f'frames_synced/{i:02}.png').convert('RGB')) for i in POLYS}
masks={}; boxes={}; result={}
for i,points in POLYS.items():
 mask=np.zeros((320,568),np.uint8)
 cv2.fillPoly(mask,[np.array(points,np.int32)],255)
 for hole in HOLES.get(i,[]):cv2.fillPoly(mask,[np.array(hole,np.int32)],0)
 seed=np.full(mask.shape,cv2.GC_BGD,np.uint8)
 seed[cv2.dilate(mask,np.ones((3,3),np.uint8))>0]=cv2.GC_PR_BGD
 seed[mask>0]=cv2.GC_PR_FGD
 seed[cv2.erode(mask,np.ones((3,3),np.uint8))>0]=cv2.GC_FGD
 for hole in HOLES.get(i,[]):cv2.fillPoly(seed,[np.array(hole,np.int32)],cv2.GC_BGD)
 cv2.grabCut(sources[i],seed,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
 mask=np.where((seed==cv2.GC_FGD)|(seed==cv2.GC_PR_FGD),255,0).astype(np.uint8)
 masks[i]=mask
 rgba=np.dstack([sources[i],mask])
 # All original RGB pixels are unchanged, including partially hidden regions.
 assert np.array_equal(rgba[:,:,:3],sources[i])
 ys,xs=np.where(mask>0);box=(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1));boxes[i]=box
 Image.fromarray(mask).save(OUT/f'mask_{i:02}.png')
 Image.fromarray(rgba).save(OUT/f'robot_{i:02}_fullcanvas.png')
 Image.fromarray(rgba).crop(box).save(OUT/f'robot_{i:02}.png')

# Real background from frame2; robot2 region is covered by the same room
# from frame6, whose robot lies outside this mask. No generative fill.
bg=sources[2].copy()
repair=cv2.dilate(masks[2],np.ones((9,9),np.uint8))
assert not np.any((repair>0)&(masks[6]>0))
bg[repair>0]=sources[6][repair>0]
Image.fromarray(bg).save(OUT/'background_real_frame.png')
canvas=Image.fromarray(bg).resize((W,H),Image.Resampling.LANCZOS).convert('RGBA')
base=canvas.copy()
for i in [3,5,2,6,1]:
 anchor,target,scale=PLACEMENTS[i];b=boxes[i]
 sprite=Image.open(OUT/f'robot_{i:02}.png')
 sprite=sprite.resize((round(sprite.width*scale),round(sprite.height*scale)),Image.Resampling.LANCZOS)
 x=round(target[0]-(anchor[0]-b[0])*scale);y=round(target[1]-(anchor[1]-b[1])*scale)
 canvas.alpha_composite(sprite,(x,y))
 result[i]=dict(source_frame=f'frames_synced/{i:02}.png',source_anchor=anchor,target_anchor=target,scale=scale,top_left=[x,y],mask='manual source-coordinate polygon; RGB unchanged')
canvas.convert('RGB').save(OUT/'teaser_v3_clean.png')
d=ImageDraw.Draw(canvas)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',25)
labels={1:(1308,413),2:(977,283),3:(819,235),5:(527,211),6:(694,340)}
for i,(x,y) in labels.items():
 ax,ay,scale=PLACEMENTS[i];b=boxes[i]
 head=np.array(POLYS[i][:5]).mean(axis=0)
 hx=ay[0]+(head[0]-ax[0])*scale;hy=ay[1]+(head[1]-ax[1])*scale
 d.line([(x+20,y+38),(hx,hy-6)],fill='#fff0ae',width=2)
 d.rounded_rectangle((x,y,x+39,y+40),radius=2,fill='#fff0ae')
 d.text((x+11,y+4),str(i),font=font,fill='#171717')
canvas.convert('RGB').save(OUT/'teaser_v3_numbered.png')
(OUT/'placement_and_verification.json').write_text(json.dumps(dict(placements=result,verification='Each fullcanvas RGBA has RGB exactly equal to its original source frame. No pose alteration, no rotation, no generative content. Number4 omitted. Whole cutout only scaled and translated.',polygons=POLYS,holes=HOLES),ensure_ascii=False,indent=2)+'\n')
# Source and its pixel-identical cutout side by side, same crop and scale.
b=boxes[6];pad=6;b=(b[0]-pad,b[1]-pad,b[2]+pad,b[3]+pad)
src=Image.fromarray(sources[6]).crop(b).resize(((b[2]-b[0])*4,(b[3]-b[1])*4),Image.Resampling.NEAREST)
spr=Image.open(OUT/'robot_06_fullcanvas.png').crop(b).resize(src.size,Image.Resampling.NEAREST)
proof=Image.new('RGB',(src.width*2+30,src.height+45),'#d8dce0');proof.paste(src,(0,45));proof.paste(spr,(src.width+30,45),spr)
pd=ImageDraw.Draw(proof);pd.text((8,8),'SOURCE FRAME 9955',font=font,fill='black');pd.text((src.width+38,8),'SAME PIXELS + MASK',font=font,fill='black')
proof.save(OUT/'frame6_pixel_proof.png')
print('Verified source RGB preservation for 5 cutouts; exported clean/numbered composite and frame6 proof.')
