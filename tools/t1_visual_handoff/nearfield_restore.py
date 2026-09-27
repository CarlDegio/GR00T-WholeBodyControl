"""Approximate inverse of the recorded diagnostic renderer, with synthetic controls.

This is numerical preprocessing of scientific data, not original-frame recovery.
Original JPEGs are never modified. Restored images must remain labelled as such.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from PIL import Image, ImageDraw

from nearfield_t2_20260921 import OUT, read, write_csv, CHECKPOINT
from prepare import sha, write_json
from metrics import neighbor_distances, unit


def rgb_read(p):
    im=cv2.imread(str(p));assert im is not None
    return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)


def write_image(p,a):
    p.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(a).save(p)


def mask_components(m):
    n,labels,stats,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),8)
    return np.isin(labels,[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=300])


def recover(rgb, threshold=55, fill_lines=True):
    x=rgb.astype(np.float32);r,g,b=np.moveaxis(x,-1,0)
    purple=(r-g>threshold)&(b-g>threshold)&(r>100)&(b>100)&(g<160)
    mask=mask_components(purple)
    # Diagnostic edges may divide the same mask into separate components.
    mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)
    neighborhood=cv2.dilate(mask.astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool)
    green=(g>150)&(g-r>90)&(g-b>80)
    # One-pixel yellow lines are antialiased against magenta and JPEG chroma
    # subsampling desaturates them; require a green surplus, not pure yellow.
    yellow=(r>150)&(g>125)&(r-b>30)&(g-b>0)&(r-g>-20)
    lines=(green|yellow)&neighborhood
    # Fill only small holes created by overdrawn lines; do not fill large scene holes.
    filled=binary_fill_holes(mask)
    holes=filled&~mask
    n,labels,stats,_=cv2.connectedComponentsWithStats(holes.astype(np.uint8),8)
    mask|=np.isin(labels,[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]<1000])
    repaired=x.copy()
    # Renderer: uint8(0.55*RGB + 0.45*[255,0,255]); +0.5 is quantization midpoint.
    repaired[mask]=(x[mask]+.5-.45*np.array([255,0,255]))/.55
    repaired=np.clip(np.rint(repaired),0,255).astype(np.uint8)
    line_halo=cv2.dilate(lines.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    if fill_lines and np.any(line_halo):
        repaired=cv2.inpaint(repaired,line_halo.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
    return repaired,dict(mask=mask,lines=lines,line_halo=line_halo,green=green&lines,yellow=yellow&lines)


def synthetic(rgb,masks):
    x=rgb.copy();m=masks['mask']
    x[m]=np.clip(.55*x[m].astype(np.float32)+.45*np.array([255,0,255]),0,255).astype(np.uint8)
    x[masks['yellow']]=[255,215,0];x[masks['green']]=[0,255,0]
    ok,buf=cv2.imencode('.jpg',cv2.cvtColor(x,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
    return cv2.cvtColor(cv2.imdecode(buf,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)


def prepare():
    trials=read(OUT/'inputs/trials.json');samples=read(OUT/'inputs/samples.json')
    # Idempotent: retain the clean post input list, then append restoration variants.
    samples=[s for s in samples if s['stage']=='post']
    references=read(OUT/'references/samples.json');controls=[];pairs=[];restorations=[]
    def sample(sid,trial,stage,p):
        samples.append(dict(sample_id=sid,trial_id=trial,stage=stage,split='experiment',view='head',
                            image=str(p),image_sha256=sha(p)))
    for t in trials:
        if not t.get('pre_diagnostic'):
            continue
        tid=t['trial_id'];raw=rgb_read(t['pre_diagnostic']);primary,masks=recover(raw)
        p=OUT/'inputs/reconstructed_pre'/f'{tid}.png';write_image(p,primary)
        np.savez_compressed(OUT/'inputs/reconstructed_pre'/f'{tid}_masks.npz',**masks)
        record=dict(trial_id=tid,image=str(p),source=t['pre_diagnostic'],source_sha256=sha(t['pre_diagnostic']),
                    restored_sha256=sha(p),purple_pixels=int(masks['mask'].sum()),
                    line_inpaint_pixels=int(masks['line_halo'].sum()),
                    original_available=False,exact_recovery=False)
        restorations.append(record)
        if not t.get('post_image'):
            continue
        pairs.append(dict(trial_id=tid,version=t['version'],physical_success=t['physical_success'],**{k:record[k] for k in ['purple_pixels','line_inpaint_pixels']},
                          pre_source='approximate_inverse_overlay' if masks['mask'].any() else 'diagnostic_image_no_overlay_detected',
                          post_source='unaltered_pre_VLA_head_check',
                          pre_image=str(p),post_image=t['post_image']))
        sample('pre_'+tid,tid,'pre_reconstructed',p)
        sample('diagnostic_'+tid,tid,'pre_diagnostic',Path(t['pre_diagnostic']))
        for mode,threshold,fill in [('unfilled',55,False),('threshold70',70,True)]:
            im,_=recover(raw,threshold,fill)
            q=OUT/'inputs/restoration_variants'/f'{tid}_{mode}.png';write_image(q,im)
            sample(mode+'_'+tid,tid,mode,q)
        # Synthetic roundtrip on BOTH same-trial clean post and a training image.
        refidx=round((len(pairs)-1)*178/21)
        for label,clean,refid in [('post',rgb_read(t['post_image']),None),('train',rgb_read(references[refidx]['image']),refidx)]:
            overlay=synthetic(clean,masks); restored,_=recover(overlay)
            q=OUT/'inputs/synthetic_controls'/f'{tid}_{label}.png';write_image(q,restored)
            sid=f'control_{label}_{tid}';sample(sid,tid,'synthetic_control',q)
            controls.append(dict(sample_id=sid,trial_id=tid,clean_post_sample_id='post_'+tid if label=='post' else None,
                                 clean_training_index=refid,mean_absolute_pixel_error=float(np.abs(clean.astype(float)-restored).mean()),
                                 mask_area_fraction=float(masks['mask'].mean()),line_area_fraction=float(masks['line_halo'].mean())))
    assert len(pairs)==22
    write_json(OUT/'inputs/approximate_pairs.json',pairs);write_json(OUT/'inputs/restoration_audit.json',restorations)
    write_json(OUT/'inputs/synthetic_controls.json',controls);write_json(OUT/'inputs/samples.json',samples)
    cfg=read(OUT/'analysis_config.json')
    cfg.update(analysis='Exploratory paired nearfield trend using approximate inverse-rendered pre images',
        restoration={'renderer':'uint8(0.55*RGB + 0.45*[255,0,255]); diagnostic JPEG quality95',
                     'inverse':'(decodedRGB+0.5-0.45*[255,0,255])/0.55 inside inferred purple mask',
                     'line_repair':'OpenCV TELEA radius3 over line detections dilated 5x5; approximate spatial interpolation',
                     'selection_independent':'All mask thresholds and repair settings fixed before embeddings; no outcome/distance tuning',
                     'limitations':'Mask inferred from rendered colors, no original segmentation mask. JPEG and overwritten edges irreversible. Synthetic controls approximate the renderer and do not certify real-image recovery.',
                     'sensitivities':['inverse_without_line_fill','purple_threshold70','raw_annotated_pre_not_primary'],
                     'controls':'22 clean post + 22 evenly spaced training frames, same spatial overlay templates, JPEG95, then same recovery algorithm'},
        paired_n=22,training_reference_n=179,checkpoint=CHECKPOINT)
    write_json(OUT/'analysis_config.json',cfg)
    for page,offset in enumerate(range(0,len(pairs),6),1):
        sheet=Image.new('RGB',(1200,6*268),(247,249,250));d=ImageDraw.Draw(sheet)
        for j,t in enumerate(pairs[offset:offset+6]):
            tid=t['trial_id'];r=next(x for x in restorations if x['trial_id']==tid)
            for col,(label,path) in enumerate([('Original diagnostic',r['source']),('Approximate recovery',t['pre_image']),('Clean post-alignment',t['post_image'])]):
                d.text((col*400+5,j*268+4),f'{tid} | {label}',fill=(20,35,50))
                im=Image.open(path).convert('RGB');im.thumbnail((320,240));sheet.paste(im,(col*400+40,j*268+25))
        sheet.save(OUT/f'restoration_audit_{page:02d}.jpg',quality=94)
    print('Prepared',len(samples),'feature inputs;',len(pairs),'approximate pairs;',len(controls),'synthetic controls',flush=True)


def analyze():
    f=np.load(OUT/'features/embeddings.npz');ref=np.load(OUT/'references/head_reference_features.npz')
    prov=read(OUT/'features/provenance.json');old=read(OUT/'references/original_provenance.json')
    assert prov['vision_bfloat16_parameter_sha256']==old['vision_bfloat16_parameter_sha256']
    assert sha(OUT/'features/embeddings.npz')==prov['feature_file_sha256']
    samples=read(OUT/'inputs/samples.json');assert list(f['sample_ids'])==[s['sample_id'] for s in samples]
    ix={s:i for i,s in enumerate(f['sample_ids'])};rows=[]
    for s in samples:assert sha(s['image'])==s['image_sha256']
    distances={}
    for mode in ['spatial','global_mean']:
        ds,neighbors=neighbor_distances(f[mode],ref[mode]);distances[mode]=ds
        for t in read(OUT/'inputs/approximate_pairs.json'):
            if mode=='spatial':rows.append(dict(t))
            row=next(r for r in rows if r['trial_id']==t['trial_id']);tid=t['trial_id']
            for stage in ['pre','post','unfilled','threshold70','diagnostic']:
                i=ix[stage+'_'+tid];row[f'{stage}_{mode}']=float(ds[i])
                if mode=='spatial':row[f'{stage}_neighbors']=','.join(map(str,neighbors[i]))
            row[f'delta_{mode}']=row[f'post_{mode}']-row[f'pre_{mode}']
            row[f'percent_{mode}']=100*row[f'delta_{mode}']/row[f'pre_{mode}']
            for stage in ['unfilled','threshold70','diagnostic']:
                row[f'delta_{stage}_{mode}']=row[f'post_{mode}']-row[f'{stage}_{mode}']
    def summary(group):
        return dict(n=len(group),reduced=sum(r['delta_spatial']<0 for r in group),
            median_pre=float(np.median([r['pre_spatial'] for r in group])),median_post=float(np.median([r['post_spatial'] for r in group])),
            median_paired_change=float(np.median([r['delta_spatial'] for r in group])),
            median_paired_percent=float(np.median([r['percent_spatial'] for r in group])),
            unfilled_reduced=sum(r['delta_unfilled_spatial']<0 for r in group),
            threshold70_reduced=sum(r['delta_threshold70_spatial']<0 for r in group),
            global_reduced=sum(r['delta_global_mean']<0 for r in group))
    controls=[]
    for c0 in read(OUT/'inputs/synthetic_controls.json'):
        c=dict(c0);i=ix[c['sample_id']]
        clean=f['spatial'][ix[c['clean_post_sample_id']]] if c['clean_post_sample_id'] else ref['spatial'][c['clean_training_index']]
        dc,_=neighbor_distances(clean[None],ref['spatial'])
        # Training controls use same-reference distance for both clean/recovered.
        c.update(clean_reference_distance=float(dc[0]),recovered_reference_distance=float(distances['spatial'][i]),
                 recovery_score_error=float(distances['spatial'][i]-dc[0]),
                 recovery_descriptor_cosine_distance=float(1-unit(clean)@unit(f['spatial'][i])))
        controls.append(c)
    st={g:summary([r for r in rows if g=='combined' or r['version']==g]) for g in ['combined','V2','V3']}
    err=np.array([abs(c['recovery_score_error']) for c in controls])
    st['synthetic_recovery_check']=dict(n=len(controls),median_absolute_distance_error=float(np.median(err)),
        p95_absolute_distance_error=float(np.quantile(err,.95)),max_absolute_distance_error=float(err.max()),
        median_pixel_mae=float(np.median([c['mean_absolute_pixel_error'] for c in controls])),
        scope='Synthetic overlays on clean images; error is NOT a confidence interval or a bound on real reconstruction error')
    for r in rows:
        r['decrease_larger_than_synthetic_p95']=bool(r['delta_spatial']<-np.quantile(err,.95))
    st['combined']['decrease_larger_than_synthetic_p95']=sum(r['decrease_larger_than_synthetic_p95'] for r in rows)
    st['limitations']=['21 reconstructed pre images; 1 diagnostic pre has no detected overlay',
       'V2 successes only, outcome-selected cohort; no unbiased success-rate or causal claim',
       'Repeated trials in same setup, not 22 independently varied environments',
       'Sigmoid curves connect two measured endpoints only, not temporal measurements',
       '4 selected trials lack completed-alignment head checks; all retained in selection and missing-data CSV']
    write_csv(OUT/'results/approximate_paired_scores.csv',rows);write_json(OUT/'results/approximate_paired_scores.json',rows)
    write_csv(OUT/'results/synthetic_recovery_controls.csv',controls);write_json(OUT/'results/approximate_statistics.json',st)
    print(json.dumps(st,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','analyze']);a=p.parse_args();globals()[a.action]()
