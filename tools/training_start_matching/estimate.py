"""Estimate camera-to-front-wall poses and expose missingness and ambiguity."""
from pathlib import Path
from paths import ROOT
import argparse
import json
import math

import cv2
import numpy as np




def wrap_deg(x):
    return (np.asarray(x)+180.0)%360.0-180.0


def object_points(config):
    g=config['geometry_m']; top=g['top_width']; bottom=g['bottom_width']
    height=math.sqrt(g['side']**2-((top-bottom)/2)**2)
    return np.array([[-top/2,height,0],[top/2,height,0],[bottom/2,0,0],[-bottom/2,0,0]],dtype=np.float64)


def solve_candidates(points, object_xyz, K, distortion):
    try:
        result=cv2.solvePnPGeneric(object_xyz,points,K,distortion,flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return []
    if not result[0]: return []
    candidates=[]
    for rvec,tvec in zip(result[1],result[2]):
        rvec=rvec.copy();tvec=tvec.copy()
        try:
            rvec,tvec=cv2.solvePnPRefineLM(object_xyz,points,K,distortion,rvec,tvec)
        except cv2.error:
            pass
        R=cv2.Rodrigues(rvec)[0]
        camera=(-R.T@tvec).reshape(3)
        depths=(R@object_xyz.T+tvec)[2]
        if np.any(depths<=0) or camera[2]<=0 or not np.isfinite(camera).all(): continue
        projection=cv2.projectPoints(object_xyz,rvec,tvec,K,distortion)[0].reshape(-1,2)
        rms=float(np.sqrt(np.mean(np.sum((projection-points)**2,axis=1))))
        forward=R.T@np.array([0.0,0.0,1.0])
        heading=float(np.degrees(np.arctan2(forward[0],-forward[2])))
        pitch=float(np.degrees(np.arctan2(forward[1],np.hypot(forward[0],forward[2]))))
        candidate={'rvec':rvec.reshape(3).tolist(),'tvec':tvec.reshape(3).tolist(),'rotation_object_to_camera':R.tolist(),'camera_position_object_m':camera.tolist(),'camera_heading_deg':heading,'camera_pitch_in_wall_frame_deg':pitch,'reprojection_rms_px':rms,'projected_points':projection.tolist(),'state':[float(camera[0]),float(camera[2]),heading]}
        duplicate=any(np.linalg.norm(camera-np.array(c['camera_position_object_m']))<1e-5 and abs(float(wrap_deg(heading-c['camera_heading_deg'])))<1e-3 for c in candidates)
        if not duplicate:candidates.append(candidate)
    return sorted(candidates,key=lambda c:c['reprojection_rms_px'])


def annotation_files():
    files=[ROOT/'annotations/experiments.jsonl']
    files.extend(sorted((ROOT/'annotations').glob('new_batch_*.jsonl')))
    for job in sorted((ROOT/'jobs').glob('train_*.json')):
        files.append(ROOT/'annotations'/f'{job.stem}.jsonl')
    return [p for p in files if p.exists()]


def approved_off_frame_point(sample_id, point, config):
    revision=config.get('off_frame_revision', {})
    approved=revision.get('approved_points', {}).get(sample_id, {}).get(point['label'])
    if not revision.get('user_confirmed') or not approved:
        return False
    bounds=revision['canvas_bounds'][sample_id]
    return (point.get('status')=='inferred'
            and point.get('inference_kind')=='off_frame_extrapolation'
            and point.get('out_of_frame') is True
            and all(point.get(key)==value for key,value in approved.items())
            and bounds[0]<=point['u']<bounds[2] and bounds[1]<=point['v']<bounds[3])


def load_annotations():
    annotations={};provenance={}
    for path in annotation_files():
        for line_no,line in enumerate(path.read_text().splitlines(),1):
            if not line.strip():continue
            row=json.loads(line)
            sid=row['sample_id']
            if sid in annotations:raise ValueError(f'Duplicate primary annotation: {sid}')
            if row.get('annotator',{}).get('model')!='gpt-6-astra' or row.get('annotator',{}).get('reasoning_effort')!='low':
                raise ValueError(f'Unapproved annotation model in {path}:{line_no}')
            annotations[sid]=row;provenance[sid]={'file':str(path),'line':line_no}
    overrides=ROOT/'annotations/review_overrides.jsonl'
    if overrides.exists():
        for line_no,line in enumerate(overrides.read_text().splitlines(),1):
            if not line.strip():continue
            row=json.loads(line);sid=row['sample_id']
            assert row['annotator']=={'model':'gpt-6-astra','reasoning_effort':'low'}
            row['previous_annotation']=annotations.get(sid)
            annotations[sid]=row;provenance[sid]={'file':str(overrides),'line':line_no,'is_review_override':True}
    # User-authorized occlusion estimates are a separate layer: never overwrite
    # the reviewed observations or silently move an already localized point.
    config=json.loads((ROOT/'analysis_config.json').read_text())
    revision=config.get('occlusion_revision')
    if revision:
        allowed=set(revision['selected_sample_ids']);seen=set()
        for path in sorted((ROOT/'annotations').glob('occlusion_batch_*.jsonl')):
            for line_no,line in enumerate(path.read_text().splitlines(),1):
                if not line.strip():continue
                row=json.loads(line);sid=row['sample_id']
                assert sid in allowed and sid not in seen, sid
                seen.add(sid)
                assert row['annotator']=={'model':'gpt-6-astra','reasoning_effort':'low'}
                assert row['review_kind']=='bottle_occlusion_shape_prior'
                previous=annotations[sid]
                before={p['label']:p for p in previous['points']}
                assert len(row['points'])==4 and {p['label'] for p in row['points']}==set('ABCD')
                estimated=[]
                for point in row['points']:
                    old=before[point['label']]
                    if old['u'] is not None and old['v'] is not None:
                        assert all(point.get(k)==old.get(k) for k in ['u','v','status','uncertainty_px']), (sid,point['label'])
                    elif point['u'] is not None and point['v'] is not None:
                        assert point['status']=='inferred' and point['uncertainty_px']>0 and point.get('note')
                        estimated.append(point['label'])
                row['occlusion_estimated_labels']=estimated
                row['previous_annotation']=previous
                annotations[sid]=row
                provenance[sid]={'file':str(path),'line':line_no,'is_occlusion_estimate':True}
        assert seen==allowed, {'missing_occlusion_annotations':sorted(allowed-seen)}
    overrides=ROOT/'annotations/new_review_overrides.jsonl'
    if overrides.exists():
        allowed=set(config['cohort_extension']['added_sample_ids'])
        for line_no,line in enumerate(overrides.read_text().splitlines(),1):
            if not line.strip():continue
            row=json.loads(line);sid=row['sample_id']
            assert sid in allowed
            assert row['annotator']=={'model':'gpt-6-astra','reasoning_effort':'low'}
            row['previous_annotation']=annotations[sid]
            annotations[sid]=row
            provenance[sid]={'file':str(overrides),'line':line_no,'is_review_override':True}
    revision=config.get('off_frame_revision')
    if revision:
        path=ROOT/'annotations/off_frame_estimates.jsonl'
        seen=set()
        for line_no,line in enumerate(path.read_text().splitlines(),1):
            if not line.strip():continue
            row=json.loads(line);sid=row['sample_id']
            assert sid in revision['selected_sample_ids'] and sid not in seen
            seen.add(sid)
            assert row['annotator']=={'model':'gpt-6-astra','reasoning_effort':'low'}
            previous=annotations[sid]
            before={p['label']:p for p in previous['points']}
            approved_labels=set(revision['approved_points'][sid])
            assert len(row['points'])==4 and {p['label'] for p in row['points']}==set('ABCD')
            for point in row['points']:
                old=before[point['label']]
                if point['label'] in approved_labels:
                    assert old['u'] is None and old['v'] is None
                    assert approved_off_frame_point(sid,point,config)
                else:
                    assert point==old, (sid,point['label'])
            row['off_frame_estimated_labels']=sorted(approved_labels)
            row['previous_annotation']=previous
            annotations[sid]=row
            provenance[sid]={'file':str(path),'line':line_no,'is_off_frame_estimate':True,
                             'user_confirmed':True,'accepted_annotation_sha256':revision['accepted_annotation_sha256']}
        assert seen==set(revision['selected_sample_ids'])
    return annotations,provenance


def estimate_one(sample,annotation,config,K,distortion):
    result={'sample_id':sample['sample_id'],'split':sample['split'],'episode_id':sample.get('episode_id'),'trial_id':sample.get('trial_id'),'method':sample.get('method'),'case_id':sample.get('case_id'),'image':sample['image'],'annotation':annotation,'valid':False,'strict_valid':False,'failure_reason':None}
    if annotation is None:
        result['failure_reason']='annotation_pending';return result
    result['occlusion_estimated_labels']=annotation.get('occlusion_estimated_labels',[])
    result['occlusion_estimated']=bool(result['occlusion_estimated_labels'])
    points_by_label={p['label']:p for p in annotation['points']}
    if set(points_by_label)!=set('ABCD'):raise ValueError(f'Wrong labels: {sample["sample_id"]}')
    incomplete=[k for k,p in points_by_label.items() if p.get('u') is None or p.get('v') is None or p.get('status')=='unresolvable']
    if incomplete or not annotation.get('resolvable',False):
        result['failure_reason']='unresolvable_front_wall_corners';result['missing_corners']=incomplete;return result
    points=np.array([[points_by_label[k]['u'],points_by_label[k]['v']] for k in 'ABCD'],dtype=np.float64)
    width,height=sample['image_size']
    outside=(points[:,0]<0)|(points[:,0]>=width)|(points[:,1]<0)|(points[:,1]>=height)
    allowed=np.array([approved_off_frame_point(sample['sample_id'],points_by_label[k],config) for k in 'ABCD'])
    if not np.isfinite(points).all() or (outside & ~allowed).any():
        result['failure_reason']='out_of_image_coordinates';return result
    if outside.any():
        result['off_frame_estimated']=True
        result['off_frame_estimated_labels']=[k for k,is_out in zip('ABCD',outside) if is_out]
    area=float(abs(cv2.contourArea(points.astype(np.float32))))
    result['quad_area_px2']=area
    estimator=config['pose_estimator']
    if not cv2.isContourConvex(points.astype(np.float32)) or area<estimator['minimum_quad_area_px2']:
        result['failure_reason']='degenerate_or_nonconvex_quadrilateral';return result
    edges=np.linalg.norm(points-np.roll(points,1,axis=0),axis=1)
    if edges.min()<estimator['minimum_edge_px']:
        result['failure_reason']='edge_too_short';return result
    obj=object_points(config)
    candidates=solve_candidates(points,obj,K,distortion)
    result['pose_candidates']=candidates
    if not candidates:
        result['failure_reason']='no_physical_pnp_solution';return result
    best=candidates[0];result.update(best)
    if best['reprojection_rms_px']>estimator['max_reprojection_rms_px']:
        result['failure_reason']='high_reprojection_error';return result
    ambiguous=[]
    for alternative in candidates[1:]:
        position_gap=float(np.linalg.norm(np.array(best['camera_position_object_m'])-np.array(alternative['camera_position_object_m'])))
        heading_gap=abs(float(wrap_deg(best['camera_heading_deg']-alternative['camera_heading_deg'])))
        if alternative['reprojection_rms_px']-best['reprojection_rms_px']<=estimator['ambiguity_error_gap_px'] and (position_gap>estimator['ambiguity_translation_m'] or heading_gap>estimator['ambiguity_heading_deg']):
            ambiguous.append({'position_gap_m':position_gap,'heading_gap_deg':heading_gap,'alternative_rms_px':alternative['reprojection_rms_px']})
    result['ambiguous']=bool(ambiguous);result['ambiguity_details']=ambiguous
    result['inferred_corners']=[k for k,p in points_by_label.items() if p['status']!='visible']
    sensitivity=config['sensitivity']
    seed=sensitivity['random_seed']+int(sample['sample_id'].split('_')[-1])
    rng=np.random.default_rng(seed)
    uncertainty=np.array([float(points_by_label[k].get('uncertainty_px') or 0) for k in 'ABCD'])
    perturbed=[]
    nominal=np.array(best['state'])
    for _ in range(sensitivity['pixel_perturbations']):
        radii=uncertainty*np.sqrt(rng.random(4));angles=rng.uniform(0,2*np.pi,4)
        pp=points+np.c_[radii*np.cos(angles),radii*np.sin(angles)]
        if not cv2.isContourConvex(pp.astype(np.float32)):continue
        options=solve_candidates(pp,obj,K,distortion)
        options=[c for c in options if c['reprojection_rms_px']<=estimator['max_reprojection_rms_px']]
        if not options:continue
        def displacement(c):
            delta=np.array(c['state'])-nominal;delta[2]=float(wrap_deg(delta[2]));return np.linalg.norm(delta/np.array([.10,.10,15.]))
        # Perturbation stability is conditional on the nominal branch. Separate
        # ambiguity flags preserve the uncertainty between distinct branches.
        perturbed.append(min(options,key=displacement)['state'])
    solve_fraction=len(perturbed)/sensitivity['pixel_perturbations']
    result['perturbation_solve_fraction']=solve_fraction
    result['perturbation_states']=perturbed
    result['stable']=False
    if perturbed:
        arr=np.array(perturbed);delta=arr-nominal;delta[:,2]=wrap_deg(delta[:,2])
        position95=float(np.quantile(np.linalg.norm(delta[:,:2],axis=1),.95))
        heading95=float(np.quantile(np.abs(delta[:,2]),.95))
        result['perturbation_xz_p95_radius_m']=position95
        result['perturbation_heading_p95_deviation_deg']=heading95
        result['perturbation_state_delta_q025_q975']=np.quantile(delta,[.025,.975],axis=0).tolist()
        result['stable']=solve_fraction>=sensitivity['minimum_solve_fraction'] and position95<=sensitivity['max_xz_p95_radius_m'] and heading95<=sensitivity['max_heading_p95_deviation_deg']
    result['valid']=True
    result['strict_valid']=not result['inferred_corners'] and not result['ambiguous'] and result['stable']
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    samples=json.loads((ROOT/'samples.json').read_text())
    config=json.loads((ROOT/'analysis_config.json').read_text())
    calibration=json.loads((ROOT/'provenance/camera_intrinsics.json').read_text())['streams']['chest_view']
    K=np.array([[calibration['fx'],0,calibration['cx']],[0,calibration['fy'],calibration['cy']],[0,0,1]],dtype=float)
    distortion=np.array(calibration['distortion_coeffs'],dtype=float)
    annotations,provenance=load_annotations()
    assert set(annotations)<=set(s['sample_id'] for s in samples)
    if not args.partial:assert len(annotations)==len(samples),(len(annotations),len(samples))
    results=[]
    for sample in samples:
        row=estimate_one(sample,annotations.get(sample['sample_id']),config,K,distortion)
        row['annotation_source']=provenance.get(sample['sample_id'])
        results.append(row)
    out=ROOT/('partial' if args.partial else 'results');out.mkdir(exist_ok=True)
    (out/'poses.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    summary={split:{'total':sum(r['split']==split for r in results),'annotated':sum(r['split']==split and r['annotation'] is not None for r in results),'valid':sum(r['split']==split and r['valid'] for r in results),'strict_valid':sum(r['split']==split and r['strict_valid'] for r in results),'inferred_or_unstable_or_ambiguous':sum(r['split']==split and r['valid'] and not r['strict_valid'] for r in results)} for split in ['train','experiment']}
    (out/'pose_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
