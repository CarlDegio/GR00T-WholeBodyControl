"""Two frozen-reference comparisons with explicit trial and geometry denominators."""
from pathlib import Path
from paths import ROOT
import csv
import json
import math

import numpy as np
from scipy.spatial.distance import cdist, pdist


METHODS=['full_vln','nav_direct_vla','geometric_vla','full_objectnav']


def wrap_deg(x):return (np.asarray(x)+180.)%360.-180.


def distance_matrix(a,b,scales):
    a=np.asarray(a,float).reshape(-1,3);b=np.asarray(b,float).reshape(-1,3)
    delta=a[:,None,:]-b[None,:,:]
    delta[:,:,2]=wrap_deg(delta[:,:,2])
    return np.sqrt(np.sum((delta/np.asarray(scales))**2,axis=-1))


def fit_calibration(training,config):
    states=np.array([p['state'] for p in training])
    if len(states)<3:raise ValueError('Fewer than three valid training poses; cannot calibrate neighborhood.')
    center=np.median(states,axis=0)
    center[2]=np.degrees(np.arctan2(np.sin(np.radians(states[:,2])).mean(),np.cos(np.radians(states[:,2])).mean()))
    deviations=states-center;deviations[:,2]=wrap_deg(deviations[:,2])
    scales=1.4826*np.median(np.abs(deviations),axis=0)
    floors=config['metric']['scale_floor']
    scales=np.maximum(scales,[floors['x_m'],floors['normal_distance_m'],floors['heading_deg']])
    distances=distance_matrix(states,states,scales);np.fill_diagonal(distances,np.inf)
    loo=distances.min(axis=1)
    threshold=max(float(np.quantile(loo,.95)),1e-8)
    embedded=embedding(states,scales)
    bandwidth=max(float(np.median(pdist(embedded))),1e-8)
    return {'fitted_on':'Valid ALL-training poses only; fixed for both reference sets','training_pose_count':len(training),'center':center.tolist(),'scales':scales.tolist(),'threshold':threshold,'loo_distances':loo.tolist(),'loo_distance_quantiles':dict(zip(['q05','q50','q95','q99'],np.quantile(loo,[.05,.5,.95,.99]).tolist())),'mmd_bandwidth':bandwidth,'mmd_embedding':'x/sx,z/sz,cos(heading)/sangle_rad,sin(heading)/sangle_rad'}


def embedding(states,scales):
    states=np.asarray(states,float).reshape(-1,3);angle=np.radians(states[:,2]);sa=np.radians(scales[2])
    return np.c_[states[:,0]/scales[0],states[:,1]/scales[1],np.cos(angle)/sa,np.sin(angle)/sa]


def mmd_squared(a,b,scales,bandwidth):
    if len(a)<2 or len(b)<2:return None
    a=embedding(a,scales);b=embedding(b,scales)
    def kernel(x,y):return np.exp(-cdist(x,y,'sqeuclidean')/(2*bandwidth**2))
    aa=kernel(a,a);bb=kernel(b,b);ab=kernel(a,b)
    return float((aa.sum()-np.trace(aa))/(len(a)*(len(a)-1))+(bb.sum()-np.trace(bb))/(len(b)*(len(b)-1))-2*ab.mean())


def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    margin=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0.0,100*(center-margin)),min(100.0,100*(center+margin))]


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize_reference(name,reference,poses,preflight,calibration,config,all_reference_count):
    by_trial={p['trial_id']:p for p in poses if p['split']=='experiment'}
    scales=calibration['scales'];threshold=calibration['threshold']
    reference_states=np.array([p['state'] for p in reference]).reshape(-1,3)
    strict_reference=[p for p in reference if p['strict_valid']]
    strict_states=np.array([p['state'] for p in strict_reference]).reshape(-1,3)
    rows=[];nearest=[]
    for trial in preflight['trials']:
        pose=by_trial.get(trial['trial_id'])
        valid=pose is not None and pose['valid']
        row={'method':trial['method'],'case_id':trial['case_id'],'trial_id':trial['trial_id'],'vla_started':trial['vla_first_action'],'has_chest_snapshot':trial['first_policy_chest_frame'] is not None,'pose_valid':valid,'strict_pose_valid':bool(valid and pose['strict_valid']),'failure_reason':pose['failure_reason'] if pose is not None else 'vla_not_started','nearest_episode_id':None,'normalized_distance':None,'neighborhood_hit':False,'strict_neighborhood_hit':False,'nearest_position_distance_m':None,'nearest_heading_difference_deg':None,'camera_x_m':None,'camera_normal_distance_m':None,'camera_heading_deg':None,'inferred_corners':None,'ambiguous':None,'perturbation_xz_p95_radius_m':None,'perturbation_heading_p95_deviation_deg':None,'perturbation_hit_fraction':None,'independent_success':trial['annotation_values'].get('success')}
        row['occlusion_estimated']=bool(pose and pose.get('occlusion_estimated'))
        row['occlusion_estimated_labels']=','.join(pose.get('occlusion_estimated_labels',[])) if pose else ''
        if config.get('off_frame_revision'):
            row['off_frame_estimated']=bool(pose and pose.get('off_frame_estimated'))
            row['off_frame_estimated_labels']=','.join(pose.get('off_frame_estimated_labels',[])) if pose else ''
        if valid:
            row.update({'camera_x_m':pose['state'][0],'camera_normal_distance_m':pose['state'][1],'camera_heading_deg':pose['state'][2],'inferred_corners':','.join(pose['inferred_corners']),'ambiguous':pose['ambiguous'],'perturbation_xz_p95_radius_m':pose.get('perturbation_xz_p95_radius_m'),'perturbation_heading_p95_deviation_deg':pose.get('perturbation_heading_p95_deviation_deg')})
            if len(reference):
                distances=distance_matrix([pose['state']],reference_states,scales)[0]
                order=np.argsort(distances);j=int(order[0]);target=reference[j]
                row.update({'nearest_episode_id':target['episode_id'],'normalized_distance':float(distances[j]),'neighborhood_hit':bool(distances[j]<=threshold),'nearest_position_distance_m':float(np.linalg.norm(np.array(pose['state'][:2])-np.array(target['state'][:2]))),'nearest_heading_difference_deg':abs(float(wrap_deg(pose['state'][2]-target['state'][2])))})
                nearest.append({'trial_id':trial['trial_id'],'sample_id':pose['sample_id'],'experimental_image':pose['image'],'neighbors':[{'sample_id':reference[int(k)]['sample_id'],'episode_id':reference[int(k)]['episode_id'],'image':reference[int(k)]['image'],'distance':float(distances[int(k)])} for k in order[:3]]})
                if pose['perturbation_states']:
                    dd=distance_matrix(pose['perturbation_states'],reference_states,scales).min(axis=1)
                    row['perturbation_hit_fraction']=float(np.mean(dd<=threshold))
            if pose['strict_valid'] and len(strict_reference):
                row['strict_neighborhood_hit']=bool(distance_matrix([pose['state']],strict_states,scales).min()<=threshold)
        rows.append(row)
    methods=[]
    for method in METHODS:
        mr=[r for r in rows if r['method']==method];good=[r for r in mr if r['normalized_distance'] is not None]
        n=len(mr);hits=sum(r['neighborhood_hit'] for r in mr);strict_hits=sum(r['strict_neighborhood_hit'] for r in mr)
        distances=[r['normalized_distance'] for r in good]
        position=[r['nearest_position_distance_m'] for r in good]
        heading=[r['nearest_heading_difference_deg'] for r in good]
        valid_poses=[by_trial[r['trial_id']] for r in good]
        states=[p['state'] for p in valid_poses]
        factors=config['metric']['threshold_sensitivity_factors']
        methods.append({'method':method,'n_trials':n,'n_vla_started':sum(r['vla_started'] for r in mr),'n_pose_valid':sum(r['pose_valid'] for r in mr),'n_pose_missing':sum(not r['pose_valid'] for r in mr),'n_strict_pose_valid':sum(r['strict_pose_valid'] for r in mr),'n_neighborhood_hits':hits,'neighborhood_hit_rate_pct':100*hits/n if n else None,'hit_rate_wilson95_pct':wilson(hits,n),'conditional_hit_rate_pct':100*hits/len(good) if good else None,'mean_normalized_distance':float(np.mean(distances)) if distances else None,'median_normalized_distance':float(np.median(distances)) if distances else None,'mean_nearest_position_distance_m':float(np.mean(position)) if position else None,'mean_nearest_heading_difference_deg':float(np.mean(heading)) if heading else None,'strict_neighborhood_hits':strict_hits,'strict_hit_rate_pct':100*strict_hits/n if n else None,'threshold_sensitivity':[{'factor':factor,'hits':sum(d<=threshold*factor for d in distances),'rate_pct':100*sum(d<=threshold*factor for d in distances)/n if n else None} for factor in factors],'mmd2_unbiased':mmd_squared(states,reference_states,scales,calibration['mmd_bandwidth']) if len(reference) else None})
    summary={'reference_name':name,'reference_source_frames':all_reference_count,'reference_pose_valid':len(reference),'reference_pose_missing':all_reference_count-len(reference),'strict_reference_pose_valid':len(strict_reference),'calibration':calibration,'methods':methods,'trials':rows,'notes':['Missing geometry is not evidence that the physical state was outside the neighborhood; the end-to-end hit rate combines geometric availability and proximity.','Only first-frame chest-view geometry is assessed; no complete VLA observation-distribution equivalence is implied.','Unbiased MMD2 may be slightly negative at finite sample size and cannot be interpreted as a percentage.','Small and unequal trial counts provide descriptive evidence, not a stable estimate of a population distribution.']}
    diagnostic_path=ROOT/'results/geometry_diagnostics.json'
    if diagnostic_path.exists():
        diagnostic=json.loads(diagnostic_path.read_text())
        summary['physical_geometry_status']=diagnostic['status']
        summary['notes'].insert(0,diagnostic['reporting_rule'])
    folder=ROOT/'results'/name;folder.mkdir(exist_ok=True)
    (folder/'statistics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (folder/'nearest_neighbors.json').write_text(json.dumps(nearest,ensure_ascii=False,indent=2)+'\n')
    write_csv(folder/'trials.csv',rows)
    method_csv=[{k:(json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in m.items()} for m in methods]
    write_csv(folder/'methods.csv',method_csv)
    return summary


def main():
    poses=json.loads((ROOT/'results/poses.json').read_text());preflight=json.loads((ROOT/'preflight.json').read_text());config=json.loads((ROOT/'analysis_config.json').read_text())
    training=[p for p in poses if p['split']=='train' and p['valid']]
    calibration=fit_calibration(training,config)
    (ROOT/'results/calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    all_summary=summarize_reference('all_training',training,poses,preflight,calibration,config,238)
    tail_summary=summarize_reference('from_00145',[p for p in training if p['episode_id']>=145],poses,preflight,calibration,config,93)
    for a,b in zip(all_summary['trials'],tail_summary['trials']):
        assert a['trial_id']==b['trial_id']
        if a['normalized_distance'] is not None and b['normalized_distance'] is not None:
            assert a['normalized_distance']<=b['normalized_distance']+1e-9
        assert not b['neighborhood_hit'] or a['neighborhood_hit']
    flat=[]
    for p in poses:
        flat.append({'sample_id':p['sample_id'],'split':p['split'],'episode_id':p['episode_id'],'trial_id':p['trial_id'],'method':p['method'],'valid':p['valid'],'strict_valid':p['strict_valid'],'failure_reason':p['failure_reason'],'camera_x_m':p.get('state',[None]*3)[0],'camera_normal_distance_m':p.get('state',[None]*3)[1],'camera_heading_deg':p.get('state',[None]*3)[2],'camera_height_in_wall_frame_m':p.get('camera_position_object_m',[None]*3)[1],'reprojection_rms_px':p.get('reprojection_rms_px'),'inferred_corners':','.join(p.get('inferred_corners',[])),'ambiguous':p.get('ambiguous'),'stable':p.get('stable'),'perturbation_xz_p95_radius_m':p.get('perturbation_xz_p95_radius_m'),'perturbation_heading_p95_deviation_deg':p.get('perturbation_heading_p95_deviation_deg'),'image':p['image']})
    for row,p in zip(flat,poses):
        row['camera_pitch_in_wall_frame_deg']=p.get('camera_pitch_in_wall_frame_deg')
        row['occlusion_estimated']=p.get('occlusion_estimated',False)
        row['occlusion_estimated_labels']=','.join(p.get('occlusion_estimated_labels',[]))
        if config.get('off_frame_revision'):
            row['off_frame_estimated']=p.get('off_frame_estimated',False)
            row['off_frame_estimated_labels']=','.join(p.get('off_frame_estimated_labels',[]))
    write_csv(ROOT/'results/poses.csv',flat)
    print(json.dumps({'all_valid_reference':len(training),'tail_valid_reference':tail_summary['reference_pose_valid'],'shared_scales':calibration['scales'],'shared_threshold':calibration['threshold'],'methods_all':all_summary['methods'],'methods_tail':tail_summary['methods']}))


if __name__=='__main__':main()
