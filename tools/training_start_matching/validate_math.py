"""Independent synthetic checks for coordinate conventions and set comparisons."""
from pathlib import Path
from paths import ROOT
import json
import copy

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from estimate import object_points, solve_candidates, estimate_one, load_annotations
from metrics import distance_matrix, mmd_squared




def main():
    config=json.loads((ROOT/'analysis_config.json').read_text())
    calibration=json.loads((ROOT/'provenance/camera_intrinsics.json').read_text())['streams']['chest_view']
    K=np.array([[calibration['fx'],0,calibration['cx']],[0,calibration['fy'],calibration['cy']],[0,0,1]],dtype=float)
    dist=np.array(calibration['distortion_coeffs'])
    model=object_points(config)
    edges=np.linalg.norm(model-np.roll(model,1,axis=0),axis=1)
    geometry=config['geometry_m']
    assert np.allclose(edges,[geometry['side'],geometry['top_width'],geometry['side'],geometry['bottom_width']])
    examples=[]
    for camera,yaw,pitch in [([0,.30,.8],0,.20),([.25,.30,1.2],.3,.15),([-.3,.25,1.8],-.4,.1)]:
        camera=np.array(camera)
        R=Rotation.from_euler('xyz',[np.pi-pitch,yaw,0.02]).as_matrix()
        rvec=cv2.Rodrigues(R)[0];tvec=(-R@camera).reshape(3,1)
        pixels=cv2.projectPoints(model,rvec,tvec,K,dist)[0].reshape(-1,2)
        solutions=solve_candidates(pixels,model,K,dist)
        assert solutions
        best=solutions[0]
        error=float(np.linalg.norm(np.array(best['camera_position_object_m'])-camera))
        assert error<1e-5,error
        assert best['reprojection_rms_px']<1e-5
        examples.append({'true_camera_center_m':camera.tolist(),'camera_center_error_m':error,'reprojection_rms_px':best['reprojection_rms_px']})
    wrap_distance=float(distance_matrix([[0,1,179]],[[0,1,-179]],[1,1,1])[0,0])
    assert np.isclose(wrap_distance,2.)
    train=np.array([[0,1,0],[.2,1.2,20],[-.1,.9,-10]])
    query=np.array([[.01,1.01,1],[.3,1.2,19]])
    full=distance_matrix(query,train,[.1,.1,10]).min(axis=1)
    subset=distance_matrix(query,train[1:],[.1,.1,10]).min(axis=1)
    assert np.all(full<=subset)
    aa=mmd_squared(query,train,[.1,.1,10],1.)
    bb=mmd_squared(train,query,[.1,.1,10],1.)
    assert np.isclose(aa,bb)
    result={'status':'passed','geometry_edge_lengths_m':edges.tolist(),'synthetic_pose_recovery':examples,'wrap_179_minus_minus179_deg':wrap_distance,'nested_reference_monotonicity':True,'mmd_symmetry':True,'scope':'Software math checks only; not an independent physical calibration or validation of image annotations.'}
    if config.get('off_frame_revision'):
        annotations,_=load_annotations()
        samples={s['sample_id']:s for s in json.loads((ROOT/'samples.json').read_text())}
        checks=[]
        for sid in config['off_frame_revision']['selected_sample_ids']:
            sample=samples[sid];annotation=annotations[sid]
            accepted=estimate_one(sample,annotation,config,K,dist)
            assert accepted['valid'] and not accepted['strict_valid']
            ordinary=copy.deepcopy(config);ordinary.pop('off_frame_revision')
            assert estimate_one(sample,annotation,ordinary,K,dist)['failure_reason']=='out_of_image_coordinates'
            altered=copy.deepcopy(annotation)
            point=next(p for p in altered['points'] if p.get('out_of_frame'))
            point['v']+=1
            assert estimate_one(sample,altered,config,K,dist)['failure_reason']=='out_of_image_coordinates'
            false_visible=copy.deepcopy(annotation)
            next(p for p in false_visible['points'] if p.get('out_of_frame'))['status']='visible'
            assert estimate_one(sample,false_visible,config,K,dist)['failure_reason']=='out_of_image_coordinates'
            checks.append({'sample_id':sid,'confirmed_coordinates_accepted':True,
                           'without_confirmation_or_with_changed_coordinate_rejected':True,
                           'out_of_frame_visible_claim_rejected':True,'excluded_from_strict':True})
        result['off_frame_regression_checks']=checks
    (ROOT/'provenance/math_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
