"""Visible horizontal support patches for receptacle-side BasePose.

Only RGB-D and current instance masks are used. Unknown surface is not assumed
available, and an associated anchor is always remeasured in the current frame.
"""

import math

import cv2
import numpy as np


def point_image(depth, calibration):
    v,u=np.indices(depth.shape)
    optical=np.stack(((u-calibration.cx)*depth/calibration.fx,
                      (v-calibration.cy)*depth/calibration.fy,depth),axis=-1)
    return calibration.camera_to_body(optical)


def horizontal_pixels(points, depth, visible_mask):
    valid=np.asarray(visible_mask,dtype=bool)&np.isfinite(depth)&(depth>.1)&(depth<3)
    valid=cv2.erode(valid.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    dx=np.zeros_like(points);dy=np.zeros_like(points)
    dx[:,3:-3]=points[:,6:]-points[:,:-6]
    dy[3:-3]=points[6:]-points[:-6]
    normal=np.cross(dx,dy)
    length=np.linalg.norm(normal,axis=-1)
    horizontal=valid&(length>1e-8)&(np.abs(normal[...,2])>=.90*length)
    horizontal&=(np.linalg.norm(dx,axis=-1)<.10)&(np.linalg.norm(dy,axis=-1)<.10)
    horizontal&=(points[...,2]>.25)&(points[...,2]<1.4)&(points[...,0]>.15)
    return horizontal


def _candidate(points, region, minimum_margin_m, resolution, preferred_body=None):
    pixels=np.column_stack(np.nonzero(region))
    cloud=points[region]
    origin=np.floor(cloud[:,:2].min(axis=0)/resolution)*resolution-3*resolution
    cells=np.rint((cloud[:,:2]-origin)/resolution).astype(int)
    shape=cells.max(axis=0)+4
    if np.prod(shape)>500000 or np.any(shape>1200):return None
    observed=np.zeros(tuple(shape[::-1]),np.uint8)
    observed[cells[:,1],cells[:,0]]=1
    # Interpolate only sampling cracks of at most one cell. All candidate
    # centers remain on the original measured support and retain a full cell
    # of reserve in the reported metric margin.
    filled=cv2.morphologyEx(observed,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    distance=cv2.distanceTransform(filled,cv2.DIST_L2,5)*resolution-resolution
    margins=distance[cells[:,1],cells[:,0]]
    eligible=margins>=minimum_margin_m
    if not eligible.any():return None
    # Choose the nearest sufficiently interior observed point, keeping large
    # bed/table centers from pulling the base unnecessarily far underneath.
    ranges=np.linalg.norm(cloud[:,:2],axis=1)
    score=(ranges+.15*np.abs(cloud[:,1]) if preferred_body is None
           else np.linalg.norm(cloud-np.asarray(preferred_body),axis=1))
    index=int(np.argmin(np.where(eligible,score,np.inf)))
    anchor=cloud[index]
    if preferred_body is not None and np.linalg.norm(anchor-preferred_body)>.12:
        return None
    if observed.sum()*resolution**2 < .03:
        return None
    result=dict(body_point=anchor.tolist(),pixel=pixels[index,::-1].tolist(),
                distance_m=float(ranges[index]),height_m=float(np.median(cloud[:,2])),
                observed_margin_m=float(margins[index]),
                observed_area_m2=float(observed.sum()*resolution**2),
                support_pixels=int(region.sum()),yaw_error_rad=None,
                reference_source='visible_horizontal_surface_contour')
    boundary=filled & ~cv2.erode(filled,np.ones((3,3),np.uint8))
    row,col=np.nonzero(boundary)
    boundary_xy=origin+np.column_stack((col,row))*resolution
    # Prefer the visible boundary between the support anchor and the robot.
    front=np.linalg.norm(boundary_xy,axis=1)<np.linalg.norm(anchor[:2])-.03
    if not front.any():return result
    candidates=boundary_xy[front]
    center=candidates[np.argmin(np.linalg.norm(candidates-anchor[:2],axis=1))]
    local=boundary_xy[np.linalg.norm(boundary_xy-center,axis=1)<.12]
    if len(local)<8:return result
    mean=local.mean(axis=0)
    _,_,axes=np.linalg.svd(local-mean,full_matrices=False)
    tangent=axes[0]
    normal=np.array([-tangent[1],tangent[0]])
    if np.dot(normal,anchor[:2])<0:normal=-normal
    residual=np.abs((local-mean)@normal)
    along=(local-mean)@tangent
    if np.percentile(residual,90)>.018 or np.ptp(along)<.08:return result
    ends=mean+np.outer([np.percentile(along,10),np.percentile(along,90)],tangent)
    image_ends=[]
    for endpoint in ends:
        match=int(np.argmin(np.linalg.norm(cloud[:,:2]-endpoint,axis=1)))
        image_ends.append(pixels[match,::-1].tolist())
    image_ends_array=np.asarray(image_ends)
    height,width=region.shape
    if (np.all(image_ends_array[:,0]<8) or np.all(image_ends_array[:,0]>=width-8)
            or np.all(image_ends_array[:,1]<8) or np.all(image_ends_array[:,1]>=height-8)):
        result['reference_rejected_reason']='surface contour follows image boundary'
        return result
    result.update(yaw_error_rad=float(math.atan2(normal[1],normal[0])),
                  reference_center_body=[float(mean[0]),float(mean[1]),result['height_m']],
                  reference_endpoints_px=image_ends,
                  contour_residual_90_m=float(np.percentile(residual,90)))
    return result


def support_candidates(depth, calibration, visible_mask, *, minimum_margin_m=.07,
                       resolution=.01, preferred_body=None):
    points=point_image(depth,calibration)
    horizontal=horizontal_pixels(points,depth,visible_mask)
    if horizontal.sum()<200:return []
    heights=points[...,2]
    bins=np.arange(.25,1.45,.025)
    counts,_=np.histogram(heights[horizontal],bins=bins)
    options=[]
    for index in np.argsort(counts)[::-1][:8]:
        if counts[index]<200:continue
        height=.5*(bins[index]+bins[index+1])
        plane=horizontal&(np.abs(heights-height)<.025)
        count,labels,stats,_=cv2.connectedComponentsWithStats(plane.astype(np.uint8))
        for component in range(1,count):
            if stats[component,cv2.CC_STAT_AREA]<200:continue
            candidate=_candidate(points,labels==component,minimum_margin_m,resolution,preferred_body)
            if candidate is None:continue
            if any(np.linalg.norm(np.asarray(candidate['body_point'])-other['body_point'])<.05 for other in options):
                continue
            options.append(candidate)
    return sorted(options,key=lambda x:(x['distance_m'],x['yaw_error_rad'] is None,
                                        -x['observed_margin_m']))
