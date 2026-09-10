"""Fresh RGB-D references with optional physical-edge association."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

from ..servo import YawAlignGeometry


def _half_turn(angle):
    return (float(angle) + math.pi / 2) % math.pi - math.pi / 2


@dataclass(frozen=True)
class _MeasuredEdge:
    candidate: object
    center: np.ndarray
    direction: np.ndarray
    normal: float


def _measure_edge(candidate, snapshot, calibration, start):
    raster = np.zeros(snapshot.depth_raw.shape, np.uint8)
    a, b = candidate.line_endpoints_px
    cv2.line(raster, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), 1, 1)
    v, u = np.nonzero(raster)
    depth = snapshot.depth_raw[v, u]
    valid = np.isfinite(depth) & (depth >= .1) & (depth <= 3)
    u, v, depth = u[valid], v[valid], depth[valid]
    if len(depth) < 20:
        return None
    cloud = calibration.camera_to_body(np.column_stack((
        (u-calibration.cx)*depth/calibration.fx,
        (v-calibration.cy)*depth/calibration.fy, depth,
    )))
    center = np.median(cloud, axis=0)
    _, _, axes = np.linalg.svd(cloud-center, full_matrices=False)
    direction = axes[0]
    projected = (cloud-center)@direction
    span = float(np.ptp(np.quantile(projected, [.05, .95])))
    residual = float(np.quantile(np.linalg.norm(
        cloud-center-np.outer(projected, direction), axis=1), .9))
    # Depth discontinuities and nearly vertical lines cannot anchor a physical
    # horizontal reference edge. They do not silently establish yaw history.
    if span < .08 or residual > .015 or abs(direction[2]) > .20:
        return None
    center = start[:3, :3]@center + start[:3, 3]
    direction = start[:3, :3]@direction
    normal = _half_turn(math.atan2(direction[0], -direction[1]))
    return _MeasuredEdge(candidate, center, direction, normal)


class ReferenceEdgeTracker:
    """Keep the image-angle controller on a freshly observed reference direction.

    The three-dimensional line is only used to associate an edge. It never
    supplies a stale yaw command when that line is absent from the new frame.
    """

    def __init__(self, parallel_max_offset_m=.40):
        self.parallel_max_offset_m = float(parallel_max_offset_m)
        self.reset()

    def reset(self):
        self.anchor = None

    def resolve(self, observation, snapshot, calibration, start, *, update=True):
        geometry = observation.yaw_align_geometry
        diagnostic = dict(source='rgb_edge' if geometry is not None else 'unavailable')
        if geometry is None:
            return None, diagnostic
        selected = next((item for item in geometry.candidate_depth_stats if item.selected), None)
        measured = None if selected is None else _measure_edge(selected, snapshot, calibration, start)
        if self.anchor is None:
            if measured is not None and update:
                self.anchor = measured
            return geometry, diagnostic
        if measured is None:
            # Without a qualified new 3-D line there is no evidence of an
            # orientation jump; preserve the original image-edge behavior.
            return geometry, diagnostic
        angle = None if measured is None else abs(_half_turn(measured.normal-self.anchor.normal))
        if angle is not None:
            diagnostic['selected_start_angle_change_deg'] = math.degrees(angle)
        if angle is not None and angle <= math.radians(25):
            if update:
                self.anchor = measured
            return geometry, diagnostic
        matches = []
        for candidate in geometry.candidate_depth_stats:
            fresh = _measure_edge(candidate, snapshot, calibration, start)
            if fresh is None:
                continue
            angle = abs(_half_turn(fresh.normal-self.anchor.normal))
            displacement = fresh.center-self.anchor.center
            offset = float(np.linalg.norm(
                displacement-self.anchor.direction*np.dot(displacement,self.anchor.direction)))
            # The closest edge can leave view while a parallel edge of the
            # same associated receptacle remains visible. Its current image
            # angle provides a measured reference with the same orientation.
            if (angle > math.radians(12) or offset > self.parallel_max_offset_m
                    or abs(displacement[2]) > .08 or np.linalg.norm(displacement) > .75):
                continue
            matches.append(((offset + .1*angle, candidate.median_depth_m, -candidate.line_length_px), fresh, offset))
        if not matches:
            diagnostic.update(source='unavailable', reason='physical reference edge changed without a fresh match')
            return None, diagnostic
        _, fresh, offset = min(matches, key=lambda item:item[0])
        a, b = np.asarray(fresh.candidate.line_endpoints_px)
        if tuple(b) < tuple(a):
            a, b = b, a
        delta = b-a
        geometry = YawAlignGeometry(
            yaw_error_rad=math.atan2(-float(delta[1]), float(delta[0])),
            line_length_px=float(np.linalg.norm(delta)),
            valid_depth_samples=fresh.candidate.valid_depth_samples,
            line_center_px=tuple(((a+b)/2).tolist()),
            line_endpoints_px=(tuple(a.tolist()), tuple(b.tolist())),
            candidate_depth_stats=tuple(replace(item, selected=item is fresh.candidate)
                                        for item in geometry.candidate_depth_stats),
        )
        if update:
            self.anchor = fresh
        diagnostic.update(source='tracked_rgb_edge' if offset <= .08 else 'parallel_rgb_edge',
                          reference_line_offset_m=offset, reference_start_normal_deg=math.degrees(fresh.normal),
                          reference_start_center_m=fresh.center.tolist())
        return geometry, diagnostic


def target_bearing_reference(targets):
    """A measured object-bearing reference, explicitly without an image line."""
    target = targets.target_geometry
    if target.forward_m <= 0 or not np.isfinite([target.forward_m, target.right_m]).all():
        return None
    center = target.lateral_anchor_px
    if center is None:
        a, b, c, d = targets.target.bbox_xyxy
        center = ((a+c)/2, (b+d)/2)
    return YawAlignGeometry(
        yaw_error_rad=math.atan2(-target.right_m, target.forward_m),
        line_length_px=0.0,
        valid_depth_samples=target.valid_depth_pixels,
        line_center_px=center,
        line_endpoints_px=None,
    )
