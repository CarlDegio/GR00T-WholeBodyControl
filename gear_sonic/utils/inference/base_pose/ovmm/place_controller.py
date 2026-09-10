"""Receptacle-side geometry adapter for the unchanged BasePose servo loop."""

from dataclasses import replace
import math

import numpy as np

from gear_sonic.utils.inference.base_pose.servo import (
    ServoPhase, TargetGeometry, TrackedInstance, YawAlignGeometry, _observation,
)
from .controller import BasePoseSession
from .place_surface import point_image, support_candidates
from .targets import AlignmentTargets, MissingTarget, _goal_id, _instances


class PlaceSurfaceProvider:
    """Associate one measured support anchor in the odometry frame.

    BasePoseSession's provider hook supplies the legacy 'pick' argument. This
    provider's task is explicitly fixed to placement: it reads only
    end_recep_goal, never object_goal or start_recep_goal.
    """

    def __init__(self, minimum_margin_m=.07):
        self.minimum_margin_m = float(minimum_margin_m)
        self.reset()

    def reset(self):
        self.anchor_start = None
        self.reference_normal_start = None
        self.reference_mode = None
        self.last_candidate = None
        self.last_reference_yaw = None
        self.last_reason = None
        self.reference_fallback_reason = None

    def get_alignment_targets(self, obs, _legacy_mode, snapshot, calibration, start):
        self.last_candidate = self.last_reference_yaw = None
        self.last_reason = None
        category = _goal_id(obs.task_observations.get('end_recep_goal'))
        preferred = (None if self.anchor_start is None else
                     start[:3, :3].T @ (self.anchor_start-start[:3, 3]))
        options = []
        for instance in _instances(obs, category):
            for candidate in support_candidates(
                snapshot.depth_raw, calibration, instance.mask,
                minimum_margin_m=self.minimum_margin_m,
                preferred_body=preferred,
            ):
                body = np.asarray(candidate['body_point'])
                if not .35 <= body[0] <= 2.5:
                    continue
                position = start[:3, :3] @ body + start[:3, 3]
                if self.anchor_start is not None:
                    score = np.linalg.norm(position-self.anchor_start)
                    if score > .12:
                        continue
                else:
                    score = candidate['distance_m'] + .15*abs(body[1])
                options.append((float(score), instance, candidate, position))
        if not options:
            raise MissingTarget('no associated visible support patch with sufficient interior margin')
        _, instance, candidate, position = min(options, key=lambda item: item[0])
        body = np.asarray(candidate['body_point'])
        if self.anchor_start is None:
            self.anchor_start = position.copy()
            self.reference_mode = ('surface_contour_normal'
                                   if candidate['yaw_error_rad'] is not None
                                   else 'surface_anchor_bearing')
            if self.reference_mode == 'surface_contour_normal':
                yaw = candidate['yaw_error_rad']
                self.reference_normal_start = start[:2, :2] @ [math.cos(yaw), math.sin(yaw)]
        if self.reference_mode == 'surface_anchor_bearing':
            # Bearing uses the measured current support point, never a cached
            # target point or simulated object position as a control input.
            yaw = math.atan2(body[1], body[0])
        else:
            yaw = candidate['yaw_error_rad']
            if yaw is not None:
                normal = start[:2, :2] @ [math.cos(yaw), math.sin(yaw)]
                difference = math.acos(float(np.clip(np.dot(normal, self.reference_normal_start), -1, 1)))
                if difference > math.radians(20):
                    yaw = None
                    self.last_reason = 'fresh support contour changed orientation'
            else:
                self.last_reason = 'associated support visible but selected contour unavailable'
            if yaw is None:
                # A horizontal patch can remain reliably visible while its
                # front contour leaves view. Use its freshly measured anchor
                # bearing and explicitly abandon the old edge reference.
                self.reference_fallback_reason = self.last_reason
                self.reference_mode = 'surface_anchor_bearing'
                self.last_reason = None
                yaw = math.atan2(body[1], body[0])
        self.last_candidate = candidate
        self.last_reference_yaw = yaw
        self.last_candidate = dict(candidate, reference_mode=self.reference_mode,
                                   reference_fallback_reason=self.reference_fallback_reason,
                                   anchor_association_error_m=float(np.linalg.norm(position-self.anchor_start)))
        # Give the servo the actual support anchor's image position, rather
        # than the bounding-box center of a large table or a cabinet front.
        points = point_image(snapshot.depth_raw, calibration)
        anchor_mask = instance.mask & (np.linalg.norm(points-body, axis=-1) <= .04)
        v, u = np.nonzero(anchor_mask)
        if not len(u):
            raise MissingTarget('measured support anchor is absent from the current instance')
        target = TrackedInstance(1, category, instance.confidence,
            (float(u.min()), float(v.min()), float(u.max()+1), float(v.max()+1)), anchor_mask)
        reference = TrackedInstance(2, category, instance.confidence, instance.bbox_xyxy, instance.mask)
        pixel = tuple(map(float, candidate['pixel']))
        geometry = TargetGeometry(
            forward_m=float(body[0]), right_m=float(-body[1]),
            body_xyz_m=tuple(map(float, body)),
            valid_depth_pixels=candidate['support_pixels'], valid_ratio=1.0,
            median_depth_m=float(snapshot.depth_raw[int(pixel[1]), int(pixel[0])]),
            lateral_anchor_px=pixel,
        )
        reference_body = (np.asarray(candidate['reference_center_body'])
                          if self.reference_mode == 'surface_contour_normal'
                          and candidate.get('reference_center_body') is not None else body)
        reference_position = start[:3, :3] @ reference_body + start[:3, 3]
        return AlignmentTargets(
            target, reference, geometry, position, reference_position,
            reason=self.last_reason, reference_mask_source='visible_end_receptacle_instance',
            target_geometry_source='visible_horizontal_support_interior',
        )


class BasePosePlaceSession(BasePoseSession):
    """Change target/reference geometry while preserving the base control loop."""

    def __init__(self, config, limits, *, use_opencv_camera_pose=False, minimum_margin_m=.07):
        super().__init__(config, limits, use_opencv_camera_pose=use_opencv_camera_pose)
        self.targets = PlaceSurfaceProvider(minimum_margin_m)

    def reset(self, start_frame_id=None):
        self._place_reference_source = None
        self.placement_source_switches = 0
        super().reset(start_frame_id)

    def _result(self, xyt, steps, started):
        result = super()._result(xyt, steps, started)
        # Both references here are metric RGB-D geometry. Keep the common
        # reference-angle field without presenting it as an image-line angle.
        self.last_result = replace(result, diagnostics={**result.diagnostics, 'image_yaw_error_rad': None})
        return self.last_result

    def observe_geometry(self, snapshot, targets, calibration, start_from_body, *,
                         include_yaw_align_geometry=True, update_reference=False):
        observation = _observation(
            snapshot, targets.target, targets.yaw_reference, calibration,
            include_yaw_align_geometry=False, target_geometry=targets.target_geometry,
        )
        candidate = self.targets.last_candidate
        yaw = self.targets.last_reference_yaw if include_yaw_align_geometry else None
        geometry = None
        if yaw is not None:
            endpoints = (candidate.get('reference_endpoints_px')
                         if self.targets.reference_mode == 'surface_contour_normal' else None)
            endpoints = None if endpoints is None else tuple(tuple(map(float, p)) for p in endpoints)
            geometry = YawAlignGeometry(
                yaw_error_rad=yaw,
                line_length_px=(0.0 if endpoints is None else
                                float(np.linalg.norm(np.asarray(endpoints[1])-endpoints[0]))),
                valid_depth_samples=candidate['support_pixels'],
                line_center_px=tuple(map(float, candidate['pixel'])),
                line_endpoints_px=endpoints,
            )
        source = (self.targets.reference_mode if geometry is not None else
                  'distance_gated_approach' if not include_yaw_align_geometry else 'unavailable')
        if update_reference and geometry is not None:
            if self._place_reference_source is not None and self._place_reference_source != source:
                self.controller.reset(snapshot.timestamp, initial_phase=ServoPhase.FORWARD_APPROACH)
                self.actions.reset()
                self.placement_source_switches += 1
            self._place_reference_source = source
        self.geometry_diagnostics = dict(
            yaw_geometry_source=source,
            target_geometry_source=targets.target_geometry_source,
            reference_source_switches=self.placement_source_switches,
            support_geometry=candidate,
        )
        return replace(observation, yaw_align_geometry=geometry,
                       yaw_align_geometry_error=self.targets.last_reason)
