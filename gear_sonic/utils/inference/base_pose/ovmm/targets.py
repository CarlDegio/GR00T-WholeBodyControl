"""Associate task instances across frames using masks, RGB-D, and odometry."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gear_sonic.utils.inference.base_pose.servo import (
    TargetGeometry,
    TrackedInstance,
    estimate_target_geometry,
)


class MissingTarget(ValueError):
    pass


@dataclass(frozen=True)
class AlignmentTargets:
    target: TrackedInstance
    yaw_reference: TrackedInstance | None
    target_geometry: TargetGeometry
    target_position_start: np.ndarray
    yaw_reference_position_start: np.ndarray | None
    reason: str | None = None
    reference_mask_source: str = "semantic_instance"
    target_geometry_source: str = "standard_mask"


def _goal_id(value) -> int:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise MissingTarget("Task semantic category ID is unavailable") from exc
    if (
        array.size != 1
        or not np.isfinite(array[0])
        or int(array[0]) < 0
        or int(array[0]) != array[0]
    ):
        raise MissingTarget("Task semantic category ID is unavailable")
    return int(array[0])


def _instances(obs, category: int, *, raw: bool = False) -> list[TrackedInstance]:
    semantic = np.asarray(obs.semantic)
    class_mask = semantic == category
    instance_map = getattr(obs, "instance", None)
    scores = obs.task_observations.get("instance_scores", ())
    raw_masks = obs.task_observations.get("instance_masks_raw") if raw else None
    if raw_masks is not None:
        raw_masks = np.asarray(raw_masks)
        classes = np.asarray(obs.task_observations.get("instance_classes", ()))
        if (
            raw_masks.ndim != 3
            or raw_masks.shape[1:] != semantic.shape
            or classes.shape != (len(raw_masks),)
        ):
            raise MissingTarget("Raw DETIC instances do not match the current frame")
        masks = [
            (i, np.asarray(raw_masks[i], dtype=bool))
            for i in np.flatnonzero(classes == category)
        ]
    elif instance_map is not None:
        instance_map = np.asarray(instance_map)
        if instance_map.shape != semantic.shape:
            raise MissingTarget("Instance map does not match the semantic image")
        masks = [
            (int(i), class_mask & (instance_map == i))
            for i in np.unique(instance_map[class_mask])
            if i >= 0
        ]
    else:
        # GT-semantic debugging may have no predicted instance map. Connected
        # components also avoid merging disjoint objects of the same category.
        count, labels = cv2.connectedComponents(class_mask.astype(np.uint8))
        masks = [(i, labels == i) for i in range(1, count)]
    instances = []
    for instance_id, mask in masks:
        v, u = np.nonzero(mask)
        if not u.size:
            continue
        confidence = (
            float(scores[instance_id]) if 0 <= instance_id < len(scores) else 1.0
        )
        instances.append(
            TrackedInstance(
                instance_id,
                category,
                confidence,
                (
                    float(u.min()),
                    float(v.min()),
                    float(u.max() + 1),
                    float(v.max() + 1),
                ),
                mask,
            )
        )
    return instances


def _in_start_frame(point, start_from_body):
    return start_from_body[:3, :3] @ np.asarray(point) + start_from_body[:3, 3]


class SemanticTargetProvider:
    """Locks to a physical target, never to a per-frame detector index.

    Positions are reconstructed from public RGB-D and GPS/compass. A failed
    association stays missing until reacquisition or the controller timeout;
    it does not silently jump to another nearby task instance.
    """

    def __init__(
        self, *, match_distance_m: float = 0.35, support_distance_m: float = 0.75,
        small_mask_fallback: bool = False, small_mask_min_pixels: int = 30,
    ):
        self.match_distance_m = float(match_distance_m)
        self.support_distance_m = float(support_distance_m)
        self.small_mask_fallback = bool(small_mask_fallback)
        self.small_mask_min_pixels = int(small_mask_min_pixels)
        self.reset()

    def reset(self) -> None:
        self.target_position = None
        self.reference_position = None

    def _geometry(self, snapshot, target, calibration):
        try:
            return estimate_target_geometry(
                snapshot, target.mask, calibration, bbox_xyxy=target.bbox_xyxy
            ), "standard_mask"
        except ValueError:
            if not self.small_mask_fallback:
                raise
        # Only the erosion and sample count relax; depth validity becomes
        # stricter, and a visible source receptacle must still be associated.
        return estimate_target_geometry(
            snapshot, target.mask, calibration, bbox_xyxy=target.bbox_xyxy,
            erode_px=1, min_pixels=self.small_mask_min_pixels, min_valid_ratio=0.50,
        ), "small_mask"

    def _reference(
        self, references, snapshot, calibration, start_from_body, target_position
    ):
        options = []
        depth = snapshot.depth_raw
        for reference in references:
            valid = (
                reference.mask & np.isfinite(depth) & (depth >= 0.1) & (depth <= 3.0)
            )
            v, u = np.nonzero(valid)
            if len(u) < 5:
                continue
            stride = max(1, int(np.ceil(len(u) / 1024)))
            u, v = u[::stride], v[::stride]
            z = depth[v, u]
            points = np.column_stack(
                (
                    (u - calibration.cx) * z / calibration.fx,
                    (v - calibration.cy) * z / calibration.fy,
                    z,
                )
            )
            body_points = calibration.camera_to_body(points)
            start_points = (
                body_points @ start_from_body[:3, :3].T + start_from_body[:3, 3]
            )
            distance = np.linalg.norm(start_points - target_position, axis=1)
            index = int(np.argmin(distance))
            anchor = start_points[index]
            if distance[index] > self.support_distance_m:
                continue
            if (
                self.reference_position is not None
                and np.linalg.norm(anchor - self.reference_position)
                > self.support_distance_m
            ):
                continue
            options.append(
                (float(distance[index]), -reference.confidence, reference, anchor)
            )
        if not options:
            return None, None
        _, _, reference, anchor = min(options, key=lambda item: item[:2])
        # The persistent ID represents the associated physical reference, not
        # DETIC's changing instance index.
        reference = TrackedInstance(
            2,
            reference.class_index,
            reference.confidence,
            reference.bbox_xyxy,
            reference.mask,
        )
        return reference, anchor

    def get_alignment_targets(
        self, obs, stage: str, snapshot, calibration, start_from_body
    ) -> AlignmentTargets:
        if stage != "pick":
            raise ValueError(
                "The first OVMM adapter supports only object-side alignment"
            )
        if (
            obs.semantic is None
            or np.asarray(obs.semantic).shape != snapshot.depth_raw.shape
        ):
            raise MissingTarget("Aligned semantic perception is unavailable")
        target_category = _goal_id(obs.task_observations.get("object_goal"))
        reference_category = _goal_id(obs.task_observations.get("start_recep_goal"))
        targets = _instances(obs, target_category)
        references = _instances(obs, reference_category, raw=True)
        options = []
        errors = []
        for target in targets:
            try:
                geometry, geometry_source = self._geometry(snapshot, target, calibration)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            position = _in_start_frame(geometry.body_xyz_m, start_from_body)
            association_distance = 0.0
            if self.target_position is not None:
                association_distance = float(
                    np.linalg.norm(position - self.target_position)
                )
                if association_distance > self.match_distance_m:
                    continue
            reference, reference_position = self._reference(
                references, snapshot, calibration, start_from_body, position
            )
            if geometry_source == "small_mask" and reference is None:
                errors.append("small target has no associated visible source receptacle")
                continue
            distance = float(np.linalg.norm(np.asarray(geometry.body_xyz_m)[:2]))
            rank = (
                (association_distance, distance, -target.confidence)
                if self.target_position is not None
                else (reference is None, distance, -target.confidence)
            )
            options.append(
                (rank, target, geometry, position, reference, reference_position, geometry_source)
            )
        if not options:
            detail = errors[0] if errors else "no matching visible instance"
            raise MissingTarget("target unavailable: " + detail)
        _, target, geometry, position, reference, reference_position, geometry_source = min(
            options, key=lambda x: x[0]
        )
        self.target_position = position.copy()
        if reference_position is not None:
            self.reference_position = reference_position.copy()
        target = TrackedInstance(
            1, target.class_index, target.confidence, target.bbox_xyxy, target.mask
        )
        return AlignmentTargets(
            target,
            reference,
            geometry,
            position,
            reference_position,
            None if reference is not None else "supporting receptacle not associated",
            (
                obs.task_observations.get("instance_mask_source", "detic_raw")
                if obs.task_observations.get("instance_masks_raw") is not None
                else "semantic_instance"
            ),
            geometry_source,
        )
