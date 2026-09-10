"""Public-observation target memory and a bounded Stretch head tilt sweep."""

from __future__ import annotations

import math
import numpy as np

from .observation import adapt_observation, camera_pose_in_start_frame
from .targets import MissingTarget, SemanticTargetProvider
from ..types import BasePoseCameraError


def tilt_sweep(current_deg, minimum_deg, maximum_deg, step_deg):
    """Look down to the lower bound, then up to the upper bound, once."""
    values = (current_deg, minimum_deg, maximum_deg, step_deg)
    if (
        not all(math.isfinite(x) for x in values)
        or step_deg <= 0
        or minimum_deg >= maximum_deg
    ):
        raise ValueError("Invalid finite tilt sweep range or step")
    current = float(np.clip(current_deg, minimum_deg, maximum_deg))
    result = []
    while current > minimum_deg + 1e-8:
        current = max(minimum_deg, current - step_deg)
        result.append(current)
    while current < maximum_deg - 1e-8:
        current = min(maximum_deg, current + step_deg)
        result.append(current)
    return result


class TargetMemory:
    def __init__(self, config, *, use_opencv_camera_pose=False):
        self.config = config
        self.use_opencv_camera_pose = use_opencv_camera_pose
        self.provider = SemanticTargetProvider(
            match_distance_m=config.instance_match_distance_m,
            support_distance_m=config.support_match_distance_m,
        )
        self.reset()

    def reset(self):
        self.provider.reset()
        self.position = None
        self.reference_position = None
        self.last_seen_frame = None
        self.last_checked_frame = None
        self.goal = None
        self.confidence = None
        self.source = None

    @staticmethod
    def goal_key(obs):
        task = obs.task_observations
        return (
            str(task.get("object_name", task.get("object_goal"))),
            str(task.get("start_recep_name", task.get("start_recep_goal"))),
        )

    def available(self, frame_id):
        return (
            self.position is not None
            and self.last_seen_frame is not None
            and 0
            <= frame_id - self.last_seen_frame
            <= self.config.target_memory_max_age_frames
        )

    def remember(self, selection, obs, frame_id, *, source):
        self.position = selection.target_position_start.copy()
        self.reference_position = (
            None
            if selection.yaw_reference_position_start is None
            else selection.yaw_reference_position_start.copy()
        )
        self.last_seen_frame = int(frame_id)
        self.goal = self.goal_key(obs)
        self.confidence = float(selection.target.confidence)
        self.source = source
        self.provider.target_position = self.position.copy()
        self.provider.reference_position = self.reference_position

    def observe_navigation(self, obs, frame_id):
        if self.last_checked_frame is not None and frame_id <= self.last_checked_frame:
            return
        if self.goal != self.goal_key(obs) or not self.available(frame_id):
            self.reset()
        self.last_checked_frame = int(frame_id)
        try:
            snapshot, calibration, start = adapt_observation(
                obs, timestamp=0.0, use_opencv_camera_pose=self.use_opencv_camera_pose
            )
            selection = self.provider.get_alignment_targets(
                obs, "pick", snapshot, calibration, start
            )
        except (MissingTarget, BasePoseCameraError, ValueError):
            return
        self.remember(selection, obs, frame_id, source="NAV_TO_OBJ")

    def seed(self, provider, obs, frame_id):
        if self.available(frame_id) and self.goal == self.goal_key(obs):
            provider.target_position = self.position.copy()
            provider.reference_position = (
                None
                if self.reference_position is None
                else self.reference_position.copy()
            )
            return True
        return False

    def guided_tilt(self, obs, frame_id):
        """Project the remembered point and aim vertically; keep pan at zero."""
        diagnostic = self.diagnostics(frame_id)
        if not self.available(frame_id) or self.goal != self.goal_key(obs):
            return None, {
                **diagnostic,
                "guidance_reason": "no recent matching target memory",
            }
        pose = camera_pose_in_start_frame(
            obs.camera_pose, use_opencv_camera_pose=self.use_opencv_camera_pose
        )
        point = pose[:3, :3].T @ (self.position - pose[:3, 3])
        if point[2] <= 1e-6 or not np.isfinite(point).all():
            return None, {
                **diagnostic,
                "guidance_reason": "remembered target behind camera",
            }
        k = np.asarray(obs.camera_K, dtype=float)
        uv = [
            float(k[0, 0] * point[0] / point[2] + k[0, 2]),
            float(k[1, 1] * point[1] / point[2] + k[1, 2]),
        ]
        correction = math.atan2(float(point[1]), float(np.hypot(point[0], point[2])))
        tilt = float(
            np.clip(
                float(obs.joint[9]) - correction,
                math.radians(self.config.acquisition_tilt_min_deg),
                math.radians(self.config.acquisition_tilt_max_deg),
            )
        )
        return tilt, {
            **diagnostic,
            "predicted_target_uv": uv,
            "guided_tilt_deg": math.degrees(tilt),
            "guidance_reason": "public RGB-D target reprojected into current camera",
        }

    def diagnostics(self, frame_id):
        return {
            "memory_available": self.available(frame_id),
            "memory_last_seen_frame": self.last_seen_frame,
            "memory_age_frames": (
                None
                if self.last_seen_frame is None
                else frame_id - self.last_seen_frame
            ),
            "memory_position_start": (
                None if self.position is None else self.position.tolist()
            ),
            "memory_confidence": self.confidence,
            "memory_source": self.source,
        }
