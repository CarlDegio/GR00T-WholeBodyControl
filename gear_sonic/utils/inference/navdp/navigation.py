#!/usr/bin/env python3
"""Continuous NavDP trajectory executor with integrated hard safety guards."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from gear_sonic.runtime.protocol import (
    COMMAND_TYPE,
    STATUS_TYPE,
    NavigationCommand,
    build_navigation_message,
    decode_navigation_message,
)

__all__ = [
    "COMMAND_TYPE",
    "STATUS_TYPE",
    "NavigationCommand",
    "build_navigation_message",
    "decode_navigation_message",
    "Pose2D",
    "HeadingControlResult",
    "HeadingGoalController",
    "heading_goal_target",
    "wrap_angle",
]


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi] while preserving finite validation."""
    value = float(angle_rad)
    if not math.isfinite(value):
        raise ValueError("heading angle must be finite")
    return math.remainder(value, 2.0 * math.pi)


def heading_goal_target(current_yaw: float, delta_rad: float) -> float:
    """Convert an Agent-relative turn into an absolute Fast-LIO heading."""
    return wrap_angle(float(current_yaw) + float(delta_rad))


@dataclass(frozen=True)
class HeadingControlResult:
    state: str
    angular_velocity_rad_s: float
    target_rad: float
    reference_rad: float
    error_rad: float
    reason: str


class HeadingGoalController:
    """Deterministic Fast-LIO heading loop used instead of the X-NavDP network."""

    def __init__(
        self,
        *,
        angular_speed_rad_s: float = 0.4,
        tolerance_rad: float = math.radians(3.0),
        stable_frames: int = 3,
        timeout_s: float = 12.0,
    ) -> None:
        if not math.isfinite(angular_speed_rad_s) or angular_speed_rad_s <= 0.0:
            raise ValueError("heading angular speed must be finite and positive")
        if not math.isfinite(tolerance_rad) or tolerance_rad <= 0.0:
            raise ValueError("heading tolerance must be finite and positive")
        if int(stable_frames) <= 0:
            raise ValueError("heading stable_frames must be positive")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("heading timeout must be finite and positive")
        self.angular_speed_rad_s = float(angular_speed_rad_s)
        self.tolerance_rad = float(tolerance_rad)
        self.stable_frames = int(stable_frames)
        self.timeout_s = float(timeout_s)
        self.reference_rad = 0.0
        self.target_rad = 0.0
        self.started_at_s = 0.0
        self._stable_count = 0
        self._active = False

    def start(self, *, current_yaw: float, delta_rad: float, now: float) -> None:
        self.reference_rad = wrap_angle(current_yaw)
        self.target_rad = heading_goal_target(current_yaw, delta_rad)
        self.started_at_s = float(now)
        if not math.isfinite(self.started_at_s):
            raise ValueError("heading start time must be finite")
        self._stable_count = 0
        self._active = True

    def update(self, *, current_yaw: float, now: float) -> HeadingControlResult:
        if not self._active:
            raise RuntimeError("heading controller has not been started")
        error = wrap_angle(self.target_rad - float(current_yaw))
        if float(now) - self.started_at_s > self.timeout_s:
            self._active = False
            return HeadingControlResult(
                "failed", 0.0, self.target_rad, self.reference_rad, error,
                "heading_timeout",
            )
        if abs(error) <= self.tolerance_rad:
            self._stable_count += 1
            velocity = 0.0
        else:
            self._stable_count = 0
            velocity = math.copysign(self.angular_speed_rad_s, error)
        if self._stable_count >= self.stable_frames:
            self._active = False
            return HeadingControlResult(
                "reached", 0.0, self.target_rad, self.reference_rad, error,
                "heading_stable",
            )
        return HeadingControlResult(
            "active", velocity, self.target_rad, self.reference_rad, error,
            "heading_tracking",
        )


def base_goal_to_world(goal: Sequence[float], pose: Pose2D) -> tuple[float, float]:
    x, y = map(float, goal)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return pose.x + c * x - s * y, pose.y + s * x + c * y


def local_goal_from_world(goal: Sequence[float], pose: Pose2D) -> tuple[float, float]:
    dx, dy = float(goal[0]) - pose.x, float(goal[1]) - pose.y
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return c * dx + s * dy, -s * dx + c * dy


def local_trajectory_to_world(trajectory: np.ndarray, pose: Pose2D) -> np.ndarray:
    """Transform a robot-local x-forward/y-left trajectory into odometry XY."""
    points = np.asarray(trajectory, dtype=np.float32).reshape(-1, 2)
    if not len(points):
        return points.copy()
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ rotation.T + np.array([pose.x, pose.y], dtype=np.float32)


def closest_timestamped_pose(
    history: Sequence[tuple[float, Pose2D]], timestamp: float
) -> Pose2D | None:
    """Return the Fast-LIO pose closest to a camera capture timestamp."""
    if not history:
        return None
    return min(history, key=lambda sample: abs(sample[0] - float(timestamp)))[1]


def update_slam_map(
    existing_xy: np.ndarray,
    incoming_xy: np.ndarray,
    *,
    center_xy: Sequence[float],
    voxel_size_m: float = 0.08,
    retain_radius_m: float = 20.0,
    max_points: int = 120_000,
) -> np.ndarray:
    """Merge absolute XY samples into a bounded, voxelized display-only map."""
    old = np.asarray(existing_xy, dtype=np.float32).reshape(-1, 2)
    new = np.asarray(incoming_xy, dtype=np.float32).reshape(-1, 2)
    combined = np.concatenate((new, old), axis=0)
    if not len(combined):
        return combined
    center = np.asarray(center_xy, dtype=np.float32)
    keep = np.isfinite(combined).all(axis=1)
    keep &= np.linalg.norm(combined - center, axis=1) <= float(retain_radius_m)
    combined = combined[keep]
    if not len(combined):
        return combined
    voxel = np.floor(combined / float(voxel_size_m)).astype(np.int64)
    _, indices = np.unique(voxel, axis=0, return_index=True)
    result = combined[np.sort(indices)]
    if len(result) > max_points:
        result = result[-max_points:]
    return result.astype(np.float32, copy=False)
