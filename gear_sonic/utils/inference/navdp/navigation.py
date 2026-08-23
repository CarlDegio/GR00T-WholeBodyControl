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
    """Convert an Agent-relative turn into an absolute source-frame heading."""
    return wrap_angle(float(current_yaw) + float(delta_rad))


@dataclass(frozen=True)
class HeadingControlResult:
    state: str
    angular_velocity_rad_s: float
    target_rad: float
    reference_rad: float
    remaining_rad: float
    reason: str


class HeadingGoalController:
    """Drive a SONIC-yaw heading goal with a coarse/fine speed profile."""

    def __init__(
        self,
        *,
        angular_speed_rad_s: float = 0.4,
        fine_angular_speed_rad_s: float = 0.2,
        slowdown_angle_rad: float = math.radians(20.0),
        tolerance_rad: float = math.radians(5.0),
    ) -> None:
        if not math.isfinite(angular_speed_rad_s) or angular_speed_rad_s <= 0.0:
            raise ValueError("heading angular speed must be finite and positive")
        if (
            not math.isfinite(fine_angular_speed_rad_s)
            or fine_angular_speed_rad_s <= 0.0
            or fine_angular_speed_rad_s > angular_speed_rad_s
        ):
            raise ValueError(
                "fine heading angular speed must be positive and no greater "
                "than the coarse speed"
            )
        if (
            not math.isfinite(slowdown_angle_rad)
            or not math.isfinite(tolerance_rad)
            or tolerance_rad <= 0.0
            or slowdown_angle_rad <= tolerance_rad
            or slowdown_angle_rad > math.pi
        ):
            raise ValueError("heading slowdown and tolerance angles are invalid")
        self.angular_speed_rad_s = float(angular_speed_rad_s)
        self.fine_angular_speed_rad_s = float(fine_angular_speed_rad_s)
        self.slowdown_angle_rad = float(slowdown_angle_rad)
        self.tolerance_rad = float(tolerance_rad)
        self.reference_rad = 0.0
        self.target_rad = 0.0
        self.started_at_s = 0.0
        self.turn_delta_rad = 0.0
        self.turn_direction: str | None = None
        self.angular_velocity_rad_s = 0.0
        self.accumulated_yaw_rad = 0.0
        self._last_yaw_rad = 0.0
        self._goal_speed_limit_rad_s = self.angular_speed_rad_s
        self._goal_max_duration_s: float | None = None
        self._active = False

    def start(
        self,
        *,
        current_yaw: float,
        delta_rad: float,
        turn_direction: str | None = None,
        now: float,
        max_angular_speed_rad_s: float | None = None,
        max_duration_s: float | None = None,
    ) -> None:
        self.reference_rad = wrap_angle(current_yaw)
        if turn_direction not in {None, "left", "right"}:
            raise ValueError("heading turn direction must be left or right")
        wrapped_delta = wrap_angle(delta_rad)
        if turn_direction == "left":
            self.turn_delta_rad = float(delta_rad) % (2.0 * math.pi)
        elif turn_direction == "right":
            self.turn_delta_rad = -((-float(delta_rad)) % (2.0 * math.pi))
        else:
            self.turn_delta_rad = wrapped_delta
        self.turn_direction = turn_direction
        self.target_rad = heading_goal_target(current_yaw, self.turn_delta_rad)
        self.started_at_s = float(now)
        if not math.isfinite(self.started_at_s):
            raise ValueError("heading start time must be finite")
        if max_angular_speed_rad_s is None:
            self._goal_speed_limit_rad_s = self.angular_speed_rad_s
        else:
            speed_limit = float(max_angular_speed_rad_s)
            if not math.isfinite(speed_limit) or speed_limit <= 0.0:
                raise ValueError("heading goal speed limit must be positive")
            self._goal_speed_limit_rad_s = min(
                self.angular_speed_rad_s, speed_limit,
            )
        if max_duration_s is None:
            self._goal_max_duration_s = None
        else:
            duration = float(max_duration_s)
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError("heading goal maximum duration must be positive")
            self._goal_max_duration_s = duration
        self.angular_velocity_rad_s = (
            0.0
            if self.turn_delta_rad == 0.0
            else math.copysign(
                self._goal_speed_limit_rad_s, self.turn_delta_rad,
            )
        )
        self.accumulated_yaw_rad = 0.0
        self._last_yaw_rad = self.reference_rad
        self._active = True

    def update(self, *, current_yaw: float, now: float) -> HeadingControlResult:
        if not self._active:
            raise RuntimeError("heading controller has not been started")
        yaw = wrap_angle(current_yaw)
        self.accumulated_yaw_rad += wrap_angle(yaw - self._last_yaw_rad)
        self._last_yaw_rad = yaw
        elapsed_s = max(0.0, float(now) - self.started_at_s)
        shortest_remaining_rad = wrap_angle(self.target_rad - yaw)
        if abs(shortest_remaining_rad) <= self.tolerance_rad:
            self._active = False
            self.angular_velocity_rad_s = 0.0
            return HeadingControlResult(
                "reached",
                0.0,
                self.target_rad,
                self.reference_rad,
                0.0,
                "heading_sonic_yaw_reached",
            )
        remaining_rad = shortest_remaining_rad
        requested_magnitude = abs(self.turn_delta_rad)
        direction_sign = (
            1.0 if self.turn_direction == "left"
            else -1.0 if self.turn_direction == "right"
            else 0.0
        )
        directed_progress = direction_sign * self.accumulated_yaw_rad
        if (
            direction_sign != 0.0
            and directed_progress < requested_magnitude
        ):
            if direction_sign > 0.0 and remaining_rad < 0.0:
                remaining_rad += 2.0 * math.pi
            elif direction_sign < 0.0 and remaining_rad > 0.0:
                remaining_rad -= 2.0 * math.pi
        remaining_magnitude = abs(remaining_rad)
        if (
            self._goal_max_duration_s is not None
            and elapsed_s >= self._goal_max_duration_s
        ):
            self._active = False
            self.angular_velocity_rad_s = 0.0
            return HeadingControlResult(
                "reached",
                0.0,
                self.target_rad,
                self.reference_rad,
                remaining_rad,
                "heading_adjustment_time_limit",
            )
        speed = (
            min(self.fine_angular_speed_rad_s, self._goal_speed_limit_rad_s)
            if remaining_magnitude <= self.slowdown_angle_rad
            else self._goal_speed_limit_rad_s
        )
        self.angular_velocity_rad_s = math.copysign(speed, remaining_rad)
        return HeadingControlResult(
            "active",
            self.angular_velocity_rad_s,
            self.target_rad,
            self.reference_rad,
            remaining_rad,
            "heading_sonic_yaw_tracking",
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
