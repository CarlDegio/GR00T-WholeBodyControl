#!/usr/bin/env python3
"""Continuous NavDP trajectory executor with integrated hard safety guards."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gear_sonic.runtime.protocol import (
    COMMAND_TYPE,
    STATUS_TYPE,
    NavigationCommand,
    build_navigation_message,
    decode_navigation_message,
)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


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
