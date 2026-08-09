#!/usr/bin/env python3
"""Continuous NavDP trajectory executor with integrated hard safety guards."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any, Literal, Mapping, Sequence

import numpy as np

COMMAND_TYPE = "sonic_navigation_command"
STATUS_TYPE = "sonic_navigation_status"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class NavigationCommand:
    mode: Literal["manual_velocity", "nav_goal", "stop"]
    generation: int
    timestamp: float
    velocity: tuple[float, float, float] | None = None
    goal_base: tuple[float, float] | None = None
    target: str = ""
    target_type: str = ""
    confidence: float = 0.0


def build_navigation_message(
    *,
    mode: str,
    generation: int,
    timestamp: float | None = None,
    velocity: Sequence[float] | None = None,
    goal_base: Sequence[float] | None = None,
    target: str = "",
    target_type: str = "",
    confidence: float = 0.0,
) -> str:
    payload: dict[str, Any] = {
        "type": COMMAND_TYPE,
        "version": 1,
        "generation": int(generation),
        "mode": mode,
        "timestamp": time.time() if timestamp is None else float(timestamp),
    }
    if velocity is not None:
        payload["velocity"] = dict(zip(("vx", "vy", "wz"), map(float, velocity)))
    if goal_base is not None:
        payload["goal_base"] = {"x": float(goal_base[0]), "y": float(goal_base[1])}
        payload.update(
            target=str(target), target_type=str(target_type), confidence=float(confidence)
        )
    return json.dumps(payload)


def decode_navigation_message(message: str | bytes | Mapping[str, Any]) -> NavigationCommand:
    payload = json.loads(message) if isinstance(message, (str, bytes)) else dict(message)
    if payload.get("type") != COMMAND_TYPE or payload.get("version") != 1:
        raise ValueError("unsupported navigation command")
    mode = payload.get("mode")
    if mode not in {"manual_velocity", "nav_goal", "stop"}:
        raise ValueError("invalid navigation mode")
    velocity = payload.get("velocity")
    goal = payload.get("goal_base")
    return NavigationCommand(
        mode=mode,
        generation=int(payload["generation"]),
        timestamp=float(payload["timestamp"]),
        velocity=None
        if velocity is None
        else tuple(float(velocity[key]) for key in ("vx", "vy", "wz")),
        goal_base=None if goal is None else (float(goal["x"]), float(goal["y"])),
        target=str(payload.get("target", "")),
        target_type=str(payload.get("target_type", "")),
        confidence=float(payload.get("confidence", 0.0)),
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
