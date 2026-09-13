"""HTTP adapter for uni-lavira-code's Unitree G1 iPlanner server.

The gateway supplies aligned BGR and metric depth. iPlanner returns local
waypoints with x forward and y left for LaViRA's pure-pursuit controller.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np


def encode_frames(rgb: np.ndarray, depth_m: np.ndarray) -> tuple[bytes, bytes]:
    import cv2

    rgb = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("iPlanner requires a uint8 BGR image")
    if depth.ndim != 2 or depth.shape != rgb.shape[:2] or not depth.size:
        raise ValueError("iPlanner requires depth aligned to the RGB image")
    depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0)
    # The upstream server divides uint16 PNG samples by 10000. Saturation
    # prevents distant surfaces from wrapping around into near obstacles.
    depth_units = np.rint(np.clip(depth, 0.0, 6.5535) * 10000.0).astype(np.uint16)
    ok_rgb, rgb_png = cv2.imencode(".png", rgb)
    ok_depth, depth_png = cv2.imencode(".png", depth_units)
    if not ok_rgb or not ok_depth:
        raise RuntimeError("failed to encode iPlanner RGB-D")
    return rgb_png.tobytes(), depth_png.tobytes()


def reset_iplanner(server: str, info: Mapping[str, Any]) -> None:
    import requests

    intrinsic = [
        [float(info["fx"]), 0.0, float(info["cx"])],
        [0.0, float(info["fy"]), float(info["cy"])],
        [0.0, 0.0, 1.0],
    ]
    response = requests.post(
        f"{server.rstrip('/')}/navigator_reset",
        json={"intrinsic": intrinsic, "stop_threshold": 0.1, "batch_size": 1},
        timeout=30.0,
    )
    response.raise_for_status()
    if response.json().get("algo") != "iplanner":
        raise ValueError("expected a LaViRA iPlanner server at the navigation endpoint")


def request_trajectory(
    server: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    goal: tuple[float, float],
    *,
    timeout: float = 10.0,
) -> np.ndarray:
    import requests

    local_goal = np.asarray(goal, dtype=np.float64)
    if local_goal.shape != (2,) or not np.isfinite(local_goal).all():
        raise ValueError("iPlanner requires a finite two-dimensional local goal")
    if not np.any(np.isfinite(depth) & (np.asarray(depth) > 0.0)):
        raise ValueError("iPlanner requires at least one valid positive depth sample")
    # Match LaViRA's process_goal(), including goals behind the robot.
    local_goal = np.clip(local_goal, -5.0, 5.0)
    rgb_png, depth_png = encode_frames(rgb, depth)
    response = requests.post(
        f"{server.rstrip('/')}/pointgoal_step",
        files={
            "image": ("rgb.png", rgb_png, "image/png"),
            "depth": ("depth.png", depth_png, "image/png"),
        },
        data={"goal_data": json.dumps({
            "goal_x": [float(local_goal[0])], "goal_y": [float(local_goal[1])],
        })},
        timeout=timeout,
    )
    response.raise_for_status()
    trajectory = np.asarray(response.json()["trajectory"], dtype=np.float32)
    if (trajectory.ndim != 3 or trajectory.shape[0] != 1
            or trajectory.shape[2] != 3 or trajectory.shape[1] < 2
            or not np.isfinite(trajectory).all()):
        raise ValueError("invalid iPlanner trajectory: expected finite [1, N>=2, 3]")
    return trajectory[0, :, :2].copy()
