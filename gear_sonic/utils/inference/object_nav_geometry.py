"""Pure RGB-D geometry for the SONIC ObjectNav workflow."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


ROTATION_SPEED = 0.4
FORWARD_SPEED = 0.3
TARGET_STANDOFF_DISTANCE = 0.0
MAX_DIRECT_TRAVEL = 8.0
MIN_ROTATION_ANGLE = math.radians(2.0)
DEPTH_WINDOW_RADIUS = 3
MIN_VALID_DEPTH_MM = 100.0
FRAME_COUNT = 5
POLICY_FRAME_INDEX = 2
MIN_VALID_FRAME_COUNT = 3


class ObjectNavGeometryError(ValueError):
    """Raised when a policy/depth pair cannot produce a safe command."""


def _json_number(value: float) -> float:
    rounded = round(float(value), 3)
    return 0.0 if rounded == 0 else rounded


def _validated_bbox(policy: Mapping[str, Any]) -> tuple[float, float, float, float]:
    if policy.get("action") != "NAVIGATE":
        raise ObjectNavGeometryError("geometry requires a NAVIGATE policy")
    bbox = policy.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ObjectNavGeometryError("NAVIGATE policy requires bbox_2d")
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ObjectNavGeometryError("bbox_2d must contain numbers") from exc
    if not all(math.isfinite(value) and 0 <= value <= 1000 for value in values):
        raise ObjectNavGeometryError("bbox_2d coordinates must be in [0, 1000]")
    x1, y1, x2, y2 = values
    if x1 >= x2 or y1 >= y2:
        raise ObjectNavGeometryError("bbox_2d has invalid corner ordering")
    return x1, y1, x2, y2


def _validated_depth(depth_mm: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or depth.size == 0:
        raise ObjectNavGeometryError("aligned depth must be a non-empty 2D array")
    return depth


def _bbox_projection(
    policy: Mapping[str, Any], depth_shape: tuple[int, ...], fx: float, cx: float
) -> dict[str, Any]:
    x1, y1, x2, y2 = _validated_bbox(policy)
    if len(depth_shape) != 2 or depth_shape[0] <= 0 or depth_shape[1] <= 0:
        raise ObjectNavGeometryError("aligned depth must be a non-empty 2D array")
    if not math.isfinite(float(fx)) or float(fx) <= 0:
        raise ObjectNavGeometryError("camera fx must be positive")
    if not math.isfinite(float(cx)):
        raise ObjectNavGeometryError("camera cx must be finite")

    height, width = depth_shape
    u = int(round(((x1 + x2) / 2.0) * (width - 1) / 1000.0))
    v = int(round(((y1 + y2) / 2.0) * (height - 1) / 1000.0))
    radius = DEPTH_WINDOW_RADIUS
    x_start, x_end = max(0, u - radius), min(width, u + radius + 1)
    y_start, y_end = max(0, v - radius), min(height, v + radius + 1)
    angle = math.atan2(-(u - float(cx)) / float(fx), 1.0)
    return {
        "bbox_normalized": [x1, y1, x2, y2],
        "bbox_center_pixel": [u, v],
        "depth_window_pixel": [x_start, y_start, x_end, y_end],
        "angle_rad": angle,
    }


def measure_object_nav_target(
    policy: Mapping[str, Any], depth_mm: np.ndarray, fx: float, cx: float
) -> dict[str, Any]:
    """Measure the selected bbox centre using one aligned depth frame."""
    depth = _validated_depth(depth_mm)
    projection = _bbox_projection(policy, depth.shape, fx, cx)
    x_start, y_start, x_end, y_end = projection["depth_window_pixel"]
    patch = depth[y_start:y_end, x_start:x_end].astype(np.float64, copy=False)
    valid = patch[np.isfinite(patch) & (patch >= MIN_VALID_DEPTH_MM)]
    if valid.size == 0:
        raise ObjectNavGeometryError("bbox center has no valid aligned depth")

    depth_value_mm = float(np.median(valid))
    z = depth_value_mm / 1000.0
    u = projection["bbox_center_pixel"][0]
    x_camera = (u - float(cx)) * z / float(fx)
    goal_y = -x_camera
    distance = math.hypot(z, goal_y)
    return {
        **projection,
        "valid_depth_samples": int(valid.size),
        "depth_mm": depth_value_mm,
        "goal_x": z,
        "goal_y": goal_y,
        "angle_deg": math.degrees(projection["angle_rad"]),
        "range": distance,
    }


def _commands_from_angle_and_range(
    angle: float,
    distance: float,
    *,
    rotation_speed: float,
    forward_speed: float,
    target_standoff_distance: float,
    max_direct_travel: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    limits = (rotation_speed, forward_speed, max_direct_travel)
    if not all(math.isfinite(value) and value > 0 for value in limits):
        raise ValueError("ObjectNav speeds and maximum travel must be positive")
    if not math.isfinite(target_standoff_distance) or target_standoff_distance < 0:
        raise ValueError("target_standoff_distance must be finite and non-negative")

    travel = max(0.0, distance - target_standoff_distance)
    if travel > max_direct_travel:
        raise ObjectNavGeometryError(
            f"direct travel {travel:.3f}m exceeds {max_direct_travel:.3f}m limit"
        )

    if abs(angle) < MIN_ROTATION_ANGLE:
        rotation_wz = 0.0
        rotation_duration = 0.0
    else:
        rotation_wz = math.copysign(rotation_speed, angle)
        rotation_duration = abs(angle) / rotation_speed
    forward_duration = travel / forward_speed
    commands = {
        "commands": [
            {
                "vx": 0.0,
                "vy": 0.0,
                "wz": _json_number(rotation_wz),
                "duration": _json_number(rotation_duration),
            },
            {
                "vx": _json_number(forward_speed),
                "vy": 0.0,
                "wz": 0.0,
                "duration": _json_number(forward_duration),
            },
        ]
    }
    geometry = {
        "goal_x": _json_number(distance * math.cos(angle)),
        "goal_y": _json_number(distance * math.sin(angle)),
        "angle_rad": _json_number(angle),
        "angle_deg": _json_number(math.degrees(angle)),
        "range": _json_number(distance),
        "travel": _json_number(travel),
    }
    return commands, geometry


def _measurement_json(measurement: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bbox_normalized": [_json_number(value) for value in measurement["bbox_normalized"]],
        "bbox_center_pixel": measurement["bbox_center_pixel"],
        "depth_window_pixel": measurement["depth_window_pixel"],
        "valid_depth_samples": measurement["valid_depth_samples"],
        "depth_mm": _json_number(measurement["depth_mm"]),
        "goal_x": _json_number(measurement["goal_x"]),
        "goal_y": _json_number(measurement["goal_y"]),
        "angle_rad": _json_number(measurement["angle_rad"]),
        "angle_deg": _json_number(measurement["angle_deg"]),
        "range": _json_number(measurement["range"]),
    }


def build_object_nav_commands_from_frames(
    policy: Mapping[str, Any],
    frames: Sequence[tuple[np.ndarray, float, float]],
    *,
    rotation_speed: float = ROTATION_SPEED,
    forward_speed: float = FORWARD_SPEED,
    target_standoff_distance: float = TARGET_STANDOFF_DISTANCE,
    max_direct_travel: float = MAX_DIRECT_TRAVEL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Average five valid per-frame ranges and build rotate/translate commands."""
    if len(frames) != FRAME_COUNT:
        raise ObjectNavGeometryError("ObjectNav requires exactly five depth frames")

    frame_measurements: list[dict[str, Any]] = []
    valid_ranges: list[float] = []
    for frame_index, (depth_mm, fx, cx) in enumerate(frames, start=1):
        try:
            measurement = measure_object_nav_target(policy, depth_mm, fx, cx)
        except ObjectNavGeometryError as exc:
            frame_measurements.append(
                {"frame_index": frame_index, "status": "invalid", "error": str(exc)}
            )
        else:
            valid_ranges.append(measurement["range"])
            frame_measurements.append(
                {"frame_index": frame_index, "status": "valid", **_measurement_json(measurement)}
            )

    if len(valid_ranges) < MIN_VALID_FRAME_COUNT:
        raise ObjectNavGeometryError("at least three of five depth frames must be valid")

    middle_depth, middle_fx, middle_cx = frames[POLICY_FRAME_INDEX]
    direction = _bbox_projection(
        policy, _validated_depth(middle_depth).shape, middle_fx, middle_cx
    )
    mean_range = float(np.mean(valid_ranges))
    commands, motion = _commands_from_angle_and_range(
        direction["angle_rad"],
        mean_range,
        rotation_speed=rotation_speed,
        forward_speed=forward_speed,
        target_standoff_distance=target_standoff_distance,
        max_direct_travel=max_direct_travel,
    )
    geometry = {
        "bbox_normalized": [_json_number(value) for value in direction["bbox_normalized"]],
        "bbox_center_pixel": direction["bbox_center_pixel"],
        "depth_window_pixel": direction["depth_window_pixel"],
        "frame_count": FRAME_COUNT,
        "required_valid_frames": MIN_VALID_FRAME_COUNT,
        "valid_frame_count": len(valid_ranges),
        "minimum_valid_depth_mm": _json_number(MIN_VALID_DEPTH_MM),
        "policy_rgb_frame_index": POLICY_FRAME_INDEX + 1,
        "frame_measurements": frame_measurements,
        "mean_range": _json_number(mean_range),
        **motion,
    }
    return commands, geometry
