"""Head-camera multimodal base-pose adjustment policy for Unitree G1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Literal, Mapping, Sequence

import cv2
import msgpack
import numpy as np
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.scripts.run_depth_camera_viewer import colorize_depth


BasePoseMode = Literal["rgb", "rgbd", "rgb_depth_query"]
BASE_POSE_MODES = {"rgb", "rgbd", "rgb_depth_query"}
BasePoseVisionBackend = Literal["codex", "qwenvl"]
BASE_POSE_VISION_BACKENDS = {"codex", "qwenvl"}
DEFAULT_QWENVL_PLUS_MODEL = "qwen3-vl-plus"
DEFAULT_QWENVL_BASE_URL = (
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
ALLOWED_ACTIONS = {
    "ROTATE_LEFT",
    "ROTATE_RIGHT",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
}
NON_ADJUST_STATUSES = {"READY", "UNSURE", "UNSAFE"}
HORIZONTAL_POSITIONS = {"FAR_LEFT", "LEFT", "CENTERED", "RIGHT", "FAR_RIGHT", "UNKNOWN"}
DISTANCE_ESTIMATES = {"TOO_CLOSE", "SUITABLE", "TOO_FAR", "UNKNOWN"}
ORIENTATION_ESTIMATES = {"TURNED_LEFT", "ALIGNED", "TURNED_RIGHT", "UNKNOWN"}
TOP_LEVEL_KEYS = {
    "status",
    "task_interpretation",
    "current_alignment",
    "desired_final_pose",
    "command_sequence",
    "expected_result",
    "confidence",
    "limitations",
}
TASK_INTERPRETATION_KEYS = {
    "primary_target",
    "secondary_targets",
    "manipulation_anchor",
    "interaction_direction",
    "selection_reason",
}
CURRENT_ALIGNMENT_KEYS = {
    "horizontal_position",
    "distance_estimate",
    "orientation_estimate",
    "theta",
}
DESIRED_FINAL_POSE_KEYS = {
    "target_alignment",
    "target_distance",
    "target_orientation",
}
COMMAND_KEYS = {"step", "action", "value", "unit", "purpose"}
HTTP_PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
ALL_PROXY_KEYS = ("all_proxy", "ALL_PROXY")


BASE_POSE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"enum": ["READY", "ADJUST", "UNSURE", "UNSAFE"]},
        "task_interpretation": {
            "type": "object",
            "properties": {
                "primary_target": {"type": "string"},
                "secondary_targets": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "manipulation_anchor": {"type": "string"},
                "interaction_direction": {"type": "string"},
                "selection_reason": {"type": "string"},
            },
            "required": sorted(TASK_INTERPRETATION_KEYS),
            "additionalProperties": False,
        },
        "current_alignment": {
            "type": "object",
            "properties": {
                "horizontal_position": {
                    "enum": [
                        "FAR_LEFT",
                        "LEFT",
                        "CENTERED",
                        "RIGHT",
                        "FAR_RIGHT",
                        "UNKNOWN",
                    ]
                },
                "distance_estimate": {
                    "enum": ["TOO_CLOSE", "SUITABLE", "TOO_FAR", "UNKNOWN"]
                },
                "orientation_estimate": {
                    "enum": [
                        "TURNED_LEFT",
                        "ALIGNED",
                        "TURNED_RIGHT",
                        "UNKNOWN",
                    ]
                },
                "theta": {"type": "number", "minimum": 0},
            },
            "required": sorted(CURRENT_ALIGNMENT_KEYS),
            "additionalProperties": False,
        },
        "desired_final_pose": {
            "type": "object",
            "properties": {
                "target_alignment": {"type": "string"},
                "target_distance": {"type": "string"},
                "target_orientation": {"type": "string"},
            },
            "required": sorted(DESIRED_FINAL_POSE_KEYS),
            "additionalProperties": False,
        },
        "command_sequence": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer", "minimum": 1},
                    "action": {"enum": sorted(ALLOWED_ACTIONS)},
                    "value": {"type": "number", "exclusiveMinimum": 0},
                    "unit": {"enum": ["degrees", "meters"]},
                    "purpose": {"type": "string"},
                },
                "required": sorted(COMMAND_KEYS),
                "additionalProperties": False,
            },
        },
        "expected_result": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "limitations": {"type": "string"},
    },
    "required": sorted(TOP_LEVEL_KEYS),
    "additionalProperties": False,
}


DEPTH_QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_interpretation": {
            "type": "object",
            "properties": {
                "primary_target": {"type": "string"},
                "secondary_targets": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "manipulation_anchors": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "selection_reason": {"type": "string"},
            },
            "required": [
                "primary_target",
                "secondary_targets",
                "manipulation_anchors",
                "selection_reason",
            ],
            "additionalProperties": False,
        },
        "depth_queries": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "query_id": {"type": "string"},
                    "task_role": {
                        "enum": [
                            "PRIMARY_TARGET",
                            "DESTINATION",
                            "COMBINED_WORKSPACE",
                        ]
                    },
                    "target_or_anchor": {"type": "string"},
                    "bbox_2d": {
                        "type": "array",
                        "items": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1000,
                        },
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "sampling_reason": {"type": "string"},
                },
                "required": [
                    "query_id",
                    "task_role",
                    "target_or_anchor",
                    "bbox_2d",
                    "sampling_reason",
                ],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "limitations": {"type": "string"},
    },
    "required": ["task_interpretation", "depth_queries", "confidence", "limitations"],
    "additionalProperties": False,
}


class BasePoseCameraError(RuntimeError):
    """Raised when a head-camera observation is unavailable or malformed."""


class BasePoseValidationError(ValueError):
    """Raised when a model plan cannot be executed without modification."""


@dataclass(frozen=True)
class AlignedRGBDSnapshot:
    rgb: np.ndarray
    depth_raw: np.ndarray | None
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m: float | None
    depth_aligned_to: str | None
    depth_source: str | None
    timestamp: float


@dataclass
class BasePoseConfig:
    task: str
    mode: BasePoseMode = "rgb"
    vision_backend: BasePoseVisionBackend = "codex"
    model: str = "gpt-5.6-sol"
    qwenvl_model: str = DEFAULT_QWENVL_PLUS_MODEL
    qwenvl_base_url: str = DEFAULT_QWENVL_BASE_URL
    qwenvl_thinking_budget: int = 500
    reasoning_effort: str = "max"
    codex_fast: bool = True
    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_timeout_ms: int = 15000
    camera_stream: str = "ego_view"
    camera_height_m: float = 1.2
    camera_pitch_deg: float = -47.6
    vertical_fov_deg: float = 55.2
    camera_forward_offset_m: float = 0.0
    camera_lateral_offset_m: float = 0.0
    depth_visual_max_m: float = 3.0
    codex_timeout_seconds: float = 600.0
    output_root: str = "outputs/base_pose_adjustment"


@dataclass(frozen=True)
class BasePoseResult:
    plan: dict[str, Any]
    output_dir: str
    timing_s: dict[str, float]


def _atomic_write_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}_", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: Any) -> None:
    contents = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_bytes(path, contents.encode("utf-8"))


def _write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BasePoseValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BasePoseValidationError(f"{field} must be finite")
    return result


def _require_exact_keys(value: Any, keys: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise BasePoseValidationError(f"{field} has an invalid object schema")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise BasePoseValidationError(f"{field} must be a string")
    return value


def validate_base_pose_plan(
    plan: Any,
    *,
    min_rotation_deg: float = 2.0,
    max_rotation_deg: float = 90.0,
    min_translation_m: float = 0.10,
    max_translation_m: float = 1.50,
    max_steps: int = 8,
    max_total_rotation_deg: float = 180.0,
    max_total_translation_m: float = 3.0,
) -> dict[str, Any]:
    """Validate the exact model plan without clipping or rewriting it."""
    payload = _require_exact_keys(plan, TOP_LEVEL_KEYS, "plan")
    task = _require_exact_keys(
        payload["task_interpretation"],
        TASK_INTERPRETATION_KEYS,
        "task_interpretation",
    )
    for key in TASK_INTERPRETATION_KEYS - {"secondary_targets"}:
        _require_string(task[key], f"task_interpretation.{key}")
    secondary = task["secondary_targets"]
    if not isinstance(secondary, list) or not all(
        isinstance(item, str) for item in secondary
    ):
        raise BasePoseValidationError("secondary_targets must be an array of strings")
    alignment = _require_exact_keys(
        payload["current_alignment"], CURRENT_ALIGNMENT_KEYS, "current_alignment"
    )
    if alignment["horizontal_position"] not in HORIZONTAL_POSITIONS:
        raise BasePoseValidationError("horizontal_position is invalid")
    if alignment["distance_estimate"] not in DISTANCE_ESTIMATES:
        raise BasePoseValidationError("distance_estimate is invalid")
    if alignment["orientation_estimate"] not in ORIENTATION_ESTIMATES:
        raise BasePoseValidationError("orientation_estimate is invalid")
    theta = _finite_number(alignment["theta"], "current_alignment.theta")
    if theta < 0.0:
        raise BasePoseValidationError("current_alignment.theta must be non-negative")
    if alignment["orientation_estimate"] == "UNKNOWN" and theta != 0.0:
        raise BasePoseValidationError(
            "current_alignment.theta must be 0 when orientation is UNKNOWN"
        )
    desired = _require_exact_keys(
        payload["desired_final_pose"],
        DESIRED_FINAL_POSE_KEYS,
        "desired_final_pose",
    )
    for key in DESIRED_FINAL_POSE_KEYS:
        _require_string(desired[key], f"desired_final_pose.{key}")
    _require_string(payload["expected_result"], "expected_result")
    _require_string(payload["limitations"], "limitations")
    status = payload["status"]
    if status not in {*NON_ADJUST_STATUSES, "ADJUST"}:
        raise BasePoseValidationError("status is invalid")
    confidence = _finite_number(payload["confidence"], "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise BasePoseValidationError("confidence must be in [0, 1]")
    commands = payload["command_sequence"]
    if not isinstance(commands, list):
        raise BasePoseValidationError("command_sequence must be an array")
    if status == "ADJUST" and not commands:
        raise BasePoseValidationError("ADJUST requires at least one command")
    if status in NON_ADJUST_STATUSES and commands:
        raise BasePoseValidationError(f"{status} requires an empty command_sequence")
    if len(commands) > max_steps:
        raise BasePoseValidationError(f"command_sequence exceeds {max_steps} steps")

    total_rotation = 0.0
    total_translation = 0.0
    for index, raw_command in enumerate(commands, start=1):
        command = _require_exact_keys(
            raw_command, COMMAND_KEYS, f"command_sequence[{index - 1}]"
        )
        step = command["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step != index:
            raise BasePoseValidationError(
                "step numbers must be consecutive and start at 1"
            )
        action = command["action"]
        if action not in ALLOWED_ACTIONS:
            raise BasePoseValidationError(
                f"command_sequence[{index - 1}].action is invalid"
            )
        value = _finite_number(command["value"], f"command_sequence[{index - 1}].value")
        _require_string(command["purpose"], f"command_sequence[{index - 1}].purpose")
        if value <= 0.0:
            raise BasePoseValidationError("command values must be positive")
        if action.startswith("ROTATE_"):
            if command["unit"] != "degrees":
                raise BasePoseValidationError("rotation commands must use degrees")
            if not min_rotation_deg <= value <= max_rotation_deg:
                raise BasePoseValidationError(
                    f"rotation command must be in [{min_rotation_deg}, {max_rotation_deg}] degrees"
                )
            total_rotation += value
        else:
            if command["unit"] != "meters":
                raise BasePoseValidationError("translation commands must use meters")
            if not min_translation_m <= value <= max_translation_m:
                raise BasePoseValidationError(
                    f"translation command must be in [{min_translation_m}, {max_translation_m}] meters"
                )
            total_translation += value
    if total_rotation > max_total_rotation_deg:
        raise BasePoseValidationError(
            f"total rotation exceeds {max_total_rotation_deg} degrees"
        )
    if total_translation > max_total_translation_m:
        raise BasePoseValidationError(
            f"total translation exceeds {max_total_translation_m} meters"
        )
    return dict(payload)


class AlignedRGBDCamera:
    """Read a fresh RGB or aligned RGB-D frame from a composed camera stream."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        host: str,
        port: int,
        *,
        stream_name: str = "ego_view",
        require_depth: bool = False,
        required_depth_source: str | None = None,
        timeout_ms: int = 15000,
    ):
        self.stream_name = stream_name
        self.depth_key = f"{stream_name}_depth"
        self.require_depth = bool(require_depth)
        self.required_depth_source = required_depth_source
        self.timeout_ms = int(timeout_ms)
        self._last_timestamp: float | None = None
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{int(port)}")

    def decode_payload(self, payload: Mapping[str, Any]) -> AlignedRGBDSnapshot:
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise BasePoseCameraError("unsupported camera schema_version")
        decoded = ImageMessageSchema.deserialize(
            dict(payload), decode_images=True
        ).asdict()
        images = decoded.get("images", {})
        info_map = decoded.get("camera_info", {})
        timestamps = decoded.get("timestamps", {})
        if self.stream_name not in images:
            raise BasePoseCameraError(f"camera payload requires {self.stream_name} RGB")
        info = info_map.get(self.stream_name)
        if not isinstance(info, Mapping):
            raise BasePoseCameraError(f"{self.stream_name} camera_info is missing")
        rgb = np.asarray(images[self.stream_name])
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise BasePoseCameraError(f"{self.stream_name} must be an RGB image")
        try:
            fx, fy, cx, cy = (float(info[name]) for name in ("fx", "fy", "cx", "cy"))
            width, height = int(info["width"]), int(info["height"])
            timestamp = float(timestamps[self.stream_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise BasePoseCameraError("head camera calibration is incomplete") from exc
        if not all(math.isfinite(value) for value in (fx, fy, cx, cy, timestamp)):
            raise BasePoseCameraError("head camera calibration must be finite")
        if fx <= 0.0 or fy <= 0.0 or (width, height) != (rgb.shape[1], rgb.shape[0]):
            raise BasePoseCameraError("head camera calibration is invalid")

        depth_raw: np.ndarray | None = None
        depth_scale_m: float | None = None
        depth_aligned_to: str | None = None
        depth_source: str | None = None
        if self.require_depth and self.depth_key in images:
            depth_raw = np.asarray(images[self.depth_key])
            if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
                raise BasePoseCameraError("aligned depth must be a uint16 image")
            if depth_raw.shape != rgb.shape[:2]:
                raise BasePoseCameraError("aligned RGB and depth shapes do not match")
            try:
                depth_scale_m = float(info["depth_scale_m"])
            except (KeyError, TypeError, ValueError) as exc:
                raise BasePoseCameraError("depth scale is missing") from exc
            depth_aligned_to = str(info.get("depth_aligned_to", ""))
            depth_source = str(info.get("depth_source", "")) or None
            if depth_scale_m <= 0.0 or not math.isfinite(depth_scale_m):
                raise BasePoseCameraError("depth scale must be finite and positive")
            if depth_aligned_to != self.stream_name:
                raise BasePoseCameraError(
                    "depth is not aligned to the requested RGB stream"
                )
            if (
                self.required_depth_source is not None
                and depth_source != self.required_depth_source
            ):
                raise BasePoseCameraError(
                    f"depth_source must be {self.required_depth_source!r}, got {depth_source!r}"
                )
            if self.required_depth_source == "lingbot-depth" and not math.isclose(
                depth_scale_m, 0.001, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise BasePoseCameraError(
                    "LingBot enhanced uint16 depth must use millimeter units"
                )
        elif self.require_depth:
            raise BasePoseCameraError(f"camera payload requires {self.depth_key}")

        return AlignedRGBDSnapshot(
            rgb=rgb,
            depth_raw=depth_raw,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale_m=depth_scale_m,
            depth_aligned_to=depth_aligned_to,
            depth_source=depth_source,
            timestamp=timestamp,
        )

    def capture(self) -> AlignedRGBDSnapshot:
        deadline = time.monotonic() + self.timeout_ms / 1000.0
        while True:
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            if not self._socket.poll(remaining_ms):
                raise BasePoseCameraError(
                    "timed out waiting for a fresh head-camera frame"
                )
            try:
                payload = msgpack.unpackb(self._socket.recv(), raw=False)
                snapshot = self.decode_payload(payload)
            except BasePoseCameraError:
                raise
            except Exception as exc:
                raise BasePoseCameraError(
                    "failed to decode head-camera message"
                ) from exc
            if (
                self._last_timestamp is None
                or snapshot.timestamp != self._last_timestamp
            ):
                self._last_timestamp = snapshot.timestamp
                return snapshot
            if time.monotonic() >= deadline:
                raise BasePoseCameraError(
                    "timed out waiting for a newer head-camera frame"
                )

    def close(self) -> None:
        self._socket.close()
        self._context.term()


def camera_parameters(
    config: BasePoseConfig, snapshot: AlignedRGBDSnapshot
) -> dict[str, Any]:
    height, width = snapshot.rgb.shape[:2]
    horizontal_fov = math.degrees(2.0 * math.atan(width / (2.0 * snapshot.fx)))
    return {
        "camera_stream": config.camera_stream,
        "camera_height_m": config.camera_height_m,
        "camera_pitch_deg": config.camera_pitch_deg,
        "vertical_fov_deg": config.vertical_fov_deg,
        "horizontal_fov_deg": horizontal_fov,
        "pitch_reference": "robot_body_horizontal",
        "pitch_convention": "positive_upward",
        "camera_forward_offset_m": config.camera_forward_offset_m,
        "camera_lateral_offset_m": config.camera_lateral_offset_m,
        "image_width_px": width,
        "image_height_px": height,
        "fx_px": snapshot.fx,
        "fy_px": snapshot.fy,
        "cx_px": snapshot.cx,
        "cy_px": snapshot.cy,
    }


def build_base_pose_prompt(
    config: BasePoseConfig, snapshot: AlignedRGBDSnapshot
) -> str:
    params = json.dumps(
        {
            "camera_height_m": config.camera_height_m,
            "camera_pitch_deg": config.camera_pitch_deg,
            "vertical_fov_deg": config.vertical_fov_deg,
            "pitch_reference": "robot_body_horizontal",
            "pitch_convention": "positive_upward",
        },
        ensure_ascii=False,
        indent=2,
    )
    task = json.dumps([config.task], ensure_ascii=False)
    return (
        """You are a base-pose adjustment planner for a G1 humanoid robot standing near a table and preparing to perform a manipulation task.

Your goal is to generate a complete and coherent sequence of base movements that places the robot in a suitable pose for performing the manipulation task.

## Inputs

Task description:

"""
        + task
        + """

RGB image:

[ATTACHED IMAGE]

"""
        + params
        + """

## Planning Procedure

Follow the steps below in order, using both the task description and the RGB image.

Identify the target manipulation region and the primary manipulation target. Among the objects most strongly related to the manipulation task description, select an object that is relatively large and provides a stable and reliable reference for alignment whenever possible.

Distinguish the primary manipulation target from other manipulation targets, the tabletop, other support surfaces, and any possible obstacles.

The appropriate body orientation should face the manipulation region directly. Determine whether the robot's current body orientation is turned to the left or to the right relative to the desired manipulation direction, and output TURNED_LEFT or TURNED_RIGHT accordingly. Estimate the angle by which the robot should rotate in order to directly face the manipulation region, and denote this estimated angle as theta.

Select a task-relevant alignment anchor on the primary manipulation target.

Determine the desired final horizontal position of the alignment anchor in the robot's ego-view image.

Imagine the robot applying the corrective rotation by theta toward the desired manipulation direction. Estimate where the alignment anchor would appear after this rotation. Compare the estimated post-rotation position of the alignment anchor with the horizontal center of the current camera view, and use this comparison together with the overall image geometry to determine the robot's lateral position relative to the target. Output FAR_LEFT or FAR_RIGHT when a substantial lateral repositioning is required.

Determine whether the robot should move closer to or farther from the primary manipulation target in order to reach a suitable manipulation distance. The suitable manipulation distance should keep all manipulation targets within the current camera view as much as possible while placing the robot as close to the manipulation region as reasonably possible.

Generate a continuous multi-step motion sequence that adjusts both the robot's position and orientation.

When lateral repositioning is required and the path is safe, prefer the following motion structure:

First move backward to create sufficient maneuvering space for turning and positional adjustment;

Then rotate toward the required lateral repositioning direction;

Move forward along the adjusted heading;

Finally rotate again so that the robot restores an appropriate manipulation orientation facing the manipulation region.

When the robot is positioned to the left of the target, a rightward repositioning is normally required. Prefer:

MOVE_BACKWARD;

ROTATE_RIGHT;

MOVE_FORWARD;

ROTATE_LEFT.

When the robot is positioned to the right of the target, a leftward repositioning is normally required. Prefer:

MOVE_BACKWARD;

ROTATE_LEFT;

MOVE_FORWARD;

ROTATE_RIGHT.

The final rotation should restore the robot's body orientation so that it faces the manipulation region and is approximately perpendicular to the relevant table edge when such an edge can be identified.

The final pose should:

place the center of the manipulation region near the horizontal center of the camera view;

place the robot at the smallest suitable manipulation distance from the manipulation region while maintaining safe whole-body operation;

orient the robot so that it faces the relevant manipulation region;

provide a reasonable shared base pose that allows the robot to reach the manipulation region using the specified hands.

## Allowed Motion Commands

ROTATE_LEFT: rotate counterclockwise by a specified number of degrees.

ROTATE_RIGHT: rotate clockwise by a specified number of degrees.

MOVE_FORWARD: move forward along the robot's current heading by a specified number of meters.

MOVE_BACKWARD: move backward along the robot's current heading by a specified number of meters.

Direct lateral translation is not available.

## Motion-Planning Rules

Output a complete and coherent sequence containing all base movements required to reach the estimated manipulation pose.

Commands must be ordered exactly as they should be executed.

Each command is defined relative to the robot pose resulting from all previous commands.

Multiple rotations and translations may be used.

Do not require a new observation between commands.

Do not limit the plan to small incremental movements.

Avoid unnecessary movements and redundant direction changes.

Prefer a smooth and geometrically consistent trajectory.

Do not output zero-valued or negative-valued motion commands.

## Translation Constraints

Every MOVE_FORWARD or MOVE_BACKWARD command must specify a distance greater than or equal to 0.3 meters.

Do not output a translation command with a distance smaller than 0.3 meters.

If the desired positional correction is smaller than 0.3 meters, do not approximate it using an invalid smaller translation.

When geometrically appropriate and safe, use a combination of backward movement, rotation, forward movement, and a final corrective rotation to produce the required positional change.

Do not output direct lateral movement.

## Rotation Constraints

Every ROTATE_LEFT or ROTATE_RIGHT command must specify an angle greater than or equal to 30 degrees.

Do not output a rotation command with an angle smaller than 30 degrees.

If a required standalone net orientation correction is smaller than 30 degrees, it may be achieved using two rotations in opposite directions.

First rotate by at least 30 degrees in the direction opposite to the required correction, then rotate in the required direction by an angle that produces the desired net correction.

Every individual rotation in this indirect correction must still be greater than or equal to 30 degrees.

For example, if a net 10-degree right rotation is required, use ROTATE_LEFT by 30 degrees followed by ROTATE_RIGHT by 40 degrees.

Use this indirect small-angle correction only when the intermediate rotation is safe and does not create unnecessary collision or stability risk.

Do not add an indirect small-angle correction when the required orientation change can already be achieved as part of the lateral repositioning sequence.

## Task and Safety Constraints

Do not output arm or hand commands.

Do not include target-recognition, image-acquisition, or observation commands in the motion sequence.

Account for the visible table, support surfaces, objects, and obstacles when evaluating motion safety.

Because no depth image or complete camera calibration is provided, all distances and angles are approximate visual estimates.

Do not claim that the estimated motion values are geometrically exact.

If the main target cannot be identified, return UNSURE.

If the required interaction direction cannot be reasonably inferred, return UNSURE.

If the required movement cannot be reasonably inferred from the image, return UNSURE.

Do not invent a motion sequence when the visual evidence is insufficient.

If the scene appears unsafe for the proposed base motion, return UNSAFE.

Apply status priority in the following order: UNSAFE first, then UNSURE, then ADJUST.

## Output Requirements

Output only one valid JSON object.

Do not include Markdown, code fences, explanations, comments, or any additional text outside the JSON object.

Use the following format:

{
"status": "ADJUST | UNSURE | UNSAFE",
"task_interpretation": {
"primary_target": "",
"secondary_targets": [],
"manipulation_anchor": "",
"interaction_direction": "",
"selection_reason": ""
},
"current_alignment": {
"horizontal_position": "FAR_LEFT | LEFT | CENTERED | RIGHT | FAR_RIGHT | UNKNOWN",
"distance_estimate": "TOO_CLOSE | SUITABLE | TOO_FAR | UNKNOWN",
"orientation_estimate": "TURNED_LEFT | ALIGNED | TURNED_RIGHT | UNKNOWN",
"theta": 0.0
},
"desired_final_pose": {
"target_alignment": "",
"target_distance": "",
"target_orientation": ""
},
"command_sequence": [
{
"step": 1,
"action": "ROTATE_LEFT | ROTATE_RIGHT | MOVE_FORWARD | MOVE_BACKWARD",
"value": 30.0,
"unit": "degrees | meters",
"purpose": ""
}
],
"expected_result": "",
"confidence": 0.0,
"limitations": ""
}

## Field Interpretation

primary_target should identify  the primary manipulation target.

secondary_targets should identify other task-relevant objects.

manipulation_anchor should identify a task-relevant alignment anchor on the primary manipulation target.

interaction_direction should describe the desired body-forward direction facing the manipulation region. 

horizontal_position describes the robot's estimated lateral position relative to the manipulation region after accounting for the estimated orientation correction theta, rather than simply describing the raw image position of the manipulation anchor.

If the estimated post-rotation manipulation anchor remains substantially to the right of the camera-view center, the robot is positioned to the LEFT or FAR_LEFT of the desired manipulation position.

If the estimated post-rotation manipulation anchor remains substantially to the left of the camera-view center, the robot is positioned to the RIGHT or FAR_RIGHT of the desired manipulation position.

If the estimated post-rotation manipulation anchor is near the camera-view center, the robot is approximately CENTERED relative to the desired manipulation position.

Use FAR_LEFT or FAR_RIGHT when the estimated lateral displacement is visually substantial and meaningful base repositioning is required.

distance_estimate should be judged relative to a suitable whole-body manipulation distance from the manipulation region. The preferred distance should keep all task-relevant manipulation targets visible whenever reasonably possible while placing the robot as close to the manipulation region as is appropriate for safe manipulation.

orientation_estimate should be judged relative to the desired body orientation facing the manipulation region.

Use TURNED_LEFT when the robot's current body-forward direction points to the left of the desired manipulation direction.

Use TURNED_RIGHT when the robot's current body-forward direction points to the right of the desired manipulation direction.

Use ALIGNED when the robot already approximately faces the desired manipulation direction.

theta should represent the estimated magnitude, in degrees, of the corrective body rotation required for the robot to face the manipulation region from its current orientation.

theta describes only the magnitude of the estimated orientation correction. The rotation direction is determined by orientation_estimate:

TURNED_LEFT means the robot should normally rotate right by approximately theta degrees to face the manipulation region.

TURNED_RIGHT means the robot should normally rotate left by approximately theta degrees to face the manipulation region.

ALIGNED means theta should be approximately 0 degrees.

If the required orientation correction cannot be reasonably estimated from the image, orientation_estimate should be UNKNOWN and theta should be 0.0.

theta is an estimated geometric quantity used for scene reasoning and is not itself a motion command. Therefore, theta is not subject to the minimum 30-degree individual rotation-command constraint.

target_alignment should state that the center of the manipulation region should be near the horizontal center of the camera view.

target_distance should describe the desired manipulation distance, keeping the robot as close as reasonably possible while maintaining a safe and useful whole-body manipulation pose and preserving visibility of the relevant manipulation targets whenever possible.

target_orientation should state that the robot should face the manipulation region. When a relevant table edge is identifiable, the robot should be approximately perpendicular to that edge.

purpose should briefly explain the geometric role of each command.

## Consistency Constraints

UNSURE must have an empty command_sequence.

UNSAFE must have an empty command_sequence.

ADJUST must contain at least one command.

Rotation commands must use degrees.

Translation commands must use meters.

Every rotation-command value must be greater than or equal to 30 degrees.

Every translation-command value must be greater than or equal to 0.3 meters.

Every command value must be positive.

Every step number must be consecutive, starting from 1.

The action and unit of every command must match.

theta must be a non-negative numerical value representing degrees.

theta is an approximate visual estimate and must not be described as geometrically exact.

When orientation_estimate is TURNED_LEFT, theta represents the approximate magnitude of the required rightward corrective rotation.

When orientation_estimate is TURNED_RIGHT, theta represents the approximate magnitude of the required leftward corrective rotation.

When orientation_estimate is ALIGNED, theta should be approximately 0.0.

When orientation_estimate is UNKNOWN, theta must be 0.0.

The command sequence does not need to contain a single rotation command exactly equal to theta. The required orientation correction may be distributed across multiple rotations as part of the complete position-and-orientation adjustment trajectory.

confidence must be a number from 0.0 to 1.0.

expected_result must describe the estimated final robot-to-manipulation-region position, distance, and orientation after the complete command_sequence has been executed.

For UNSURE or UNSAFE, explain the reason in limitations without inventing unsupported geometry.
"""
    )


def build_depth_query_prompt(
    config: BasePoseConfig, snapshot: AlignedRGBDSnapshot
) -> str:
    params = json.dumps(
        camera_parameters(config, snapshot), ensure_ascii=False, indent=2
    )
    task = json.dumps(config.task, ensure_ascii=False)
    return f"""You select task-relevant image regions for metric depth lookup.

TASK DESCRIPTION: {task}
CAMERA PARAMETERS:
{params}
RGB IMAGE: [ATTACHED_IMAGE_1]

Interpret the full manipulation task and identify the anchors that determine the final robot base pose. Return at most three tight bounding boxes in normalized [0,1000] coordinates. Use PRIMARY_TARGET for grasp/contact anchors, DESTINATION for openings or placement anchors, and COMBINED_WORKSPACE when one region represents multiple interactions. Do not output movement commands. If no reliable anchor can be identified, return an empty depth_queries array and explain why in limitations.

Output only one JSON object matching the supplied schema, with no Markdown or extra text.
"""


def _validate_depth_queries(value: Any) -> dict[str, Any]:
    payload = _require_exact_keys(
        value,
        {"task_interpretation", "depth_queries", "confidence", "limitations"},
        "depth query response",
    )
    interpretation = _require_exact_keys(
        payload["task_interpretation"],
        {
            "primary_target",
            "secondary_targets",
            "manipulation_anchors",
            "selection_reason",
        },
        "depth query task_interpretation",
    )
    _require_string(interpretation["primary_target"], "primary_target")
    _require_string(interpretation["selection_reason"], "selection_reason")
    for key in ("secondary_targets", "manipulation_anchors"):
        if not isinstance(interpretation[key], list) or not all(
            isinstance(item, str) for item in interpretation[key]
        ):
            raise BasePoseValidationError(f"{key} must be an array of strings")
    confidence = _finite_number(payload["confidence"], "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise BasePoseValidationError("confidence must be in [0, 1]")
    _require_string(payload["limitations"], "limitations")
    queries = payload["depth_queries"]
    if not isinstance(queries, list) or len(queries) > 3:
        raise BasePoseValidationError(
            "depth_queries must contain at most three entries"
        )
    seen: set[str] = set()
    for index, query in enumerate(queries):
        if not isinstance(query, Mapping):
            raise BasePoseValidationError(f"depth_queries[{index}] must be an object")
        _require_exact_keys(
            query,
            {
                "query_id",
                "task_role",
                "target_or_anchor",
                "bbox_2d",
                "sampling_reason",
            },
            f"depth_queries[{index}]",
        )
        query_id = query.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in seen:
            raise BasePoseValidationError(
                "depth query IDs must be non-empty and unique"
            )
        seen.add(query_id)
        if query.get("task_role") not in {
            "PRIMARY_TARGET",
            "DESTINATION",
            "COMBINED_WORKSPACE",
        }:
            raise BasePoseValidationError("depth query task_role is invalid")
        _require_string(query.get("target_or_anchor"), "target_or_anchor")
        _require_string(query.get("sampling_reason"), "sampling_reason")
        bbox = query.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise BasePoseValidationError(
                "depth query bbox_2d must contain four values"
            )
        coords = [_finite_number(item, "bbox_2d") for item in bbox]
        if not all(0.0 <= item <= 1000.0 for item in coords):
            raise BasePoseValidationError(
                "depth query bbox coordinates must be in [0,1000]"
            )
        if coords[0] >= coords[2] or coords[1] >= coords[3]:
            raise BasePoseValidationError("depth query bbox corner order is invalid")
    return dict(payload)


def query_depth_regions(
    selection: Mapping[str, Any], snapshot: AlignedRGBDSnapshot
) -> list[dict[str, Any]]:
    if snapshot.depth_raw is None or snapshot.depth_scale_m is None:
        raise BasePoseCameraError("numeric depth lookup requires aligned depth")
    height, width = snapshot.depth_raw.shape
    depth_mm = snapshot.depth_raw.astype(np.float64) * snapshot.depth_scale_m * 1000.0
    results: list[dict[str, Any]] = []
    for query in selection["depth_queries"]:
        x1, y1, x2, y2 = (float(item) for item in query["bbox_2d"])
        px1 = max(0, min(width - 1, int(math.floor(x1 * width / 1000.0))))
        py1 = max(0, min(height - 1, int(math.floor(y1 * height / 1000.0))))
        px2 = max(px1 + 1, min(width, int(math.ceil(x2 * width / 1000.0))))
        py2 = max(py1 + 1, min(height, int(math.ceil(y2 * height / 1000.0))))
        roi = depth_mm[py1:py2, px1:px2]
        valid_mask = np.isfinite(roi) & (roi >= 100.0) & (roi <= 10000.0)
        valid = roi[valid_mask]
        u = (px1 + px2) // 2
        v = (py1 + py2) // 2
        radius = 3
        center = depth_mm[
            max(0, v - radius) : min(height, v + radius + 1),
            max(0, u - radius) : min(width, u + radius + 1),
        ]
        center_valid = center[
            np.isfinite(center) & (center >= 100.0) & (center <= 10000.0)
        ]
        result: dict[str, Any] = {
            "query_id": query["query_id"],
            "task_role": query["task_role"],
            "target_or_anchor": query["target_or_anchor"],
            "bbox_normalized": [x1, y1, x2, y2],
            "bbox_pixel": [px1, py1, px2, py2],
            "center_pixel": [u, v],
            "valid_depth_samples": int(valid.size),
            "total_roi_pixels": int(roi.size),
            "valid_ratio": float(valid.size / roi.size),
            "depth_source": snapshot.depth_source,
        }
        if valid.size:
            percentiles = np.percentile(valid, [10, 25, 50, 75, 90])
            result["depth_percentiles_mm"] = {
                name: float(value)
                for name, value in zip(("p10", "p25", "p50", "p75", "p90"), percentiles)
            }
        else:
            result["depth_percentiles_mm"] = None
        center_median_mm = float(np.median(center_valid)) if center_valid.size else None
        result["center_7x7_median_mm"] = center_median_mm
        chosen_mm = center_median_mm
        if chosen_mm is None and valid.size:
            chosen_mm = float(np.median(valid))
        if chosen_mm is None:
            result["camera_xyz_m"] = None
        else:
            z = chosen_mm / 1000.0
            result["camera_xyz_m"] = {
                "x_right": (u - snapshot.cx) * z / snapshot.fx,
                "y_down": (v - snapshot.cy) * z / snapshot.fy,
                "z_forward": z,
            }
        results.append(result)
    return results


class CodexStructuredVisionClient:
    """Execute a read-only Codex CLI request with one or more images."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "max",
        fast: bool = True,
        timeout_seconds: float = 600.0,
        codex_bin: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.fast = bool(fast)
        self.timeout_seconds = float(timeout_seconds)
        self.codex_bin = codex_bin or os.environ.get("CODEX_BIN", "codex")
        self.runner = runner
        self._authenticated = False

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        child_env = os.environ.copy()
        http_proxy = os.environ.get("BASE_POSE_CODEX_HTTP_PROXY")
        all_proxy = os.environ.get("BASE_POSE_CODEX_ALL_PROXY")
        for keys, value in ((HTTP_PROXY_KEYS, http_proxy), (ALL_PROXY_KEYS, all_proxy)):
            if value is None:
                continue
            for key in keys:
                if value:
                    child_env[key] = value
                else:
                    child_env.pop(key, None)
        return child_env

    def _check_login(self) -> None:
        if self._authenticated:
            return
        result = self.runner(
            [self.codex_bin, "login", "status"],
            capture_output=True,
            text=True,
            timeout=min(15.0, self.timeout_seconds),
            check=False,
            env=self._subprocess_env(),
        )
        login_text = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        if result.returncode != 0 or "chatgpt" not in login_text:
            raise RuntimeError(
                "Codex CLI must be logged in with a ChatGPT subscription"
            )
        self._authenticated = True

    def run(
        self,
        *,
        prompt: str,
        image_paths: Sequence[str | Path],
        schema: Mapping[str, Any],
        schema_filename: str,
        cwd: str | Path,
    ) -> dict[str, Any]:
        self._check_login()
        resolved_images = [Path(path).resolve() for path in image_paths]
        for path in resolved_images:
            if not path.is_file():
                raise FileNotFoundError(f"model image input not found: {path}")
        workdir = Path(cwd).resolve()
        schema_path = workdir / schema_filename
        _write_json(schema_path, schema)
        command = [
            self.codex_bin,
            "--ask-for-approval",
            "never",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--model",
            self.model,
            "--config",
            f'model_reasoning_effort="{self.reasoning_effort}"',
        ]
        command.extend(
            ("--config", f"features.fast_mode={str(self.fast).lower()}")
        )
        if self.fast:
            command.extend(("--config", 'service_tier="fast"'))
        for path in resolved_images:
            command.extend(("--image", str(path)))
        command.extend(("--output-schema", str(schema_path), prompt))
        result = self.runner(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
            cwd=str(workdir),
            env=self._subprocess_env(),
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "unknown error").strip()
            raise RuntimeError(f"Codex base-pose policy failed: {message}")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise BasePoseValidationError("Codex output is not valid JSON") from exc
        if not isinstance(value, dict):
            raise BasePoseValidationError("Codex output must be a JSON object")
        return value


def _dashscope_api_key(env_file: str | Path | None = None) -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    path = Path(env_file) if env_file is not None else Path(sys.prefix) / ".env"
    if path.is_file():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "DASHSCOPE_API_KEY" and value.strip():
                return value.strip().strip("\"'")
    raise RuntimeError(
        "Qwen-VL requires DASHSCOPE_API_KEY in the environment or "
        f"{path}"
    )


class QwenVLStructuredVisionClient:
    """Call Qwen-VL Plus through DashScope's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_QWENVL_PLUS_MODEL,
        base_url: str = DEFAULT_QWENVL_BASE_URL,
        timeout_seconds: float = 600.0,
        thinking_budget: int = 500,
        api_key: str | None = None,
        env_file: str | Path | None = None,
        client: Any | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = float(timeout_seconds)
        self.thinking_budget = int(thinking_budget)
        self.last_reasoning_content = ""
        self.last_answer_content = ""
        if self.thinking_budget <= 0:
            raise ValueError("Qwen-VL thinking_budget must be positive")
        if client is not None:
            self.client = client
            return
        key = api_key or _dashscope_api_key(env_file)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Qwen-VL requires the openai package in .venv_inference"
            ) from exc
        import httpx

        proxy = (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
        )
        http_client = (
            httpx.Client(proxy=proxy) if proxy else httpx.Client(trust_env=False)
        )
        self.client = OpenAI(
            api_key=key,
            base_url=base_url,
            http_client=http_client,
        )

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        text = content.strip()
        if not text:
            raise BasePoseValidationError("Qwen-VL base-pose output is empty")
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BasePoseValidationError(
                "Qwen-VL base-pose output is not valid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise BasePoseValidationError(
                "Qwen-VL base-pose output must be a JSON object"
            )
        return value

    def run(
        self,
        *,
        prompt: str,
        image_paths: Sequence[str | Path],
        schema: Mapping[str, Any],
        schema_filename: str,
        cwd: str | Path,
    ) -> dict[str, Any]:
        resolved_images = [Path(path).resolve() for path in image_paths]
        for path in resolved_images:
            if not path.is_file():
                raise FileNotFoundError(f"model image input not found: {path}")
        workdir = Path(cwd).resolve()
        _write_json(workdir / schema_filename, schema)
        content: list[dict[str, Any]] = []
        for path in resolved_images:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            )
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        image_manifest = "\n".join(
            f"ATTACHED_IMAGE_{index} is image_url item {index} above."
            for index in range(1, len(resolved_images) + 1)
        )
        content.append(
            {
                "type": "text",
                "text": (
                    f"{image_manifest}\n\n{prompt}\n\n"
                    "Return only one JSON object matching this schema:\n"
                    f"{schema_text}"
                ),
            }
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            stream=True,
            timeout=self.timeout_seconds,
            extra_body={
                "enable_thinking": True,
                "thinking_budget": self.thinking_budget,
            },
        )
        reasoning_parts: list[str] = []
        answer_parts: list[str] = []
        for chunk in completion:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            answer = getattr(delta, "content", None)
            if answer:
                answer_parts.append(answer)
        self.last_reasoning_content = "".join(reasoning_parts)
        self.last_answer_content = "".join(answer_parts)
        return self._parse_json_content(self.last_answer_content)


class BasePoseRunner:
    """Capture one observation and obtain one validated model-authored plan."""

    def __init__(
        self,
        config: BasePoseConfig,
        *,
        camera: AlignedRGBDCamera | None = None,
        client: CodexStructuredVisionClient | QwenVLStructuredVisionClient | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        if config.mode not in BASE_POSE_MODES:
            raise ValueError(f"unsupported base-pose mode: {config.mode}")
        if config.vision_backend not in BASE_POSE_VISION_BACKENDS:
            raise ValueError(
                f"unsupported base-pose vision backend: {config.vision_backend}"
            )
        if not config.task.strip():
            raise ValueError("base-pose task must be non-empty")
        if config.depth_visual_max_m <= 0.0:
            raise ValueError("depth_visual_max_m must be positive")
        self.config = config
        self._monotonic = monotonic
        require_depth = config.mode != "rgb"
        self.camera = camera or AlignedRGBDCamera(
            config.camera_host,
            config.camera_port,
            stream_name=config.camera_stream,
            require_depth=require_depth,
            required_depth_source="lingbot-depth" if require_depth else None,
            timeout_ms=config.camera_timeout_ms,
        )
        if client is not None:
            self.client = client
        elif config.vision_backend == "qwenvl":
            self.client = QwenVLStructuredVisionClient(
                model=config.qwenvl_model,
                base_url=config.qwenvl_base_url,
                timeout_seconds=config.codex_timeout_seconds,
                thinking_budget=config.qwenvl_thinking_budget,
            )
        else:
            self.client = CodexStructuredVisionClient(
                model=config.model,
                reasoning_effort=config.reasoning_effort,
                fast=config.codex_fast,
                timeout_seconds=config.codex_timeout_seconds,
            )

    def _output_dir(self) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"{time.time_ns() % 1_000_000_000:09d}"
        path = Path(self.config.output_root).resolve() / f"{stamp}_{suffix}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    @staticmethod
    def _save_rgb(path: Path, rgb: np.ndarray) -> None:
        success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("failed to encode RGB observation")
        _atomic_write_bytes(path, encoded.tobytes())

    def run_once(self) -> BasePoseResult:
        started = self._monotonic()
        output_dir = self._output_dir()
        timing: dict[str, float] = {}
        stage = "camera"
        try:
            capture_started = self._monotonic()
            snapshot = self.camera.capture()
            timing["camera"] = self._monotonic() - capture_started
            rgb_path = output_dir / "rgb.png"
            self._save_rgb(rgb_path, snapshot.rgb)
            _write_json(
                output_dir / "camera_parameters.json",
                camera_parameters(self.config, snapshot),
            )

            prompt = build_base_pose_prompt(self.config, snapshot)
            image_paths: list[Path] = [rgb_path]
            depth_path: Path | None = None
            color_path: Path | None = None
            if snapshot.depth_raw is not None:
                assert snapshot.depth_scale_m is not None
                depth_path = output_dir / "depth_uint16_mm.png"
                success, encoded = cv2.imencode(".png", snapshot.depth_raw)
                if not success:
                    raise RuntimeError("failed to encode uint16 depth observation")
                _atomic_write_bytes(depth_path, encoded.tobytes())
                color, stats = colorize_depth(
                    snapshot.depth_raw,
                    max_depth_m=self.config.depth_visual_max_m,
                    depth_scale_m=snapshot.depth_scale_m,
                )
                color_path = output_dir / "depth_color_0_3m.png"
                success, encoded = cv2.imencode(".png", color)
                if not success:
                    raise RuntimeError("failed to encode colorized depth observation")
                _atomic_write_bytes(color_path, encoded.tobytes())
                _write_json(output_dir / "depth_visual_stats.json", stats.__dict__)

            if self.config.mode == "rgbd":
                assert depth_path is not None and color_path is not None
                image_paths.extend((depth_path, color_path))
                prompt += f"""

## Aligned depth inputs

* ATTACHED_IMAGE_2 is the lossless aligned uint16 LingBot depth image in millimeters; zero is invalid.
* ATTACHED_IMAGE_3 is a visualization over 0-{self.config.depth_visual_max_m:g} meters. Near is warm, far is cool, and invalid pixels are black.
* Both depth images are pixel-aligned to ATTACHED_IMAGE_1. Use them as approximate evidence; do not claim false pixel-level precision.
"""
            elif self.config.mode == "rgb_depth_query":
                assert snapshot.depth_raw is not None
                stage = "depth_query_model"
                query_prompt = build_depth_query_prompt(self.config, snapshot)
                _write_text(output_dir / "depth_query_prompt.txt", query_prompt)
                query_started = self._monotonic()
                selection = self.client.run(
                    prompt=query_prompt,
                    image_paths=[rgb_path],
                    schema=DEPTH_QUERY_SCHEMA,
                    schema_filename="depth_query.schema.json",
                    cwd=output_dir,
                )
                timing["depth_query_model"] = self._monotonic() - query_started
                selection = _validate_depth_queries(selection)
                measurements = query_depth_regions(selection, snapshot)
                _write_json(output_dir / "depth_query_selection.json", selection)
                _write_json(output_dir / "depth_measurements.json", measurements)
                prompt += """

## Numeric aligned-depth evidence

The following evidence was queried from the same LingBot-enhanced uint16 frame as the attached RGB. Values are measurements, not motion commands. A box may contain foreground and background, so compare the center median, valid ratio, and percentiles. You must still author the complete final movement sequence yourself.

""" + json.dumps(
                    {
                        "selection": selection,
                        "measurements": measurements,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            stage = "final_model"
            _write_text(output_dir / "final_prompt.txt", prompt)
            model_started = self._monotonic()
            raw_plan = self.client.run(
                prompt=prompt,
                image_paths=image_paths,
                schema=BASE_POSE_OUTPUT_SCHEMA,
                schema_filename="base_pose_plan.schema.json",
                cwd=output_dir,
            )
            timing["final_model"] = self._monotonic() - model_started
            _write_json(output_dir / "model_output.json", raw_plan)
            stage = "validation"
            validation_started = self._monotonic()
            try:
                plan = validate_base_pose_plan(raw_plan)
            except Exception as exc:
                _write_json(
                    output_dir / "validation.json",
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                )
                raise
            timing["validation"] = self._monotonic() - validation_started
            timing["total"] = self._monotonic() - started
            _write_json(output_dir / "validation.json", {"ok": True, "error": None})
            _write_json(output_dir / "base_pose_plan.json", plan)
            _write_json(output_dir / "timing.json", timing)
            return BasePoseResult(
                plan=plan, output_dir=str(output_dir), timing_s=timing
            )
        except Exception as exc:
            timing["total"] = self._monotonic() - started
            try:
                _write_json(
                    output_dir / "failure.json",
                    {
                        "stage": stage,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                _write_json(output_dir / "timing.json", timing)
            except Exception:
                pass
            raise

    def close(self) -> None:
        self.camera.close()
