"""Head-camera multimodal base-pose adjustment policy for Unitree G1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import subprocess
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
    model: str = "gpt-5.6"
    reasoning_effort: str = "high"
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
    codex_timeout_seconds: float = 180.0
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
        camera_parameters(config, snapshot), ensure_ascii=False, indent=2
    )
    task = json.dumps(config.task, ensure_ascii=False)
    return f"""You are a base-pose adjustment planner for a G1 humanoid robot standing near a table and preparing to perform a manipulation task.

Your goal is to generate a complete and coherent sequence of robot base movements that places the robot in a suitable pose for completing the full manipulation task.

## Inputs

Task description:

{task}

RGB image:

[ATTACHED_IMAGE_1]

Camera parameters:

{params}

## Camera-parameter interpretation

* `camera_height_m` is the height of the camera optical center above the ground, measured in meters.
* `camera_pitch_deg` is the pitch angle of the camera optical axis relative to the robot-body horizontal plane.
* Under the `positive_upward` convention, positive pitch points upward and negative pitch points downward.
* Therefore, a negative `camera_pitch_deg` means that the camera optical axis points downward relative to the robot-body horizontal plane.
* `vertical_fov_deg` and `horizontal_fov_deg` are full fields of view.
* Use the camera height, camera pitch, fields of view, live intrinsics, image perspective, object scale, support-surface geometry, and visible spatial relationships when estimating the required base adjustment.
* Assume that the camera forward direction in the horizontal plane is aligned with the robot's forward body direction.
* The configured camera forward/lateral offsets are relative to the robot base; zero means that translation extrinsics are not yet calibrated.
* Do not assume access to any information other than the task description, attached inputs, and camera parameters listed above.
* All numerical movement values are approximate visual estimates.

## Planning objectives

Use the task description, RGB image, and camera parameters to:

1. Interpret the complete manipulation task, including the required order of object interactions.
2. Identify the primary manipulation target or targets.
3. Distinguish among manipulation targets, destination containers, support surfaces, nearby objects, environmental obstacles, and irrelevant background objects.
4. Select the task-relevant manipulation anchor for each interaction, such as a handle, button, opening, graspable region, contact surface, insertion point, small-object center, or container opening.
5. Determine which manipulation anchor or combined manipulation workspace should guide the final base pose.
6. Determine the most suitable final horizontal alignment between the task-relevant manipulation workspace and the robot's ego-view.
7. Estimate whether the robot should move closer to or farther from the manipulation workspace.
8. Estimate whether the robot should rotate left, rotate right, or remain aligned.
9. Generate a continuous multi-step base-motion sequence that adjusts both robot position and orientation.
10. Prefer one final base pose that supports the complete sequence of manipulations without requiring another base adjustment between individual arm actions.
11. For tasks involving both hands, keep right-hand objects in the right workspace, left-hand objects in the left workspace, and shared containers near the central workspace when practical.
12. After the planned movement, the robot should face the workspace with a practical table standoff and body orientation.

The objective is not always to align the geometric center of an entire object with the center of the image. Align the task-relevant anchor, opening, interaction axis, or combined workspace best suited to the complete task.

## Visual reasoning guidelines

* Classify horizontal workspace position as FAR_LEFT, LEFT, CENTERED, RIGHT, or FAR_RIGHT.
* Classify current standoff as TOO_CLOSE, SUITABLE, TOO_FAR, or UNKNOWN.
* Do not infer distance from vertical image position alone; account for the downward pitch and support-surface geometry.
* Prioritize the complete manipulation workspace for multi-object tasks.
* Balance reachability, visibility, collision clearance, both-arm access, and destination access.
* Prefer small purposeful corrections when the current pose is close to suitable.

## Allowed motion commands

* `ROTATE_LEFT`: counterclockwise rotation in degrees.
* `ROTATE_RIGHT`: clockwise rotation in degrees.
* `MOVE_FORWARD`: forward motion along the current heading in meters.
* `MOVE_BACKWARD`: backward motion along the current heading in meters.

Direct lateral translation is unavailable.

## Motion-planning rules

* Output every movement required to reach the final manipulation pose in exact execution order.
* Every command is relative to the pose resulting from all previous commands.
* Do not require a new observation between commands.
* For lateral repositioning, use a geometrically coherent backward/rotation/forward/corrective-rotation sequence.
* Restore a suitable final orientation toward the manipulation workspace.
* Avoid redundant commands and unnecessary direction changes.
* Do not output arm, hand, gripper, head, gaze, perception, or manipulation commands.
* Keep reasonable clearance from the table and visible obstacles.
* Do not move forward when already extremely close to the table.
* If backing away is required before rotating, move backward first.
* Rotation values must be in degrees and translations in meters.
* Each rotation must be between 2 and 90 degrees.
* Each translation must be between 0.10 and 1.50 meters.
* Use at most 8 commands, at most 180 cumulative rotation degrees, and at most 3.0 cumulative translation meters.
* Numerical values must be plausible estimates rather than false precision.

## Status-selection rules

Return `READY` when the complete manipulation workspace is suitably aligned and reachable without base movement.
Return `ADJUST` when a coherent correction can be inferred and one or more commands are required.
Return `UNSURE` when targets, anchors, necessary direction, or important workspace areas cannot be reliably determined.
Return `UNSAFE` when a coherent collision-free base sequence cannot be inferred from the visible scene.

## Output requirements

Output only one valid JSON object.

Do not include Markdown, comments, explanations, hidden reasoning, or any text outside the JSON object.

Use exactly this structure:

{{
  "status": "READY | ADJUST | UNSURE | UNSAFE",
  "task_interpretation": {{
    "primary_target": "",
    "secondary_targets": [],
    "manipulation_anchor": "",
    "interaction_direction": "",
    "selection_reason": ""
  }},
  "current_alignment": {{
    "horizontal_position": "FAR_LEFT | LEFT | CENTERED | RIGHT | FAR_RIGHT | UNKNOWN",
    "distance_estimate": "TOO_CLOSE | SUITABLE | TOO_FAR | UNKNOWN",
    "orientation_estimate": "TURNED_LEFT | ALIGNED | TURNED_RIGHT | UNKNOWN"
  }},
  "desired_final_pose": {{
    "target_alignment": "",
    "target_distance": "",
    "target_orientation": ""
  }},
  "command_sequence": [
    {{
      "step": 1,
      "action": "ROTATE_LEFT | ROTATE_RIGHT | MOVE_FORWARD | MOVE_BACKWARD",
      "value": 2.0,
      "unit": "degrees | meters",
      "purpose": ""
    }}
  ],
  "expected_result": "",
  "confidence": 0.0,
  "limitations": ""
}}

## Field-definition requirements

### `task_interpretation.primary_target`

Describe the object or objects that must be directly manipulated. For a multi-object task, include all primary manipulation objects in one concise string.

### `task_interpretation.secondary_targets`

List destination containers, support surfaces, and task-relevant environmental objects. Do not include irrelevant background objects unless they affect safety or movement.

### `task_interpretation.manipulation_anchor`

Describe the task-relevant grasping, contact, insertion, or placement regions. For a multi-stage task, include both pickup anchors and destination anchors when necessary.

### `task_interpretation.interaction_direction`

Describe the preferred direction from which the robot should face or approach the combined manipulation workspace.

### `task_interpretation.selection_reason`

Explain why the selected workspace and anchor are appropriate for the complete task.

### `current_alignment.horizontal_position`

Classify the task-relevant manipulation workspace relative to the current ego-view. Base this classification on the anchor or combined workspace, not necessarily on the geometric center of the largest object.

### `current_alignment.distance_estimate`

Classify the current table or workspace standoff using visible scene geometry and the supplied camera parameters.

### `current_alignment.orientation_estimate`

Use `TURNED_LEFT` when the robot appears oriented too far toward the left side of the desired workspace; `TURNED_RIGHT` when it appears oriented too far toward the right; `ALIGNED` when the current body-facing direction is suitable; and `UNKNOWN` when orientation cannot be inferred.

### `desired_final_pose.target_alignment`

Describe where the manipulation workspace should appear horizontally after all commands are executed.

### `desired_final_pose.target_distance`

Describe the intended final table or workspace standoff.

### `desired_final_pose.target_orientation`

Describe the intended final body-facing direction.

### `command_sequence`

Include only base-motion commands. Each command must contain a consecutive step number, one allowed action, a positive numerical value, the correct unit, and a concise purpose.

### `expected_result`

Describe the estimated final relationship after the complete sequence. Mention final horizontal alignment, table or workspace standoff, body orientation, and expected accessibility of the targets and container.

### `confidence`

Use a number from `0.0` to `1.0`. Use lower confidence when target boundaries are unclear, the table edge is partially visible, dimensions are uncertain, the exact camera-to-base relationship is uncertain, or motion relies heavily on approximate visual interpretation.

### `limitations`

Briefly state the main causes of uncertainty. Do not use this field to contradict the selected status or command sequence.

## Consistency constraints

* `READY`, `UNSURE`, and `UNSAFE` require an empty `command_sequence`.
* `ADJUST` requires at least one command.
* Step numbers must be consecutive and start from `1`.
* Rotation commands must use `"unit": "degrees"`.
* Translation commands must use `"unit": "meters"`.
* Every command value must be greater than `0` and within the limits above.
* Do not emit zero-value commands or unsupported actions.
* `expected_result` must describe the estimated final pose after every command has executed.
* The sequence must be geometrically consistent with the desired final pose.
* The final corrective rotation must face the robot toward the selected workspace.
* Output must be syntactically valid JSON.
"""


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
        model: str = "gpt-5.6",
        reasoning_effort: str = "high",
        timeout_seconds: float = 180.0,
        codex_bin: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
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


class BasePoseRunner:
    """Capture one observation and obtain one validated model-authored plan."""

    def __init__(
        self,
        config: BasePoseConfig,
        *,
        camera: AlignedRGBDCamera | None = None,
        client: CodexStructuredVisionClient | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        if config.mode not in BASE_POSE_MODES:
            raise ValueError(f"unsupported base-pose mode: {config.mode}")
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
        self.client = client or CodexStructuredVisionClient(
            model=config.model,
            reasoning_effort=config.reasoning_effort,
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
