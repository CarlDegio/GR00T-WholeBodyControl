"""Single-cycle RGB-D perception, Codex policy, and navigation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping

import cv2
import msgpack
import numpy as np
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.utils.inference.object_nav_geometry import (
    FORWARD_SPEED,
    FRAME_COUNT,
    MAX_DIRECT_TRAVEL,
    POLICY_FRAME_INDEX,
    ROTATION_SPEED,
    TARGET_STANDOFF_DISTANCE,
    build_object_nav_commands_from_frames,
)


DEFAULT_SCHEMA_FILENAME = "object_nav_policy.schema.json"
HTTP_PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
ALL_PROXY_KEYS = ("all_proxy", "ALL_PROXY")
POLICY_KEYS = (
    "visual_check",
    "action",
    "bbox_2d",
    "target",
    "target_type",
    "estimated_distance_m",
    "target_center_normalized",
    "target_center_pixel",
    "horizontal_offset_pixel",
    "camera_bearing_deg",
    "rotation_direction",
    "rotation_angle_deg",
    "confidence",
    "distance_confidence",
    "stop_reasoning",
)
REQUIRED_POLICY_KEYS = set(POLICY_KEYS)
TARGET_TYPES = {"global_target", "intermediate_landmark", "traversable_opening"}
ROTATION_DIRECTIONS = {"LEFT", "RIGHT", "CENTERED"}
DEFAULT_POLICY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "visual_check": {"type": "string"},
        "action": {"enum": ["NAVIGATE", "STOP"]},
        "bbox_2d": {
            "type": ["array", "null"],
            "items": {"type": "number", "minimum": 0, "maximum": 1000},
            "minItems": 4,
            "maxItems": 4,
        },
        "target": {"type": "string"},
        "target_type": {
            "enum": [
                "global_target",
                "intermediate_landmark",
                "traversable_opening",
            ]
        },
        "estimated_distance_m": {
            "type": ["number", "null"],
            "exclusiveMinimum": 0,
        },
        "target_center_normalized": {
            "type": ["array", "null"],
            "items": {"type": "number", "minimum": 0, "maximum": 1000},
            "minItems": 2,
            "maxItems": 2,
        },
        "target_center_pixel": {
            "type": ["array", "null"],
            "items": {"type": "number", "minimum": 0},
            "minItems": 2,
            "maxItems": 2,
        },
        "horizontal_offset_pixel": {"type": ["number", "null"]},
        "camera_bearing_deg": {"type": ["number", "null"]},
        "rotation_direction": {
            "enum": ["LEFT", "RIGHT", "CENTERED", None]
        },
        "rotation_angle_deg": {
            "type": ["number", "null"],
            "minimum": 0,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "distance_confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "stop_reasoning": {"type": "string"},
    },
    "required": list(POLICY_KEYS),
    "additionalProperties": False,
}


@dataclass(frozen=True)
class RGBDSnapshot:
    rgb_bgr: np.ndarray
    depth_raw: np.ndarray
    depth_mm: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m: float
    depth_aligned_to: str
    timestamp: float


@dataclass
class ObjectNavConfig:
    mission: str
    global_target: str
    model: str = "gpt-5.6-luna"
    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_timeout_ms: int = 3000
    codex_timeout_seconds: float = 180.0
    min_confidence: float = 0.6
    rotation_speed: float = ROTATION_SPEED
    forward_speed: float = FORWARD_SPEED
    target_standoff_distance: float = TARGET_STANDOFF_DISTANCE
    max_direct_travel: float = MAX_DIRECT_TRAVEL
    output_root: str = "outputs/object_nav"


@dataclass(frozen=True)
class ObjectNavResult:
    outcome: str
    policy: dict[str, Any]
    commands: dict[str, Any]
    geometry: dict[str, Any]
    output_dir: str
    error: str | None = None


class ObjectNavCameraError(RuntimeError):
    """Raised for malformed, missing, or stale composed-camera frames."""


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}_", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: Mapping[str, Any], *, compact: bool = False) -> None:
    contents = json.dumps(value, ensure_ascii=False, indent=None if compact else 2) + "\n"
    _atomic_write_bytes(path, contents.encode("utf-8"))


class ComposedRGBDCamera:
    """Read fresh aligned chest RGB-D frames from the composed camera server."""

    COLOR_KEY = "chest_view"
    DEPTH_KEY = "chest_view_depth"
    SCHEMA_VERSION = 2

    def __init__(self, host: str, port: int, timeout_ms: int = 3000):
        self.timeout_ms = int(timeout_ms)
        self._last_timestamp: float | None = None
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{int(port)}")

    @classmethod
    def decode_payload(cls, payload: Mapping[str, Any]) -> RGBDSnapshot:
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ObjectNavCameraError("unsupported camera schema_version")
        decoded = ImageMessageSchema.deserialize(
            dict(payload), decode_images=True
        ).asdict()
        images = decoded.get("images", {})
        camera_info = decoded.get("camera_info", {})
        timestamps = decoded.get("timestamps", {})
        if cls.COLOR_KEY not in images or cls.DEPTH_KEY not in images:
            raise ObjectNavCameraError("camera payload requires chest_view RGB and depth")
        info = camera_info.get(cls.COLOR_KEY)
        if not isinstance(info, Mapping):
            raise ObjectNavCameraError("chest_view camera_info is missing")
        if info.get("depth_aligned_to") != cls.COLOR_KEY:
            raise ObjectNavCameraError("depth is not aligned to chest_view")

        rgb = np.asarray(images[cls.COLOR_KEY])
        depth_raw = np.asarray(images[cls.DEPTH_KEY])
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ObjectNavCameraError("chest_view must be an RGB image")
        if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
            raise ObjectNavCameraError("chest depth must be uint16")
        if rgb.shape[:2] != depth_raw.shape:
            raise ObjectNavCameraError("aligned chest RGB and depth shapes do not match")

        try:
            numeric = {
                name: float(info[name])
                for name in ("fx", "fy", "cx", "cy", "depth_scale_m")
            }
            timestamp = float(timestamps[cls.COLOR_KEY])
            width = int(info["width"])
            height = int(info["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ObjectNavCameraError("chest camera calibration is incomplete") from exc
        if not all(math.isfinite(value) for value in (*numeric.values(), timestamp)):
            raise ObjectNavCameraError("chest camera calibration must be finite")
        if numeric["fx"] <= 0 or numeric["fy"] <= 0 or numeric["depth_scale_m"] <= 0:
            raise ObjectNavCameraError("focal lengths and depth scale must be positive")
        if (width, height) != (rgb.shape[1], rgb.shape[0]):
            raise ObjectNavCameraError(
                "camera_info dimensions do not match chest images"
            )

        return RGBDSnapshot(
            rgb_bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            depth_raw=depth_raw,
            depth_mm=depth_raw.astype(np.float32)
            * np.float32(numeric["depth_scale_m"] * 1000),
            fx=numeric["fx"],
            fy=numeric["fy"],
            cx=numeric["cx"],
            cy=numeric["cy"],
            depth_scale_m=numeric["depth_scale_m"],
            depth_aligned_to=str(info["depth_aligned_to"]),
            timestamp=timestamp,
        )

    def capture_aligned_rgbd(self, timeout_ms: int | None = None) -> RGBDSnapshot:
        effective_timeout = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        deadline = time.monotonic() + effective_timeout / 1000.0
        while True:
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            if not self._socket.poll(remaining_ms):
                raise ObjectNavCameraError("timed out waiting for fresh chest RGB-D")
            try:
                payload = msgpack.unpackb(self._socket.recv(), raw=False)
                snapshot = self.decode_payload(payload)
            except ObjectNavCameraError:
                raise
            except Exception as exc:
                raise ObjectNavCameraError(
                    "failed to decode composed-camera message"
                ) from exc
            if self._last_timestamp is None or snapshot.timestamp != self._last_timestamp:
                self._last_timestamp = snapshot.timestamp
                return snapshot
            if time.monotonic() >= deadline:
                raise ObjectNavCameraError(
                    "timed out waiting for a newer chest RGB-D frame"
                )

    def close(self) -> None:
        self._socket.close()
        self._context.term()


def _escape_prompt_value(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)[1:-1]


def get_object_nav_policy_prompt(
    mission: str, global_target: str, snapshot: RGBDSnapshot
) -> str:
    """Build the isolated Codex image-policy prompt using live intrinsics."""
    height, width = snapshot.rgb_bgr.shape[:2]
    return f"""You are the visual navigation policy for a Unitree G1 humanoid robot.
Analyse only the supplied current chest-camera image. Visible robot arms, grippers,
body parts, reflections, and shadows are never navigation targets.

MISSION: \"{_escape_prompt_value(mission)}\"
GLOBAL TARGET: \"{_escape_prompt_value(global_target)}\"
IMAGE SIZE: width={width}, height={height}
CAMERA INTRINSICS: fx={snapshot.fx}, fy={snapshot.fy}, cx={snapshot.cx}, cy={snapshot.cy}

Select exactly one target. Prefer the visible global target; otherwise select a useful
intermediate landmark, doorway, passage, or traversable opening. Return a tight bbox
[x1,y1,x2,y2] in normalized [0,1000] coordinates. Estimate distance for diagnostics.
Calculate horizontal bearing from the bbox centre and the supplied fx/cx. Positive
bearing means RIGHT, negative means LEFT, and absolute bearing <=2 degrees is CENTERED.

Return NAVIGATE unless the global target is clearly identified, reached, approximately
centred/directly reachable, and no further forward motion is needed. For a discrete
object, STOP also requires its bbox to occupy at least 20 percent of image height.
Never return STOP merely because the target is absent or uncertain.

Return exactly one JSON object matching the supplied schema, without Markdown or any
text before or after it. Do not output hidden reasoning."""


class CodexBBoxClient:
    """Run a read-only Codex CLI image request and validate its policy JSON."""

    def __init__(
        self,
        *,
        schema_path: str | Path | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout_seconds: float = 180.0,
        codex_bin: str | None = None,
        model: str = "gpt-5.6-sol",
        reasoning_effort: str | None = "high",
        http_proxy_url: str | None = None,
        all_proxy_url: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.schema_path = (
            None if schema_path is None else Path(schema_path).resolve()
        )
        self.runner = runner
        self.timeout_seconds = float(timeout_seconds)
        self.codex_bin = codex_bin or os.environ.get("CODEX_BIN", "codex")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._monotonic = monotonic
        self.last_auth_check_seconds = 0.0
        self.last_api_inference_seconds = 0.0
        self.http_proxy_url = (
            os.environ.get("OBJECT_NAV_CODEX_HTTP_PROXY")
            if http_proxy_url is None
            else http_proxy_url
        )
        self.all_proxy_url = (
            os.environ.get("OBJECT_NAV_CODEX_ALL_PROXY")
            if all_proxy_url is None
            else all_proxy_url
        )

    def _subprocess_env(self) -> dict[str, str]:
        child_env = os.environ.copy()
        for keys, value in (
            (HTTP_PROXY_KEYS, self.http_proxy_url),
            (ALL_PROXY_KEYS, self.all_proxy_url),
        ):
            if value is None:
                continue
            for key in keys:
                if value:
                    child_env[key] = value
                else:
                    child_env.pop(key, None)
        return child_env

    def _check_chatgpt_login(self) -> None:
        result = self.runner(
            [self.codex_bin, "login", "status"],
            capture_output=True,
            text=True,
            timeout=min(self.timeout_seconds, 15),
            check=False,
            env=self._subprocess_env(),
        )
        login_text = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        if result.returncode != 0 or "chatgpt" not in login_text:
            raise RuntimeError("Codex CLI must be logged in with a ChatGPT subscription")

    def _schema_path(self, cwd: str | Path) -> Path:
        if self.schema_path is not None:
            if not self.schema_path.is_file():
                raise FileNotFoundError(
                    f"ObjectNav output schema not found: {self.schema_path}"
                )
            return self.schema_path
        schema_path = Path(cwd).resolve() / DEFAULT_SCHEMA_FILENAME
        _write_json(schema_path, DEFAULT_POLICY_SCHEMA)
        return schema_path

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )

    @classmethod
    def _valid_point(cls, value: Any, *, maximum: float | None = None) -> bool:
        return isinstance(value, list) and len(value) == 2 and all(
            cls._is_finite_number(item)
            and float(item) >= 0
            and (maximum is None or float(item) <= maximum)
            for item in value
        )

    @classmethod
    def _valid_bbox(cls, value: Any) -> bool:
        if not isinstance(value, list) or len(value) != 4:
            return False
        if not all(
            cls._is_finite_number(item) and 0 <= float(item) <= 1000
            for item in value
        ):
            return False
        x1, y1, x2, y2 = (float(item) for item in value)
        return x1 < x2 and y1 < y2

    @classmethod
    def validate_policy(cls, policy: Any) -> dict[str, Any]:
        if not isinstance(policy, dict) or set(policy) != REQUIRED_POLICY_KEYS:
            raise ValueError("Codex policy has an invalid object schema")
        if policy["action"] not in {"NAVIGATE", "STOP"}:
            raise ValueError("Codex policy has an invalid action")
        if policy["target_type"] not in TARGET_TYPES:
            raise ValueError("Codex policy has an invalid target_type")
        if any(
            not isinstance(policy[key], str)
            for key in ("visual_check", "target", "stop_reasoning")
        ):
            raise ValueError("Codex policy text fields must be strings")
        for key in ("confidence", "distance_confidence"):
            if not cls._is_finite_number(policy[key]) or not 0 <= float(
                policy[key]
            ) <= 1:
                raise ValueError(f"Codex policy {key} is invalid")

        bbox = policy["bbox_2d"]
        if bbox is not None and not cls._valid_bbox(bbox):
            raise ValueError("Codex policy bbox_2d is invalid")
        if policy["action"] == "NAVIGATE" and bbox is None:
            raise ValueError("NAVIGATE requires a valid bbox_2d")
        distance = policy["estimated_distance_m"]
        if distance is not None and (
            not cls._is_finite_number(distance) or float(distance) <= 0
        ):
            raise ValueError("Codex policy estimated distance is invalid")
        if policy["target_center_normalized"] is not None and not cls._valid_point(
            policy["target_center_normalized"], maximum=1000
        ):
            raise ValueError("Codex policy normalized target center is invalid")
        if policy["target_center_pixel"] is not None and not cls._valid_point(
            policy["target_center_pixel"]
        ):
            raise ValueError("Codex policy pixel target center is invalid")
        for key in ("horizontal_offset_pixel", "camera_bearing_deg"):
            if policy[key] is not None and not cls._is_finite_number(policy[key]):
                raise ValueError(f"Codex policy {key} is invalid")
        if (
            policy["rotation_direction"] is not None
            and policy["rotation_direction"] not in ROTATION_DIRECTIONS
        ):
            raise ValueError("Codex policy rotation_direction is invalid")
        angle = policy["rotation_angle_deg"]
        if angle is not None and (
            not cls._is_finite_number(angle) or float(angle) < 0
        ):
            raise ValueError("Codex policy rotation_angle_deg is invalid")
        if bbox is not None and any(
            policy[key] is None
            for key in (
                "estimated_distance_m",
                "target_center_normalized",
                "target_center_pixel",
                "horizontal_offset_pixel",
                "camera_bearing_deg",
                "rotation_direction",
                "rotation_angle_deg",
            )
        ):
            raise ValueError("boxable Codex target requires distance and rotation fields")
        if policy["action"] == "STOP":
            if policy["target_type"] != "global_target":
                raise ValueError("STOP is only valid for the global target")
            if not policy["stop_reasoning"].strip():
                raise ValueError("STOP requires stop_reasoning")
        return policy

    def locate(
        self,
        *,
        image_path: str | Path,
        mission: str,
        global_target: str,
        snapshot: RGBDSnapshot,
        cwd: str | Path,
    ) -> dict[str, Any]:
        auth_started = self._monotonic()
        try:
            self._check_chatgpt_login()
        finally:
            self.last_auth_check_seconds = self._monotonic() - auth_started
        image_path = Path(image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"ObjectNav RGB image not found: {image_path}")
        schema_path = self._schema_path(cwd)
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
            "--image",
            str(image_path),
            "--output-schema",
            str(schema_path),
            get_object_nav_policy_prompt(mission, global_target, snapshot),
        ]
        if self.reasoning_effort is not None:
            model_index = command.index("--image")
            command[model_index:model_index] = [
                "--config",
                f'model_reasoning_effort="{self.reasoning_effort}"',
            ]
        api_started = self._monotonic()
        try:
            result = self.runner(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                cwd=str(cwd),
                env=self._subprocess_env(),
            )
        finally:
            self.last_api_inference_seconds = self._monotonic() - api_started
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "unknown error").strip()
            raise RuntimeError(f"Codex ObjectNav policy failed: {message}")
        try:
            return self.validate_policy(json.loads(result.stdout))
        except json.JSONDecodeError as exc:
            raise ValueError("Codex ObjectNav output is not valid JSON") from exc


def _draw_bbox(rgb_bgr: np.ndarray, bbox: Any) -> np.ndarray:
    image = rgb_bgr.copy()
    if not isinstance(bbox, list) or len(bbox) != 4:
        return image
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (
        int(round(float(bbox[0]) * (width - 1) / 1000.0)),
        int(round(float(bbox[1]) * (height - 1) / 1000.0)),
        int(round(float(bbox[2]) * (width - 1) / 1000.0)),
        int(round(float(bbox[3]) * (height - 1) / 1000.0)),
    )
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def _save_raw_depth_outputs(
    snapshots: list[RGBDSnapshot], output_dir: Path
) -> None:
    frame_info = []
    for frame_index, snapshot in enumerate(snapshots, start=1):
        encoded, depth_png = cv2.imencode(".png", snapshot.depth_raw)
        if not encoded:
            raise RuntimeError("failed to encode ObjectNav raw depth")
        filename = f"depth_raw_{frame_index:02d}.png"
        _atomic_write_bytes(output_dir / filename, depth_png.tobytes())
        frame_info.append(
            {
                "frame_index": frame_index,
                "depth_file": filename,
                "timestamp": snapshot.timestamp,
                "width": int(snapshot.depth_raw.shape[1]),
                "height": int(snapshot.depth_raw.shape[0]),
                "fx": snapshot.fx,
                "fy": snapshot.fy,
                "cx": snapshot.cx,
                "cy": snapshot.cy,
                "depth_scale_m": snapshot.depth_scale_m,
                "depth_aligned_to": snapshot.depth_aligned_to,
            }
        )
    _write_json(
        output_dir / "camera_info.json",
        {
            "frame_count": FRAME_COUNT,
            "policy_rgb_frame_index": POLICY_FRAME_INDEX + 1,
            "depth_dtype": "uint16",
            "depth_unit_formula": "depth_m = depth_raw * depth_scale_m",
            "frames": frame_info,
        },
    )


class ObjectNavRunner:
    """Capture one RGB-D burst and return one fail-closed navigation result."""

    def __init__(
        self,
        config: ObjectNavConfig,
        camera: Any | None = None,
        codex: CodexBBoxClient | None = None,
    ):
        if not config.mission.strip() or not config.global_target.strip():
            raise ValueError("mission and global_target are required")
        self.config = config
        self._owns_camera = camera is None
        self.camera = camera or ComposedRGBDCamera(
            config.camera_host, config.camera_port, config.camera_timeout_ms
        )
        self.codex = codex or CodexBBoxClient(
            model=config.model,
            reasoning_effort=None,
            timeout_seconds=config.codex_timeout_seconds,
        )

    def run_once(self, *, iteration: int = 1) -> ObjectNavResult:
        total_started = time.monotonic()
        timing = {
            "camera_rgbd": 0.0,
            "image_io": 0.0,
            "auth_check": 0.0,
            "api_inference": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
        }
        postprocess_started: float | None = None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path(self.config.output_root)
            / f"{timestamp}_{os.getpid()}"
            / f"iteration_{iteration:04d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        commands: dict[str, Any] = {"commands": []}
        policy: dict[str, Any] = {}
        geometry: dict[str, Any] = {"status": "failed"}
        error: str | None = None
        outcome = "FAILED"
        try:
            camera_started = time.monotonic()
            snapshots = [
                self.camera.capture_aligned_rgbd() for _ in range(FRAME_COUNT)
            ]
            timing["camera_rgbd"] = time.monotonic() - camera_started
            image_io_started = time.monotonic()
            _save_raw_depth_outputs(snapshots, output_dir)
            snapshot = snapshots[POLICY_FRAME_INDEX]
            input_path = output_dir / "object_nav_input.png"
            if not cv2.imwrite(str(input_path), snapshot.rgb_bgr):
                raise RuntimeError("failed to save ObjectNav RGB input")
            timing["image_io"] = time.monotonic() - image_io_started
            policy_call_started = time.monotonic()
            try:
                policy = self.codex.locate(
                    image_path=input_path,
                    mission=self.config.mission,
                    global_target=self.config.global_target,
                    snapshot=snapshot,
                    cwd=output_dir,
                )
            finally:
                policy_call_seconds = time.monotonic() - policy_call_started
                timing["auth_check"] = float(
                    getattr(self.codex, "last_auth_check_seconds", 0.0)
                )
                timing["api_inference"] = float(
                    getattr(
                        self.codex,
                        "last_api_inference_seconds",
                        max(0.0, policy_call_seconds - timing["auth_check"]),
                    )
                )
            postprocess_started = time.monotonic()
            _write_json(output_dir / "object_nav_policy.json", policy)
            if not cv2.imwrite(
                str(output_dir / "object_nav_bbox_vis.png"),
                _draw_bbox(snapshot.rgb_bgr, policy.get("bbox_2d")),
            ):
                raise RuntimeError("failed to save ObjectNav bbox visualization")

            if float(policy["confidence"]) < self.config.min_confidence:
                outcome = "REJECTED"
                geometry = {
                    "status": "rejected",
                    "reason": "policy confidence below threshold",
                    "confidence": float(policy["confidence"]),
                }
            elif policy["action"] == "STOP":
                outcome = "STOP"
                geometry = {
                    "status": "stopped",
                    "reason": policy["stop_reasoning"],
                }
            else:
                commands, geometry = build_object_nav_commands_from_frames(
                    policy,
                    [(item.depth_mm, item.fx, item.cx) for item in snapshots],
                    rotation_speed=self.config.rotation_speed,
                    forward_speed=self.config.forward_speed,
                    target_standoff_distance=self.config.target_standoff_distance,
                    max_direct_travel=self.config.max_direct_travel,
                )
                outcome = "NAVIGATE"
                geometry["status"] = "ok"

            geometry["codex_prediction"] = {
                "distance_m": policy.get("estimated_distance_m"),
                "rotation_direction": policy.get("rotation_direction"),
                "rotation_angle_deg": policy.get("rotation_angle_deg"),
            }
            if outcome == "NAVIGATE":
                geometry["depth_camera_measurement"] = {
                    "distance_m": geometry["mean_range"],
                    "rotation_angle_deg": abs(float(geometry["angle_deg"])),
                }
        except Exception as exc:
            error = str(exc)
            geometry = {"status": "failed", "error": error}
            print(f"[ObjectNav] {error}", file=os.sys.stderr, flush=True)
        finally:
            if postprocess_started is not None:
                timing["postprocess"] = time.monotonic() - postprocess_started
            timing["total"] = time.monotonic() - total_started
            geometry["timing_s"] = timing
            if not (output_dir / "object_nav_policy.json").exists():
                _write_json(output_dir / "object_nav_policy.json", policy)
            _write_json(output_dir / "object_nav_geometry.json", geometry)
            _write_json(
                output_dir / "object_nav_commands.json", commands, compact=True
            )

        print(f"[ObjectNav] outcome={outcome} output={output_dir}", file=os.sys.stderr)
        if outcome == "NAVIGATE":
            prediction = geometry["codex_prediction"]
            measured = geometry["depth_camera_measurement"]
            print(
                "[ObjectNav] Codex vs depth: "
                f"distance {prediction['distance_m']} vs {measured['distance_m']}m; "
                f"rotation {prediction['rotation_direction']} "
                f"{prediction['rotation_angle_deg']} vs "
                f"{measured['rotation_angle_deg']}deg",
                file=os.sys.stderr,
            )
        return ObjectNavResult(
            outcome=outcome,
            policy=policy,
            commands=commands,
            geometry=geometry,
            output_dir=str(output_dir),
            error=error,
        )

    def warmup(self) -> ObjectNavResult:
        """Execute one real policy request whose motion result is never consumed."""
        return self.run_once(iteration=0)

    def close(self) -> None:
        if self._owns_camera:
            self.camera.close()
