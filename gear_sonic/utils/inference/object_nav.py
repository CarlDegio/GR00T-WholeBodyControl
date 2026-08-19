"""Single-cycle RGB-D perception, Qwen-VL policy, and navigation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import json
import math
import os
import tempfile
import time
from typing import Any, Callable, Mapping

import cv2
import msgpack
import numpy as np
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.runtime.client import SensorGatewayClient, SensorGatewayClientError
from gear_sonic.runtime.snapshot import SnapshotRequest, TimestampBasis
from gear_sonic.utils.inference.object_nav_geometry import (
    FORWARD_SPEED,
    FRAME_COUNT,
    MAX_DIRECT_TRAVEL,
    POLICY_FRAME_INDEX,
    ROTATION_SPEED,
    TARGET_STANDOFF_DISTANCE,
    build_object_nav_commands_from_frames,
)


DEFAULT_QWENVL_MODEL = "qwen3-vl-32b-instruct"
DEFAULT_QWENVL_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWENVL_PROXY_URL = "http://127.0.0.1:7890"
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
QWENVL_POLICY_KEYS = {
    "action",
    "bbox_2d",
    "target",
    "target_type",
    "confidence",
    "stop_reasoning",
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
    qwenvl_model: str = DEFAULT_QWENVL_MODEL
    qwenvl_base_url: str = DEFAULT_QWENVL_BASE_URL
    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_timeout_ms: int = 3000
    qwenvl_timeout_seconds: float = 180.0
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


class SensorGatewayRGBDCamera:
    """Read chest RGB and Depth Anything metric depth from shared memory."""

    RGB_STREAM = "camera/chest_view"
    DEPTH_STREAM = "derived/depth_anything/chest_view"

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_ms: int = 3000,
        request_timeout_ms: int = 100,
        max_age_ms: float = 1000.0,
        max_skew_ms: float = 5.0,
        client: SensorGatewayClient | None = None,
    ) -> None:
        self.timeout_ms = int(timeout_ms)
        self.max_age_ms = float(max_age_ms)
        self.max_skew_ms = float(max_skew_ms)
        self.client = client or SensorGatewayClient(
            endpoint, request_timeout_ms=int(request_timeout_ms)
        )
        self._owns_client = client is None
        self._last_timestamp_ns: int | None = None

    @staticmethod
    def _decode(snapshot) -> RGBDSnapshot:
        rgb_frame = snapshot.snapshot.frames[SensorGatewayRGBDCamera.RGB_STREAM]
        depth_frame = snapshot.snapshot.frames[SensorGatewayRGBDCamera.DEPTH_STREAM]
        rgb = np.asarray(snapshot.arrays[SensorGatewayRGBDCamera.RGB_STREAM])
        depth_raw = np.asarray(snapshot.arrays[SensorGatewayRGBDCamera.DEPTH_STREAM])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ObjectNavCameraError(
                f"Gateway chest RGB must be HxWx3 uint8, got {rgb.shape} {rgb.dtype}"
            )
        if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
            raise ObjectNavCameraError(
                f"Gateway Depth Anything depth must be HxW uint16, got "
                f"{depth_raw.shape} {depth_raw.dtype}"
            )
        if rgb.shape[:2] != depth_raw.shape:
            raise ObjectNavCameraError(
                "Gateway RGB and Depth Anything depth shapes do not match"
            )
        info = dict(depth_frame.attributes.get("camera_info", {}))
        if not info:
            info = dict(rgb_frame.attributes.get("camera_info", {}))
        try:
            numeric = {
                name: float(info[name])
                for name in ("fx", "fy", "cx", "cy", "depth_scale_m")
            }
            width = int(info["width"])
            height = int(info["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ObjectNavCameraError("Gateway RGB-D calibration is incomplete") from exc
        if info.get("depth_aligned_to") != "chest_view":
            raise ObjectNavCameraError(
                "Gateway Depth Anything depth is not aligned to chest_view"
            )
        depth_source = str(
            depth_frame.attributes.get("depth_source")
            or info.get("depth_source", "")
        )
        if not depth_source.startswith("depth-anything-v2-metric-"):
            raise ObjectNavCameraError(
                f"Gateway depth source is not metric Depth Anything: {depth_source!r}"
            )
        if info.get("inference_owner") != "lavira":
            raise ObjectNavCameraError(
                "Gateway Depth Anything frame is not owned by LaViRA"
            )
        if (width, height) != (rgb.shape[1], rgb.shape[0]):
            raise ObjectNavCameraError("Gateway camera_info dimensions do not match RGB-D")
        timestamp_ns = depth_frame.source_timestamp_ns or rgb_frame.source_timestamp_ns
        timestamp = timestamp_ns * 1.0e-9 if timestamp_ns > 0 else time.time()
        return RGBDSnapshot(
            rgb_bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            depth_raw=depth_raw.copy(),
            depth_mm=depth_raw.astype(np.float32)
            * np.float32(numeric["depth_scale_m"] * 1000.0),
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
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                snapshot = self.client.read_snapshot(
                    SnapshotRequest(
                        streams=(self.RGB_STREAM, self.DEPTH_STREAM),
                        max_age_ms=self.max_age_ms,
                        max_skew_ms=self.max_skew_ms,
                        timestamp_basis=TimestampBasis.SOURCE,
                    ),
                    retries=1,
                )
                depth_frame = snapshot.snapshot.frames[self.DEPTH_STREAM]
                timestamp_ns = depth_frame.source_timestamp_ns
                if self._last_timestamp_ns is None or timestamp_ns != self._last_timestamp_ns:
                    decoded = self._decode(snapshot)
                    self._last_timestamp_ns = timestamp_ns
                    return decoded
            except (SensorGatewayClientError, ObjectNavCameraError) as exc:
                last_error = exc
            time.sleep(0.01)
        raise ObjectNavCameraError(
            f"timed out waiting for fresh Gateway RGB-D: {last_error or 'no new frame'}"
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


def _escape_prompt_value(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)[1:-1]


def get_qwenvl_policy_prompt(
    mission: str, global_target: str, snapshot: RGBDSnapshot
) -> str:
    """Build Qwen-VL's explicit minimal JSON contract."""
    height, width = snapshot.rgb_bgr.shape[:2]
    return f"""You are the visual navigation policy for a Unitree G1 humanoid robot.
Analyse only the supplied current chest-camera image. Robot body parts, reflections,
and shadows are never navigation targets.

MISSION: \"{_escape_prompt_value(mission)}\"
GLOBAL TARGET: \"{_escape_prompt_value(global_target)}\"
IMAGE SIZE: width={width}, height={height}

Select exactly one target. Prefer the visible global target; otherwise select a useful
intermediate landmark or traversable opening. Return NAVIGATE unless the global target
is clearly reached and no further forward motion is needed. Never return STOP merely
because the target is absent or uncertain.

Return one JSON object with exactly these fields:
{{
  \"action\": \"NAVIGATE\" or \"STOP\",
  \"bbox_2d\": [x1, y1, x2, y2] or null,
  \"target\": string,
  \"target_type\": \"global_target\", \"intermediate_landmark\", or \"traversable_opening\",
  \"confidence\": number from 0 to 1,
  \"stop_reasoning\": string
}}

For NAVIGATE, bbox_2d is required, ordered [left, top, right, bottom], and every
coordinate is normalized to [0,1000]. For STOP, bbox_2d must be null, target_type must
be global_target, and stop_reasoning must be non-empty. Do not output target_center,
pixel offsets, distance, bearing, rotation direction, rotation angle, Markdown, or
hidden reasoning."""


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _valid_point(value: Any, *, maximum: float | None = None) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(
        _is_finite_number(item)
        and float(item) >= 0
        and (maximum is None or float(item) <= maximum)
        for item in value
    )


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(
        _is_finite_number(item) and 0 <= float(item) <= 1000
        for item in value
    ):
        return False
    x1, y1, x2, y2 = (float(item) for item in value)
    return x1 < x2 and y1 < y2


def validate_object_nav_policy(policy: Any) -> dict[str, Any]:
    """Validate the canonical policy shared by Qwen-VL and geometry code."""

    if not isinstance(policy, dict) or set(policy) != REQUIRED_POLICY_KEYS:
        raise ValueError("ObjectNav policy has an invalid object schema")
    if policy["action"] not in {"NAVIGATE", "STOP"}:
        raise ValueError("ObjectNav policy has an invalid action")
    if policy["target_type"] not in TARGET_TYPES:
        raise ValueError("ObjectNav policy has an invalid target_type")
    if any(
        not isinstance(policy[key], str)
        for key in ("visual_check", "target", "stop_reasoning")
    ):
        raise ValueError("ObjectNav policy text fields must be strings")
    for key in ("confidence", "distance_confidence"):
        if not _is_finite_number(policy[key]) or not 0 <= float(
            policy[key]
        ) <= 1:
            raise ValueError(f"ObjectNav policy {key} is invalid")

    bbox = policy["bbox_2d"]
    if bbox is not None and not _valid_bbox(bbox):
        raise ValueError("ObjectNav policy bbox_2d is invalid")
    if policy["action"] == "NAVIGATE" and bbox is None:
        raise ValueError("NAVIGATE requires a valid bbox_2d")
    distance = policy["estimated_distance_m"]
    if distance is not None and (
        not _is_finite_number(distance) or float(distance) <= 0
    ):
        raise ValueError("ObjectNav policy estimated distance is invalid")
    if policy["target_center_normalized"] is not None and not _valid_point(
        policy["target_center_normalized"], maximum=1000
    ):
        raise ValueError("ObjectNav policy normalized target center is invalid")
    if policy["target_center_pixel"] is not None and not _valid_point(
        policy["target_center_pixel"]
    ):
        raise ValueError("ObjectNav policy pixel target center is invalid")
    for key in ("horizontal_offset_pixel", "camera_bearing_deg"):
        if policy[key] is not None and not _is_finite_number(policy[key]):
            raise ValueError(f"ObjectNav policy {key} is invalid")
    if (
        policy["rotation_direction"] is not None
        and policy["rotation_direction"] not in ROTATION_DIRECTIONS
    ):
        raise ValueError("ObjectNav policy rotation_direction is invalid")
    angle = policy["rotation_angle_deg"]
    if angle is not None and (
        not _is_finite_number(angle) or float(angle) < 0
    ):
        raise ValueError("ObjectNav policy rotation_angle_deg is invalid")
    if bbox is not None and any(
        policy[key] is None
        for key in (
            "target_center_normalized",
            "target_center_pixel",
            "horizontal_offset_pixel",
            "camera_bearing_deg",
            "rotation_direction",
            "rotation_angle_deg",
        )
    ):
        raise ValueError(
            "boxable ObjectNav target requires distance and rotation fields"
        )
    if policy["action"] == "STOP":
        if policy["target_type"] != "global_target":
            raise ValueError("STOP is only valid for the global target")
        if not policy["stop_reasoning"].strip():
            raise ValueError("STOP requires stop_reasoning")
    return policy


class QwenVLBBoxClient:
    """Call Qwen-VL through DashScope's OpenAI-compatible chat API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_QWENVL_MODEL,
        base_url: str = DEFAULT_QWENVL_BASE_URL,
        timeout_seconds: float = 180.0,
        api_key: str | None = None,
        proxy_url: str | None = DEFAULT_QWENVL_PROXY_URL,
        client: Any | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = float(timeout_seconds)
        self._monotonic = monotonic
        self.last_auth_check_seconds = 0.0
        self.last_api_inference_seconds = 0.0

        if client is not None:
            self.client = client
            return
        key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError(
                "Qwen-VL requires the DASHSCOPE_API_KEY environment variable"
            )
        try:
            from openai import OpenAI
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "Qwen-VL requires the openai and httpx packages; install the inference extra"
            ) from exc
        http_client = httpx.Client(proxy=proxy_url, trust_env=False)
        self.client = OpenAI(
            api_key=key,
            base_url=base_url,
            http_client=http_client,
        )

    @staticmethod
    def _parse_json_content(content: Any) -> dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Qwen-VL ObjectNav output is empty")
        text = content.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Qwen-VL ObjectNav output is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Qwen-VL ObjectNav output must be a JSON object")
        return payload

    @classmethod
    def _normalize_policy(
        cls, payload: dict[str, Any], snapshot: RGBDSnapshot
    ) -> dict[str, Any]:
        if set(payload) != QWENVL_POLICY_KEYS:
            raise ValueError("Qwen-VL policy has an invalid object schema")
        action = payload["action"]
        if action not in {"NAVIGATE", "STOP"}:
            raise ValueError("Qwen-VL policy has an invalid action")
        if payload["target_type"] not in TARGET_TYPES:
            raise ValueError("Qwen-VL policy has an invalid target_type")
        if not isinstance(payload["target"], str) or not isinstance(
            payload["stop_reasoning"], str
        ):
            raise ValueError("Qwen-VL policy text fields must be strings")
        confidence = payload["confidence"]
        if not _is_finite_number(confidence) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("Qwen-VL policy confidence is invalid")

        bbox = payload["bbox_2d"]
        if action == "NAVIGATE" and not _valid_bbox(bbox):
            raise ValueError("Qwen-VL NAVIGATE requires a valid bbox_2d")
        if action == "STOP":
            if bbox is not None:
                raise ValueError("Qwen-VL STOP requires bbox_2d=null")
            if payload["target_type"] != "global_target":
                raise ValueError("Qwen-VL STOP is only valid for the global target")
            if not payload["stop_reasoning"].strip():
                raise ValueError("Qwen-VL STOP requires stop_reasoning")

        derived: dict[str, Any] = {
            "estimated_distance_m": None,
            "target_center_normalized": None,
            "target_center_pixel": None,
            "horizontal_offset_pixel": None,
            "camera_bearing_deg": None,
            "rotation_direction": None,
            "rotation_angle_deg": None,
        }
        if bbox is not None:
            x1, y1, x2, y2 = map(float, bbox)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            height, width = snapshot.rgb_bgr.shape[:2]
            pixel_x = center_x * (width - 1) / 1000.0
            pixel_y = center_y * (height - 1) / 1000.0
            offset = pixel_x - float(snapshot.cx)
            bearing = math.degrees(math.atan2(offset, float(snapshot.fx)))
            if abs(bearing) <= 2.0:
                direction = "CENTERED"
            else:
                direction = "RIGHT" if bearing > 0.0 else "LEFT"
            derived.update(
                {
                    "target_center_normalized": [center_x, center_y],
                    "target_center_pixel": [round(pixel_x, 3), round(pixel_y, 3)],
                    "horizontal_offset_pixel": round(offset, 3),
                    "camera_bearing_deg": round(bearing, 3),
                    "rotation_direction": direction,
                    "rotation_angle_deg": round(abs(bearing), 3),
                }
            )

        canonical = {
            "visual_check": f"Qwen-VL selected {payload['target']}",
            **payload,
            **derived,
            "distance_confidence": 0.0,
        }
        return validate_object_nav_policy(canonical)

    def locate(
        self,
        *,
        image_path: str | Path,
        mission: str,
        global_target: str,
        snapshot: RGBDSnapshot,
        cwd: str | Path,
    ) -> dict[str, Any]:
        del cwd
        image_path = Path(image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"ObjectNav RGB image not found: {image_path}")
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                    {
                        "type": "text",
                        "text": get_qwenvl_policy_prompt(
                            mission, global_target, snapshot
                        ),
                    },
                ],
            }
        ]
        started = self._monotonic()
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=self.timeout_seconds,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
        finally:
            self.last_api_inference_seconds = self._monotonic() - started
        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("Qwen-VL ObjectNav response is malformed") from exc
        return self._normalize_policy(self._parse_json_content(content), snapshot)


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
        policy_client: Any | None = None,
        *,
        own_camera: bool | None = None,
    ):
        if not config.mission.strip() or not config.global_target.strip():
            raise ValueError("mission and global_target are required")
        self.config = config
        self._owns_camera = camera is None if own_camera is None else bool(own_camera)
        self.camera = camera or ComposedRGBDCamera(
            config.camera_host, config.camera_port, config.camera_timeout_ms
        )
        self.policy_client = (
            policy_client
            if policy_client is not None
            else QwenVLBBoxClient(
                model=config.qwenvl_model,
                base_url=config.qwenvl_base_url,
                timeout_seconds=config.qwenvl_timeout_seconds,
            )
        )

    def run_once(
        self,
        *,
        iteration: int = 1,
        rgbd_capture_complete: Callable[[], None] | None = None,
    ) -> ObjectNavResult:
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
            snapshots = []
            try:
                snapshots = [
                    self.camera.capture_aligned_rgbd() for _ in range(FRAME_COUNT)
                ]
            finally:
                if rgbd_capture_complete is not None:
                    rgbd_capture_complete()
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
                policy = self.policy_client.locate(
                    image_path=input_path,
                    mission=self.config.mission,
                    global_target=self.config.global_target,
                    snapshot=snapshot,
                    cwd=output_dir,
                )
            finally:
                policy_call_seconds = time.monotonic() - policy_call_started
                timing["auth_check"] = float(
                    getattr(self.policy_client, "last_auth_check_seconds", 0.0)
                )
                timing["api_inference"] = float(
                    getattr(
                        self.policy_client,
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

            geometry["qwen_prediction"] = {
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
            prediction = geometry["qwen_prediction"]
            measured = geometry["depth_camera_measurement"]
            print(
                "[ObjectNav] Qwen-VL vs depth: "
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
