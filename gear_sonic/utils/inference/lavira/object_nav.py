"""Single-cycle RGB-D perception, Qwen-VL policy, and navigation diagnostics."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import logging
import math
import os
import time
from typing import Any, Callable

import cv2
import numpy as np

from gear_sonic.runtime.gateway.sensor_client import (
    SensorGatewayClient,
    SensorGatewayClientError,
)
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest, TimestampBasis
from gear_sonic.utils.inference.lavira.geometry import (
    FRAME_COUNT,
    MAX_DIRECT_TRAVEL,
    POLICY_FRAME_INDEX,
    build_object_nav_geometry_from_frames,
)

LOGGER = logging.getLogger("sonic.lavira")


DEFAULT_QWENVL_MODEL = "qwen3-vl-32b-instruct"
DEFAULT_QWENVL_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWENVL_PROXY_URL = "http://127.0.0.1:7890"
TARGET_TYPES = {"global_target", "intermediate_landmark", "traversable_opening"}
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
    depth_mm: np.ndarray
    fx: float
    cx: float


@dataclass
class ObjectNavConfig:
    mission: str
    global_target: str
    qwenvl_model: str = DEFAULT_QWENVL_MODEL
    qwenvl_base_url: str = DEFAULT_QWENVL_BASE_URL
    qwenvl_timeout_seconds: float = 180.0
    min_confidence: float = 0.6
    max_direct_travel: float = MAX_DIRECT_TRAVEL


@dataclass(frozen=True)
class ObjectNavResult:
    outcome: str
    policy: dict[str, Any]
    geometry: dict[str, Any]
    error: str | None = None


class ObjectNavCameraError(RuntimeError):
    """Raised for malformed, missing, or stale Gateway RGB-D frames."""


class SensorGatewayRGBDCamera:
    """Read chest RGB and Depth Anything metric depth from shared memory."""

    RGB_STREAM = "camera/chest_view"
    DEPTH_STREAM = "derived/depth_anything/chest_view"
    ODOMETRY_STREAM = "ros/odometry"

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
        self._last_rgb_timestamp_ns: int | None = None
        self._last_rgb_timestamp_by_stream: dict[str, int] = {}
        self._expected_generation: int | None = None
        self._expected_skill_id: int | None = None
        self._expected_segment_id: int | None = None

    @staticmethod
    def _decode(
        snapshot,
        *,
        expected_generation: int | None = None,
        expected_skill_id: int | None = None,
        expected_segment_id: int | None = None,
    ) -> RGBDSnapshot:
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
                for name in ("fx", "cx", "depth_scale_m")
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
        if expected_generation is not None and int(
            info.get("inference_generation", -1)
        ) != int(expected_generation):
            raise ObjectNavCameraError(
                "Gateway Depth Anything frame belongs to a stale generation"
            )
        if expected_segment_id is not None and int(
            info.get("inference_segment_id", -1)
        ) != int(expected_segment_id):
            raise ObjectNavCameraError(
                "Gateway Depth Anything frame belongs to a stale segment"
            )
        if expected_skill_id is not None and int(
            info.get("inference_skill_id", 0)
        ) != int(expected_skill_id):
            raise ObjectNavCameraError(
                "Gateway Depth Anything frame belongs to a stale skill"
            )
        if (width, height) != (rgb.shape[1], rgb.shape[0]):
            raise ObjectNavCameraError("Gateway camera_info dimensions do not match RGB-D")
        return RGBDSnapshot(
            rgb_bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            depth_mm=depth_raw.astype(np.float32)
            * np.float32(numeric["depth_scale_m"] * 1000.0),
            fx=numeric["fx"],
            cx=numeric["cx"],
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
                    decoded = self._decode(
                        snapshot,
                        expected_generation=self._expected_generation,
                        expected_skill_id=self._expected_skill_id,
                        expected_segment_id=self._expected_segment_id,
                    )
                    self._last_timestamp_ns = timestamp_ns
                    return decoded
            except (SensorGatewayClientError, ObjectNavCameraError) as exc:
                last_error = exc
            time.sleep(0.01)
        raise ObjectNavCameraError(
            f"timed out waiting for fresh Gateway RGB-D: {last_error or 'no new frame'}"
        )

    def begin_depth_lease(
        self, generation: int, skill_id: int, segment_id: int
    ) -> None:
        self._expected_generation = int(generation)
        self._expected_skill_id = int(skill_id)
        self._expected_segment_id = int(segment_id)
        self._last_timestamp_ns = None

    def capture_rgb(
        self,
        timeout_ms: int | None = None,
        *,
        camera_stream: str = "chest_view",
    ) -> np.ndarray:
        """Capture one fresh RGB frame without acquiring a depth lease."""

        stream_name = str(camera_stream).strip()
        if not stream_name:
            raise ObjectNavCameraError("Gateway RGB camera stream is empty")
        stream = (
            stream_name if stream_name.startswith("camera/")
            else f"camera/{stream_name}"
        )
        effective_timeout = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        deadline = time.monotonic() + effective_timeout / 1000.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                snapshot = self.client.read_snapshot(
                    SnapshotRequest(
                        streams=(stream,),
                        max_age_ms=self.max_age_ms,
                        max_skew_ms=0.0,
                        timestamp_basis=TimestampBasis.SOURCE,
                    ),
                    retries=1,
                )
                frame = snapshot.snapshot.frames[stream]
                timestamp_ns = frame.source_timestamp_ns
                rgb = np.asarray(snapshot.arrays[stream])
                if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                    raise ObjectNavCameraError(
                        f"Gateway chest RGB must be HxWx3 uint8, got "
                        f"{rgb.shape} {rgb.dtype}"
                    )
                if (
                    self._last_rgb_timestamp_by_stream.get(stream) != timestamp_ns
                ):
                    self._last_rgb_timestamp_by_stream[stream] = timestamp_ns
                    if stream == self.RGB_STREAM:
                        self._last_rgb_timestamp_ns = timestamp_ns
                    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except (SensorGatewayClientError, ObjectNavCameraError) as exc:
                last_error = exc
            time.sleep(0.01)
        raise ObjectNavCameraError(
            f"timed out waiting for fresh Gateway RGB {stream!r}: "
            f"{last_error or 'no new frame'}"
        )

    def current_pose(self) -> tuple[float, float, float]:
        """Return the latest Fast-LIO x/y/yaw for exploration-loop memory."""
        try:
            snapshot = self.client.read_snapshot(
                SnapshotRequest(
                    streams=(self.ODOMETRY_STREAM,),
                    max_age_ms=self.max_age_ms,
                    max_skew_ms=0.0,
                    timestamp_basis=TimestampBasis.SOURCE,
                ),
                retries=1,
            )
            state = np.asarray(
                snapshot.arrays[self.ODOMETRY_STREAM], dtype=np.float64
            ).reshape(-1)
        except SensorGatewayClientError as exc:
            raise ObjectNavCameraError("Fast-LIO odometry is unavailable") from exc
        if state.shape != (13,) or not np.all(np.isfinite(state[:7])):
            raise ObjectNavCameraError("Fast-LIO odometry is malformed")
        x, y, z, w = map(float, state[3:7])
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        return float(state[0]), float(state[1]), yaw

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

    if not isinstance(policy, dict) or set(policy) != QWENVL_POLICY_KEYS:
        raise ValueError("ObjectNav policy has an invalid object schema")
    if policy["action"] not in {"NAVIGATE", "STOP"}:
        raise ValueError("ObjectNav policy has an invalid action")
    if policy["target_type"] not in TARGET_TYPES:
        raise ValueError("ObjectNav policy has an invalid target_type")
    if any(not isinstance(policy[key], str) for key in ("target", "stop_reasoning")):
        raise ValueError("ObjectNav policy text fields must be strings")
    if not _is_finite_number(policy["confidence"]) or not 0 <= float(
        policy["confidence"]
    ) <= 1:
        raise ValueError("ObjectNav policy confidence is invalid")

    bbox = policy["bbox_2d"]
    if bbox is not None and not _valid_bbox(bbox):
        raise ValueError("ObjectNav policy bbox_2d is invalid")
    if policy["action"] == "NAVIGATE" and bbox is None:
        raise ValueError("NAVIGATE requires a valid bbox_2d")
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
        self.timeout_seconds = float(timeout_seconds)
        self._monotonic = monotonic
        self.last_image_encode_seconds = 0.0
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
            import httpx
            from openai import OpenAI
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

    def locate(
        self,
        *,
        mission: str,
        global_target: str,
        snapshot: RGBDSnapshot,
    ) -> dict[str, Any]:
        started = self._monotonic()
        try:
            encoded, image_png = cv2.imencode(".png", snapshot.rgb_bgr)
            if not encoded:
                raise RuntimeError("failed to encode ObjectNav RGB input")
            image_data = base64.b64encode(image_png.tobytes()).decode("ascii")
        finally:
            self.last_image_encode_seconds = self._monotonic() - started
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
        return validate_object_nav_policy(self._parse_json_content(content))


class ObjectNavRunner:
    """Capture one RGB-D burst and return one fail-closed navigation result."""

    def __init__(
        self,
        config: ObjectNavConfig,
        camera: Any,
        policy_client: Any | None = None,
        *,
        own_camera: bool | None = None,
    ):
        if not config.mission.strip() or not config.global_target.strip():
            raise ValueError("mission and global_target are required")
        self.config = config
        self._owns_camera = False if own_camera is None else bool(own_camera)
        self.camera = camera
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
        rgbd_capture_complete: Callable[[], None] | None = None,
    ) -> ObjectNavResult:
        total_started = time.monotonic()
        timing = {
            "camera_rgbd": 0.0,
            "image_encode": 0.0,
            "api_inference": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
        }
        postprocess_started: float | None = None
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
            snapshot = snapshots[POLICY_FRAME_INDEX]
            policy_call_started = time.monotonic()
            try:
                policy = self.policy_client.locate(
                    mission=self.config.mission,
                    global_target=self.config.global_target,
                    snapshot=snapshot,
                )
            finally:
                policy_call_seconds = time.monotonic() - policy_call_started
                timing["image_encode"] = float(
                    getattr(self.policy_client, "last_image_encode_seconds", 0.0)
                )
                timing["api_inference"] = float(
                    getattr(
                        self.policy_client,
                        "last_api_inference_seconds",
                        max(0.0, policy_call_seconds - timing["image_encode"]),
                    )
                )
            postprocess_started = time.monotonic()
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
                geometry = build_object_nav_geometry_from_frames(
                    policy,
                    [(item.depth_mm, item.fx, item.cx) for item in snapshots],
                    max_direct_travel=self.config.max_direct_travel,
                )
                outcome = "NAVIGATE"
                geometry["status"] = "ok"

        except Exception as exc:
            error = str(exc)
            geometry = {"status": "failed", "error": error}
            LOGGER.exception("object_nav failed")
        finally:
            if postprocess_started is not None:
                timing["postprocess"] = time.monotonic() - postprocess_started
            timing["total"] = time.monotonic() - total_started
            geometry["timing_s"] = timing
        LOGGER.info(
            "object_nav outcome=%s policy=%s geometry=%s",
            outcome,
            policy,
            geometry,
        )
        return ObjectNavResult(
            outcome=outcome,
            policy=policy,
            geometry=geometry,
            error=error,
        )

    def close(self) -> None:
        if self._owns_camera:
            self.camera.close()
