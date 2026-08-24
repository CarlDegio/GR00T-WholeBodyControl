"""Sensor Gateway camera adapter used by the LaViRA agent."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from gear_sonic.runtime.gateway.rgbd import materialize_rgbd
from gear_sonic.runtime.gateway.sensor_client import (
    SensorGatewayClient,
    SensorGatewayClientError,
)
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest, TimestampBasis
from gear_sonic.runtime.protocol import decode_cpp_state_array
from gear_sonic.utils.math3d.quaternions import yaw_from_quaternion_wxyz


@dataclass(frozen=True)
class RGBDSnapshot:
    rgb_bgr: np.ndarray
    depth_mm: np.ndarray
    fx: float
    cx: float


class ObjectNavCameraError(RuntimeError):
    """Raised for malformed, missing, or stale Gateway RGB-D frames."""


class SensorGatewayRGBDCamera:
    """Read chest RGB and Depth Anything metric depth from shared memory."""

    RGB_STREAM = "camera/chest_view"
    DEPTH_STREAM = "derived/depth_anything/chest_view"
    ODOMETRY_STREAM = "ros/odometry"
    ROBOT_STATE_STREAM = "cpp/state_msgpack"
    SONIC_YAW_MAX_AGE_MS = 300.0

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
        self._last_rgbd_timestamp_by_stream: dict[str, int] = {}
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
        decoded = materialize_rgbd(
            snapshot,
            rgb_stream=SensorGatewayRGBDCamera.RGB_STREAM,
            depth_stream=SensorGatewayRGBDCamera.DEPTH_STREAM,
            error_type=ObjectNavCameraError,
            rgb_label="Gateway chest RGB",
            depth_label="Gateway Depth Anything depth",
            mismatch_message=(
                "Gateway RGB and Depth Anything depth shapes do not match"
            ),
        )
        rgb = decoded.rgb
        depth_raw = decoded.depth_raw
        assert depth_raw is not None
        info = dict(decoded.camera_info)
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
        depth_source = decoded.depth_source or ""
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

    @staticmethod
    def _decode_camera_rgbd(
        snapshot,
        *,
        camera_stream: str,
        rgb_stream: str,
        depth_stream: str,
    ) -> RGBDSnapshot:
        """Decode one camera's hardware-aligned RGB-D observation."""

        rgb_frame = snapshot.snapshot.frames[rgb_stream]
        depth_frame = snapshot.snapshot.frames[depth_stream]
        rgb = np.asarray(snapshot.arrays[rgb_stream])
        depth_raw = np.asarray(snapshot.arrays[depth_stream])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ObjectNavCameraError(
                f"Gateway {camera_stream} RGB must be HxWx3 uint8, got "
                f"{rgb.shape} {rgb.dtype}"
            )
        if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
            raise ObjectNavCameraError(
                f"Gateway {camera_stream} depth must be HxW uint16, got "
                f"{depth_raw.shape} {depth_raw.dtype}"
            )
        if rgb.shape[:2] != depth_raw.shape:
            raise ObjectNavCameraError(
                f"Gateway {camera_stream} RGB and depth shapes do not match"
            )
        info = dict(depth_frame.attributes.get("camera_info", {}))
        if not info:
            info = dict(rgb_frame.attributes.get("camera_info", {}))
        try:
            fx = float(info["fx"])
            cx = float(info["cx"])
            depth_scale_m = float(info["depth_scale_m"])
            width = int(info["width"])
            height = int(info["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ObjectNavCameraError(
                f"Gateway {camera_stream} RGB-D calibration is incomplete"
            ) from exc
        if not math.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
            raise ObjectNavCameraError(
                f"Gateway {camera_stream} depth scale is invalid"
            )
        if str(info.get("depth_aligned_to", "")) != camera_stream:
            raise ObjectNavCameraError(
                f"Gateway depth is not aligned to {camera_stream!r}"
            )
        if (width, height) != (rgb.shape[1], rgb.shape[0]):
            raise ObjectNavCameraError(
                f"Gateway {camera_stream} calibration dimensions do not match RGB-D"
            )
        return RGBDSnapshot(
            rgb_bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            depth_mm=(
                depth_raw.astype(np.float32)
                * np.float32(depth_scale_m * 1000.0)
            ),
            fx=fx,
            cx=cx,
        )

    def capture_camera_aligned_rgbd(
        self,
        *,
        camera_stream: str,
        depth_stream: str | None = None,
        timeout_ms: int | None = None,
    ) -> RGBDSnapshot:
        """Capture fresh hardware RGB-D for a named camera stream."""

        stream_name = str(camera_stream).strip()
        if not stream_name:
            raise ObjectNavCameraError("Gateway RGB-D camera stream is empty")
        rgb_stream = (
            stream_name if stream_name.startswith("camera/")
            else f"camera/{stream_name}"
        )
        camera_name = rgb_stream.removeprefix("camera/")
        resolved_depth_stream = str(
            depth_stream or f"camera/{camera_name}_depth"
        ).strip()
        if not resolved_depth_stream:
            raise ObjectNavCameraError("Gateway RGB-D depth stream is empty")
        if not resolved_depth_stream.startswith(("camera/", "derived/")):
            resolved_depth_stream = f"camera/{resolved_depth_stream}"
        effective_timeout = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        deadline = time.monotonic() + effective_timeout / 1000.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                snapshot = self.client.read_snapshot(
                    SnapshotRequest(
                        streams=(rgb_stream, resolved_depth_stream),
                        max_age_ms=self.max_age_ms,
                        max_skew_ms=self.max_skew_ms,
                        timestamp_basis=TimestampBasis.SOURCE,
                    ),
                    retries=1,
                )
                depth_frame = snapshot.snapshot.frames[resolved_depth_stream]
                timestamp_ns = depth_frame.source_timestamp_ns
                if (
                    self._last_rgbd_timestamp_by_stream.get(resolved_depth_stream)
                    != timestamp_ns
                ):
                    decoded = self._decode_camera_rgbd(
                        snapshot,
                        camera_stream=camera_name,
                        rgb_stream=rgb_stream,
                        depth_stream=resolved_depth_stream,
                    )
                    self._last_rgbd_timestamp_by_stream[resolved_depth_stream] = (
                        timestamp_ns
                    )
                    return decoded
            except (SensorGatewayClientError, ObjectNavCameraError) as exc:
                last_error = exc
            time.sleep(0.01)
        raise ObjectNavCameraError(
            f"timed out waiting for fresh Gateway {camera_name} RGB-D: "
            f"{last_error or 'no new frame'}"
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

    def current_sonic_yaw(self) -> float:
        """Return fresh measured SONIC base yaw from the C++ robot state."""

        try:
            snapshot = self.client.read_snapshot(
                SnapshotRequest(
                    streams=(self.ROBOT_STATE_STREAM,),
                    max_age_ms=min(self.max_age_ms, self.SONIC_YAW_MAX_AGE_MS),
                    max_skew_ms=0.0,
                    timestamp_basis=TimestampBasis.RECEIVE,
                ),
                retries=1,
            )
            state = decode_cpp_state_array(
                snapshot.arrays[self.ROBOT_STATE_STREAM]
            )
            yaw = yaw_from_quaternion_wxyz(state["base_quat"])
        except (KeyError, TypeError, ValueError, SensorGatewayClientError) as exc:
            raise ObjectNavCameraError(
                "fresh SONIC measured yaw is unavailable"
            ) from exc
        return float(yaw)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


