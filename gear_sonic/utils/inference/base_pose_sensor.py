"""SensorGateway adapter for caller-owned base-pose observations."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping

import numpy as np

from gear_sonic.runtime.client import SensorGatewayClient, SensorGatewayClientError
from gear_sonic.runtime.snapshot import SnapshotRequest, TimestampBasis
from gear_sonic.utils.inference.base_pose import (
    AlignedRGBDSnapshot,
    BasePoseCameraError,
)


def _decode_rgbd_snapshot(
    snapshot,
    *,
    camera_stream: str,
    rgb_stream: str,
    depth_stream: str,
    require_depth: bool,
) -> AlignedRGBDSnapshot:
    rgb_frame = snapshot.snapshot.frames[rgb_stream]
    rgb = np.asarray(snapshot.arrays[rgb_stream])
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise BasePoseCameraError(
            f"Gateway RGB must be HxWx3 uint8, got {rgb.shape} {rgb.dtype}"
        )
    info = dict(rgb_frame.attributes.get("camera_info", {}))
    depth_raw: np.ndarray | None = None
    depth_scale_m: float | None = None
    depth_aligned_to: str | None = None
    depth_source: str | None = None
    timestamp_ns = rgb_frame.source_timestamp_ns
    if require_depth:
        depth_frame = snapshot.snapshot.frames[depth_stream]
        depth_raw = np.asarray(snapshot.arrays[depth_stream])
        if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
            raise BasePoseCameraError(
                f"Gateway raw depth must be HxW uint16, got "
                f"{depth_raw.shape} {depth_raw.dtype}"
            )
        if depth_raw.shape != rgb.shape[:2]:
            raise BasePoseCameraError("Gateway RGB and raw depth shapes do not match")
        depth_info = dict(depth_frame.attributes.get("camera_info", {}))
        if depth_info:
            info = depth_info
        depth_scale_m = float(info.get("depth_scale_m", 0.0))
        depth_aligned_to = str(info.get("depth_aligned_to", ""))
        depth_source = str(
            depth_frame.attributes.get("depth_source")
            or info.get("depth_source", "")
        ) or None
        timestamp_ns = depth_frame.source_timestamp_ns or timestamp_ns
    try:
        fx, fy, cx, cy = (float(info[name]) for name in ("fx", "fy", "cx", "cy"))
        width, height = int(info["width"]), int(info["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BasePoseCameraError("Gateway camera calibration is incomplete") from exc
    if (width, height) != (rgb.shape[1], rgb.shape[0]):
        raise BasePoseCameraError("Gateway calibration dimensions do not match RGB")
    if require_depth and depth_aligned_to != camera_stream:
        raise BasePoseCameraError(
            f"Gateway depth is not aligned to {camera_stream!r}"
        )
    timestamp = timestamp_ns * 1.0e-9 if timestamp_ns > 0 else time.time()
    return AlignedRGBDSnapshot(
        rgb=rgb.copy(),
        depth_raw=None if depth_raw is None else depth_raw.copy(),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        depth_scale_m=depth_scale_m,
        depth_aligned_to=depth_aligned_to,
        depth_source=depth_source,
        timestamp=timestamp,
    )


class SensorGatewayBasePoseCamera:
    """Read one fresh RGB or aligned raw RGB-D snapshot from SensorGateway."""

    def __init__(
        self,
        endpoint: str,
        *,
        camera_stream: str = "ego_view",
        depth_stream: str = "derived/lingbot_depth",
        require_depth: bool = False,
        timeout_ms: int = 15000,
        request_timeout_ms: int = 100,
        max_age_ms: float = 1000.0,
        max_skew_ms: float = 5.0,
        client: SensorGatewayClient | None = None,
    ) -> None:
        self.camera_stream = camera_stream
        self.rgb_stream = f"camera/{camera_stream}"
        self.depth_stream = depth_stream
        self.require_depth = bool(require_depth)
        self.timeout_ms = int(timeout_ms)
        self.max_age_ms = float(max_age_ms)
        self.max_skew_ms = float(max_skew_ms)
        self.client = client or SensorGatewayClient(
            endpoint, request_timeout_ms=int(request_timeout_ms)
        )
        self._owns_client = client is None
        self._last_timestamp_ns: int | None = None

    def _decode(self, snapshot) -> AlignedRGBDSnapshot:
        return _decode_rgbd_snapshot(
            snapshot,
            camera_stream=self.camera_stream,
            rgb_stream=self.rgb_stream,
            depth_stream=self.depth_stream,
            require_depth=self.require_depth,
        )

    def capture(self) -> AlignedRGBDSnapshot:
        streams = (
            (self.rgb_stream, self.depth_stream)
            if self.require_depth
            else (self.rgb_stream,)
        )
        deadline = time.monotonic() + self.timeout_ms / 1000.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                snapshot = self.client.read_snapshot(
                    SnapshotRequest(
                        streams=streams,
                        max_age_ms=self.max_age_ms,
                        max_skew_ms=self.max_skew_ms,
                        timestamp_basis=TimestampBasis.SOURCE,
                    ),
                    retries=1,
                )
                frame = snapshot.snapshot.frames[streams[-1]]
                timestamp_ns = frame.source_timestamp_ns
                if self._last_timestamp_ns is None or timestamp_ns != self._last_timestamp_ns:
                    decoded = self._decode(snapshot)
                    self._last_timestamp_ns = timestamp_ns
                    return decoded
            except (SensorGatewayClientError, BasePoseCameraError) as exc:
                last_error = exc
            time.sleep(0.01)
        raise BasePoseCameraError(
            f"timed out waiting for fresh Gateway base-pose observation: "
            f"{last_error or 'no new frame'}"
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


@dataclass(frozen=True)
class DualBasePoseCapture:
    """Fresh per-camera RGB-D snapshots; one unavailable view does not block the other."""

    snapshots: Mapping[str, AlignedRGBDSnapshot]
    errors: Mapping[str, str]

    def require(self, stream_name: str) -> AlignedRGBDSnapshot:
        snapshot = self.snapshots.get(stream_name)
        if snapshot is not None:
            return snapshot
        raise BasePoseCameraError(
            self.errors.get(
                stream_name,
                f"SensorGateway capture has no fresh {stream_name} RGB-D",
            )
        )


class SensorGatewayDualBasePoseCamera:
    """Read head and chest RGB-D independently through SensorGateway."""

    def __init__(
        self,
        endpoint: str,
        *,
        stream_depths: Mapping[str, str],
        timeout_ms: int = 15000,
        request_timeout_ms: int = 100,
        max_age_ms: float = 1000.0,
        max_skew_ms: float = 5.0,
        client: SensorGatewayClient | None = None,
    ) -> None:
        values = {str(name): str(depth) for name, depth in stream_depths.items()}
        if len(values) != 2 or any(
            not name or not depth for name, depth in values.items()
        ):
            raise ValueError("dual BasePose requires two camera/depth stream pairs")
        self.stream_depths = values
        self.timeout_ms = int(timeout_ms)
        self.max_age_ms = float(max_age_ms)
        self.max_skew_ms = float(max_skew_ms)
        self.client = client or SensorGatewayClient(
            endpoint,
            request_timeout_ms=int(request_timeout_ms),
        )
        self._owns_client = client is None
        self._last_timestamp_ns: dict[str, int] = {}

    def _capture_stream(self, stream_name: str) -> tuple[int, AlignedRGBDSnapshot]:
        rgb_stream = f"camera/{stream_name}"
        depth_stream = self.stream_depths[stream_name]
        materialized = self.client.read_snapshot(
            SnapshotRequest(
                streams=(rgb_stream, depth_stream),
                max_age_ms=self.max_age_ms,
                max_skew_ms=self.max_skew_ms,
                timestamp_basis=TimestampBasis.SOURCE,
            ),
            retries=0,
        )
        rgb_frame = materialized.snapshot.frames[rgb_stream]
        depth_frame = materialized.snapshot.frames[depth_stream]
        timestamp_ns = (
            depth_frame.source_timestamp_ns or rgb_frame.source_timestamp_ns
        )
        return timestamp_ns, _decode_rgbd_snapshot(
            materialized,
            camera_stream=stream_name,
            rgb_stream=rgb_stream,
            depth_stream=depth_stream,
            require_depth=True,
        )

    def capture(self) -> DualBasePoseCapture:
        deadline = time.monotonic() + self.timeout_ms / 1000.0
        latest_errors: dict[str, str] = {}
        while time.monotonic() < deadline:
            snapshots: dict[str, AlignedRGBDSnapshot] = {}
            errors: dict[str, str] = {}
            timestamps: dict[str, int] = {}
            for stream_name in self.stream_depths:
                try:
                    timestamp_ns, snapshot = self._capture_stream(stream_name)
                    if self._last_timestamp_ns.get(stream_name) == timestamp_ns:
                        raise BasePoseCameraError(
                            f"waiting for newer {stream_name} RGB-D"
                        )
                    snapshots[stream_name] = snapshot
                    timestamps[stream_name] = timestamp_ns
                except (SensorGatewayClientError, BasePoseCameraError) as exc:
                    errors[stream_name] = str(exc)
            if snapshots:
                self._last_timestamp_ns.update(timestamps)
                return DualBasePoseCapture(snapshots=snapshots, errors=errors)
            latest_errors = errors
            time.sleep(0.01)
        detail = "; ".join(
            f"{name}: {error}" for name, error in sorted(latest_errors.items())
        )
        raise BasePoseCameraError(
            "timed out waiting for fresh dual-camera Gateway RGB-D"
            + (f": {detail}" if detail else "")
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
