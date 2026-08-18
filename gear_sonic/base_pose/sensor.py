"""SensorGateway adapter for caller-owned base-pose observations."""

from __future__ import annotations

import time

import numpy as np

from gear_sonic.base_pose.policy import BasePoseCameraError, BasePoseObservation
from gear_sonic.runtime.client import SensorGatewayClient, SensorGatewayClientError
from gear_sonic.runtime.snapshot import SnapshotRequest, TimestampBasis


class SensorGatewayBasePoseCamera:
    """Read one fresh RGB or aligned LingBot RGB-D snapshot from SensorGateway."""

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

    def _decode(self, snapshot) -> BasePoseObservation:
        rgb_frame = snapshot.snapshot.frames[self.rgb_stream]
        rgb = np.asarray(snapshot.arrays[self.rgb_stream])
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
        if self.require_depth:
            depth_frame = snapshot.snapshot.frames[self.depth_stream]
            depth_raw = np.asarray(snapshot.arrays[self.depth_stream])
            if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
                raise BasePoseCameraError(
                    f"Gateway LingBot depth must be HxW uint16, got "
                    f"{depth_raw.shape} {depth_raw.dtype}"
                )
            if depth_raw.shape != rgb.shape[:2]:
                raise BasePoseCameraError("Gateway RGB and LingBot depth shapes do not match")
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
        timestamp = timestamp_ns * 1.0e-9 if timestamp_ns > 0 else time.time()
        return BasePoseObservation(
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

    def capture(self) -> BasePoseObservation:
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
