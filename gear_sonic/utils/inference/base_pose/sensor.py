"""SensorGateway adapter for caller-owned base-pose observations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

from gear_sonic.runtime.gateway.rgbd import materialize_rgbd
from gear_sonic.runtime.gateway.sensor_client import (
    SensorGatewayClient,
    SensorGatewayClientError,
)
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest, TimestampBasis


class BasePoseCameraError(RuntimeError):
    """Raised when a head-camera observation is unavailable or malformed."""


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


def _decode_rgbd_snapshot(
    snapshot,
    *,
    camera_stream: str,
    rgb_stream: str,
    depth_stream: str,
    require_depth: bool,
    expected_inference_generation: int | None = None,
) -> AlignedRGBDSnapshot:
    decoded = materialize_rgbd(
        snapshot,
        rgb_stream=rgb_stream,
        depth_stream=depth_stream,
        require_depth=require_depth,
        error_type=BasePoseCameraError,
    )
    rgb = decoded.rgb
    info = dict(decoded.camera_info)
    depth_raw = decoded.depth_raw
    depth_scale_m: float | None = None
    depth_aligned_to: str | None = None
    depth_source = decoded.depth_source
    timestamp_ns = decoded.source_timestamp_ns
    if require_depth:
        depth_scale_m = float(info.get("depth_scale_m", 0.0))
        depth_aligned_to = str(info.get("depth_aligned_to", ""))
        if depth_stream.startswith("derived/depth_anything/"):
            if not (depth_source or "").startswith(
                "depth-anything-v2-metric-"
            ):
                raise BasePoseCameraError(
                    f"Gateway derived depth is not metric Depth Anything: "
                    f"{depth_source!r}"
                )
            if not math.isclose(
                depth_scale_m,
                0.001,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise BasePoseCameraError(
                    "Depth Anything uint16 depth must use millimetre units"
                )
            if bool(info.get("uses_raw_depth", True)):
                raise BasePoseCameraError(
                    "Depth Anything stream must be estimated from RGB only"
                )
            if info.get("inference_owner") != "base_pose":
                raise BasePoseCameraError(
                    "Depth Anything frame is not owned by BasePose"
                )
            if expected_inference_generation is not None:
                try:
                    inference_generation = int(info["inference_generation"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise BasePoseCameraError(
                        "Depth Anything frame has no valid BasePose generation"
                    ) from exc
                if inference_generation != expected_inference_generation:
                    raise BasePoseCameraError(
                        "Depth Anything frame belongs to generation "
                        f"{inference_generation}, expected "
                        f"{expected_inference_generation}"
                    )
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



@dataclass(frozen=True)
class DualBasePoseCapture:
    """Fresh per-camera RGB-D snapshots; one unavailable view does not block the other."""

    snapshots: Mapping[str, AlignedRGBDSnapshot]
    errors: Mapping[str, str]


class _AsyncRGBDStreamBuffer:
    """Continuously materialize one camera's aligned RGB-D into a small buffer."""

    def __init__(
        self,
        stream_name: str,
        depth_stream: str,
        client: Any,
        *,
        max_age_ms: float,
        max_skew_ms: float,
        buffer_size: int,
        poll_hz: float,
        owns_client: bool,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("dual RGB-D buffer size must be positive")
        if not math.isfinite(poll_hz) or poll_hz <= 0.0:
            raise ValueError("dual RGB-D poll frequency must be positive")
        self.stream_name = stream_name
        self.rgb_stream = f"camera/{stream_name}"
        self.depth_stream = depth_stream
        self.client = client
        self.max_age_ms = float(max_age_ms)
        self.max_skew_ms = float(max_skew_ms)
        self.poll_period_s = 1.0 / float(poll_hz)
        self.owns_client = bool(owns_client)
        self._frames: deque[tuple[int, AlignedRGBDSnapshot]] = deque(
            maxlen=int(buffer_size)
        )
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._error = "waiting for first RGB-D frame"
        self._epoch = 0
        self._expected_generation: int | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"base-pose-rgbd-{stream_name}",
            daemon=True,
        )
        self._thread.start()

    def _materialize(
        self,
        expected_generation: int | None,
    ) -> tuple[int, AlignedRGBDSnapshot]:
        materialized = self.client.read_snapshot(
            SnapshotRequest(
                streams=(self.rgb_stream, self.depth_stream),
                max_age_ms=self.max_age_ms,
                max_skew_ms=self.max_skew_ms,
                timestamp_basis=TimestampBasis.SOURCE,
            ),
            retries=0,
        )
        rgb_frame = materialized.snapshot.frames[self.rgb_stream]
        depth_frame = materialized.snapshot.frames[self.depth_stream]
        timestamp_ns = (
            depth_frame.source_timestamp_ns or rgb_frame.source_timestamp_ns
        )
        snapshot = _decode_rgbd_snapshot(
            materialized,
            camera_stream=self.stream_name,
            rgb_stream=self.rgb_stream,
            depth_stream=self.depth_stream,
            require_depth=True,
            expected_inference_generation=expected_generation,
        )
        return timestamp_ns, snapshot

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._condition:
                epoch = self._epoch
                expected_generation = self._expected_generation
            try:
                timestamp_ns, snapshot = self._materialize(expected_generation)
            except Exception as exc:
                with self._condition:
                    if epoch == self._epoch:
                        self._error = str(exc)
                        self._condition.notify_all()
            else:
                with self._condition:
                    if epoch == self._epoch:
                        if not self._frames or self._frames[-1][0] != timestamp_ns:
                            self._frames.append((timestamp_ns, snapshot))
                        self._error = ""
                        self._condition.notify_all()
            self._stop.wait(self.poll_period_s)

    def reset(self, expected_generation: int | None) -> None:
        with self._condition:
            self._epoch += 1
            self._expected_generation = expected_generation
            self._frames.clear()
            self._error = "waiting for first RGB-D frame"
            self._condition.notify_all()

    def latest_after(
        self,
        timestamp_ns: int | None,
    ) -> tuple[int, AlignedRGBDSnapshot] | None:
        with self._condition:
            if not self._frames:
                return None
            latest = self._frames[-1]
            if timestamp_ns is not None and latest[0] <= timestamp_ns:
                return None
            return latest

    def latest(self) -> tuple[int, AlignedRGBDSnapshot] | None:
        """Return the newest buffered frame without advancing a consumer."""

        with self._condition:
            return None if not self._frames else self._frames[-1]

    def wait_after(
        self,
        timestamp_ns: int | None,
        timeout_s: float,
    ) -> tuple[int, AlignedRGBDSnapshot]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while True:
                if self._frames:
                    latest = self._frames[-1]
                    if timestamp_ns is None or latest[0] > timestamp_ns:
                        return latest
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    detail = self._error or "no newer frame"
                    raise BasePoseCameraError(
                        f"timed out waiting for newer {self.stream_name} RGB-D: "
                        f"{detail}"
                    )
                self._condition.wait(remaining)

    @property
    def error(self) -> str:
        with self._condition:
            return self._error or f"waiting for newer {self.stream_name} RGB-D"

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=max(1.0, self.poll_period_s * 2.0))
        if self.owns_client:
            self.client.close()


class SensorGatewayDualBasePoseCamera:
    """Asynchronously buffer head and chest RGB-D with independent cursors."""

    def __init__(
        self,
        endpoint: str,
        *,
        stream_depths: Mapping[str, str],
        timeout_ms: int = 15000,
        request_timeout_ms: int = 100,
        max_age_ms: float = 1000.0,
        max_skew_ms: float = 5.0,
        buffer_size: int = 8,
        poll_hz: float = 60.0,
        client: SensorGatewayClient | None = None,
        client_factory: Callable[[str], Any] | None = None,
        allow_single: bool = False,
    ) -> None:
        values = {str(name): str(depth) for name, depth in stream_depths.items()}
        if len(values) not in ({1, 2} if allow_single else {2}) or any(
            not name or not depth for name, depth in values.items()
        ):
            raise ValueError("dual BasePose requires two camera/depth stream pairs")
        self.stream_depths = values
        self.timeout_ms = int(timeout_ms)
        self.max_age_ms = float(max_age_ms)
        self.max_skew_ms = float(max_skew_ms)
        self._last_timestamp_ns: dict[str, int] = {}
        if client is not None and client_factory is not None:
            raise ValueError("provide either client or client_factory, not both")

        def make_client(stream_name: str) -> tuple[Any, bool]:
            if client_factory is not None:
                return client_factory(stream_name), True
            if client is not None:
                return client, False
            return (
                SensorGatewayClient(
                    endpoint,
                    request_timeout_ms=int(request_timeout_ms),
                ),
                True,
            )

        self._buffers: dict[str, _AsyncRGBDStreamBuffer] = {}
        try:
            for stream_name, depth_stream in self.stream_depths.items():
                stream_client, owns_client = make_client(stream_name)
                self._buffers[stream_name] = _AsyncRGBDStreamBuffer(
                    stream_name,
                    depth_stream,
                    stream_client,
                    max_age_ms=self.max_age_ms,
                    max_skew_ms=self.max_skew_ms,
                    buffer_size=buffer_size,
                    poll_hz=poll_hz,
                    owns_client=owns_client,
                )
        except Exception:
            for stream_buffer in self._buffers.values():
                stream_buffer.close()
            raise

    def begin_generation(self, generation: int) -> None:
        """Discard buffered frames and require derived depth from this generation."""

        self._last_timestamp_ns.clear()
        for stream_buffer in self._buffers.values():
            stream_buffer.reset(int(generation))

    def capture(self) -> DualBasePoseCapture:
        """Wait for initial frames from both streams without cross-stream races."""

        deadline = time.monotonic() + self.timeout_ms / 1000.0
        snapshots: dict[str, AlignedRGBDSnapshot] = {}
        timestamps: dict[str, int] = {}
        while time.monotonic() < deadline and len(snapshots) < len(self._buffers):
            for stream_name, stream_buffer in self._buffers.items():
                if stream_name in snapshots:
                    continue
                buffered = stream_buffer.latest_after(
                    self._last_timestamp_ns.get(stream_name)
                )
                if buffered is not None:
                    timestamp_ns, snapshot = buffered
                    snapshots[stream_name] = snapshot
                    timestamps[stream_name] = timestamp_ns
            time.sleep(0.01)
        errors = {
            stream_name: stream_buffer.error
            for stream_name, stream_buffer in self._buffers.items()
            if stream_name not in snapshots
        }
        if not snapshots:
            detail = "; ".join(
                f"{name}: {error}" for name, error in sorted(errors.items())
            )
            raise BasePoseCameraError(
                "timed out waiting for fresh dual-camera Gateway RGB-D"
                + (f": {detail}" if detail else "")
            )
        self._last_timestamp_ns.update(timestamps)
        return DualBasePoseCapture(snapshots=snapshots, errors=errors)

    def capture_stream(
        self,
        stream_name: str,
        *,
        timeout_ms: int | None = None,
    ) -> AlignedRGBDSnapshot:
        """Wait only for the requested stream and advance only its cursor."""

        if stream_name not in self._buffers:
            raise BasePoseCameraError(f"unknown BasePose camera stream {stream_name!r}")
        timeout = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        timestamp_ns, snapshot = self._buffers[stream_name].wait_after(
            self._last_timestamp_ns.get(stream_name),
            max(0, timeout) / 1000.0,
        )
        self._last_timestamp_ns[stream_name] = timestamp_ns
        return snapshot

    def poll_stream(self, stream_name: str) -> AlignedRGBDSnapshot | None:
        """Return a buffered newer frame immediately, without blocking."""

        if stream_name not in self._buffers:
            raise BasePoseCameraError(f"unknown BasePose camera stream {stream_name!r}")
        buffered = self._buffers[stream_name].latest_after(
            self._last_timestamp_ns.get(stream_name)
        )
        if buffered is None:
            return None
        timestamp_ns, snapshot = buffered
        self._last_timestamp_ns[stream_name] = timestamp_ns
        return snapshot

    def peek_stream(self, stream_name: str) -> AlignedRGBDSnapshot | None:
        """Return the latest buffered frame without advancing its cursor."""

        if stream_name not in self._buffers:
            raise BasePoseCameraError(f"unknown BasePose camera stream {stream_name!r}")
        buffered = self._buffers[stream_name].latest()
        return None if buffered is None else buffered[1]

    def close(self) -> None:
        for stream_buffer in self._buffers.values():
            stream_buffer.close()
