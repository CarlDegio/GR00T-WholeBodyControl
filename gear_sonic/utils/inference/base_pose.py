"""RGB-D camera utilities for Unitree G1 base-pose adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import msgpack
import numpy as np
import zmq

from gear_sonic.camera.calibration import (
    CameraCalibrationError,
    DEFAULT_CAMERA_INTRINSICS_PATH,
    load_camera_intrinsics,
)
from gear_sonic.camera.sensor_server import ImageMessageSchema

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


@dataclass(frozen=True)
class DualRGBDCapture:
    """Independently decoded RGB-D streams from one composed-camera packet."""

    snapshots: Mapping[str, AlignedRGBDSnapshot]
    errors: Mapping[str, str]

    def require(self, stream_name: str) -> AlignedRGBDSnapshot:
        snapshot = self.snapshots.get(stream_name)
        if snapshot is not None:
            return snapshot
        raise BasePoseCameraError(
            self.errors.get(stream_name, f"camera payload requires {stream_name} RGB-D")
        )


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
        calibration_path: str | Path | None = DEFAULT_CAMERA_INTRINSICS_PATH,
    ):
        self.stream_name = stream_name
        self.depth_key = f"{stream_name}_depth"
        self.require_depth = bool(require_depth)
        self.required_depth_source = required_depth_source
        self.timeout_ms = int(timeout_ms)
        self._configured_camera_info: dict[str, Any] | None = None
        if calibration_path is not None:
            try:
                self._configured_camera_info = load_camera_intrinsics(
                    calibration_path
                )[stream_name].asdict()
            except CameraCalibrationError as exc:
                raise BasePoseCameraError(str(exc)) from exc
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
        packet_info = info_map.get(self.stream_name)
        configured_info = getattr(self, "_configured_camera_info", None)
        if isinstance(configured_info, Mapping):
            info = dict(configured_info)
            if isinstance(packet_info, Mapping) and packet_info.get("depth_source"):
                info["depth_source"] = packet_info["depth_source"]
        else:
            info = packet_info
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
            if (
                self.required_depth_source is not None
                and self.required_depth_source.startswith("depth-anything-v2-metric-")
                and not math.isclose(
                    depth_scale_m, 0.001, rel_tol=0.0, abs_tol=1.0e-9
                )
            ):
                raise BasePoseCameraError(
                    "Depth Anything uint16 depth must use millimetre units"
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


class DualAlignedRGBDCamera:
    """Read multiple aligned RGB-D streams through one composed-camera socket."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        stream_names: Sequence[str] = ("ego_view", "chest_view"),
        timeout_ms: int = 15000,
        calibration_path: str | Path | None = DEFAULT_CAMERA_INTRINSICS_PATH,
    ):
        names = tuple(str(name) for name in stream_names)
        if (
            len(names) < 2
            or len(set(names)) != len(names)
            or any(not name for name in names)
        ):
            raise ValueError("dual RGB-D stream names must be distinct and non-empty")
        self.stream_names = names
        self.timeout_ms = int(timeout_ms)
        configured: Mapping[str, Any] = {}
        if calibration_path is not None:
            try:
                configured = load_camera_intrinsics(calibration_path)
            except CameraCalibrationError as exc:
                raise BasePoseCameraError(str(exc)) from exc
        self._decoders: dict[str, AlignedRGBDCamera] = {}
        for stream_name in names:
            decoder = object.__new__(AlignedRGBDCamera)
            decoder.stream_name = stream_name
            decoder.depth_key = f"{stream_name}_depth"
            decoder.require_depth = True
            decoder.required_depth_source = None
            camera_info = configured.get(stream_name)
            decoder._configured_camera_info = (
                None if camera_info is None else camera_info.asdict()
            )
            self._decoders[stream_name] = decoder
        self._last_timestamps: dict[str, float] = {}
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{int(port)}")

    def decode_payload(self, payload: Mapping[str, Any]) -> DualRGBDCapture:
        snapshots: dict[str, AlignedRGBDSnapshot] = {}
        errors: dict[str, str] = {}
        for stream_name in self.stream_names:
            try:
                snapshots[stream_name] = self._decoders[stream_name].decode_payload(
                    payload
                )
            except BasePoseCameraError as exc:
                errors[stream_name] = str(exc)
        return DualRGBDCapture(snapshots=snapshots, errors=errors)

    def capture(self) -> DualRGBDCapture:
        deadline = time.monotonic() + self.timeout_ms / 1000.0
        while True:
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            if not self._socket.poll(remaining_ms):
                raise BasePoseCameraError(
                    "timed out waiting for a fresh dual-camera frame"
                )
            try:
                payload = msgpack.unpackb(self._socket.recv(), raw=False)
                decoded = self.decode_payload(payload)
            except BasePoseCameraError:
                raise
            except Exception as exc:
                raise BasePoseCameraError(
                    "failed to decode dual-camera message"
                ) from exc
            fresh: dict[str, AlignedRGBDSnapshot] = {}
            errors = dict(decoded.errors)
            for stream_name, snapshot in decoded.snapshots.items():
                if self._last_timestamps.get(stream_name) != snapshot.timestamp:
                    fresh[stream_name] = snapshot
                else:
                    errors[stream_name] = (
                        f"timed out waiting for newer {stream_name} frame"
                    )
            if fresh:
                self._last_timestamps.update(
                    {name: snapshot.timestamp for name, snapshot in fresh.items()}
                )
                return DualRGBDCapture(snapshots=fresh, errors=errors)
            if time.monotonic() >= deadline:
                raise BasePoseCameraError(
                    "timed out waiting for newer dual-camera frames"
                )

    def close(self) -> None:
        self._socket.close()
        self._context.term()
