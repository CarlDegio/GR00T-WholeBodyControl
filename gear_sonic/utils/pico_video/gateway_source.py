"""Validated encoded-camera access through the local SensorGateway API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gear_sonic.runtime.gateway.sensor_client import (
    SensorGatewayClient,
    SensorGatewayClientError,
    SensorGatewayTimeoutError,
    SnapshotUnavailableError,
)
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest


class InvalidGatewayFrameError(ValueError):
    """A materialized frame violates the encoded camera stream contract."""


@dataclass(frozen=True)
class GatewayFrame:
    """Synchronized JPEG views and ego-view metadata copied out of shared memory."""

    jpeg: bytes
    generation: int
    sequence: int
    received_timestamp_ns: int
    source_timestamp_ns: int
    source_shape: tuple[int, int, int]
    left_wrist_jpeg: bytes = b""
    right_wrist_jpeg: bytes = b""
    left_wrist_source_shape: tuple[int, int, int] = (0, 0, 0)
    right_wrist_source_shape: tuple[int, int, int] = (0, 0, 0)


class SensorGatewayVideoSource:
    """Read synchronized ego and wrist JPEGs exclusively through SensorGateway."""

    STREAM = "camera_encoded/ego_view"
    LEFT_WRIST_STREAM = "camera_encoded/left_wrist"
    RIGHT_WRIST_STREAM = "camera_encoded/right_wrist"

    def __init__(
        self,
        client: SensorGatewayClient,
        *,
        max_age_ms: float,
        stream: str = STREAM,
    ) -> None:
        if stream != self.STREAM:
            raise ValueError(f"unsupported PICO video stream: {stream}")
        self._client = client
        self._streams = (
            stream,
            self.LEFT_WRIST_STREAM,
            self.RIGHT_WRIST_STREAM,
        )
        self._request = SnapshotRequest(
            streams=self._streams,
            max_age_ms=max_age_ms,
            max_skew_ms=0.0,
        )
        self._stream = stream
        self._last_key: tuple[tuple[int, int], ...] | None = None
        self.status = "WAITING FOR SENSORGATEWAY"

    @staticmethod
    def _image_shape(attributes: Any) -> tuple[int, int, int]:
        raw_shape = attributes.get("image_shape", ())
        if not isinstance(raw_shape, list | tuple):
            raise InvalidGatewayFrameError("encoded frame image_shape must be a list")
        try:
            shape = tuple(int(value) for value in raw_shape)
        except (TypeError, ValueError) as exc:
            raise InvalidGatewayFrameError(
                "encoded frame image_shape must contain integers"
            ) from exc
        if len(shape) != 3 or any(value <= 0 for value in shape) or shape[2] != 3:
            raise InvalidGatewayFrameError(
                "encoded frame image_shape must be positive HxWx3"
            )
        return shape

    def _encoded_camera(
        self,
        materialized: Any,
        stream: str,
    ) -> tuple[bytes, tuple[int, int, int], Any]:
        try:
            reference = materialized.snapshot.frames[stream]
            encoded = materialized.arrays[stream]
        except (AttributeError, KeyError, TypeError) as exc:
            raise InvalidGatewayFrameError(
                f"materialized snapshot omitted encoded camera stream {stream!r}"
            ) from exc

        if not isinstance(encoded, np.ndarray):
            raise InvalidGatewayFrameError("encoded frame must be a NumPy array")
        if encoded.dtype != np.dtype(np.uint8) or encoded.ndim != 1:
            raise InvalidGatewayFrameError("encoded frame must be 1-D uint8")
        try:
            reference_dtype = np.dtype(reference.dtype)
        except TypeError as exc:
            raise InvalidGatewayFrameError(
                "encoded frame shared-memory dtype is invalid"
            ) from exc
        if reference_dtype != np.dtype(np.uint8) or tuple(reference.shape) != encoded.shape:
            raise InvalidGatewayFrameError(
                "encoded frame shared-memory metadata does not match its array"
            )
        if reference.attributes.get("encoding") != "jpeg_bytes":
            raise InvalidGatewayFrameError("encoded frame must use jpeg_bytes")

        jpeg = encoded.tobytes()
        if (
            len(jpeg) < 4
            or not jpeg.startswith(b"\xff\xd8")
            or not jpeg.endswith(b"\xff\xd9")
        ):
            raise InvalidGatewayFrameError("encoded frame is not a complete JPEG")
        source_shape = self._image_shape(reference.attributes)
        return jpeg, source_shape, reference

    def _materialize_frame(self, materialized: Any) -> GatewayFrame | None:
        cameras = {
            stream: self._encoded_camera(materialized, stream)
            for stream in self._streams
        }

        key = tuple(
            (reference.metadata.generation, reference.metadata.sequence)
            for _, _, reference in cameras.values()
        )
        if key == self._last_key:
            self.status = "READY"
            return None
        self._last_key = key
        self.status = "READY"
        jpeg, source_shape, reference = cameras[self._stream]
        left_jpeg, left_shape, _ = cameras[self.LEFT_WRIST_STREAM]
        right_jpeg, right_shape, _ = cameras[self.RIGHT_WRIST_STREAM]
        return GatewayFrame(
            jpeg=jpeg,
            generation=reference.metadata.generation,
            sequence=reference.metadata.sequence,
            received_timestamp_ns=reference.metadata.timestamp_ns,
            source_timestamp_ns=reference.source_timestamp_ns,
            source_shape=source_shape,
            left_wrist_jpeg=left_jpeg,
            right_wrist_jpeg=right_jpeg,
            left_wrist_source_shape=left_shape,
            right_wrist_source_shape=right_shape,
        )

    def poll(self) -> GatewayFrame | None:
        """Return one new current frame, or ``None`` with a typed display status."""

        try:
            materialized = self._client.read_snapshot(self._request, retries=0)
            return self._materialize_frame(materialized)
        except SnapshotUnavailableError as exc:
            cause: BaseException | None = exc
            seen: set[int] = set()
            while cause is not None and not isinstance(
                cause, SensorGatewayTimeoutError
            ) and id(cause) not in seen:
                seen.add(id(cause))
                cause = cause.__cause__ or cause.__context__
            self.status = (
                "SENSORGATEWAY OFFLINE"
                if isinstance(cause, SensorGatewayTimeoutError)
                else "SENSOR FRAME STALE"
            )
        except SensorGatewayClientError:
            self.status = "SENSORGATEWAY OFFLINE"
        except InvalidGatewayFrameError:
            self.status = "INVALID CAMERA FRAME"
        return None
