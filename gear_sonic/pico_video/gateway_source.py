"""Validated encoded-camera access through the local SensorGateway API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gear_sonic.runtime.client import (
    SensorGatewayClient,
    SensorGatewayClientError,
    SnapshotUnavailableError,
)
from gear_sonic.runtime.snapshot import SnapshotRequest


class InvalidGatewayFrameError(ValueError):
    """A materialized frame violates the encoded camera stream contract."""


@dataclass(frozen=True)
class GatewayFrame:
    """Immutable JPEG and ordering metadata copied out of shared memory."""

    jpeg: bytes
    generation: int
    sequence: int
    received_timestamp_ns: int
    source_timestamp_ns: int
    source_shape: tuple[int, int, int]


class SensorGatewayVideoSource:
    """Read fresh JPEG head-camera frames exclusively through SensorGateway."""

    STREAM = "camera_encoded/ego_view"

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
        self._request = SnapshotRequest(
            streams=(stream,),
            max_age_ms=max_age_ms,
            max_skew_ms=0.0,
        )
        self._stream = stream
        self._last_key: tuple[int, int] | None = None
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

    def _materialize_frame(self, materialized: Any) -> GatewayFrame | None:
        try:
            reference = materialized.snapshot.frames[self._stream]
            encoded = materialized.arrays[self._stream]
        except (AttributeError, KeyError, TypeError) as exc:
            raise InvalidGatewayFrameError(
                "materialized snapshot omitted the encoded camera stream"
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

        key = (reference.metadata.generation, reference.metadata.sequence)
        if key == self._last_key:
            self.status = "READY"
            return None
        self._last_key = key
        self.status = "READY"
        return GatewayFrame(
            jpeg=jpeg,
            generation=key[0],
            sequence=key[1],
            received_timestamp_ns=reference.metadata.timestamp_ns,
            source_timestamp_ns=reference.source_timestamp_ns,
            source_shape=source_shape,
        )

    def poll(self) -> GatewayFrame | None:
        """Return one new current frame, or ``None`` with a typed display status."""

        try:
            materialized = self._client.read_snapshot(self._request, retries=0)
            return self._materialize_frame(materialized)
        except SnapshotUnavailableError:
            self.status = "SENSOR FRAME STALE"
        except SensorGatewayClientError:
            self.status = "SENSORGATEWAY OFFLINE"
        except InvalidGatewayFrameError:
            self.status = "INVALID CAMERA FRAME"
        return None
