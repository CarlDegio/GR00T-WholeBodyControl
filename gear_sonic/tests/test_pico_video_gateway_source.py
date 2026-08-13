from __future__ import annotations

from collections import deque
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from gear_sonic.pico_video.gateway_source import GatewayFrame, SensorGatewayVideoSource
from gear_sonic.runtime.client import (
    MaterializedSnapshot,
    SensorGatewayClientError,
    SensorGatewayTimeoutError,
    SnapshotUnavailableError,
)
from gear_sonic.runtime.contracts import MessageMetadata, SharedMemoryFrame
from gear_sonic.runtime.snapshot import SensorSnapshot, SnapshotRequest, TimestampBasis


class SequenceClient:
    def __init__(self, responses: list[MaterializedSnapshot | Exception]) -> None:
        self._responses = deque(responses)
        self.requests: list[tuple[SnapshotRequest, int]] = []

    def read_snapshot(
        self,
        request: SnapshotRequest,
        *,
        retries: int = 2,
        retry_backoff_s: float = 0.005,
    ) -> MaterializedSnapshot:
        del retry_backoff_s
        self.requests.append((request, retries))
        response = self._responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


def _materialized(
    jpeg: bytes,
    *,
    sequence: int,
    generation: int = 3,
    dtype: Any = np.uint8,
    array_shape: tuple[int, ...] | None = None,
    encoding: str = "jpeg_bytes",
    image_shape: list[int] | None = None,
) -> MaterializedSnapshot:
    stream = "camera_encoded/ego_view"
    metadata = MessageMetadata(
        source="camera_zmq",
        sequence=sequence,
        timestamp_ns=10_000_000_000,
        ttl_ms=1000,
        generation=generation,
    )
    array = np.frombuffer(jpeg, dtype=np.uint8).astype(dtype)
    if array_shape is not None:
        array = array.reshape(array_shape)
    frame = SharedMemoryFrame(
        metadata=metadata,
        stream=stream,
        shared_memory="test-ring",
        shape=array.shape,
        dtype=array.dtype.str,
        offset_bytes=0,
        size_bytes=array.nbytes,
        source_timestamp_ns=9_900_000_000,
        source_clock="camera_unix",
        attributes={
            "encoding": encoding,
            "decoded_color_order": "RGB",
            "image_shape": image_shape or [480, 640, 3],
            "camera_info": {},
        },
    )
    snapshot = SensorSnapshot(
        complete=True,
        reason="",
        anchor_timestamp_ns=metadata.timestamp_ns,
        timestamp_basis=TimestampBasis.RECEIVE,
        frames=MappingProxyType({stream: frame}),
        skew_ms=0.0,
        ages_ms=MappingProxyType({stream: 1.0}),
    )
    return MaterializedSnapshot(
        snapshot=snapshot,
        arrays=MappingProxyType({stream: array}),
        attempts=1,
    )


def test_source_returns_valid_jpeg_once_and_deduplicates_generation_sequence() -> None:
    materialized = _materialized(b"\xff\xd8data\xff\xd9", sequence=8)
    source = SensorGatewayVideoSource(
        SequenceClient([materialized, materialized]), max_age_ms=250.0
    )

    assert source.poll() == GatewayFrame(
        jpeg=b"\xff\xd8data\xff\xd9",
        generation=3,
        sequence=8,
        received_timestamp_ns=10_000_000_000,
        source_timestamp_ns=9_900_000_000,
        source_shape=(480, 640, 3),
    )
    assert source.poll() is None
    assert source.status == "READY"


def test_source_requests_only_the_encoded_ego_stream_without_internal_retries() -> None:
    client = SequenceClient([_materialized(b"\xff\xd8x\xff\xd9", sequence=1)])
    source = SensorGatewayVideoSource(client, max_age_ms=250.0)

    source.poll()

    request, retries = client.requests[0]
    assert request.streams == ("camera_encoded/ego_view",)
    assert request.max_age_ms == 250.0
    assert request.max_skew_ms == 0.0
    assert retries == 0


def test_source_accepts_same_sequence_after_gateway_generation_changes() -> None:
    first = _materialized(b"\xff\xd8first\xff\xd9", sequence=4, generation=2)
    restarted = _materialized(b"\xff\xd8second\xff\xd9", sequence=4, generation=3)
    source = SensorGatewayVideoSource(
        SequenceClient([first, restarted]), max_age_ms=250.0
    )

    assert source.poll() is not None
    assert source.poll() == GatewayFrame(
        jpeg=b"\xff\xd8second\xff\xd9",
        generation=3,
        sequence=4,
        received_timestamp_ns=10_000_000_000,
        source_timestamp_ns=9_900_000_000,
        source_shape=(480, 640, 3),
    )


@pytest.mark.parametrize(
    ("snapshot", "expected_status"),
    [
        (
            _materialized(
                b"\xff\xd8data\xff\xd9", sequence=1, encoding="base64_jpeg"
            ),
            "INVALID CAMERA FRAME",
        ),
        (
            _materialized(b"\xff\xd8data\xff\xd9", sequence=1, dtype=np.uint16),
            "INVALID CAMERA FRAME",
        ),
        (
            _materialized(
                b"\xff\xd8data\xff\xd9", sequence=1, array_shape=(2, 4)
            ),
            "INVALID CAMERA FRAME",
        ),
        (
            _materialized(
                b"\xff\xd8data\xff\xd9", sequence=1, image_shape=[480, 640]
            ),
            "INVALID CAMERA FRAME",
        ),
        (
            _materialized(b"not-a-jpeg", sequence=1),
            "INVALID CAMERA FRAME",
        ),
    ],
)
def test_source_rejects_frames_outside_the_encoded_camera_contract(
    snapshot: MaterializedSnapshot,
    expected_status: str,
) -> None:
    source = SensorGatewayVideoSource(SequenceClient([snapshot]), max_age_ms=250.0)

    assert source.poll() is None
    assert source.status == expected_status


def test_source_reports_stale_and_gateway_failures_as_distinct_statuses() -> None:
    source = SensorGatewayVideoSource(
        SequenceClient(
            [
                SnapshotUnavailableError("frame too old"),
                SensorGatewayClientError("offline"),
            ]
        ),
        max_age_ms=250.0,
    )

    assert source.poll() is None
    assert source.status == "SENSOR FRAME STALE"
    assert source.poll() is None
    assert source.status == "SENSORGATEWAY OFFLINE"


def test_source_preserves_gateway_timeout_status_through_snapshot_wrapper() -> None:
    wrapped = SnapshotUnavailableError("snapshot unavailable after one attempt")
    wrapped.__cause__ = SensorGatewayTimeoutError("RPC timed out")
    source = SensorGatewayVideoSource(
        SequenceClient([wrapped]),
        max_age_ms=250.0,
    )

    assert source.poll() is None
    assert source.status == "SENSORGATEWAY OFFLINE"


def test_source_recovers_after_an_invalid_frame() -> None:
    source = SensorGatewayVideoSource(
        SequenceClient(
            [
                _materialized(b"bad", sequence=1),
                _materialized(b"\xff\xd8good\xff\xd9", sequence=2),
            ]
        ),
        max_age_ms=250.0,
    )

    assert source.poll() is None
    assert source.status == "INVALID CAMERA FRAME"
    assert source.poll() is not None
    assert source.status == "READY"
