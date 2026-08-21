from __future__ import annotations

import pytest

from gear_sonic.runtime.protocol import MessageMetadata, SharedMemoryFrame
from gear_sonic.runtime.gateway.snapshot import (
    SensorSnapshot,
    SensorSnapshotStore,
    SnapshotRequest,
    TimestampBasis,
)


def _frame(
    stream: str,
    *,
    sequence: int,
    received_ms: float,
    source_ms: float,
    source_clock: str = "robot_unix",
) -> SharedMemoryFrame:
    return SharedMemoryFrame(
        metadata=MessageMetadata(
            source="sensor_gateway",
            sequence=sequence,
            timestamp_ns=int(received_ms * 1_000_000),
            ttl_ms=1000,
        ),
        stream=stream,
        shared_memory=f"fake-{stream}",
        shape=(1,),
        dtype="float32",
        offset_bytes=8,
        size_bytes=4,
        source_timestamp_ns=int(source_ms * 1_000_000),
        source_clock=source_clock,
    )


def test_snapshot_selects_nearest_frames_on_gateway_receive_clock() -> None:
    store = SensorSnapshotStore(history_size=4)
    store.add(_frame("camera", sequence=0, received_ms=100.0, source_ms=1000.0))
    store.add(_frame("camera", sequence=1, received_ms=140.0, source_ms=1040.0))
    store.add(_frame("state", sequence=0, received_ms=110.0, source_ms=1005.0))
    store.add(_frame("state", sequence=1, received_ms=150.0, source_ms=1045.0))

    snapshot = store.select(
        SnapshotRequest(
            streams=("camera", "state"),
            max_age_ms=100.0,
            max_skew_ms=15.0,
        ),
        now_ns=160_000_000,
    )

    assert snapshot.complete
    assert snapshot.anchor_timestamp_ns == 140_000_000
    assert snapshot.frames["camera"].metadata.sequence == 1
    assert snapshot.frames["state"].metadata.sequence == 1
    assert snapshot.skew_ms == pytest.approx(10.0)
    decoded = SensorSnapshot.from_dict(snapshot.to_dict())
    assert decoded.complete
    assert decoded.frames["camera"] == snapshot.frames["camera"]


def test_snapshot_can_align_compatible_source_clocks() -> None:
    store = SensorSnapshotStore(history_size=4)
    store.add(_frame("camera", sequence=0, received_ms=100.0, source_ms=1000.0))
    store.add(_frame("state", sequence=0, received_ms=130.0, source_ms=1005.0))

    snapshot = store.select(
        SnapshotRequest(
            streams=("camera", "state"),
            max_age_ms=100.0,
            max_skew_ms=10.0,
            timestamp_basis=TimestampBasis.SOURCE,
        ),
        now_ns=150_000_000,
    )

    assert snapshot.complete
    assert snapshot.skew_ms == pytest.approx(5.0)


def test_snapshot_rejects_incompatible_source_clocks_and_stale_streams() -> None:
    store = SensorSnapshotStore(history_size=4)
    store.add(
        _frame(
            "camera",
            sequence=0,
            received_ms=100.0,
            source_ms=1000.0,
            source_clock="camera_unix",
        )
    )
    store.add(
        _frame(
            "state",
            sequence=0,
            received_ms=110.0,
            source_ms=1005.0,
            source_clock="robot_ros",
        )
    )
    source_snapshot = store.select(
        SnapshotRequest(
            streams=("camera", "state"),
            max_age_ms=100.0,
            max_skew_ms=20.0,
            timestamp_basis=TimestampBasis.SOURCE,
        ),
        now_ns=120_000_000,
    )
    stale_snapshot = store.select(
        SnapshotRequest(
            streams=("camera", "state"),
            max_age_ms=50.0,
            max_skew_ms=20.0,
        ),
        now_ns=300_000_000,
    )

    assert not source_snapshot.complete
    assert "incompatible" in source_snapshot.reason
    assert not stale_snapshot.complete
    assert "stale streams" in stale_snapshot.reason


def test_snapshot_request_wire_round_trip() -> None:
    request = SnapshotRequest(
        streams=("camera", "state"),
        max_age_ms=250.0,
        max_skew_ms=50.0,
        anchor_timestamp_ns=123,
        timestamp_basis=TimestampBasis.RECEIVE,
    )

    assert SnapshotRequest.from_dict(request.to_dict()) == request


def test_explicit_anchor_is_included_in_snapshot_skew_limit() -> None:
    store = SensorSnapshotStore(history_size=4)
    store.add(
        _frame(
            "camera",
            sequence=0,
            received_ms=100.0,
            source_ms=2000.0,
            source_clock="camera_unix",
        )
    )

    snapshot = store.select(
        SnapshotRequest(
            streams=("camera",),
            max_age_ms=100.0,
            max_skew_ms=10.0,
            anchor_timestamp_ns=2_050_000_000,
            timestamp_basis=TimestampBasis.SOURCE,
        ),
        now_ns=110_000_000,
    )

    assert not snapshot.complete
    assert snapshot.skew_ms == 50.0
    assert "exceeds limit" in snapshot.reason
