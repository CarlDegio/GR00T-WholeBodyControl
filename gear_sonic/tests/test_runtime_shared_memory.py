from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from gear_sonic.runtime.shared_memory import (
    FrameOverwrittenError,
    SharedMemoryRing,
    read_shared_memory_frame,
)


def test_shared_memory_ring_round_trip_and_slot_overwrite_detection() -> None:
    first_array = np.arange(12, dtype=np.float32).reshape(3, 4)
    second_array = first_array + 100.0
    third_array = first_array + 200.0

    with SharedMemoryRing(
        "ego_view_depth",
        slot_count=2,
        slot_size_bytes=first_array.nbytes,
    ) as ring:
        first = ring.write(
            first_array,
            received_ns=100,
            source_timestamp_ns=1000,
            source_clock="camera_unix",
        )
        second = ring.write(
            second_array,
            received_ns=200,
            source_timestamp_ns=1100,
            source_clock="camera_unix",
        )
        np.testing.assert_array_equal(read_shared_memory_frame(first), first_array)
        np.testing.assert_array_equal(read_shared_memory_frame(second), second_array)

        third = ring.write(
            third_array,
            received_ns=300,
            source_timestamp_ns=1200,
            source_clock="camera_unix",
        )
        with pytest.raises(FrameOverwrittenError):
            read_shared_memory_frame(first)
        np.testing.assert_array_equal(read_shared_memory_frame(third), third_array)


def test_shared_memory_ring_rejects_oversized_frame() -> None:
    with SharedMemoryRing("tiny", slot_count=2, slot_size_bytes=8) as ring:
        with pytest.raises(ValueError, match="slot capacity"):
            ring.write(np.zeros((3,), dtype=np.float32), received_ns=1)


def test_shared_memory_reader_rejects_inconsistent_shape_metadata() -> None:
    with SharedMemoryRing("depth", slot_count=2, slot_size_bytes=32) as ring:
        frame = ring.write(np.ones((4,), dtype=np.float32), received_ns=1)
        payload = frame.to_dict()
        payload["shape"] = [8]
        malformed = type(frame).from_dict(payload)

        with pytest.raises(ValueError, match="metadata requires"):
            read_shared_memory_frame(malformed)


def test_external_reader_does_not_unlink_producer_owned_memory() -> None:
    values = np.arange(8, dtype=np.float32)
    with SharedMemoryRing("external-reader", slot_count=2, slot_size_bytes=values.nbytes) as ring:
        frame = ring.write(values, received_ns=1)
        child_code = (
            "import json,sys; "
            "from gear_sonic.runtime.contracts import SharedMemoryFrame; "
            "from gear_sonic.runtime.shared_memory import read_shared_memory_frame; "
            "frame=SharedMemoryFrame.from_dict(json.loads(sys.argv[1])); "
            "print(float(read_shared_memory_frame(frame).sum()))"
        )

        result = subprocess.run(
            [sys.executable, "-c", child_code, json.dumps(frame.to_dict())],
            check=True,
            capture_output=True,
            text=True,
        )

        assert result.stdout.strip() == "28.0"
        assert "resource_tracker" not in result.stderr
        np.testing.assert_array_equal(read_shared_memory_frame(frame), values)
