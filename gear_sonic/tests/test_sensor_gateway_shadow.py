from __future__ import annotations

import time

import numpy as np
import zmq

from gear_sonic.runtime.client import SensorGatewayClient
from gear_sonic.runtime.sensor_gateway import SensorGatewayCore, SensorGatewayRpc
from gear_sonic.runtime.shadow import (
    SensorGatewayShadowComparator,
    ShadowComparisonStats,
    compare_sensor_arrays,
    unavailable_comparison,
)


def test_array_comparison_reports_exact_data_and_timestamp_match() -> None:
    core = SensorGatewayCore(slot_count=8, history_size=16)
    try:
        source_ns = 2_000_000_000
        values = np.arange(12, dtype=np.float32).reshape(4, 3)
        frame = core.publish_array(
            "ros/livox_lidar_xyz",
            values,
            received_ns=1_000_000_000,
            source_timestamp_ns=source_ns,
            source_clock="ros_time",
        )
        comparison = compare_sensor_arrays(
            "ros/livox_lidar_xyz",
            values.copy(),
            values.copy(),
            legacy_source_timestamp_ns=source_ns,
            gateway_frame=frame,
            max_source_skew_ms=1.0,
        )

        assert comparison.matched
        assert comparison.comparable
        assert comparison.exact_values
        assert comparison.source_skew_ms == 0.0
        assert comparison.max_abs_error == 0.0
    finally:
        core.close()


def test_array_comparison_explains_shape_dtype_value_and_timestamp_mismatches() -> None:
    core = SensorGatewayCore(slot_count=8, history_size=16)
    try:
        frame = core.publish_array(
            "camera/chest_view",
            np.zeros((2, 2, 3), dtype=np.uint8),
            received_ns=1_000_000_000,
            source_timestamp_ns=2_000_000_000,
            source_clock="camera_unix",
        )
        comparison = compare_sensor_arrays(
            "camera/chest_view",
            np.ones((4,), dtype=np.float32),
            np.zeros((2, 2, 3), dtype=np.uint8),
            legacy_source_timestamp_ns=2_010_000_000,
            gateway_frame=frame,
            max_source_skew_ms=1.0,
        )

        assert not comparison.matched
        assert comparison.comparable
        assert "shape mismatch" in comparison.reason
        assert "dtype mismatch" in comparison.reason
        assert "value mismatch" in comparison.reason
        assert "source timestamp mismatch" in comparison.reason
    finally:
        core.close()


def test_shadow_comparator_matches_gateway_by_explicit_source_timestamp() -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=8, history_size=16)
    rpc = SensorGatewayRpc(context, "inproc://shadow-comparator", core)
    source_ns = 2_000_000_000
    values = np.arange(10, dtype=np.float64)
    core.publish_array(
        "ros/livox_imu",
        values,
        received_ns=time.monotonic_ns(),
        source_timestamp_ns=source_ns,
        source_clock="ros_time",
        expected_hz=200.0,
    )
    client = SensorGatewayClient(
        "inproc://shadow-comparator",
        context=context,
        request_timeout_ms=100,
    )
    comparator = SensorGatewayShadowComparator(client, retries=0)
    result: list = []

    import threading

    worker = threading.Thread(
        target=lambda: result.append(
            comparator.compare(
                "ros/livox_imu",
                values.copy(),
                source_timestamp_ns=source_ns,
            )
        )
    )
    try:
        worker.start()
        assert rpc.serve_once(timeout_ms=100)
        worker.join(1.0)
        assert not worker.is_alive()
        assert result[0].matched
        stats = comparator.stats.to_dict()["ros/livox_imu"]
        assert stats["samples"] == 1
        assert stats["match_rate"] == 1.0
    finally:
        client.close()
        rpc.close()
        core.close()
        context.term()


def test_shadow_stats_preserve_last_mismatch_for_future_gui() -> None:
    stats = ShadowComparisonStats()
    core = SensorGatewayCore(slot_count=8, history_size=16)
    try:
        frame = core.publish_array(
            "test",
            np.zeros((1,), dtype=np.float32),
            received_ns=1,
            source_timestamp_ns=2,
        )
        mismatch = compare_sensor_arrays(
            "test",
            np.ones((1,), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            legacy_source_timestamp_ns=2,
            gateway_frame=frame,
            max_source_skew_ms=0.0,
        )
        stats.record(mismatch)

        payload = stats.to_dict()["test"]
        assert payload["mismatches"] == 1
        assert payload["unpaired"] == 0
        assert payload["compared"] == 1
        assert payload["match_rate"] == 0.0
        assert payload["coverage_rate"] == 1.0
        assert payload["last"]["reason"] == "value mismatch"
    finally:
        core.close()


def test_shadow_stats_keep_unpaired_samples_out_of_value_match_rate() -> None:
    stats = ShadowComparisonStats()
    stats.record(
        unavailable_comparison(
            "camera/chest_view",
            np.zeros((2, 2, 3), dtype=np.uint8),
            "no identical source timestamp",
        )
    )

    payload = stats.to_dict()["camera/chest_view"]
    assert payload["samples"] == 1
    assert payload["compared"] == 0
    assert payload["unpaired"] == 1
    assert payload["mismatches"] == 0
    assert payload["coverage_rate"] == 0.0
