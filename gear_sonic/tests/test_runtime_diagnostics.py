from __future__ import annotations

import pytest

from gear_sonic.runtime.diagnostics import EndpointHealthMonitor, EndpointState


def test_endpoint_health_transitions_without_active_probes() -> None:
    monitor = EndpointHealthMonitor(
        "cpp_state",
        expected_hz=10.0,
        stale_after_s=0.3,
        down_after_s=1.0,
        started_ns=0,
    )

    assert monitor.snapshot(now_ns=100_000_000).state is EndpointState.WAITING

    monitor.observe(
        received_ns=100_000_000,
        source_timestamp_ns=80_000_000,
        sequence=5,
    )
    monitor.observe(
        received_ns=200_000_000,
        source_timestamp_ns=175_000_000,
        sequence=8,
    )
    healthy = monitor.snapshot(now_ns=250_000_000)
    assert healthy.state is EndpointState.HEALTHY
    assert healthy.message_count == 2
    assert healthy.dropped_messages == 2
    assert healthy.rate_hz == pytest.approx(10.0)
    assert healthy.latency_ms == pytest.approx(25.0)

    assert monitor.snapshot(now_ns=500_000_000).state is EndpointState.STALE
    assert monitor.snapshot(now_ns=1_200_000_000).state is EndpointState.DOWN


def test_endpoint_failure_and_recovery_are_explicit() -> None:
    monitor = EndpointHealthMonitor(
        "policy_server",
        expected_hz=2.0,
        stale_after_s=1.0,
        down_after_s=3.0,
        started_ns=0,
    )
    monitor.observe(received_ns=100_000_000, round_trip_ms=12.5)
    monitor.record_failure("request timed out")

    failed = monitor.snapshot(now_ns=200_000_000)
    assert failed.state is EndpointState.DOWN
    assert failed.failure_count == 1
    assert failed.consecutive_failures == 1
    assert failed.last_error == "request timed out"

    monitor.observe(received_ns=300_000_000, round_trip_ms=8.0)
    recovered = monitor.snapshot(now_ns=350_000_000)
    assert recovered.state is EndpointState.HEALTHY
    assert recovered.consecutive_failures == 0
    assert recovered.latency_ms == pytest.approx(8.0)
    assert recovered.last_error == ""


def test_out_of_order_sequence_is_not_counted_as_packet_loss() -> None:
    monitor = EndpointHealthMonitor("camera_server", expected_hz=30.0, started_ns=0)
    monitor.observe(received_ns=10, sequence=4)
    monitor.observe(received_ns=20, sequence=4)
    monitor.observe(received_ns=30, sequence=3)
    monitor.observe(received_ns=40, sequence=5)

    snapshot = monitor.snapshot(now_ns=50)
    assert snapshot.out_of_order_messages == 2
    assert snapshot.dropped_messages == 0
