"""Contracts for SONIC heading and measured-yaw diagnostics."""

from __future__ import annotations

import json
import math

import pytest

from gear_sonic.utils.teleop.sonic_orientation_telemetry import (
    LatestOrientationTelemetry,
    OrientationTelemetrySample,
    OrientationTracker,
    decode_orientation_telemetry,
    encode_orientation_telemetry,
    quaternion_yaw_wxyz,
)


def yaw_quaternion(yaw_rad: float) -> list[float]:
    return [
        math.cos(yaw_rad / 2.0),
        0.0,
        0.0,
        math.sin(yaw_rad / 2.0),
    ]


@pytest.mark.parametrize(
    ("quaternion", "expected_yaw"),
    [
        ([1.0, 0.0, 0.0, 0.0], 0.0),
        (yaw_quaternion(0.5), 0.5),
        ([item * 2.0 for item in yaw_quaternion(-0.4)], -0.4),
    ],
)
def test_quaternion_yaw_wxyz_normalizes_and_extracts_yaw(
    quaternion: list[float], expected_yaw: float
) -> None:
    assert quaternion_yaw_wxyz(quaternion) == pytest.approx(expected_yaw)


@pytest.mark.parametrize(
    "quaternion",
    [
        [],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [math.nan, 0.0, 0.0, 0.0],
        [math.inf, 0.0, 0.0, 0.0],
    ],
)
def test_quaternion_yaw_wxyz_rejects_invalid_values(
    quaternion: list[float],
) -> None:
    with pytest.raises(ValueError, match="quaternion"):
        quaternion_yaw_wxyz(quaternion)


def test_orientation_tracker_reports_heading_and_wrapped_lag() -> None:
    tracker = OrientationTracker()
    tracker.update_state(
        {"base_quat": yaw_quaternion(0.4)},
        received_at_monotonic_s=10.0,
        heading_setpoint_rad=0.0,
    )
    tracker.update_state(
        {"base_quat": yaw_quaternion(0.5)},
        received_at_monotonic_s=10.1,
        heading_setpoint_rad=0.15,
    )

    sample = tracker.sample(
        now_monotonic_s=10.12,
        heading_setpoint_rad=0.15,
    )

    assert sample.actual_yaw_rad == pytest.approx(0.5)
    assert sample.actual_heading_rad == pytest.approx(0.1)
    assert sample.heading_setpoint_rad == pytest.approx(0.15)
    assert sample.heading_lag_rad == pytest.approx(0.05)
    assert sample.state_age_s == pytest.approx(0.02)


def test_orientation_tracker_without_state_keeps_actual_fields_null() -> None:
    sample = OrientationTracker().sample(
        now_monotonic_s=2.0,
        heading_setpoint_rad=0.2,
    )

    assert sample.actual_yaw_rad is None
    assert sample.actual_heading_rad is None
    assert sample.heading_setpoint_rad == pytest.approx(0.2)
    assert sample.heading_lag_rad is None
    assert sample.state_age_s is None


def test_orientation_tracker_does_not_invent_origin_after_heading_moves() -> None:
    tracker = OrientationTracker()
    tracker.sample(now_monotonic_s=1.0, heading_setpoint_rad=0.1)
    tracker.update_state(
        {"base_quat": yaw_quaternion(0.4)},
        received_at_monotonic_s=1.1,
        heading_setpoint_rad=0.1,
    )

    sample = tracker.sample(
        now_monotonic_s=1.2,
        heading_setpoint_rad=0.1,
    )

    assert sample.actual_yaw_rad == pytest.approx(0.4)
    assert sample.actual_heading_rad is None
    assert sample.heading_lag_rad is None


def test_orientation_tracker_wraps_lag_across_pi_boundary() -> None:
    tracker = OrientationTracker()
    tracker.update_state(
        {"base_quat": yaw_quaternion(0.0)},
        received_at_monotonic_s=1.0,
        heading_setpoint_rad=0.0,
    )
    tracker.update_state(
        {"base_quat": yaw_quaternion(-math.pi + 0.05)},
        received_at_monotonic_s=1.1,
        heading_setpoint_rad=math.pi - 0.05,
    )

    sample = tracker.sample(
        now_monotonic_s=1.1,
        heading_setpoint_rad=math.pi - 0.05,
    )

    assert sample.heading_lag_rad == pytest.approx(-0.1)


def test_orientation_telemetry_round_trip_and_latest_sample_age() -> None:
    older = OrientationTelemetrySample(
        emitted_at_monotonic_s=10.0,
        actual_yaw_rad=0.4,
        actual_heading_rad=0.1,
        heading_setpoint_rad=0.15,
        heading_lag_rad=0.05,
        state_age_s=0.01,
    )
    newer = OrientationTelemetrySample(
        emitted_at_monotonic_s=10.2,
        actual_yaw_rad=0.5,
        actual_heading_rad=0.2,
        heading_setpoint_rad=0.25,
        heading_lag_rad=0.05,
        state_age_s=0.02,
    )
    latest = LatestOrientationTelemetry()

    latest.update(encode_orientation_telemetry(older))
    latest.update(encode_orientation_telemetry(newer))
    latest.update(encode_orientation_telemetry(older))

    assert decode_orientation_telemetry(
        encode_orientation_telemetry(newer)
    ) == newer
    assert latest.diagnostics(now_monotonic_s=10.3) == pytest.approx(
        {
            "actual_yaw_rad": 0.5,
            "actual_heading_rad": 0.2,
            "heading_setpoint_rad": 0.25,
            "heading_lag_rad": 0.05,
            "state_age_s": 0.02,
            "telemetry_age_s": 0.1,
        }
    )


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "{}",
        json.dumps({"type": "wrong", "version": 1}),
        json.dumps({"type": "sonic_orientation_telemetry", "version": 2}),
        json.dumps(
            {
                "type": "sonic_orientation_telemetry",
                "version": 1,
                "emitted_at_monotonic_s": 1.0,
                "actual_yaw_rad": None,
                "actual_heading_rad": 0.1,
                "heading_setpoint_rad": 0.2,
                "heading_lag_rad": 0.1,
                "state_age_s": 0.0,
            }
        ),
        (
            '{"type":"sonic_orientation_telemetry","version":1,'
            '"emitted_at_monotonic_s":1.0,"actual_yaw_rad":null,'
            '"actual_heading_rad":null,"heading_setpoint_rad":NaN,'
            '"heading_lag_rad":null,"state_age_s":null}'
        ),
    ],
)
def test_orientation_telemetry_decoder_rejects_malformed_payloads(
    payload: str,
) -> None:
    with pytest.raises(ValueError):
        decode_orientation_telemetry(payload)
