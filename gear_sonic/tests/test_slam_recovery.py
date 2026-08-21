from __future__ import annotations

import math

from gear_sonic.utils.inference.navdp.recovery import (
    MaliciousDriftDetector,
    OdometrySample,
    SlamRecoveryLimits,
)


def limits(**overrides: object) -> SlamRecoveryLimits:
    values: dict[str, object] = {
        "startup_grace_s": 5.0,
        "max_planar_speed_m_s": 1.5,
        "max_position_step_m": 1.0,
        "max_yaw_rate_rad_s": 2.5,
        "consecutive_samples": 3,
    }
    values.update(overrides)
    return SlamRecoveryLimits.from_mapping(values)


def sample(
    timestamp_s: float,
    x: float,
    *,
    y: float = 0.0,
    yaw: float = 0.0,
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
    yaw_rate: float = 0.0,
) -> OdometrySample:
    return OdometrySample(
        timestamp_s=timestamp_s,
        x=x,
        y=y,
        yaw=yaw,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        yaw_rate=yaw_rate,
    )


def test_startup_grace_ignores_fastlio_initialization_motion() -> None:
    detector = MaliciousDriftDetector(limits(), monotonic=lambda: 100.0)

    assert detector.observe(sample(1.0, 0.0), monotonic_s=100.0) is None
    assert detector.observe(sample(1.1, 10.0), monotonic_s=102.0) is None
    assert detector.observe(sample(1.2, 20.0), monotonic_s=104.9) is None


def test_commanded_navigation_speed_never_triggers_recovery() -> None:
    detector = MaliciousDriftDetector(
        limits(startup_grace_s=0.0),
        monotonic=lambda: 0.0,
    )

    for index in range(20):
        assert (
            detector.observe(
                sample(index * 0.1, index * 0.03, velocity_x=0.3),
                monotonic_s=index * 0.1,
            )
            is None
        )


def test_one_speed_spike_is_not_malicious_drift() -> None:
    detector = MaliciousDriftDetector(
        limits(startup_grace_s=0.0),
        monotonic=lambda: 0.0,
    )

    assert detector.observe(sample(0.0, 0.0), monotonic_s=0.0) is None
    assert (
        detector.observe(
            sample(0.1, 0.05, velocity_x=2.0),
            monotonic_s=0.1,
        )
        is None
    )
    assert detector.observe(sample(0.2, 0.08), monotonic_s=0.2) is None


def test_sustained_impossible_translation_triggers_before_large_displacement() -> None:
    detector = MaliciousDriftDetector(
        limits(startup_grace_s=0.0),
        monotonic=lambda: 0.0,
    )
    positions = (0.0, 0.04, 0.08, 0.25, 0.48, 0.75)
    reasons = [
        detector.observe(sample(index * 0.1, x), monotonic_s=index * 0.1)
        for index, x in enumerate(positions)
    ]

    assert reasons[-1] is not None
    assert reasons[-1].startswith("planar_speed:")
    assert positions[-1] < limits().max_position_step_m


def test_meter_scale_position_jump_triggers_immediately() -> None:
    detector = MaliciousDriftDetector(
        limits(startup_grace_s=0.0),
        monotonic=lambda: 0.0,
    )

    assert detector.observe(sample(0.0, 0.0), monotonic_s=0.0) is None
    reason = detector.observe(sample(0.1, 1.01), monotonic_s=0.1)

    assert reason is not None
    assert reason.startswith("position_jump:")


def test_non_finite_odometry_triggers_immediately_after_grace() -> None:
    detector = MaliciousDriftDetector(
        limits(startup_grace_s=0.0),
        monotonic=lambda: 0.0,
    )

    reason = detector.observe(sample(0.0, math.nan), monotonic_s=0.0)

    assert reason == "non_finite_odometry"


def test_sustained_impossible_yaw_rate_triggers_recovery() -> None:
    detector = MaliciousDriftDetector(
        limits(startup_grace_s=0.0),
        monotonic=lambda: 0.0,
    )
    reasons = [
        detector.observe(
            sample(index * 0.1, 0.0, yaw_rate=3.0),
            monotonic_s=index * 0.1,
        )
        for index in range(3)
    ]

    assert reasons[:2] == [None, None]
    assert reasons[2] is not None
    assert reasons[2].startswith("yaw_rate:")


def test_reset_reapplies_grace_after_fastlio_restart() -> None:
    detector = MaliciousDriftDetector(
        limits(startup_grace_s=5.0),
        monotonic=lambda: 0.0,
    )
    detector.reset(started_s=20.0)

    assert detector.observe(sample(0.0, 0.0), monotonic_s=20.0) is None
    assert detector.observe(sample(0.1, 5.0), monotonic_s=22.0) is None
