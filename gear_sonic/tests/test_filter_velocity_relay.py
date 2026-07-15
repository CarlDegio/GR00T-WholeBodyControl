import math
import struct
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from filter_velocity_relay import (  # noqa: E402
    LatestVelocity,
    PlannerState,
    ReadySignal,
    StartHeartbeat,
    build_planner_from_velocity,
)


def test_latest_velocity_holds_then_times_out_to_zero():
    latest = LatestVelocity(timeout_s=1.0)
    latest.update((0.4, -0.2, 0.7), now=10.0)

    assert latest.value(now=10.999) == pytest.approx((0.4, -0.2, 0.7))
    assert latest.value(now=11.0) == (0.0, 0.0, 0.0)


def test_planner_conversion_integrates_yaw_and_rotates_local_velocity():
    state = PlannerState()

    message = build_planner_from_velocity(state, (1.0, 0.0, math.pi / 2), dt=1.0)

    assert message.startswith(b"planner")
    payload = message[7 + 1280 :]
    mode = struct.unpack_from("<i", payload, 0)[0]
    movement = struct.unpack_from("<fff", payload, 4)
    facing = struct.unpack_from("<fff", payload, 16)
    speed = struct.unpack_from("<f", payload, 28)[0]
    assert mode == 2
    assert movement == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert facing == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert speed == pytest.approx(1.0)


def test_zero_velocity_uses_idle_without_changing_heading():
    state = PlannerState(heading=0.7)

    message = build_planner_from_velocity(state, (0.0, 0.0, 0.0), dt=0.1)

    payload = message[7 + 1280 :]
    assert struct.unpack_from("<i", payload, 0)[0] == 0
    assert struct.unpack_from("<fff", payload, 4) == (0.0, 0.0, 0.0)
    assert state.heading == pytest.approx(0.7)


def test_start_heartbeat_repeats_for_late_cpp_subscriber():
    heartbeat = StartHeartbeat(interval_s=0.5)

    assert heartbeat.due(now=10.0)
    assert not heartbeat.due(now=10.49)
    assert heartbeat.due(now=10.5)


def test_ready_signal_exists_only_after_first_filter_packet(tmp_path):
    path = tmp_path / "ready"
    signal = ReadySignal(path)

    assert not path.exists()
    signal.mark()
    assert path.exists()
    signal.clear()
    assert not path.exists()
