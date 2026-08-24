from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gear_sonic.utils.inference.vla.runtime import _current_vla_safety_reason
from gear_sonic.utils.inference.vla.safety import VlaSafetyGate
from gear_sonic.utils.planner_control.executor import SafetySnapshot


def test_vla_safety_reuses_lidar_depth_and_state_freshness_gate():
    gate = VlaSafetyGate(radar_timeout_s=0.75, robot_state_timeout_s=0.5)
    clear = SafetySnapshot(
        radar_timestamp_s=9.8,
        depth_m=np.full((80, 80), 1.0, np.float32),
    )
    assert gate.reason(
        now=10.0, safety=clear, robot_state_timestamp_s=9.8,
    ) == "clear"
    assert gate.reason(
        now=10.0, safety=clear, robot_state_timestamp_s=9.0,
    ) == "robot_state_timeout"
    assert gate.reason(
        now=10.0,
        safety=SafetySnapshot(8.0, clear.depth_m),
        robot_state_timestamp_s=9.8,
    ) == "radar_timeout"
    blocked_depth = np.full((80, 80), 1.0, np.float32)
    blocked_depth[:50, :50] = 0.05
    assert gate.reason(
        now=10.0,
        safety=SafetySnapshot(9.8, blocked_depth),
        robot_state_timestamp_s=9.8,
    ) == "depth_hard_stop"


def test_vla_safety_captures_monitor_snapshots_before_current_time():
    """A state update during slow command handling must not look future-dated."""

    call_order = []
    clear = SafetySnapshot(
        radar_timestamp_s=12.0,
        depth_m=np.full((80, 80), 1.0, np.float32),
    )

    class Monitor:
        def snapshot(self):
            call_order.append("safety")
            return clear

        def orientation_snapshot(self):
            call_order.append("orientation")
            return SimpleNamespace(received_at_s=12.0)

    def monotonic():
        call_order.append("now")
        return 12.1

    reason = _current_vla_safety_reason(
        VlaSafetyGate(radar_timeout_s=0.75, robot_state_timeout_s=0.5),
        Monitor(),
        monotonic=monotonic,
    )

    assert reason == "clear"
    assert call_order == ["safety", "orientation", "now"]
