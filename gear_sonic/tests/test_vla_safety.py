from __future__ import annotations

import numpy as np

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
