from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from gear_sonic.utils.planner_control import (
    NavigationCommand,
    PlannerVelocityCommand,
    PlannerVelocityExecutorCore,
    SafetySnapshot,
    build_navigation_runtime_status_message,
    build_planner_velocity_message,
    decode_navigation_runtime_status_message,
    decode_planner_velocity_message,
)
from gear_sonic.utils.planner_control.executor_service import PlannerSafetySensorMonitor
from gear_sonic.utils.teleop.sonic_orientation_telemetry import OrientationTracker


def navigation(
    mode: str,
    *,
    generation: int = 1,
    velocity: tuple[float, float, float] | None = None,
    source: str = "",
) -> NavigationCommand:
    return NavigationCommand(
        mode=mode,
        generation=generation,
        timestamp=1.0,
        velocity=velocity,
        source=source,
    )


def fresh_safety(now: float, depth: np.ndarray | None = None) -> SafetySnapshot:
    return SafetySnapshot(now, depth)


class FakeSafetySensorClient:
    def __init__(self, *, depth_error: Exception | None = None) -> None:
        self.depth_error = depth_error
        self.lidar_frame = SimpleNamespace(
            metadata=SimpleNamespace(timestamp_ns=12_300_000_000),
        )
        self.depth_frame = SimpleNamespace(
            attributes={"camera_info": {"depth_scale_m": 0.001}},
        )
        self.state_frame = SimpleNamespace(
            metadata=SimpleNamespace(
                sequence=9,
                timestamp_ns=12_350_000_000,
            ),
        )

    def request_snapshot(self, _request):
        return SimpleNamespace(
            complete=True,
            frames={PlannerSafetySensorMonitor.LIDAR_STREAM: self.lidar_frame},
        )

    def read_snapshot(self, _request, *, retries: int):
        assert retries == 0
        stream = _request.streams[0]
        if stream == PlannerSafetySensorMonitor.DEPTH_STREAM:
            if self.depth_error is not None:
                raise self.depth_error
            return SimpleNamespace(
                arrays={stream: np.full((2, 3), 750, dtype=np.uint16)},
                snapshot=SimpleNamespace(frames={stream: self.depth_frame}),
            )
        assert stream == PlannerSafetySensorMonitor.ROBOT_STATE_STREAM
        payload = msgpack.packb(
            {"base_quat": [1.0, 0.0, 0.0, 0.0]},
            use_bin_type=True,
        )
        return SimpleNamespace(
            arrays={stream: np.frombuffer(payload, dtype=np.uint8)},
            snapshot=SimpleNamespace(frames={stream: self.state_frame}),
        )


def test_navdp_velocity_protocol_preserves_generation_and_heading_frame() -> None:
    payload = build_planner_velocity_message(
        generation=8,
        source="navdp",
        velocity=(0.3, 0.0, 0.4),
        timestamp=12.5,
        heading_target_rad=0.7,
        heading_reference_rad=0.2,
    )

    assert decode_planner_velocity_message(payload) == PlannerVelocityCommand(
        generation=8,
        timestamp=12.5,
        source="navdp",
        velocity=(0.3, 0.0, 0.4),
        heading_target_rad=0.7,
        heading_reference_rad=0.2,
    )


def test_navigation_runtime_status_preserves_requested_and_final_velocity() -> None:
    message = build_navigation_runtime_status_message(
        generation=3,
        timestamp=12.5,
        mode="manual_velocity",
        source="operator_console",
        requested_velocity=(0.3, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        reason="depth_hard_stop",
    )

    status = decode_navigation_runtime_status_message(message)

    assert status.generation == 3
    assert status.mode == "manual_velocity"
    assert status.source == "operator_console"
    assert status.requested_velocity == (0.3, 0.0, 0.0)
    assert status.velocity == (0.0, 0.0, 0.0)
    assert status.reason == "depth_hard_stop"

def test_safety_monitor_uses_sensor_receive_times_and_scales_raw_depth() -> None:
    monitor = PlannerSafetySensorMonitor(
        "inproc://unused",
        poll_hz=20.0,
        request_timeout_ms=100,
        max_age_ms=1000.0,
        client=FakeSafetySensorClient(),
    )

    monitor.poll_once()
    snapshot = monitor.snapshot()

    assert snapshot.radar_timestamp_s == pytest.approx(12.3)
    np.testing.assert_allclose(snapshot.depth_m, 0.75)
    monitor.close()


def test_safety_monitor_preserves_fresh_radar_when_depth_is_unavailable() -> None:
    monitor = PlannerSafetySensorMonitor(
        "inproc://unused",
        poll_hz=20.0,
        request_timeout_ms=100,
        max_age_ms=1000.0,
        client=FakeSafetySensorClient(depth_error=RuntimeError("depth unavailable")),
    )

    with pytest.raises(RuntimeError, match="depth unavailable"):
        monitor.poll_once()

    snapshot = monitor.snapshot()
    assert snapshot.radar_timestamp_s == pytest.approx(12.3)
    assert snapshot.depth_m is None
    monitor.close()


def test_sensor_monitor_decodes_latest_g1_debug_base_quaternion() -> None:
    monitor = PlannerSafetySensorMonitor(
        "inproc://unused",
        poll_hz=20.0,
        request_timeout_ms=100,
        max_age_ms=1000.0,
        include_robot_state=True,
        client=FakeSafetySensorClient(),
    )

    monitor.poll_once()
    orientation = monitor.orientation_snapshot()

    assert orientation.sequence == 9
    assert orientation.received_at_s == pytest.approx(12.35)
    assert orientation.state is not None
    np.testing.assert_allclose(
        orientation.state["base_quat"],
        [1.0, 0.0, 0.0, 0.0],
    )
    monitor.close()


def test_safety_monitor_lidar_refresh_isolated_from_blocked_depth_reader() -> None:
    lidar_refreshed = threading.Event()
    release_depth = threading.Event()
    depth_entered = threading.Event()

    class LidarClient:
        calls = 0

        def request_snapshot(self, _request):
            self.calls += 1
            if self.calls >= 3:
                lidar_refreshed.set()
            frame = SimpleNamespace(
                metadata=SimpleNamespace(timestamp_ns=time.monotonic_ns())
            )
            return SimpleNamespace(
                complete=True,
                frames={PlannerSafetySensorMonitor.LIDAR_STREAM: frame},
            )

    class BlockingAuxiliaryClient:
        def read_snapshot(self, _request, *, retries):
            assert retries == 0
            depth_entered.set()
            release_depth.wait(timeout=1.0)
            raise RuntimeError("depth reader released")

    monitor = PlannerSafetySensorMonitor(
        "inproc://unused",
        poll_hz=100.0,
        request_timeout_ms=100,
        max_age_ms=1000.0,
        lidar_client=LidarClient(),
        auxiliary_client=BlockingAuxiliaryClient(),
    )
    monitor.start()
    try:
        assert depth_entered.wait(timeout=0.5)
        assert lidar_refreshed.wait(timeout=0.5)
        assert monitor.snapshot().radar_timestamp_s > 0.0
    finally:
        release_depth.set()
        monitor.close()


def test_wasd_and_base_pose_share_the_same_manual_execution_path() -> None:
    core = PlannerVelocityExecutorCore()
    assert core.accept_navigation(
        navigation(
            "manual_velocity",
            velocity=(0.0, 0.15, 0.0),
            source="operator_console",
        ),
        now=10.0,
    )
    wasd = core.decide(now=10.1, safety=fresh_safety(10.1))
    assert wasd.velocity == pytest.approx((0.0, 0.15, 0.0))
    assert wasd.source == "operator_console"

    assert core.accept_navigation(
        navigation(
            "manual_velocity",
            generation=2,
            velocity=(0.3, 0.0, 0.0),
            source="base_pose_agent",
        ),
        now=11.0,
    )
    base_pose = core.decide(now=11.1, safety=fresh_safety(11.1))
    assert base_pose.velocity == pytest.approx((0.3, 0.0, 0.0))
    assert base_pose.source == "base_pose_agent"


def test_orientation_sample_uses_executor_post_integration_heading() -> None:
    core = PlannerVelocityExecutorCore(control_hz=20.0)
    tracker = OrientationTracker()
    tracker.update_state(
        {"base_quat": [1.0, 0.0, 0.0, 0.0]},
        received_at_monotonic_s=10.0,
        heading_setpoint_rad=core.sonic.heading,
    )
    core.accept_navigation(
        navigation("manual_velocity", velocity=(0.0, 0.0, 0.4)),
        now=10.0,
    )

    core.decide(now=10.05, safety=fresh_safety(10.05))
    sample = tracker.sample(10.05, core.sonic.heading)

    assert sample.actual_heading_rad == pytest.approx(0.0)
    assert sample.heading_setpoint_rad == pytest.approx(0.02)
    assert sample.heading_lag_rad == pytest.approx(0.02)


def test_common_radar_and_depth_safety_apply_to_every_source() -> None:
    core = PlannerVelocityExecutorCore(radar_timeout_s=0.75)
    core.accept_navigation(
        navigation("manual_velocity", velocity=(0.3, 0.0, 0.1)), now=1.0
    )

    radar_stop = core.decide(now=2.0, safety=SafetySnapshot())
    assert radar_stop.velocity == (0.0, 0.0, 0.0)
    assert radar_stop.reason == "radar_timeout"

    depth = np.ones((60, 60), dtype=np.float32)
    depth.flat[:2001] = 0.09
    depth_stop = core.decide(now=2.1, safety=fresh_safety(2.1, depth))
    assert depth_stop.velocity == (0.0, 0.0, 0.0)
    assert depth_stop.reason == "depth_hard_stop"


def test_missing_or_future_dated_radar_is_fail_safe() -> None:
    core = PlannerVelocityExecutorCore()
    core.accept_navigation(
        navigation("manual_velocity", velocity=(0.3, 0.0, 0.0)), now=0.0
    )

    missing = core.decide(now=0.1, safety=SafetySnapshot())
    future = core.decide(now=1.0, safety=SafetySnapshot(radar_timestamp_s=2.0))

    assert missing.reason == "radar_timeout"
    assert future.reason == "radar_timeout"
    assert missing.velocity == future.velocity == (0.0, 0.0, 0.0)


def test_stale_or_wrong_generation_navdp_output_cannot_move_robot() -> None:
    core = PlannerVelocityExecutorCore(navdp_timeout_s=0.3)
    core.accept_navigation(navigation("nav_goal", generation=5), now=10.0)
    assert not core.accept_planner_velocity(
        PlannerVelocityCommand(4, 1.0, "navdp", (0.3, 0.0, 0.0)), now=10.1
    )
    waiting = core.decide(now=10.1, safety=fresh_safety(10.1))
    assert waiting.velocity == (0.0, 0.0, 0.0)

    assert core.accept_planner_velocity(
        PlannerVelocityCommand(5, 1.0, "navdp", (0.3, 0.0, 0.2)), now=10.2
    )
    assert core.decide(now=10.49, safety=fresh_safety(10.49)).velocity == pytest.approx(
        (0.3, 0.0, 0.2)
    )
    expired = core.decide(now=10.51, safety=fresh_safety(10.51))
    assert expired.velocity == (0.0, 0.0, 0.0)
    assert expired.reason == "navdp_velocity_timeout"


def test_navdp_fastlio_heading_is_rebased_to_current_sonic_heading() -> None:
    core = PlannerVelocityExecutorCore()
    core.sonic.heading = 1.0
    core.accept_navigation(navigation("nav_goal", generation=3), now=5.0)
    core.accept_planner_velocity(
        PlannerVelocityCommand(
            3,
            1.0,
            "navdp",
            (0.2, 0.0, 0.3),
            heading_target_rad=0.7,
            heading_reference_rad=0.2,
        ),
        now=5.1,
    )

    core.decide(now=5.1, safety=fresh_safety(5.1))

    assert core.sonic.heading == pytest.approx(math.remainder(1.5, 2.0 * math.pi))


def test_heading_goal_integrates_angular_velocity_instead_of_jumping_to_target() -> None:
    core = PlannerVelocityExecutorCore(control_hz=20.0)
    core.sonic.heading = 1.0
    core.accept_navigation(
        NavigationCommand(
            mode="heading_goal",
            generation=3,
            timestamp=1.0,
            heading_delta_rad=-math.pi / 2.0,
        ),
        now=5.0,
    )
    core.accept_planner_velocity(
        PlannerVelocityCommand(
            3,
            1.0,
            "navdp",
            (0.0, 0.0, -0.4),
            heading_target_rad=-1.3,
            heading_reference_rad=0.2,
        ),
        now=5.1,
    )

    core.decide(now=5.1, safety=fresh_safety(5.1))
    assert core.sonic.heading == pytest.approx(0.98)

    core.decide(now=5.15, safety=fresh_safety(5.15))
    assert core.sonic.heading == pytest.approx(0.96)


def test_heading_goal_safety_block_freezes_incremental_facing() -> None:
    core = PlannerVelocityExecutorCore(control_hz=20.0)
    core.accept_navigation(
        NavigationCommand(
            mode="heading_goal",
            generation=4,
            timestamp=1.0,
            heading_delta_rad=math.pi / 2.0,
        ),
        now=8.0,
    )
    core.accept_planner_velocity(
        PlannerVelocityCommand(
            4,
            1.0,
            "navdp",
            (0.0, 0.0, 0.4),
            heading_target_rad=1.5,
            heading_reference_rad=0.0,
        ),
        now=8.1,
    )

    blocked = core.decide(now=8.1, safety=SafetySnapshot())
    assert blocked.reason == "radar_timeout"
    assert blocked.velocity == (0.0, 0.0, 0.0)
    assert core.sonic.heading == pytest.approx(0.0)

    core.decide(now=8.15, safety=fresh_safety(8.15))
    assert core.sonic.heading == pytest.approx(0.02)


def test_heading_goal_keeps_replanning_until_zero_velocity_snaps_to_target() -> None:
    core = PlannerVelocityExecutorCore(control_hz=20.0)
    core.sonic.heading = 1.0
    core.accept_navigation(
        NavigationCommand(
            mode="heading_goal",
            generation=5,
            timestamp=1.0,
            heading_delta_rad=0.01,
        ),
        now=9.0,
    )
    core.accept_planner_velocity(
        PlannerVelocityCommand(
            5,
            1.0,
            "navdp",
            (0.0, 0.0, 0.4),
            heading_target_rad=0.21,
            heading_reference_rad=0.2,
        ),
        now=9.1,
    )

    core.decide(now=9.1, safety=fresh_safety(9.1))
    assert core.sonic.heading == pytest.approx(1.02)

    core.accept_planner_velocity(
        PlannerVelocityCommand(
            5,
            1.1,
            "navdp",
            (0.0, 0.0, 0.0),
            heading_target_rad=0.21,
            heading_reference_rad=0.2,
        ),
        now=9.15,
    )
    core.decide(now=9.15, safety=fresh_safety(9.15))

    assert core.sonic.heading == pytest.approx(1.01)
