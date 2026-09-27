from __future__ import annotations

from dataclasses import replace
import math
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from baselinelavira import BASELINE_OVERLAY, build_profile
from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.utils.inference.iplanner.control import (
    IPlannerNavigation,
    load_navigation_config,
    pure_pursuit,
    truncate_trajectory,
)
from gear_sonic.utils.inference.navdp.gateway import load_navdp_planner_config
from gear_sonic.utils.inference.navdp.navigation import Pose2D
from gear_sonic.utils.inference.navdp.runtime import _NavDPRuntimeState


@pytest.fixture
def nav_config():
    return load_navigation_config(load_runtime_profile(overlays=(BASELINE_OVERLAY,)))


def test_pure_pursuit_selects_half_meter_lookahead_and_applies_configured_speed_range(nav_config):
    path = [[0.1, 0], [0.5, 0], [1.0, 1.0], [2.0, 0.0]]
    velocity, index = pure_pursuit(path, 4.0, nav_config)
    assert index == 2  # The exact 0.5 m point is skipped.
    assert velocity == pytest.approx((0.3, 0.0, 0.4))
    assert pure_pursuit([[1, -1]], 4.0, nav_config)[0] == pytest.approx((0.3, 0.0, -0.4))
    assert pure_pursuit([[0, 0.6]], 4.0, nav_config)[0] == pytest.approx((0.3, 0.0, 0.5))
    assert pure_pursuit([[-1, 0]], 4.0, nav_config)[0][0] == pytest.approx(0.3)
    assert pure_pursuit([[1, 0]], 4.0, nav_config)[0] == pytest.approx((0.4, 0.0, 0.0))
    # Steering continues outside the configured arrival tolerance.
    assert pure_pursuit(path, 2.01, nav_config)[0] == pytest.approx((0.3, 0.0, 0.4))
    assert pure_pursuit(path, 1.99, nav_config)[0] == (0.0, 0.0, 0.0)
    assert pure_pursuit([], 4.0, nav_config)[0] == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("minimum", [-0.1, 0.41, float("nan"), float("inf")])
def test_invalid_forward_speed_bounds_are_rejected(nav_config, minimum):
    with pytest.raises(ValueError, match="min_forward_speed"):
        replace(nav_config, min_forward_speed=minimum)


def test_older_runtime_snapshots_keep_their_original_speed_floor(tmp_path):
    import yaml

    payload = build_profile()
    navigation = payload["components"]["iplanner_navigation"]
    navigation.pop("min_forward_speed")
    navigation["target_speed"] = 0.3
    path = tmp_path / "older_runtime.yaml"
    path.write_text(yaml.safe_dump(payload))
    config = load_navigation_config(load_runtime_profile(path))
    assert config.min_forward_speed == 0.1
    assert pure_pursuit([[0, 0.6]], 4.0, config)[0] == pytest.approx((0.15, 0.0, 0.5))


def test_safe_distance_is_applied_only_to_the_initial_plan(nav_config, monkeypatch):
    from gear_sonic.utils.inference.iplanner import control

    nav_config = replace(nav_config, safe_distance=0.5)
    monkeypatch.setattr(control.time, "monotonic", lambda: 10.0)
    path = np.column_stack((np.arange(0, 3.01, 0.1), np.zeros(31)))
    shortened = truncate_trajectory(path, 0.5)
    np.testing.assert_allclose(shortened[-1], [2.5, 0])
    state = _NavDPRuntimeState(mode="nav_goal", generation=2, segment_id=4, world_goal=(10, 23))
    nav = IPlannerNavigation(nav_config)
    nav.sync_command(state, now=9.0)
    results = queue.Queue()
    results.put((1, 4, path * 999, Pose2D(10, 20, math.pi / 2), None, 0.1))
    results.put((2, 4, path, Pose2D(10, 20, math.pi / 2), None, 0.1))
    callbacks = dict(send_metrics=lambda *a, **kw: None, send_status=lambda *a: None,
                     report_event=lambda *a, **kw: None)
    nav.consume_results(state, results, **callbacks)
    np.testing.assert_allclose(state.world_goal, [10, 23])
    np.testing.assert_allclose(nav.world_path[-1], [10, 22.5])
    assert state.mpc_reference.size == 0
    assert nav.control(state, pose=Pose2D(10, 20, math.pi / 2), now=10.0,
                       config=load_navdp_planner_config()) == pytest.approx((0.4, 0, 0))
    results.put((2, 4, path[:21], Pose2D(10, 20.5, math.pi / 2), None, 0.1))
    nav.consume_results(state, results, **callbacks)
    np.testing.assert_allclose(state.world_goal, [10, 23])
    assert len(state.trajectory) == 21  # No second safety-buffer subtraction.
    assert nav.control(state, pose=Pose2D(10, 20.5, math.pi / 2), now=20.0,
                       config=load_navdp_planner_config()) == (0.0, 0.0, 0.0)
    state.mode = "stop"
    nav.sync_command(state, now=20.0)
    assert nav.world_path.size == 0


def test_navigation_times_out_and_rejects_repeated_invalid_plans(nav_config, monkeypatch):
    from gear_sonic.utils.inference.iplanner import control

    nav = IPlannerNavigation(nav_config)
    state = _NavDPRuntimeState(mode="nav_goal", generation=1, segment_id=1, world_goal=(5, 0))
    nav.sync_command(state, now=1.0)
    monkeypatch.setattr(control.time, "monotonic", lambda: 62.0)
    statuses = []
    nav.schedule(state, SimpleNamespace(pose=Pose2D(0, 0, 0)), config=load_navdp_planner_config(),
                 infer=lambda *a: pytest.fail("must not infer after timeout"),
                 send_status=lambda *a: statuses.append(a))
    assert statuses == [("failed", "iplanner_execution_timeout")]
    state.mode = "nav_goal"
    nav.sync_command(state, now=63.0)
    results = queue.Queue()
    for _ in range(3):
        results.put((1, 1, None, None, "invalid response", 0.1))
    nav.consume_results(state, results, send_metrics=lambda *a, **kw: None,
                        send_status=lambda *a: statuses.append(a), report_event=lambda *a, **kw: None)
    assert state.mode == "stop"
    assert statuses[-1] == ("failed", "three_invalid_trajectories")


@pytest.mark.parametrize("endpoint", [(0.6, 0.0), (0.1, 0.5), (0.1, -0.5)])
def test_short_or_deviated_model_path_cannot_replace_the_requested_goal(nav_config, endpoint):
    import time

    state = _NavDPRuntimeState(mode="nav_goal", generation=1, skill_id=1, segment_id=1, world_goal=(3.0, 0.0))
    nav = IPlannerNavigation(nav_config)
    nav.sync_command(state, now=time.monotonic())
    results = queue.Queue()
    results.put((1, 1, np.array([[0.0, 0.0], endpoint]), Pose2D(0, 0, 0), None, 0.02))
    statuses = []
    nav.consume_results(state, results, send_metrics=lambda *a, **kw: None,
                        send_status=lambda *a: statuses.append(a), report_event=lambda *a, **kw: None)
    assert state.world_goal == (3.0, 0.0)
    nav.schedule(state, SimpleNamespace(pose=Pose2D(0, 0, 0)), config=load_navdp_planner_config(),
                 infer=lambda *a: pytest.fail("server is not initialized"), send_status=lambda *a: statuses.append(a))
    assert state.mode == "nav_goal" and statuses == []
    assert state.current_local_goal == pytest.approx((3.0, 0.0))


@pytest.mark.parametrize("safe_distance", [0.0, 0.5])
def test_degenerate_trajectory_is_a_failure_and_never_arrival(nav_config, safe_distance):
    import time

    nav = IPlannerNavigation(replace(nav_config, safe_distance=safe_distance))
    state = _NavDPRuntimeState(mode="nav_goal", generation=1, segment_id=1, world_goal=(3.0, 0.0))
    nav.sync_command(state, now=time.monotonic())
    statuses = []
    for attempt in range(3):
        results = queue.Queue()
        results.put((1, 1, np.zeros((51, 2)), Pose2D(0, 0, 0), None, 0.02))
        nav.consume_results(state, results, send_metrics=lambda *a, **kw: None,
                            send_status=lambda *a: statuses.append(a), report_event=lambda *a, **kw: None)
        assert state.world_goal == (3.0, 0.0)
        assert not any(status == "reached" for status, reason in statuses)
        assert state.invalid_count == attempt + 1
    assert statuses == [("failed", "three_invalid_trajectories")]
    assert nav.control(state, pose=Pose2D(0, 0, 0), now=time.monotonic(),
                       config=load_navdp_planner_config()) == (0.0, 0.0, 0.0)


def test_arrival_checks_original_waypoint_at_two_meters(nav_config):
    import time

    assert nav_config.goal_tolerance == pytest.approx(2.0)
    nav = IPlannerNavigation(nav_config)
    state = _NavDPRuntimeState(mode="nav_goal", generation=1, segment_id=1, world_goal=(3.0, 0.0))
    nav.sync_command(state, now=time.monotonic())
    statuses = []
    for x in (0.0, 0.99):
        nav.schedule(state, SimpleNamespace(pose=Pose2D(x, 0, 0)), config=load_navdp_planner_config(),
                     infer=lambda *a: pytest.fail("server is not initialized"), send_status=lambda *a: statuses.append(a))
        assert state.mode == "nav_goal" and statuses == []
    nav.schedule(state, SimpleNamespace(pose=Pose2D(1.01, 0, 0)), config=load_navdp_planner_config(),
                 infer=lambda *a: pytest.fail("arrival does not infer"), send_status=lambda *a: statuses.append(a))
    assert statuses == [("reached", "iplanner_goal_tolerance")]


def test_replanning_uses_g1_interval_and_the_camera_pose(nav_config, monkeypatch):
    from gear_sonic.utils.inference.iplanner import control

    clock = [10.0]
    monkeypatch.setattr(control.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(control.threading, "Thread", lambda *, target, args, daemon:
                        SimpleNamespace(start=lambda: target(*args)))
    state = _NavDPRuntimeState(
        mode="nav_goal", generation=2, segment_id=4, world_goal=(5, 2), navdp_initialized=True,
        latest_rgb=np.zeros((2, 2, 3), np.uint8), latest_depth=np.ones((2, 2)),
        latest_camera_timestamp=123.0,
    )
    cycle = SimpleNamespace(pose=Pose2D(2, 1, 0), pose_history=[(123.0, Pose2D(1, 1, math.pi / 2))])
    nav = IPlannerNavigation(nav_config)
    nav.sync_command(state, now=10.0)
    calls = []

    def infer(*args):
        calls.append(args)
        state.inference_busy = False

    for tick in (10.0, 10.05, 10.11):
        clock[0] = tick
        nav.schedule(state, cycle, config=load_navdp_planner_config(), infer=infer,
                     send_status=lambda *a: pytest.fail("goal should still be active"))
    assert len(calls) == 2
    assert calls[0][-2] == pytest.approx((1.0, -4.0))
    assert calls[0][-1] == cycle.pose_history[0][1]
    assert calls[0][2] is not state.latest_rgb  # Worker owns a stable image copy.


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -math.pi / 2, math.pi - 0.01, -math.pi + 0.01])
def test_nearby_right_goal_steers_right_in_every_world_heading(nav_config, yaw):
    from gear_sonic.utils.inference.navdp.navigation import base_goal_to_world, local_trajectory_to_world

    # Exercise the former hardcoded 2 m steering cutoff with a smaller tolerance.
    nav_config = replace(nav_config, goal_tolerance=1.0)
    pose = Pose2D(10, -4, yaw)
    local_goal = (1.5, -0.4)
    state = _NavDPRuntimeState(mode="nav_goal", world_goal=base_goal_to_world(local_goal, pose), trajectory_time=10.0)
    nav = IPlannerNavigation(nav_config)
    nav.world_path = local_trajectory_to_world([[0, 0], [0.6, -0.16], local_goal], pose)
    velocity = nav.control(state, pose=pose, now=10.0, config=load_navdp_planner_config())
    assert nav_config.min_forward_speed <= velocity[0] <= nav_config.max_forward_speed
    assert velocity[2] < 0  # The original <2m branch returned zero in all these cases.
    mirrored = pure_pursuit([[0, 0], [0.6, 0.16], [1.5, 0.4]], math.hypot(*local_goal), nav_config)[0]
    assert velocity[2] == pytest.approx(-mirrored[2], abs=2e-6)
