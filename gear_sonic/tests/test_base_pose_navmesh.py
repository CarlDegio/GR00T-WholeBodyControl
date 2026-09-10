"""Behavioral recovery tests with continuous obstacle geometry and real servo."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.inference.base_pose.ovmm.actions import ActionLimits
from gear_sonic.utils.inference.base_pose.ovmm.config import BasePoseConfig
from gear_sonic.utils.inference.base_pose.ovmm.controller import BasePoseSession
from gear_sonic.utils.inference.base_pose.ovmm.navmesh import (
    RecoveryExecution,
    RecoveryPlanner,
)
from gear_sonic.tests.test_base_pose_ovmm import observation, targets
from gear_sonic.utils.inference.base_pose.ovmm.targets import SemanticTargetProvider


class AnalyticOracle:
    def __init__(self, obstacle):
        self.obstacle = obstacle

    def endpoint(self, point, clearance):
        ok = not self.obstacle(*point, clearance)
        return {"reachable": ok, "reason": "free" if ok else "obstacle"}

    def segment(self, start, end, clearance):
        return all(
            self.endpoint(start + t * (end - start), clearance)["reachable"]
            for t in np.linspace(0, 1, 301)[1:]
        )


def validate_executable(plan, oracle):
    start = np.zeros(2)
    for end in plan.waypoints:
        command = np.asarray(end - start, dtype=np.float32)
        assert 0.1 <= np.linalg.norm(command) <= 0.25
        assert oracle.segment(start, end, 0.005)
        start = end


def test_reachable_endpoint_switches_axis_before_completing_blocked_axis():
    oracle = AnalyticOracle(
        lambda x, y, m: -0.03 - m < x < 0.08 + m and -0.12 - m < y < -0.04 + m
    )
    assert not oracle.segment(np.zeros(2), np.array([0.0, -0.15]), 0.005)
    plan = RecoveryPlanner(BasePoseConfig(), ActionLimits()).plan(
        oracle, 0.80, 0.15, blocked_axis=1
    )
    assert plan.diagnostics["branch"] == "nominal_reachable"
    assert plan.diagnostics["strategy"] == "switch_axis_order"
    assert plan.effective_distance_m == 0.65
    np.testing.assert_allclose(plan.waypoints[0], [0.15, 0])
    np.testing.assert_allclose(plan.waypoints[-1], [0.15, -0.15])
    validate_executable(plan, oracle)


def test_unreachable_nominal_distance_selects_largest_safe_grid_distance():
    oracle = AnalyticOracle(lambda x, y, m: x < 0.121 + m - 1e-10 and y < -0.04 + m)
    planner = RecoveryPlanner(BasePoseConfig(), ActionLimits())
    plan = planner.plan(oracle, 0.703, 0.093, blocked_axis=1)
    assert plan.diagnostics["branch"] == "nominal_unreachable"
    assert plan.effective_distance_m == pytest.approx(0.577)
    assert plan.effective_distance_m < 0.65
    np.testing.assert_allclose(plan.waypoints[-1], [0.126, -0.093])
    # Every larger sampled standoff has been rejected; safety margin moves
    # the theoretical boundary from .582 to .577 without changing lateral.
    trials = plan.diagnostics["tested_distances"]
    assert len(trials) == 73
    assert all(not p["reachable"] for p in trials[:-1])
    assert not oracle.endpoint(np.array([0.125, -0.093]), 0.005)["reachable"]
    validate_executable(plan, oracle)


def test_short_corrections_use_legal_bridge_instead_of_sub_deadzone_axes():
    oracle = AnalyticOracle(lambda x, y, m: False)
    plan = RecoveryPlanner(BasePoseConfig(), ActionLimits()).plan(
        oracle, 0.703, 0.093, 1
    )
    assert plan.diagnostics["strategy"] == "switch_axis_bridge"
    assert len(plan.waypoints) == 2
    np.testing.assert_allclose(plan.waypoints[-1], [0.053, -0.093])
    validate_executable(plan, oracle)


def test_boundary_keeps_original_lateral_tolerance_and_maximizes_distance():
    # Zero lateral error is impossible anywhere closer than nominal, but the
    # original 6 cm tolerance permits a safe endpoint at a 4.6 cm residual.
    oracle = AnalyticOracle(lambda x, y, m: x < 0.5 + m and y < -0.052 + m - 1e-10)
    planner = RecoveryPlanner(BasePoseConfig(), ActionLimits())
    plan = planner.plan(oracle, 0.703, 0.093, 1)
    assert plan.diagnostics["branch"] == "nominal_unreachable"
    assert plan.effective_distance_m == pytest.approx(0.646)
    assert plan.effective_distance_m + 0.003 < 0.65
    assert plan.diagnostics["planned_right_residual_m"] == pytest.approx(0.046)
    assert abs(plan.diagnostics["planned_right_residual_m"]) < 0.06
    assert len(plan.diagnostics["tested_distances"]) == 4
    validate_executable(plan, oracle)


def test_reachable_but_unplannable_nominal_does_not_claim_it_is_unreachable():
    oracle = AnalyticOracle(lambda x, y, m: False)
    oracle.segment = lambda start, end, clearance: False
    plan = RecoveryPlanner(
        BasePoseConfig(navmesh_local_radius_m=0.2, navmesh_search_resolution_m=0.05),
        ActionLimits(),
    ).plan(oracle, 0.8, 0.15, 1)
    assert not plan.waypoints
    assert plan.diagnostics["branch"] == "nominal_reachable"
    trials = plan.diagnostics["tested_distances"]
    original = [p for p in trials if p["search_phase"] == "original"]
    expanded = [p for p in trials if p["search_phase"] == "expanded"]
    assert trials == original + expanded
    assert original[0]["distance_m"] == pytest.approx(0.65)
    assert original[-1]["distance_m"] == pytest.approx(0.30)
    assert expanded[-1]["distance_m"] == pytest.approx(0.90)
    assert all(p["reachable"] for p in trials)
    assert plan.diagnostics["expanded_search"]


def test_actual_no_progress_requires_environment_rejection_and_verifies_each_leg():
    recovery = RecoveryExecution(BasePoseConfig(), ActionLimits())
    provider = SimpleNamespace(navmesh_violated=lambda: False)
    recovery.provider = provider
    obs = SimpleNamespace(gps=np.zeros(2), compass=np.zeros(1))
    recovery.record(np.array([0, -0.1, 0]), obs, 10)
    recovery.observe(obs, 11)
    assert recovery.blocked_axis is None
    provider.navmesh_violated = lambda: True
    recovery.record(np.array([0, -0.1, 0]), obs, 11)
    recovery.observe(obs, 12)
    assert recovery.blocked_axis == 1
    recovery.path = [np.array([0.15, 0])]
    provider.navmesh_violated = lambda: False
    recovery.record(np.array([0.15, 0, 0]), obs, 12)
    recovery.observe(obs, 13)  # no actual movement despite optimistic preflight
    assert recovery.failure and not recovery.completed


def scene_observation(position):
    obs = observation(distance=0.703 - position[0], gps=position)
    # Keep a fixed RGB-D target in episode-start coordinates as the robot moves.
    initial_right = targets(SemanticTargetProvider(), obs).target_geometry.right_m
    obs.camera_pose[1, 3] -= 0.093 + position[1] - initial_right
    return obs


@pytest.mark.parametrize("boundary", [False, True])
def test_recovery_returns_to_fresh_stability_and_poststop_without_budget_reset(
    boundary,
):
    config = BasePoseConfig(navmesh_recovery_enabled=True)
    session = BasePoseSession(config, ActionLimits(), use_opencv_camera_pose=True)
    position = np.zeros(2)

    def snapshot():
        anchor = position.copy()
        return AnalyticOracle(
            lambda x, y, m: boundary
            and x + anchor[0] < 0.5 + m
            and y + anchor[1] < -0.052 + m - 1e-10
        )

    provider = SimpleNamespace(navmesh_violated=lambda: True, snapshot=snapshot)
    session.bind_navmesh(provider)
    session.reset(start_frame_id=0)
    session.recovery.record(
        np.array([0, -0.1000001, 0]), scene_observation(position), 19
    )
    old_budget = session.start_frame_id
    recovery_frames = []
    poststop_frames = []
    stable_counts = []
    for frame in range(20, 120):
        obs = scene_observation(position)
        result = session.step(obs, frame_id=frame)
        stable_counts.append(session.controller.stable_frames)
        if result.diagnostics["navmesh_recovery"]["active"]:
            recovery_frames.append(frame)
            assert session.controller.stable_frames == 0
            assert session.controller.post_stop_valid_sample_count == 0
        if result.diagnostics["phase"] == "post_stop_sampling":
            poststop_frames.append(frame)
        duplicate = session.step(obs, frame_id=frame)
        np.testing.assert_array_equal(duplicate.xyt, 0)
        assert (
            duplicate.diagnostics["controller_updates"]
            == result.diagnostics["controller_updates"]
        )
        position += result.xyt[:2]
        provider.navmesh_violated = lambda: False
        if result.terminate:
            break
    assert result.status.value == "READY"
    assert len(recovery_frames) == 2
    assert len(poststop_frames) >= 30
    assert max(stable_counts) >= 4  # Entering post-stop resets the core counter.
    assert min(poststop_frames) - max(recovery_frames) >= 5
    assert session.controller.post_stop_valid_sample_count == 30
    assert session.start_frame_id == old_budget
    assert result.diagnostics["steps"] == frame
    assert min(poststop_frames) > max(recovery_frames)
    assert abs(result.diagnostics["raw_right_m"]) < 0.06
    expected_distance = 0.646 if boundary else 0.65
    assert result.diagnostics["effective_target_distance_m"] == pytest.approx(
        expected_distance
    )
    assert session.controller.last_errors[0] == pytest.approx(
        result.diagnostics["raw_forward_m"] - expected_distance, abs=1e-5
    )


def test_recovery_never_renews_total_environment_step_budget():
    session = BasePoseSession(
        BasePoseConfig(navmesh_recovery_enabled=True, max_steps=10),
        ActionLimits(),
        use_opencv_camera_pose=True,
    )
    session.bind_navmesh(SimpleNamespace(navmesh_violated=lambda: False))
    session.reset(start_frame_id=100)
    session.recovery.path = [np.array([0.2, 0])]
    result = session.step(observation(), frame_id=110)
    assert result.status.value == "TIMEOUT"
    np.testing.assert_array_equal(result.xyt, 0)


def test_lateral_expansion_waits_until_all_strict_distance_routes_fail():
    config = BasePoseConfig(navmesh_max_lateral_tolerance_m=0.15)
    planner = RecoveryPlanner(config, ActionLimits())
    free = planner.plan(AnalyticOracle(lambda x, y, margin: False), 1.05, 0, 0)
    assert free.waypoints and not free.diagnostics['lateral_expanded_search']
    assert free.effective_lateral_tolerance_m == .06
    oracle = AnalyticOracle(lambda x, y, margin: x > .02-margin and y > -.13+margin)
    strict = RecoveryPlanner(BasePoseConfig(), ActionLimits()).plan(oracle, 1.05, 0, 0)
    assert not strict.waypoints
    expanded = planner.plan(oracle, 1.05, 0, 0)
    assert expanded.waypoints and expanded.diagnostics['lateral_expanded_search']
    assert expanded.diagnostics['strict_lateral_search']['original_and_outer_distances_exhausted']
    residual = abs(expanded.diagnostics['planned_right_residual_m'])
    assert .06 < residual <= .147
    assert expanded.effective_lateral_tolerance_m == pytest.approx(residual+.003)
    validate_executable(expanded, oracle)


def test_relaxed_endpoint_retains_its_tolerance_through_fresh_confirmation():
    config = BasePoseConfig(navmesh_recovery_enabled=True, navmesh_max_lateral_tolerance_m=.15)
    session = BasePoseSession(config, ActionLimits(), use_opencv_camera_pose=True)
    position = np.zeros(2)

    def obs():
        value = observation(distance=1.05-position[0], gps=position)
        initial_right = targets(SemanticTargetProvider(), value).target_geometry.right_m
        value.camera_pose[1, 3] -= position[1]-initial_right
        return value

    def snapshot():
        anchor = position.copy()
        return AnalyticOracle(lambda x, y, margin:
                              x+anchor[0] > .02-margin and y+anchor[1] > -.13+margin)

    provider = SimpleNamespace(navmesh_violated=lambda: True, snapshot=snapshot)
    session.bind_navmesh(provider)
    session.reset(start_frame_id=0)
    session.recovery.record(np.array([.100001,0,0]),obs(),19)
    for frame in range(20,160):
        result = session.step(obs(),frame_id=frame)
        position += result.xyt[:2]
        provider.navmesh_violated = lambda: False
        if result.terminate:
            break
    assert result.status.value == 'READY'
    assert session.controller.post_stop_valid_sample_count == 30
    assert .06 < abs(result.diagnostics['raw_right_m']) <= session.controller.lateral_tolerance_m
    assert session.start_frame_id == 0
    tolerance = session.controller.lateral_tolerance_m
    session.restart_after_acquisition(frame+1)
    assert session.controller.lateral_tolerance_m == tolerance and session.start_frame_id == 0
    session.reset(start_frame_id=0)
    assert session.controller.lateral_tolerance_m == config.lateral_tolerance_m
