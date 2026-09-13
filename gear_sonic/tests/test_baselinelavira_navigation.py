from __future__ import annotations

import importlib.util
from dataclasses import replace
import json
import math
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from baselinelavira import BASELINE_OVERLAY, build_profile
from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.utils.inference.iplanner.agent import IPlannerLaViRAClient
from gear_sonic.utils.inference.iplanner.control import (
    IPlannerNavigation,
    load_navigation_config,
    pure_pursuit,
    truncate_trajectory,
)
from gear_sonic.utils.inference.iplanner.geometry import build_waypoint_geometry
from gear_sonic.utils.inference.lavira.agent import LaViRAAgent, LaViRAAgentError, LaViRAClient
from gear_sonic.utils.inference.lavira.geometry import ObjectNavGeometryError
from gear_sonic.utils.inference.navdp.gateway import load_navdp_planner_config
from gear_sonic.utils.inference.navdp.navigation import Pose2D
from gear_sonic.utils.inference.navdp.runtime import _NavDPRuntimeState


@pytest.fixture
def nav_config():
    return load_navigation_config(load_runtime_profile(overlays=(BASELINE_OVERLAY,)))


def test_waypoint_uses_va_selected_area_bottom_and_single_paired_depth():
    depth = np.full((100, 100), 9000.0)
    depth[77:84, 47:54] = np.arange(49).reshape(7, 7) * 10 + 1000
    frames = [(np.full((100, 100), 5000), 100, 50)] * 5
    frames[2] = (depth, 100, 50)
    geometry = build_waypoint_geometry(
        {"action": "NAVIGATE", "bbox_2d": [400, 400, 600, 800]}, frames,
    )
    assert geometry["waypoint_pixel"] == [50, 80]
    assert geometry["goal_x"] == pytest.approx(1.144)
    assert geometry["goal_y"] == 0
    # The bbox center and other burst frames deliberately have different depth.
    assert geometry["goal_x"] != 9.0 and geometry["goal_x"] != 5.0


def test_missing_waypoint_depth_does_not_default_to_blind_forward_motion():
    frames = [(np.zeros((100, 100)), 100, 50)] * 5
    with pytest.raises(ObjectNavGeometryError, match="no valid aligned depth"):
        build_waypoint_geometry({"action": "NAVIGATE", "bbox_2d": [400, 400, 600, 800]}, frames)


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


@pytest.mark.parametrize("experiment_mode", [False, True])
def test_agent_factory_wires_autonomous_selection_without_changing_camera(tmp_path, monkeypatch, experiment_mode):
    import yaml
    from gear_sonic.experiments.agent import ExperimentAgent
    from gear_sonic.experiments.config import CONFIGS, resolve
    from gear_sonic.utils.inference.lavira import service

    path = tmp_path / "runtime.yaml"
    payload = resolve(CONFIGS / "baselinelavira.yaml")[0] if experiment_mode else build_profile()
    path.write_text(yaml.safe_dump(payload))
    config = service.load_lavira_config(str(path))
    camera = object()
    monkeypatch.setattr(service, "SensorGatewayRGBDCamera", lambda *a, **kw: camera)
    monkeypatch.setattr(IPlannerLaViRAClient, "__init__", lambda self, **kw: None)
    runtime = SimpleNamespace(**{name: (lambda *a, **kw: None) for name in (
        "submit_intent", "wait_status", "is_cancelled", "pop_failure", "report_agent_event", "report_todo",
    )})
    agent = service._agent(config, "inproc://unused", runtime)
    client = agent.client.client if experiment_mode else agent.client
    assert (agent.camera.camera if experiment_mode else agent.camera) is camera
    assert isinstance(client, IPlannerLaViRAClient)
    assert agent.navigation_waypoint.__self__ is client
    assert agent.navigation_geometry is build_waypoint_geometry
    expected_type = ExperimentAgent if experiment_mode else LaViRAAgent
    assert type(agent)._align is expected_type._align
    assert type(agent)._manipulate is expected_type._manipulate


def _navigation_client(monkeypatch, responses):
    client = object.__new__(IPlannerLaViRAClient)
    prompts_path = Path(build_profile()["components"]["iplanner"]["root"]).parent / "prompts.py"
    if not prompts_path.exists():
        pytest.skip("local LaViRA navigation prompts unavailable")
    spec = importlib.util.spec_from_file_location("test_lavira_prompts", prompts_path)
    client.navigation_prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client.navigation_prompts)
    client.navigation_strategy = ""
    client.navigation_stop = False
    client.la_client = client.va_client = object()
    client.la_model = client.va_model = "test"
    client.la_enable_thinking = client.va_enable_thinking = False
    client.la_timeout_seconds = client.va_timeout_seconds = 1.0
    calls = []

    def create(_client, *, response_parser, **kwargs):
        calls.append(kwargs)
        text = responses.pop(0)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
        return response_parser(completion)

    monkeypatch.setattr(client, "_create", create)
    return client, calls


def _current_views(image):
    return [SimpleNamespace(direction=direction, image_bgr=image) for direction in
            ("front", "front_right", "right", "left", "front_left")]


def test_navigation_uses_original_la_strategy_then_va_autonomous_waypoint(monkeypatch):
    client, calls = _navigation_client(monkeypatch, [
        "- [ ] Pass through the opening\n- [ ] Reach the desk",
        json.dumps(dict(progress_analysis="Opening ahead", updated_todo_list="- [ ] Pass through the opening",
                        reasoning="Use the open passage to continue along the corridor", turn_direction="front_left",
                        chosen_view_index=5, stop=False, expected_landmark="doorway", global_target="desk")),
        json.dumps(dict(action="NAVIGATE", target="open corridor area", bbox_2d=[200, 300, 700, 900],
                        confidence=0.9, visual_check="Open floor continuing the route", stop_reasoning="")),
    ])
    image = np.zeros((100, 100, 3), np.uint8)
    decision = client.language_action(
        mission="Follow the corridor and reach the desk", navigation_mode="vln", global_target="desk",
        current_step=1, todo_list="", scan_views=_current_views(image),
        move_to_views=[], transition_result=None, manipulation_prompt="Pick up the bottle",
    )
    selected = client.navigation_waypoint(
        mission="Follow the corridor and reach the desk", global_target="desk",
        strategic_goal="Requested target: 'doorway'", image_bgr=image,
    )
    assert decision["skill_args"]["view_direction"] == "front_left"
    assert selected["target_description"] == "open corridor area"
    assert [call["request_kind"] for call in calls] == [
        "la_navigation_todo", "la_navigation", "va_navigation_waypoint",
    ]
    text = json.dumps(calls[-1]["messages"])
    assert "best navigation target" in text and "CURRENT STRATEGY" in text
    assert "Requested target" not in text
    assert "Use the open passage" in text
    monkeypatch.setattr(LaViRAClient, "language_action", lambda self, **kw: {"skill": "ALIGN"})
    assert client.language_action(transition_result={"transition": "READY_TO_ALIGN"}) == {"skill": "ALIGN"}


def test_navigation_prompt_examples_are_valid_complete_json(monkeypatch):
    client, _ = _navigation_client(monkeypatch, [])

    def follow_example(_client, *, response_parser, **kwargs):
        text = kwargs["messages"][-1]["content"][-1]["text"]
        marker = "**JSON FORMAT**:" if kwargs["request_kind"].startswith("va_") else "**JSON RESPONSE FORMAT**:"
        schema = text.split(marker, 1)[1]
        example, _ = json.JSONDecoder().raw_decode(schema[schema.index("{"):])
        if kwargs["request_kind"].startswith("va_"):
            assert isinstance(example["confidence"], (int, float))
            assert "best navigation target" in text  # The upstream targeting rules are preserved.
        else:
            assert example["global_target"]
            assert example["turn_direction"] == "front"
        return response_parser(SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(example)),
        )]))

    monkeypatch.setattr(client, "_create", follow_example)
    image = np.zeros((10, 10, 3), np.uint8)
    decision = client.language_action(
        mission="Find the desk", navigation_mode="vln", global_target=None,
        current_step=1, todo_list="- [ ] Reach the desk", scan_views=_current_views(image), move_to_views=[],
        manipulation_prompt="Pick up the bottle",
    )
    assert decision["global_target"] != "Pick up the bottle"
    assert client.navigation_waypoint(mission="Find the desk", global_target="desk",
                                      strategic_goal="Go forward", image_bgr=image)["confidence"] == 0.85


def test_missing_confidence_retries_with_precise_feedback_and_saves_raw_responses(monkeypatch, tmp_path):
    client, _ = _navigation_client(monkeypatch, [])
    client.request_context_dir = tmp_path
    client._request_context_lock = threading.Lock()
    client._request_context_sequence = 0
    attempts = []

    def create(**request):
        attempts.append(request)
        response = dict(action="NAVIGATE", target="desk", bbox_2d=[100, 200, 700, 900],
                        visual_check="Desk ahead", stop_reasoning="")
        if len(attempts) > 1:
            assert "confidence is missing" in request["messages"][0]["content"]
            assert "JSON number in [0,1]" in request["messages"][0]["content"]
            response["confidence"] = 0.82
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))])

    client.va_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(client, "_create", LaViRAClient._create.__get__(client))
    result = client.navigation_waypoint(mission="Find the desk", global_target="desk",
                                        strategic_goal="Go forward", image_bgr=np.zeros((10, 10, 3), np.uint8))
    assert len(attempts) == 2
    assert result["confidence"] == 0.82
    responses = [json.loads(p.read_text()) for p in sorted((tmp_path / "responses").glob("*.json"))]
    assert len(responses) == 2
    assert "confidence" not in json.loads(responses[0]["content"])
    assert "confidence is missing" in responses[0]["validation_error"]
    assert json.loads(responses[1]["content"])["confidence"] == 0.82
    assert responses[1]["validation_error"] is None


@pytest.mark.parametrize("confidence", [None, "0.9", 90, True, -0.1, float("nan"), float("inf")])
def test_invalid_waypoint_confidence_is_never_fabricated_or_allowed(monkeypatch, confidence):
    client, _ = _navigation_client(monkeypatch, [json.dumps(dict(
        action="NAVIGATE", target="desk", bbox_2d=[100, 200, 700, 900], confidence=confidence,
    ))])
    with pytest.raises(LaViRAAgentError, match="confidence.*JSON number"):
        client.navigation_waypoint(mission="Find the desk", global_target="desk",
                                   strategic_goal="Go forward", image_bgr=np.zeros((10, 10, 3), np.uint8))


def test_initial_global_target_cannot_copy_the_manipulation_instruction(monkeypatch):
    manipulation = "Pick up the bottle and place it in the blue basket."
    client, _ = _navigation_client(monkeypatch, [json.dumps(dict(
        progress_analysis="Desk visible", updated_todo_list="- [ ] Approach the desk",
        reasoning="Desk ahead", turn_direction="front", stop=False, expected_landmark="desk",
        global_target=manipulation.rstrip("."),
    ))])
    with pytest.raises(LaViRAAgentError, match="object/area name"):
        client.language_action(mission="Find the desk", global_target=None, current_step=1,
                               todo_list="- [ ] Approach the desk",
                               scan_views=_current_views(np.zeros((10, 10, 3), np.uint8)), move_to_views=[],
                               manipulation_prompt=manipulation)


def test_move_to_submits_va_selected_area_without_object_grounding(monkeypatch):
    agent = object.__new__(LaViRAAgent)
    agent.mission, agent.global_target, agent._segment_id = "follow the corridor", "desk", 1
    agent.min_confidence = 0.5
    snapshot = SimpleNamespace(rgb_bgr=np.zeros((100, 100, 3), np.uint8),
                               depth_mm=np.full((100, 100), 2500), fx=100, cx=50)
    agent.navigation_waypoint = lambda **kw: {
        "status": "FOUND", "target_description": "open corridor area", "confidence": 0.9,
        "bbox_2d": [400, 400, 600, 800], "navigation_action": "NAVIGATE",
    }
    agent.navigation_geometry = build_waypoint_geometry
    agent._capture_va_burst = lambda *a: [snapshot] * 5
    agent._face_scan_direction = lambda *a: None
    agent._event = lambda *a, **kw: None
    agent._next_segment = lambda: 2
    agent._ground = lambda **kw: pytest.fail("baseline must not ground a predetermined object before planning")
    intents = []
    agent.submit_intent = lambda *a: intents.append(a)

    def wait(*args):
        raise InterruptedError("end test after goal submission")

    agent._wait = wait
    with pytest.raises(InterruptedError):
        agent._move_to(1, 1, {"view_direction": "front", "target": "doorway"}, "follow the corridor")
    assert intents[0][0] == "navigation_goal"
    assert intents[0][1]["target"] == "open corridor area"
    assert intents[0][1]["goal_base"] == pytest.approx([2.5, 0.0])


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


def test_la_cannot_execute_ninety_degree_left_for_chosen_forty_five_degree_view(monkeypatch):
    client, _ = _navigation_client(monkeypatch, [json.dumps(dict(
        progress_analysis="The table is in the 45-degree left view, Image 5",
        updated_todo_list="- [ ] Approach table", reasoning="Turn left to face Image 5",
        chosen_view_index=5, turn_direction="left", stop=False, expected_landmark="table", global_target="table",
    ))])
    with pytest.raises(LaViRAAgentError, match="CURRENT_VIEW_5 requires front_left"):
        client.language_action(mission="Approach table", global_target="table", current_step=2,
                               todo_list="- [ ] Approach table", scan_views=_current_views(np.zeros((10, 10, 3), np.uint8)),
                               move_to_views=[])


def test_history_is_labeled_as_old_outcome_and_does_not_renumber_current_views(monkeypatch):
    client, calls = _navigation_client(monkeypatch, [json.dumps(dict(
        progress_analysis="Table now to right", updated_todo_list="- [ ] Approach table", reasoning="Use Image 2",
        chosen_view_index=2, turn_direction="front_right", stop=False, expected_landmark="table", global_target="table",
    ))])
    image = np.zeros((10, 10, 3), np.uint8)
    history = SimpleNamespace(skill_id=1, target="door", image_bgr=image, controller_state="reached",
                              va_result="SATISFIED", evidence="Doorway crossed")
    decision = client.language_action(mission="Approach table", global_target="table", current_step=2,
                                     todo_list="- [ ] Approach table", scan_views=_current_views(image), move_to_views=[history])
    text = json.dumps(calls[0]["messages"])
    assert "PAST_MOVE_TO_1" in text and "History Plan" not in text
    assert "CURRENT_VIEW_2: turn_direction=front_right" in text
    assert decision["skill_args"]["view_direction"] == "front_right"


def test_va_can_request_reobservation_without_inventing_an_unrelated_target(monkeypatch):
    client, calls = _navigation_client(monkeypatch, [json.dumps(dict(
        visual_check="Only a computer desk is visible; no basket table or relevant passage",
        action="REOBSERVE", bbox_2d=None, target="table with blue basket", confidence=0.9, stop_reasoning="",
    ))])
    client.navigation_strategy = "Go left. Find the basket table."
    result = client.navigation_waypoint(mission="Approach basket table", global_target="table with blue basket",
                                        strategic_goal="Find table", image_bgr=np.zeros((10, 10, 3), np.uint8))
    assert result["status"] == "NOT_FOUND" and result["bbox_2d"] is None
    assert result["navigation_action"] == "REOBSERVE"
    text = json.dumps(calls[0]["messages"])
    assert "ALREADY been executed" in text and "PRE-TURN panorama" in text
    assert "Do not substitute an unrelated table" in text
