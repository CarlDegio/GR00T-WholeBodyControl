"""Free action ordering in one original LA loop, using real gateway/VLA state."""

import json
import math
from types import SimpleNamespace

import pytest

from gear_sonic.tests.test_lavira_agent import (
    decision, grounding, alignment_grounding, ready_to_manipulate, task_complete, postcheck,
)
from gear_sonic.tests.test_multi_skill_mission import mission_harness
from gear_sonic.utils.inference.lavira.agent import LaViRAClient, LaViRAAgentError
from gear_sonic.utils.inference.lavira.flexible import FlexibleMissionAgent, validate_flexible_action
from gear_sonic.utils.inference.lavira.mission import parse_skill_definitions


def flexible_harness(monkeypatch, actions):
    h = mission_harness(monkeypatch)
    h.agent.__class__ = FlexibleMissionAgent
    h.agent.mission = "先抓起袋子，拿着袋子走到垃圾桶，踩开桶盖，最后回到门口。"
    h.agent.skills = parse_skill_definitions({
        "bag": dict(backend="pill", vla_prompt="grasp and hold bag", vla_trained_prompt="collect the pill bottle",
                    completion={"kind": "visual"}, exit="hold", completion_settle_seconds=0,
                    timeout_seconds=40, on_timeout="continue"),
        "bin": dict(backend="rubbish", vla_prompt="step on pedal", vla_trained_prompt="trained pedal prompt ",
                    completion={"kind": "visual"}, exit="hold", completion_settle_seconds=0),
    }, flexible=True)
    h.agent.monotonic = lambda: h.now[0]
    h.agent.sleep = lambda seconds: h.now.__setitem__(0, h.now[0] + seconds)
    h.agent.manipulation_window_seconds = 5
    capture = h.camera.capture_rgb
    h.camera.capture_rgb = lambda *, camera_stream="chest_view", timeout_ms=None: capture(camera_stream=camera_stream)
    h.client.grounding = lambda **kwargs: grounding(description=kwargs["target"])
    h.client.alignment_groundings = iter([alignment_grounding()] * 10)
    h.client.postchecks = iter([ready_to_manipulate()] * 20)
    original_postcheck = h.client.postcheck
    h.first_verdict = task_complete

    def check(**kwargs):
        if kwargs.get("skill") == "MANIPULATE":
            h.client.postcheck_calls.append(kwargs)
            return h.first_verdict() if kwargs["mission"] == "grasp and hold bag" else task_complete()
        return original_postcheck(**kwargs)

    h.client.postcheck = check
    pending = iter(actions)
    initial_todo = "\n".join(f"- [ ] {index}: {skill} {key}" for index, (skill, key) in enumerate(actions) if skill != "COMPLETE")

    def language_action(**kwargs):
        h.client.la_calls.append(kwargs)
        skill, key = next(pending)
        todo = kwargs["todo_list"] or initial_todo
        if skill == "MANIPULATE":
            lines = todo.splitlines()
            for index, line in enumerate(lines):
                if "[ ]" in line:
                    lines[index] = line.replace("[ ]", "[x]", 1) + " Result: ALIGN ready."
                    break
            todo = "\n".join(lines)
        result = decision(None if skill == "COMPLETE" else skill, global_target=key or "basket")
        result.update(updated_todo_list=todo, expected_postcondition="The current operation is physically finished.")
        if skill == "COMPLETE":
            result.update(decision="COMPLETE", skill=None, skill_args={})
        elif skill == "MOVE_TO":
            result["skill_args"] = {"view_direction": "front", "target": key}
        elif skill == "ALIGN":
            result["skill_args"] = {"skill_key": key, "task": "pick the bag from the table" if key == "bag" else "press the trash can pedal"}
        else:
            result["skill_args"] = {"skill_key": key}
        return result

    h.client.language_action = language_action
    h.actual_prompts, h.start_times, h.stops = [], [], []
    start = h.agent._start_manipulation

    def start_and_record(*args):
        start(*args)
        h.actual_prompts.append(h.handler.language_prompt_ref[0])
        h.start_times.append(h.now[0])

    h.agent._start_manipulation = start_and_record
    submit = h.agent.submit_intent

    def record_stop(name, params):
        if name == "stop_vla_task":
            h.stops.append((h.now[0], dict(params)))
        return submit(name, params)

    h.agent.submit_intent = record_stop
    return h


@pytest.mark.parametrize("actions,backends", [
    ([("MOVE_TO", "table"), ("ALIGN", "bag"), ("MANIPULATE", "bag"),
      ("MOVE_TO", "trash can"), ("ALIGN", "bin"), ("MANIPULATE", "bin"), ("COMPLETE", "trash can")], ["pill", "rubbish"]),
    ([("ALIGN", "bag"), ("MANIPULATE", "bag"), ("ALIGN", "bin"), ("MANIPULATE", "bin"),
      ("MOVE_TO", "door"), ("COMPLETE", "door")], ["pill", "rubbish"]),
    ([("ALIGN", "bag"), ("MANIPULATE", "bag"), ("ALIGN", "bag"), ("MANIPULATE", "bag"), ("COMPLETE", "bag")], ["pill", "pill"]),
    ([("MOVE_TO", "hallway"), ("MOVE_TO", "door"), ("COMPLETE", "door")], []),
])
def test_one_la_loop_follows_model_order_without_preconfigured_steps(monkeypatch, actions, backends):
    h = flexible_harness(monkeypatch, actions)
    result = h.agent.run(1)
    assert result.reason == "mission_completed", result
    assert h.selected == backends
    assert len(h.client.la_calls) == len(actions)  # No extra task planner or hidden LA pass.
    assert all(call["mission"] == h.agent.mission for call in h.client.la_calls)
    assert [call["current_step"] for call in h.client.la_calls] == list(range(1, len(actions) + 1))
    assert all("vla_trained_prompt" not in json.dumps(call["available_skills"]) for call in h.client.la_calls)
    assert all(call["available_skills"] == {
        "bag": {"description": "grasp and hold bag"},
        "bin": {"description": "step on pedal"},
    } for call in h.client.la_calls)
    va_prompts = {"bag": "grasp and hold bag", "bin": "step on pedal"}
    assert [call["manipulation_prompt"] for call in h.client.alignment_grounding_calls] == [
        va_prompts[key] for skill, key in actions if skill == "ALIGN"
    ]
    names = [name for name, _ in h.intents if name in {"navigation_goal", "start_base_pose", "start_vla_task"}]
    mapping = {"MOVE_TO": "navigation_goal", "ALIGN": "start_base_pose", "MANIPULATE": "start_vla_task"}
    assert names == [mapping[skill] for skill, _ in actions if skill != "COMPLETE"]
    prompts = {"pill": "collect the pill bottle", "rubbish": "trained pedal prompt "}
    assert h.actual_prompts == [prompts[backend] for backend in backends]
    assert len([event for event in h.events if event["code"] == "TASK_COMPLETED"]) == 1
    assert not h.terminal


def test_flexible_timeout_returns_to_same_la_loop_and_does_not_claim_success(monkeypatch):
    h = flexible_harness(monkeypatch, [("ALIGN", "bag"), ("MANIPULATE", "bag"),
                                       ("MOVE_TO", "door"), ("COMPLETE", "door")])
    h.first_verdict = lambda: postcheck("NOT_SATISFIED")
    result = h.agent.run(1)
    assert result.reason == "mission_completed_with_timeouts", result
    assert h.stops[0][0] - h.start_times[0] == pytest.approx(40)
    assert h.stops[0][1]["reason"] == "timeout_continue"
    after = h.client.la_calls[2]
    assert after["transition_result"]["timeout_continued"]
    assert "without success confirmation" in after["todo_list"]


def test_next_waypoint_is_selected_from_panorama_at_arrival_heading(monkeypatch):
    h = flexible_harness(monkeypatch, [
        ("MOVE_TO", "black cabinet"), ("MOVE_TO", "white rubbish bin"),
        ("ALIGN", "bag"), ("MANIPULATE", "bag"), ("COMPLETE", "bag"),
    ])
    original_la = h.client.language_action
    original_capture = h.camera.capture_camera_aligned_rgbd
    arrivals = []

    def select_visible_direction(**kwargs):
        result = original_la(**kwargs)
        if result["skill"] == "MOVE_TO":
            result["skill_args"]["view_direction"] = "front_left"
        return result

    def observe_arrival(**kwargs):
        # NAV's head image is captured after navigation and target-facing finish.
        arrivals.append(h.camera.sonic_yaw)
        return original_capture(**kwargs)

    h.client.language_action = select_visible_direction
    h.camera.capture_camera_aligned_rgbd = observe_arrival
    result = h.agent.run(1)
    assert result.reason == "mission_completed", result
    views = h.client.la_calls[1]["scan_views"]
    assert [v.direction for v in views] == [
        "front", "front_right", "right", "left", "front_left",
    ]
    assert arrivals[0] != pytest.approx(0.0)
    assert views[0].reference_pose[2] == pytest.approx(arrivals[0])
    assert [math.remainder(v.absolute_yaw_rad - arrivals[0], 2 * math.pi)
            for v in views] == pytest.approx([0, -math.pi / 4, -math.pi / 2,
                                            math.pi / 2, math.pi / 4])
    # Successful alignment still hands off to manipulation without a scan.
    assert len(h.client.la_calls[3]["scan_views"]) == 1


def test_unobserved_move_direction_is_rejected_before_robot_turns(monkeypatch):
    h = flexible_harness(monkeypatch, [("ALIGN", "bag"), ("MOVE_TO", "door")])
    original_la = h.client.language_action
    intents_at_decision = []

    def guess_right(**kwargs):
        result = original_la(**kwargs)
        if result["skill"] == "MOVE_TO":
            assert [v.direction for v in kwargs["scan_views"]] == ["front"]
            intents_at_decision.extend(h.intents)
            result["skill_args"]["view_direction"] = "right"
        return result

    h.client.language_action = guess_right
    result = h.agent.run(1)
    assert result.state == "failed"
    assert "MOVE_TO direction has no current observation" in result.reason
    assert h.intents == intents_at_decision


@pytest.mark.parametrize("actions", [
    [("MANIPULATE", "bag")],
    [("ALIGN", "bag"), ("MANIPULATE", "bin")],
    [("ALIGN", "bag"), ("MANIPULATE", "bag"), ("MANIPULATE", "bag")],
])
def test_every_vla_call_requires_alignment_for_its_selected_backend(monkeypatch, actions):
    h = flexible_harness(monkeypatch, actions)
    result = h.agent.run(1)
    assert result.state == "failed"
    assert len(h.selected) <= int(len(actions) == 3)


def test_va_alignment_out_of_range_returns_control_to_la_for_navigation(monkeypatch):
    h = flexible_harness(monkeypatch, [("ALIGN", "bag"), ("MOVE_TO", "table"),
                                       ("ALIGN", "bag"), ("MANIPULATE", "bag"), ("COMPLETE", "bag")])
    original = h.agent._single_view_nav_handoff
    checked = []

    def out_of_range_once(**kwargs):
        if not checked:
            checked.append(True)
            return {"status": "NOT_SATISFIED", "visual_evidence": "target is 5 m away"}, {}
        return original(**kwargs)

    h.agent._single_view_nav_handoff = out_of_range_once
    # The LA may replace a failed, unfinished ALIGN with a navigation recovery item.
    response = h.client.language_action

    def recover(**kwargs):
        result = response(**kwargs)
        if len(h.client.la_calls) == 2:
            result["updated_todo_list"] = "\n".join(result["updated_todo_list"].splitlines()[1:])
        return result

    h.client.language_action = recover
    result = h.agent.run(1)
    assert result.reason == "mission_completed", result
    assert h.client.la_calls[1]["transition_result"]["transition"] == "RETURN_TO_NAVIGATION"
    assert len([name for name, _ in h.intents if name == "start_base_pose"]) == 1


@pytest.mark.parametrize("args", [
    {"skill_key": "missing", "task": "grasp bag"},
    {"skill_key": "bag", "task": "grasp bag", "trained_prompt": "arbitrary"},
    {"skill_key": "bag", "task": ""},
])
def test_la_cannot_select_unknown_models_or_override_training_prompts(args):
    value = decision("ALIGN")
    value["skill_args"] = args
    with pytest.raises(LaViRAAgentError):
        validate_flexible_action(value, available_skills={"bag": {}})


def test_same_la_client_retries_new_action_schema_without_extra_planning_call(tmp_path):
    requests = []
    valid = decision("ALIGN")
    valid["updated_todo_list"] = "- [ ] Align for bag\n- [ ] Grasp bag"
    valid["skill_args"] = {"skill_key": "bag", "task": "grasp the bag"}
    invalid = {**valid, "skill_args": {"skill_key": "unknown", "task": "grasp the bag"}}
    replies = iter([invalid, valid])

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(replies))))])

    cloud = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = LaViRAClient(la_client=cloud, va_client=cloud, la_base_url="https://example.invalid",
                         va_base_url="https://example.invalid", request_context_dir=tmp_path)
    result = client.language_action(mission="whole mission", navigation_mode="vln", global_target=None,
                                    current_step=1, todo_list="", scan_views=(), move_to_views=(),
                                    available_skills={"bag": {"description": "grasp bag"}})
    assert result["skill_args"]["skill_key"] == "bag"
    assert len(requests) == 2
    assert "previous response failed validation" in requests[1]["messages"][0]["content"]
    assert "whole mission" in json.dumps(requests[0]["messages"])


def test_two_resident_vlas_on_same_host_use_distinct_ports_and_exact_prompts(monkeypatch):
    from gear_sonic.utils.inference.vla.backends import Backend, PolicyRouter

    h = flexible_harness(monkeypatch, [("ALIGN", "bag"), ("MANIPULATE", "bag"),
                                       ("ALIGN", "bin"), ("MANIPULATE", "bin"), ("COMPLETE", "bin")])
    pings, warmups = [], []

    class Client:
        def __init__(self, host, port, **kwargs):
            self.endpoint = (host, port)

        def ping(self, **kwargs):
            pings.append(self.endpoint)
            return True

        def get_action(self, observation):
            warmups.append((self.endpoint, observation["language"]["annotation.human.task_description"][0][0]))

    router = PolicyRouter({
        "pill": Backend("127.0.0.1", 29999, "collect the pill bottle"),
        "rubbish": Backend("127.0.0.1", 30000, "trained pedal prompt "),
    }, "pill", Client, require_warmup=True)
    router.warmup({"language": {}}, stopped=lambda: False)
    h.handler.policy = router
    result = h.agent.run(1)
    assert result.reason == "mission_completed", result
    assert pings == [("127.0.0.1", 29999), ("127.0.0.1", 30000)]
    assert [prompt for _, prompt in warmups] == h.actual_prompts == ["collect the pill bottle", "trained pedal prompt "]


def test_la_cannot_finish_with_pending_actions_or_rewrite_runtime_completions():
    value = decision(None)
    value.update(decision="COMPLETE", skill=None, skill_args={}, updated_todo_list="- [x] fabricated completion")
    with pytest.raises(LaViRAAgentError, match="fully completed runtime TODO"):
        validate_flexible_action(value, available_skills={"bag": {}}, todo_list="- [ ] pick bag")
    value = decision("MOVE_TO")
    value.update(skill_args={"view_direction": "front", "target": "door"}, updated_todo_list="- [ ] door")
    with pytest.raises(LaViRAAgentError, match="preserve completed TODO"):
        validate_flexible_action(value, available_skills={"bag": {}}, todo_list="- [x] bag grasped\n- [ ] door")


def test_service_builds_flexible_agent_with_original_la_and_va_clients(monkeypatch, tmp_path):
    from gear_sonic.scripts.prepare_mission import prepare_mission
    from gear_sonic.utils.inference.lavira import service

    path = prepare_mission(tmp_path / "mission.yaml", overlays=["gear_sonic/config/missions/bag_then_bin.yaml"])
    h = mission_harness(monkeypatch)
    monkeypatch.setattr(service, "SensorGatewayRGBDCamera", lambda *args, **kwargs: h.camera)
    monkeypatch.setattr(service, "LaViRAClient", lambda **kwargs: h.client)
    agent = service._agent(service.load_lavira_config(str(path)), "unused", h.mailbox)
    assert isinstance(agent, FlexibleMissionAgent)
    assert agent.client is h.client and agent.camera is h.camera
    assert agent.skills["pick_bag"]["backend"] == "bag_vla"
    assert agent.skills["step_bin"]["backend"] == "bin_vla"


def test_live_smoke_keeps_navigation_alignment_and_skips_only_vla(monkeypatch):
    h = flexible_harness(monkeypatch, [
        ("MOVE_TO", "table"), ("ALIGN", "bag"), ("MANIPULATE", "bag"),
        ("MOVE_TO", "trash can"), ("ALIGN", "bin"), ("MANIPULATE", "bin"),
        ("COMPLETE", "trash can"),
    ])
    h.agent.smoke_assume_vla_success = True
    result = h.agent.run(1)
    assert result.reason == "smoke_mission_completed", result
    assert not h.selected and not h.actual_prompts and not h.stops
    assert not h.transitions  # Never enters CPP POSE or fabricates a handoff.
    assert not any(name.endswith("vla_task") for name, _ in h.intents)
    assert [name for name, _ in h.intents if name in {"navigation_goal", "start_base_pose"}] == [
        "navigation_goal", "start_base_pose", "navigation_goal", "start_base_pose",
    ]
    assert len(h.client.alignment_grounding_calls) == 2
    assert not any(call.get("skill") == "MANIPULATE" for call in h.client.postcheck_calls)
    assert len(h.client.la_calls) == 7
    for index in (3, 6):
        transition = h.client.la_calls[index]["transition_result"]
        assert transition["smoke_assumed_success"] and transition["status"] == "UNKNOWN"
    assert h.client.la_calls[-1]["todo_list"].count("SMOKE:") == 2
    assert "VA confirmed operation completion" not in h.client.la_calls[-1]["todo_list"]
    assert len([e for e in h.events if e["code"] == "SMOKE_VLA_ASSUMED_SUCCESS"]) == 2
    assert not any(e["code"] == "MISSION_OPERATION_COMPLETED" for e in h.events)


@pytest.mark.parametrize("smoke", [False, True])
def test_final_operation_keeps_front_view_but_intermediate_operation_scans(monkeypatch, smoke):
    h = flexible_harness(monkeypatch, [
        ("ALIGN", "bag"), ("MANIPULATE", "bag"),
        ("MOVE_TO", "trash can"), ("ALIGN", "bin"), ("MANIPULATE", "bin"),
        ("COMPLETE", "trash can"),
    ])
    h.agent.smoke_assume_vla_success = smoke
    offsets = []
    manipulate = h.agent._manipulate

    def record_after_operation(*args):
        result = manipulate(*args)
        offsets.append(len(h.intents))
        return result

    h.agent._manipulate = record_after_operation
    result = h.agent.run(1)
    assert result.state == "reached", result
    assert len(h.client.la_calls[2]["scan_views"]) == 5
    assert [v.direction for v in h.client.la_calls[-1]["scan_views"]] == ["front"]
    assert not any(name == "navigation_heading_goal" for name, _ in h.intents[offsets[-1]:])
    assert h.client.la_calls[-1]["transition_result"]["executed_skill"] == "MANIPULATE"


@pytest.mark.parametrize("failure", ["missing_alignment", "expired_alignment", "cancelled", "safety"])
def test_live_smoke_does_not_bypass_alignment_cancel_or_failure(monkeypatch, failure):
    actions = [("ALIGN", "bag"), ("MANIPULATE", "bag"), ("COMPLETE", "bag")]
    if failure == "missing_alignment":
        actions.pop(0)
    h = flexible_harness(monkeypatch, actions)
    h.agent.smoke_assume_vla_success = True
    manipulate = h.agent._manipulate

    def interrupt(*args):
        if failure == "expired_alignment":
            h.now[0] += h.agent.operation_alignment_max_age_seconds + 1
        elif failure == "cancelled":
            h.agent.cancelled = lambda generation: True
        elif failure == "safety":
            h.agent.poll_failure = lambda *args: {"reason": "radar_timeout"}
        return manipulate(*args)

    h.agent._manipulate = interrupt
    result = h.agent.run(1)
    assert result.state in {"failed", "stopped"}, result
    assert not any(e["code"] == "SMOKE_VLA_ASSUMED_SUCCESS" for e in h.events)
    assert not h.selected


def test_live_smoke_overlay_builds_agent_and_leaves_normal_profile_disabled(monkeypatch, tmp_path):
    from gear_sonic.scripts.prepare_mission import prepare_mission
    from gear_sonic.utils.inference.lavira import service
    overlay = "gear_sonic/config/missions/bag_then_bin.yaml"
    smoke = "gear_sonic/config/missions/smoke_assume_vla_success.yaml"
    normal = prepare_mission(tmp_path / "normal.yaml", overlays=[overlay])
    path = prepare_mission(tmp_path / "smoke.yaml", overlays=[overlay, smoke])
    assert not service.load_lavira_config(str(normal)).smoke_assume_vla_success
    config = service.load_lavira_config(str(path))
    h = mission_harness(monkeypatch)
    monkeypatch.setattr(service, "SensorGatewayRGBDCamera", lambda *a, **k: h.camera)
    monkeypatch.setattr(service, "LaViRAClient", lambda **k: h.client)
    agent = service._agent(config, "unused", h.mailbox)
    assert agent.smoke_assume_vla_success
    assert agent.client is h.client
    assert config.mission == service.load_lavira_config(str(normal)).mission


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("delay, moved, succeeds", [(7.1, False, True), (31, False, False), (7.1, True, False)])
def test_alignment_age_is_independent_of_vla_start_ack_timeout(monkeypatch, smoke, delay, moved, succeeds):
    h = flexible_harness(monkeypatch, [("ALIGN", "bag"), ("MANIPULATE", "bag"), ("COMPLETE", "bag")])
    h.agent.smoke_assume_vla_success = smoke
    h.agent.vla_start_timeout_seconds = 6
    h.agent.operation_alignment_max_age_seconds = 30
    original_la = h.client.language_action

    def slow_la(**kwargs):
        result = original_la(**kwargs)
        if result["skill"] == "MANIPULATE":
            h.now[0] += delay
            if moved:
                x, y, yaw = h.camera.pose
                h.camera.pose = (x + 0.5, y, yaw)
        return result

    h.client.language_action = slow_la
    result = h.agent.run(1)
    assert (result.state == "reached") == succeeds, result
    if succeeds:
        assert result.reason == ("smoke_mission_completed" if smoke else "mission_completed")
    else:
        assert "alignment_expired_or_moved" in result.reason
        assert not h.selected
        assert any(e["code"] == "OPERATION_ALIGNMENT_INVALIDATED" for e in h.events)


@pytest.mark.parametrize("changes", [
    {"mission_steps": [{"id": "prewritten"}]}, {"global_target": "fixed destination"},
    {"mission": ""}, {"flexible_mission": "true"}, {"skills": {}},
    {"smoke_assume_vla_success": "true"},
    {"smoke_assume_vla_success": True, "flexible_mission": False},
])
def test_flexible_profile_rejects_fixed_steps_or_missing_whole_prompt(tmp_path, changes):
    from pathlib import Path
    import yaml
    from gear_sonic.scripts.prepare_mission import prepare_mission

    overlay = yaml.safe_load(Path("gear_sonic/config/missions/bag_then_bin.yaml").read_text())
    overlay["components"]["lavira"].update(changes)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(overlay))
    with pytest.raises(ValueError):
        prepare_mission(tmp_path / "runtime.yaml", overlays=[str(path)])
