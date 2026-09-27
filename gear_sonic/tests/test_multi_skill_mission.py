"""Exercise the real mission, gateway and VLA state machines with fake sensors/CPP."""

from dataclasses import asdict, replace
import queue
import threading
from types import SimpleNamespace

import pytest

from gear_sonic.runtime.gateway.control import ControlGatewayCore, NavigationControlState
from gear_sonic.runtime.gateway.services.control import ControlGatewayRuntime
from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.tests.test_lavira_agent import (
    build_agent, move, align, decision, grounding, alignment_grounding,
    ready_to_manipulate, task_complete,
)
from gear_sonic.utils.inference.lavira.mission import MissionAgent, MissionStep, validate_mission_profile
from gear_sonic.utils.inference.lavira.service import LaviraPlannerRuntime, load_lavira_config
from gear_sonic.utils.inference.vla.runtime import _VlaCommandHandler, _VlaRuntimeState
from gear_sonic.utils.inference.vla.handoff import PlannerHandoff


def mission_harness(monkeypatch, *, handoff="ready"):
    now = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    monkeypatch.setattr("gear_sonic.utils.inference.vla.runtime._current_vla_safety_reason", lambda *a: "clear")
    events = []
    agent, camera, client, intents, _ = build_agent(
        [move(), align(), decision("MANIPULATE"),
         move("door", global_target="door"), align(global_target="door"),
         decision("MANIPULATE", global_target="door"), move("basket")],
        groundings=[grounding()] * 3 + [grounding(description="door")] * 3 + [grounding()] * 3,
        alignment_groundings=[alignment_grounding()] * 2,
        postchecks=[ready_to_manipulate(), ready_to_manipulate(), task_complete()] * 2 + [ready_to_manipulate()],
        events=events,
    )
    agent.__class__ = MissionAgent
    agent.mission_steps = (
        MissionStep("first", "approach basket", "basket", "pill", "pick pill", "pill collected", completion_settle_seconds=1),
        MissionStep("second", "approach door", "door", "rubbish", "pick rubbish", "rubbish collected",
                    completion_settle_seconds=0, navigation_upper_body="hold_current"),
        MissionStep("return", "return to basket", "basket"),
    )
    mailbox = LaviraPlannerRuntime(load_lavira_config(), submit_intent=lambda *a: None)
    mailbox.start_navigation(1)
    gateway = object.__new__(ControlGatewayRuntime)
    gateway.navigation = NavigationControlState()
    gateway.navigation.handle_key("n", now=now[0])
    gateway.pending_vla_completion = None
    gateway.show_event = gateway.show_command = lambda *a, **k: None
    gateway.navigation_pub = SimpleNamespace(send_string=lambda _: None)
    terminal = []
    gateway.experiment = SimpleNamespace(
        config={}, active=None, recorder=SimpleNamespace(write=lambda *a, **k: None),
        tick=lambda: None, finish=lambda *a: terminal.append(a),
    )
    state = _VlaRuntimeState(cpp_mode="PLANNER", cpp_loop_running=True)
    selected = []
    agent_core = ControlGatewayCore(source="lavira_agent")
    vla_core = ControlGatewayCore(source="vla_service")
    transitions = []

    def send_cpp(*, start, planner):
        state.cpp_request_id += 1
        state.cpp_mode = "PLANNER" if planner else "POSE"
        transitions.append((now[0], state.cpp_mode))
        return True

    def invalidate(_):
        state.inference_generation += 1
        state.cached_action_chunk = None

    def vla_status(name, payload):
        if name == "vla_handoff_status":
            gateway._handle_vla_handoff_status(vla_core.accept_command(name, parameters=payload).command)
        else:
            mailbox.accept_status(payload, channel=name)

    def fail(reason, message):
        raise AssertionError((reason, message))

    handler = _VlaCommandHandler(
        state, control_listener=None, language_prompt_ref=["initial"],
        inference_failures=queue.Queue(), inference_failed_event=threading.Event(),
        vla_safety_gate=None, vla_safety_monitor=None,
        task_status_intent=SimpleNamespace(send=vla_status),
        policy=SimpleNamespace(select=selected.append, ping=lambda **kw: True),
        record_event=lambda *a, **k: None, invalidate_inference=invalidate,
        publish_initial_pose=lambda: True, send_cpp_control_command=send_cpp,
        activate_vla_metrics=lambda: None, fail_active_task=fail,
        publish_task_status=lambda status, reason, window_id: vla_status("vla_task_status", {
            "generation": state.task_generation, "skill_id": state.task_skill_id,
            "segment_id": window_id, "state": status, "reason": reason}),
        read_cpp_state=lambda: {"control_status": [1, 1, state.cpp_request_id if handoff == "ready" else 0]},
        monotonic=lambda: now[0],
    )

    def dispatch(name, params):
        if name in handler.TASK_COMMANDS:
            handler._handle_task_command(SimpleNamespace(name=name, parameters=params))
        elif name == "vla_handoff_status":
            mailbox.accept_status(params, channel=name)

    gateway.dispatch_navigation_event = dispatch
    original_wait = agent.wait_status

    def submit(name, params):
        intents.append((name, dict(params)))
        if name.endswith("vla_task"):
            command = agent_core.accept_command(name, parameters=params).command
            gateway._handle_vla_task(command)
            if name == "stop_vla_task" and params.get("continue_mission"):
                # Duplicate delivery with a new envelope must not postpone the deadline.
                pending = gateway.pending_vla_completion
                gateway._handle_vla_task(command)
                assert gateway.pending_vla_completion is pending
        elif name in {"navigation_goal", "navigation_heading_goal"}:
            gateway.navigation.accept_goal(params)
        elif name == "start_base_pose":
            gateway.navigation.accept_base_pose_start(params)

    def wait(generation, skill_id, segment_id, timeout):
        if segment_id == -1:
            with pytest.raises(ValueError):
                gateway.navigation.accept_goal(dict(generation=generation, skill_id=skill_id + 1, segment_id=0))
            for _ in range(28):
                now[0] += 0.5
                if handoff == "cancel":
                    action = gateway.navigation.handle_key(" ", now=now[0])
                    mailbox.cancel(action.generation, "operator_stop")
                    handler._cancel_navigation(action.generation)
                    break
                gateway._handle_timeout()
                handler.tick()
                if state.handoff and state.handoff.result:
                    break
            return mailbox.wait_status(generation, skill_id, -1, 0)
        if intents[-1][0] == "start_vla_task":
            return mailbox.wait_status(generation, skill_id, 0, 0)
        status = original_wait(generation, skill_id, segment_id, timeout)
        owner = "base_pose" if intents[-1][0] == "start_base_pose" else "lavira"
        assert gateway.navigation.accept_status(status, owner=owner, agent_final=False)
        return status

    agent.submit_intent, agent.wait_status = submit, wait
    agent.cancelled = mailbox.is_cancelled
    return SimpleNamespace(**locals())


def test_two_skills_then_navigation_waits_for_cpp_and_finishes_once(monkeypatch):
    h = mission_harness(monkeypatch)
    result = h.agent.run(1)
    assert result.state == "reached", result.reason
    assert result.reason == "mission_completed"
    assert h.selected == ["pill", "rubbish"]
    assert h.gateway.navigation.generation == 1
    assert not h.terminal  # Individual skills never finish the mission/trial.
    assert h.state.hold_current_upper_body
    assert not h.state.initial_pose_left_hand_closed and not h.state.initial_pose_right_hand_closed
    assert len([e for e in h.events if e["code"] == "TASK_COMPLETED"]) == 1
    assert [c["global_target"] for c in h.client.la_calls] == ["basket"] * 3 + ["door"] * 3 + ["basket"]
    starts = [p for n, p in h.intents if n == "start_vla_task"]
    assert [p["handoff_context"] for p in starts] == ["pick pill", "pick rubbish"]
    assert [c["expected_postcondition"] for c in h.client.postcheck_calls
            if c.get("skill") == "MANIPULATE"] == ["pill collected", "rubbish collected"]
    assert starts[0]["skill_id"] < starts[1]["skill_id"] < result.skill_id
    h.gateway._handle_navigation_agent_status(h.agent_core.accept_command(
        "navigation_agent_status", parameters=asdict(result)).command)
    assert h.gateway.navigation.mode == "listen_wasd"
    assert len(h.terminal) == 1


@pytest.mark.parametrize("handoff,expected", [("timeout", "failed"), ("cancel", "stopped")])
def test_handoff_failure_or_cancel_never_starts_next_goal(monkeypatch, handoff, expected):
    h = mission_harness(monkeypatch, handoff=handoff)
    result = h.agent.run(1)
    assert result.state == expected, result.reason
    assert h.selected == ["pill"]
    assert len(h.client.la_calls) == 3
    if handoff == "timeout":
        assert "planner_handoff_timeout" in result.reason
        assert h.gateway.navigation.mode == "lavira_handoff"


def test_handoff_rejects_old_ack_and_requires_continuous_fresh_readiness():
    monitor = PlannerHandoff(55, 10)
    assert monitor.update({"control_status": [1, 1, 54]}, 0) is None
    assert monitor.update({"control_status": [1, 1, 55]}, 1) is None
    assert monitor.update(None, 2) is None
    assert monitor.update({"control_status": [1, 1, 55]}, 3) is None
    assert monitor.update({"control_status": [1, 1, 55]}, 4.9) is None
    assert monitor.update({"control_status": [1, 1, 55]}, 5) == "ready"
    nav = NavigationControlState()
    nav.handle_key("n", now=0)
    nav.skill_id = 4
    nav.mode = "lavira_handoff"
    assert not nav.accept_vla_handoff(dict(generation=1, skill_id=3, state="ready"))
    assert not nav.accept_vla_handoff(dict(generation=0, skill_id=4, state="ready"))
    assert nav.mode == "lavira_handoff"


def test_example_profile_and_unknown_backend(tmp_path):
    profile = load_runtime_profile(overlays=["gear_sonic/config/missions/two_skills.yaml"])
    assert len(validate_mission_profile(profile)) == 3
    assert len(load_lavira_config(overlays=("gear_sonic/config/missions/two_skills.yaml",)).mission_steps) == 3
    from gear_sonic.scripts.prepare_mission import prepare_mission
    from gear_sonic.scripts.launch_inference import load_inference_launch_config
    from gear_sonic.utils.inference.vla.service import load_inference_config
    output = prepare_mission(tmp_path / "mission.yaml", overlays=["gear_sonic/config/missions/two_skills.yaml"])
    assert load_inference_launch_config(output).planner_input == "lavira"
    assert len(load_lavira_config(str(output)).mission_steps) == 3
    assert set(load_inference_config(str(output)).backends) == {"collect_pill", "collect_rubbish"}
    from gear_sonic.utils.inference.lavira.mission import parse_mission_steps
    with pytest.raises(ValueError, match="unknown mission skill"):
        parse_mission_steps([asdict(MissionStep("one", "find basket", "basket", "bad", "pick", "done"))], backends={})


def test_legacy_profiles_need_no_new_fields_and_existing_fields_stay_required(tmp_path):
    import yaml
    from gear_sonic.runtime.profile import default_runtime_profile_path
    from gear_sonic.utils.inference.vla.service import load_inference_config
    payload = yaml.safe_load(default_runtime_profile_path().read_text())
    assert "mission_steps" not in payload["components"]["lavira"]
    assert "backends" not in payload["components"]["vla"]
    assert load_lavira_config().mission_steps == []
    assert load_inference_config().backends == {}
    del payload["components"]["vla"]["action_horizon"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="missing fields: action_horizon"):
        load_inference_config(str(path))


def test_handoff_wait_propagates_safety_failure_from_execution_window():
    runtime = LaviraPlannerRuntime(load_lavira_config(), submit_intent=lambda *a: None)
    runtime.start_navigation(1)
    runtime.accept_status({"generation": 1, "skill_id": 3, "segment_id": 6,
                           "state": "failed", "reason": "radar_timeout"}, channel="vla_task_status")
    assert runtime.wait_status(1, 3, -1, 0)["reason"] == "radar_timeout"


def test_mode_command_ack_id_survives_wire_encoding():
    from gear_sonic.runtime.protocol.cpp_control import build_command_message
    from gear_sonic.runtime.protocol.array_message import unpack_array_message
    token = 2**52 + 101
    fields = unpack_array_message(build_command_message(True, False, True, request_id=token,
                                   hold_current_upper_body=True), expected_topic="command").fields
    assert fields["control_request_id"].tolist() == [token]
    assert fields["hold_current_upper_body"].tolist() == [1]
    with pytest.raises(ValueError):
        build_command_message(True, False, True, request_id=2**53)


def operation(name, *, backend="pill", exit="hold", completion="visual", attempts=3):
    return MissionStep(name=name, type="skill", mission=f"perform {name}",
                       global_target="basket", skill_key=backend,
                       manipulation_prompt=f"perform {name}", expected_postcondition=f"{name} completed",
                       align={"target": f"basket for {name}", "max_attempts": attempts},
                       exit=exit, completion_kind=completion, completion_settle_seconds=0)


def explicit_harness(monkeypatch, *, manual=False, handoff="ready"):
    h = mission_harness(monkeypatch, handoff=handoff)
    h.agent.mission_steps = (
        operation("a", completion="manual" if manual else "visual"),
        MissionStep("move_b", "approach door", "door", type="navigate"),
        operation("b", backend="rubbish", exit="default"),
        operation("c"),  # Adjacent operations; reuse the first model, but realign.
        MissionStep("return", "return to basket", "basket", type="navigate"),
    )
    h.client.decisions = iter([move("door", global_target="door"), move("basket")])
    h.client.groundings = iter([grounding(description="door")] * 3 + [grounding()] * 3)
    h.client.alignment_groundings = iter([alignment_grounding()] * 3)
    first = [ready_to_manipulate(), ready_to_manipulate()]
    h.client.postchecks = iter(first + ([] if manual else [task_complete()]) +
                              [ready_to_manipulate(), ready_to_manipulate(), task_complete()] * 2)
    h.state.initial_pose_left_hand_closed = True
    h.state.initial_pose_right_hand_closed = False
    h.camera.sonic_yaw = 1.1
    h.camera.pose = (0.0, 0.0, 1.1)
    h.snapshots = []
    original_send = h.handler.send_cpp_control_command

    def send(**kwargs):
        h.snapshots.append((kwargs["planner"], h.state.hold_current_upper_body,
                            h.state.initial_pose_left_hand_closed, h.state.initial_pose_right_hand_closed))
        return original_send(**kwargs)

    h.handler.send_cpp_control_command = send
    original_dispatch = h.gateway.dispatch_navigation_event

    def dispatch(name, params):
        if name == "skill_completion":
            h.mailbox.accept_status(params, channel=name)
        else:
            original_dispatch(name, params)

    h.gateway.dispatch_navigation_event = dispatch
    original_wait = h.agent.wait_status
    h.manual_events = []

    def wait(generation, skill_id, segment_id, timeout):
        if segment_id == -2:
            core = ControlGatewayCore(source="operator_console")
            command = core.accept_console_line("skill_done").command
            h.gateway._handle_complete_skill(command)
            h.manual_events.append((generation, skill_id))
            assert h.gateway.navigation.complete_skill() is None
            return h.mailbox.wait_status(generation, skill_id, -2, 0)
        return original_wait(generation, skill_id, segment_id, timeout)

    h.agent.wait_status = wait
    return h


@pytest.mark.parametrize("manual", [False, True])
def test_explicit_interleaving_realigns_each_skill_and_maps_only_exit_targets(monkeypatch, manual):
    h = explicit_harness(monkeypatch, manual=manual)
    result = h.agent.run(1)
    assert result.state == "reached", result.reason
    assert h.selected == ["pill", "rubbish", "pill"]
    assert len(h.client.la_calls) == 2
    assert all(call["navigation_only"] for call in h.client.la_calls)
    assert [call["global_target"] for call in h.client.la_calls] == ["door", "basket"]
    starts = [params for name, params in h.intents if name == "start_vla_task"]
    aligns = [params for name, params in h.intents if name == "start_base_pose"]
    assert [params["operation_id"] for params in aligns] == ["a", "b", "c"]
    assert len(starts) == len(aligns) == 3
    for start, aligned in zip(starts, aligns):
        assert start["operation_id"] == aligned["operation_id"]
        assert start["alignment_skill_id"] == aligned["skill_id"]
        assert start["alignment_segment_id"] == aligned["segment_id"]
        assert start["skill_id"] > aligned["skill_id"]
    assert h.snapshots[1::2] == [(True, True, True, False),
                                  (True, False, False, False), (True, True, False, False)]
    assert len(h.manual_events) == int(manual)
    assert len([call for call in h.client.postcheck_calls if call.get("skill") == "MANIPULATE"]) == 3 - int(manual)
    assert all(f"basket for {name}" in call["manipulation_prompt"]
               for name, call in zip(("a", "b", "c"), h.client.alignment_grounding_calls))
    assert h.intents[0][0] == "navigation_heading_goal"
    assert h.intents[0][1]["heading_delta_rad"] == pytest.approx(0.0)
    assert len([event for event in h.events if event["code"] == "TASK_COMPLETED"]) == 1
    # Manual keys remain independent after task exits; no whole-body lock was added.
    h.handler._toggle_right_hand(None)
    assert h.state.initial_pose_right_hand_closed
    assert h.state.hold_current_upper_body
    late = dict(generation=1, skill_id=starts[1]["skill_id"], window_id=0,
                reason="postcondition_satisfied", continue_mission=True, exit_policy="default")
    h.handler._handle_task_command(SimpleNamespace(name="stop_vla_task", parameters=late))
    assert h.state.initial_pose_right_hand_closed  # A late exit cannot overwrite the operator.


@pytest.mark.parametrize("bad_alignment", ["not_found", "controller_failed", "vision_failed", "out_of_range"])
def test_explicit_skill_never_runs_without_its_successful_alignment(monkeypatch, bad_alignment):
    h = explicit_harness(monkeypatch)
    h.agent.mission_steps = (operation("a", attempts=1), operation("b"))
    if bad_alignment == "not_found":
        h.client.alignment_groundings = iter([alignment_grounding(status="NOT_FOUND")])
    elif bad_alignment == "out_of_range":
        h.camera.depth_mm = 5000.0
    elif bad_alignment == "vision_failed":
        h.client.postchecks = iter([ready_to_manipulate(status="NOT_SATISFIED")] * 2)
    else:
        original = h.agent.wait_status

        def wait(generation, skill_id, segment_id, timeout):
            status = original(generation, skill_id, segment_id, timeout)
            if h.intents[-1][0] == "start_base_pose":
                status = {**status, "state": "failed", "reason": "alignment_failed"}
            return status

        h.agent.wait_status = wait
    result = h.agent.run(1)
    assert result.state == "failed"
    assert not h.selected
    assert not any(name == "start_vla_task" for name, _ in h.intents)
    assert h.state.initial_pose_left_hand_closed  # Failure does not apply a default exit.


def test_gateway_binds_alignment_to_operation_and_rejects_reuse():
    nav = NavigationControlState()
    nav.handle_key("n", now=0)
    command = dict(generation=1, skill_id=2, operation_id="a", alignment_skill_id=1,
                   alignment_segment_id=0, completion_kind="manual")
    assert not nav.accept_vla_command("start_vla_task", command)
    nav.accept_base_pose_start(dict(generation=1, skill_id=1, segment_id=0, operation_id="a"))
    assert not nav.accept_status(dict(generation=1, skill_id=1, segment_id=9, state="reached", reason="aligned"), owner="base_pose")
    assert nav.accept_status(dict(generation=1, skill_id=1, segment_id=0, state="reached", reason="aligned"), owner="base_pose")
    assert not nav.accept_vla_command("start_vla_task", {**command, "operation_id": "b"})
    assert not nav.accept_vla_command("start_vla_task", dict(generation=1, skill_id=2))
    assert nav.accept_vla_command("start_vla_task", command)
    assert nav.accept_vla_command("start_vla_task", command)  # Idempotent retransmission.
    assert not nav.accept_vla_command("start_vla_task", {**command, "operation_id": "b"})
    assert nav.complete_skill()["operation_id"] == "a"
    assert nav.complete_skill() is None
    nav.mode = "lavira_handoff"
    assert nav.accept_vla_handoff(dict(generation=1, skill_id=2, state="ready"))
    assert not nav.accept_vla_command("start_vla_task", {**command, "skill_id": 3})
    nav.handle_key(" ", now=1)
    assert nav.complete_skill() is None
    assert nav.alignment_ticket is None and nav.pending_operation_id == nav.operation_id == ""
    assert not nav.accept_vla_command("start_vla_task", command)


@pytest.mark.parametrize("handoff", ["timeout", "cancel"])
def test_explicit_operation_handoff_failure_never_starts_next_step(monkeypatch, handoff):
    h = explicit_harness(monkeypatch, handoff=handoff)
    result = h.agent.run(1)
    assert result.state == ("failed" if handoff == "timeout" else "stopped")
    assert h.selected == ["pill"]
    assert not h.client.la_calls


def test_explicit_nav_rejects_la_attempt_to_start_a_skill(monkeypatch):
    h = explicit_harness(monkeypatch)
    h.agent.mission_steps = (MissionStep("nav", "approach basket", "basket", type="navigate"),)
    h.client.decisions = iter([decision("MANIPULATE")])
    result = h.agent.run(1)
    assert result.state == "failed"
    assert "only permits MOVE_TO" in result.reason
    assert not h.selected


def test_legacy_exit_keeps_manual_grippers_without_an_explicit_new_policy(monkeypatch):
    h = mission_harness(monkeypatch)
    h.state.initial_pose_left_hand_closed = True
    result = h.agent.run(1)
    assert result.state == "reached", result.reason
    assert h.state.initial_pose_left_hand_closed
    assert not h.state.initial_pose_right_hand_closed


def test_completed_alignment_is_rechecked_for_motion_before_vla_start(monkeypatch):
    h = explicit_harness(monkeypatch)
    original_start = h.agent._start_manipulation

    def start(generation, skill_id):
        x, y, yaw = h.camera.pose
        h.camera.pose = (x + 0.5, y, yaw)
        return original_start(generation, skill_id)

    h.agent._start_manipulation = start
    result = h.agent.run(1)
    assert result.state == "failed"
    assert "alignment_expired_or_moved" in result.reason
    assert not h.selected


def test_manual_completion_wait_propagates_failure_without_applying_exit(monkeypatch):
    h = explicit_harness(monkeypatch, manual=True)
    h.agent.mission_steps = (operation("a", exit="default", completion="manual"),)
    original_wait = h.agent.wait_status

    def wait(generation, skill_id, segment_id, timeout):
        if segment_id == -2:
            h.mailbox.accept_status(dict(generation=generation, skill_id=skill_id,
                                         segment_id=4, state="failed", reason="radar_timeout"),
                                    channel="vla_task_status")
            return h.mailbox.wait_status(generation, skill_id, -2, 0)
        return original_wait(generation, skill_id, segment_id, timeout)

    h.agent.wait_status = wait
    result = h.agent.run(1)
    assert result.state == "failed"
    assert "radar_timeout" in result.reason
    assert h.state.initial_pose_left_hand_closed
    stops = [params for name, params in h.intents if name == "stop_vla_task"]
    assert stops and all(params["reason"] != "postcondition_satisfied" for params in stops)


def test_mission_separates_agent_prompts_from_actual_vla_language_and_resets_each_step(monkeypatch):
    h = explicit_harness(monkeypatch)
    h.agent.mission_steps = tuple(
        replace(step, vla_trained_prompt={"a": "collect the pill bottle", "b": "trained bin prompt "}.get(step.name))
        for step in h.agent.mission_steps
    )
    actual_vla_prompts = []
    start = h.agent._start_manipulation

    def record_start(*args):
        start(*args)
        actual_vla_prompts.append(h.handler.language_prompt_ref[0])

    h.agent._start_manipulation = record_start
    result = h.agent.run(1)
    assert result.reason == "mission_completed"
    assert actual_vla_prompts == ["collect the pill bottle", "trained bin prompt ", "perform c"]
    starts = [params for name, params in h.intents if name == "start_vla_task"]
    assert [params["task"] for params in starts] == ["perform a", "perform b", "perform c"]
    assert [params["handoff_context"] for params in starts] == actual_vla_prompts
    assert [call["mission"] for call in h.client.postcheck_calls if call.get("skill") == "MANIPULATE"] == [
        "perform a", "perform b", "perform c",
    ]
    assert all("trained bin prompt" not in call["manipulation_prompt"]
               for call in h.client.alignment_grounding_calls)


@pytest.mark.parametrize("changes", [
    {"vla_prompt": ""}, {"vla_prompt": None}, {"vla_prompt": "  "},
    {"vla_trained_prompt": ""}, {"vla_trained_prompt": None}, {"vla_trained_prompt": 42},
    {"prompt": "ambiguous old prompt"},
])
def test_split_prompt_config_rejects_missing_or_ambiguous_text(changes):
    from gear_sonic.utils.inference.lavira.mission import parse_skill_definitions
    spec = dict(backend="model", vla_prompt="Agent task", vla_trained_prompt="trained task ",
                completion={"kind": "manual"}, exit="hold")
    with pytest.raises(ValueError):
        parse_skill_definitions({"skill": {**spec, **changes}})


def test_interleaved_profile_resolves_logical_skills_and_keeps_legacy_launcher(tmp_path):
    from gear_sonic.scripts.prepare_mission import prepare_mission
    overlay = "gear_sonic/config/missions/interleaved_skills.yaml"
    profile = load_runtime_profile(overlays=[overlay])
    steps = validate_mission_profile(profile)
    assert [step.type for step in steps] == ["skill", "navigate", "skill", "skill", "navigate"]
    assert [step.skill_key for step in steps if step.type == "skill"] == ["model_a", "model_b", "model_a"]
    assert [step.exit for step in steps if step.type == "skill"] == ["hold", "hold", "default"]
    path = prepare_mission(tmp_path / "launch.yaml", overlays=[overlay])
    config = load_lavira_config(str(path))
    assert len(config.skills) == 3 and len(config.mission_steps) == 5


@pytest.mark.parametrize("changes", [
    {"align": {}}, {"exit": "inherit"}, {"exit": "preset"},
    {"align": {"target": "basket", "enabled": False}},
    {"navigation_upper_body": "hold_current"}, {"grippers": "hold"},
    {"completion": {"kind": "grasp_detection"}}, {"completion": {"kind": []}},
])
def test_explicit_skill_config_rejects_missing_alignment_or_split_exit(changes):
    from gear_sonic.utils.inference.lavira.mission import parse_mission_steps
    step = dict(id="one", type="skill", skill_key="model_a", manipulation_prompt="task a",
                completion={"kind": "manual"}, align={"target": "basket"}, exit="hold")
    with pytest.raises(ValueError):
        parse_mission_steps([{**step, **changes}], backends={"model_a": {}})


@pytest.mark.parametrize("invalid", [None, [], True, "open_only"])
def test_gateway_rejects_invalid_exit_before_accepting_vla_command(monkeypatch, invalid):
    h = mission_harness(monkeypatch)
    before = (h.gateway.navigation.mode, h.gateway.navigation.skill_id)
    command = h.agent_core.accept_command("start_vla_task", parameters={
        "generation": 1, "skill_id": 1, "exit_policy": invalid,
    }).command
    with pytest.raises(ValueError, match="exit_policy"):
        h.gateway._handle_vla_task(command)
    assert (h.gateway.navigation.mode, h.gateway.navigation.skill_id) == before


def test_navigation_client_uses_navigation_prompt_and_retries_forbidden_action(tmp_path):
    import json
    import numpy as np
    from gear_sonic.utils.inference.lavira.agent import LaViRAClient, ScanView
    calls = []
    replies = iter([decision("MANIPULATE"), move()])

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(replies))))])

    transport = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = LaViRAClient(la_base_url="https://example.invalid", va_base_url="https://example.invalid",
                          la_client=transport, va_client=transport, request_context_dir=tmp_path)
    result = client.language_action(mission="find basket", navigation_mode="object_nav", global_target="basket",
                                    current_step=1, todo_list="", navigation_only=True,
                                    scan_views=[ScanView(1, "front", np.zeros((8, 8, 3), np.uint8), (0, 0, 0), 0)],
                                    move_to_views=[])
    assert result["skill"] == "MOVE_TO" and len(calls) == 2
    prompt = json.dumps(calls[0]["messages"])
    assert "Only plan MOVE_TO waypoints" in prompt
    assert "MANIPULATE skill_args must" not in prompt
    assert "MANIPULATION TASK AFTER NAVIGATION" not in prompt
