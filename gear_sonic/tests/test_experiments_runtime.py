"""Offline behavioral checks: no robot sockets, models, or real launches."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gear_sonic.experiments.agent import ExperimentAgent
from gear_sonic.experiments.base_pose import GeometricController
from gear_sonic.experiments.config import CONFIGS, launch, resolve
from gear_sonic.experiments.recording import read_events
from gear_sonic.experiments.supervisor import ExperimentSupervisor
from gear_sonic.runtime.gateway.control import NavigationControlState
from gear_sonic.tests import test_lavira_agent as helpers

METHODS = sorted(p.stem for p in CONFIGS.glob("*.yaml") if not p.stem.startswith(("cases_", "tasks")))


def experiment(tmp_path, method="near_dual"):
    payload, _ = resolve(CONFIGS / f"{method}.yaml")
    exp = payload["components"]["experiment"]
    exp.update(run_dir=str(tmp_path), session_id="offline-test")
    exp["task"].update(vla_prompt="SEMANTIC real physical task", vla_trained_prompt="TRAINED exact policy string")
    if exp["navigation_backend"] == "navila":
        exp["condition"]["navila_alignment"] = dict(target="blue basket", yaw_align_target="desk")
    return exp


def make_agent(monkeypatch, tmp_path, method, decisions=(), **kwargs):
    exp = experiment(tmp_path, method)
    monkeypatch.setattr(helpers, "LaViRAAgent", lambda **options: ExperimentAgent(experiment=exp, **options))
    agent, camera, client, intents, waited = helpers.build_agent(
        decisions, manipulation_prompt=exp["task"]["vla_prompt"], mission="NAVIGATION ROUTE", **kwargs
    )
    if agent.navila is not None:
        capture = camera.capture_rgb
        camera.capture_cutoffs = []

        def capture_after(*, camera_stream="chest_view", after_timestamp_ns=None):
            camera.capture_cutoffs.append(after_timestamp_ns)
            return capture(camera_stream=camera_stream)

        camera.capture_rgb = capture_after
        wait = agent.wait_status

        def wait_navila(*args):
            status = wait(*args)
            if intents[-1][0] == "experiment_nav_step":
                status["reason"] = "navila_step_completed"
            return status

        agent.wait_status = wait_navila
    return agent, camera, client, intents


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("task", ["T1", "T2", "T3", "T4"])
def test_every_profile_loads_strict_runtime_components(tmp_path, method, task):
    from gear_sonic.scripts.launch_inference import load_inference_launch_config
    from gear_sonic.utils.inference.base_pose.agent import load_base_pose_config
    from gear_sonic.utils.inference.lavira.service import load_lavira_config
    from gear_sonic.utils.inference.vla.service import load_inference_config

    payload, missing = resolve(CONFIGS / f"{method}.yaml", task)
    assert payload["components"]["base_pose"]["raw_diagnostic_image_interval_frames"] == 5
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(payload))
    for loader in (load_inference_launch_config, load_base_pose_config, load_lavira_config, load_inference_config):
        loader(str(path))
    assert payload["components"]["vla"]["prompt"] == (
        payload["components"]["experiment"]["task"]["vla_trained_prompt"] or "REQUIRED"
    )
    if method == "semantic_roles":
        assert not missing


def test_all_dry_runs_do_not_launch_or_create_results(tmp_path, capsys, monkeypatch):
    scripts = CONFIGS.parents[1] / "scripts" / "experiments"
    assert set(METHODS) == {path.stem for path in scripts.glob("*.sh")}
    monkeypatch.setattr(
        "gear_sonic.experiments.config.subprocess.run", lambda *a, **k: pytest.fail("dry-run invoked a process")
    )
    for method in METHODS:
        assert launch(method, ["--dry-run", "--output", str(tmp_path / "runs")]) == 0
        output = yaml.safe_load(capsys.readouterr().out)
        assert output["ready"] is (not output["missing"])
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("method,head_only", [("near_dual", False), ("near_head", True), ("near_vla", False)])
def test_nearfield_executes_vla_with_separate_prompts(monkeypatch, tmp_path, method, head_only):
    agent, camera, client, intents = make_agent(
        monkeypatch,
        tmp_path,
        method,
        alignment_groundings=[helpers.alignment_grounding()],
        postchecks=(
            [helpers.task_complete()]
            if method == "near_vla"
            else [helpers.ready_to_manipulate()] * (1 if head_only else 2) + [helpers.task_complete()]
        ),
    )
    result = agent.run(11)
    assert result.state == "reached", result
    assert not client.la_calls
    assert not any(name == "navigation_goal" for name, _ in intents)
    starts = [p for name, p in intents if name == "start_vla_task"]
    assert len(starts) == 1
    assert starts[0]["handoff_context"] == "TRAINED exact policy string"
    assert all(c["mission"] == "SEMANTIC real physical task" for c in client.postcheck_calls)
    assert all(
        np.mean(c["image_bgr"]) > 100
        for c in client.postcheck_calls if c.get("skill") == "MANIPULATE"
    )
    completion_checks = [
        e for e in read_events(agent.recorder.path) if e["type"] == "check" and e["stage"] == "MANIPULATE"
    ]
    assert len(completion_checks) == 1
    assert completion_checks[0]["camera"] == "head"
    assert completion_checks[0]["observation"]["stream"] == "ego_view"
    if method == "near_vla":
        assert not any(name == "start_base_pose" for name, _ in intents)
    else:
        assert client.alignment_grounding_calls[0]["manipulation_prompt"] == "SEMANTIC real physical task"
        if head_only:
            assert np.mean(client.alignment_grounding_calls[0]["image_bgr"]) > 100
            assert np.mean(client.postcheck_calls[0]["image_bgr"]) > 100
            assert (
                len(
                    [e for e in read_events(agent.recorder.path) if e["type"] == "check" and e["stage"] == "ALIGN"]
                )
                == 2
            )


@pytest.mark.parametrize(
    "method",
    ["full_vln", "full_objectnav", "nav_direct_vla", "gate_nav_on", "gate_align_on", "gate_completion_on"],
)
def test_navigation_branches_keep_shared_agent_and_semantic_checks(monkeypatch, tmp_path, method):
    direct = method == "nav_direct_vla"
    decisions = [helpers.move()] + ([] if direct else [helpers.align(), helpers.decision("MANIPULATE")])
    agent, _, client, intents = make_agent(
        monkeypatch,
        tmp_path,
        method,
        decisions,
        groundings=[helpers.grounding()] * 3,
        alignment_groundings=[helpers.alignment_grounding()],
        postchecks=(
            [helpers.task_complete()]
            if direct
            else [helpers.ready_to_manipulate()] * 2 + [helpers.task_complete()]
        ),
    )
    result = agent.run(1)
    assert result.state == "reached", result
    assert len(client.la_calls) == (1 if direct else 3)
    assert all(c["mission"] == "SEMANTIC real physical task" for c in client.grounding_calls)
    assert all(c["manipulation_prompt"] == "SEMANTIC real physical task" for c in client.la_calls)
    assert all(
        p["handoff_context"] == "TRAINED exact policy string" for name, p in intents if name == "start_vla_task"
    )


def test_geometric_timeout_stops_alignment_but_still_starts_vla(monkeypatch, tmp_path):
    agent, _, _, intents = make_agent(
        monkeypatch,
        tmp_path,
        "geometric_vla",
        [helpers.move(), helpers.align()],
        groundings=[helpers.grounding()] * 3,
        postchecks=[helpers.postcheck("NOT_SATISFIED")] * 2 + [helpers.task_complete()],
    )
    original = agent.wait_status

    def wait(*args):
        if intents[-1][0] == "start_base_pose":
            return dict(
                generation=args[0],
                skill_id=args[1],
                segment_id=args[2],
                state="failed",
                reason="geometric_timeout",
            )
        return original(*args)

    agent.wait_status = wait
    result = agent.run(7)
    assert result.state == "reached", result
    assert any(name == "start_vla_task" for name, _ in intents)
    gate = next(e for e in read_events(agent.recorder.path) if e["type"] == "gate" and e["gate"] == "align")
    assert gate["applied"] == "allow" and gate["va_recommendation"] is False
    assert gate["controller_state"] == "geometric_timeout"


@pytest.mark.parametrize("method", ["gate_nav_off", "gate_align_off"])
def test_disabled_gate_preserves_negative_recommendation_and_executes(monkeypatch, tmp_path, method):
    decisions = (
        [helpers.move()]
        + ([helpers.align()] if method == "gate_align_off" else [])
        + [helpers.decision("MANIPULATE")]
    )
    checks = (
        [helpers.postcheck("NOT_SATISFIED")] * 2
        if method == "gate_align_off"
        else [helpers.ready_to_manipulate()] * 2
    )
    agent, _, _, intents = make_agent(
        monkeypatch,
        tmp_path,
        method,
        decisions,
        groundings=[helpers.grounding()] * 3,
        handoff_depth_mm=5000 if method == "gate_nav_off" else 2000,
        alignment_groundings=[helpers.alignment_grounding()],
        postchecks=checks + [helpers.task_complete()],
    )
    result = agent.run(5)
    assert result.state == "reached", result
    gate = next(
        e
        for e in read_events(agent.recorder.path)
        if e["type"] == "gate" and e["gate"] == agent.experiment["gate_under_test"]
    )
    assert gate["va_recommendation"] is False and gate["applied"] == "allow"
    assert any(name == "start_vla_task" for name, _ in intents)


def test_geometry_phases_freshness_loss_and_reverse(tmp_path):
    cfg = experiment(tmp_path)["geometric"]
    control = GeometricController(cfg, 0.0)
    assert control.update(2.0, 1.0, 1, 0.0) == (0.0, 0.0, -0.3)
    assert control.update(2.0, 0.0, 2, 1.0) == (0.4, 0.0, 0.0)
    assert control.update(None, None, 3, 2.0) == (0.0, 0.0, 0.0)
    assert control.update(0.8, 0.0, 4, 3.0) == (-0.4, 0.0, 0.0)
    assert control.update(1.1, 0.3, 5, 4.0) == (0.0, 0.0, -0.2)
    for stamp in (6, 6, 7):
        assert control.update(1.1, 0.0, stamp, 5.0) == (0.0, 0.0, 0.0)
        assert control.reason is None
    control.update(1.1, 0.0, 8, 5.0)
    assert control.reason == "aligned"
    timeout = GeometricController(cfg, 0.0)
    assert timeout.update(None, None, None, 60.0) == (0.0, 0.0, 0.0)
    assert timeout.reason == "geometric_timeout"


def supervisor(monkeypatch, tmp_path, method="near_vla"):
    exp = experiment(tmp_path, method)
    profile = SimpleNamespace(components={"experiment": exp})
    navigation = NavigationControlState()
    action = navigation.handle_key("n", now=0.0)
    sent = []
    gateway = SimpleNamespace(
        profile=profile,
        navigation=navigation,
        pending_vla_completion=None,
        publish_navigation_action=lambda action, **kw: sent.append(action),
        _send_navigation_stop=lambda **kw: sent.append(("stop", kw)),
        dispatch_navigation_event=lambda *args: sent.append(args),
        navigation_pub=SimpleNamespace(send_string=sent.append),
    )
    clock = [0.0]
    monkeypatch.setattr("gear_sonic.experiments.snapshots.record_pose_async", lambda *a, **k: None)
    controller = ExperimentSupervisor(gateway, clock=lambda: clock[0])
    gateway.experiment = controller
    controller.start(action.generation)
    return controller, clock, sent


def test_gateway_total_clock_cancels_blocked_model_and_isolates_new_trial(monkeypatch, tmp_path):
    controller, clock, sent = supervisor(monkeypatch, tmp_path)
    old = controller.active[0]
    clock[0] = 600.0
    controller.tick()
    assert controller.active is None
    assert sent[-1].agent_event == "cancel_navigation"
    new = controller.gateway.navigation.handle_key("n", now=601.0).generation
    controller.start(new)
    controller.first_action({"generation": old, "skill_id": 1})
    controller.finish(old, "reached", "late_model_reply")
    assert controller.active[0] == new and controller.vla_deadline is None
    ends = [e for e in read_events(controller.recorder.path) if e["type"] == "trial_end"]
    assert len(ends) == 1 and ends[0]["reason"] == "total_timeout"


@pytest.mark.parametrize("mode", ["lavira_pending", "lavira_nav", "base_pose_motion", "lavira_manipulate"])
@pytest.mark.parametrize("method", [method for method in METHODS if method != "semantic_roles"])
@pytest.mark.parametrize("key,success", [("g", True), ("h", False)])
def test_operator_result_records_key_time_and_invalidates_old_commands(
    monkeypatch, tmp_path, mode, method, key, success,
):
    from gear_sonic.experiments.results import trial_rows
    from gear_sonic.runtime.gateway.control import ControlGatewayCore, OperatorConsoleRouter
    from gear_sonic.runtime.gateway.services.control import ControlGatewayRuntime

    event_clock = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: event_clock[0])
    monkeypatch.setattr("time.monotonic_ns", lambda: round(event_clock[0] * 1e9))
    monkeypatch.setattr("time.time_ns", lambda: round((1000.0 + event_clock[0]) * 1e9))
    controller, clock, sent = supervisor(monkeypatch, tmp_path, method)
    gateway = controller.gateway
    shown = []
    gateway.show_command = lambda *a, **k: shown.append((a, k))
    gateway.show_event = lambda *a, **k: shown.append((a, k))
    gateway._require_source = ControlGatewayRuntime._require_source
    controller.recorder.write("session", experiment=controller.config)
    nav = gateway.navigation
    generation = nav.generation
    nav.mode, nav.skill_id, nav.segment_id = mode, 3, 6
    nav.manual_velocity = (0.3, 0.0, 0.0)
    nav.manual_deadline = 100.0
    clock[0] = event_clock[0] = 42.75
    core = ControlGatewayCore()
    command = OperatorConsoleRouter().accept_line(key, core=core).command
    # A model reply and gateway processing arrive after the operator's key.
    clock[0] = event_clock[0] = 43.25
    controller.recorder.runtime("lavira", "SKILL_STARTED", generation=generation, skill="ALIGN")

    ControlGatewayRuntime._handle_complete_agent_result(gateway, command)

    assert controller.active is None and nav.mode == "listen_wasd"
    assert not nav.lavira_task_active and nav.task_started_at is None
    assert nav.manual_velocity == (0.0, 0.0, 0.0) and nav.manual_deadline == 0.0
    action = sent[-1]
    assert action.mode == "stop" and action.agent_event == "cancel_navigation"
    reason = "operator_success" if success else "operator_failure"
    assert action.generation > generation and action.reason == reason
    events = read_events(controller.recorder.path)
    end = next(e for e in events if e["type"] == "trial_end")
    assert end["state"] == ("reached" if success else "failed") and end["reason"] == reason
    assert end["completion_time_s"] == pytest.approx(42.75)
    assert end["completed_wall_time_ns"] == 1_042_750_000_000
    assert end["completed_monotonic_ns"] == 42_750_000_000
    assert end["monotonic_ns"] == 43_250_000_000
    row = trial_rows(events)[0]
    assert row["success"] is success and row["physical_success"] is success
    assert row["completion_time_s"] == pytest.approx(42.75)
    assert row["actual_runtime_s"] == pytest.approx(42.75)
    # Published experiment scores retain their fixed penalty for failed tasks.
    assert row["time_s"] == pytest.approx(42.75 if success else controller.config["task"]["total_timeout_s"])
    navigation_entry = controller.config["entry_stage"] == "navigation"
    assert row["progress"] == (1.0 if success else None)
    assert row["nav_success"] is (True if success and navigation_entry else None)
    missing = [] if success else ["progress"] + (["nav_success"] if navigation_entry else [])
    assert row["missing_annotations"] == missing
    assert not row["intervention"] and row["align_attempts"] == 0
    assert shown[-1][0][1] == ("TASK_COMPLETED" if success else "TASK_FAILED")

    # Neither repeated nor opposite result keys can overwrite a finished trial.
    for repeated_key in ("g", "h"):
        ControlGatewayRuntime._handle_complete_agent_result(gateway, core.accept_console_line(repeated_key).command)
    assert len(sent) == 1
    assert not nav.accept_status(dict(generation=generation, skill_id=3, state="failed"), owner="lavira")
    assert not nav.accept_vla_command("start_vla_task", dict(generation=generation, skill_id=3))
    controller.finish(generation, "failed", "late_failure")
    controller.finish(generation, "reached", "late_success")
    assert read_events(controller.recorder.path) == events
    clock[0] = event_clock[0] = 50.0
    restarted = nav.handle_key("n", now=50.0)
    controller.start(restarted.generation)
    controller.finish(generation, "failed", "late_failure")
    ControlGatewayRuntime._handle_complete_agent_result(gateway, command)
    assert controller.active[0] == restarted.generation and nav.task_started_at == 50.0
    assert len(sent) == 1 and shown[-1][1]["reason"] == "result_before_task_start"


@pytest.mark.parametrize("key,state", [("g", "reached"), ("h", "failed")])
def test_operator_result_also_works_without_experiment_profile(monkeypatch, key, state):
    from gear_sonic.runtime.gateway.control import ControlGatewayCore
    from gear_sonic.runtime.gateway.services.control import ControlGatewayRuntime

    actions, events = [], []
    gateway = SimpleNamespace(
        profile=SimpleNamespace(components={}), navigation=NavigationControlState(),
        publish_navigation_action=lambda action, **kw: actions.append(action),
        show_event=lambda *a, **kw: events.append(kw),
        show_command=lambda *a, **kw: events.append(kw),
        _require_source=ControlGatewayRuntime._require_source,
    )
    gateway.experiment = ExperimentSupervisor(gateway)
    monkeypatch.setattr("time.monotonic", lambda: 17.0)
    monkeypatch.setattr("time.monotonic_ns", lambda: 17_000_000_000)
    monkeypatch.setattr("time.time_ns", lambda: 1_017_000_000_000)
    command = ControlGatewayCore().accept_console_line(key).command
    ControlGatewayRuntime._handle_complete_agent_result(gateway, command)
    assert not actions and events[-1]["reason"] == "no_active_agent"
    gateway.navigation.handle_key("n", now=10.0)

    ControlGatewayRuntime._handle_complete_agent_result(gateway, command)

    assert actions[-1].agent_event == "cancel_navigation"
    assert events[-1]["state"] == state and events[-1]["completion_time_s"] == 7.0


def test_operator_failure_preserves_partial_progress_and_navigation_labels(monkeypatch, tmp_path):
    from gear_sonic.experiments.results import trial_rows

    monkeypatch.setattr("time.monotonic_ns", lambda: 0)
    controller, _, _ = supervisor(monkeypatch, tmp_path, "full_vln")
    generation = controller.active[0]
    controller.recorder.write("session", experiment=controller.config)
    controller.recorder.write("annotation", generation=generation, values=dict(progress=1, nav_success=True))
    monkeypatch.setattr("time.monotonic_ns", lambda: 42_750_000_000)

    controller.confirm_result(
        generation, success=False, completion_time_s=42.75,
        completed_wall_time_ns=1_042_750_000_000, completed_monotonic_ns=42_750_000_000,
    )

    row = trial_rows(read_events(controller.recorder.path))[0]
    assert row["success"] is False and row["nav_success"] is True
    assert row["progress"] == 1 / len(controller.config["task"]["milestones"])
    assert row["actual_runtime_s"] == row["completion_time_s"] == 42.75
    assert not row["missing_annotations"]


def test_fixed_vla_deadline_starts_on_action_and_resume_does_not_reset(monkeypatch, tmp_path):
    controller, clock, sent = supervisor(monkeypatch, tmp_path, "gate_completion_off")
    gen = controller.active[0]
    controller.gateway.navigation.mode = "lavira_manipulate"
    controller.gateway.navigation.skill_id = 3
    clock[0] = 10.0
    controller.first_action({"generation": gen, "skill_id": 3})
    clock[0] = 90.0
    controller.first_action({"generation": gen, "skill_id": 3})
    assert controller.vla_deadline == 190.0
    clock[0] = 190.0
    controller.tick()
    assert controller.active is None
    end = next(e for e in read_events(controller.recorder.path) if e["type"] == "trial_end")
    assert end["reason"] == "fixed_duration" and end["state"] == "completion_candidate"


def test_vla_deadline_uses_action_time_not_delayed_notification(monkeypatch, tmp_path):
    controller, clock, _ = supervisor(monkeypatch, tmp_path, "gate_completion_off")
    controller.gateway.navigation.mode = "lavira_manipulate"
    controller.gateway.navigation.skill_id = 1
    clock[0] = 20.0
    controller.first_action(dict(generation=controller.active[0], skill_id=1, action_monotonic_ns=10_000_000_000))
    assert controller.vla_deadline == 190.0


def test_motion_watchdog_closes_experiment_log(monkeypatch, tmp_path):
    from gear_sonic.runtime.gateway.services.control import ControlGatewayRuntime

    controller, _, sent = supervisor(monkeypatch, tmp_path)
    nav = controller.gateway.navigation
    nav.mode = "base_pose_motion"
    nav.manual_deadline = 1.0
    ControlGatewayRuntime._handle_timeout(controller.gateway)
    assert controller.active is None
    assert (
        next(e for e in read_events(controller.recorder.path) if e["type"] == "trial_end")["reason"]
        == "base_pose_velocity_timeout"
    )


def test_repeat_start_creates_distinct_trials_and_closes_old(monkeypatch, tmp_path):
    controller, _, _ = supervisor(monkeypatch, tmp_path)
    gen = controller.active[0]
    controller.start(gen + 1)
    events = read_events(controller.recorder.path)
    assert len({e["trial_id"] for e in events if e["type"] == "trial_start"}) == 2
    assert next(e for e in events if e["type"] == "trial_end")["reason"] == "restarted_with_n"


def test_shadow_completion_can_be_cancelled_while_va_is_blocked(monkeypatch, tmp_path):
    agent, _, _, intents = make_agent(monkeypatch, tmp_path, "gate_completion_off")
    agent.generation, agent._skill_id = 1, 1
    cancelled, entered, release, returned = (threading.Event() for _ in range(4))
    agent.cancelled = lambda generation: cancelled.is_set()

    def postcheck(**kwargs):
        assert np.mean(kwargs["image_bgr"]) > 100, "Shadow completion checks must also use the head view"
        entered.set()
        release.wait(2.0)
        return helpers.task_complete()

    agent.client.client.postcheck = postcheck

    def execute():
        try:
            agent._manipulate(1, 1, "physical condition", "physical task")
        except helpers.LaViRAAgentError:
            returned.set()

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    assert entered.wait(2.0)
    cancelled.set()
    assert returned.wait(1.0), "Blocked VA must not block cancellation"
    release.set()
    thread.join(1.0)
    assert len([1 for name, _ in intents if name == "start_vla_task"]) == 1


def test_navila_discrete_action_uses_gateway_and_stops_at_deadline(monkeypatch, tmp_path):
    from gear_sonic.experiments.navila import parse_action
    from gear_sonic.runtime.protocol import decode_navigation_message
    from gear_sonic.runtime.protocol.navigation import NavigationRuntimeStatus

    controller, clock, sent = supervisor(monkeypatch, tmp_path, "navila_basepose_vla")
    gen = controller.active[0]
    velocity, duration = parse_action(
        dict(type="navila_action", version=1, sequence=1, text="Move forward 50 cm."),
        1,
        controller.config["navila"],
    )
    assert velocity == (0.4, 0.0, 0.0) and duration == pytest.approx(1.25)
    step = dict(generation=gen, skill_id=1, segment_id=1, velocity=velocity, duration_s=duration)
    controller.begin_step(step)
    controller.tick()
    assert decode_navigation_message(sent[-1]).velocity == (0.4, 0.0, 0.0)
    clock[0] = 1.20
    controller.tick()
    assert controller.step is not None
    clock[0] = 1.25
    controller.tick()
    assert sent[-1][0] == "stop" and controller.step is not None
    assert controller.gateway.navigation.mode == "lavira_nav"
    assert controller.accept_executor_status(NavigationRuntimeStatus(
        generation=gen, skill_id=1, segment_id=1, timestamp=time.time(),
        mode="stop", source="stop", requested_velocity=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0), reason="stopped",
    ))
    assert sent[-1][0] == "navigation_status" and controller.step is None
    assert controller.gateway.navigation.mode == "lavira_pending"
    with pytest.raises(ValueError, match="Stale"):
        controller.begin_step(step)
    for velocity in ([0, 0.2, 0], [0.41, 0, 0], [-0.31, 0, 0], [0, 0, 0.31]):
        with pytest.raises(ValueError, match="envelope"):
            controller.begin_step(dict(generation=gen, skill_id=1, segment_id=2, velocity=velocity, duration_s=1.0))


def test_operator_repeat_n_cancels_old_generation_and_perturb_requires_cue(monkeypatch, tmp_path):
    from gear_sonic.runtime.gateway.control import ControlGatewayCore
    from gear_sonic.runtime.gateway.services.control import ControlGatewayRuntime

    controller, clock, sent = supervisor(monkeypatch, tmp_path, "gate_nav_off")
    gateway = controller.gateway
    gateway._require_source = ControlGatewayRuntime._require_source
    core = ControlGatewayCore()
    command = core.accept_command("navigation_key", parameters={"key": "n"}).command
    old = controller.active[0]
    ControlGatewayRuntime._handle_navigation_key(gateway, command)
    assert controller.active[0] > old and sent[-2].agent_event == "cancel_navigation"
    assert sent[-1].agent_event == "start_navigation"
    mark = core.accept_command("experiment_perturbation", parameters={}).command
    with pytest.raises(ValueError, match="cue"):
        ControlGatewayRuntime._handle_experiment_perturbation(gateway, mark)
    controller.config["condition"]["perturbation"]["enabled"] = True
    controller.recorder.write("perturbation_cue", generation=controller.active[0], gate="nav")
    ControlGatewayRuntime._handle_experiment_perturbation(gateway, mark)
    with pytest.raises(ValueError, match="already"):
        ControlGatewayRuntime._handle_experiment_perturbation(gateway, mark)
    assert len([e for e in read_events(controller.recorder.path) if e["type"] == "perturbation"]) == 1


@pytest.fixture
def fake_navila(monkeypatch):
    from gear_sonic.experiments.navila import NavilaClient

    class FakeNavila(NavilaClient):
        def __init__(self, config):
            super().__init__(config)
            self.resets = 0
            self.stop_text = "stop"

        def call(self, endpoint, data=None):
            if endpoint == "reset":
                self.resets += 1
            if endpoint in {"ping", "reset"}:
                return {"status": "ok"}
            assert endpoint == "get_action" and data["instruction"] == "NAVIGATION ROUTE"
            return dict(
                type="navila_action",
                version=1,
                sequence=data["sequence"],
                text="The next action is move forward 50 cm." if self.sequence == 1 else self.stop_text,
            )

    monkeypatch.setattr("gear_sonic.experiments.agent.NavilaClient", FakeNavila)


@pytest.mark.parametrize(
    "stop_text",
    [
        "stop",
        "i think i should stop because i have finished the instruction",
        "The next action is I think I should stop because I have finished the instruction.",
    ],
)
def test_navila_agent_executes_navigation_alignment_and_manipulation(monkeypatch, tmp_path, fake_navila, stop_text):
    payload, _ = resolve(CONFIGS / "navila_basepose_vla.yaml", "T1")
    agent, camera, client, intents = make_agent(
        monkeypatch,
        tmp_path,
        "navila_basepose_vla",
        global_target=payload["components"]["lavira"]["global_target"],
    )
    agent.navila.stop_text = stop_text
    cancelled = threading.Event()
    agent.cancelled = lambda generation: cancelled.is_set()
    agent.sleep = lambda seconds: cancelled.set()  # Operator ends the VLA phase.
    camera.capture_camera_aligned_rgbd = lambda **kwargs: pytest.fail("NaVILA stop must skip NAV RGB-D checks")
    result = agent.run(1)
    assert result.state == "stopped" and result.reason == "operator_cancelled", result
    assert agent.navila.resets == 1 and not client.la_calls
    assert agent.navila.sequence == 2 and not client.grounding_calls
    assert agent.global_target == payload["components"]["lavira"]["global_target"]
    assert not client.alignment_grounding_calls and not client.postcheck_calls
    names = [name for name, _ in intents]
    assert names == ["experiment_nav_step", "start_base_pose", "start_vla_task"]
    base_pose = next(args for name, args in intents if name == "start_base_pose")
    assert (base_pose["target"], base_pose["yaw_align_target"]) == ("blue basket", "desk")
    events = read_events(agent.recorder.path)
    navila_responses = [e for e in events if e["type"] == "navila_response"]
    assert [e["response"]["sequence"] for e in navila_responses] == [1, 2]
    assert [e["response"]["text"] for e in navila_responses] == [
        "The next action is move forward 50 cm.", stop_text
    ]
    assert all(e["generation"] == 1 and e["skill_id"] == 1 and e["camera"] == "chest_view" for e in navila_responses)
    nav_gates = [e for e in events if e["type"] == "gate" and e["gate"] == "nav"]
    assert len(nav_gates) == 1
    gate = nav_gates[0]
    assert gate["enabled"] is False and gate["applied"] == "allow"
    assert gate["reason"] == "navila_stop" and gate["va_recommendation"] is None
    assert gate["check_ids"] == [] and gate["result"]["status"] == "SKIPPED"
    assert not any(e["type"] == "check" for e in events)
    gates = [e for e in events if e["type"] == "gate"]
    assert [e["gate"] for e in gates] == ["nav", "align", "completion"]
    assert all(e["enabled"] is False and e["va_recommendation"] is None for e in gates)
    assert gates[1]["controller_aligned"] is True
    assert gates[2]["applied"] == "wait_operator_or_fixed_duration"


@pytest.mark.parametrize("state,reason", [("failed", "target_lost"), ("failed", "timeout"), ("reached", "unverified")])
def test_navila_alignment_failure_never_starts_vla(monkeypatch, tmp_path, fake_navila, state, reason):
    agent, _, client, intents = make_agent(
        monkeypatch,
        tmp_path,
        "navila_basepose_vla",
        global_target="",
    )
    wait = agent.wait_status

    def failed_alignment(*args):
        status = wait(*args)
        if intents[-1][0] == "start_base_pose":
            status.update(state=state, reason=reason)
        return status

    agent.wait_status = failed_alignment
    result = agent.run(1)
    assert result.state == "failed" and result.reason == f"navila_base_pose_failed:{reason}"
    assert agent.global_target == "" and not client.grounding_calls and not client.la_calls
    assert not client.alignment_grounding_calls and not client.postcheck_calls
    assert agent.navila.resets == 1 and agent.navila.sequence == 2
    assert [name for name, _ in intents] == ["experiment_nav_step", "start_base_pose"]
    events = read_events(agent.recorder.path)
    align_gates = [e for e in events if e["type"] == "gate" and e["gate"] == "align"]
    assert [e["applied"] for e in align_gates] == ["block"]
    assert all(e["enabled"] is False for e in align_gates)


def test_navila_waits_for_each_controller_before_next_frame_or_vla(monkeypatch, tmp_path, fake_navila):
    agent, camera, client, intents = make_agent(monkeypatch, tmp_path, "navila_basepose_vla", global_target="")
    nav_waiting, align_waiting, vla_started = (threading.Event() for _ in range(3))
    nav_release, align_release, cancelled = (threading.Event() for _ in range(3))
    wait = agent.wait_status

    def blocking_wait(*args):
        name = intents[-1][0]
        if name == "experiment_nav_step":
            nav_waiting.set()
            assert nav_release.wait(3)
        elif name == "start_base_pose":
            align_waiting.set()
            assert align_release.wait(3)
        elif name == "start_vla_task":
            vla_started.set()
        return wait(*args)

    agent.wait_status = blocking_wait
    agent.cancelled = lambda generation: cancelled.is_set()
    agent.sleep = lambda seconds: cancelled.wait(0.01)
    results = []
    worker = threading.Thread(target=lambda: results.append(agent.run(1)), daemon=True)
    worker.start()
    try:
        assert nav_waiting.wait(2)
        assert agent.navila.sequence == 1 and len(camera.capture_cutoffs) == 1
        assert [name for name, _ in intents] == ["experiment_nav_step"]
        released_ns = time.time_ns()
        nav_release.set()
        assert align_waiting.wait(2)
        assert agent.navila.sequence == 2 and camera.capture_cutoffs[1] >= released_ns
        assert not vla_started.is_set()
        align_release.set()
        assert vla_started.wait(2)
    finally:
        cancelled.set()
        nav_release.set()
        align_release.set()
        worker.join(3)
    assert not worker.is_alive() and results[0].state == "stopped"
    assert not (client.la_calls or client.grounding_calls or client.alignment_grounding_calls or client.postcheck_calls)


@pytest.mark.parametrize("state,reason", [("failed", "controller_error"), ("reached", "unverified")])
def test_navila_bad_step_result_does_not_request_another_image(monkeypatch, tmp_path, fake_navila, state, reason):
    agent, camera, _, intents = make_agent(monkeypatch, tmp_path, "navila_basepose_vla")
    wait = agent.wait_status

    def bad_step(*args):
        return {**wait(*args), "state": state, "reason": reason}

    agent.wait_status = bad_step
    result = agent.run(1)
    assert result.state == "failed" and result.reason == f"navila_step_failed:{reason}"
    assert agent.navila.sequence == 1 and len(camera.capture_cutoffs) == 1
    assert [name for name, _ in intents] == ["experiment_nav_step"]


def test_nearfield_missing_role_retries_alignment_without_navigation(monkeypatch, tmp_path):
    agent, _, _, intents = make_agent(
        monkeypatch,
        tmp_path,
        "near_head",
        alignment_groundings=[helpers.alignment_grounding("NOT_FOUND"), helpers.alignment_grounding()],
        postchecks=[helpers.postcheck("NOT_SATISFIED"), helpers.ready_to_manipulate(), helpers.task_complete()],
    )
    result = agent.run(1)
    assert result.state == "reached", result
    assert not any(name == "navigation_goal" for name, _ in intents)
