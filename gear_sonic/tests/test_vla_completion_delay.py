"""Exercise VA completion through the gateway and VLA without robot sockets."""

from types import SimpleNamespace

import pytest

from gear_sonic.experiments.recording import read_events
from gear_sonic.experiments.results import trial_rows
from gear_sonic.experiments.supervisor import ExperimentSupervisor
from gear_sonic.runtime.gateway.control import ControlGatewayCore, NavigationControlState
from gear_sonic.runtime.gateway.services.control import ControlGatewayRuntime
from gear_sonic.tests.test_experiments_runtime import experiment
from gear_sonic.utils.inference.vla import runtime as vla_runtime


def completion_runtime(monkeypatch, tmp_path, *, experimental=True):
    clock = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("time.monotonic_ns", lambda: round(clock[0] * 1e9))
    monkeypatch.setattr("time.time_ns", lambda: round((1000.0 + clock[0]) * 1e9))
    monkeypatch.setattr("gear_sonic.experiments.snapshots.record_pose_async", lambda *a, **k: None)
    gateway = object.__new__(ControlGatewayRuntime)
    components = {"experiment": experiment(tmp_path, "full_vln")} if experimental else {}
    gateway.profile = SimpleNamespace(components=components)
    gateway.navigation = NavigationControlState()
    generation = gateway.navigation.handle_key("n", now=clock[0]).generation
    gateway.navigation.mode = "lavira_manipulate"
    gateway.navigation.skill_id = 3
    gateway.navigation.segment_id = 10
    gateway.pending_vla_completion = None
    gateway.command_handlers = {"navigation_key": gateway._handle_navigation_key}
    gateway.show_command = gateway.show_event = lambda *a, **k: None
    dispatched, cpp_commands, navigation_messages = [], [], []
    gateway.navigation_pub = SimpleNamespace(send_string=navigation_messages.append)
    gateway.experiment = ExperimentSupervisor(gateway, clock=lambda: clock[0])
    if experimental:
        gateway.experiment.recorder.write("session", experiment=components["experiment"])
    gateway.experiment.start(generation)
    state = vla_runtime._VlaRuntimeState(
        task_active=True, task_generation=generation, task_skill_id=3,
        cpp_loop_running=True, cpp_mode="POSE", pause_loop=False,
    )
    handler = object.__new__(vla_runtime._VlaCommandHandler)
    handler.state = state
    handler.invalidate_inference = lambda _reason: None

    def cpp_control(*, start, planner):
        cpp_commands.append((clock[0], start, planner))
        state.cpp_loop_running = start
        state.cpp_mode = "PLANNER" if planner else "POSE"

    handler.send_cpp_control_command = cpp_control

    def dispatch(name, parameters):
        dispatched.append((clock[0], name, dict(parameters)))
        if name in handler.TASK_COMMANDS:
            handler._handle_task_command(SimpleNamespace(name=name, parameters=parameters))

    gateway.dispatch_navigation_event = dispatch
    agent_core = ControlGatewayCore(source="lavira_agent")

    def agent_command(name, **parameters):
        return agent_core.accept_command(
            name, parameters=dict(generation=generation, skill_id=3, **parameters),
        ).command

    operator_core = ControlGatewayCore()

    def operator_command(name, **parameters):
        return operator_core.accept_command(name, parameters=parameters).command

    return SimpleNamespace(
        gateway=gateway, clock=clock, state=state, dispatched=dispatched,
        cpp_commands=cpp_commands, generation=generation,
        agent_command=agent_command, operator_command=operator_command,
    )


def confirm_va(h, *, publish_result=True):
    h.gateway._handle_vla_task(h.agent_command(
        "stop_vla_task", window_id=13, reason="postcondition_satisfied",
    ))
    if publish_result:
        h.gateway._handle_navigation_agent_status(h.agent_command(
            "navigation_agent_status", state="reached", reason="manipulation_completed", segment_id=10,
        ))


@pytest.mark.parametrize("experimental", [False, True])
@pytest.mark.parametrize("publish_result", [False, True])
def test_va_completion_runs_vla_five_more_seconds_and_keeps_task_time(
    monkeypatch, tmp_path, experimental, publish_result,
):
    h = completion_runtime(monkeypatch, tmp_path, experimental=experimental)
    recorder = h.gateway.experiment.recorder
    if experimental:
        # Both original budgets would expire during the completion interval.
        h.gateway.experiment.active = (h.generation, 101.0)
        h.gateway.experiment.vla_deadline = 100.5
    h.clock[0] = 100.0
    confirm_va(h, publish_result=publish_result)
    assert h.gateway.experiment.active is None
    assert h.gateway.navigation.mode == "lavira_completing"
    assert h.gateway.pending_vla_completion.deadline == 105.0
    assert h.state.task_active and not h.state.pause_loop and not h.state.task_stream_hold_active
    assert h.state.cpp_mode == "POSE" and not h.cpp_commands
    assert all(name not in {"stop_vla_task", "cancel_navigation"} for _, name, _ in h.dispatched)
    if experimental:
        end_before = next(e for e in read_events(recorder.path) if e["type"] == "trial_end")
        assert end_before["monotonic_ns"] == 100_000_000_000
        assert end_before["reason"] == "manipulation_completed"
        # Independent labels remain independent, with their original time.
        recorder.write("annotation", generation=h.generation, values={
            "success": True, "progress": 2, "completion_time_s": 99.0, "nav_success": True,
        })
        row_before = trial_rows(read_events(recorder.path))[0]

    for now in (100.5, 101.0, 104.999):
        h.clock[0] = now
        h.gateway._handle_timeout()
        assert h.state.cpp_mode == "POSE" and h.state.task_active and not h.state.pause_loop
        assert not h.cpp_commands

    h.clock[0] = 105.0
    h.gateway._handle_timeout()
    assert h.cpp_commands == [(105.0, True, True)]
    assert not h.state.task_active and h.state.pause_loop
    assert h.gateway.navigation.mode == "listen_wasd"
    assert h.gateway.pending_vla_completion is None
    assert [name for _, name, _ in h.dispatched][-2:] == ["stop_vla_task", "cancel_navigation"]
    h.clock[0] = 110.0
    h.gateway._handle_timeout()
    assert len(h.cpp_commands) == 1
    if experimental:
        events = read_events(recorder.path)
        assert [e for e in events if e["type"] == "trial_end"] == [end_before]
        assert trial_rows(events)[0] == row_before
        assert row_before["actual_runtime_s"] == 100.0 and row_before["completion_time_s"] == 99.0
        stop = next(e for e in events if e["type"] == "vla_command")
        assert stop["monotonic_ns"] == 105_000_000_000


@pytest.mark.parametrize("key", [" ", "n"])
def test_operator_interrupts_delay_and_old_timer_cannot_stop_new_trial(monkeypatch, tmp_path, key):
    h = completion_runtime(monkeypatch, tmp_path)
    h.clock[0] = 100.0
    confirm_va(h)
    old_end = next(e for e in read_events(h.gateway.experiment.recorder.path) if e["type"] == "trial_end")
    h.clock[0] = 101.0
    h.gateway._handle_navigation_key(h.operator_command("navigation_key", key=key))
    assert h.cpp_commands == [(101.0, True, True)]
    assert h.gateway.pending_vla_completion is None
    if key == " ":
        h.gateway._handle_navigation_key(h.operator_command("navigation_key", key="n"))
    new_generation = h.gateway.navigation.generation
    assert new_generation > h.generation
    assert h.gateway.experiment.active[0] == new_generation
    dispatch_count = len(h.dispatched)
    h.clock[0] = 106.0
    h.gateway._handle_timeout()
    assert len(h.dispatched) == dispatch_count
    assert h.gateway.navigation.generation == new_generation
    assert h.gateway.navigation.mode == "lavira_pending"
    with pytest.raises(ValueError, match="stale"):
        h.gateway._handle_navigation_agent_status(h.agent_command(
            "navigation_agent_status", state="reached", reason="manipulation_completed", segment_id=10,
        ))
    assert [e for e in read_events(h.gateway.experiment.recorder.path) if e["type"] == "trial_end"] == [old_end]


def test_repeated_completion_does_not_restart_five_second_timer(monkeypatch, tmp_path):
    h = completion_runtime(monkeypatch, tmp_path)
    h.clock[0] = 100.0
    confirm_va(h)
    h.clock[0] = 102.0
    with pytest.raises(ValueError, match="stale"):
        confirm_va(h)
    assert h.gateway.pending_vla_completion.deadline == 105.0


@pytest.mark.parametrize("name", ["select_planner_mode", "toggle_policy_pause", "toggle_control_loop"])
def test_operator_mode_control_interrupts_delay_immediately(monkeypatch, tmp_path, name):
    h = completion_runtime(monkeypatch, tmp_path)
    h.clock[0] = 100.0
    confirm_va(h)
    h.clock[0] = 101.0
    h.gateway._dispatch_command(h.operator_command(name))
    assert h.gateway.pending_vla_completion is None
    assert h.cpp_commands == [(101.0, True, True)]


@pytest.mark.parametrize("reason", ["timeout", "window_limit", "postcheck_unknown", "radar_timeout"])
def test_noncompletion_stops_remain_immediate(monkeypatch, tmp_path, reason):
    h = completion_runtime(monkeypatch, tmp_path)
    h.clock[0] = 100.0
    h.gateway._handle_vla_task(h.agent_command("stop_vla_task", window_id=13, reason=reason))
    assert h.gateway.pending_vla_completion is None
    assert h.cpp_commands == [(100.0, True, True)]
    assert not h.state.task_active


def test_vla_safety_can_stop_during_completion_delay_without_changing_result(monkeypatch, tmp_path):
    h = completion_runtime(monkeypatch, tmp_path)
    h.clock[0] = 100.0
    confirm_va(h)
    end = next(e for e in read_events(h.gateway.experiment.recorder.path) if e["type"] == "trial_end")
    h.clock[0] = 101.0
    monkeypatch.setattr(vla_runtime, "_current_vla_safety_reason", lambda *a, **k: "radar_timeout")
    mode_switches = []
    vla_runtime._enforce_active_task_safety(
        h.state, vla_safety_gate=None, vla_safety_monitor=None,
        invalidate_inference=lambda *_: None,
        send_cpp_control_command=lambda **kw: mode_switches.append(kw),
        task_status_intent=SimpleNamespace(send=lambda *a, **k: None),
        record_event=lambda *a, **k: None,
    )
    assert not h.state.task_active and h.state.pause_loop
    assert mode_switches == [{"start": True, "planner": True}]
    assert [e for e in read_events(h.gateway.experiment.recorder.path) if e["type"] == "trial_end"] == [end]
