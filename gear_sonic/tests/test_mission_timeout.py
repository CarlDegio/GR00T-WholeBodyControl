"""Per-operation time limits exercise real mission/gateway/VLA handoffs."""

from dataclasses import replace
import threading

import pytest

from gear_sonic.tests.test_lavira_agent import postcheck, ready_to_manipulate, task_complete
from gear_sonic.tests.test_multi_skill_mission import explicit_harness
from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.utils.inference.lavira.mission import parse_mission_steps, validate_mission_profile


def timed_harness(monkeypatch, *, on_timeout="continue", handoff="ready"):
    h = explicit_harness(monkeypatch, handoff=handoff)
    h.agent.mission_steps = (
        replace(h.agent.mission_steps[0], timeout_seconds=40, on_timeout=on_timeout),
        *h.agent.mission_steps[1:3],
    )
    h.agent.manipulation_window_seconds = 5
    h.agent.manipulation_max_windows = 36
    h.agent.monotonic = lambda: h.now[0]
    h.agent.sleep = lambda seconds: h.now.__setitem__(0, h.now[0] + seconds)
    capture = h.camera.capture_rgb
    h.camera.capture_rgb = lambda *, camera_stream="chest_view", timeout_ms=None: capture(camera_stream=camera_stream)
    h.client.postchecks = iter([ready_to_manipulate()] * 4)
    original_postcheck = h.client.postcheck
    h.first_verdict = lambda: postcheck("NOT_SATISFIED")

    def check(**kwargs):
        if kwargs.get("skill") == "MANIPULATE":
            return h.first_verdict() if kwargs["mission"] == "perform a" else task_complete()
        return original_postcheck(**kwargs)

    h.client.postcheck = check
    original_start = h.agent._start_manipulation
    h.starts = []

    def start(*args):
        original_start(*args)
        h.starts.append((h.now[0], dict(h.agent.vla_task_parameters)))

    h.agent._start_manipulation = start
    original_submit = h.agent.submit_intent
    h.stops = []

    def submit(name, params):
        if name == "stop_vla_task":
            h.stops.append((h.now[0], dict(params)))
        return original_submit(name, params)

    h.agent.submit_intent = submit
    original_record = h.agent._record
    h.history = []

    def record(entry):
        h.history.append(entry)
        original_record(entry)

    h.agent._record = record
    return h


def test_first_skill_expires_at_40_seconds_then_navigates_and_runs_second(monkeypatch):
    h = timed_harness(monkeypatch)
    result = h.agent.run(1)
    assert result.state == "reached", result.reason
    assert result.reason == "mission_completed_with_timeouts"
    assert h.selected == ["pill", "rubbish"]
    assert h.stops[0][0] - h.starts[0][0] == pytest.approx(40)
    assert h.stops[0][1]["reason"] == "timeout_continue"
    assert h.stops[0][1]["completion_settle_seconds"] == 0
    assert "timeout_seconds" not in h.starts[1][1]
    assert h.snapshots[1][1:] == (True, True, False)  # Hold arm posture and closed left gripper.
    first = next(entry for entry in h.history if entry.skill == "MANIPULATE")
    assert (first.controller_state, first.va_result) == ("timeout_continued", "UNKNOWN")
    assert any(event["code"] == "MISSION_STEP_TIMEOUT_CONTINUED" for event in h.events)
    assert not any(event["code"] == "MISSION_STEP_COMPLETED" and event["fields"]["mission_step"] == 1
                   for event in h.events)
    assert not h.terminal  # No experiment success annotation.


def test_visual_success_advances_early_without_timeout_label(monkeypatch):
    h = timed_harness(monkeypatch)
    h.first_verdict = task_complete
    result = h.agent.run(1)
    assert result.reason == "mission_completed"
    assert h.stops[0][0] - h.starts[0][0] == pytest.approx(5)
    assert h.stops[0][1]["reason"] == "postcondition_satisfied"
    assert h.selected == ["pill", "rubbish"]
    assert not any(event["code"] == "SKILL_TIMEOUT_CONTINUING" for event in h.events)


def test_timeout_defaults_to_failure_and_does_not_run_next_skill(monkeypatch):
    h = timed_harness(monkeypatch, on_timeout="fail")
    result = h.agent.run(1)
    assert (result.state, result.reason) == ("failed", "manipulate_timeout")
    assert h.selected == ["pill"]
    assert h.stops[0][1]["reason"] == "timeout"


@pytest.mark.parametrize("handoff", ["timeout", "cancel"])
def test_timeout_continue_still_requires_planner_ack_and_honors_cancel(monkeypatch, handoff):
    h = timed_harness(monkeypatch, handoff=handoff)
    result = h.agent.run(1)
    assert result.state != "reached"
    assert h.selected == ["pill"]


@pytest.mark.parametrize("fault", ["cancel", "safety"])
def test_fault_while_va_is_pending_cannot_become_timeout_continue(monkeypatch, fault):
    h = timed_harness(monkeypatch)
    release = threading.Event()
    returned = threading.Event()

    def blocked():
        if fault == "cancel":
            h.agent.cancelled = lambda _: True
        else:
            h.agent.poll_failure = lambda *args: {"reason": "radar_timeout"}
        h.now[0] = h.starts[0][0] + 40
        release.wait(2)
        returned.set()
        return task_complete()

    h.first_verdict = blocked
    try:
        result = h.agent.run(1)
        assert result.state == ("stopped" if fault == "cancel" else "failed")
        assert h.selected == ["pill"]
        assert not any(params["reason"] == "timeout_continue" for _, params in h.stops)
    finally:
        release.set()
        assert returned.wait(2)


def test_slow_va_cannot_extend_deadline_or_publish_late_success(monkeypatch):
    h = timed_harness(monkeypatch)
    release = threading.Event()
    returned = threading.Event()

    def blocked():
        h.now[0] = h.starts[0][0] + 40
        release.wait(2)
        returned.set()
        return task_complete("late first-operation success")

    h.first_verdict = blocked
    try:
        result = h.agent.run(1)
        assert result.reason == "mission_completed_with_timeouts"
        assert h.selected == ["pill", "rubbish"]
        assert h.stops[0][0] - h.starts[0][0] == pytest.approx(40)
    finally:
        release.set()
        assert returned.wait(2)
    assert not any(event["fields"].get("visual_evidence") == "late first-operation success" for event in h.events)


@pytest.mark.parametrize("changes", [
    {"timeout_seconds": 0}, {"timeout_seconds": -1}, {"timeout_seconds": True},
    {"timeout_seconds": float("nan")}, {"timeout_seconds": float("inf")},
    {"timeout_seconds": "40"}, {"on_timeout": []}, {"on_timeout": "skip"},
    {"timeout_seconds": None, "on_timeout": "continue"},
    {"completion": {"kind": "manual"}},
])
def test_invalid_per_skill_timeouts_are_rejected(changes):
    step = dict(id="a", type="skill", skill_key="model", manipulation_prompt="pick",
                expected_postcondition="bag held", align={"target": "bag"}, exit="hold",
                timeout_seconds=40)
    with pytest.raises(ValueError):
        parse_mission_steps([{**step, **changes}], backends={"model": {}})


def test_bag_bin_yaml_resolves_one_whole_mission_and_fixed_vla_prompts(tmp_path):
    from gear_sonic.scripts.prepare_mission import prepare_mission
    from gear_sonic.utils.inference.lavira.service import load_lavira_config

    output = prepare_mission(tmp_path / "mission.yaml",
                             overlays=["gear_sonic/config/missions/bag_then_bin.yaml"])
    profile = load_runtime_profile(output)
    assert validate_mission_profile(profile) == ()
    config = load_lavira_config(str(output))
    assert config.flexible_mission and config.mission_steps == [] and config.global_target == ""
    skills = config.skills
    assert (skills["pick_bag"]["timeout_seconds"], skills["pick_bag"]["on_timeout"]) == (40, "continue")
    assert skills["pick_bag"]["completion_settle_seconds"] == 0
    assert "timeout_seconds" not in skills["step_bin"]
    assert skills["pick_bag"]["vla_trained_prompt"] == "collect the pill bottle"
    assert skills["step_bin"]["vla_trained_prompt"] == "collect the pill bottle"
    for skill in skills.values():
        assert profile.component("vla")["backends"][skill["backend"]]["prompt"] == skill["vla_trained_prompt"]
    assert config.manipulation_timeout_seconds == 180
