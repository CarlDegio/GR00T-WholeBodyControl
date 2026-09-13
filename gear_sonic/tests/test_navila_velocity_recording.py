"""Audit real executor decisions without connecting to robot services."""

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.experiments.recording import read_events
from gear_sonic.experiments.velocity_recording import NavilaVelocityRecorder
from gear_sonic.runtime.protocol import NavigationCommand
from gear_sonic.utils.planner_control import PlannerVelocityExecutorCore, SafetySnapshot


def recorder(tmp_path, backend="navila"):
    return NavilaVelocityRecorder(SimpleNamespace(components={"experiment": dict(
        navigation_backend=backend, run_dir=str(tmp_path), session_id="velocity-test", minimal_logging=True,
    )}))


def begin(core, *, generation=7, skill=2, segment=3, source="navila", mode="manual_velocity", now=10.0):
    assert core.accept_navigation(NavigationCommand(
        mode=mode, generation=generation, skill_id=skill, segment_id=segment,
        source=source, velocity=(0.4, 0.0, 0.0) if mode == "manual_velocity" else None, timestamp=now,
    ), now=now)


def sample(recording, core, now, *, safety=None):
    decision = core.decide(now=now, safety=SafetySnapshot(now) if safety is None else safety)
    recording.record_sent(decision, mode=core.mode, now=now)


@pytest.mark.parametrize("blocked_reason", ["radar_timeout", "depth_hard_stop"])
def test_records_requested_and_sent_velocity_through_safety_and_stop(monkeypatch, tmp_path, blocked_reason):
    monkeypatch.setenv("SONIC_EXPERIMENT_MINIMAL_LOGGING", "1")
    recording, core = recorder(tmp_path), PlannerVelocityExecutorCore()
    begin(core)
    sample(recording, core, 10.0)
    sample(recording, core, 10.05)
    assert len(read_events(recording.recorder.path)) == 1
    sample(recording, core, 10.5)
    safety = SafetySnapshot() if blocked_reason == "radar_timeout" else SafetySnapshot(
        radar_timestamp_s=10.55, depth_m=np.full((50, 50), 0.05),
    )
    sample(recording, core, 10.55, safety=safety)
    sample(recording, core, 10.6)
    sample(recording, core, 10.8)  # Input timeout produces a zero command.
    begin(core, mode="stop", source="", now=10.85)
    sample(recording, core, 10.85)
    sample(recording, core, 12.0)  # Do not keep logging idle after the segment.

    events = read_events(recording.recorder.path)
    assert [e["reason"] for e in events] == [
        "clear", "clear", blocked_reason, "clear", "navila_velocity_timeout", "stopped",
    ]
    assert all(e["type"] == "navila_velocity" and e["trial_id"] == "velocity-test-g7" for e in events)
    assert all((e["skill_id"], e["segment_id"]) == (2, 3) for e in events)
    assert events[0]["sonic"]["movement_direction"] == [1.0, 0.0, 0.0]
    assert events[0]["sonic"]["facing_direction"] == [1.0, 0.0, 0.0]
    assert [e["sonic"]["speed_m_s"] for e in events] == pytest.approx([0.4, 0.4, 0, 0.4, 0, 0])
    assert events[2]["requested_velocity"] == [0.4, 0.0, 0.0]
    assert events[2]["velocity"] == [0.0, 0.0, 0.0]
    assert events[-1]["source"] == "stop"


def test_segment_changes_and_cancel_keep_the_correct_trial(tmp_path):
    recording, core = recorder(tmp_path), PlannerVelocityExecutorCore()
    begin(core)
    sample(recording, core, 10.0)
    begin(core, segment=4, now=10.05)
    sample(recording, core, 10.05)
    begin(core, generation=8, skill=0, segment=0, mode="stop", source="", now=10.1)
    sample(recording, core, 10.1)
    begin(core, generation=8, skill=1, segment=0, now=10.15)
    sample(recording, core, 10.15)

    events = read_events(recording.recorder.path)
    assert [(e["generation"], e["skill_id"], e["segment_id"]) for e in events] == [
        (7, 2, 3), (7, 2, 4), (7, 2, 4), (8, 1, 0),
    ]
    assert events[2]["executor_identity"] == dict(generation=8, skill_id=0, segment_id=0)
    assert events[2]["sonic"]["speed_m_s"] == 0.0


def test_executor_shutdown_records_one_final_zero(tmp_path):
    recording, core = recorder(tmp_path), PlannerVelocityExecutorCore()
    recording.record_shutdown(core.sonic.message((0.0, 0.0, 0.0)))
    assert not recording.recorder.path.exists()
    begin(core)
    sample(recording, core, 10.0)
    message = core.sonic.message((0.0, 0.0, 0.0))
    for _ in range(3):
        recording.record_shutdown(message)
    events = read_events(recording.recorder.path)
    assert len(events) == 2
    assert events[-1]["reason"] == "executor_shutdown"
    assert events[-1]["velocity"] == [0.0, 0.0, 0.0]
    assert events[-1]["sonic"]["speed_m_s"] == 0.0


@pytest.mark.parametrize(
    "backend,source", [("lavira", "navila"), ("navila", "base_pose_agent"), ("navila", "operator_console")]
)
def test_other_backends_and_producers_do_not_create_velocity_records(tmp_path, backend, source):
    recording, core = recorder(tmp_path, backend), PlannerVelocityExecutorCore()
    begin(core, source=source)
    sample(recording, core, 10.0)
    recording.record_shutdown(core.sonic.message((0.0, 0.0, 0.0)))
    assert not recording.recorder.path.exists()
