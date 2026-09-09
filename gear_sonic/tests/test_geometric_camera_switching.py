"""Exercise camera switching using fresh RGB-D frames and fake YOLO detections."""

from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gear_sonic.experiments import base_pose
from gear_sonic.experiments.config import CONFIGS, resolve
from gear_sonic.experiments.recording import Recorder, read_events
from gear_sonic.tests import test_lavira_agent as helpers
from gear_sonic.tests.test_experiments_runtime import experiment, make_agent

HEAD, CHEST = "ego_view", "chest_view"


@pytest.fixture
def perception(monkeypatch):
    frames = defaultdict(deque)
    calls, resets, holds = [], [], []
    calibrations = {HEAD: object(), CHEST: object()}

    class Camera:
        def capture_stream(self, stream, **kwargs):
            calls.append(stream)
            if not frames[stream]:
                raise TimeoutError("no new frame")
            return frames[stream].popleft()

    class Tracker:
        last_stream = None

        def reset_tracking(self):
            resets.append(self.last_stream)
            self.last_stream = None

        def track(self, frame):
            assert self.last_stream in (None, frame.stream), "Tracker state leaked across camera views"
            self.last_stream = frame.stream
            if not frame.found:
                return []
            return [SimpleNamespace(track_id=1, class_index=0, confidence=0.9, bbox_xyxy=(10, 20, 30, 40))]

    def observe(snap, instance, yaw, calibration, **kwargs):
        assert calibration is calibrations[snap.stream]
        assert kwargs["include_yaw_align_geometry"] is False
        return SimpleNamespace(target=SimpleNamespace(forward_m=snap.forward, right_m=snap.right))

    tracker = Tracker()
    monkeypatch.setattr(base_pose, "dual_calibrations_from_config", lambda config: calibrations)
    monkeypatch.setattr(base_pose, "tracker_for", lambda *args: tracker)
    monkeypatch.setattr(base_pose, "_observation", observe)
    config = SimpleNamespace(dual_head_camera_stream=HEAD, dual_chest_camera_stream=CHEST)
    value = base_pose.GeometricPerception(config, Camera(), "blue basket")

    def feed(stream, stamp, found, forward=2.0, right=0.0):
        frame = SimpleNamespace(stream=stream, timestamp=stamp, found=found, forward=forward, right=right)
        frame.rgb = frame
        frames[stream].append(frame)

    def step():
        return value.step(hold=lambda: holds.append(len(calls)))

    return SimpleNamespace(
        value=value, feed=feed, step=step, calls=calls, resets=resets, holds=holds, tracker=tracker,
    )


def test_initial_head_preference(perception):
    p = perception
    p.feed(HEAD, 1, True)
    p.feed(CHEST, 1, True)
    sample = p.step()
    assert sample.camera_stream == HEAD and sample.forward == 2.0
    assert p.calls == [HEAD] and not p.holds


@pytest.mark.parametrize("head_has_frame", [True, False])
def test_initial_chest_fallback_and_keep_chest_while_visible(perception, head_has_frame):
    p = perception
    if head_has_frame:
        p.feed(HEAD, 1, False)
    p.feed(CHEST, 1, True)
    sample = p.step()
    assert sample.camera_stream == CHEST and sample.both_lost_frames == 0
    assert p.holds == [1], "Old motion must be held before reading the fallback camera"
    p.feed(HEAD, 2, True)
    p.feed(CHEST, 2, True)
    assert p.step().camera_stream == CHEST
    assert p.calls == [HEAD, CHEST, CHEST]


def test_switch_both_directions_on_loss_and_reset_tracker(perception):
    p = perception
    p.feed(HEAD, 1, True)
    assert p.step().camera_stream == HEAD
    p.feed(HEAD, 2, False)
    p.feed(CHEST, 1, True)
    assert p.step().camera_stream == CHEST
    p.feed(CHEST, 2, False)
    p.feed(HEAD, 3, True)
    assert p.step().camera_stream == HEAD
    assert len(p.resets) == 3
    assert p.holds == [2, 4]


def test_twentieth_fresh_pair_ends_alignment(perception):
    p = perception
    for stamp in range(1, 21):
        p.feed(HEAD, stamp, False)
        p.feed(CHEST, stamp, False)
        sample = p.step()
        assert sample.both_lost_frames == stamp
        assert sample.reason == ("geometric_target_lost" if stamp == 20 else None)
        assert sample.forward is None and sample.right is None


def test_reacquisition_resets_joint_loss_count(perception):
    p = perception
    for stamp in range(1, 20):
        p.feed(HEAD, stamp, False)
        p.feed(CHEST, stamp, False)
        assert p.step().reason is None
    p.feed(HEAD, 20, False)
    p.feed(CHEST, 20, True)
    assert p.step().both_lost_frames == 0
    p.feed(HEAD, 21, False)
    p.feed(CHEST, 21, False)
    sample = p.step()
    assert sample.both_lost_frames == 1 and sample.reason is None


def test_repeated_and_unavailable_frames_do_not_count_as_new_losses(perception):
    p = perception
    for _ in range(25):
        p.feed(HEAD, 1, False)
        p.feed(CHEST, 1, False)
        sample = p.step()
        assert sample.both_lost_frames == 1 and sample.reason is None
    for _ in range(25):
        sample = p.step()
        assert sample.both_lost_frames == 1 and sample.reason is None


def test_one_camera_missing_frames_cannot_fake_twenty_joint_losses(perception):
    p = perception
    for stamp in range(1, 25):
        p.feed(HEAD, stamp, False)
        sample = p.step()
        assert sample.both_lost_frames == 0 and sample.reason is None


def test_invalid_depth_uses_other_camera_and_records_detection_separately(perception):
    p = perception
    p.feed(HEAD, 1, True, forward=float("nan"))
    p.feed(CHEST, 1, True, forward=1.7)
    sample = p.step()
    assert sample.camera_stream == CHEST and sample.forward == 1.7
    assert sample.views[HEAD]["target_visible"] is True
    assert sample.views[HEAD]["usable"] is False
    assert sample.views[CHEST]["usable"] is True


def test_model_failure_is_not_treated_as_target_loss(perception, monkeypatch):
    p = perception
    p.feed(HEAD, 1, True)

    def broken_model(_frame):
        raise RuntimeError("model failure")

    monkeypatch.setattr(p.tracker, "track", broken_model)
    with pytest.raises(RuntimeError, match="model failure"):
        p.step()
    assert p.value.both_lost_frames == 0


def test_switch_resets_alignment_phase_and_stability_without_extending_timeout(tmp_path):
    controller = base_pose.GeometricController(experiment(tmp_path)["geometric"], 0.0)
    controller.select_camera(HEAD)
    for stamp in (1, 2):
        controller.update(1.1, 0.0, (HEAD, stamp), 1.0)
    assert controller.stable == 2 and controller.phase == "fine"
    controller.select_camera(CHEST)
    assert controller.stable == 0 and controller.last_stamp is None
    assert controller.update(1.1, 0.3, (CHEST, 2), 2.0) == (0.0, 0.0, -0.3)
    controller.update(None, None, None, 60.0)
    assert controller.reason == "geometric_timeout"


@pytest.mark.parametrize("va_status", ["SATISFIED", "NOT_SATISFIED", "UNKNOWN"])
def test_joint_loss_checks_both_views_before_vla_even_if_va_fails(monkeypatch, tmp_path, va_status):
    transition = {
        "SATISFIED": "READY_TO_MANIPULATE", "NOT_SATISFIED": "RETRY_ALIGN", "UNKNOWN": "UNKNOWN",
    }[va_status]
    agent, _, client, intents = make_agent(
        monkeypatch, tmp_path, "geometric_vla", [helpers.move(), helpers.align()],
        groundings=[helpers.grounding()] * 3,
        postchecks=[helpers.postcheck(va_status, transition=transition)] * 2 + [helpers.task_complete()],
    )
    original_wait = agent.wait_status
    original_postcheck = client.postcheck

    def wait(*args):
        if intents[-1][0] == "start_base_pose":
            return dict(
                generation=args[0], skill_id=args[1], segment_id=args[2], state="failed",
                reason="geometric_target_lost", both_lost_frames=20,
            )
        return original_wait(*args)

    def postcheck(**kwargs):
        if kwargs.get("skill") != "MANIPULATE":
            assert not any(name == "start_vla_task" for name, _ in intents), "ALIGN VA must finish before VLA"
        return original_postcheck(**kwargs)

    agent.wait_status = wait
    monkeypatch.setattr(client, "postcheck", postcheck)
    result = agent.run(7)
    assert result.state == "reached", result
    assert any(name == "start_vla_task" for name, _ in intents)
    assert len(client.la_calls) == 2
    assert [call.get("skill") for call in client.postcheck_calls] == [None, None, "MANIPULATE"]
    events = read_events(agent.recorder.path)
    checks = [e for e in events if e["type"] == "check" and e["stage"] == "ALIGN"]
    assert [e["camera"] for e in checks] == ["chest", "head"]
    assert all(e["result"]["status"] == va_status and Path(e["image"]).is_file() for e in checks)
    gate = next(e for e in events if e["type"] == "gate" and e["gate"] == "align")
    assert gate["controller_aligned"] is False and gate["applied"] == "allow"
    assert gate["controller_state"] == "geometric_target_lost"
    assert gate["enabled"] is False and gate["va_recommendation"] is False
    assert gate["check_ids"] == [e["event_id"] for e in checks]
    assert gate["result"]["status"] == ("UNKNOWN" if va_status == "UNKNOWN" else "NOT_SATISFIED")


@pytest.mark.parametrize("wrong_generation,reason", [(True, "geometric_target_lost"), (False, "hardware_error")])
def test_stale_or_hard_failure_cannot_trigger_loss_handoff(monkeypatch, tmp_path, wrong_generation, reason):
    agent, _, _, intents = make_agent(monkeypatch, tmp_path, "geometric_vla")
    agent.generation = 7
    agent.forced_skill = None
    agent.wait_status = lambda *args: dict(
        generation=args[0] + int(wrong_generation), skill_id=args[1], segment_id=args[2],
        state="failed", reason=reason,
    )
    with pytest.raises(helpers.LaViRAAgentError):
        agent._align(7, 1, {}, "align")
    assert not any(name == "start_vla_task" for name, _ in intents)
    assert agent.forced_skill is None


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_loss_frame_limit_must_be_a_positive_integer(tmp_path, limit):
    spec = yaml.safe_load((CONFIGS / "geometric_vla.yaml").read_text())
    for key in ("runtime_profile", "tasks_file", "cases_file"):
        spec[key] = str((CONFIGS / spec[key]).resolve())
    spec["geometric"] = {"both_lost_frames": limit}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        resolve(path)


def test_camera_and_loss_evidence_survives_minimal_logging(tmp_path):
    recorder = Recorder(path=tmp_path / "events.jsonl")
    recorder.runtime(
        "base_pose", "GEOMETRIC_CAMERA_SELECTED", generation=1, camera_stream=CHEST,
        views={HEAD: {"target_visible": False}, CHEST: {"target_visible": True, "usable": True}},
    )
    recorder.runtime("base_pose", "GEOMETRIC_FINISHED", generation=1,
                     reason="geometric_target_lost", both_lost_frames=20)
    events = read_events(recorder.path)
    assert len(events) == 2 and events[0]["camera_stream"] == CHEST
    assert events[0]["views"][HEAD]["target_visible"] is False
    assert events[1]["reason"] == "geometric_target_lost" and events[1]["both_lost_frames"] == 20
