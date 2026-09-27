"""Skill distances reach the controller without leaking into the next operation."""

import pytest

from gear_sonic.tests.test_flexible_mission import flexible_harness
from gear_sonic.utils.inference.lavira.mission import parse_skill_definitions
from gear_sonic.utils.inference.base_pose.agent import BasePoseAgentConfig, GatewayRawServoAdapter
from gear_sonic.utils.inference.base_pose.servo import RawServoEvent


DISTANCES = {"raw_head_target_distance_m": 1.25, "raw_chest_target_distance_m": 0.9}
OVERRIDES = {**DISTANCES, "raw_head_target_confidence": 0.3, "raw_chest_target_confidence": 0.3,
             "raw_stable_frames": 3, "raw_lateral_tolerance_m": 0.12}


def test_flexible_align_sends_only_selected_skill_distances(monkeypatch):
    h = flexible_harness(monkeypatch, [
        ("ALIGN", "bag"), ("MANIPULATE", "bag"),
        ("ALIGN", "bin"), ("MANIPULATE", "bin"),
        ("ALIGN", "bag"), ("MANIPULATE", "bag"), ("COMPLETE", "bag"),
    ])
    h.agent.skills["bin"]["base_pose"] = dict(OVERRIDES)
    h.agent.skills = parse_skill_definitions(h.agent.skills, flexible=True)
    result = h.agent.run(1)
    assert result.reason == "mission_completed", result
    starts = [p for name, p in h.intents if name == "start_base_pose"]
    assert [p.get("base_pose", {}) for p in starts] == [{}, OVERRIDES, {}]


@pytest.mark.parametrize("overrides", [
    {"raw_head_target_distance_m": -1}, {"raw_chest_target_distance_m": float("nan")},
    {"raw_head_target_distance_m": True}, {"raw_max_lateral_speed_m_s": 2},
    {"raw_head_target_confidence": 0}, {"raw_chest_target_confidence": 1.1},
    {"raw_head_target_confidence": True},
    {"raw_stable_frames": 0}, {"raw_stable_frames": 2.5}, {"raw_stable_frames": True},
    {"raw_lateral_tolerance_m": 0}, {"raw_lateral_tolerance_m": -0.1},
    {"raw_lateral_tolerance_m": float("nan")}, {"raw_lateral_tolerance_m": True},
])
def test_skill_rejects_invalid_distance_overrides(monkeypatch, overrides):
    h = flexible_harness(monkeypatch, [])
    h.agent.skills["bin"]["base_pose"] = overrides
    with pytest.raises(ValueError, match="base_pose"):
        parse_skill_definitions(h.agent.skills, flexible=True)


def test_adapter_applies_distances_and_restores_defaults(tmp_path):
    config = BasePoseAgentConfig(task="align", output_root=str(tmp_path), raw_head_target_distance_m=1.1,
                                raw_head_approach_cutoff_m=1.5, raw_chest_target_distance_m=0.9,
                                raw_head_target_confidence=0.3, raw_chest_target_confidence=0.15,
                                raw_lateral_tolerance_m=0.08)
    adapter = GatewayRawServoAdapter(config, submit_intent=lambda *args: None)
    assert adapter.start(1, skill_id=1, base_pose=OVERRIDES, now=1)
    assert adapter.runtime._target_distance_for_stream(config.dual_head_camera_stream) == 1.25
    assert adapter.runtime._target_distance_for_stream(config.dual_chest_camera_stream) == 0.9
    assert adapter.runtime.controller.stable_frames_required == 3
    adapter.runtime.controller.yaw_error_trusted = True
    assert not adapter.runtime.controller._post_stop_pose_out_of_tolerance(
        forward_error=0, right_error=0.12, yaw_error=0)
    assert adapter.runtime.controller._post_stop_pose_out_of_tolerance(
        forward_error=0, right_error=0.121, yaw_error=0)
    from types import SimpleNamespace
    from gear_sonic.utils.inference.base_pose.dual_servo import _set_target_confidence
    tracker = SimpleNamespace(confidence=0.25)
    for stream in (config.dual_head_camera_stream, config.dual_chest_camera_stream):
        _set_target_confidence(config, tracker, stream)
        assert tracker.target_confidence == 0.3
    adapter.runtime.requests.get_nowait()
    adapter.runtime.events.put(RawServoEvent(adapter.runtime.generation, "error", error="retry", hard=True))
    adapter.tick(now=1.1)
    assert adapter.start(1, skill_id=2, now=1.2)
    assert config.raw_head_target_distance_m == 1.1
    assert config.raw_chest_target_distance_m == 0.9
    assert config.raw_head_target_confidence == 0.3
    assert config.raw_chest_target_confidence == 0.15
    assert adapter.runtime.controller.stable_frames_required == 5
    adapter.runtime.controller.yaw_error_trusted = True
    assert adapter.runtime.controller._post_stop_pose_out_of_tolerance(
        forward_error=0, right_error=0.10, yaw_error=0)
    assert not adapter.runtime.controller._post_stop_pose_out_of_tolerance(
        forward_error=0, right_error=0.08, yaw_error=0)


@pytest.mark.parametrize("overrides,required", [({}, 5), ({"raw_stable_frames": 3}, 3)])
def test_alignment_enters_stop_review_after_configured_stable_frames(tmp_path, overrides, required):
    from gear_sonic.tests.test_base_pose_yolo_adapter import _servo_observation, _orientation
    from gear_sonic.utils.inference.base_pose.servo import ServoPhase
    config = BasePoseAgentConfig(task="align", output_root=str(tmp_path), raw_head_target_distance_m=1.2)
    adapter = GatewayRawServoAdapter(config, submit_intent=lambda *args: None)
    assert adapter.start(1, base_pose=overrides, now=1)
    controller = adapter.runtime.controller
    controller.phase = ServoPhase.TRANSLATE_TARGET
    for index in range(required):
        controller.update(_servo_observation(), now=1.05 + index * 0.05, orientation=_orientation(0))
        expected = ServoPhase.POST_STOP_SAMPLING if index == required - 1 else ServoPhase.TRANSLATE_TARGET
        assert controller.phase is expected
    assert controller.post_stop_sample_frames == 30


def test_adapter_rejects_distance_beyond_approach_cutoff(tmp_path):
    config = BasePoseAgentConfig(task="align", output_root=str(tmp_path), raw_head_target_distance_m=1.1,
                                raw_head_approach_cutoff_m=1.5)
    intents = []
    adapter = GatewayRawServoAdapter(config, submit_intent=lambda name, p: intents.append((name, p)))
    assert not adapter.start(1, base_pose={"raw_head_target_distance_m": 1.6}, now=1)
    assert adapter.runtime.phase == "idle"
    assert config.raw_head_target_distance_m == 1.1
    assert intents[-1][1]["reason"].startswith("invalid_base_pose:")
