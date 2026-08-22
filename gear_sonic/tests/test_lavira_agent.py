from __future__ import annotations

import math

import numpy as np
import pytest

from gear_sonic.runtime.protocol import (
    NavigationCommand,
    PlannerVelocityCommand,
    build_navigation_message,
    build_planner_velocity_message,
    decode_navigation_message,
    decode_planner_velocity_message,
)
from gear_sonic.utils.inference.lavira.agent import (
    LaViRAAgent,
    LaViRAAgentError,
    validate_language_action,
    validate_vision_action,
)
from gear_sonic.utils.inference.lavira.object_nav import RGBDSnapshot
from gear_sonic.utils.inference.navdp.navigation import HeadingGoalController
from gear_sonic.utils.planner_control.executor import (
    PlannerVelocityExecutorCore,
    SafetySnapshot,
)


def la(*, stop: bool = False, direction: str = "front") -> dict[str, object]:
    return {
        "progress_analysis": "progress",
        "updated_todo_list": "- [ ] continue",
        "reasoning": "reason",
        "turn_direction": direction,
        "stop": stop,
        "expected_landmark": "chair",
    }


def va(*, action: str = "NAVIGATE") -> dict[str, object]:
    return {
        "visual_check": "chair visible",
        "action": action,
        "bbox_2d": [200, 200, 800, 800] if action == "NAVIGATE" else None,
        "target": "chair",
        "target_type": "global_target",
        "confidence": 0.9,
        "stop_reasoning": "reached" if action == "STOP" else "",
    }


class FakeCamera:
    def __init__(self) -> None:
        self.rgb_count = 0
        self.depth_count = 0
        self.leases: list[tuple[int, int]] = []

    def capture_rgb(self) -> np.ndarray:
        self.rgb_count += 1
        return np.full((8, 8, 3), self.rgb_count, dtype=np.uint8)

    def begin_depth_lease(self, generation: int, segment_id: int) -> None:
        self.leases.append((generation, segment_id))

    def capture_aligned_rgbd(self) -> RGBDSnapshot:
        self.depth_count += 1
        return RGBDSnapshot(
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8), 2000.0, dtype=np.float32),
            100.0,
            3.5,
        )


class FakeClient:
    def __init__(self, language: list[dict], vision: list[dict]) -> None:
        self.language = iter(language)
        self.vision = iter(vision)
        self.answer_calls = 0

    def initial_todo(self, **_kwargs):
        return "- [ ] find chair"

    def language_action(self, **_kwargs):
        return next(self.language)

    def vision_action(self, **_kwargs):
        return next(self.vision)

    def answer(self, **_kwargs):
        self.answer_calls += 1
        return {"reasoning": "I see blue", "answer": "blue"}


def build_agent(*, task_type="vln", language, vision, max_steps=20):
    camera = FakeCamera()
    client = FakeClient(language, vision)
    intents: list[tuple[str, dict]] = []
    waited: list[int] = []

    def wait_status(generation: int, segment_id: int, _timeout: float):
        assert generation == 7
        waited.append(segment_id)
        return {"generation": generation, "segment_id": segment_id, "state": "reached"}

    agent = LaViRAAgent(
        task_type=task_type,
        mission="find the chair",
        global_target="chair",
        question="what color is it?" if task_type == "eqa" else "",
        max_steps=max_steps,
        history_size=5,
        min_confidence=0.6,
        segment_timeout_seconds=1.0,
        camera=camera,
        client=client,
        submit_intent=lambda name, values: intents.append((name, dict(values))),
        wait_status=wait_status,
        cancelled=lambda _generation: False,
    )
    return agent, camera, client, intents, waited


def test_protocol_round_trip_preserves_heading_goal_and_segment() -> None:
    command = decode_navigation_message(
        build_navigation_message(
            mode="heading_goal",
            generation=3,
            segment_id=4,
            timestamp=1.0,
            heading_delta_rad=math.pi / 2,
        )
    )
    assert command == NavigationCommand(
        "heading_goal", 3, 1.0, segment_id=4, heading_delta_rad=math.pi / 2
    )
    old = decode_navigation_message(
        {
            "type": "sonic_navigation_command",
            "version": 1,
            "generation": 2,
            "mode": "stop",
            "timestamp": 1.0,
        }
    )
    assert old.segment_id == 0


def test_velocity_segment_is_wire_scoped_and_old_segment_is_rejected() -> None:
    velocity = decode_planner_velocity_message(
        build_planner_velocity_message(
            generation=3,
            segment_id=5,
            source="navdp",
            velocity=(0.0, 0.0, 0.4),
        )
    )
    assert velocity.segment_id == 5
    core = PlannerVelocityExecutorCore()
    core.accept_navigation(
        NavigationCommand("heading_goal", 3, 1.0, segment_id=5, heading_delta_rad=0.5),
        now=1.0,
    )
    assert not core.accept_planner_velocity(
        PlannerVelocityCommand(3, 1.0, "navdp", (0.0, 0.0, 0.4), segment_id=4),
        now=1.1,
    )


def test_safety_block_freezes_heading_update() -> None:
    core = PlannerVelocityExecutorCore()
    core.sonic.heading = 0.25
    core.accept_navigation(
        NavigationCommand("heading_goal", 1, 1.0, segment_id=0, heading_delta_rad=1.0),
        now=1.0,
    )
    core.accept_planner_velocity(
        PlannerVelocityCommand(
            1,
            1.0,
            "navdp",
            (0.0, 0.0, 0.4),
            segment_id=0,
            heading_target_rad=1.0,
            heading_reference_rad=0.0,
        ),
        now=1.1,
    )
    decision = core.decide(now=1.1, safety=SafetySnapshot())
    assert decision.reason == "radar_timeout"
    assert core.sonic.heading == pytest.approx(0.25)


def test_heading_controller_wrap_stability_and_timeout() -> None:
    controller = HeadingGoalController(timeout_s=1.0)
    controller.start(current_yaw=math.pi - 0.1, delta_rad=math.pi / 2, now=0.0)
    assert controller.update(current_yaw=math.pi - 0.1, now=0.1).angular_velocity_rad_s > 0
    target = controller.target_rad
    assert controller.update(current_yaw=target, now=0.2).state == "active"
    assert controller.update(current_yaw=target, now=0.3).state == "active"
    assert controller.update(current_yaw=target, now=0.4).state == "reached"

    timed = HeadingGoalController(timeout_s=0.2)
    timed.start(current_yaw=0.0, delta_rad=-math.pi / 2, now=0.0)
    assert timed.update(current_yaw=0.0, now=0.3).reason == "heading_timeout"

    behind = HeadingGoalController()
    behind.start(current_yaw=0.0, delta_rad=math.pi, now=0.0)
    assert behind.target_rad == pytest.approx(math.pi)
    assert behind.update(current_yaw=0.0, now=0.1).angular_velocity_rad_s == pytest.approx(0.4)


def test_la_stop_with_va_bbox_executes_one_final_navdp_approach() -> None:
    agent, camera, _client, intents, waited = build_agent(
        language=[la(stop=True, direction="left")],
        vision=[va()],
    )
    result = agent.run(7)
    assert result.state == "reached"
    assert result.reason == "la_stop_final_approach_reached"
    assert waited == [0, 1]
    assert camera.depth_count == 5
    assert [name for name, _ in intents] == [
        "navigation_heading_goal",
        "lavira_depth_request",
        "lavira_rgbd_captured",
        "navigation_goal",
    ]


def test_vln_multistep_and_eqa_answer_only_after_success() -> None:
    agent, camera, client, _intents, waited = build_agent(
        task_type="eqa",
        language=[la(), la(direction="right")],
        vision=[va(), va(action="STOP")],
    )
    result = agent.run(7)
    assert result.state == "reached"
    assert result.answer == "blue"
    assert client.answer_calls == 1
    assert waited == [0, 1, 2]
    assert camera.rgb_count == 4  # initial, two LA observations, fresh EQA frame


def test_step_limit_and_strict_json_schemas_fail_closed() -> None:
    agent, _camera, client, _intents, _waited = build_agent(
        language=[la()], vision=[va()], max_steps=1
    )
    result = agent.run(7)
    assert result.state == "failed"
    assert result.reason == "max_steps_exceeded:1"
    assert client.answer_calls == 0

    invalid_la = la()
    invalid_la["extra"] = "not allowed"
    with pytest.raises(LaViRAAgentError, match="schema"):
        validate_language_action(invalid_la)
    invalid_va = va()
    invalid_va["bbox_2d"] = [900, 1, 100, 2]
    with pytest.raises(LaViRAAgentError, match="corner"):
        validate_vision_action(invalid_va)
