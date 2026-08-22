from __future__ import annotations

import json
import logging
import math
from types import SimpleNamespace

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
    LaViRAClient,
    MoveToView,
    ScanView,
    mean_bbox_depth_m,
    validate_alignment_grounding,
    validate_grounding,
    validate_language_action,
    validate_postcheck,
)
from gear_sonic.utils.inference.lavira.object_nav import RGBDSnapshot
from gear_sonic.utils.planner_control.executor import PlannerVelocityExecutorCore


def decision(skill: str | None, args: dict | None = None, *, result="EXECUTE"):
    return {
        "progress_analysis": "progress",
        "updated_todo_list": "- [x] observed\n- [ ] continue",
        "reasoning": "reason",
        "decision": result,
        "skill": skill,
        "skill_args": args or {},
        "expected_postcondition": "requested condition is visible",
    }


def move(target="basket", direction="front"):
    return decision("MOVE_TO", {
        "view_direction": direction,
        "target": target,
    })


def align():
    return decision("ALIGN")


def grounding(status="FOUND", description="basket"):
    return {
        "mode": "GROUNDING",
        "status": status,
        "bbox_2d": [200, 200, 800, 800] if status == "FOUND" else None,
        "point_2d": [500, 500] if status == "FOUND" else None,
        "target_description": description,
        "confidence": 0.9,
    }


def alignment_grounding(
    status="FOUND", target="basket", surface="desk",
    evidence="basket is on desk",
):
    return {
        "mode": "ALIGN_GROUNDING",
        "status": status,
        "target": target,
        "surface": surface,
        "bbox_2d": [200, 200, 800, 800] if status == "FOUND" else None,
        "visual_evidence": evidence,
        "confidence": 0.9,
    }


def postcheck(
    status="SATISFIED", evidence="condition visible", transition=None,
):
    if transition is None:
        transition = (
            "UNKNOWN" if status == "UNKNOWN" else "CONTINUE_NAVIGATION"
        )
    return {
        "mode": "POSTCHECK",
        "status": status,
        "transition": transition,
        "visual_evidence": evidence,
        "confidence": 0.9,
    }


def ready_to_align(status="SATISFIED", evidence="ready to align"):
    return postcheck(status, evidence, "READY_TO_ALIGN")


def ready_to_manipulate(status="SATISFIED", evidence="ready to manipulate"):
    return postcheck(status, evidence, "READY_TO_MANIPULATE")


def task_complete(evidence="task complete"):
    return postcheck("SATISFIED", evidence, "TASK_COMPLETE")


class FakeCamera:
    def __init__(self, depth_mm=2000.0, handoff_depth_mm=None):
        self.rgb_count = 0
        self.depth_count = 0
        self.depth_mm = float(depth_mm)
        self.handoff_depth_mm = (
            None if handoff_depth_mm is None else float(handoff_depth_mm)
        )
        self.pose = (0.0, 0.0, 0.0)
        self.leases = []

    def capture_rgb(self, *, camera_stream="chest_view"):
        self.rgb_count += 1
        offset = 100 if camera_stream == "ego_view" else 0
        return np.full(
            (8, 8, 3), (self.rgb_count + offset) % 255, np.uint8,
        )

    def current_pose(self):
        return self.pose

    def begin_depth_lease(self, generation, skill_id, segment_id):
        self.leases.append((generation, skill_id, segment_id))

    def capture_aligned_rgbd(self):
        self.depth_count += 1
        depth_mm = (
            self.handoff_depth_mm
            if self.handoff_depth_mm is not None and self.depth_count > 5
            else self.depth_mm
        )
        return RGBDSnapshot(
            np.zeros((8, 8, 3), np.uint8),
            np.full((8, 8), depth_mm, np.float32),
            100.0,
            3.5,
        )


class FakeClient:
    def __init__(
        self, decisions, groundings=(), alignment_groundings=(), postchecks=(),
    ):
        self.decisions = iter(decisions)
        self.groundings = iter(groundings)
        self.alignment_groundings = iter(alignment_groundings)
        self.postchecks = iter(postchecks)
        self.la_calls = []
        self.grounding_calls = []
        self.alignment_grounding_calls = []
        self.postcheck_calls = []
        self.todo_calls = []

    def initial_todo(self, **kwargs):
        self.todo_calls.append(kwargs)
        return "- [ ] complete mission"

    def language_action(self, **kwargs):
        self.la_calls.append(kwargs)
        return next(self.decisions)

    def grounding(self, **kwargs):
        self.grounding_calls.append(kwargs)
        return next(self.groundings)

    def alignment_grounding(self, **kwargs):
        self.alignment_grounding_calls.append(kwargs)
        return next(self.alignment_groundings)

    def postcheck(self, **kwargs):
        self.postcheck_calls.append(kwargs)
        return next(self.postchecks)


def test_lavira_cloud_calls_preserve_role_specific_thinking_mode(tmp_path) -> None:
    calls = []
    responses = iter([
        "- [ ] complete mission",
        json.dumps(move()),
        json.dumps(grounding()),
        json.dumps(alignment_grounding()),
        json.dumps(ready_to_manipulate()),
    ])

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=next(responses)),
                )],
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
    )

    lavira = LaViRAClient(
        la_base_url="https://example.invalid/v1",
        va_base_url="https://example.invalid/v1",
        la_client=client,
        va_client=client,
        request_context_dir=tmp_path / "lavira_requests",
    )
    image = np.zeros((8, 8, 3), np.uint8)
    scans = [
        ScanView(2, direction, image, (1.0, 2.0, 0.0), yaw)
        for direction, yaw in (
            ("front", 0.0), ("right", -math.pi / 2),
            ("behind", math.pi), ("left", math.pi / 2),
        )
    ]
    move_view = MoveToView(
        4, "basket", image, "reached", "SATISFIED", "basket is near",
    )
    lavira.initial_todo(mission="find basket", scan_views=scans)
    lavira.language_action(
        mission="find basket",
        navigation_mode="object_nav",
        global_target="basket",
        current_step=3,
        todo_list="- [ ] find basket",
        scan_views=scans,
        move_to_views=[move_view],
    )
    lavira.grounding(
        mission="find basket",
        global_target="basket",
        strategic_goal="Approach the requested basket",
        strategic_stop=False,
        target="basket",
        direction="front",
        image_bgr=image,
    )
    lavira.alignment_grounding(
        mission="find basket",
        global_target="basket",
        strategic_goal="Prepare the basket for the final task",
        strategic_stop=False,
        direction="front",
        image_bgr=image,
    )
    lavira.postcheck(
        mission="find basket",
        global_target="basket",
        strategic_goal="The basket should be ready for the final task",
        strategic_stop=False,
        expected_postcondition="basket aligned",
        image_bgr=image,
    )

    assert [call["extra_body"] for call in calls] == [
        {"enable_thinking": True},
        {"enable_thinking": True},
        {"enable_thinking": False},
        {"enable_thinking": False},
        {"enable_thinking": False},
    ]
    assert calls[0]["messages"][0]["content"].startswith("Reason carefully")
    assert calls[2]["messages"][0]["content"] == "/no_think"
    todo_content = calls[0]["messages"][1]["content"]
    assert todo_content[0] == {
        "type": "text",
        "text": (
            'Instruction: "find basket"\n\n'
            "The images provided are the 4-directional views from the "
            "starting position."
        ),
    }
    assert sum(item["type"] == "image_url" for item in todo_content) == 4
    la_content = calls[1]["messages"][1]["content"]
    assert sum(item["type"] == "image_url" for item in la_content) == 5
    la_labels = [
        item["text"] for item in la_content if item["type"] == "text"
    ]
    assert la_labels[0] == 'Navigation Task: "find basket"\n\n- Current Step: 3'
    assert la_labels[1] == "PLAN-1"
    assert la_labels[2:6] == [
        "Image 1: The current FORWARD view (Step 3).",
        "Image 2: The view after turning 90 deg to the RIGHT (Step 3).",
        "Image 3: The view directly BEHIND (180 deg turn) (Step 3).",
        "Image 4: The view after turning 90 deg to the LEFT (Step 3).",
    ]
    plan_image_index = next(
        index for index, item in enumerate(la_content)
        if item.get("text") == "PLAN-1"
    ) - 1
    current_image_index = next(
        index for index, item in enumerate(la_content)
        if item.get("text")
        == "Image 1: The current FORWARD view (Step 3)."
    ) - 1
    assert plan_image_index < current_image_index
    assert "absolute_yaw" not in json.dumps(la_content)
    assert "controller=" not in json.dumps(la_content)
    la_prompt = la_content[-1]["text"]
    assert '**MISSION**: "find basket"' in la_prompt
    assert '**GLOBAL TARGET**: "basket"' in la_prompt
    assert "**Current Step**: 3" in la_prompt
    assert "**Current TODO List**" in la_prompt
    assert "RECENT SKILLS/RESULTS" not in la_prompt
    assert "FAST-LIO EXPLORATION MEMORY" not in la_prompt
    assert '"view_direction":"front|left|right"' in la_prompt
    for index in (2, 3, 4):
        va_content = calls[index]["messages"][1]["content"]
        assert sum(item["type"] == "image_url" for item in va_content) == 1
        va_prompt = va_content[-1]["text"]
        assert '**MISSION**: "find basket"' in va_prompt
        assert '**GLOBAL TARGET**: "basket"' in va_prompt
        assert "**CURRENT STRATEGY**" in va_prompt
        assert "**STRATEGIC STOP SIGNAL**: false" in va_prompt
    postcheck_prompt_text = calls[4]["messages"][1]["content"][-1]["text"]
    assert "latest controller\naction" in postcheck_prompt_text
    assert "after ALIGN" not in postcheck_prompt_text
    assert "skill" not in postcheck_prompt_text.lower()
    saved = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / "lavira_requests").glob("*.json"))
    ]
    assert {item["request_kind"] for item in saved} == {
        "la_todo", "la_decision", "va_grounding", "va_align_grounding",
        "va_postcheck",
    }
    assert all(item["type"] == "sonic.model_request_context" for item in saved)
    assert all(item["attempt"] == 1 for item in saved)
    assert all(item["request"]["messages"] for item in saved)
    assert any(
        "data:image/jpeg;base64," in json.dumps(item["request"])
        for item in saved
    )


def build_agent(
    decisions, *, groundings=(), alignment_groundings=(), postchecks=(),
    max_steps=20, poll_failure=None, events=None, todos=None, depth_mm=2000.0,
    handoff_depth_mm=None,
):
    camera = FakeCamera(
        depth_mm=depth_mm, handoff_depth_mm=handoff_depth_mm,
    )
    client = FakeClient(
        decisions, groundings, alignment_groundings, postchecks,
    )
    intents = []
    waited = []

    def wait_status(generation, skill_id, segment_id, _timeout):
        waited.append((skill_id, segment_id))
        name = intents[-1][0]
        if name == "start_vla_task":
            return {
                "generation": generation,
                "skill_id": skill_id,
                "segment_id": segment_id,
                "state": "active",
                "reason": "started",
            }
        reason = "aligned" if name == "start_base_pose" else "reached"
        return {
            "generation": generation,
            "skill_id": skill_id,
            "segment_id": segment_id,
            "state": "reached",
            "reason": reason,
        }

    agent = LaViRAAgent(
        navigation_mode="object_nav",
        mission="find the basket and put the bottle in it",
        global_target="basket",
        max_steps=max_steps,
        history_size=5,
        min_confidence=0.6,
        segment_timeout_seconds=1.0,
        camera=camera,
        client=client,
        submit_intent=lambda name, args: intents.append((name, dict(args))),
        wait_status=wait_status,
        cancelled=lambda _generation: False,
        manipulation_window_seconds=0.0,
        sleep=lambda _seconds: None,
        poll_failure=poll_failure,
        report_event=(
            None
            if events is None
            else lambda level, code, message, **fields: events.append(
                {
                    "level": level,
                    "code": code,
                    "message": message,
                    "fields": fields,
                }
            )
        ),
        report_todo=(
            None
            if todos is None
            else lambda generation, step, todo: todos.append(
                (generation, step, todo)
            )
        ),
    )
    return agent, camera, client, intents, waited


def test_protocol_preserves_skill_and_old_messages_default_to_zero():
    new = decode_navigation_message(build_navigation_message(
        mode="heading_goal", generation=3, skill_id=4, segment_id=5,
        timestamp=1.0, heading_delta_rad=math.pi / 2,
    ))
    assert new == NavigationCommand(
        "heading_goal", 3, 1.0, segment_id=5, skill_id=4,
        heading_delta_rad=math.pi / 2,
    )
    old = decode_navigation_message({
        "type": "sonic_navigation_command", "version": 1,
        "generation": 2, "mode": "stop", "timestamp": 1.0,
    })
    assert old.skill_id == old.segment_id == 0


def test_old_skill_or_segment_velocity_cannot_move_robot():
    core = PlannerVelocityExecutorCore()
    core.accept_navigation(NavigationCommand(
        "heading_goal", 3, 1.0, segment_id=5, skill_id=2,
        heading_delta_rad=0.5,
    ), now=1.0)
    stale_skill = decode_planner_velocity_message(build_planner_velocity_message(
        generation=3, skill_id=1, segment_id=5, source="navdp",
        velocity=(0.0, 0.0, 0.4), timestamp=1.1,
    ))
    stale_segment = PlannerVelocityCommand(
        3, 1.1, "navdp", (0.0, 0.0, 0.4), segment_id=4, skill_id=2,
    )
    assert not core.accept_planner_velocity(stale_skill, now=1.1)
    assert not core.accept_planner_velocity(stale_segment, now=1.1)


def test_each_la_step_gets_panorama_and_consecutive_move_to_are_supported():
    steps = [
        move("doorway", "left"),
        move("desk", "front"),
        move("basket", "right"),
        decision(None, result="FAIL"),
    ]
    agent, camera, client, intents, _waited = build_agent(
        steps,
        groundings=[grounding() for _ in range(6)],
    )
    result = agent.run(7)
    assert result.state == "failed" and result.reason.startswith("la_fail")
    assert len(client.todo_calls[0]["scan_views"]) == 4
    assert client.todo_calls[0]["scan_views"] == client.la_calls[0]["scan_views"]
    names = [name for name, _ in intents]
    assert names.count("navigation_goal") == 3
    assert names.count("navigation_heading_goal") == 4 + 3
    heading_deltas = [
        args["heading_delta_rad"]
        for name, args in intents if name == "navigation_heading_goal"
    ]
    assert heading_deltas[:5] == pytest.approx([
        -math.pi / 2, math.pi, math.pi / 2, 0.0, math.pi / 2,
    ])
    first_goal = names.index("navigation_goal")
    assert names[:first_goal].count("navigation_heading_goal") == 5
    assert names.index("lavira_depth_request") > max(
        index for index, name in enumerate(names[:first_goal])
        if name == "navigation_heading_goal"
    )
    assert camera.depth_count == 18
    skill_ids = [args["skill_id"] for name, args in intents if name.startswith("navigation_")]
    assert skill_ids == sorted(skill_ids)


def test_la_context_keeps_fresh_panorama_and_last_five_completed_moves():
    moves = [move(f"landmark-{index}") for index in range(6)]
    agent, _camera, client, _intents, _waited = build_agent(
        [*moves, decision(None, result="FAIL")],
        groundings=[grounding() for _ in range(2 * len(moves))],
    )

    result = agent.run(17)

    assert result.state == "failed"
    final_context = client.la_calls[-1]
    assert final_context["current_step"] == 7
    assert len(final_context["scan_views"]) == 1
    assert [view.direction for view in final_context["scan_views"]] == [
        "front",
    ]
    assert len(final_context["move_to_views"]) == 5
    assert [view.target for view in final_context["move_to_views"]] == [
        f"landmark-{index}" for index in range(1, 6)
    ]
    assert [view.skill_id for view in final_context["move_to_views"]] == [
        2, 3, 4, 5, 6,
    ]
    assert all(
        view.controller_state == "reached"
        for view in final_context["move_to_views"]
    )


def test_va_context_contains_mission_strategy_and_terminal_stop_flag():
    agent, _camera, client, _intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding(), grounding()],
        alignment_groundings=[alignment_grounding()],
        postchecks=[
            ready_to_manipulate(), ready_to_manipulate(), task_complete(),
        ],
    )

    result = agent.run(18)

    assert result.state == "reached"
    assert result.steps == 3
    assert len(client.la_calls) == 3
    calls = [
        *client.grounding_calls,
        *client.alignment_grounding_calls,
        *client.postcheck_calls,
    ]
    assert all(
        call["mission"] == "find the basket and put the bottle in it"
        and call["global_target"] == "basket"
        and call["strategic_goal"]
        for call in calls
    )
    assert all(not call["strategic_stop"] for call in client.grounding_calls)
    assert [call["strategic_stop"] for call in client.postcheck_calls] == [
        False, False, True,
    ]
    assert all("skill" not in call for call in client.postcheck_calls)
    assert all(
        not any(
            skill_name in call["strategic_goal"]
            for skill_name in ("MOVE_TO", "ALIGN", "MANIPULATE")
        )
        for call in calls
    )


def test_new_generation_clears_visual_context_from_reused_agent():
    agent, _camera, client, _intents, _waited = build_agent(
        [move(), decision(None, result="FAIL"), decision(None, result="FAIL")],
        groundings=[grounding("NOT_FOUND")],
    )

    first = agent.run(20)
    second = agent.run(21)

    assert first.state == second.state == "failed"
    assert len(client.la_calls[1]["scan_views"]) == 4
    assert len(client.la_calls[2]["scan_views"]) == 4
    assert client.la_calls[2]["move_to_views"] == ()
    assert client.la_calls[2]["current_step"] == 1


def test_todo_pane_updates_only_when_full_markdown_todo_changes():
    todos = []
    unchanged = move()
    unchanged["updated_todo_list"] = "- [ ] complete mission"
    agent, _camera, _client, _intents, _waited = build_agent(
        [unchanged, decision(None, result="FAIL")],
        groundings=[grounding("NOT_FOUND")],
        todos=todos,
    )

    agent.run(22)

    assert todos == [
        (22, 0, "- [ ] complete mission"),
        (22, 2, "- [x] observed\n- [ ] continue"),
    ]


def test_lost_move_target_is_followed_by_a_fresh_automatic_panorama():
    agent, _camera, client, _intents, _waited = build_agent(
        [move("doorway", "left"), decision(None, result="FAIL")],
        groundings=[grounding("NOT_FOUND", "doorway absent")],
    )
    result = agent.run(7)
    assert result.state == "failed"
    assert client.la_calls[1]["move_to_views"] == ()
    assert len(client.la_calls[1]["scan_views"]) == 4
    assert client.la_calls[1]["scan_views"][0].scan_id == 2
    assert agent._history[-1].skill == "MOVE_TO"


def test_move_outside_handoff_depth_does_not_enter_la_image_history():
    agent, _camera, client, _intents, _waited = build_agent(
        [move(), decision(None, result="FAIL")],
        groundings=[grounding(), grounding()],
        depth_mm=5000.0,
    )

    result = agent.run(7)

    assert result.state == "failed"
    assert client.la_calls[1]["move_to_views"] == ()
    assert agent._history[-1].skill == "MOVE_TO"
    assert agent._history[-1].va_result == "NOT_SATISFIED"


def test_align_retry_and_align_can_return_to_move_to():
    agent, _camera, client, intents, _waited = build_agent(
        [
            move("basket"), align(), move("basket"), align(),
            decision(None, result="FAIL"),
        ],
        groundings=[grounding() for _ in range(4)],
        alignment_groundings=[
            alignment_grounding("NOT_FOUND"), alignment_grounding(),
        ],
        postchecks=[
            postcheck("NOT_SATISFIED"), postcheck("NOT_SATISFIED"),
            postcheck("NOT_SATISFIED"), postcheck("NOT_SATISFIED"),
        ],
    )
    result = agent.run(7)
    assert result.state == "failed"
    assert [name for name, _ in intents].count("start_base_pose") == 1
    base_pose = next(args for name, args in intents if name == "start_base_pose")
    assert base_pose["target"] == "basket"
    assert base_pose["surface"] == "desk"
    assert base_pose["reference_bbox"] == [200, 200, 800, 800]
    assert agent._history[1].controller_state == "target_not_found"


def test_ready_nav_allows_another_move_and_uses_front_observation():
    agent, _camera, client, intents, _waited = build_agent(
        [move(), move("another landmark"), decision(None, result="FAIL")],
        groundings=[grounding() for _ in range(4)],
    )

    result = agent.run(7)

    assert result.reason.startswith("la_fail")
    assert len(client.la_calls[0]["scan_views"]) == 4
    assert [view.direction for view in client.la_calls[1]["scan_views"]] == [
        "front",
    ]
    assert [name for name, _args in intents].count("navigation_goal") == 2


def test_nav_handoff_depth_uses_mean_after_discarding_invalid_values():
    depth = np.array([
        [0.0, np.nan, np.inf],
        [1000.0, 3000.0, 9000.0],
    ], dtype=np.float32)

    assert mean_bbox_depth_m(
        depth, [0.0, 0.0, 1000.0, 1000.0], max_depth_m=8.0,
    ) == pytest.approx(2.0)
    assert mean_bbox_depth_m(
        np.zeros((4, 4), np.float32), [0.0, 0.0, 1000.0, 1000.0],
    ) is None


def test_nav_handoff_with_no_valid_depth_cannot_advance_to_align():
    agent, _camera, _client, intents, _waited = build_agent(
        [move(), align()],
        groundings=[grounding(), grounding()],
        handoff_depth_mm=0.0,
    )

    result = agent.run(39)

    assert result.state == "failed"
    assert result.reason == "handoff_gate:UNKNOWN_does_not_allow_ALIGN"
    assert [name for name, _args in intents].count("start_base_pose") == 0


def test_align_handoff_accepts_one_complete_camera_view():
    events = []
    agent, _camera, client, intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding(), grounding()],
        alignment_groundings=[alignment_grounding()],
        postchecks=[
            postcheck("NOT_SATISFIED", "chest is missing the bottle"),
            ready_to_manipulate("SATISFIED", "head contains bottle and basket"),
            task_complete(),
        ],
        events=events,
    )

    result = agent.run(37)

    assert result.state == "reached"
    align_event = next(
        event for event in events
        if event["code"] == "ALIGN_HANDOFF_EVALUATED"
    )
    assert align_event["fields"]["common_view"] == "head"
    align_checks = client.postcheck_calls[:2]
    assert all(
        "simultaneously visible in this single image"
        in call["expected_postcondition"]
        for call in align_checks
    )
    assert [name for name, _args in intents].count("start_vla_task") == 1


def test_align_handoff_never_combines_partial_visibility_across_cameras():
    agent, _camera, _client, intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding(), grounding()],
        alignment_groundings=[alignment_grounding()],
        postchecks=[
            postcheck("NOT_SATISFIED", "chest has bottle but no basket"),
            postcheck("NOT_SATISFIED", "head has basket but no bottle"),
        ],
    )

    result = agent.run(38)

    assert result.state == "failed"
    assert result.reason == "handoff_gate:RETRY_ALIGN_does_not_allow_MANIPULATE"
    assert [name for name, _args in intents].count("start_vla_task") == 0


def test_same_pose_gets_a_fresh_panorama_on_every_agent_step():
    agent, _camera, client, _intents, _waited = build_agent(
        [move()] * 3 + [decision(None, result="FAIL")],
        groundings=[grounding("NOT_FOUND")] * 3,
    )
    agent.run(7)
    assert [call["scan_views"][0].scan_id for call in client.la_calls] == [
        1, 2, 3, 4,
    ]
    assert all(len(call["scan_views"]) == 4 for call in client.la_calls)


def test_twenty_agent_step_limit_and_monotonic_skill_ids():
    agent, _camera, _client, intents, _waited = build_agent(
        [move()] * 20,
        groundings=[grounding("NOT_FOUND")] * 20,
        max_steps=20,
    )
    result = agent.run(7)
    assert result.reason == "max_steps_exceeded:20"
    assert result.steps == result.skill_id == 20
    assert all(args.get("skill_id", 0) <= 20 for _name, args in intents)


def test_manipulation_recovers_in_vla_then_system_completes_after_va_success():
    agent, _camera, _client, intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding(), grounding()],
        alignment_groundings=[alignment_grounding()],
        postchecks=[
            ready_to_manipulate(), ready_to_manipulate(),
            postcheck(
                "NOT_SATISFIED", "bottle still on desk",
                "CONTINUE_MANIPULATION",
            ),
            task_complete("bottle is in basket"),
        ],
    )
    result = agent.run(7)
    assert result.state == "reached"
    assert result.steps == 3
    names = [name for name, _ in intents]
    assert names.count("start_vla_task") == 1
    assert names.count("resume_vla_task") == 1
    assert names.count("stop_vla_task") == 1
    start = next(args for name, args in intents if name == "start_vla_task")
    assert start["task"] == "find the basket and put the bottle in it"
    assert "ORIGINAL TASK:" in start["handoff_context"]
    assert not any(name.startswith("navigation_") for name in names[names.index("start_vla_task") + 1:])


def test_manipulate_requires_a_ready_align_handoff():
    agent, _camera, client, intents, _waited = build_agent(
        [decision("MANIPULATE")],
        postchecks=[task_complete("mission complete")],
    )

    result = agent.run(31)

    assert result.state == "failed"
    assert result.reason == "handoff_gate:no_nav_readiness_for_MANIPULATE"
    assert result.steps == 1
    assert len(client.grounding_calls) == 0
    assert len(client.alignment_grounding_calls) == 0
    assert [name for name, _args in intents].count("start_vla_task") == 0


def test_vla_unknown_three_times_and_async_safety_fail_closed():
    unknowns = [postcheck("UNKNOWN")] * 6
    agent, _camera, _client, intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding(), grounding()],
        alignment_groundings=[alignment_grounding()],
        postchecks=[
            ready_to_manipulate(), ready_to_manipulate(), *unknowns,
        ],
    )
    result = agent.run(7)
    assert result.reason == "manipulate_postcheck_unknown"
    assert any(name == "stop_vla_task" for name, _ in intents)

    failures = iter([{"state": "failed", "reason": "radar_timeout"}])
    guarded, _camera, _client, _intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding(), grounding()],
        alignment_groundings=[alignment_grounding()],
        postchecks=[ready_to_manipulate(), ready_to_manipulate()],
        poll_failure=lambda _generation, _skill_id: next(failures, None),
    )
    guarded_result = guarded.run(7)
    assert guarded_result.reason == "manipulate_safety:radar_timeout"


def test_missing_vla_service_ends_agent_without_execution_window_retries():
    events = []
    agent, _camera, _client, intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding(), grounding()],
        alignment_groundings=[alignment_grounding()],
        postchecks=[ready_to_manipulate(), ready_to_manipulate()],
        events=events,
    )
    normal_wait = agent.wait_status

    def wait_or_timeout(generation, skill_id, segment_id, timeout):
        if intents[-1][0] == "start_vla_task":
            raise TimeoutError("VLA ACK timeout")
        return normal_wait(generation, skill_id, segment_id, timeout)

    agent.wait_status = wait_or_timeout

    result = agent.run(7)

    assert result.state == "failed"
    assert result.reason == "unexpected_termination:vla_service_unavailable"
    names = [name for name, _parameters in intents]
    assert names.count("start_vla_task") == 1
    assert names.count("stop_vla_task") == 1
    assert "hold_vla_task" not in names
    assert "resume_vla_task" not in names
    assert not any(
        event["code"] == "MANIPULATION_WINDOW_STARTED" for event in events
    )


def test_vla_rejected_start_ends_agent_without_retrying():
    agent, _camera, _client, intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding(), grounding()],
        alignment_groundings=[alignment_grounding()],
        postchecks=[ready_to_manipulate(), ready_to_manipulate()],
    )
    normal_wait = agent.wait_status

    def wait_or_reject(generation, skill_id, segment_id, timeout):
        if intents[-1][0] == "start_vla_task":
            return {
                "generation": generation,
                "skill_id": skill_id,
                "segment_id": segment_id,
                "state": "failed",
                "reason": "vla_policy_unreachable",
            }
        return normal_wait(generation, skill_id, segment_id, timeout)

    agent.wait_status = wait_or_reject

    result = agent.run(7)

    assert result.reason == "unexpected_termination:vla_policy_unreachable"
    assert [name for name, _parameters in intents].count("start_vla_task") == 1
    assert not any(name == "hold_vla_task" for name, _parameters in intents)


def test_strict_schemas_fail_closed():
    invalid = move()
    invalid["extra"] = True
    with pytest.raises(LaViRAAgentError, match="schema"):
        validate_language_action(invalid)
    with pytest.raises(LaViRAAgentError, match="invalid schema"):
        validate_language_action(decision("ALIGN", {"target": "basket"}))
    with pytest.raises(LaViRAAgentError, match="bbox"):
        validate_grounding({**grounding(), "bbox_2d": [900, 1, 100, 2]})
    with pytest.raises(LaViRAAgentError, match="status"):
        validate_postcheck({**postcheck(), "status": "FOUND"})
    with pytest.raises(LaViRAAgentError, match="invalid for ALIGN"):
        validate_postcheck(ready_to_align(), skill="ALIGN")
    with pytest.raises(LaViRAAgentError, match="status/transition mismatch"):
        validate_postcheck(
            postcheck("SATISFIED", transition="CONTINUE_MANIPULATION"),
            skill="MANIPULATE",
        )
    with pytest.raises(LaViRAAgentError, match="schema"):
        validate_alignment_grounding({**alignment_grounding(), "extra": True})
    with pytest.raises(LaViRAAgentError, match="decision"):
        validate_language_action(decision(None, result="COMPLETE"))
    with pytest.raises(LaViRAAgentError, match="only allowed on the first"):
        validate_language_action(
            move(direction="behind"), current_step=2,
        )


def test_structured_events_cover_decisions_skills_va_warnings_and_errors():
    events = []
    agent, _camera, _client, _intents, _waited = build_agent(
        [move("doorway", "left"), decision(None, result="FAIL")],
        groundings=[grounding("NOT_FOUND", "doorway absent")],
        events=events,
    )

    result = agent.run(7)

    assert result.state == "failed"
    by_code = {event["code"]: event for event in events}
    execute_decision = next(
        event
        for event in events
        if event["code"] == "LA_DECISION"
        and event["fields"]["decision"] == "EXECUTE"
    )
    assert by_code["TASK_STARTED"]["fields"]["generation"] == 7
    assert execute_decision["fields"]["skill"] == "MOVE_TO"
    assert by_code["SKILL_STARTED"]["fields"]["skill_id"] == 1
    assert by_code["VA_GROUNDING"]["level"] == logging.WARNING
    assert by_code["SKILL_COMPLETED"]["level"] == logging.WARNING
    assert by_code["TASK_FAILED"]["level"] == logging.ERROR
    for code in {"SKILL_STARTED", "SKILL_COMPLETED"}:
        assert by_code[code]["fields"]["generation"] == 7
        assert by_code[code]["fields"]["skill_id"] == 1
        assert "segment_id" in by_code[code]["fields"]


def test_structured_events_cover_panorama_three_skills_and_completion():
    events = []
    agent, _camera, _client, intents, _waited = build_agent(
        [
            move(),
            align(),
            decision("MANIPULATE"),
        ],
        groundings=[grounding(), grounding()],
        alignment_groundings=[alignment_grounding()],
        postchecks=[
            ready_to_manipulate(), ready_to_manipulate(), task_complete(),
        ],
        events=events,
    )

    result = agent.run(9)

    assert result.state == "reached"
    assert [
        event["fields"]["skill"]
        for event in events
        if event["code"] == "SKILL_STARTED"
    ] == ["MOVE_TO", "ALIGN", "MANIPULATE"]
    assert sum(event["code"] == "PANORAMA_COMPLETED" for event in events) == 1
    assert sum(
        event["code"] == "FRONT_OBSERVATION_COMPLETED" for event in events
    ) == 2
    handoff_events = [
        event for event in events
        if event["code"] in {
            "NAV_HANDOFF_EVALUATED", "ALIGN_HANDOFF_EVALUATED",
        }
    ]
    assert [event["fields"]["transition"] for event in handoff_events] == [
        "READY_TO_ALIGN", "READY_TO_MANIPULATE",
    ]
    postcheck_events = [
        event for event in events if event["code"] == "VA_POSTCHECK"
    ]
    assert [event["fields"]["transition"] for event in postcheck_events] == [
        "TASK_COMPLETE",
    ]
    assert [name for name, _args in intents].count("navigation_heading_goal") == 6
    assert events[-1]["code"] == "TASK_COMPLETED"
