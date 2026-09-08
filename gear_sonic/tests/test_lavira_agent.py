from __future__ import annotations

import json
import logging
import math
import sys
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
    _complete_move_to_todo,
    _first_incomplete_todo_line,
    _move_to_facing_geometry,
    alignment_grounding_prompt,
    language_action_prompt,
    mean_bbox_depth_m,
    validate_alignment_grounding,
    validate_grounding,
    validate_language_action,
    validate_postcheck,
)
from gear_sonic.utils.inference.lavira.camera import RGBDSnapshot
from gear_sonic.utils.planner_control.executor import PlannerVelocityExecutorCore


def decision(
    skill: str | None,
    args: dict | None = None,
    *,
    result="EXECUTE",
    global_target="basket",
    todo_list="- [x] observed\n- [ ] continue",
):
    return {
        "global_target": global_target,
        "progress_analysis": "progress",
        "updated_todo_list": todo_list,
        "reasoning": "reason",
        "decision": result,
        "skill": skill,
        "skill_args": args or {},
        "expected_postcondition": "requested condition is visible",
    }


def move(
    target="basket",
    direction="front",
    *,
    global_target="basket",
    todo_list="- [x] observed\n- [ ] continue",
):
    return decision("MOVE_TO", {
        "view_direction": direction,
        "target": target,
    }, global_target=global_target, todo_list=todo_list)


def align(*, global_target="basket"):
    return decision("ALIGN", global_target=global_target)


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
    status="FOUND",
    target="basket",
    yaw_align_target="desk",
    evidence="basket and yaw-align target are visible",
    *,
    bbox=None,
    target_confidence=0.9,
    yaw_align_target_confidence=0.9,
    target_visible=None,
    yaw_align_target_visible=None,
):
    if target_visible is None:
        target_visible = status == "FOUND"
    if yaw_align_target_visible is None:
        yaw_align_target_visible = status == "FOUND"
    if bbox is None and target_visible:
        bbox = [200, 200, 800, 800]
    return {
        "mode": "ALIGN_GROUNDING",
        "status": status,
        "target": {
            "name": target,
            "visible": target_visible,
            "bbox_2d": bbox,
            "confidence": target_confidence,
        },
        "yaw_align_target": {
            "name": yaw_align_target,
            "visible": yaw_align_target_visible,
            "confidence": yaw_align_target_confidence,
        },
        "visual_evidence": evidence,
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


def test_runtime_move_to_todo_helper_completes_only_recorded_item():
    todo = (
        "- [x] pass the glass door\n"
        "  - [ ] approach the black trash can\n"
        "- [ ] approach the dark blue trash can"
    )

    active_line = _first_incomplete_todo_line(todo)
    assert active_line == 1
    assert _complete_move_to_todo(todo, active_line, "black trash can") == (
        "- [x] pass the glass door\n"
        "  - [x] approach the black trash can Result: Runtime VA confirmed "
        'MOVE_TO target "black trash can" as SATISFIED.\n'
        "- [ ] approach the dark blue trash can"
    )


def test_runtime_move_to_todo_helper_fails_closed_if_item_changed():
    with pytest.raises(
        LaViRAAgentError,
        match="active MOVE_TO TODO item is no longer incomplete",
    ):
        _complete_move_to_todo("- [x] already complete", 0, "basket")


class FakeCamera:
    def __init__(self, depth_mm=2000.0, handoff_depth_mm=None):
        self.rgb_count = 0
        self.depth_count = 0
        self.depth_mm = float(depth_mm)
        self.handoff_depth_mm = (
            None if handoff_depth_mm is None else float(handoff_depth_mm)
        )
        self.pose = (0.0, 0.0, 0.0)
        self.sonic_yaw = 0.0
        self.capture_yaws = []
        self.rgbd_streams = []
        self.leases = []

    def capture_rgb(self, *, camera_stream="chest_view"):
        self.rgb_count += 1
        self.capture_yaws.append(self.sonic_yaw)
        offset = 100 if camera_stream == "ego_view" else 0
        return np.full(
            (8, 8, 3), (self.rgb_count + offset) % 255, np.uint8,
        )

    def current_pose(self):
        return self.pose

    def current_sonic_yaw(self):
        return self.sonic_yaw

    def begin_depth_lease(self, generation, skill_id, segment_id):
        self.leases.append((generation, skill_id, segment_id))

    def capture_aligned_rgbd(self):
        self.depth_count += 1
        self.rgbd_streams.append("chest_view")
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

    def capture_camera_aligned_rgbd(self, *, camera_stream):
        self.depth_count += 1
        self.rgbd_streams.append(camera_stream)
        depth_mm = (
            self.handoff_depth_mm
            if self.handoff_depth_mm is not None and self.depth_count > 5
            else self.depth_mm
        )
        return RGBDSnapshot(
            np.full((8, 8, 3), 100, np.uint8),
            np.full((8, 8), depth_mm, np.float32),
            100.0,
            3.5,
        )


class FakeClient:
    def __init__(
        self, decisions, groundings=(), alignment_groundings=(),
        postchecks=(),
    ):
        self.decisions = iter(decisions)
        self.groundings = iter(groundings)
        self.alignment_groundings = iter(alignment_groundings)
        self.postchecks = iter(postchecks)
        self.la_calls = []
        self.grounding_calls = []
        self.alignment_grounding_calls = []
        self.postcheck_calls = []

    def language_action(self, **kwargs):
        self.la_calls.append(kwargs)
        return next(self.decisions)

    def grounding(self, **kwargs):
        self.grounding_calls.append(kwargs)
        return next(self.groundings)

    def alignment_grounding(self, **kwargs):
        self.alignment_grounding_calls.append(kwargs)
        return validate_alignment_grounding(next(self.alignment_groundings))

    def postcheck(self, **kwargs):
        self.postcheck_calls.append(kwargs)
        return next(self.postchecks)


def test_lavira_clients_reuse_shared_dashscope_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = []

    def fake_openai(**kwargs):
        created.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=fake_openai),
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "shared-key")
    monkeypatch.delenv("LAVIRA_LA_API_KEY", raising=False)
    monkeypatch.delenv("LAVIRA_VA_API_KEY", raising=False)

    LaViRAClient(
        la_base_url="https://example.invalid/la",
        va_base_url="https://example.invalid/va",
    )

    assert [call["api_key"] for call in created] == ["shared-key", "shared-key"]


def test_lavira_client_rejects_missing_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "DASHSCOPE_API_KEY",
        "LAVIRA_LA_API_KEY",
        "LAVIRA_VA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="LaViRA LA requires"):
        LaViRAClient(
            la_base_url="https://example.invalid/la",
            va_base_url="https://example.invalid/va",
        )


def test_align_grounding_retries_after_legacy_surface_schema(
    tmp_path,
) -> None:
    calls = []
    invalid = alignment_grounding(
        target="cardboard box",
        yaw_align_target="cardboard box",
    )
    invalid["surface"] = "floor"
    responses = iter([
        json.dumps(invalid),
        json.dumps(alignment_grounding(
            target="cardboard box",
            yaw_align_target="cardboard box",
        )),
    ])

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses)),
            )])

    cloud_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
    )
    lavira = LaViRAClient(
        la_base_url="https://example.invalid/v1",
        va_base_url="https://example.invalid/v1",
        la_client=cloud_client,
        va_client=cloud_client,
        request_context_dir=tmp_path / "lavira_requests",
    )

    result = lavira.alignment_grounding(
        manipulation_prompt=(
            "Collect the bag and put it into the cardboard box, then carry "
            "the cardboard box."
        ),
        direction="front",
        image_bgr=np.zeros((8, 8, 3), np.uint8),
    )

    assert result["target"]["name"] == "cardboard box"
    assert result["yaw_align_target"]["name"] == "cardboard box"
    assert len(calls) == 2
    assert "previous response failed validation" in (
        calls[1]["messages"][0]["content"].lower()
    )
    assert "invalid object schema" in calls[1]["messages"][0]["content"]


def test_lavira_cloud_calls_preserve_role_specific_thinking_mode(tmp_path) -> None:
    calls = []
    responses = iter([
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
            ("front", 0.0),
            ("front_right", -math.pi / 4),
            ("right", -math.pi / 2),
            ("left", math.pi / 2),
            ("front_left", math.pi / 4),
        )
    ]
    move_view = MoveToView(
        4, "basket", image, "reached", "SATISFIED", "basket is near",
    )
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
        manipulation_prompt="Put the medicine bottle into the basket.",
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
        skill="MANIPULATE",
    )

    assert [call["extra_body"] for call in calls] == [
        {"enable_thinking": False},
        {"enable_thinking": False},
        {"enable_thinking": False},
        {"enable_thinking": False},
    ]
    la_system_prompt = calls[0]["messages"][0]["content"]
    assert la_system_prompt.startswith(
        "/no_think\n\nALIGN is forbidden until navigation has completed"
    )
    assert "ALIGN is forbidden until navigation has completed" in la_system_prompt
    assert 'frozen GLOBAL TARGET "basket" unchanged' in la_system_prompt
    assert "intermediate landmark cannot authorize ALIGN" in la_system_prompt
    assert calls[1]["messages"][0]["content"] == "/no_think"
    postcheck_text = calls[3]["messages"][1]["content"][1]["text"]
    assert "SATISFIED -> TASK_COMPLETE" in postcheck_text
    assert "NOT_SATISFIED -> CONTINUE_MANIPULATION" in postcheck_text
    la_content = calls[0]["messages"][1]["content"]
    assert sum(item["type"] == "image_url" for item in la_content) == 6
    la_labels = [
        item["text"] for item in la_content if item["type"] == "text"
    ]
    assert la_labels[0].startswith("**ROLE**: You are an intelligent humanoid")
    assert "**JSON RESPONSE FORMAT**" in la_labels[0]
    assert '**MISSION**: "find basket"' in la_labels[1]
    assert la_labels[2] == 'Navigation Task: "find basket"\n\n- Current Step: 3'
    assert la_labels[3:8] == [
        "Image 1: The current FORWARD view (Step 3).",
        "Image 2: The view 45 deg to the RIGHT of the forward view (Step 3).",
        "Image 3: The view after turning 90 deg to the RIGHT (Step 3).",
        "Image 4: The view 90 deg to the LEFT of the original forward view "
        "(Step 3).",
        "Image 5: The view 45 deg to the LEFT of the forward view (Step 3).",
    ]
    assert la_labels[8] == "PLAN-1"
    plan_image_index = next(
        index for index, item in enumerate(la_content)
        if item.get("text") == "PLAN-1"
    ) - 1
    current_image_index = next(
        index for index, item in enumerate(la_content)
        if item.get("text")
        == "Image 1: The current FORWARD view (Step 3)."
    ) - 1
    assert current_image_index < plan_image_index
    assert current_image_index > 1
    assert "absolute_yaw" not in json.dumps(la_content)
    assert "controller=" not in json.dumps(la_content)
    la_prompt = "\n\n".join(la_labels[:2])
    assert '**MISSION**: "find basket"' in la_prompt
    assert '**FROZEN GLOBAL TARGET**: "basket"' in la_prompt
    assert "**Current Step**: 3" in la_prompt
    assert "**Current TODO List**" in la_prompt
    assert "RECENT SKILLS/RESULTS" not in la_prompt
    assert "FAST-LIO EXPLORATION MEMORY" not in la_prompt
    assert (
        '"view_direction":"front|front_right|right|left|front_left"'
        in la_prompt
    )
    assert "45-degree intermediate views" in la_prompt
    assert "rear/behind direction" in la_prompt
    assert "successful MOVE_TO to the exact GLOBAL TARGET" in la_prompt
    assert "runtime exclusively completes MOVE_TO TODO items" in la_prompt
    assert "Never mark a MOVE_TO item complete yourself" in la_prompt
    assert "exactly one target waypoint" in la_prompt
    assert "never create a standalone turn" in la_prompt
    assert "ALIGN TODO completion remains your responsibility" in la_prompt
    assert "Never select the same completed waypoint again" in la_prompt
    assert "VA independently" in la_prompt
    for index in (1, 3):
        va_content = calls[index]["messages"][1]["content"]
        assert sum(item["type"] == "image_url" for item in va_content) == 1
        va_prompt = va_content[-1]["text"]
        assert '**MISSION**: "find basket"' in va_prompt
        assert '**GLOBAL TARGET**: "basket"' in va_prompt
        assert "**CURRENT STRATEGY**" in va_prompt
        assert "**STRATEGIC STOP SIGNAL**: false" in va_prompt
    alignment_content = calls[2]["messages"][1]["content"]
    assert sum(
        item["type"] == "image_url" for item in alignment_content
    ) == 1
    alignment_prompt_text = alignment_content[-1]["text"]
    assert "**OVERALL MANIPULATION TASK**" in alignment_prompt_text
    assert "Put the medicine bottle into the basket." in alignment_prompt_text
    assert "GLOBAL TARGET" not in alignment_prompt_text
    assert "CURRENT STRATEGY" not in alignment_prompt_text
    assert "STRATEGIC STOP" not in alignment_prompt_text
    postcheck_prompt_text = calls[3]["messages"][1]["content"][-1]["text"]
    assert "latest controller\naction" in postcheck_prompt_text
    assert "after ALIGN" not in postcheck_prompt_text
    assert "skill" not in postcheck_prompt_text.lower()
    assert "Select exactly one `target`" in alignment_prompt_text
    assert "Select exactly one `yaw_align_target`" in alignment_prompt_text
    assert "Both roles may name the same physical object" in alignment_prompt_text
    assert "only\n   supporting surfaces are excluded" in alignment_prompt_text
    assert "prefer that supporting surface over `target` itself" in (
        alignment_prompt_text
    )
    assert "Never select the floor as a supporting yaw target" in (
        alignment_prompt_text
    )
    saved = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / "lavira_requests").glob("*.json"))
    ]
    assert {item["request_kind"] for item in saved} == {
        "la_decision", "va_grounding", "va_align_grounding", "va_postcheck",
    }
    assert all(item["type"] == "sonic.model_request_context" for item in saved)
    assert all(item["attempt"] == 1 for item in saved)
    assert all(item["request"]["messages"] for item in saved)
    assert any(
        "data:image/jpeg;base64," in json.dumps(item["request"])
        for item in saved
    )


def test_first_la_prompt_derives_global_target_without_preloaded_value() -> None:
    prompt = language_action_prompt(
        "walk to the trash can, then go to the desk",
        "vln",
        None,
        1,
        "",
        "panorama",
        None,
        "grasp the bottle and put it in the blue basket on the desk",
    )

    assert "**FROZEN GLOBAL TARGET**" not in prompt
    assert "Infer one GLOBAL TARGET now" in prompt
    assert "runtime freezes this first" in prompt
    assert '"global_target":"..."' in prompt


def test_lavira_retries_invalid_la_json_before_returning_a_skill(tmp_path) -> None:
    calls = []
    responses = iter([
        '{"progress_analysis": "unterminated',
        json.dumps(move("trash can", global_target="trash can")),
    ])

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=next(responses)),
                )],
            )

    cloud_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
    )
    lavira = LaViRAClient(
        la_base_url="https://example.invalid/v1",
        va_base_url="https://example.invalid/v1",
        la_client=cloud_client,
        va_client=cloud_client,
        request_context_dir=tmp_path / "lavira_requests",
    )
    image = np.zeros((8, 8, 3), np.uint8)

    result = lavira.language_action(
        mission="walk to the trash can",
        navigation_mode="vln",
        global_target="trash can",
        current_step=1,
        todo_list="",
        scan_views=[
            ScanView(1, "front", image, (0.0, 0.0, 0.0), 0.0),
        ],
        move_to_views=[],
    )

    assert result["skill"] == "MOVE_TO"
    assert result["skill_args"]["target"] == "trash can"
    assert len(calls) == 2
    assert "RETRY REQUIREMENT" not in calls[0]["messages"][0]["content"]
    assert "RETRY REQUIREMENT" in calls[1]["messages"][0]["content"]
    assert "Return one complete strict JSON object" in (
        calls[1]["messages"][0]["content"]
    )
    saved = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / "lavira_requests").glob("*.json"))
    ]
    assert [item["attempt"] for item in saved] == [1, 2]


def test_lavira_stops_only_after_three_invalid_json_responses(tmp_path) -> None:
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"reasoning": "broken'),
                )],
            )

    cloud_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
    )
    lavira = LaViRAClient(
        la_base_url="https://example.invalid/v1",
        va_base_url="https://example.invalid/v1",
        la_client=cloud_client,
        va_client=cloud_client,
        request_context_dir=tmp_path / "lavira_requests",
    )
    image = np.zeros((8, 8, 3), np.uint8)

    with pytest.raises(LaViRAAgentError, match="LA output is not strict JSON"):
        lavira.language_action(
            mission="walk to the trash can",
            navigation_mode="vln",
            global_target="trash can",
            current_step=1,
            todo_list="",
            scan_views=[
                ScanView(1, "front", image, (0.0, 0.0, 0.0), 0.0),
            ],
            move_to_views=[],
        )

    assert len(calls) == 3


def build_agent(
    decisions, *, groundings=(), alignment_groundings=(),
    postchecks=(),
    max_steps=20, poll_failure=None, events=None, todos=None, depth_mm=2000.0,
    handoff_depth_mm=None, manipulation_prompt=None, sleeps=None,
    trace=None,
    mission="find the basket and put the bottle in it",
    global_target="basket",
    alignment_prompt="Use the basket as both alignment targets.",
):
    camera = FakeCamera(
        depth_mm=depth_mm, handoff_depth_mm=handoff_depth_mm,
    )
    client = FakeClient(
        decisions, groundings, alignment_groundings,
        postchecks,
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
        if name == "navigation_heading_goal":
            heading_delta = float(intents[-1][1]["heading_delta_rad"])
            camera.sonic_yaw = math.remainder(
                camera.sonic_yaw + heading_delta, 2 * math.pi,
            )
            pose_x, pose_y, pose_yaw = camera.pose
            camera.pose = (
                pose_x,
                pose_y,
                math.remainder(pose_yaw + heading_delta, 2 * math.pi),
            )
        reason = "aligned" if name == "start_base_pose" else "reached"
        result = {
            "generation": generation,
            "skill_id": skill_id,
            "segment_id": segment_id,
            "state": "reached",
            "reason": reason,
        }
        if name == "navigation_goal":
            goal_x, goal_y = map(float, intents[-1][1]["goal_base"])
            pose_x, pose_y, pose_yaw = camera.pose
            cosine, sine = math.cos(pose_yaw), math.sin(pose_yaw)
            result["goal_world"] = {
                "x": pose_x + cosine * goal_x - sine * goal_y,
                "y": pose_y + sine * goal_x + cosine * goal_y,
            }
        return result

    def report_event(level, code, message, **fields):
        if events is not None:
            events.append({
                "level": level,
                "code": code,
                "message": message,
                "fields": fields,
            })
        if trace is not None:
            trace.append(("event", code))

    def report_todo(generation, step, todo):
        if todos is not None:
            todos.append((generation, step, todo))
        if trace is not None:
            trace.append(("todo", todo))

    agent = LaViRAAgent(
        navigation_mode="object_nav",
        mission=mission,
        global_target=global_target,
        manipulation_prompt=manipulation_prompt,
        alignment_prompt=alignment_prompt,
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
        sleep=(
            (lambda _seconds: None)
            if sleeps is None else lambda seconds: sleeps.append(seconds)
        ),
        poll_failure=poll_failure,
        report_event=(
            None
            if events is None and trace is None
            else report_event
        ),
        report_todo=(
            None
            if todos is None and trace is None
            else report_todo
        ),
    )
    return agent, camera, client, intents, waited


def test_first_la_response_derives_and_freezes_global_target_for_generation():
    events = []
    inferred = "desk with blue basket"
    agent, _camera, client, _intents, _waited = build_agent(
        [
            move("trash can", global_target=inferred),
            decision(None, result="FAIL", global_target=inferred),
        ],
        groundings=[grounding() for _ in range(3)],
        global_target="",
        events=events,
    )

    result = agent.run(70)

    assert result.reason.startswith("la_fail:")
    assert client.la_calls[0]["global_target"] is None
    assert client.la_calls[1]["global_target"] == inferred
    assert agent.global_target == inferred
    frozen = next(
        event for event in events if event["code"] == "GLOBAL_TARGET_FROZEN"
    )
    assert frozen["fields"]["generation"] == 70
    assert frozen["fields"]["global_target"] == inferred
    agent._reset_task_context()
    assert agent.global_target == ""


def test_later_la_response_cannot_change_frozen_global_target():
    agent, _camera, client, _intents, _waited = build_agent(
        [
            move("trash can", global_target="desk with blue basket"),
            decision(None, result="FAIL", global_target="different desk"),
        ],
        groundings=[grounding() for _ in range(3)],
        global_target="",
    )

    result = agent.run(71)

    assert result.state == "failed"
    assert result.reason.startswith("LA changed the frozen global_target:")
    assert client.la_calls[0]["global_target"] is None
    assert client.la_calls[1]["global_target"] == "desk with blue basket"


def test_protocol_preserves_skill_and_old_messages_default_to_zero():
    new = decode_navigation_message(build_navigation_message(
        mode="heading_goal", generation=3, skill_id=4, segment_id=5,
        timestamp=1.0, heading_delta_rad=math.pi / 2,
        heading_turn_direction="left",
        heading_max_angular_speed_rad_s=0.2,
        heading_max_duration_s=10.0,
    ))
    assert new == NavigationCommand(
        "heading_goal", 3, 1.0, segment_id=5, skill_id=4,
        heading_delta_rad=math.pi / 2,
        heading_turn_direction="left",
        heading_max_angular_speed_rad_s=0.2,
        heading_max_duration_s=10.0,
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
        move("doorway", "front_right"),
        move("desk", "front"),
        move("basket", "right"),
        decision(None, result="FAIL"),
    ]
    sleeps = []
    agent, camera, client, intents, _waited = build_agent(
        steps,
        groundings=[grounding() for _ in range(9)],
        sleeps=sleeps,
    )
    result = agent.run(7)
    assert result.state == "failed" and result.reason.startswith("la_fail")
    assert client.la_calls[0]["todo_list"] == ""
    assert len(client.la_calls[0]["scan_views"]) == 5
    names = [name for name, _ in intents]
    assert names.count("navigation_goal") == 3
    assert names.count("navigation_heading_goal") == 5 * 3 + 3 + 3
    heading_deltas = [
        args["heading_delta_rad"]
        for name, args in intents if name == "navigation_heading_goal"
    ]
    assert heading_deltas[:6] == pytest.approx(
        [
            -math.pi / 4,
            -math.pi / 4,
            math.pi,
            -math.pi / 4,
            -math.pi / 4,
            -math.pi / 4,
        ]
    )
    assert camera.capture_yaws[:5] == pytest.approx([
        0.0, -math.pi / 4, -math.pi / 2, math.pi / 2, math.pi / 4,
    ])
    heading_count = names.count("navigation_heading_goal")
    assert len(sleeps) == 30 * heading_count
    assert sleeps == pytest.approx([1.0 / 30.0] * len(sleeps))
    assert sum(sleeps) == pytest.approx(float(heading_count))
    first_goal = names.index("navigation_goal")
    assert names[:first_goal].count("navigation_heading_goal") == 6
    assert names.index("lavira_depth_request") > max(
        index for index, name in enumerate(names[:first_goal])
        if name == "navigation_heading_goal"
    )
    assert camera.depth_count == 21
    skill_ids = [args["skill_id"] for name, args in intents if name.startswith("navigation_")]
    assert skill_ids == sorted(skill_ids)


def test_move_to_facing_geometry_uses_unrounded_fastlio_pose() -> None:
    facing = _move_to_facing_geometry(
        (2.12349, -1.45678),
        (1.02341, -1.05672, math.pi / 2.0),
    )

    expected_yaw = math.atan2(-0.40006, 1.10008)
    assert facing["fastlio_x"] == pytest.approx(1.02341)
    assert facing["fastlio_y"] == pytest.approx(-1.05672)
    assert facing["target_yaw_rad"] == pytest.approx(expected_yaw)
    assert facing["heading_delta_rad"] == pytest.approx(
        math.remainder(expected_yaw - math.pi / 2.0, 2.0 * math.pi)
    )


def test_move_to_facing_geometry_uses_shortest_turn_across_wraparound() -> None:
    target_yaw = math.pi - 0.1
    facing = _move_to_facing_geometry(
        (math.cos(target_yaw), math.sin(target_yaw)),
        (0.0, 0.0, -math.pi + 0.1),
    )

    assert facing["target_yaw_rad"] == pytest.approx(target_yaw)
    assert facing["heading_delta_rad"] == pytest.approx(-0.2)


def test_move_to_facing_geometry_rejects_zero_length_target_direction() -> None:
    with pytest.raises(
        LaViRAAgentError, match="move_to_target_direction_undefined"
    ):
        _move_to_facing_geometry(
            (1.25, -3.5),
            (1.25, -3.5, 0.7),
        )


def test_move_to_faces_exact_world_goal_before_post_navigation_capture():
    events = []
    agent, camera, _client, intents, _waited = build_agent(
        [move(), decision(None, result="FAIL")],
        groundings=[grounding() for _ in range(3)],
        events=events,
    )
    camera.pose = (1.1234, -2.5678, 2.9)
    camera.sonic_yaw = -0.7
    normal_wait = agent.wait_status
    nav_fastlio_yaw = None
    nav_sonic_yaw = None
    desired_yaw = -3.0

    def return_exact_world_goal(generation, skill_id, segment_id, timeout):
        nonlocal nav_fastlio_yaw, nav_sonic_yaw
        if intents[-1][0] == "navigation_goal":
            pose_x, pose_y, nav_fastlio_yaw = camera.pose
            nav_sonic_yaw = camera.sonic_yaw
            return {
                "generation": generation,
                "skill_id": skill_id,
                "segment_id": segment_id,
                "state": "reached",
                "reason": "goal_within_2m",
                "goal_world": {
                    "x": pose_x + math.cos(desired_yaw),
                    "y": pose_y + math.sin(desired_yaw),
                },
            }
        return normal_wait(generation, skill_id, segment_id, timeout)

    agent.wait_status = return_exact_world_goal
    result = agent.run(72)

    assert result.reason.startswith("la_fail:")
    assert nav_fastlio_yaw is not None and nav_sonic_yaw is not None
    expected_delta = math.remainder(
        desired_yaw - nav_fastlio_yaw, 2.0 * math.pi,
    )
    names = [name for name, _args in intents]
    nav_index = names.index("navigation_goal")
    facing_index = names.index("navigation_heading_goal", nav_index + 1)
    capture_index = names.index("lavira_depth_request", nav_index + 1)
    assert nav_index < facing_index < capture_index
    facing_intent = intents[facing_index][1]
    assert facing_intent["heading_delta_rad"] == pytest.approx(expected_delta)
    assert camera.sonic_yaw == pytest.approx(
        math.remainder(nav_sonic_yaw + expected_delta, 2.0 * math.pi)
    )
    assert camera.leases[-1] == (
        72, facing_intent["skill_id"], facing_intent["segment_id"],
    )
    assert any(
        event["code"] == "MOVE_TO_TARGET_FACING_COMPLETED"
        for event in events
    )


def test_move_to_terminal_facing_failure_warns_and_continues_handoff():
    events = []
    agent, _camera, _client, intents, _waited = build_agent(
        [move(), decision(None, result="FAIL")],
        groundings=[grounding() for _ in range(3)],
        events=events,
    )
    normal_wait = agent.wait_status
    fail_next_heading = False

    def fail_facing(generation, skill_id, segment_id, timeout):
        nonlocal fail_next_heading
        if intents[-1][0] == "navigation_goal":
            fail_next_heading = True
            return normal_wait(generation, skill_id, segment_id, timeout)
        if intents[-1][0] == "navigation_heading_goal" and fail_next_heading:
            fail_next_heading = False
            return {
                "generation": generation,
                "skill_id": skill_id,
                "segment_id": segment_id,
                "state": "failed",
                "reason": "sonic_orientation_timeout",
            }
        return normal_wait(generation, skill_id, segment_id, timeout)

    agent.wait_status = fail_facing
    result = agent.run(73)

    assert result.reason.startswith("la_fail:")
    assert agent._history[0].controller_state == "reached"
    failure = next(
        event for event in events
        if event["code"] == "MOVE_TO_TARGET_FACING_FAILED"
    )
    assert failure["fields"]["controller_reason"] == (
        "sonic_orientation_timeout"
    )
    names = [name for name, _args in intents]
    nav_index = names.index("navigation_goal")
    failed_heading_index = names.index("navigation_heading_goal", nav_index + 1)
    capture_index = names.index("lavira_depth_request", nav_index + 1)
    assert failed_heading_index < capture_index
    assert intents[capture_index][1]["segment_id"] == (
        intents[failed_heading_index][1]["segment_id"]
    )


def test_move_to_missing_world_goal_warns_and_continues_without_facing():
    events = []
    agent, _camera, _client, intents, _waited = build_agent(
        [move(), decision(None, result="FAIL")],
        groundings=[grounding() for _ in range(3)],
        events=events,
    )
    normal_wait = agent.wait_status

    def omit_world_goal(generation, skill_id, segment_id, timeout):
        status = dict(normal_wait(generation, skill_id, segment_id, timeout))
        if intents[-1][0] == "navigation_goal":
            status.pop("goal_world", None)
        return status

    agent.wait_status = omit_world_goal
    result = agent.run(74)

    assert result.reason.startswith("la_fail:")
    failure = next(
        event for event in events
        if event["code"] == "MOVE_TO_TARGET_FACING_FAILED"
    )
    assert failure["fields"]["error"] == "move_to_goal_world_unavailable"
    names = [name for name, _args in intents]
    nav_index = names.index("navigation_goal")
    assert names[nav_index + 1] == "lavira_depth_request"


def test_move_to_facing_wait_timeout_still_terminates_task():
    agent, _camera, _client, intents, _waited = build_agent(
        [move()],
        groundings=[grounding()],
    )
    normal_wait = agent.wait_status
    timeout_next_heading = False

    def timeout_facing(generation, skill_id, segment_id, timeout):
        nonlocal timeout_next_heading
        if intents[-1][0] == "navigation_goal":
            timeout_next_heading = True
            return normal_wait(generation, skill_id, segment_id, timeout)
        if intents[-1][0] == "navigation_heading_goal" and timeout_next_heading:
            raise TimeoutError("move_to_facing_timeout")
        return normal_wait(generation, skill_id, segment_id, timeout)

    agent.wait_status = timeout_facing
    result = agent.run(75)

    assert result.state == "failed"
    assert result.reason == "move_to_facing_timeout"
    names = [name for name, _args in intents]
    nav_index = names.index("navigation_goal")
    facing_index = names.index("navigation_heading_goal", nav_index + 1)
    assert "lavira_depth_request" not in names[facing_index + 1:]


def test_move_to_facing_stale_segment_status_still_terminates_task():
    agent, _camera, _client, intents, _waited = build_agent(
        [move()],
        groundings=[grounding()],
    )
    normal_wait = agent.wait_status
    stale_next_heading = False

    def stale_facing(generation, skill_id, segment_id, timeout):
        nonlocal stale_next_heading
        if intents[-1][0] == "navigation_goal":
            stale_next_heading = True
            return normal_wait(generation, skill_id, segment_id, timeout)
        if intents[-1][0] == "navigation_heading_goal" and stale_next_heading:
            return {
                "generation": generation,
                "skill_id": skill_id,
                "segment_id": segment_id + 1,
                "state": "reached",
                "reason": "heading_sonic_yaw_reached",
            }
        return normal_wait(generation, skill_id, segment_id, timeout)

    agent.wait_status = stale_facing
    result = agent.run(76)

    assert result.state == "failed"
    assert result.reason == "stale segment status"
    names = [name for name, _args in intents]
    nav_index = names.index("navigation_goal")
    facing_index = names.index("navigation_heading_goal", nav_index + 1)
    assert "lavira_depth_request" not in names[facing_index + 1:]


def test_panorama_uses_sonic_measured_yaw_not_fastlio_yaw():
    sleeps = []
    agent, camera, client, intents, _waited = build_agent(
        [decision(None, result="FAIL")],
        sleeps=sleeps,
    )
    camera.pose = (1.0, 2.0, 2.4)
    camera.sonic_yaw = 0.3

    result = agent.run(8)

    assert result.state == "failed"
    heading_deltas = [
        args["heading_delta_rad"]
        for name, args in intents if name == "navigation_heading_goal"
    ]
    assert heading_deltas == pytest.approx([
        -math.pi / 4,
        -math.pi / 4,
        math.pi,
        -math.pi / 4,
        -math.pi / 4,
    ])
    heading_intents = [
        args for name, args in intents if name == "navigation_heading_goal"
    ]
    assert [args.get("heading_turn_direction") for args in heading_intents] == [
        None, None, "left", None, None,
    ]
    views = client.la_calls[0]["scan_views"]
    assert views[0].reference_pose == pytest.approx((1.0, 2.0, 0.3))
    assert [view.direction for view in views] == [
        "front", "front_right", "right", "left", "front_left",
    ]
    assert [view.absolute_yaw_rad for view in views] == pytest.approx([
        0.3,
        0.3 - math.pi / 4,
        0.3 - math.pi / 2,
        0.3 + math.pi / 2,
        0.3 + math.pi / 4,
    ])
    assert len(sleeps) == 30 * 5
    assert sleeps == pytest.approx([1.0 / 30.0] * len(sleeps))
    assert sum(sleeps) == pytest.approx(5.0)


def test_heading_settle_requests_point_two_radian_fine_correction():
    events = []
    agent, camera, _client, intents, _waited = build_agent(
        [decision(None, result="FAIL")],
        events=events,
    )
    target_yaw = -math.pi / 2.0
    samples = 0

    def drift_during_first_window(_seconds):
        nonlocal samples
        samples += 1
        if samples <= 30:
            camera.sonic_yaw = math.remainder(
                target_yaw - math.radians(8.0), 2 * math.pi,
            )

    agent.sleep = drift_during_first_window
    agent._face_absolute_yaw(1, 1, target_yaw)

    headings = [
        args for name, args in intents if name == "navigation_heading_goal"
    ]
    assert len(headings) == 2
    assert headings[0]["heading_delta_rad"] == pytest.approx(target_yaw)
    assert headings[1]["heading_delta_rad"] == pytest.approx(math.radians(8.0))
    assert headings[1]["heading_max_angular_speed_rad_s"] == pytest.approx(0.2)
    assert 0.0 < headings[1]["heading_max_duration_s"] <= 10.0
    assert camera.sonic_yaw == pytest.approx(target_yaw)
    assert any(event["code"] == "HEADING_SETTLE_CORRECTION" for event in events)
    assert events[-1]["code"] == "HEADING_SETTLE_COMPLETED"


def test_heading_settle_time_limit_allows_the_next_capture():
    events = []
    agent, camera, _client, intents, _waited = build_agent(
        [decision(None, result="FAIL")],
        events=events,
    )
    target_yaw = -math.pi / 2.0
    clock = 0.0

    def monotonic():
        return clock

    def persistent_drift(seconds):
        nonlocal clock
        clock += seconds
        camera.sonic_yaw = math.remainder(
            target_yaw - math.radians(8.0), 2 * math.pi,
        )

    agent.monotonic = monotonic
    agent.sleep = persistent_drift
    status = agent._face_absolute_yaw(1, 1, target_yaw)

    headings = [
        args for name, args in intents if name == "navigation_heading_goal"
    ]
    assert status["state"] == "reached"
    assert len(headings) > 2
    assert all(
        args["heading_max_angular_speed_rad_s"] == pytest.approx(0.2)
        for args in headings[1:]
    )
    assert clock <= 11.0 + 1.0 / 30.0 + 1.0e-9
    assert events[-1]["code"] == "HEADING_SETTLE_TIME_LIMIT"


def test_panorama_fails_closed_without_sonic_measured_yaw():
    agent, camera, _client, intents, _waited = build_agent([
        decision(None, result="FAIL"),
    ])
    camera.current_sonic_yaw = lambda: math.nan

    result = agent.run(9)

    assert result.state == "failed"
    assert result.reason == "sonic_measured_yaw_invalid"
    assert not any(name == "navigation_heading_goal" for name, _ in intents)


def test_la_context_keeps_fresh_panorama_and_last_five_completed_moves():
    moves = [move(f"landmark-{index}") for index in range(6)]
    agent, _camera, client, _intents, _waited = build_agent(
        [*moves, decision(None, result="FAIL")],
        groundings=[grounding() for _ in range(3 * len(moves))],
    )

    result = agent.run(17)

    assert result.state == "failed"
    final_context = client.la_calls[-1]
    assert final_context["current_step"] == 7
    assert len(final_context["scan_views"]) == 5
    assert [view.direction for view in final_context["scan_views"]] == [
        "front", "front_right", "right", "left", "front_left",
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


def test_manipulation_postchecks_and_unknown_retry_use_fresh_head_views():
    agent, camera, client, _intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding()] * 3,
        alignment_groundings=[alignment_grounding()],
        postchecks=[ready_to_manipulate()] * 2 + [
            postcheck("NOT_SATISFIED"), postcheck("UNKNOWN"), task_complete(),
        ],
    )

    result = agent.run(18)

    assert result.state == "reached"
    checks = [c for c in client.postcheck_calls if c.get("skill") == "MANIPULATE"]
    assert len(checks) == 3
    pixels = [int(c["image_bgr"][0, 0, 0]) for c in checks]
    assert all(p > 100 for p in pixels), "All manipulation checks must use the head camera"
    assert len(set(pixels)) == 3, "Each retry must capture a fresh image"


def test_va_context_routes_manipulation_prompt_to_align_and_handoff():
    agent, _camera, client, _intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding(), grounding(), grounding()],
        alignment_groundings=[alignment_grounding()],
        postchecks=[
            ready_to_manipulate(), ready_to_manipulate(), task_complete(),
        ],
    )

    result = agent.run(18)

    assert result.state == "reached"
    assert result.steps == 3
    assert len(client.la_calls) == 3
    calls = [*client.grounding_calls, *client.postcheck_calls]
    assert all(
        call["global_target"] == "basket"
        and call["strategic_goal"]
        for call in calls
    )
    assert all(
        call["mission"] == "find the basket and put the bottle in it"
        for call in client.grounding_calls
    )
    assert all(
        call["mission"] == "find the basket and put the bottle in it"
        for call in client.postcheck_calls[:2]
    )
    assert all(not call["strategic_stop"] for call in client.grounding_calls)
    assert len(client.alignment_grounding_calls) == 1
    assert set(client.alignment_grounding_calls[0]) == {
        "manipulation_prompt", "direction", "image_bgr",
    }
    assert client.alignment_grounding_calls[0]["manipulation_prompt"] == (
        "find the basket and put the bottle in it"
    )
    assert [call["strategic_stop"] for call in client.postcheck_calls] == [
        False, False, True,
    ]
    assert [call.get("skill") for call in client.postcheck_calls] == [
        None, None, "MANIPULATE",
    ]
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
    assert len(client.la_calls[1]["scan_views"]) == 5
    assert len(client.la_calls[2]["scan_views"]) == 5
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
        (22, 1, "- [ ] complete mission"),
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
    assert len(client.la_calls[1]["scan_views"]) == 5
    assert client.la_calls[1]["scan_views"][0].scan_id == 2
    assert agent._history[-1].skill == "MOVE_TO"


def test_move_outside_handoff_depth_does_not_enter_la_image_history():
    agent, _camera, client, _intents, _waited = build_agent(
        [move(), decision(None, result="FAIL")],
        groundings=[grounding(), grounding(), grounding()],
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
        groundings=[grounding() for _ in range(6)],
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
    assert base_pose["yaw_align_target"] == "desk"
    assert base_pose["reference_bbox"] == [200, 200, 800, 800]
    assert agent._history[1].controller_state == "target_not_found"


def test_ready_nav_allows_another_move_and_uses_front_observation():
    agent, _camera, client, intents, _waited = build_agent(
        [move(), move("another landmark"), decision(None, result="FAIL")],
        groundings=[grounding() for _ in range(6)],
    )

    result = agent.run(7)

    assert result.reason.startswith("la_fail")
    assert len(client.la_calls[0]["scan_views"]) == 5
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
        groundings=[grounding(), grounding(), grounding()],
        handoff_depth_mm=0.0,
    )

    result = agent.run(39)

    assert result.state == "failed"
    assert result.reason == (
        "handoff_gate:global_target_navigation_not_ready_for_ALIGN"
    )
    assert [name for name, _args in intents].count("start_base_pose") == 0


def test_intermediate_navigation_target_cannot_open_align():
    events = []
    agent, _camera, client, intents, _waited = build_agent(
        [
            move("trash can", global_target="desk with blue basket"),
            align(global_target="desk with blue basket"),
        ],
        groundings=[grounding() for _ in range(3)],
        events=events,
        global_target="desk with blue basket",
    )

    result = agent.run(41)

    assert result.state == "failed"
    assert result.reason == (
        "handoff_gate:global_target_navigation_not_ready_for_ALIGN"
    )
    assert agent._latest_transition is not None
    assert agent._latest_transition["transition"] == "CONTINUE_NAVIGATION"
    assert len(client.la_calls[1]["scan_views"]) == 5
    assert not any(name == "start_base_pose" for name, _args in intents)
    handoff = next(
        event for event in events
        if event["code"] == "NAV_HANDOFF_EVALUATED"
    )
    assert handoff["fields"]["target_is_global_target"] is False
    assert handoff["fields"]["global_target_navigation_ready"] is False
    assert "intermediate navigation target" in handoff["fields"][
        "visual_evidence"
    ]


def test_align_grounding_can_select_operation_target_independent_of_global_target():
    agent, _camera, _client, intents, _waited = build_agent(
        [
            move(
                "desk with blue basket",
                global_target="desk with blue basket",
            ),
            align(global_target="desk with blue basket"),
            decision(
                None,
                result="FAIL",
                global_target="desk with blue basket",
            ),
        ],
        groundings=[grounding() for _ in range(3)],
        alignment_groundings=[alignment_grounding(
            target="medicine bottle", yaw_align_target="desk",
        )],
        postchecks=[ready_to_manipulate(), ready_to_manipulate()],
        global_target="desk with blue basket",
    )

    result = agent.run(42)

    assert result.reason.startswith("la_fail:")
    base_pose = next(
        args for name, args in intents if name == "start_base_pose"
    )
    assert base_pose["target"] == "medicine bottle"
    assert base_pose["yaw_align_target"] == "desk"


def test_base_pose_target_comes_from_align_va_grounding():
    agent, _camera, client, intents, _waited = build_agent(
        [
            move(
                "desk with blue basket",
                global_target="desk with blue basket",
            ),
            align(global_target="desk with blue basket"),
            decision(
                None,
                result="FAIL",
                global_target="desk with blue basket",
            ),
        ],
        groundings=[grounding() for _ in range(3)],
        alignment_groundings=[alignment_grounding(
            target="  blue basket  ", yaw_align_target="  desk  ",
        )],
        postchecks=[ready_to_manipulate(), ready_to_manipulate()],
        global_target="desk with blue basket",
    )

    result = agent.run(43)

    assert result.reason.startswith("la_fail:")
    base_pose = next(
        args for name, args in intents if name == "start_base_pose"
    )
    assert base_pose["target"] == "blue basket"
    assert base_pose["yaw_align_target"] == "desk"
    assert set(client.alignment_grounding_calls[0]) == {
        "manipulation_prompt", "direction", "image_bgr",
    }


def test_failed_base_pose_does_not_run_an_intra_align_fallback():
    events = []
    agent, _camera, _client, intents, _waited = build_agent(
        [move(), align(), decision(None, result="FAIL")],
        groundings=[grounding() for _ in range(3)],
        alignment_groundings=[alignment_grounding(
            target="cardboard box",
            yaw_align_target="cardboard box",
        )],
        postchecks=[ready_to_manipulate(), ready_to_manipulate()],
        events=events,
    )
    normal_wait = agent.wait_status

    def fail_base_pose(generation, skill_id, segment_id, timeout):
        if intents[-1][0] == "start_base_pose":
            return {
                "generation": generation,
                "skill_id": skill_id,
                "segment_id": segment_id,
                "state": "failed",
                "reason": "YOLOE text prompt found no target",
            }
        return normal_wait(generation, skill_id, segment_id, timeout)

    agent.wait_status = fail_base_pose

    result = agent.run(44)

    assert result.reason.startswith("la_fail:")
    attempts = [
        args for name, args in intents if name == "start_base_pose"
    ]
    assert attempts == [{
        "generation": 44,
        "skill_id": 2,
        "segment_id": attempts[0]["segment_id"],
        "target": "cardboard box",
        "yaw_align_target": "cardboard box",
        "reference_bbox": [200.0, 200.0, 800.0, 800.0],
    }]
    assert not any(
        event["code"] == "BASE_POSE_TARGET_FALLBACK" for event in events
    )


def test_nav_handoff_accepts_head_when_chest_does_not_pass():
    events = []
    agent, camera, client, _intents, _waited = build_agent(
        [move(), decision(None, result="FAIL")],
        groundings=[
            grounding(),
            grounding("NOT_FOUND", "chest cannot see basket"),
            grounding("FOUND", "head can see basket"),
        ],
        events=events,
    )

    result = agent.run(40)

    assert result.state == "failed"
    assert agent._history[-1].va_result == "SATISFIED"
    assert len(client.grounding_calls) == 3
    assert camera.rgbd_streams[-2:] == ["chest_view", "ego_view"]
    handoff = next(
        event for event in events
        if event["code"] == "NAV_HANDOFF_EVALUATED"
    )
    assert handoff["fields"]["chest_status"] == "NOT_SATISFIED"
    assert handoff["fields"]["head_status"] == "SATISFIED"
    assert handoff["fields"]["ready_view"] == "head"
    assert handoff["fields"]["transition"] == "READY_TO_ALIGN"


@pytest.mark.parametrize(
    ("chest_grounding", "head_grounding", "ready_view"),
    [
        ("FOUND", "NOT_FOUND", "chest"),
        ("NOT_FOUND", "FOUND", "head"),
        ("FOUND", "FOUND", "both"),
    ],
)
def test_runtime_completes_move_to_todo_when_either_nav_view_is_satisfied(
    chest_grounding,
    head_grounding,
    ready_view,
):
    open_todo = "- [ ] approach the basket\n- [ ] align with the basket"
    completed_todo = (
        "- [x] approach the basket Result: Runtime VA confirmed MOVE_TO target "
        '"basket" as SATISFIED.\n'
        "- [ ] align with the basket"
    )
    todos = []
    events = []
    trace = []
    agent, _camera, client, _intents, _waited = build_agent(
        [
            move(todo_list=open_todo),
            decision(None, result="FAIL", todo_list=completed_todo),
        ],
        groundings=[
            grounding("FOUND", "basket"),
            grounding(chest_grounding, "basket"),
            grounding(head_grounding, "basket"),
        ],
        todos=todos,
        events=events,
        trace=trace,
    )

    result = agent.run(50)

    assert result.reason.startswith("la_fail:")
    assert client.la_calls[1]["todo_list"] == completed_todo
    assert [item[2] for item in todos] == [open_todo, completed_todo]
    assert trace.count(("todo", completed_todo)) == 1
    assert trace.index(("event", "SKILL_COMPLETED")) < trace.index(
        ("todo", completed_todo)
    )
    handoff = next(
        event for event in events
        if event["code"] == "NAV_HANDOFF_EVALUATED"
    )
    assert handoff["fields"]["status"] == "SATISFIED"
    assert handoff["fields"]["transition"] == "READY_TO_ALIGN"
    assert handoff["fields"]["ready_view"] == ready_view


@pytest.mark.parametrize("unknown", [False, True])
def test_runtime_leaves_move_to_todo_open_without_satisfied_nav_view(unknown):
    open_todo = "- [ ] approach the basket\n- [ ] align with the basket"
    if unknown:
        handoff_groundings = [
            grounding("FOUND", "basket"),
            grounding("FOUND", "basket"),
        ]
    else:
        handoff_groundings = [
            grounding("NOT_FOUND", "basket"),
            grounding("NOT_FOUND", "basket"),
        ]
    todos = []
    events = []
    agent, _camera, client, _intents, _waited = build_agent(
        [
            move(todo_list=open_todo),
            decision(None, result="FAIL", todo_list=open_todo),
        ],
        groundings=[grounding("FOUND", "basket"), *handoff_groundings],
        handoff_depth_mm=0.0 if unknown else None,
        todos=todos,
        events=events,
    )

    result = agent.run(51)

    assert result.reason.startswith("la_fail:")
    assert client.la_calls[1]["todo_list"] == open_todo
    assert [item[2] for item in todos] == [open_todo]
    handoff = next(
        event for event in events
        if event["code"] == "NAV_HANDOFF_EVALUATED"
    )
    assert handoff["fields"]["status"] == (
        "UNKNOWN" if unknown else "NOT_SATISFIED"
    )


def test_intermediate_move_to_completion_advances_black_to_dark_blue_trash_can():
    global_target = "desk with blue basket"
    initial_todo = (
        "- [ ] approach the black trash can\n"
        "- [ ] approach the dark blue trash can\n"
        "- [ ] approach the desk with blue basket"
    )
    black_completed = (
        "- [x] approach the black trash can Result: Runtime VA confirmed "
        'MOVE_TO target "black trash can" as SATISFIED.\n'
        "- [ ] approach the dark blue trash can\n"
        "- [ ] approach the desk with blue basket"
    )
    dark_blue_completed = (
        "- [x] approach the black trash can Result: Runtime VA confirmed "
        'MOVE_TO target "black trash can" as SATISFIED.\n'
        "- [x] approach the dark blue trash can Result: Runtime VA confirmed "
        'MOVE_TO target "dark blue trash can" as SATISFIED.\n'
        "- [ ] approach the desk with blue basket"
    )
    todos = []
    events = []
    agent, _camera, client, _intents, _waited = build_agent(
        [
            move(
                "black trash can",
                global_target=global_target,
                todo_list=initial_todo,
            ),
            move(
                "dark blue trash can",
                global_target=global_target,
                todo_list=black_completed,
            ),
            decision(
                None,
                result="FAIL",
                global_target=global_target,
                todo_list=dark_blue_completed,
            ),
        ],
        groundings=[
            grounding("FOUND", "black trash can"),
            grounding("FOUND", "black trash can"),
            grounding("NOT_FOUND", "black trash can"),
            *[grounding("FOUND", "dark blue trash can") for _ in range(3)],
        ],
        global_target=global_target,
        todos=todos,
        events=events,
    )

    result = agent.run(52)

    assert result.reason.startswith("la_fail:")
    assert client.la_calls[1]["todo_list"] == black_completed
    assert client.la_calls[2]["todo_list"] == dark_blue_completed
    assert [call["target"] for call in client.grounding_calls] == [
        "black trash can",
        "black trash can",
        "black trash can",
        "dark blue trash can",
        "dark blue trash can",
        "dark blue trash can",
    ]
    assert [item[2] for item in todos] == [
        initial_todo,
        black_completed,
        dark_blue_completed,
    ]
    handoffs = [
        event for event in events
        if event["code"] == "NAV_HANDOFF_EVALUATED"
    ]
    assert [event["fields"]["transition"] for event in handoffs] == [
        "CONTINUE_NAVIGATION",
        "CONTINUE_NAVIGATION",
    ]
    assert handoffs[0]["fields"]["chest_status"] == "SATISFIED"
    assert handoffs[0]["fields"]["head_status"] == "NOT_SATISFIED"


def test_align_handoff_accepts_one_complete_camera_view():
    events = []
    agent, _camera, client, intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding(), grounding(), grounding()],
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
    assert all(
        'distance-and-centering target "basket"' in call["expected_postcondition"]
        and 'yaw-alignment target "desk"' in call["expected_postcondition"]
        for call in align_checks
    )
    assert [name for name, _args in intents].count("start_vla_task") == 1


def test_align_handoff_never_combines_partial_visibility_across_cameras():
    agent, _camera, _client, intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding(), grounding(), grounding()],
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
    assert all(len(call["scan_views"]) == 5 for call in client.la_calls)


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
    static_prompt = (
        "Move in front of the desk with the blue basket, grasp the medicine "
        "bottle, and place it into the blue basket."
    )
    agent, _camera, client, intents, _waited = build_agent(
        [move(), align(), decision("MANIPULATE")],
        groundings=[grounding(), grounding(), grounding()],
        alignment_groundings=[alignment_grounding()],
        postchecks=[
            ready_to_manipulate(), ready_to_manipulate(),
            postcheck(
                "NOT_SATISFIED", "bottle still on desk",
                "CONTINUE_MANIPULATION",
            ),
            task_complete("bottle is in basket"),
        ],
        manipulation_prompt=static_prompt,
    )
    result = agent.run(7)
    assert result.state == "reached"
    assert result.steps == 3
    names = [name for name, _ in intents]
    assert names.count("start_vla_task") == 1
    assert names.count("hold_vla_task") == 0
    assert names.count("resume_vla_task") == 0
    assert names.count("stop_vla_task") == 1
    start = next(args for name, args in intents if name == "start_vla_task")
    assert start["task"] == static_prompt
    assert start["handoff_context"] == static_prompt
    assert all(
        call["mission"] == "find the basket and put the bottle in it"
        for call in client.grounding_calls
    )
    assert all(
        call["manipulation_prompt"] == static_prompt
        for call in client.la_calls
    )
    assert set(client.alignment_grounding_calls[0]) == {
        "manipulation_prompt", "direction", "image_bgr",
    }
    assert client.alignment_grounding_calls[0]["manipulation_prompt"] == (
        static_prompt
    )
    assert all(
        call["mission"] == static_prompt
        for call in client.postcheck_calls[:2]
    )
    assert all(
        call["mission"] == static_prompt
        for call in client.postcheck_calls[2:]
    )
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
        groundings=[grounding(), grounding(), grounding()],
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
        groundings=[grounding(), grounding(), grounding()],
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
        groundings=[grounding(), grounding(), grounding()],
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
        groundings=[grounding(), grounding(), grounding()],
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


def test_align_grounding_same_object_canonicalizes_detector_text() -> None:
    result = validate_alignment_grounding(alignment_grounding(
        target="Cardboard Box",
        yaw_align_target="cardboard   box",
    ))

    assert result["target"]["name"] == "Cardboard Box"
    assert result["yaw_align_target"]["name"] == "Cardboard Box"
    assert result["target"]["bbox_2d"] == [200.0, 200.0, 800.0, 800.0]
    assert result["confidence"] == pytest.approx(0.9)


def test_align_grounding_accepts_different_target_roles() -> None:
    result = validate_alignment_grounding(alignment_grounding(
        target="medicine bottle",
        yaw_align_target="desk",
        target_confidence=0.95,
        yaw_align_target_confidence=0.81,
    ))

    assert result["target"]["name"] == "medicine bottle"
    assert result["yaw_align_target"]["name"] == "desk"
    assert result["confidence"] == pytest.approx(0.81)


def test_align_grounding_not_found_allows_visible_low_confidence_roles() -> None:
    result = validate_alignment_grounding(alignment_grounding(
        status="NOT_FOUND",
        target="cardboard box",
        yaw_align_target="cardboard box",
        target_visible=True,
        yaw_align_target_visible=True,
        target_confidence=0.2,
        yaw_align_target_confidence=0.1,
    ))

    assert result["status"] == "NOT_FOUND"
    assert result["confidence"] == pytest.approx(0.1)


def test_align_runtime_applies_minimum_role_confidence_gate() -> None:
    agent, _camera, _client, _intents, _waited = build_agent(
        [],
        alignment_groundings=[alignment_grounding(
            target="cardboard box",
            yaw_align_target="cardboard box",
            target_confidence=0.95,
            yaw_align_target_confidence=0.59,
        )],
    )

    result = agent._alignment_ground(
        generation=7,
        skill_id=1,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )

    assert result["status"] == "NOT_FOUND"
    assert result["confidence"] == pytest.approx(0.59)


def test_align_grounding_prompt_infers_roles_from_manipulation_task() -> None:
    prompt = alignment_grounding_prompt(
        manipulation_prompt=(
            "Collect the bag and put it into the cardboard box, then carry "
            "the cardboard box."
        ),
        direction="front",
    )

    assert "Use only OVERALL MANIPULATION TASK and the current image" in prompt
    assert "Collect the bag and put it into the cardboard box" in prompt
    assert "Select exactly one `target`" in prompt
    assert "destination\n   containers or receptacles" in prompt
    assert "only\n   supporting surfaces are excluded" in prompt
    assert "Select exactly one `yaw_align_target`" in prompt
    assert "prefer that supporting surface over `target` itself" in prompt
    assert "Never select the floor as a supporting yaw target" in prompt
    assert "Both roles may name the same physical object" in prompt
    assert "GLOBAL TARGET" not in prompt
    assert "CURRENT STRATEGY" not in prompt
    assert "STRATEGIC STOP" not in prompt


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
    normalized = validate_postcheck(
        postcheck("SATISFIED", transition="READY_TO_MANIPULATE"),
        skill="MANIPULATE",
    )
    assert normalized["transition"] == "TASK_COMPLETE"
    normalized = validate_postcheck(
        postcheck("NOT_SATISFIED", transition="READY_TO_MANIPULATE"),
        skill="MANIPULATE",
    )
    assert normalized["transition"] == "CONTINUE_MANIPULATION"
    with pytest.raises(LaViRAAgentError, match="schema"):
        validate_alignment_grounding({**alignment_grounding(), "extra": True})
    legacy = alignment_grounding()
    legacy["surface"] = "desk"
    with pytest.raises(LaViRAAgentError, match="schema"):
        validate_alignment_grounding(legacy)
    with pytest.raises(LaViRAAgentError, match="stable target bbox"):
        validate_alignment_grounding(alignment_grounding(
            target="medicine bottle",
            yaw_align_target="desk",
            bbox=[925, 406, 971, 513],
        ))
    with pytest.raises(LaViRAAgentError, match="corner ordering"):
        validate_alignment_grounding(alignment_grounding(
            bbox=[900, 100, 100, 900],
        ))
    with pytest.raises(LaViRAAgentError, match="English detector"):
        validate_alignment_grounding(alignment_grounding(target="纸箱"))
    with pytest.raises(LaViRAAgentError, match="confidence"):
        validate_alignment_grounding(alignment_grounding(
            yaw_align_target_confidence=math.inf,
        ))
    with pytest.raises(LaViRAAgentError, match="decision"):
        validate_language_action(decision(None, result="COMPLETE"))
    with pytest.raises(LaViRAAgentError, match="view_direction is invalid"):
        validate_language_action(
            move(direction="behind"),
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
        groundings=[grounding(), grounding(), grounding()],
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
    assert [name for name, _args in intents].count("navigation_heading_goal") == 8
    assert events[-1]["code"] == "TASK_COMPLETED"
