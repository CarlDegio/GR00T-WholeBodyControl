"""Observation-grounded manipulation agent using separate LA and VA roles.

The Language Agent owns task decomposition and selects one skill per agent step.
The Vision Agent only grounds a requested target or checks a requested visual
postcondition. Navigation handoff combines fresh target grounding with bbox
depth; alignment handoff requires every operation object in one fresh chest or
head view. MOVE_TO and ALIGN may repeat, while MANIPULATE is terminal.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Literal, Mapping, Sequence

import cv2
import numpy as np

from gear_sonic.utils.inference.lavira.geometry import (
    FRAME_COUNT,
    MAX_DIRECT_TRAVEL,
    POLICY_FRAME_INDEX,
    build_object_nav_geometry_from_frames,
)
from gear_sonic.utils.inference.lavira.object_nav import RGBDSnapshot
from gear_sonic.runtime.telemetry import default_inference_log_dir

LOGGER = logging.getLogger("sonic.lavira")
DEFAULT_LA_MODEL = "qwen3.8-max"
DEFAULT_VA_MODEL = "qwen3.5-27b"
# Compatibility alias for external imports that previously assumed one model.
DEFAULT_MODEL = DEFAULT_VA_MODEL
NAVIGATION_MODES = {"vln", "object_nav"}
SKILLS = {"MOVE_TO", "ALIGN", "MANIPULATE"}
DIRECTIONS = {"front", "left", "behind", "right"}
DECISIONS = {"EXECUTE", "FAIL"}
POSTCHECK_RESULTS = {"SATISFIED", "NOT_SATISFIED", "UNKNOWN"}
EventReporter = Callable[..., None]
TodoReporter = Callable[[int, int, str], None]

LA_KEYS = {
    "progress_analysis", "updated_todo_list", "reasoning", "decision",
    "skill", "skill_args", "expected_postcondition",
}
GROUNDING_KEYS = {
    "mode", "status", "bbox_2d", "point_2d", "target_description", "confidence",
}
ALIGN_GROUNDING_KEYS = {
    "mode", "status", "target", "surface", "bbox_2d", "visual_evidence",
    "confidence",
}
POSTCHECK_KEYS = {
    "mode", "status", "transition", "visual_evidence", "confidence",
}
TRANSITIONS_BY_SKILL = {
    "MOVE_TO": {"CONTINUE_NAVIGATION", "READY_TO_ALIGN", "UNKNOWN"},
    "ALIGN": {
        "RETRY_ALIGN", "RETURN_TO_NAVIGATION", "READY_TO_MANIPULATE", "UNKNOWN",
    },
    "MANIPULATE": {"CONTINUE_MANIPULATION", "TASK_COMPLETE", "UNKNOWN"},
}
DIRECTION_DELTAS = {
    "front": 0.0,
    "left": math.pi / 2.0,
    "behind": math.pi,
    "right": -math.pi / 2.0,
}


class LaViRAAgentError(RuntimeError):
    """A fail-closed task error suitable for a structured task result."""


class LaViRAAgentCancelled(LaViRAAgentError):
    """Raised after Space invalidates the active task generation."""


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _strict_json_object(text: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise LaViRAAgentError(f"{role} output is empty")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LaViRAAgentError(f"{role} output is not strict JSON") from exc
    if not isinstance(value, dict):
        raise LaViRAAgentError(f"{role} output must be a JSON object")
    return value


def _validate_bbox(value: Any, *, required: bool) -> list[float] | None:
    if value is None and not required:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise LaViRAAgentError("VA bbox_2d is invalid")
    if not all(_finite(item) and 0.0 <= float(item) <= 1000.0 for item in value):
        raise LaViRAAgentError("VA bbox_2d is invalid")
    x1, y1, x2, y2 = map(float, value)
    if x1 >= x2 or y1 >= y2:
        raise LaViRAAgentError("VA bbox_2d has invalid corner ordering")
    return [x1, y1, x2, y2]


def _validate_skill_args(
    skill: str | None,
    args: Mapping[str, Any],
    *,
    current_step: int | None = None,
) -> None:
    if skill is None:
        if args:
            raise LaViRAAgentError("LA terminal decision requires empty skill_args")
        return
    expected: set[str]
    if skill == "MOVE_TO":
        expected = {"view_direction", "target"}
        if args.get("view_direction") not in DIRECTIONS:
            raise LaViRAAgentError("MOVE_TO view_direction is invalid")
        if current_step is not None and current_step > 1 and (
            args.get("view_direction") == "behind"
        ):
            raise LaViRAAgentError(
                "MOVE_TO behind direction is only allowed on the first Agent step"
            )
        if not isinstance(args.get("target"), str) or not str(args["target"]).strip():
            raise LaViRAAgentError("MOVE_TO target is required")
    elif skill == "ALIGN":
        expected = set()
    else:
        expected = set()
    if set(args) != expected:
        raise LaViRAAgentError(f"{skill} skill_args have an invalid schema")


def validate_language_action(
    value: Any, *, current_step: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != LA_KEYS:
        raise LaViRAAgentError("LA output has an invalid object schema")
    for key in {
        "progress_analysis", "updated_todo_list", "reasoning", "decision",
        "expected_postcondition",
    }:
        if not isinstance(value[key], str):
            raise LaViRAAgentError(f"LA {key} must be a string")
    decision = value["decision"].upper()
    if decision not in DECISIONS:
        raise LaViRAAgentError("LA decision is invalid")
    skill = value["skill"]
    if skill is not None and (not isinstance(skill, str) or skill.upper() not in SKILLS):
        raise LaViRAAgentError("LA skill is invalid")
    if not isinstance(value["skill_args"], dict):
        raise LaViRAAgentError("LA skill_args must be an object")
    if decision == "EXECUTE" and skill is None:
        raise LaViRAAgentError("LA EXECUTE requires a skill")
    if decision != "EXECUTE" and skill is not None:
        raise LaViRAAgentError("LA non-EXECUTE decision requires skill=null")
    result = dict(value)
    result["decision"] = decision
    result["skill"] = None if skill is None else skill.upper()
    _validate_skill_args(
        result["skill"], result["skill_args"], current_step=current_step,
    )
    return result


def validate_grounding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != GROUNDING_KEYS:
        raise LaViRAAgentError("VA GROUNDING output has an invalid object schema")
    if value["mode"] != "GROUNDING" or value["status"] not in {"FOUND", "NOT_FOUND"}:
        raise LaViRAAgentError("VA GROUNDING status is invalid")
    if not isinstance(value["target_description"], str):
        raise LaViRAAgentError("VA target_description must be a string")
    if not _finite(value["confidence"]) or not 0.0 <= float(value["confidence"]) <= 1.0:
        raise LaViRAAgentError("VA confidence is invalid")
    found = value["status"] == "FOUND"
    _validate_bbox(value["bbox_2d"], required=found)
    point = value["point_2d"]
    if point is not None and (
        not isinstance(point, list) or len(point) != 2
        or not all(_finite(item) and 0.0 <= float(item) <= 1000.0 for item in point)
    ):
        raise LaViRAAgentError("VA point_2d is invalid")
    if not found and (value["bbox_2d"] is not None or point is not None):
        raise LaViRAAgentError("VA NOT_FOUND cannot include a location")
    return dict(value)


def validate_postcheck(
    value: Any, *, skill: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != POSTCHECK_KEYS:
        raise LaViRAAgentError("VA POSTCHECK output has an invalid object schema")
    if value["mode"] != "POSTCHECK" or value["status"] not in POSTCHECK_RESULTS:
        raise LaViRAAgentError("VA POSTCHECK status is invalid")
    transition = value["transition"]
    valid_transitions = set().union(*TRANSITIONS_BY_SKILL.values())
    if transition not in valid_transitions:
        raise LaViRAAgentError("VA POSTCHECK transition is invalid")
    if skill is not None and transition not in TRANSITIONS_BY_SKILL.get(skill, set()):
        raise LaViRAAgentError(
            f"VA POSTCHECK transition is invalid for {skill}"
        )
    if value["status"] == "UNKNOWN" and transition != "UNKNOWN":
        raise LaViRAAgentError("VA UNKNOWN postcheck requires UNKNOWN transition")
    if transition == "TASK_COMPLETE" and value["status"] != "SATISFIED":
        raise LaViRAAgentError("VA TASK_COMPLETE requires SATISFIED postcheck")
    if skill == "MANIPULATE":
        expected_transition = {
            "SATISFIED": "TASK_COMPLETE",
            "NOT_SATISFIED": "CONTINUE_MANIPULATION",
            "UNKNOWN": "UNKNOWN",
        }[value["status"]]
        if transition != expected_transition:
            raise LaViRAAgentError(
                "VA MANIPULATE postcheck status/transition mismatch"
            )
    if not isinstance(value["visual_evidence"], str):
        raise LaViRAAgentError("VA visual_evidence must be a string")
    if not _finite(value["confidence"]) or not 0.0 <= float(value["confidence"]) <= 1.0:
        raise LaViRAAgentError("VA confidence is invalid")
    return dict(value)


def validate_alignment_grounding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != ALIGN_GROUNDING_KEYS:
        raise LaViRAAgentError(
            "VA ALIGN_GROUNDING output has an invalid object schema"
        )
    if (
        value["mode"] != "ALIGN_GROUNDING"
        or value["status"] not in {"FOUND", "NOT_FOUND"}
    ):
        raise LaViRAAgentError("VA ALIGN_GROUNDING status is invalid")
    for key in ("target", "surface", "visual_evidence"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise LaViRAAgentError(f"VA ALIGN_GROUNDING {key} is required")
    if not _finite(value["confidence"]) or not 0.0 <= float(value["confidence"]) <= 1.0:
        raise LaViRAAgentError("VA confidence is invalid")
    _validate_bbox(value["bbox_2d"], required=value["status"] == "FOUND")
    if value["status"] == "NOT_FOUND" and value["bbox_2d"] is not None:
        raise LaViRAAgentError("VA ALIGN_GROUNDING NOT_FOUND cannot include a bbox")
    return dict(value)


def mean_bbox_depth_m(
    depth_mm: np.ndarray,
    bbox_2d: Sequence[float],
    *,
    max_depth_m: float = MAX_DIRECT_TRAVEL,
) -> float | None:
    """Return the mean valid metric depth inside a normalized VA bbox."""

    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or depth.size == 0:
        raise LaViRAAgentError("NAV handoff depth must be a non-empty 2D array")
    bbox = _validate_bbox(list(bbox_2d), required=True)
    assert bbox is not None
    height, width = depth.shape
    x1 = max(0, min(width - 1, int(math.floor(bbox[0] * width / 1000.0))))
    y1 = max(0, min(height - 1, int(math.floor(bbox[1] * height / 1000.0))))
    x2 = max(x1 + 1, min(width, int(math.ceil(bbox[2] * width / 1000.0))))
    y2 = max(y1 + 1, min(height, int(math.ceil(bbox[3] * height / 1000.0))))
    roi = depth[y1:y2, x1:x2].astype(np.float64, copy=False)
    maximum_mm = float(max_depth_m) * 1000.0
    valid = roi[
        np.isfinite(roi) & (roi > 0.0) & (roi <= maximum_mm)
    ]
    if valid.size == 0:
        return None
    return float(np.mean(valid) / 1000.0)


# Migration-friendly import name; semantics are now strictly GROUNDING.
validate_vision_action = validate_grounding


def _image_data_url(image_bgr: np.ndarray) -> str:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise LaViRAAgentError("LaViRA RGB must be HxWx3 uint8")
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise LaViRAAgentError("failed to encode LaViRA chest RGB")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def todo_prompt(mission: str) -> str:
    return f"""
Your task is to create a dynamic checklist (TODO list) to complete the
instruction based on the visual context.

Instruction: {json.dumps(mission, ensure_ascii=False)}

Requirements:
- Break down the instruction into logical, sequential steps.
- Use the visual information to identify landmarks or initial direction if possible.
- Include navigation, approach, alignment, manipulation, and final verification.
- Format as a Markdown checklist:
  - [ ] Step 1 description
  - [ ] Step 2 description

Response format:
Return ONLY the markdown checklist string. Do not use JSON.
"""


def _g1_view_label(direction: str, current_step: int, image_index: int) -> str:
    labels = {
        "front": "The current FORWARD view",
        "right": "The view after turning 90 deg to the RIGHT",
        "behind": "The view directly BEHIND (180 deg turn)",
        "left": "The view after turning 90 deg to the LEFT",
    }
    return f"Image {image_index}: {labels[direction]} (Step {current_step})."


def language_action_prompt(
    mission: str,
    navigation_mode: str,
    global_target: str,
    current_step: int,
    todo_list: str,
    observation_mode: str,
    transition_result: Mapping[str, Any] | None,
) -> str:
    navigation_guidance = (
        "Follow the route language and use visible route landmarks/openings."
        if navigation_mode == "vln"
        else "Explore for the named object and use useful landmarks/openings."
    )
    visual_context = (
        "The first four labeled images are the fresh Current Panorama Views "
        "captured at this step: front, right, behind, and left."
        if observation_mode == "panorama"
        else "The first labeled image is the fresh fixed-front observation. "
        "The robot did not rotate for this LA request."
    )
    transition_text = (
        "No previous skill transition is available for the first step."
        if transition_result is None
        else json.dumps(dict(transition_result), ensure_ascii=False)
    )
    allowed_move_directions = (
        "front|left|right|behind" if current_step == 1 else "front|left|right"
    )
    return f"""
**ROLE**: You are an intelligent humanoid robot agent using a generic checklist
to guide your actions.
**MISSION**: {json.dumps(mission, ensure_ascii=False)}
**GLOBAL TARGET**: {json.dumps(global_target, ensure_ascii=False)}
**NAVIGATION MODE**: {navigation_mode}. {navigation_guidance}

**Current TODO List**:
{todo_list}

**Current Step**: {current_step}

**VISUAL CONTEXT**:
- {visual_context}
- Any following images are the five most recent successful MOVE_TO result
  observations, ordered from oldest to newest.

**LATEST VA TRANSITION DISCRIMINATOR**:
{transition_text}

**Task**:
1. **Update the TODO list**:
   - Check if the current step is completed based on the visual views.
   - If completed, mark it as [x] and append "Result: ...".
   - If the plan is stuck, add new items or modify steps.
2. **Decide the next action**:
   - Base it on the first incomplete TODO item.
   - Choose strictly from MOVE_TO, ALIGN, MANIPULATE, or FAIL.
   - Respect the latest VA transition discriminator. It describes which stage
     is visually ready, but only your skill call can request a mode change.
   - MOVE_TO skill_args must be exactly
     {{"view_direction":"{allowed_move_directions}","target":"..."}}.
     The direction refers only to the Current Panorama Views in this request.
   - ALIGN skill_args must be {{}}. Alignment always uses the front view; VA
     derives the BasePose target and surface from the original mission.
   - MANIPULATE skill_args must be {{}} and is terminal.
   - The runtime applies fixed NAV and ALIGN handoff contracts. Your
     expected_postcondition describes intended progress but cannot override
     target-visible/depth readiness or single-camera common visibility.
3. **Completion and failure**:
   - Never return COMPLETE. After MANIPULATE, the system completes the task
     when VA confirms the final visual postcondition.
   - Use FAIL only for an unrecoverable task failure.

**JSON RESPONSE FORMAT**:
{{"progress_analysis":"...","updated_todo_list":"- [x] ...",
"reasoning":"...","decision":"EXECUTE|FAIL",
"skill":"MOVE_TO|ALIGN|MANIPULATE|null","skill_args":{{}},
"expected_postcondition":"..."}}"""


def _va_context(
    mission: str,
    global_target: str,
    strategic_goal: str,
    strategic_stop: bool,
) -> str:
    return f"""**MISSION**: {json.dumps(mission, ensure_ascii=False)}
**GLOBAL TARGET**: {json.dumps(global_target, ensure_ascii=False)}
**CURRENT STRATEGY**: {json.dumps(strategic_goal, ensure_ascii=False)}
**STRATEGIC STOP SIGNAL**: {json.dumps(strategic_stop)}"""


def grounding_prompt(
    *, mission: str, global_target: str, strategic_goal: str,
    strategic_stop: bool, target: str, direction: str,
) -> str:
    return f"""**ROLE**: You are a humanoid robot agent's TACTICAL EYES in
GROUNDING mode.
{_va_context(mission, global_target, strategic_goal, strategic_stop)}

**INPUT**: You are looking at the CURRENT VIEW after turning to the requested
{direction} panorama direction.

**TASK**:
1. **Verification**: Do you see this requested target instance:
   {json.dumps(target, ensure_ascii=False)}?
2. **Targeting**: Draw a bbox around that target. Do not choose another target,
   skill, or direction, and do not alter the strategic stop signal.
3. **Grounding Decision**: Return FOUND only when the requested target has a
   confident location in this image; otherwise return NOT_FOUND.

Coordinates are normalized [0,1000]. Return exactly:
{{"mode":"GROUNDING","status":"FOUND|NOT_FOUND","bbox_2d":null,
"point_2d":null,"target_description":"...","confidence":0.0}}"""


def alignment_grounding_prompt(
    *, mission: str, global_target: str, strategic_goal: str,
    strategic_stop: bool, direction: str,
) -> str:
    return f"""**ROLE**: You are a humanoid robot agent's TACTICAL EYES in
ALIGN_GROUNDING mode.
{_va_context(mission, global_target, strategic_goal, strategic_stop)}

**INPUT**: You are looking at the CURRENT VIEW after turning to the fixed
{direction} panorama direction.

**TASK**:
1. **Instruction Decomposition**: Derive from MISSION the concrete target that
   BasePose must align the robot with and its supporting/context surface.
2. **Verification**: Check whether that target and surface agree with the
   current image.
3. **Targeting**: Draw a bbox around the derived alignment target. Do not choose
   a skill or direction.
4. **Grounding Decision**: Return FOUND only when the target has a confident
   location; otherwise return NOT_FOUND while still returning target/surface.

Coordinates are normalized [0,1000]. Return exactly:
{{"mode":"ALIGN_GROUNDING","status":"FOUND|NOT_FOUND","target":"...",
"surface":"...","bbox_2d":null,"visual_evidence":"...","confidence":0.0}}"""


def postcheck_prompt(
    *, mission: str, global_target: str, strategic_goal: str,
    strategic_stop: bool, expected: str,
) -> str:
    return f"""**ROLE**: You are a humanoid robot agent's TACTICAL EYES in
POSTCHECK mode.
{_va_context(mission, global_target, strategic_goal, strategic_stop)}

**INPUT**: You are looking at the fresh CURRENT VIEW after the latest controller
action.

**TASK**:
1. **Verification**: Check the image against this requested visual postcondition:
   {json.dumps(expected, ensure_ascii=False)}.
2. **Transition Discriminator**: Select the visually appropriate transition:
   CONTINUE_NAVIGATION, READY_TO_ALIGN, RETRY_ALIGN, RETURN_TO_NAVIGATION,
   READY_TO_MANIPULATE, CONTINUE_MANIPULATION, TASK_COMPLETE, or UNKNOWN.
   Base the decision only on visible evidence and the requested postcondition,
   without inferring or naming the controller action. UNKNOWN status must use
   UNKNOWN transition; TASK_COMPLETE requires SATISFIED.

Return exactly:
{{"mode":"POSTCHECK","status":"SATISFIED|NOT_SATISFIED|UNKNOWN",
"transition":"...","visual_evidence":"...","confidence":0.0}}"""


def alignment_handoff_postcondition(camera_label: str) -> str:
    """Fixed verifier contract for one camera's ALIGN handoff view."""

    return (
        f"This is the fresh {camera_label} camera image. Derive all concrete "
        "operation objects required by the complete original mission, including "
        "the manipulated object and its destination, container, support, or "
        "control object. SATISFIED requires every operation object to be "
        "simultaneously visible in this single image. Do not use or infer "
        "visibility from another camera, and do not count navigation landmarks "
        "or rooms that are not operated on."
    )


class LaViRAClient:
    """Two-endpoint OpenAI-compatible client: LA plans, VA localizes/checks."""

    def __init__(
        self, *, la_base_url: str, va_base_url: str,
        la_model: str = DEFAULT_LA_MODEL, va_model: str = DEFAULT_VA_MODEL,
        la_enable_thinking: bool = True, va_enable_thinking: bool = False,
        la_timeout_seconds: float = 180.0, va_timeout_seconds: float = 180.0,
        la_api_key: str | None = None, va_api_key: str | None = None,
        la_client: Any | None = None, va_client: Any | None = None,
        request_context_dir: str | Path | None = None,
    ) -> None:
        if la_client is None or va_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("LaViRA requires the openai inference dependency") from exc
            la_client = la_client or OpenAI(
                api_key=la_api_key or os.getenv("LAVIRA_LA_API_KEY") or "no-key",
                base_url=la_base_url, timeout=float(la_timeout_seconds),
            )
            va_client = va_client or OpenAI(
                api_key=va_api_key or os.getenv("LAVIRA_VA_API_KEY") or "no-key",
                base_url=va_base_url, timeout=float(va_timeout_seconds),
            )
        self.la_client = la_client
        self.va_client = va_client
        self.la_model = str(la_model)
        self.va_model = str(va_model)
        self.la_enable_thinking = bool(la_enable_thinking)
        self.va_enable_thinking = bool(va_enable_thinking)
        self.la_timeout_seconds = float(la_timeout_seconds)
        self.va_timeout_seconds = float(va_timeout_seconds)
        self.request_context_dir = (
            Path(request_context_dir)
            if request_context_dir is not None
            else default_inference_log_dir() / "lavira_requests"
        )
        self._request_context_lock = threading.Lock()
        self._request_context_sequence = 0

    def _save_request_context(
        self, request_kind: str, attempt: int, request: Mapping[str, Any],
    ) -> None:
        """Persist the exact OpenAI-compatible request before transmission."""

        try:
            with self._request_context_lock:
                self._request_context_sequence += 1
                sequence = self._request_context_sequence
                self.request_context_dir.mkdir(parents=True, exist_ok=True)
                timestamp_ns = time.time_ns()
                filename = (
                    f"{timestamp_ns}_{os.getpid()}_{sequence:06d}_"
                    f"{request_kind}_attempt{attempt}.json"
                )
                payload = {
                    "type": "sonic.model_request_context",
                    "version": 1,
                    "timestamp_ns": timestamp_ns,
                    "component": "lavira",
                    "request_kind": request_kind,
                    "attempt": attempt,
                    "request": dict(request),
                }
                (self.request_context_dir / filename).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            LOGGER.exception(
                "Failed to persist LaViRA request context kind=%s",
                request_kind,
            )

    def _create(
        self, client: Any, *, request_kind: str, **kwargs: Any,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            self._save_request_context(request_kind, attempt + 1, kwargs)
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    LOGGER.warning("LaViRA model call failed; retrying: %s", exc)
                    time.sleep(10.0)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _messages(
        content: Sequence[Mapping[str, Any]], *, enable_thinking: bool
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": (
                    "Reason carefully and follow the requested output format exactly."
                    if enable_thinking
                    else "/no_think"
                ),
            },
            {"role": "user", "content": list(content)},
        ]

    @staticmethod
    def _content(completion: Any, *, role: str) -> str:
        try:
            value = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LaViRAAgentError(f"{role} response is malformed") from exc
        if not isinstance(value, str):
            raise LaViRAAgentError(f"{role} response content is malformed")
        return value

    def initial_todo(
        self, *, mission: str, scan_views: Sequence["ScanView"],
    ) -> str:
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"Instruction: {json.dumps(mission, ensure_ascii=False)}\n\n"
                "The images provided are the 4-directional views from the "
                "starting position."
            ),
        }]
        for index, view in enumerate(scan_views, start=1):
            content.extend([
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(view.image_bgr)},
                },
                {
                    "type": "text",
                    "text": _g1_view_label(view.direction, 1, index),
                },
            ])
        content.append({"type": "text", "text": todo_prompt(mission)})
        completion = self._create(
            self.la_client, request_kind="la_todo", model=self.la_model,
            messages=self._messages(
                content, enable_thinking=self.la_enable_thinking,
            ),
            max_tokens=512, temperature=0.1, timeout=self.la_timeout_seconds,
            extra_body={"enable_thinking": self.la_enable_thinking},
        )
        result = self._content(completion, role="LA TODO").strip()
        if "- [ ]" not in result:
            raise LaViRAAgentError("LA TODO output is not a Markdown checklist")
        return result

    def language_action(
        self, *, mission: str, navigation_mode: str, global_target: str,
        current_step: int, todo_list: str,
        scan_views: Sequence["ScanView"],
        move_to_views: Sequence["MoveToView"],
        transition_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"Navigation Task: {json.dumps(mission, ensure_ascii=False)}\n\n"
                f"- Current Step: {current_step}"
            ),
        }]
        recent_move_views = move_to_views[-5:]
        for index, view in enumerate(recent_move_views):
            content.extend([
                {"type": "image_url", "image_url": {
                    "url": _image_data_url(view.image_bgr),
                }},
                {
                    "type": "text",
                    "text": f"PLAN-{len(recent_move_views) - index}",
                },
            ])
        for index, view in enumerate(scan_views[-4:], start=1):
            content.extend([
                {"type": "image_url", "image_url": {
                    "url": _image_data_url(view.image_bgr),
                }},
                {
                    "type": "text",
                    "text": _g1_view_label(
                        view.direction, current_step, index,
                    ),
                },
            ])
        content.append({
            "type": "text",
            "text": language_action_prompt(
                mission, navigation_mode, global_target, current_step, todo_list,
                "panorama" if len(scan_views) == 4 else "front",
                transition_result,
            ),
        })
        completion = self._create(
            self.la_client, request_kind="la_decision", model=self.la_model,
            messages=self._messages(
                content, enable_thinking=self.la_enable_thinking,
            ),
            max_tokens=1200, temperature=0, timeout=self.la_timeout_seconds,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": self.la_enable_thinking},
        )
        return validate_language_action(_strict_json_object(
            self._content(completion, role="LA"), role="LA",
        ), current_step=current_step)

    def grounding(
        self, *, mission: str, global_target: str, strategic_goal: str,
        strategic_stop: bool, target: str, direction: str,
        image_bgr: np.ndarray,
    ) -> dict[str, Any]:
        completion = self._create(
            self.va_client, request_kind="va_grounding", model=self.va_model,
            messages=self._messages([
                {"type": "image_url", "image_url": {"url": _image_data_url(image_bgr)}},
                {"type": "text", "text": grounding_prompt(
                    mission=mission, global_target=global_target,
                    strategic_goal=strategic_goal,
                    strategic_stop=strategic_stop, target=target,
                    direction=direction,
                )},
            ], enable_thinking=self.va_enable_thinking),
            max_tokens=768, temperature=0, timeout=self.va_timeout_seconds,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": self.va_enable_thinking},
        )
        return validate_grounding(_strict_json_object(
            self._content(completion, role="VA GROUNDING"), role="VA GROUNDING",
        ))

    def alignment_grounding(
        self, *, mission: str, global_target: str, strategic_goal: str,
        strategic_stop: bool, direction: str, image_bgr: np.ndarray,
    ) -> dict[str, Any]:
        completion = self._create(
            self.va_client, request_kind="va_align_grounding", model=self.va_model,
            messages=self._messages([
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image_bgr)},
                },
                {"type": "text", "text": alignment_grounding_prompt(
                    mission=mission, global_target=global_target,
                    strategic_goal=strategic_goal,
                    strategic_stop=strategic_stop, direction=direction,
                )},
            ], enable_thinking=self.va_enable_thinking),
            max_tokens=768, temperature=0, timeout=self.va_timeout_seconds,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": self.va_enable_thinking},
        )
        return validate_alignment_grounding(_strict_json_object(
            self._content(completion, role="VA ALIGN_GROUNDING"),
            role="VA ALIGN_GROUNDING",
        ))

    def postcheck(
        self, *, mission: str, global_target: str, strategic_goal: str,
        strategic_stop: bool, expected_postcondition: str,
        image_bgr: np.ndarray,
    ) -> dict[str, Any]:
        completion = self._create(
            self.va_client, request_kind="va_postcheck", model=self.va_model,
            messages=self._messages([
                {"type": "image_url", "image_url": {"url": _image_data_url(image_bgr)}},
                {"type": "text", "text": postcheck_prompt(
                    mission=mission, global_target=global_target,
                    strategic_goal=strategic_goal,
                    strategic_stop=strategic_stop,
                    expected=expected_postcondition,
                )},
            ], enable_thinking=self.va_enable_thinking),
            max_tokens=768, temperature=0, timeout=self.va_timeout_seconds,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": self.va_enable_thinking},
        )
        return validate_postcheck(_strict_json_object(
            self._content(completion, role="VA POSTCHECK"), role="VA POSTCHECK",
        ))


@dataclass(frozen=True)
class ScanView:
    scan_id: int
    direction: str
    image_bgr: np.ndarray = field(repr=False, compare=False)
    reference_pose: tuple[float, float, float]
    absolute_yaw_rad: float


@dataclass(frozen=True)
class MoveToView:
    skill_id: int
    target: str
    image_bgr: np.ndarray = field(repr=False, compare=False)
    controller_state: str
    va_result: str
    evidence: str = ""


@dataclass(frozen=True)
class AgentHistoryEntry:
    skill_id: int
    skill: str
    target: str
    controller_state: str
    va_result: str
    evidence: str = ""

    def summary(self) -> str:
        return (
            f"skill_id={self.skill_id} skill={self.skill} target={self.target!r} "
            f"controller={self.controller_state} va={self.va_result} "
            f"evidence={self.evidence!r}"
        )


@dataclass(frozen=True)
class LaViRATaskResult:
    generation: int
    state: str
    reason: str
    steps: int
    skill_id: int
    segment_id: int
    navigation_mode: str


class LaViRAAgent:
    """Execute a unified manipulation mission without publishing direct velocity."""

    def __init__(
        self, *, navigation_mode: Literal["vln", "object_nav"], mission: str,
        global_target: str, max_steps: int, history_size: int,
        min_confidence: float, segment_timeout_seconds: float, camera: Any,
        client: Any,
        submit_intent: Callable[[str, Mapping[str, object]], None],
        wait_status: Callable[..., Mapping[str, Any]],
        cancelled: Callable[[int], bool],
        readiness: Callable[[int, int], tuple[bool, str]] | None = None,
        poll_failure: (
            Callable[[int, int], Mapping[str, Any] | None] | None
        ) = None,
        nav_handoff_min_depth_m: float = 0.3,
        nav_handoff_max_depth_m: float = 3.0,
        alignment_head_camera_stream: str = "ego_view",
        manipulation_window_seconds: float = 5.0,
        manipulation_max_windows: int = 12,
        manipulation_timeout_seconds: float = 180.0,
        vla_start_timeout_seconds: float = 3.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        report_event: EventReporter | None = None,
        report_todo: TodoReporter | None = None,
    ) -> None:
        if navigation_mode not in NAVIGATION_MODES:
            raise ValueError("navigation_mode must be vln or object_nav")
        if not mission.strip() or not global_target.strip():
            raise ValueError("mission and global_target are required")
        if max_steps <= 0 or history_size <= 0:
            raise ValueError("max_steps and history_size must be positive")
        if (
            manipulation_window_seconds < 0
            or manipulation_max_windows <= 0
            or manipulation_timeout_seconds <= 0
            or vla_start_timeout_seconds <= 0
        ):
            raise ValueError("manipulation limits are invalid")
        if (
            not math.isfinite(float(nav_handoff_min_depth_m))
            or not math.isfinite(float(nav_handoff_max_depth_m))
            or float(nav_handoff_min_depth_m) <= 0.0
            or float(nav_handoff_max_depth_m) <= float(nav_handoff_min_depth_m)
            or float(nav_handoff_max_depth_m) > MAX_DIRECT_TRAVEL
        ):
            raise ValueError("NAV handoff depth range is invalid")
        if not str(alignment_head_camera_stream).strip():
            raise ValueError("ALIGN handoff head camera stream is required")
        self.navigation_mode = navigation_mode
        self.mission = mission
        self.global_target = global_target
        self.max_steps = int(max_steps)
        self.history_size = min(5, int(history_size))
        self.min_confidence = float(min_confidence)
        self.segment_timeout_seconds = float(segment_timeout_seconds)
        self.camera = camera
        self.client = client
        self.submit_intent = submit_intent
        self.wait_status = wait_status
        self.cancelled = cancelled
        self.readiness = readiness or (lambda _generation, _skill_id: (True, "ready"))
        self.poll_failure = poll_failure or (lambda _generation, _skill_id: None)
        self.nav_handoff_min_depth_m = float(nav_handoff_min_depth_m)
        self.nav_handoff_max_depth_m = float(nav_handoff_max_depth_m)
        self.alignment_head_camera_stream = str(
            alignment_head_camera_stream
        ).strip()
        self.manipulation_window_seconds = float(manipulation_window_seconds)
        self.manipulation_max_windows = int(manipulation_max_windows)
        self.manipulation_timeout_seconds = float(manipulation_timeout_seconds)
        self.vla_start_timeout_seconds = float(vla_start_timeout_seconds)
        self.monotonic = monotonic
        self.sleep = sleep
        self.report_event = report_event
        self.report_todo = report_todo
        self._segment_id = -1
        self._skill_id = 0
        self._scan_id = 0
        self._latest_scan: list[ScanView] = []
        self._scan_anchor_yaw: float | None = None
        self._recent_move_to_views: list[MoveToView] = []
        self._history: list[AgentHistoryEntry] = []
        self._failed_targets: list[str] = []
        self._latest_transition: dict[str, Any] | None = None
        self._next_observation_mode = "panorama"
        self._manipulation_started = False

    def _reset_task_context(self) -> None:
        """Start one generation with no images or state from an older task."""

        self._segment_id = -1
        self._skill_id = 0
        self._scan_id = 0
        self._latest_scan = []
        self._scan_anchor_yaw = None
        self._recent_move_to_views = []
        self._history = []
        self._failed_targets = []
        self._latest_transition = None
        self._next_observation_mode = "panorama"
        self._manipulation_started = False

    def _event(
        self,
        level: int,
        code: str,
        message: str,
        **fields: object,
    ) -> None:
        """Report one best-effort structured event without affecting control."""

        if self.report_event is None:
            return
        try:
            self.report_event(level, code, message, **fields)
        except Exception:
            LOGGER.exception(
                "LaViRA runtime event reporter failed code=%s",
                code,
            )

    def _todo(self, generation: int, step: int, todo_list: str) -> None:
        if self.report_todo is None:
            return
        try:
            self.report_todo(generation, step, todo_list)
        except Exception:
            LOGGER.exception("LaViRA TODO pane reporter failed")

    def _check_cancelled(self, generation: int) -> None:
        if self.cancelled(generation):
            raise LaViRAAgentCancelled("operator_cancelled")

    def _next_segment(self) -> int:
        self._segment_id += 1
        return self._segment_id

    def _wait(self, generation: int, skill_id: int, segment_id: int) -> Mapping[str, Any]:
        try:
            status = self.wait_status(
                generation, skill_id, segment_id, self.segment_timeout_seconds,
            )
        except TypeError:
            status = self.wait_status(
                generation, segment_id, self.segment_timeout_seconds,
            )
        self._check_cancelled(generation)
        if int(status.get("generation", generation)) != generation:
            raise LaViRAAgentError("stale controller status")
        if int(status.get("skill_id", skill_id)) != skill_id:
            raise LaViRAAgentError("stale skill status")
        return status

    def _pose(self) -> tuple[float, float, float]:
        if hasattr(self.camera, "current_pose"):
            try:
                raw = self.camera.current_pose()
            except Exception:
                raw = None
            if raw is None:
                return 0.0, 0.0, 0.0
            if isinstance(raw, Mapping):
                values = (raw.get("x"), raw.get("y"), raw.get("yaw"))
            else:
                values = tuple(raw)
            if len(values) == 3 and all(_finite(item) for item in values):
                return tuple(round(float(item), 2) for item in values)  # type: ignore[return-value]
        return 0.0, 0.0, 0.0

    def _heading(
        self, generation: int, skill_id: int, delta: float,
    ) -> Mapping[str, Any]:
        segment = self._next_segment()
        self.submit_intent("navigation_heading_goal", {
            "generation": generation, "skill_id": skill_id,
            "segment_id": segment, "heading_delta_rad": float(delta),
        })
        status = self._wait(generation, skill_id, segment)
        if status.get("state") != "reached":
            raise LaViRAAgentError(
                f"heading_{status.get('state', 'failed')}:"
                f"{status.get('reason', 'unknown')}"
            )
        return status

    def _capture_va_burst(
        self, generation: int, skill_id: int, segment_id: int,
    ) -> list[RGBDSnapshot]:
        identity = {
            "generation": generation, "skill_id": skill_id,
            "segment_id": segment_id,
        }
        self.submit_intent("lavira_depth_request", identity)
        if hasattr(self.camera, "begin_depth_lease"):
            try:
                self.camera.begin_depth_lease(generation, skill_id, segment_id)
            except TypeError:
                self.camera.begin_depth_lease(generation, segment_id)
        try:
            return [self.camera.capture_aligned_rgbd() for _ in range(FRAME_COUNT)]
        finally:
            self.submit_intent("lavira_rgbd_captured", identity)

    def _capture_va_snapshot(
        self, generation: int, skill_id: int, segment_id: int,
    ) -> RGBDSnapshot:
        """Capture one fresh leased RGB-D observation for NAV handoff."""

        identity = {
            "generation": generation, "skill_id": skill_id,
            "segment_id": segment_id,
        }
        self.submit_intent("lavira_depth_request", identity)
        if hasattr(self.camera, "begin_depth_lease"):
            try:
                self.camera.begin_depth_lease(generation, skill_id, segment_id)
            except TypeError:
                self.camera.begin_depth_lease(generation, segment_id)
        try:
            return self.camera.capture_aligned_rgbd()
        finally:
            self.submit_intent("lavira_rgbd_captured", identity)

    def _record(self, entry: AgentHistoryEntry) -> None:
        self._history.append(entry)
        self._history = self._history[-self.history_size :]

    def _remember_move_to(
        self, entry: AgentHistoryEntry, image_bgr: np.ndarray,
    ) -> None:
        """Retain an observation captured after a successful MOVE_TO ends."""

        self._recent_move_to_views.append(MoveToView(
            skill_id=entry.skill_id,
            target=entry.target,
            image_bgr=np.asarray(image_bgr).copy(),
            controller_state=entry.controller_state,
            va_result=entry.va_result,
            evidence=entry.evidence,
        ))
        self._recent_move_to_views = self._recent_move_to_views[-5:]

    def _face_absolute_yaw(
        self, generation: int, skill_id: int, target_yaw: float,
    ) -> Mapping[str, Any]:
        """Turn to an absolute Fast-LIO yaw using the relative wire command."""

        current_yaw = self._pose()[2]
        delta = math.remainder(float(target_yaw) - current_yaw, 2 * math.pi)
        return self._heading(generation, skill_id, delta)

    def _capture_panorama(
        self, generation: int, skill_id: int,
    ) -> list[ScanView]:
        """Capture the fresh four-direction observation for one LA step."""

        anchor_pose = self._pose()
        anchor_yaw = anchor_pose[2]
        self._scan_anchor_yaw = anchor_yaw
        self._scan_id += 1
        scan_id = self._scan_id
        front_pose = self._pose()
        views = [ScanView(
            scan_id, "front", self.camera.capture_rgb(), anchor_pose,
            front_pose[2],
        )]
        for direction in ("right", "behind", "left"):
            target_yaw = math.remainder(
                anchor_yaw + DIRECTION_DELTAS[direction], 2 * math.pi,
            )
            self._face_absolute_yaw(generation, skill_id, target_yaw)
            measured_pose = self._pose()
            views.append(ScanView(
                scan_id, direction, self.camera.capture_rgb(), anchor_pose,
                measured_pose[2],
            ))
        self._face_absolute_yaw(generation, skill_id, anchor_yaw)
        self._latest_scan = views
        return views

    def _capture_front_observation(self) -> list[ScanView]:
        """Capture one fixed-front view without commanding any rotation."""

        pose = self._pose()
        self._scan_anchor_yaw = pose[2]
        self._scan_id += 1
        views = [ScanView(
            self._scan_id, "front", self.camera.capture_rgb(), pose, pose[2],
        )]
        self._latest_scan = views
        return views

    def _face_scan_direction(
        self, generation: int, skill_id: int, direction: str,
    ) -> Mapping[str, Any]:
        """Face a direction from the panorama used by the current LA call."""

        if self._scan_anchor_yaw is None or direction not in DIRECTIONS:
            raise LaViRAAgentError("current panorama direction is unavailable")
        target_yaw = math.remainder(
            self._scan_anchor_yaw + DIRECTION_DELTAS[direction], 2 * math.pi,
        )
        return self._face_absolute_yaw(generation, skill_id, target_yaw)

    def _validate_transition_gate(self, skill: str) -> None:
        """Gate forward handoffs while allowing same-stage recovery calls."""

        if self._latest_transition is None:
            if skill != "MOVE_TO":
                raise LaViRAAgentError(
                    f"handoff_gate:no_nav_readiness_for_{skill}"
                )
            return
        transition = str(self._latest_transition["transition"])
        executed_skill = str(self._latest_transition["executed_skill"])
        allowed = {
            "CONTINUE_NAVIGATION": {"MOVE_TO"},
            "READY_TO_ALIGN": {"MOVE_TO", "ALIGN"},
            "RETRY_ALIGN": {"MOVE_TO", "ALIGN"},
            "RETURN_TO_NAVIGATION": {"MOVE_TO"},
            "READY_TO_MANIPULATE": {"MOVE_TO", "ALIGN", "MANIPULATE"},
            "UNKNOWN": (
                {"MOVE_TO", "ALIGN"}
                if executed_skill == "ALIGN" else {"MOVE_TO"}
            ),
        }.get(transition, set())
        if skill not in allowed:
            raise LaViRAAgentError(
                f"handoff_gate:{transition}_does_not_allow_{skill}"
            )

    def _ground(
        self, *, generation: int, skill_id: int, target: str,
        direction: str, image: np.ndarray, strategic_goal: str,
        strategic_stop: bool,
    ) -> dict[str, Any]:
        result = validate_grounding(self.client.grounding(
            mission=self.mission, global_target=self.global_target,
            strategic_goal=strategic_goal,
            strategic_stop=strategic_stop,
            target=target, direction=direction, image_bgr=image,
        ))
        if (
            result["status"] == "FOUND"
            and float(result["confidence"]) < self.min_confidence
        ):
            result = dict(result)
            result.update(status="NOT_FOUND", bbox_2d=None, point_2d=None)
        self._event(
            logging.INFO if result["status"] == "FOUND" else logging.WARNING,
            "VA_GROUNDING",
            f"VA grounding {str(result['status']).lower()}",
            generation=generation,
            skill_id=skill_id,
            segment_id=max(0, self._segment_id),
            target=target,
            view_direction=direction,
            status=result["status"],
            confidence=result["confidence"],
            target_description=result["target_description"],
            bbox_2d=result["bbox_2d"],
            point_2d=result["point_2d"],
        )
        return result

    def _alignment_ground(
        self, *, generation: int, skill_id: int, image: np.ndarray,
        strategic_goal: str,
    ) -> dict[str, Any]:
        result = validate_alignment_grounding(self.client.alignment_grounding(
            mission=self.mission,
            global_target=self.global_target,
            strategic_goal=strategic_goal,
            strategic_stop=False,
            direction="front",
            image_bgr=image,
        ))
        if (
            result["status"] == "FOUND"
            and float(result["confidence"]) < self.min_confidence
        ):
            result = dict(result)
            result.update(status="NOT_FOUND", bbox_2d=None)
        self._event(
            logging.INFO if result["status"] == "FOUND" else logging.WARNING,
            "VA_ALIGN_GROUNDING",
            f"VA alignment grounding {str(result['status']).lower()}",
            generation=generation,
            skill_id=skill_id,
            segment_id=max(0, self._segment_id),
            target=result["target"],
            surface=result["surface"],
            view_direction="front",
            status=result["status"],
            confidence=result["confidence"],
            bbox_2d=result["bbox_2d"],
            visual_evidence=result["visual_evidence"],
        )
        return result

    def _postcheck(
        self,
        skill: str,
        expected: str,
        *,
        generation: int,
        skill_id: int,
        strategic_goal: str,
        strategic_stop: bool,
        window_id: int | None = None,
        image_bgr: np.ndarray | None = None,
    ) -> dict[str, Any]:
        result = validate_postcheck(self.client.postcheck(
            mission=self.mission, global_target=self.global_target,
            strategic_goal=strategic_goal,
            strategic_stop=strategic_stop,
            expected_postcondition=expected,
            image_bgr=(
                self.camera.capture_rgb() if image_bgr is None else image_bgr
            ),
        ), skill=skill)
        self._store_transition(skill, result)
        self._event(
            logging.INFO if result["status"] == "SATISFIED" else logging.WARNING,
            "VA_POSTCHECK",
            f"{skill} visual postcheck {str(result['status']).lower()}",
            generation=generation,
            skill_id=skill_id,
            segment_id=max(0, self._segment_id),
            window_id=window_id,
            skill=skill,
            status=result["status"],
            transition=result["transition"],
            confidence=result["confidence"],
            visual_evidence=result["visual_evidence"],
            expected_postcondition=expected,
        )
        return result

    def _store_transition(
        self, skill: str, result: Mapping[str, Any],
    ) -> None:
        """Store a harness-owned transition in the existing LA context shape."""

        self._latest_transition = {"executed_skill": skill, **dict(result)}
        if skill == "MOVE_TO":
            self._next_observation_mode = (
                "front" if result["transition"] == "READY_TO_ALIGN"
                else "panorama"
            )
        elif skill == "ALIGN":
            self._next_observation_mode = (
                "panorama"
                if result["transition"] == "RETURN_TO_NAVIGATION"
                else "front"
            )
        else:
            self._next_observation_mode = "front"

    def _nav_handoff(
        self, *, generation: int, skill_id: int, target: str,
        grounding: Mapping[str, Any], snapshot: RGBDSnapshot,
    ) -> dict[str, Any]:
        """Apply the fixed visible-target plus bbox-depth NAV contract."""

        target_visible = grounding.get("status") == "FOUND"
        mean_depth: float | None = None
        if target_visible:
            mean_depth = mean_bbox_depth_m(
                snapshot.depth_mm,
                grounding["bbox_2d"],
                max_depth_m=MAX_DIRECT_TRAVEL,
            )
        depth_valid = mean_depth is not None
        depth_in_range = bool(
            depth_valid
            and self.nav_handoff_min_depth_m <= float(mean_depth)
            <= self.nav_handoff_max_depth_m
        )
        if target_visible and depth_in_range:
            status = "SATISFIED"
            transition = "READY_TO_ALIGN"
            evidence = (
                f"{target!r} is visible and its mean bbox depth "
                f"{mean_depth:.3f}m is within the ALIGN handoff range."
            )
        elif target_visible and not depth_valid:
            status = "UNKNOWN"
            transition = "UNKNOWN"
            evidence = f"{target!r} is visible but has no valid bbox depth."
        elif target_visible:
            status = "NOT_SATISFIED"
            transition = "CONTINUE_NAVIGATION"
            evidence = (
                f"{target!r} is visible but its mean bbox depth "
                f"{mean_depth:.3f}m is outside the ALIGN handoff range."
            )
        else:
            status = "NOT_SATISFIED"
            transition = "CONTINUE_NAVIGATION"
            evidence = f"{target!r} is not visible in the fresh NAV result view."
        result = {
            "mode": "POSTCHECK",
            "status": status,
            "transition": transition,
            "visual_evidence": evidence,
            "confidence": float(grounding.get("confidence", 0.0)),
        }
        self._store_transition("MOVE_TO", result)
        self._event(
            logging.INFO if status == "SATISFIED" else logging.WARNING,
            "NAV_HANDOFF_EVALUATED",
            "NAV handoff readiness evaluated",
            generation=generation,
            skill_id=skill_id,
            segment_id=max(0, self._segment_id),
            target=target,
            target_visible=target_visible,
            depth_valid=depth_valid,
            mean_depth_m=mean_depth,
            min_depth_m=self.nav_handoff_min_depth_m,
            max_depth_m=self.nav_handoff_max_depth_m,
            depth_in_range=depth_in_range,
            status=status,
            transition=transition,
            visual_evidence=evidence,
        )
        return result

    def _single_view_alignment_check(
        self, *, camera_label: str, image_bgr: np.ndarray,
        strategic_goal: str,
    ) -> dict[str, Any]:
        """Reuse the existing VA POSTCHECK interface for one camera view."""

        result = validate_postcheck(self.client.postcheck(
            mission=self.mission,
            global_target=self.global_target,
            strategic_goal=strategic_goal,
            strategic_stop=False,
            expected_postcondition=alignment_handoff_postcondition(camera_label),
            image_bgr=image_bgr,
        ))
        if (
            result["status"] == "SATISFIED"
            and float(result["confidence"]) < self.min_confidence
        ):
            result = dict(result)
            result.update(status="UNKNOWN", transition="UNKNOWN")
        return result

    def _align_handoff(
        self, *, generation: int, skill_id: int, strategic_goal: str,
        controller_aligned: bool, controller_state: str,
    ) -> dict[str, Any]:
        """Require all operation objects in one fresh chest or head view."""

        camera_results: dict[str, dict[str, Any]] = {}
        camera_errors: dict[str, str] = {}
        for label, stream_name in (
            ("chest", "chest_view"),
            ("head", self.alignment_head_camera_stream),
        ):
            try:
                if label == "chest":
                    image = self.camera.capture_rgb()
                else:
                    image = self.camera.capture_rgb(camera_stream=stream_name)
            except Exception as exc:
                camera_errors[label] = str(exc)
                self._event(
                    logging.WARNING,
                    "ALIGN_HANDOFF_VIEW_UNAVAILABLE",
                    "ALIGN handoff camera view is unavailable",
                    generation=generation,
                    skill_id=skill_id,
                    segment_id=max(0, self._segment_id),
                    camera=label,
                    camera_stream=stream_name,
                    error=str(exc),
                )
                continue
            try:
                camera_results[label] = self._single_view_alignment_check(
                    camera_label=label,
                    image_bgr=image,
                    strategic_goal=strategic_goal,
                )
            except Exception as exc:
                camera_errors[label] = str(exc)
                self._event(
                    logging.ERROR,
                    "ALIGN_HANDOFF_POSTCHECK_FAILED",
                    "ALIGN handoff VA postcheck failed",
                    generation=generation,
                    skill_id=skill_id,
                    segment_id=max(0, self._segment_id),
                    camera=label,
                    error=str(exc),
                )

        satisfied_views = [
            label for label, result in camera_results.items()
            if result["status"] == "SATISFIED"
        ]
        common_view = (
            "both" if len(satisfied_views) == 2
            else satisfied_views[0] if satisfied_views
            else None
        )
        semantic_ready = bool(satisfied_views)
        has_unknown = bool(camera_errors) or any(
            result["status"] == "UNKNOWN"
            for result in camera_results.values()
        )
        if controller_aligned and semantic_ready:
            status = "SATISFIED"
            transition = "READY_TO_MANIPULATE"
        elif has_unknown and not semantic_ready:
            status = "UNKNOWN"
            transition = "UNKNOWN"
        else:
            status = "NOT_SATISFIED"
            transition = (
                "RETURN_TO_NAVIGATION"
                if controller_state == "target_not_found"
                else "RETRY_ALIGN"
            )
        evidence_parts = [
            f"{label}={result['status']}: {result['visual_evidence']}"
            for label, result in camera_results.items()
        ]
        evidence_parts.extend(
            f"{label}=UNKNOWN: {error}"
            for label, error in camera_errors.items()
        )
        evidence = "; ".join(evidence_parts)
        confidence_results = (
            [camera_results[label] for label in satisfied_views]
            if satisfied_views else list(camera_results.values())
        )
        result = {
            "mode": "POSTCHECK",
            "status": status,
            "transition": transition,
            "visual_evidence": evidence,
            "confidence": max(
                (
                    float(item["confidence"])
                    for item in confidence_results
                ),
                default=0.0,
            ),
        }
        self._store_transition("ALIGN", result)
        self._event(
            logging.INFO if status == "SATISFIED" else logging.WARNING,
            "ALIGN_HANDOFF_EVALUATED",
            "ALIGN handoff readiness evaluated",
            generation=generation,
            skill_id=skill_id,
            segment_id=max(0, self._segment_id),
            controller_aligned=controller_aligned,
            controller_state=controller_state,
            chest_status=(camera_results.get("chest") or {}).get("status"),
            head_status=(camera_results.get("head") or {}).get("status"),
            camera_errors=camera_errors,
            common_view=common_view,
            all_operation_objects_common_view=semantic_ready,
            status=status,
            transition=transition,
            visual_evidence=evidence,
        )
        return result

    def _move_to(
        self, generation: int, skill_id: int, args: Mapping[str, Any],
        _expected: str, strategic_goal: str,
    ) -> AgentHistoryEntry:
        direction = str(args["view_direction"])
        target = str(args["target"])
        self._face_scan_direction(generation, skill_id, direction)
        lease_segment = self._segment_id
        snapshots = self._capture_va_burst(
            generation, skill_id, lease_segment,
        )
        grounding = self._ground(
            generation=generation, skill_id=skill_id,
            target=target, direction=direction,
            image=snapshots[POLICY_FRAME_INDEX].rgb_bgr,
            strategic_goal=strategic_goal, strategic_stop=False,
        )
        if grounding["status"] != "FOUND":
            self._failed_targets.append(target)
            post = self._nav_handoff(
                generation=generation,
                skill_id=skill_id,
                target=target,
                grounding=grounding,
                snapshot=snapshots[POLICY_FRAME_INDEX],
            )
            return AgentHistoryEntry(
                skill_id, "MOVE_TO", target, "target_not_found", post["status"],
                post["visual_evidence"],
            )
        policy = {
            "action": "NAVIGATE", "bbox_2d": grounding["bbox_2d"],
            "target": target,
            "confidence": grounding["confidence"], "stop_reasoning": "",
        }
        geometry = build_object_nav_geometry_from_frames(
            policy,
            [(item.depth_mm, item.fx, item.cx) for item in snapshots],
            max_direct_travel=MAX_DIRECT_TRAVEL,
        )
        segment = self._next_segment()
        self.submit_intent("navigation_goal", {
            "generation": generation, "skill_id": skill_id,
            "segment_id": segment,
            "goal_base": [geometry["goal_x"], geometry["goal_y"]],
            "target": target,
            "confidence": float(grounding["confidence"]),
        })
        status = self._wait(generation, skill_id, segment)
        move_succeeded = status.get("state") == "reached"
        if not move_succeeded:
            self._failed_targets.append(target)
        result_snapshot = self._capture_va_snapshot(
            generation, skill_id, segment,
        )
        result_grounding = self._ground(
            generation=generation,
            skill_id=skill_id,
            target=target,
            direction="front",
            image=result_snapshot.rgb_bgr,
            strategic_goal=strategic_goal,
            strategic_stop=False,
        )
        post = self._nav_handoff(
            generation=generation,
            skill_id=skill_id,
            target=target,
            grounding=result_grounding,
            snapshot=result_snapshot,
        )
        entry = AgentHistoryEntry(
            skill_id, "MOVE_TO", target,
            str(status.get("state", "failed")), post["status"],
            post["visual_evidence"],
        )
        if move_succeeded and post["status"] == "SATISFIED":
            self._remember_move_to(entry, result_snapshot.rgb_bgr)
        return entry

    def _align(
        self, generation: int, skill_id: int, args: Mapping[str, Any],
        _expected: str, strategic_goal: str,
    ) -> AgentHistoryEntry:
        if args:
            raise LaViRAAgentError("ALIGN skill_args must be empty")
        self._face_scan_direction(generation, skill_id, "front")
        alignment_image = self.camera.capture_rgb()
        grounding = self._alignment_ground(
            generation=generation,
            skill_id=skill_id,
            image=alignment_image,
            strategic_goal=strategic_goal,
        )
        target = str(grounding["target"])
        surface = str(grounding["surface"])
        if grounding["status"] != "FOUND":
            post = self._align_handoff(
                generation=generation,
                skill_id=skill_id,
                strategic_goal=strategic_goal,
                controller_aligned=False,
                controller_state="target_not_found",
            )
            return AgentHistoryEntry(
                skill_id, "ALIGN", target, "target_not_found", post["status"],
                post["visual_evidence"],
            )
        segment = self._next_segment()
        self.submit_intent("start_base_pose", {
            "generation": generation, "skill_id": skill_id,
            "segment_id": segment, "target": target,
            "surface": surface,
            "reference_bbox": grounding["bbox_2d"],
        })
        status = self._wait(generation, skill_id, segment)
        aligned = (
            status.get("state") == "reached"
            and status.get("reason") == "aligned"
        )
        controller_state = (
            "aligned" if aligned else str(status.get("state", "failed"))
        )
        post = self._align_handoff(
            generation=generation,
            skill_id=skill_id,
            strategic_goal=strategic_goal,
            controller_aligned=aligned,
            controller_state=controller_state,
        )
        return AgentHistoryEntry(
            skill_id, "ALIGN", target,
            controller_state,
            post["status"], post["visual_evidence"],
        )

    def _manipulate(
        self, generation: int, skill_id: int, expected: str,
        strategic_goal: str,
    ) -> AgentHistoryEntry:
        ready, reason = self.readiness(generation, skill_id)
        if not ready:
            raise LaViRAAgentError(f"manipulate_gate:{reason}")
        self._manipulation_started = True
        handoff = (
            f"ORIGINAL TASK:\n{self.mission}\n\nHANDOFF STATE:\n"
            "Language Agent 已从当前四向观测中选择 MANIPULATE。\n"
            "从机器人当前状态执行并完成原始任务。"
        )
        self.submit_intent("start_vla_task", {
            "generation": generation, "skill_id": skill_id, "window_id": 0,
            "task": self.mission, "handoff_context": handoff,
        })
        try:
            try:
                start_status = self.wait_status(
                    generation,
                    skill_id,
                    0,
                    self.vla_start_timeout_seconds,
                )
            except TypeError:
                start_status = self.wait_status(
                    generation,
                    0,
                    self.vla_start_timeout_seconds,
                )
        except TimeoutError as exc:
            self.submit_intent("stop_vla_task", {
                "generation": generation,
                "skill_id": skill_id,
                "window_id": 0,
                "reason": "vla_start_timeout",
            })
            raise LaViRAAgentError(
                "unexpected_termination:vla_service_unavailable"
            ) from exc
        self._check_cancelled(generation)
        start_state = str(start_status.get("state", "failed"))
        if start_state != "active":
            reason = str(start_status.get("reason", "vla_start_failed"))
            self.submit_intent("stop_vla_task", {
                "generation": generation,
                "skill_id": skill_id,
                "window_id": 0,
                "reason": reason,
            })
            raise LaViRAAgentError(f"unexpected_termination:{reason}")
        self._event(
            logging.INFO,
            "VLA_TASK_ACKNOWLEDGED",
            "VLA service accepted the manipulation task",
            generation=generation,
            skill_id=skill_id,
            segment_id=max(0, self._segment_id),
            window_id=0,
        )
        started = self.monotonic()
        unknown_count = 0
        last_post: dict[str, Any] | None = None
        for window_id in range(1, self.manipulation_max_windows + 1):
            self._check_cancelled(generation)
            self._event(
                logging.INFO,
                "MANIPULATION_WINDOW_STARTED",
                "VLA execution window started",
                generation=generation,
                skill_id=skill_id,
                segment_id=max(0, self._segment_id),
                window_id=window_id,
                max_windows=self.manipulation_max_windows,
            )
            if self.monotonic() - started > self.manipulation_timeout_seconds:
                self.submit_intent("stop_vla_task", {
                    "generation": generation, "skill_id": skill_id,
                    "window_id": window_id, "reason": "timeout",
                })
                raise LaViRAAgentError("manipulate_timeout")
            self.sleep(self.manipulation_window_seconds)
            failure = self.poll_failure(generation, skill_id)
            if failure is not None:
                self.submit_intent("stop_vla_task", {
                    "generation": generation,
                    "skill_id": skill_id,
                    "window_id": window_id,
                    "reason": str(failure.get("reason", "safety_failure")),
                })
                raise LaViRAAgentError(
                    f"manipulate_safety:{failure.get('reason', 'unknown')}"
                )
            self.submit_intent("hold_vla_task", {
                "generation": generation, "skill_id": skill_id,
                "window_id": window_id,
            })
            post = self._postcheck(
                "MANIPULATE", expected, generation=generation,
                skill_id=skill_id, strategic_goal=strategic_goal,
                strategic_stop=True, window_id=window_id,
            )
            if post["status"] == "UNKNOWN":
                post = self._postcheck(
                    "MANIPULATE", expected, generation=generation,
                    skill_id=skill_id, strategic_goal=strategic_goal,
                    strategic_stop=True, window_id=window_id,
                )
                if post["status"] == "UNKNOWN":
                    unknown_count += 1
                    if unknown_count >= 3:
                        self.submit_intent("stop_vla_task", {
                            "generation": generation, "skill_id": skill_id,
                            "window_id": window_id,
                            "reason": "postcheck_unknown",
                        })
                        raise LaViRAAgentError("manipulate_postcheck_unknown")
                else:
                    unknown_count = 0
            else:
                unknown_count = 0
            last_post = post
            if post["transition"] == "TASK_COMPLETE":
                self.submit_intent("stop_vla_task", {
                    "generation": generation, "skill_id": skill_id,
                    "window_id": window_id,
                    "reason": "postcondition_satisfied",
                })
                return AgentHistoryEntry(
                    skill_id, "MANIPULATE", self.global_target, "stopped",
                    "SATISFIED", post["visual_evidence"],
                )
            self.submit_intent("resume_vla_task", {
                "generation": generation, "skill_id": skill_id,
                "window_id": window_id,
                "handoff_context": (
                    f"{handoff}\n\nVA INCOMPLETE EVIDENCE:\n"
                    f"{post['visual_evidence']}"
                ),
            })
        self.submit_intent("stop_vla_task", {
            "generation": generation, "skill_id": skill_id,
            "window_id": self.manipulation_max_windows,
            "reason": "window_limit",
        })
        evidence = "" if last_post is None else str(last_post["visual_evidence"])
        raise LaViRAAgentError(f"manipulate_window_limit:{evidence}")

    def run(self, generation: int) -> LaViRATaskResult:
        step = 0
        self._reset_task_context()
        try:
            self._check_cancelled(generation)
            self._event(
                logging.INFO,
                "TASK_STARTED",
                "LaViRA manipulation task started",
                generation=generation,
                navigation_mode=self.navigation_mode,
                mission=self.mission,
                global_target=self.global_target,
                max_steps=self.max_steps,
            )
            todo = ""
            for step in range(1, self.max_steps + 1):
                self._check_cancelled(generation)
                if self._manipulation_started:
                    raise LaViRAAgentError("navigation_after_manipulate_forbidden")
                self._skill_id += 1
                panorama = self._next_observation_mode == "panorama"
                event_prefix = "PANORAMA" if panorama else "FRONT_OBSERVATION"
                self._event(
                    logging.INFO,
                    f"{event_prefix}_STARTED",
                    (
                        "Fresh four-direction LA panorama started"
                        if panorama else "Fresh fixed-front LA observation started"
                    ),
                    generation=generation,
                    step=step,
                    skill_id=self._skill_id,
                    segment_id=max(0, self._segment_id),
                )
                try:
                    scan_views = (
                        self._capture_panorama(generation, self._skill_id)
                        if panorama else self._capture_front_observation()
                    )
                except Exception as exc:
                    self._event(
                        logging.ERROR,
                        f"{event_prefix}_FAILED",
                        (
                            "Fresh four-direction LA panorama failed"
                            if panorama else "Fresh fixed-front LA observation failed"
                        ),
                        generation=generation,
                        step=step,
                        skill_id=self._skill_id,
                        segment_id=max(0, self._segment_id),
                        error=str(exc),
                    )
                    raise
                self._event(
                    logging.INFO,
                    f"{event_prefix}_COMPLETED",
                    (
                        "Fresh four-direction LA panorama completed"
                        if panorama else "Fresh fixed-front LA observation completed"
                    ),
                    generation=generation,
                    step=step,
                    skill_id=self._skill_id,
                    segment_id=max(0, self._segment_id),
                    scan_id=scan_views[0].scan_id,
                    directions=[view.direction for view in scan_views],
                    anchor_pose=list(scan_views[0].reference_pose),
                )
                if step == 1:
                    todo = self.client.initial_todo(
                        mission=self.mission,
                        scan_views=tuple(scan_views),
                    )
                    self._todo(generation, 0, todo)
                la = validate_language_action(self.client.language_action(
                    mission=self.mission,
                    navigation_mode=self.navigation_mode,
                    global_target=self.global_target,
                    current_step=step,
                    todo_list=todo,
                    scan_views=tuple(scan_views),
                    move_to_views=tuple(self._recent_move_to_views),
                    transition_result=self._latest_transition,
                ), current_step=step)
                updated_todo = la["updated_todo_list"]
                if updated_todo != todo:
                    todo = updated_todo
                    self._todo(generation, step, todo)
                self._event(
                    logging.INFO,
                    "LA_DECISION",
                    f"LA returned {str(la['decision']).lower()}",
                    generation=generation,
                    step=step,
                    skill_id=self._skill_id,
                    segment_id=max(0, self._segment_id),
                    decision=la["decision"],
                    skill=la["skill"],
                    skill_args=la["skill_args"],
                    progress_analysis=la["progress_analysis"],
                    expected_postcondition=la["expected_postcondition"],
                )
                if la["decision"] == "FAIL":
                    raise LaViRAAgentError(f"la_fail:{la['reasoning']}")
                skill = str(la["skill"])
                self._validate_transition_gate(skill)
                args = la["skill_args"]
                target_hint = str(args.get("target", ""))
                strategic_goal = (
                    f"{la['reasoning']}"
                    + (f" Requested target: {target_hint!r}." if target_hint else "")
                    + " Expected visual result: "
                    f"{la['expected_postcondition']}"
                )
                self._event(
                    logging.INFO,
                    "SKILL_STARTED",
                    f"{skill} skill started",
                    generation=generation,
                    step=step,
                    skill_id=self._skill_id,
                    segment_id=max(0, self._segment_id),
                    skill=skill,
                    target=target_hint,
                    skill_args=args,
                    expected_postcondition=la["expected_postcondition"],
                )
                try:
                    if skill == "MOVE_TO":
                        entry = self._move_to(
                            generation, self._skill_id, args,
                            la["expected_postcondition"], strategic_goal,
                        )
                    elif skill == "ALIGN":
                        entry = self._align(
                            generation, self._skill_id, args,
                            la["expected_postcondition"], strategic_goal,
                        )
                    else:
                        entry = self._manipulate(
                            generation, self._skill_id,
                            la["expected_postcondition"], strategic_goal,
                        )
                except Exception as exc:
                    self._event(
                        logging.ERROR,
                        "SKILL_FAILED",
                        f"{skill} skill failed",
                        generation=generation,
                        step=step,
                        skill_id=self._skill_id,
                        segment_id=max(0, self._segment_id),
                        skill=skill,
                        error=str(exc),
                    )
                    raise
                self._record(entry)
                unsuccessful = (
                    entry.controller_state.startswith("BLOCKED")
                    or entry.controller_state in {"failed", "target_not_found"}
                    or entry.va_result
                    in {"NOT_FOUND", "NOT_SATISFIED", "UNKNOWN"}
                )
                self._event(
                    logging.WARNING if unsuccessful else logging.INFO,
                    "SKILL_COMPLETED",
                    f"{skill} skill completed",
                    generation=generation,
                    step=step,
                    skill_id=self._skill_id,
                    segment_id=max(0, self._segment_id),
                    skill=skill,
                    target=entry.target,
                    controller_state=entry.controller_state,
                    va_result=entry.va_result,
                    evidence=entry.evidence,
                )
                if skill == "MANIPULATE":
                    if entry.va_result != "SATISFIED":
                        raise LaViRAAgentError(
                            "manipulate_returned_without_satisfied_postcheck"
                        )
                    result = LaViRATaskResult(
                        generation, "reached", "manipulation_completed", step,
                        self._skill_id, max(0, self._segment_id),
                        self.navigation_mode,
                    )
                    self._event(
                        logging.INFO,
                        "TASK_COMPLETED",
                        "LaViRA manipulation task completed after VA confirmation",
                        **asdict(result),
                    )
                    return result
            raise LaViRAAgentError(f"max_steps_exceeded:{self.max_steps}")
        except LaViRAAgentCancelled:
            result = LaViRATaskResult(
                generation, "stopped", "operator_cancelled", step,
                self._skill_id, max(0, self._segment_id), self.navigation_mode,
            )
            self._event(
                logging.WARNING,
                "TASK_CANCELLED",
                "LaViRA manipulation task cancelled",
                **asdict(result),
            )
            return result
        except Exception as exc:
            LOGGER.exception(
                "LaViRA manipulation task failed generation=%d", generation,
                extra={"runtime_event_emitted": True},
            )
            result = LaViRATaskResult(
                generation, "failed", str(exc), step, self._skill_id,
                max(0, self._segment_id), self.navigation_mode,
            )
            self._event(
                logging.ERROR,
                "TASK_FAILED",
                "LaViRA manipulation task failed",
                **asdict(result),
            )
            return result
