"""Single-chest-camera Uni-LaViRA LA/VA navigation agent.

Portions of the prompts and LA/VA orchestration are adapted from Uni-LaViRA,
commit 215e7aca, Copyright the Uni-LaViRA contributors.  Those portions are
licensed under CC BY-NC-SA 4.0; see legal/THIRD-PARTY SOFTWARE NOTICES -
GEAR-SONIC.txt.  No robot driver, ROS1, iPlanner, Unitree SDK, Web, speech, or
direct velocity-control source is included here.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import logging
import math
import os
import time
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from gear_sonic.utils.inference.lavira.geometry import (
    FRAME_COUNT,
    MAX_DIRECT_TRAVEL,
    POLICY_FRAME_INDEX,
    build_object_nav_geometry_from_frames,
)
from gear_sonic.utils.inference.lavira.object_nav import RGBDSnapshot

LOGGER = logging.getLogger("sonic.lavira")
DEFAULT_MODEL = "Qwen3.5-27B-Q4_K_M"
LA_KEYS = {
    "progress_analysis",
    "updated_todo_list",
    "reasoning",
    "turn_direction",
    "stop",
    "expected_landmark",
}
VA_KEYS = {
    "visual_check",
    "action",
    "bbox_2d",
    "target",
    "target_type",
    "confidence",
    "stop_reasoning",
}
EQA_KEYS = {"reasoning", "answer"}
DIRECTION_DELTAS = {
    "front": 0.0,
    "left": math.pi / 2.0,
    "right": -math.pi / 2.0,
    "behind": math.pi,
}
TARGET_TYPES = {"global_target", "intermediate_landmark", "traversable_opening"}


class LaViRAAgentError(RuntimeError):
    """A fail-closed task error suitable for a structured task result."""


class LaViRAAgentCancelled(LaViRAAgentError):
    """Raised when the generation is invalidated by Space/cancellation."""


def _image_data_url(image_bgr: np.ndarray) -> str:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise LaViRAAgentError("LaViRA chest RGB must be HxWx3 uint8")
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise LaViRAAgentError("failed to encode LaViRA chest RGB")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


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


def validate_language_action(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != LA_KEYS:
        raise LaViRAAgentError("LA output has an invalid object schema")
    for key in LA_KEYS - {"stop"}:
        if not isinstance(value[key], str):
            raise LaViRAAgentError(f"LA {key} must be a string")
    if value["turn_direction"].lower() not in DIRECTION_DELTAS:
        raise LaViRAAgentError("LA turn_direction is invalid")
    if not isinstance(value["stop"], bool):
        raise LaViRAAgentError("LA stop must be a boolean")
    return dict(value)


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_vision_action(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != VA_KEYS:
        raise LaViRAAgentError("VA output has an invalid object schema")
    for key in {"visual_check", "target", "target_type", "stop_reasoning"}:
        if not isinstance(value[key], str):
            raise LaViRAAgentError(f"VA {key} must be a string")
    if value["action"] not in {"NAVIGATE", "STOP"}:
        raise LaViRAAgentError("VA action is invalid")
    if value["target_type"] not in TARGET_TYPES:
        raise LaViRAAgentError("VA target_type is invalid")
    confidence = value["confidence"]
    if not _finite_number(confidence) or not 0.0 <= float(confidence) <= 1.0:
        raise LaViRAAgentError("VA confidence is invalid")
    bbox = value["bbox_2d"]
    if bbox is not None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise LaViRAAgentError("VA bbox_2d is invalid")
        if not all(_finite_number(item) and 0.0 <= float(item) <= 1000.0 for item in bbox):
            raise LaViRAAgentError("VA bbox_2d is invalid")
        x1, y1, x2, y2 = map(float, bbox)
        if x1 >= x2 or y1 >= y2:
            raise LaViRAAgentError("VA bbox_2d has invalid corner ordering")
    if value["action"] == "NAVIGATE" and bbox is None:
        raise LaViRAAgentError("VA NAVIGATE requires bbox_2d")
    if value["action"] == "STOP" and not value["stop_reasoning"].strip():
        raise LaViRAAgentError("VA STOP requires stop_reasoning")
    return dict(value)


def validate_eqa_answer(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != EQA_KEYS:
        raise LaViRAAgentError("EQA output has an invalid object schema")
    if not all(isinstance(value[key], str) and value[key].strip() for key in EQA_KEYS):
        raise LaViRAAgentError("EQA reasoning and answer must be non-empty strings")
    return {key: value[key] for key in EQA_KEYS}


def todo_prompt() -> str:
    return """Create a dynamic checklist to complete the navigation instruction from
the current chest-camera view. Break the mission into logical, sequential,
visually verifiable steps. Return ONLY a Markdown checklist using '- [ ]'."""


def language_action_prompt(
    mission: str,
    global_target: str,
    todo_list: str,
    history_text: str,
) -> str:
    return f"""ROLE: You are the strategic Language-Action navigator for a humanoid.
MISSION: {json.dumps(mission, ensure_ascii=False)}
GLOBAL TARGET: {json.dumps(global_target, ensure_ascii=False)}
CURRENT TODO LIST:
{todo_list}

Use only the current single forward chest view and the recent observation/action/result
history. Update the checklist, then choose exactly one relative view direction from
front, left, right, behind. Set stop=true only when the final goal is reached.

RECENT HISTORY:
{history_text or 'No previous actions.'}

Return exactly one JSON object with these keys and no Markdown:
{{"progress_analysis":"...","updated_todo_list":"...","reasoning":"...",
"turn_direction":"front|left|right|behind","stop":false,
"expected_landmark":"..."}}"""


def vision_action_prompt(
    mission: str,
    global_target: str,
    strategic_goal: str,
    strategic_stop: bool,
) -> str:
    return f"""ROLE: You are the tactical Vision-Action eyes for a humanoid navigator.
MISSION: {json.dumps(mission, ensure_ascii=False)}
GLOBAL TARGET: {json.dumps(global_target, ensure_ascii=False)}
CURRENT STRATEGY: {json.dumps(strategic_goal, ensure_ascii=False)}
STRATEGIC STOP SIGNAL: {json.dumps(strategic_stop)}

This is the current chest view after the requested closed-loop turn. Box the visible
global target, otherwise the best strategy-aligned landmark or traversable opening.
Coordinates are normalized [0,1000]. Return STOP only when the global target is clearly
reached; otherwise return NAVIGATE with a bbox. A strategic stop may still return a bbox
for one final approach.

Return exactly one JSON object with these keys and no Markdown:
{{"visual_check":"...","action":"NAVIGATE|STOP","bbox_2d":[x1,y1,x2,y2],
"target":"...","target_type":"global_target|intermediate_landmark|traversable_opening",
"confidence":0.0,"stop_reasoning":""}}"""


def eqa_prompt(question: str) -> str:
    return f"""You are a humanoid robot that has successfully navigated to the
destination. Answer the question only from the fresh current chest-camera view.
QUESTION: {json.dumps(question, ensure_ascii=False)}
Return exactly one JSON object: {{"reasoning":"...","answer":"..."}}"""


class LaViRAClient:
    """Two-endpoint OpenAI-compatible LA/VA client adapted from Uni-LaViRA."""

    def __init__(
        self,
        *,
        la_base_url: str,
        va_base_url: str,
        la_model: str = DEFAULT_MODEL,
        va_model: str = DEFAULT_MODEL,
        la_timeout_seconds: float = 180.0,
        va_timeout_seconds: float = 180.0,
        la_api_key: str | None = None,
        va_api_key: str | None = None,
        la_client: Any | None = None,
        va_client: Any | None = None,
    ) -> None:
        if la_client is None or va_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("LaViRA requires the openai inference dependency") from exc
            if la_client is None:
                la_client = OpenAI(
                    api_key=la_api_key or os.environ.get("LAVIRA_LA_API_KEY") or "no-key",
                    base_url=la_base_url,
                    timeout=float(la_timeout_seconds),
                )
            if va_client is None:
                va_client = OpenAI(
                    api_key=va_api_key or os.environ.get("LAVIRA_VA_API_KEY") or "no-key",
                    base_url=va_base_url,
                    timeout=float(va_timeout_seconds),
                )
        self.la_client = la_client
        self.va_client = va_client
        self.la_model = str(la_model)
        self.va_model = str(va_model)
        self.la_timeout_seconds = float(la_timeout_seconds)
        self.va_timeout_seconds = float(va_timeout_seconds)

    @staticmethod
    def _create(client: Any, **kwargs: Any) -> Any:
        """Preserve Uni-LaViRA's three attempts for transient model failures."""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    break
                LOGGER.warning("LaViRA model call failed; retrying: %s", exc)
                time.sleep(10.0)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _messages(content: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": "/no_think"},
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

    def initial_todo(self, *, mission: str, image_bgr: np.ndarray) -> str:
        content = [
            {"type": "image_url", "image_url": {"url": _image_data_url(image_bgr)}},
            {"type": "text", "text": f"Instruction: {mission}\n\n{todo_prompt()}"},
        ]
        completion = self._create(
            self.la_client,
            model=self.la_model,
            messages=self._messages(content),
            max_tokens=512,
            temperature=0.1,
            timeout=self.la_timeout_seconds,
        )
        result = self._content(completion, role="LA TODO").strip()
        if not result or "- [ ]" not in result:
            raise LaViRAAgentError("LA TODO output is not a Markdown checklist")
        return result

    def language_action(
        self,
        *,
        mission: str,
        global_target: str,
        todo_list: str,
        current_image_bgr: np.ndarray,
        history: Sequence["AgentHistoryEntry"],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for index, entry in enumerate(history, start=1):
            content.extend(
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(entry.observation_bgr)},
                    },
                    {
                        "type": "text",
                        "text": f"Recent observation {index}: action={entry.action}; "
                        f"result={entry.result}",
                    },
                ]
            )
        content.extend(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(current_image_bgr)},
                },
                {"type": "text", "text": "Current chest-camera observation."},
                {
                    "type": "text",
                    "text": language_action_prompt(
                        mission,
                        global_target,
                        todo_list,
                        "\n".join(
                            f"{entry.action} -> {entry.result}" for entry in history
                        ),
                    ),
                },
            ]
        )
        completion = self._create(
            self.la_client,
            model=self.la_model,
            messages=self._messages(content),
            max_tokens=1024,
            temperature=0,
            timeout=self.la_timeout_seconds,
            response_format={"type": "json_object"},
        )
        return validate_language_action(
            _strict_json_object(self._content(completion, role="LA"), role="LA")
        )

    def vision_action(
        self,
        *,
        mission: str,
        global_target: str,
        strategic_goal: str,
        strategic_stop: bool,
        snapshot: RGBDSnapshot,
    ) -> dict[str, Any]:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(snapshot.rgb_bgr)},
            },
            {
                "type": "text",
                "text": vision_action_prompt(
                    mission, global_target, strategic_goal, strategic_stop
                ),
            },
        ]
        completion = self._create(
            self.va_client,
            model=self.va_model,
            messages=self._messages(content),
            max_tokens=1024,
            temperature=0,
            timeout=self.va_timeout_seconds,
            response_format={"type": "json_object"},
        )
        return validate_vision_action(
            _strict_json_object(self._content(completion, role="VA"), role="VA")
        )

    def answer(self, *, question: str, image_bgr: np.ndarray) -> dict[str, str]:
        content = [
            {"type": "image_url", "image_url": {"url": _image_data_url(image_bgr)}},
            {"type": "text", "text": eqa_prompt(question)},
        ]
        completion = self._create(
            self.la_client,
            model=self.la_model,
            messages=self._messages(content),
            max_tokens=1024,
            temperature=0,
            timeout=self.la_timeout_seconds,
            response_format={"type": "json_object"},
        )
        return validate_eqa_answer(
            _strict_json_object(self._content(completion, role="EQA"), role="EQA")
        )


@dataclass(frozen=True)
class AgentHistoryEntry:
    observation_bgr: np.ndarray = field(repr=False, compare=False)
    action: str
    result: str


@dataclass(frozen=True)
class LaViRATaskResult:
    generation: int
    state: str
    reason: str
    steps: int
    segment_id: int
    task_type: str
    answer: str | None = None


class LaViRAAgent:
    """Run VLN, ObjectNav, or EQA without ever producing robot velocity."""

    def __init__(
        self,
        *,
        task_type: str,
        mission: str,
        global_target: str,
        question: str,
        max_steps: int,
        history_size: int,
        min_confidence: float,
        segment_timeout_seconds: float,
        camera: Any,
        client: Any,
        submit_intent: Callable[[str, Mapping[str, object]], None],
        wait_status: Callable[[int, int, float], Mapping[str, Any]],
        cancelled: Callable[[int], bool],
    ) -> None:
        if task_type not in {"vln", "object_nav", "eqa"}:
            raise ValueError("task_type must be vln, object_nav, or eqa")
        if not mission.strip() or not global_target.strip():
            raise ValueError("mission and global_target are required")
        if task_type == "eqa" and not question.strip():
            raise ValueError("question is required for EQA")
        if max_steps <= 0 or history_size <= 0:
            raise ValueError("max_steps and history_size must be positive")
        self.task_type = task_type
        self.mission = mission
        self.global_target = global_target
        self.question = question
        self.max_steps = int(max_steps)
        self.history_size = int(history_size)
        self.min_confidence = float(min_confidence)
        self.segment_timeout_seconds = float(segment_timeout_seconds)
        self.camera = camera
        self.client = client
        self.submit_intent = submit_intent
        self.wait_status = wait_status
        self.cancelled = cancelled

    def _check_cancelled(self, generation: int) -> None:
        if self.cancelled(generation):
            raise LaViRAAgentCancelled("operator_cancelled")

    def _wait_segment(self, generation: int, segment_id: int) -> Mapping[str, Any]:
        status = self.wait_status(
            generation,
            segment_id,
            self.segment_timeout_seconds,
        )
        self._check_cancelled(generation)
        if status.get("state") != "reached":
            raise LaViRAAgentError(
                f"navdp_{status.get('state', 'failed')}:"
                f"{status.get('reason', 'unknown')}"
            )
        return status

    def _capture_va_burst(
        self, generation: int, segment_id: int
    ) -> list[RGBDSnapshot]:
        self.submit_intent(
            "lavira_depth_request",
            {"generation": generation, "segment_id": segment_id},
        )
        if hasattr(self.camera, "begin_depth_lease"):
            self.camera.begin_depth_lease(generation, segment_id)
        try:
            return [self.camera.capture_aligned_rgbd() for _ in range(FRAME_COUNT)]
        finally:
            self.submit_intent(
                "lavira_rgbd_captured",
                {"generation": generation, "segment_id": segment_id},
            )

    def run(self, generation: int) -> LaViRATaskResult:
        segment_id = -1
        step = 0
        answer: str | None = None
        try:
            self._check_cancelled(generation)
            initial_rgb = self.camera.capture_rgb()
            todo_list = self.client.initial_todo(
                mission=self.mission,
                image_bgr=initial_rgb,
            )
            history: list[AgentHistoryEntry] = []
            success_reason = ""
            for step in range(1, self.max_steps + 1):
                self._check_cancelled(generation)
                observation = self.camera.capture_rgb()
                la = validate_language_action(
                    self.client.language_action(
                        mission=self.mission,
                        global_target=self.global_target,
                        todo_list=todo_list,
                        current_image_bgr=observation,
                        history=history[-self.history_size :],
                    )
                )
                todo_list = la["updated_todo_list"]
                direction = la["turn_direction"].lower()
                segment_id += 1
                self.submit_intent(
                    "navigation_heading_goal",
                    {
                        "generation": generation,
                        "segment_id": segment_id,
                        "heading_delta_rad": DIRECTION_DELTAS[direction],
                    },
                )
                heading_status = self._wait_segment(generation, segment_id)

                snapshots = self._capture_va_burst(generation, segment_id)
                self._check_cancelled(generation)
                va = validate_vision_action(
                    self.client.vision_action(
                        mission=self.mission,
                        global_target=self.global_target,
                        strategic_goal=(
                            f"Go {direction}. {la['reasoning']} "
                            f"Look for {la['expected_landmark']}."
                        ),
                        strategic_stop=la["stop"],
                        snapshot=snapshots[POLICY_FRAME_INDEX],
                    )
                )
                if float(va["confidence"]) < self.min_confidence:
                    raise LaViRAAgentError("VA confidence below threshold")
                if va["action"] == "STOP":
                    success_reason = f"va_stop:{va['stop_reasoning']}"
                    history.append(
                        AgentHistoryEntry(observation, direction, success_reason)
                    )
                    break

                policy = {
                    "action": "NAVIGATE",
                    "bbox_2d": va["bbox_2d"],
                    "target": va["target"],
                    "target_type": va["target_type"],
                    "confidence": va["confidence"],
                    "stop_reasoning": va["stop_reasoning"],
                }
                geometry = build_object_nav_geometry_from_frames(
                    policy,
                    [(item.depth_mm, item.fx, item.cx) for item in snapshots],
                    max_direct_travel=MAX_DIRECT_TRAVEL,
                )
                segment_id += 1
                self.submit_intent(
                    "navigation_goal",
                    {
                        "generation": generation,
                        "segment_id": segment_id,
                        "goal_base": [geometry["goal_x"], geometry["goal_y"]],
                        "target": va["target"],
                        "target_type": va["target_type"],
                        "confidence": float(va["confidence"]),
                    },
                )
                nav_status = self._wait_segment(generation, segment_id)
                result_text = (
                    f"heading={heading_status.get('reason', 'reached')}; "
                    f"nav={nav_status.get('reason', 'reached')}; "
                    f"target={va['target']}"
                )
                history.append(AgentHistoryEntry(observation, direction, result_text))
                history = history[-self.history_size :]
                if la["stop"]:
                    success_reason = "la_stop_final_approach_reached"
                    break
            else:
                raise LaViRAAgentError(f"max_steps_exceeded:{self.max_steps}")

            if self.task_type == "eqa":
                self._check_cancelled(generation)
                fresh_rgb = self.camera.capture_rgb()
                eqa = validate_eqa_answer(
                    self.client.answer(question=self.question, image_bgr=fresh_rgb)
                )
                answer = eqa["answer"]
                LOGGER.info(
                    "EQA_RESULT generation=%d question=%s answer=%s reasoning=%s",
                    generation,
                    self.question,
                    answer,
                    eqa["reasoning"],
                )
            return LaViRATaskResult(
                generation=generation,
                state="reached",
                reason=success_reason or "navigation_completed",
                steps=step,
                segment_id=max(0, segment_id),
                task_type=self.task_type,
                answer=answer,
            )
        except LaViRAAgentCancelled:
            return LaViRATaskResult(
                generation,
                "stopped",
                "operator_cancelled",
                step,
                max(0, segment_id),
                self.task_type,
            )
        except Exception as exc:
            LOGGER.exception("LaViRA task failed generation=%d", generation)
            return LaViRATaskResult(
                generation,
                "failed",
                str(exc),
                step,
                max(0, segment_id),
                self.task_type,
            )
