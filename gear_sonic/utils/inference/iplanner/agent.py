"""Original LaViRA navigation decisions, adapted to Groot's camera and skills.

LA chooses a direction/strategy; VA independently selects a navigation target
or area. Object grounding remains available for the existing stage handoffs.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
from pathlib import Path
import time
import uuid

from gear_sonic.utils.inference.lavira.agent import (
    DIRECTIONS,
    LaViRAAgentError,
    LaViRAClient,
    _g1_view_label,
    _image_data_url,
    _strict_json_object,
    validate_grounding,
    validate_language_action,
)

LOGGER = logging.getLogger("sonic.lavira")


def _json_contract(prompt, marker, example):
    """Keep the upstream task rules and replace its transport-format example."""
    instructions, separator, _ = prompt.partition(marker)
    if not separator:
        raise LaViRAAgentError(f"LaViRA navigation prompt is missing {marker}")
    return (instructions + marker + "\nReturn one JSON object with every field shown below.\n"
            + json.dumps(example, indent=2, ensure_ascii=False) + "\n")


class IPlannerLaViRAClient(LaViRAClient):
    def __init__(self, *, navigation_prompts_path: Path, **kwargs):
        super().__init__(**kwargs)
        spec = importlib.util.spec_from_file_location("lavira_g1_navigation_prompts", navigation_prompts_path)
        self.navigation_prompts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.navigation_prompts)
        self.navigation_strategy = ""
        self.navigation_stop = False

    def _save_navigation_response(self, kind, role, content, error):
        directory = getattr(self, "request_context_dir", None)
        if directory is None or not getattr(self, "save_request_context", True):
            return
        try:
            directory = Path(directory) / "responses"
            directory.mkdir(parents=True, exist_ok=True)
            timestamp = time.time_ns()
            path = directory / f"{timestamp}_{kind}_{uuid.uuid4().hex[:8]}.json"
            path.write_text(json.dumps({
                "type": "sonic.model_response_context", "version": 1,
                "timestamp_ns": timestamp, "request_kind": kind, "role": role,
                "model": self.la_model if role == "LA" else self.va_model,
                "content": content, "validation_error": error,
            }, indent=2, ensure_ascii=False) + "\n")
        except OSError as exc:
            LOGGER.warning("Could not save LaViRA navigation response: %s", exc)

    def _navigation_call(self, *, role, kind, content, parser, max_tokens=1024):
        is_la = role == "LA"
        thinking = self.la_enable_thinking if is_la else self.va_enable_thinking

        def parse_completion(completion):
            raw = self._content(completion, role=role)
            try:
                parsed = parser(raw)
            except LaViRAAgentError as exc:
                self._save_navigation_response(kind, role, raw, str(exc))
                raise
            self._save_navigation_response(kind, role, raw, None)
            return parsed

        return self._create(
            self.la_client if is_la else self.va_client,
            request_kind=kind, model=self.la_model if is_la else self.va_model,
            response_parser=parse_completion,
            messages=self._messages(content, enable_thinking=thinking),
            max_tokens=max_tokens, temperature=0.1 if kind == "la_navigation_todo" else 0,
            timeout=self.la_timeout_seconds if is_la else self.va_timeout_seconds,
            extra_body={"enable_thinking": thinking},
            **({} if kind == "la_navigation_todo" else {"response_format": {"type": "json_object"}}),
        )

    def language_action(self, **kwargs):
        transition = kwargs.get("transition_result") or {}
        # Keep the current ALIGN/MANIPULATE planner and all its handoff gates.
        if (transition.get("transition") in {"READY_TO_ALIGN", "READY_TO_MANIPULATE", "RETRY_ALIGN"}
                or (transition.get("executed_skill") == "ALIGN"
                    and transition.get("transition") != "RETURN_TO_NAVIGATION")):
            return super().language_action(**kwargs)
        mission, step = kwargs["mission"], kwargs["current_step"]
        global_target = kwargs["global_target"]
        views = []
        for index, view in enumerate(kwargs["scan_views"], start=1):
            views.extend([
                {"type": "image_url", "image_url": {"url": _image_data_url(view.image_bgr)}},
                {"type": "text", "text": _g1_view_label(view.direction, step, index)},
            ])
        todo = kwargs["todo_list"]
        if not todo.strip():
            def parse_todo(text):
                if not any(line.strip().startswith("- [ ]") for line in text.splitlines()):
                    raise LaViRAAgentError("LaViRA navigation TODO must be a Markdown checklist")
                return text.strip()

            todo = self._navigation_call(
                role="LA", kind="la_navigation_todo", max_tokens=512, parser=parse_todo,
                content=[{"type": "text", "text": f"Instruction: {mission}"}, *views,
                         {"type": "text", "text": self.navigation_prompts.get_todo_generator_prompt()}],
            )
        content = [{"type": "text", "text": f"Navigation Task: {mission}\nCurrent Step: {step}"}]
        for index, view in enumerate(kwargs["move_to_views"][-5:]):
            content.extend([
                {"type": "image_url", "image_url": {"url": _image_data_url(view.image_bgr)}},
                {"type": "text", "text": f"History Plan {index + 1}"},
            ])
        content.extend(views)
        prompt = self.navigation_prompts.get_navigation_prompt_text(
            mission, global_target or "Not set yet; identify the final navigation destination",
            todo, "Recent visual history is provided above.", step,
        )
        # The robot still captures Groot's five directions, with no rear view.
        directions = '"front", "front_right", "right", "left", "front_left"'
        prompt = prompt.replace('"front", "left", "right", "behind"', directions)
        prompt = prompt.replace('"front", "left", "right"', directions)
        prompt = prompt.replace('"front" or "right" or "left" or "behind"', directions)
        prompt = _json_contract(prompt, "**JSON RESPONSE FORMAT**:", {
            "progress_analysis": "Assessment of current navigation progress",
            "updated_todo_list": "- [ ] Next navigation step",
            "reasoning": "Why choose this direction and strategy",
            "turn_direction": "front", "stop": False,
            "expected_landmark": "Next navigation object or area",
            "global_target": global_target or "Short name of the final destination object or area",
        })
        prompt += (
            f"\nturn_direction must be one of {directions}. stop must be a JSON boolean. "
            "global_target must name the final navigation object/area, using a short noun phrase; "
            "do not copy the route instruction or subsequent manipulation actions into this field. "
            + (f"preserve exactly {json.dumps(global_target)}." if global_target else
               "Infer that destination from the mission. Subsequent manipulation context: "
               + str(kwargs.get("manipulation_prompt") or mission))
            + "\nChoose direction and strategy; VA will independently choose the best navigation object/area."
        )
        content.append({"type": "text", "text": prompt})

        def parse(text):
            value = _strict_json_object(text, role="LA navigation")
            if (value.get("turn_direction") not in DIRECTIONS or type(value.get("stop")) is not bool
                    or not isinstance(value.get("expected_landmark"), str)):
                raise LaViRAAgentError("invalid LaViRA direction/stop/expected_landmark")
            destination = global_target or value.get("global_target")
            if global_target and value.get("global_target", global_target) != global_target:
                raise LaViRAAgentError("LaViRA navigation changed the frozen global target")
            if not global_target and isinstance(destination, str):
                normalized = destination.strip().rstrip(".。").casefold()
                if normalized in {str(text or "").strip().rstrip(".。").casefold()
                                  for text in (mission, kwargs.get("manipulation_prompt"))}:
                    raise LaViRAAgentError("global_target must be an object/area name, not a copied task instruction")
            updated_todo = value.get("updated_todo_list")
            if not isinstance(updated_todo, str):
                raise LaViRAAgentError("invalid LaViRA navigation TODO")
            if not any(line.strip().startswith("- [ ]") for line in updated_todo.splitlines()):
                updated_todo += "\n- [ ] Finish the navigation approach and hand off to alignment."
            decision = validate_language_action({
                "global_target": destination,
                "progress_analysis": value.get("progress_analysis"),
                "updated_todo_list": updated_todo,
                "reasoning": value.get("reasoning"),
                "decision": "EXECUTE", "skill": "MOVE_TO",
                "skill_args": {"view_direction": value["turn_direction"],
                               "target": value["expected_landmark"].strip() or destination},
                "expected_postcondition": value["expected_landmark"].strip() or str(destination),
            }, expected_global_target=global_target)
            self.navigation_strategy = f"Go {value['turn_direction']}. {value['reasoning']}"
            self.navigation_stop = value["stop"]
            return decision

        return self._navigation_call(role="LA", kind="la_navigation", content=content, parser=parse)

    def navigation_waypoint(self, *, mission, global_target, strategic_goal, image_bgr):
        # As in G1 VLNTask, request a bbox even after LA's strategic stop so a
        # final approach can run. The existing Groot handoff verifies arrival.
        prompt = self.navigation_prompts.get_tactical_eyes_prompt(
            mission, global_target, self.navigation_strategy or strategic_goal, False,
        )
        prompt = _json_contract(prompt, "**JSON FORMAT**:", {
            "visual_check": "What is visible and how it matches the navigation strategy",
            "action": "NAVIGATE", "bbox_2d": [100, 200, 700, 900],
            "target": "Name of the selected navigation object or area",
            "confidence": 0.85, "stop_reasoning": "",
        })
        prompt += (
            "\naction must be NAVIGATE or STOP. Coordinates must be normalized to [0,1000]. "
            "confidence is REQUIRED for both actions: use a JSON number between 0.0 and 1.0, "
            "not a string, percentage, boolean or null. Estimate it from the actual image; "
            "the example value and coordinates are format examples, not observations. "
            "NAVIGATE requires a valid bbox_2d; STOP may use null if no target can be located. "
            "Select the navigation object or area yourself from the mission and current strategy. "
            "If selecting GLOBAL TARGET, return its exact name in target."
        )

        def parse(text):
            value = _strict_json_object(text, role="VA navigation")
            if value.get("action") not in {"NAVIGATE", "STOP"}:
                raise LaViRAAgentError("invalid LaViRA waypoint action")
            if not isinstance(value.get("target"), str) or not value["target"].strip():
                raise LaViRAAgentError("LaViRA waypoint target/area is required")
            found = value.get("bbox_2d") is not None
            if value["action"] == "NAVIGATE" and not found:
                raise LaViRAAgentError("LaViRA NAVIGATE requires a waypoint bbox")
            confidence = value.get("confidence")
            if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                    or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                detail = "missing" if "confidence" not in value else repr(confidence)
                raise LaViRAAgentError(
                    f"VA waypoint confidence is {detail}; include confidence as a JSON number in [0,1]"
                )
            result = validate_grounding({
                "mode": "GROUNDING", "status": "FOUND" if found else "NOT_FOUND",
                "bbox_2d": value.get("bbox_2d"), "point_2d": None,
                "target_description": value["target"], "confidence": value.get("confidence"),
            })
            result["navigation_action"] = value["action"]
            return result

        return self._navigation_call(
            role="VA", kind="va_navigation_waypoint", parser=parse,
            content=[{"type": "image_url", "image_url": {"url": _image_data_url(image_bgr)}},
                     {"type": "text", "text": prompt}],
        )
