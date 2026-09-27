"""Interleaved actions in the existing LA loop, with fixed VLA language inputs."""

import json
import logging

from .agent import (
    AgentHistoryEntry, LaViRAAgent, LaViRAAgentError, LA_KEYS,
    _OperationContext, _first_incomplete_todo_line, _g1_view_label,
    _image_data_url, _strict_json_object, validate_language_action,
)


def _completed_lines(todo):
    return [line for line in todo.splitlines() if line.strip().lower().startswith("- [x]")]


def validate_flexible_action(value, *, available_skills, todo_list="", transition_result=None,
                             observed_directions=None):
    """Extend only LA's action schema; VA and controller schemas are unchanged."""
    if not isinstance(value, dict) or set(value) != LA_KEYS:
        raise LaViRAAgentError("LA output has an invalid object schema")
    normalized = dict(value)
    skill = str(value.get("skill", "")).upper()
    args = value.get("skill_args")
    if skill in {"ALIGN", "MANIPULATE"}:
        keys = {"skill_key", "task"} if skill == "ALIGN" else {"skill_key"}
        if not isinstance(args, dict) or set(args) != keys:
            raise LaViRAAgentError(f"{skill} requires {sorted(keys)}")
        key = args["skill_key"]
        if not isinstance(key, str) or key not in available_skills:
            raise LaViRAAgentError("LA selected an unknown VLA skill")
        if skill == "ALIGN" and (not isinstance(args["task"], str) or not args["task"].strip()):
            raise LaViRAAgentError("ALIGN requires the current operation's task from the whole mission")
        normalized["skill_args"] = {}
    complete = str(value.get("decision", "")).upper() == "COMPLETE"
    if complete:
        normalized["decision"] = "FAIL"  # Reuse all other original schema checks.
    result = validate_language_action(normalized, observed_directions=observed_directions)
    result["skill_args"] = dict(args)
    if complete:
        result["decision"] = "COMPLETE"
        if not todo_list or result["updated_todo_list"] != todo_list or _first_incomplete_todo_line(todo_list) is not None:
            raise LaViRAAgentError("COMPLETE requires the unchanged fully completed runtime TODO")
    previous = _completed_lines(todo_list)
    current = _completed_lines(result["updated_todo_list"])
    if [line for line in current if line in previous] != previous:
        raise LaViRAAgentError("LA must preserve completed TODO lines exactly and in order")
    # Runtime owns NAV and VLA completion; LA may acknowledge a ready ALIGN.
    added = [line for line in current if line not in previous]
    align_ready = (transition_result or {}).get("transition") == "READY_TO_MANIPULATE"
    if added and (not align_ready or len(added) != 1):
        raise LaViRAAgentError("Only runtime-confirmed ALIGN may be newly checked by LA")
    if result["decision"] == "EXECUTE" and _first_incomplete_todo_line(result["updated_todo_list"]) is None:
        raise LaViRAAgentError("EXECUTE requires an unfinished TODO action")
    return result


FLEXIBLE_CONTRACT = """You are the existing robot Language Agent (LA). Plan and execute the
WHOLE user mission below in one continuous action loop. There is no prewritten
mission_steps list and no other task planner. Split the whole instruction into
navigation, alignment and manipulation actions in your Markdown TODO. Preserve
the user's order and route constraints. Choose the next action from fresh images,
the TODO and runtime evidence; never infer success without runtime confirmation.

MOVE_TO: skill_args={"view_direction":"front|front_right|right|left|front_left",
"target":"a visually groundable waypoint"}. Infer global_target for the CURRENT
navigation objective from the WHOLE mission; it is not fixed for the entire run.
You may navigate before, between or after operations, and may use intermediate
waypoints. Every navigation TODO item must represent exactly one target waypoint.
Left turns, right turns, facing, and other orientation choices belong only in
the selected MOVE_TO view_direction; never create a standalone turn, face, or
orientation TODO item. Combine route turns with their destination waypoint.
Runtime completes MOVE_TO TODO items. Work on the next unfinished waypoint
after runtime confirms the previous one; never repeat a completed waypoint.

The direction refers only to the Current Panorama Views in this request.
For a five-view panorama, the image order and exact view_direction values are:
Image 1 = front (0 deg), Image 2 = front_right (-45 deg),
Image 3 = right (-90 deg), Image 4 = left (+90 deg),
Image 5 = front_left (+45 deg), relative to the panorama's forward view.
front_right and front_left are the 45-degree intermediate views; selecting
either one makes the runtime turn 45 degrees before fresh VA grounding.
Select the direction of the image that actually shows your target. If you cite
Image 5 as target evidence, use front_left, not left. A route instruction such
as "turn left" does not by itself select the left (+90 deg) image; preserve the
route constraint while choosing the matching observed waypoint and view.
Never select or reason about a rear/behind direction because no rear image is
provided. If only a fixed-front observation is supplied, use front only.
PLAN-* images are historical MOVE_TO observations, not current direction views.

ALIGN: skill_args={"skill_key":"a key from AVAILABLE VLA SKILLS",
"task":"the current operation described using the WHOLE mission"}.
Select the operation using each available skill's description (its configured
vla_prompt). The task field records your interpretation of the WHOLE mission;
it does not override that description. VA identifies the distance/centering
object and yaw reference from the selected skill's vla_prompt and a fresh image;
BasePose executes alignment. You may ALIGN directly when already near the object,
including at mission start or after another operation. Local visibility/depth
and BasePose/VA readiness checks remain mandatory. If alignment cannot start
because the object is far away or missing, choose MOVE_TO or retry as indicated.

MANIPULATE: skill_args={"skill_key":"the SAME key selected by the latest ALIGN"}.
Requires a fresh successful ALIGN and READY_TO_MANIPULATE. Even consecutive VLA
calls each require their own ALIGN. The runtime chooses the checkpoint and its
immutable trained prompt; NEVER emit a trained prompt, endpoint or checkpoint.
expected_postcondition describes this operation's physical completion. VA checks
it with the existing POSTCHECK interface. MANIPULATE is NOT terminal for the
whole mission. After runtime completion or an explicitly allowed timeout, plan
the next action from the WHOLE mission. A timeout is NOT evidence of success.

Maintain the full Markdown checklist. Preserve completed lines verbatim and in
order. Never check NAV or MANIPULATE items yourself; runtime does that. When the
latest transition is READY_TO_MANIPULATE, you may check the corresponding ALIGN
item only. You may revise unfinished items using new evidence. Keep completed
work and remaining navigation/operations represented. Do not add a separate
visual verification action: the runtime's VA check belongs to MANIPULATE.
Return COMPLETE only when the whole mission is finished and the received TODO
has no unfinished items; return that TODO unchanged. FAIL means unrecoverable
failure. COMPLETE/FAIL require skill=null and skill_args={}.
Return exactly these JSON keys:
{"global_target":"current destination or last relevant target",
"progress_analysis":"...","updated_todo_list":"- [ ] ...",
"reasoning":"...","decision":"EXECUTE|COMPLETE|FAIL",
"skill":"MOVE_TO|ALIGN|MANIPULATE|null","skill_args":{},
"expected_postcondition":"..."}
"""


SMOKE_CONTRACT = """If runtime_transition.smoke_assumed_success is true, or the
completed TODO contains SMOKE annotations, the operator explicitly
requested a smoke run with manipulation simulated. Treat that operation as done
for scheduling and continue the remaining mission, even if images show no
physical manipulation effect. Preserve SMOKE annotations in completed TODOs;
do not retry simulated operations or claim their physical success. Navigation
and alignment still require real controller and VA evidence.
"""


def request_flexible_action(client, *, mission, navigation_mode, global_target,
                            current_step, todo_list, scan_views, move_to_views,
                            transition_result, available_skills, alignment_mode="dual"):
    # Same LA client, cloud retry/archival path and cameras; no extra model call.
    contract = FLEXIBLE_CONTRACT
    if alignment_mode != "dual":
        start = contract.index("VA identifies the distance/centering")
        end = contract.index("The runtime chooses the checkpoint")
        preparation = (
            "ALIGN only selects and prepares the operation: this direct-VLA baseline skips all "
            "physical alignment and alignment VA checks. Choose MOVE_TO first if navigation is needed. "
            "READY_TO_MANIPULATE here means alignment was deliberately skipped, not that alignment succeeded. "
            if alignment_mode == "none" else
            "ALIGN uses UniLM VLM-driven pose alignment for this operation's semantic task: "
            "select a view, locate the interaction point, and predict a base pose. "
            "It uses reached navigation history, or five fresh frames when no navigation preceded it. "
            "READY_TO_MANIPULATE requires physical position/yaw controller success; "
            "there is no additional BasePose grounding or alignment VA gate. "
            if alignment_mode == "unilm_nav" else
            "ALIGN runs simple geometric approach/centering on the selected skill's configured object. "
            "There is no BasePose semantic role/yaw alignment. Alignment VA is recorded but does not gate "
            "VLA; geometric timeout/target loss also releases to VLA without claiming alignment success. "
        )
        contract = (contract[:start] + preparation + "\n\nMANIPULATE: "
                    'skill_args={"skill_key":"the SAME key selected by the latest ALIGN"}. '
                    "Requires READY_TO_MANIPULATE from a fresh ALIGN preparation for this operation. "
                    + contract[end:])
    if (transition_result or {}).get("smoke_assumed_success") or "SMOKE:" in todo_list:
        contract += "\n" + SMOKE_CONTRACT
    content = [{"type": "text", "text": json.dumps({
        "whole_mission": mission, "navigation_mode": navigation_mode,
        "current_global_target": global_target, "current_step": current_step,
        "todo_list": todo_list, "runtime_transition": transition_result,
        "available_vla_skills": available_skills,
    }, ensure_ascii=False)}]
    for index, view in enumerate(scan_views[-5:], 1):
        content.extend([
            {"type": "image_url", "image_url": {"url": _image_data_url(view.image_bgr)}},
            {"type": "text", "text": _g1_view_label(view.direction, current_step, index)},
        ])
    recent = move_to_views[-5:]
    for index, view in enumerate(recent):
        content.extend([
            {"type": "image_url", "image_url": {"url": _image_data_url(view.image_bgr)}},
            {"type": "text", "text": f"PLAN-{len(recent) - index}"},
        ])
    return client._create(
        client.la_client, request_kind="la_decision", model=client.la_model,
        response_parser=lambda completion: validate_flexible_action(
            _strict_json_object(client._content(completion, role="LA"), role="LA"),
            available_skills=available_skills, todo_list=todo_list,
            transition_result=transition_result,
            observed_directions={view.direction for view in scan_views},
        ),
        messages=client._messages(content, enable_thinking=client.la_enable_thinking,
                                  system_instruction=contract),
        max_tokens=1800, temperature=0, timeout=client.la_timeout_seconds,
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": client.la_enable_thinking},
    )


class FlexibleMissionAgent(LaViRAAgent):
    """Extend the original LA loop; all actions use the original VA/controllers."""

    smoke_assume_vla_success = False

    def __init__(self, *, skills, smoke_assume_vla_success=False, **kwargs):
        super().__init__(**kwargs)
        self.skills = skills
        self.smoke_assume_vla_success = smoke_assume_vla_success

    def run(self, generation, **kwargs):
        self._flex_generation = generation
        self._active_vla_skill = None
        self._timed_out_operations = 0
        self._completed_actions = 0
        self._configured_global_target = ""
        if self.smoke_assume_vla_success:
            self._event(logging.WARNING, "SMOKE_VLA_ENABLED",
                        "Live navigation/alignment; VLA operations will be skipped and assumed successful",
                        generation=generation, smoke=True)
        return super().run(generation, **kwargs)

    def _request_language_action(self, **kwargs):
        catalog = {name: {"description": spec["prompt"]} for name, spec in self.skills.items()}
        # No checkpoint language, endpoints or fixed task order are exposed to LA.
        result = validate_flexible_action(
            self.client.language_action(**kwargs, available_skills=catalog),
            available_skills=catalog, todo_list=kwargs["todo_list"],
            transition_result=kwargs["transition_result"],
            observed_directions={view.direction for view in kwargs["scan_views"]},
        )
        if result["decision"] == "COMPLETE":
            if not self._completed_actions or self._operation is not None:
                raise LaViRAAgentError("mission completion requires finished controller actions")
            return result
        if result["decision"] == "FAIL":
            return result
        skill, args = result["skill"], result["skill_args"]
        self.global_target = result["global_target"]
        if skill == "MOVE_TO":
            self._operation = None
            self._active_vla_skill = None
            self._global_target_navigation_ready = False
        elif skill == "ALIGN":
            key = args["skill_key"]
            spec = self.skills[key]
            self.base_pose_parameters = dict(spec.get("base_pose", {}))
            self._active_vla_skill = key
            # parse_skill_definitions normalizes vla_prompt to prompt. Use the
            # selected skill's configured VA context, never LA's task paraphrase
            # or the checkpoint's training language.
            self.manipulation_prompt = spec["prompt"]
            self.vla_trained_prompt = spec["vla_trained_prompt"]
            self.manipulation_postcondition = spec.get("expected_postcondition", "")
            self.vla_task_parameters = {
                "skill_key": spec["backend"], "continue_mission": True,
                "completion_settle_seconds": spec.get("completion_settle_seconds", 0.0),
                "exit_policy": spec["exit"],
            }
            if spec.get("timeout_seconds") is not None:
                self.vla_task_parameters.update(timeout_seconds=spec["timeout_seconds"],
                                                on_timeout=spec.get("on_timeout", "fail"))
            self._operation = _OperationContext(
                self._flex_generation, f"{key}_{self._skill_id}", {}, "visual",
            )
            result["skill_args"] = {}  # Original BasePose interface.
        else:
            if args["skill_key"] != self._active_vla_skill:
                raise LaViRAAgentError("MANIPULATE must use the skill from the latest ALIGN")
            result["skill_args"] = {}  # Original VLA start interface.
        return result

    def _validate_transition_gate(self, skill):
        if skill in {"MOVE_TO", "ALIGN"}:
            return  # ALIGN retains the existing explicit-operation local depth gate.
        if (self._operation is None or not self._latest_transition
            or self._latest_transition.get("transition") != "READY_TO_MANIPULATE"):
            raise LaViRAAgentError("handoff_gate:fresh_alignment_required_for_MANIPULATE")
        return super()._validate_transition_gate(skill)

    def _align(self, generation, skill_id, args, strategic_goal):
        try:
            entry = super()._align(generation, skill_id, args, strategic_goal)
        except LaViRAAgentError as exc:
            if not str(exc).startswith("alignment_requires_navigation:"):
                raise
            self._store_transition("ALIGN", {
                "status": "NOT_SATISFIED", "transition": "RETURN_TO_NAVIGATION",
                "visual_evidence": str(exc), "confidence": 0.0,
            })
            return AgentHistoryEntry(skill_id, "ALIGN", self.global_target,
                                     "target_not_found", "NOT_SATISFIED", str(exc))
        if entry.controller_state == "aligned" and self._latest_transition.get("transition") == "READY_TO_MANIPULATE":
            operation = self._operation
            operation.alignment_skill_id = skill_id
            operation.alignment_segment_id = self._segment_id
            operation.aligned_pose = self._fastlio_pose()
            operation.aligned_yaw = self._sonic_yaw()
            operation.aligned_at = self.monotonic()
        return entry

    def _record(self, entry):
        super()._record(entry)
        if entry.skill == "MOVE_TO" and entry.va_result == "SATISFIED":
            self._completed_actions += 1

    def _manipulate(self, generation, skill_id, expected, strategic_goal):
        if not self.smoke_assume_vla_success:
            return super()._manipulate(generation, skill_id, expected, strategic_goal)
        # Stay in PLANNER: no POSE entry, VLA task, policy request, or fake CPP ack.
        # Consume only a real, fresh ALIGN; cancellation/failures still stop the run.
        self._validate_transition_gate("MANIPULATE")
        operation = self._check_operation_alignment(generation)
        ready, reason = self.readiness(generation, skill_id)
        if not ready:
            raise LaViRAAgentError(f"manipulate_gate:{reason}")
        failure = self.poll_failure(generation, skill_id)
        if failure is not None:
            raise LaViRAAgentError(f"manipulate_safety:{failure.get('reason', 'unknown')}")
        self._check_cancelled(generation)
        operation.consumed = True
        return AgentHistoryEntry(
            skill_id, "MANIPULATE", self.global_target, "smoke_assumed_success", "UNKNOWN",
            "SMOKE: VLA skipped; success assumed for sequencing only. No physical completion was checked.",
        )

    def _continue_after_manipulation(self, generation, step, entry, todo):
        line = _first_incomplete_todo_line(todo)
        if line is None:
            raise LaViRAAgentError("MANIPULATE requires an unfinished TODO item")
        timed_out = entry.controller_state == "timeout_continued"
        simulated = entry.controller_state == "smoke_assumed_success"
        self._timed_out_operations += int(timed_out)
        self._completed_actions += 1
        lines = todo.splitlines()
        result = "Timed out; continued without success confirmation." if timed_out else "VA confirmed operation completion."
        if simulated:
            result = "SMOKE: VLA skipped; assumed successful for sequencing, not physically verified."
        lines[line] = lines[line].replace("[ ]", "[x]", 1) + " Result: Runtime " + result
        self._event(logging.WARNING if timed_out or simulated else logging.INFO,
                    "SMOKE_VLA_ASSUMED_SUCCESS" if simulated else
                    "MISSION_OPERATION_TIMEOUT_CONTINUED" if timed_out else "MISSION_OPERATION_COMPLETED",
                    result, generation=generation, skill_id=entry.skill_id,
                    vla_skill=self._active_vla_skill, va_result=entry.va_result)
        self._store_transition("MANIPULATE", {
            "status": entry.va_result, "transition": "UNKNOWN" if timed_out else "TASK_COMPLETE",
            "visual_evidence": entry.evidence, "completed_vla_skill": self._active_vla_skill,
            "timeout_continued": timed_out,
            "smoke_assumed_success": simulated,
        })
        self._operation = None
        self._active_vla_skill = None
        self._global_target_navigation_ready = False
        updated_todo = "\n".join(lines)
        # Keep the final LA decision, without turning just to confirm completion.
        self._next_observation_mode = (
            "front" if _first_incomplete_todo_line(updated_todo) is None else "panorama"
        )
        return updated_todo

    def _mission_completion_reason(self):
        if self.smoke_assume_vla_success:
            return "smoke_mission_completed"
        return "mission_completed_with_timeouts" if self._timed_out_operations else "mission_completed"
