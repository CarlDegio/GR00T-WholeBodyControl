"""Ordered navigation and Align→VLA operations over the existing controllers."""

from dataclasses import asdict, dataclass, field
import logging
import math
from typing import Mapping

from gear_sonic.utils.inference.base_pose.alignment_overrides import validate_alignment_overrides

from .agent import LaViRAAgent, LaViRATaskResult


@dataclass(frozen=True)
class MissionStep:
    # Keep the original positional fields and legacy goal-list format.
    name: str
    mission: str = ""
    global_target: str = ""
    skill_key: str = ""
    manipulation_prompt: str = ""
    expected_postcondition: str = ""
    navigation_mode: str = "object_nav"
    completion_settle_seconds: float = 5.0
    navigation_upper_body: str = "default"
    type: str = ""  # Empty: legacy NAV→ALIGN→VLA goal; navigate/skill: explicit step.
    align: dict = field(default_factory=dict)
    exit: str = ""  # Empty retains legacy gripper behavior.
    completion_kind: str = "visual"
    skill: str = ""  # Optional logical definition name, distinct from backend key.
    timeout_seconds: float | None = None  # Explicit visual Skill only; starts at VLA acknowledgement.
    on_timeout: str = "fail"
    vla_trained_prompt: str | None = None

    def __post_init__(self):
        for key in ("name", "mission", "global_target", "skill_key", "manipulation_prompt",
                    "expected_postcondition", "navigation_mode", "navigation_upper_body",
                    "type", "exit", "completion_kind", "skill"):
            if not isinstance(getattr(self, key), str):
                raise ValueError(f"mission step {key} must be a string")
        if not self.name.strip():
            raise ValueError("mission step requires name")
        if self.type not in {"", "navigate", "skill"}:
            raise ValueError("mission step type must be navigate or skill")
        if self.type != "skill" and (not self.mission.strip() or not self.global_target.strip()):
            raise ValueError("navigation/legacy mission step requires mission and global_target")
        if self.navigation_mode not in {"object_nav", "vln"}:
            raise ValueError("mission step navigation_mode must be object_nav or vln")
        if self.completion_kind not in {"visual", "manual"}:
            raise ValueError("completion kind must be visual or manual")
        if not isinstance(self.on_timeout, str) or self.on_timeout not in {"fail", "continue"}:
            raise ValueError("on_timeout must be fail or continue")
        if self.timeout_seconds is not None:
            if (type(self.timeout_seconds) not in {int, float}
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
                raise ValueError("timeout_seconds must be finite and positive")
            if self.type != "skill" or self.completion_kind != "visual":
                raise ValueError("timeout_seconds requires an explicit visual skill")
        if self.on_timeout == "continue" and self.timeout_seconds is None:
            raise ValueError("on_timeout: continue requires timeout_seconds")
        if self.type == "skill" and not self.skill_key.strip():
            raise ValueError("skill step requires a VLA backend")
        if self.vla_trained_prompt is not None and (
            not isinstance(self.vla_trained_prompt, str) or not self.vla_trained_prompt.strip()
        ):
            raise ValueError("vla_trained_prompt must be a nonempty string")
        if self.vla_trained_prompt is not None and not self.skill_key:
            raise ValueError("vla_trained_prompt requires a VLA skill")
        if self.skill_key:
            if not self.manipulation_prompt.strip():
                raise ValueError("VLA mission step requires manipulation_prompt")
            if self.completion_kind == "visual" and not self.expected_postcondition.strip():
                raise ValueError("visual completion requires expected_postcondition")
        if self.navigation_upper_body not in {"default", "hold_current"}:
            raise ValueError("navigation_upper_body must be default or hold_current")
        if self.exit not in {"", "hold", "default"} or (self.type == "skill" and not self.exit):
            raise ValueError("skill exit must be hold or default")
        if (type(self.completion_settle_seconds) not in {int, float}
            or not math.isfinite(self.completion_settle_seconds)
            or not 0 <= self.completion_settle_seconds <= 30):
            raise ValueError("completion_settle_seconds must be between 0 and 30")
        if not isinstance(self.align, dict):
            raise ValueError("align must be a mapping")
        if self.type == "skill":
            if set(self.align) - {"target", "yaw_align_target", "max_attempts"}:
                raise ValueError("align supports target, yaw_align_target and max_attempts")
            if not isinstance(self.align.get("target"), str) or not self.align["target"].strip():
                raise ValueError("every skill step requires align.target")
            if "yaw_align_target" in self.align and (
                not isinstance(self.align["yaw_align_target"], str) or not self.align["yaw_align_target"].strip()
            ):
                raise ValueError("align.yaw_align_target must be a nonempty description")
            attempts = self.align.get("max_attempts", 3)
            if type(attempts) is not int or not 1 <= attempts <= 20:
                raise ValueError("align.max_attempts must be between 1 and 20")
        elif self.align:
            raise ValueError("align belongs to an explicit skill step")
        if self.type == "navigate" and (self.skill_key or self.skill or self.exit):
            raise ValueError("navigate steps cannot select a skill or exit policy")
        if self.type != "skill" and self.completion_kind == "manual":
            raise ValueError("manual completion requires an explicit skill step")


def _completion_kind(raw):
    if (not isinstance(raw, dict) or set(raw) != {"kind"}
        or not isinstance(raw["kind"], str) or raw["kind"] not in {"visual", "manual"}):
        raise ValueError("completion must be {kind: visual} or {kind: manual}")
    return raw["kind"]


def parse_skill_definitions(raw, *, backends=None, flexible=False):
    if not isinstance(raw, dict):
        raise ValueError("lavira.skills must be a mapping")
    definitions = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
            raise ValueError("skill definitions require a name and mapping")
        if set(spec) - {"backend", "prompt", "vla_prompt", "vla_trained_prompt", "expected_postcondition", "completion", "exit",
                        "completion_settle_seconds", "timeout_seconds", "on_timeout", "geometry_target", "base_pose"}:
            raise ValueError(f"unsupported skill definition fields: {name}")
        spec = dict(spec)
        if "base_pose" in spec:
            if not flexible:
                raise ValueError("skill base_pose overrides require a flexible mission")
            spec["base_pose"] = validate_alignment_overrides(spec["base_pose"])
        if "geometry_target" in spec and (not isinstance(spec["geometry_target"], str) or not spec["geometry_target"].strip()):
            raise ValueError(f"skill {name}: geometry_target must name a nonempty object")
        if "vla_prompt" in spec:
            if "prompt" in spec:
                raise ValueError(f"skill {name}: specify only one of prompt/vla_prompt")
            if not isinstance(spec["vla_prompt"], str) or not spec["vla_prompt"].strip():
                raise ValueError(f"skill {name}: fill in vla_prompt for Agent/VA")
            if not isinstance(spec.get("vla_trained_prompt"), str) or not spec["vla_trained_prompt"].strip():
                raise ValueError(f"skill {name}: vla_prompt requires a nonempty vla_trained_prompt")
            spec["prompt"] = spec.pop("vla_prompt")
        if any(not isinstance(spec.get(key), str) or not spec[key].strip() for key in ("backend", "prompt")):
            raise ValueError(f"skill {name} requires backend and prompt")
        if backends is not None and spec["backend"] not in backends:
            raise ValueError(f"unknown mission skill backend: {spec['backend']}")
        kind = _completion_kind(spec.get("completion", {"kind": "visual"}))
        if flexible and (kind != "visual" or not isinstance(spec.get("vla_trained_prompt"), str)
                         or not spec["vla_trained_prompt"].strip()):
            raise ValueError("flexible mission skills require visual completion and a fixed vla_trained_prompt")
        expected = spec.get("expected_postcondition", "")
        if flexible and expected == "":
            expected = "LA supplies the current operation postcondition"
        # Validate even unused definitions; no models are loaded here.
        MissionStep(name=name, type="skill", skill_key=spec["backend"],
                    manipulation_prompt=spec["prompt"],
                    vla_trained_prompt=spec.get("vla_trained_prompt"),
                    expected_postcondition=expected,
                    align={"target": "definition validation"}, exit=spec.get("exit", ""),
                    completion_kind=kind,
                    completion_settle_seconds=spec.get("completion_settle_seconds", 5.0),
                    timeout_seconds=spec.get("timeout_seconds"), on_timeout=spec.get("on_timeout", "fail"))
        definitions[name] = dict(spec)
    return definitions


def parse_mission_steps(raw, *, backends: Mapping | None = None, skills=None):
    if not isinstance(raw, (list, tuple)):
        raise ValueError("mission_steps must be a list")
    definitions = parse_skill_definitions({} if skills is None else skills, backends=backends)
    steps = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("mission steps must be mappings")
        values = dict(item)
        if "vla_prompt" in values and values.get("vla_trained_prompt") is None:
            raise ValueError("vla_prompt requires vla_trained_prompt")
        for alias, canonical in (("id", "name"), ("target", "global_target"), ("vla_prompt", "manipulation_prompt")):
            if alias in values:
                if canonical in values:
                    raise ValueError(f"specify only one of {alias}/{canonical}")
                values[canonical] = values.pop(alias)
        logical = values.get("skill")
        if logical:
            if not isinstance(logical, str) or logical not in definitions:
                raise ValueError(f"unknown mission skill definition: {logical}")
            if (values.get("type") != "skill" or "skill_key" in values or "manipulation_prompt" in values
                or "vla_trained_prompt" in values):
                raise ValueError("logical skill requires type: skill; backend/prompt belong to its definition")
            spec = definitions[logical]
            defaults = dict(skill_key=spec["backend"], manipulation_prompt=spec["prompt"],
                            expected_postcondition=spec.get("expected_postcondition", ""),
                            completion_kind=_completion_kind(spec.get("completion", {"kind": "visual"})),
                            exit=spec["exit"], completion_settle_seconds=spec.get("completion_settle_seconds", 5.0),
                            timeout_seconds=spec.get("timeout_seconds"), on_timeout=spec.get("on_timeout", "fail"))
            defaults["vla_trained_prompt"] = spec.get("vla_trained_prompt")
            values = {**defaults, **values}
        if "completion" in values:
            values["completion_kind"] = _completion_kind(values.pop("completion"))
        if values.get("type") == "navigate":
            values.setdefault("mission", f"Navigate to {values.get('global_target', '')}.")
        elif values.get("type") == "skill":
            if "navigation_upper_body" in item:
                raise ValueError("explicit skill steps use exit: hold/default, not navigation_upper_body")
            values.setdefault("mission", values.get("manipulation_prompt", ""))
            alignment = values.get("align")
            if isinstance(alignment, dict):
                values.setdefault("global_target", alignment.get("target", ""))
        try:
            step = MissionStep(**values)
        except TypeError as exc:
            raise ValueError(f"invalid mission step fields: {exc}") from exc
        if backends is not None and step.skill_key and step.skill_key not in backends:
            raise ValueError(f"unknown mission skill_key: {step.skill_key}")
        steps.append(step)
    if len({step.name for step in steps}) != len(steps):
        raise ValueError("mission step names must be unique")
    return tuple(steps)


def validate_mission_profile(profile):
    from ..vla.backends import parse_backends
    vla = profile.component("vla")
    backends = parse_backends(vla.get("backends", {}))
    if vla.get("default_skill") and vla["default_skill"] not in backends:
        raise ValueError("default_skill must name a configured VLA backend")
    lavira = profile.component("lavira")
    flexible = lavira.get("flexible_mission", False)
    smoke = lavira.get("smoke_assume_vla_success", False)
    if type(smoke) is not bool:
        raise ValueError("smoke_assume_vla_success must be a boolean")
    if smoke and not flexible:
        raise ValueError("smoke_assume_vla_success requires flexible_mission")
    if type(flexible) is not bool:
        raise ValueError("flexible_mission must be a boolean")
    if flexible:
        if not isinstance(lavira.get("mission"), str) or not lavira["mission"].strip():
            raise ValueError("flexible_mission requires a nonempty whole mission prompt")
        if lavira.get("mission_steps"):
            raise ValueError("flexible_mission cannot be combined with mission_steps")
        if lavira.get("global_target", "").strip():
            raise ValueError("flexible_mission derives navigation targets from the whole mission; leave global_target empty")
        skills = parse_skill_definitions(lavira.get("skills", {}), backends=backends, flexible=True)
        if not skills:
            raise ValueError("flexible_mission requires a VLA skill catalog")
        experiment = profile.components.get("experiment")
        alignment = "dual"
        if experiment is not None:
            supported = {"full_vln": "dual", "nav_direct_vla": "none", "geometric_vla": "geometric",
                         "baselinelavira": "dual", "unilm_nav": "unilm_nav"}
            alignment = experiment.get("alignment")
            if (not experiment.get("flexible_mission") or experiment.get("method") not in supported
                or alignment != supported[experiment["method"]]
                or experiment.get("entry_stage") != "navigation"
                or experiment.get("navigation_backend") != "lavira"
                or experiment.get("gates") != {"nav": True, "align": True, "completion": True}):
                raise ValueError("flexible_mission requires a supported flexible experiment profile")
            if smoke:
                raise ValueError("smoke_assume_vla_success cannot be combined with experiment recording")
            if alignment == "geometric" and any(not spec.get("geometry_target") for spec in skills.values()):
                raise ValueError("geometric flexible missions require geometry_target for every skill")
        launcher = profile.component("launcher")
        if (launcher.get("planner_input") != "lavira" or not launcher.get("keyboard_planner")
            or (alignment != "none" and not launcher.get("base_pose_enabled"))):
            raise ValueError("flexible_mission requires the LaViRA planner and BasePose")
        return ()
    steps = parse_mission_steps(lavira.get("mission_steps", []), backends=backends,
                                skills=lavira.get("skills", {}))
    if steps:
        if "experiment" in profile.components:
            raise ValueError("mission_steps cannot be combined with single-trial experiment overlays")
        launcher = profile.component("launcher")
        if launcher.get("planner_input") != "lavira" or not launcher.get("keyboard_planner"):
            raise ValueError("mission_steps requires the LaViRA planner")
        if any(step.skill_key for step in steps) and not launcher.get("base_pose_enabled"):
            raise ValueError("VLA mission steps require BasePose alignment")
    return steps


class MissionAgent(LaViRAAgent):
    """One task generation; each explicit operation owns its ALIGN→VLA attempt."""

    def __init__(self, *, mission_steps, **kwargs):
        super().__init__(**kwargs)
        self.mission_steps = tuple(mission_steps)

    def run(self, generation):
        self._reset_task_context()
        total_steps = 0
        timed_out_steps = []
        for index, goal in enumerate(self.mission_steps, 1):
            self.reset_step_context(goal.global_target)
            self.mission = goal.mission
            self.navigation_mode = goal.navigation_mode
            self.manipulation_prompt = goal.manipulation_prompt or goal.mission
            # Reset on every step so a previous model's training text cannot leak.
            # Legacy prompt-only missions keep their original single-prompt behavior.
            self.vla_trained_prompt = goal.vla_trained_prompt if goal.vla_trained_prompt is not None else self.manipulation_prompt
            self.manipulation_postcondition = goal.expected_postcondition
            self.vla_task_parameters = {
                "skill_key": goal.skill_key,
                "continue_mission": True,
                "completion_settle_seconds": goal.completion_settle_seconds,
                "navigation_upper_body": goal.navigation_upper_body,
            }
            if goal.exit:
                self.vla_task_parameters["exit_policy"] = goal.exit
            if goal.timeout_seconds is not None:
                self.vla_task_parameters.update(timeout_seconds=goal.timeout_seconds, on_timeout=goal.on_timeout)
            self._event(logging.INFO, "MISSION_STEP_STARTED", goal.name,
                        generation=generation, mission_step=index, goal=asdict(goal))
            if goal.type == "skill":
                result = self.run_operation(generation, operation_id=goal.name,
                                            alignment=goal.align, completion_kind=goal.completion_kind)
            elif goal.type == "navigate":
                result = self.run_navigation(generation)
            else:
                result = super().run(generation, reset_context=False,
                                     navigation_only=not goal.skill_key, subtask=True)
            total_steps += result.steps
            if result.state != "reached":
                result = LaViRATaskResult(generation, result.state, result.reason, total_steps,
                                         result.skill_id, result.segment_id, result.navigation_mode)
                self._event(logging.ERROR if result.state == "failed" else logging.WARNING,
                            "TASK_FAILED" if result.state == "failed" else "TASK_CANCELLED",
                            "Mission stopped before completing all steps", **asdict(result))
                return result
            timed_out = result.reason == "manipulation_timeout_continued"
            if timed_out:
                timed_out_steps.append(goal.name)
            self._event(logging.WARNING if timed_out else logging.INFO,
                        "MISSION_STEP_TIMEOUT_CONTINUED" if timed_out else "MISSION_STEP_COMPLETED", goal.name,
                        generation=generation, mission_step=index, skill_id=self._skill_id, reason=result.reason)
        result = LaViRATaskResult(generation, "reached",
                                 "mission_completed_with_timeouts" if timed_out_steps else "mission_completed", total_steps,
                                 self._skill_id, max(0, self._segment_id), self.navigation_mode)
        self._event(logging.INFO, "TASK_COMPLETED", "Mission sequence finished",
                    timed_out_steps=timed_out_steps, **asdict(result))
        return result
