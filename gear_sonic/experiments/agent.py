"""Experiment branches around the existing agent's navigation and handoffs."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
import uuid

import cv2

from gear_sonic.utils.inference.lavira.agent import AgentHistoryEntry, LaViRAAgent, LaViRAAgentError, _same_target

from .navila import NavilaClient
from .recording import Recorder, read_events


class RecordedCamera:
    """Associate VA images with their capture time, without archiving LA scans."""

    def __init__(self, camera):
        self.camera = camera
        self.observations = {}

    def __getattr__(self, name):
        method = getattr(self.camera, name)
        if name not in {"capture_rgb", "capture_aligned_rgbd", "capture_camera_aligned_rgbd"}:
            return method

        def capture(*args, **kwargs):
            value = method(*args, **kwargs)
            image = value.rgb_bgr if hasattr(value, "rgb_bgr") else value
            stream = kwargs.get("camera_stream", "chest_view")
            rgb_key = f"camera/{stream}"
            depth_key = kwargs.get("depth_stream", f"camera/{stream}_depth")
            if name == "capture_aligned_rgbd":
                stamp = getattr(self.camera, "_last_timestamp_ns", None)
            elif name == "capture_camera_aligned_rgbd":
                stamp = getattr(self.camera, "_last_rgbd_timestamp_by_stream", {}).get(depth_key)
            else:
                stamp = getattr(self.camera, "_last_rgb_timestamp_by_stream", {}).get(rgb_key)
            self.observations[id(image)] = dict(
                captured_wall_ns=time.time_ns(), source_timestamp_ns=stamp, stream=stream
            )
            if len(self.observations) > 64:
                self.observations.pop(next(iter(self.observations)))
            return value

        return capture


class RecordedVisualClient:
    def __init__(self, client, agent):
        self.client, self.agent = client, agent
        self.context = threading.local()
        self.check_events = []

    def check_ids(self, generation, skill, stage):
        return [
            e["event_id"]
            for e in self.check_events
            if e.get("generation") == generation and e["skill_id"] == skill and e["stage"] == stage
        ]

    def __getattr__(self, key):
        return getattr(self.client, key)

    def _call(self, kind, **kwargs):
        generation, skill = self.agent.generation, self.agent._skill_id
        if "mission" in kwargs:
            kwargs["mission"] = self.agent.experiment["task"]["vla_prompt"]
        started = time.time_ns()
        fields = dict(
            kind=kind,
            skill_id=skill,
            stage=kwargs.get("skill") or getattr(self.context, "stage", "ALIGN"),
            camera=getattr(self.context, "camera", None),
            request_started_ns=started,
        )
        fields["round"] = 1 + sum(
            e.get("generation") == generation
            and e["skill_id"] == skill
            and e["kind"] == kind
            and e["camera"] == fields["camera"]
            for e in self.check_events
        )
        image = kwargs.get("image_bgr")
        if image is not None:
            fields["observation"] = self.agent.camera.observations.get(
                id(image), {"missing_reason": "capture_timestamp_unavailable"}
            )
            if self.agent.recorder.path is not None:
                directory = self.agent.recorder.path.parent / "checks" / self.agent.recorder.trial_id(generation)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{uuid.uuid4().hex}.png"
                if cv2.imwrite(str(path), image):
                    fields["image"] = str(path)
                else:
                    fields["image_missing_reason"] = "image_write_failed"
        try:
            result = getattr(self.client, kind)(**kwargs)
        except Exception as exc:
            event = self.agent.recorder.write(
                "check",
                generation=generation,
                **fields,
                response_received_ns=time.time_ns(),
                error=str(exc),
                result=None,
            )
            if event:
                self.check_events.append(event)
            raise
        event = self.agent.recorder.write(
            "check", generation=generation, **fields, response_received_ns=time.time_ns(), result=result
        )
        if event:
            self.check_events.append(event)
        return result

    def grounding(self, **kwargs):
        return self._call("grounding", **kwargs)

    def alignment_grounding(self, **kwargs):
        return self._call("alignment_grounding", **kwargs)

    def postcheck(self, **kwargs):
        return self._call("postcheck", **kwargs)


class ExperimentAgent(LaViRAAgent):
    def __init__(self, *, experiment, **kwargs):
        super().__init__(**kwargs)
        self.experiment = experiment
        self.recorder = Recorder(SimpleNamespace(components={"experiment": experiment}))
        self.vla_trained_prompt = experiment["task"]["vla_trained_prompt"]
        self.head_only = experiment["alignment"] == "head"
        self.camera = RecordedCamera(self.camera)
        self.client = RecordedVisualClient(self.client, self)
        self.generation = -1
        self.navila = NavilaClient(experiment["navila"]) if experiment["navigation_backend"] == "navila" else None
        self._perturbation_requested = False

    def run(self, generation):
        self.generation = generation
        self._perturbation_requested = False
        if self.navila is not None:
            self.navila.reset()
            self._check_cancelled(generation)
        return super().run(generation)

    def _reset_task_context(self):
        super()._reset_task_context()
        entry = self.experiment["entry_stage"]
        self.forced_skill = {"alignment": "ALIGN", "manipulation": "MANIPULATE"}.get(entry)
        if entry != "navigation":
            self._global_target_navigation_ready = True
            self._store_transition(
                "ALIGN" if entry == "manipulation" else "MOVE_TO",
                {
                    "transition": "READY_TO_MANIPULATE" if entry == "manipulation" else "READY_TO_ALIGN",
                    "status": "SATISFIED",
                    "visual_evidence": "Configured nearfield entry; navigation is not applicable.",
                },
            )

    def _plan_next_step(self, generation, step, todo):
        skill = self.forced_skill
        self.forced_skill = None
        if skill is None and self.navila is not None:
            return_to_nav = (
                self._latest_transition and self._latest_transition.get("transition") == "RETURN_TO_NAVIGATION"
            )
            skill = "MOVE_TO" if return_to_nav or not self._global_target_navigation_ready else "ALIGN"
        if skill is None and self.experiment["entry_stage"] != "navigation":
            skill = "ALIGN"
        if skill is None:
            return super()._plan_next_step(generation, step, todo)
        return dict(
            decision="EXECUTE",
            skill=skill,
            skill_args={"view_direction": "front", "target": self.global_target} if skill == "MOVE_TO" else {},
            updated_todo_list=f"- [ ] {skill} {self.global_target}",
            global_target=self.global_target,
            progress_analysis="Fixed experiment transition",
            reasoning="Configured experiment branch",
            expected_postcondition=self.manipulation_prompt,
        )

    def _face_scan_direction(self, generation, skill_id, direction):
        if self.experiment["entry_stage"] != "navigation" or self.navila is not None:
            return None
        return super()._face_scan_direction(generation, skill_id, direction)

    def _alignment_views(self):
        if self.head_only:
            return (("head", self.alignment_head_camera_stream),)
        return super()._alignment_views()

    def _ground(self, **kwargs):
        self.client.context.stage, self.client.context.camera = "NAV", kwargs.get("camera_label", "chest")
        return super()._ground(**kwargs)

    def _alignment_ground(self, **kwargs):
        self.client.context.stage = "ALIGN"
        self.client.context.camera = "head" if self.head_only else "chest"
        return super()._alignment_ground(**kwargs)

    def _single_view_alignment_check(self, **kwargs):
        self.client.context.stage, self.client.context.camera = "ALIGN", kwargs["camera_label"]
        return super()._single_view_alignment_check(**kwargs)

    def _postcheck(self, *args, **kwargs):
        self._perturb("completion")
        self.client.context.stage, self.client.context.camera = "MANIPULATE", "head"
        result = super()._postcheck(*args, **kwargs)
        enabled = self.experiment["gates"]["completion"]
        applied = (
            "allow"
            if enabled and result["transition"] == "TASK_COMPLETE"
            else "block"
            if enabled
            else "wait_fixed_duration"
        )
        if self.cancelled(kwargs["generation"]):
            applied = "ignored_late_result"
        self.recorder.write(
            "gate",
            generation=kwargs["generation"],
            gate="completion",
            enabled=enabled,
            skill_id=kwargs["skill_id"],
            window_id=kwargs.get("window_id"),
            va_recommendation=result["transition"] == "TASK_COMPLETE",
            applied=applied,
            check_ids=self.client.check_ids(kwargs["generation"], kwargs["skill_id"], "MANIPULATE"),
            result=result,
        )
        return result

    def _perturb(self, gate):
        condition = self.experiment["condition"].get("perturbation", {})
        if (
            self._perturbation_requested
            or not condition.get("enabled")
            or self.experiment.get("gate_under_test") != gate
        ):
            return
        self._perturbation_requested = True
        self.recorder.write("perturbation_cue", generation=self.generation, gate=gate)
        self._event(
            logging.WARNING,
            "PERTURBATION_CUE",
            "Apply the configured disturbance, then enter :perturb",
            generation=self.generation,
        )
        # This explicit barrier is only used by the prescribed perturbation
        # protocol; the gateway's independent total clock remains running.
        while True:
            self._check_cancelled(self.generation)
            if any(
                e["type"] == "perturbation" and e.get("trial_id") == self.recorder.trial_id(self.generation)
                for e in read_events(self.recorder.path)
            ):
                return
            self.sleep(0.1)

    def _nav_handoff(self, **kwargs):
        result, view = super()._nav_handoff(**kwargs)
        recommended = self._global_target_navigation_ready
        enabled = self.experiment["gates"]["nav"]
        # A completed intermediate waypoint is not the navigation stage's
        # terminal candidate, even when visual readiness is bypassed.
        candidate = kwargs["navigation_completed"] and _same_target(kwargs["target"], self.global_target)
        allow = recommended or (not enabled and candidate)
        self.recorder.write(
            "gate",
            generation=self.generation,
            gate="nav",
            enabled=enabled,
            skill_id=kwargs["skill_id"],
            va_recommendation=recommended,
            applied="allow" if allow else "block",
            terminal_candidate=candidate,
            result=result,
            check_ids=self.client.check_ids(self.generation, kwargs["skill_id"], "NAV"),
        )
        if allow:
            self._global_target_navigation_ready = True
            direct = self.experiment["alignment"] == "none"
            if direct or not enabled or self.navila is not None:
                self.forced_skill = "MANIPULATE" if direct else "ALIGN"
            self._store_transition(
                "MOVE_TO", {**result, "transition": "READY_TO_MANIPULATE" if direct else "READY_TO_ALIGN"}
            )
        return result, view

    def _before_nav_handoff(self, generation, skill_id, target):
        if self.experiment.get("runtime_profile_path"):
            from gear_sonic.runtime.profile import load_runtime_profile

            from .snapshots import record_pose_async

            record_pose_async(
                load_runtime_profile(self.experiment["runtime_profile_path"]),
                self.recorder,
                generation,
                "navigation_end",
                skill_id,
            )
        if _same_target(target, self.global_target):
            self._perturb("nav")

    def _align_handoff(self, **kwargs):
        self._perturb("align")
        result = super()._align_handoff(**kwargs)
        recommended = result["transition"] == "READY_TO_MANIPULATE"
        enabled = self.experiment["gates"]["align"]
        geometric = self.experiment["alignment"] == "geometric"
        # A failed role lookup has not run the controller to its terminal state.
        controller_ended = kwargs["controller_state"] != "target_not_found"
        # The geometric baseline records VA evidence without gating entry to VLA.
        allow = recommended or (not enabled and controller_ended) or geometric
        self.recorder.write(
            "gate",
            generation=self.generation,
            gate="align",
            enabled=enabled and not geometric,
            skill_id=kwargs["skill_id"],
            va_recommendation=recommended,
            applied="allow" if allow else "block",
            controller_aligned=kwargs["controller_aligned"],
            controller_state=kwargs["controller_state"],
            result=result,
            check_ids=self.client.check_ids(self.generation, kwargs["skill_id"], "ALIGN"),
        )
        if allow:
            if (
                not enabled
                or geometric
                or self.navila is not None
                or self.experiment["entry_stage"] != "navigation"
            ):
                self.forced_skill = "MANIPULATE"
            self._store_transition("ALIGN", {**result, "transition": "READY_TO_MANIPULATE"})
        elif self.experiment["entry_stage"] != "navigation":
            self._store_transition("ALIGN", {**result, "transition": "RETRY_ALIGN"})
        return result

    def _align(self, generation, skill_id, args, strategic_goal):
        if self.experiment["alignment"] != "geometric":
            return super()._align(generation, skill_id, args, strategic_goal)
        target = self.experiment["geometric"]["target"]
        segment = self._next_segment()
        self.submit_intent(
            "start_base_pose",
            dict(
                generation=generation,
                skill_id=skill_id,
                segment_id=segment,
                target=target,
                yaw_align_target=target,
            ),
        )
        status = self._wait(generation, skill_id, segment)
        reason = status.get("reason")
        if reason not in {"aligned", "geometric_timeout", "geometric_target_lost"}:
            raise LaViRAAgentError(f"geometric_controller_failed:{reason}")
        post = self._align_handoff(
            generation=generation,
            skill_id=skill_id,
            strategic_goal=strategic_goal,
            controller_aligned=reason == "aligned",
            controller_state=str(reason),
            target=target,
            yaw_align_target=target,
        )
        return AgentHistoryEntry(skill_id, "ALIGN", target, str(reason), post["status"], post["visual_evidence"])

    def _manipulate(self, generation, skill_id, expected, strategic_goal):
        if self.experiment["gates"]["completion"]:
            return super()._manipulate(generation, skill_id, expected, strategic_goal)
        self._start_manipulation(generation, skill_id)
        window = 0
        while True:
            self._check_cancelled(generation)
            failure = self.poll_failure(generation, skill_id)
            if failure is not None:
                raise LaViRAAgentError(f"manipulate_safety:{failure.get('reason')}")
            window += 1
            finished = threading.Event()

            def check(window_id=window):
                try:
                    self._postcheck(
                        "MANIPULATE",
                        expected,
                        generation=generation,
                        skill_id=skill_id,
                        strategic_goal=strategic_goal,
                        strategic_stop=True,
                        window_id=window_id,
                    )
                except Exception:
                    pass  # Recorded by the visual client; disabled gate cannot decide control.
                finally:
                    finished.set()

            self.sleep(self.manipulation_window_seconds)
            self._check_cancelled(generation)
            threading.Thread(target=check, daemon=True, name="experiment-shadow-va").start()
            while not finished.wait(0.05):
                self._check_cancelled(generation)
            # Gateway ends this trial at a fixed duration measured from the
            # first published action, including when a VA call never returns.

    def _move_to(self, generation, skill_id, args, strategic_goal):
        if self.navila is None:
            return super()._move_to(generation, skill_id, args, strategic_goal)
        while self.navila.sequence < self.experiment["navila"]["max_actions"]:
            self._check_cancelled(generation)
            action = self.navila.action(
                self.camera.capture_rgb(camera_stream=self.experiment["navila"]["camera"]), self.mission
            )
            self._check_cancelled(generation)
            if action is None:
                self._before_nav_handoff(generation, skill_id, self.global_target)
                views, errors = {}, {}
                for label, stream in (("head", "ego_view"), ("chest", "chest_view")):
                    try:
                        snapshot = self.camera.capture_camera_aligned_rgbd(
                            camera_stream=stream, depth_stream=f"camera/{stream}_depth"
                        )
                        grounded = self._ground(
                            generation=generation,
                            skill_id=skill_id,
                            target=self.global_target,
                            direction="front",
                            image=snapshot.rgb_bgr,
                            strategic_goal=strategic_goal,
                            strategic_stop=True,
                            camera_label=label,
                        )
                        views[label] = (grounded, snapshot)
                    except Exception as exc:
                        errors[label] = str(exc)
                post, _ = self._nav_handoff(
                    generation=generation,
                    skill_id=skill_id,
                    target=self.global_target,
                    navigation_completed=True,
                    camera_views=views,
                    camera_errors=errors,
                )
                return AgentHistoryEntry(
                    skill_id, "MOVE_TO", self.global_target, "reached", post["status"], post["visual_evidence"]
                )
            velocity, duration = action
            segment = self._next_segment()
            self.submit_intent(
                "experiment_nav_step",
                dict(
                    generation=generation,
                    skill_id=skill_id,
                    segment_id=segment,
                    velocity=list(velocity),
                    duration_s=duration,
                ),
            )
            self._wait(generation, skill_id, segment)
        raise LaViRAAgentError("navila_action_budget_exceeded")
