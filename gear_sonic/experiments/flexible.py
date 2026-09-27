"""One flexible LA loop with experiment evidence and per-operation baselines."""

from gear_sonic.utils.inference.lavira.agent import AgentHistoryEntry
from gear_sonic.utils.inference.lavira.flexible import FlexibleMissionAgent

from .agent import ExperimentAgent


class FlexibleExperimentAgent(FlexibleMissionAgent, ExperimentAgent):
    """Keep planning and skill switching identical; replace only ALIGN execution."""

    def _request_language_action(self, **kwargs):
        result = super()._request_language_action(
            **kwargs, alignment_mode=self.experiment["alignment"],
        )
        if result["skill"] == "ALIGN":
            # Every operation gets its own deadline, including cloud VA waits.
            self.vla_task_parameters.setdefault("timeout_seconds", self.manipulation_timeout_seconds)
        return result

    def _align(self, generation, skill_id, args, strategic_goal):
        mode = self.experiment["alignment"]
        if mode in {"dual", "unilm_nav"}:
            return super()._align(generation, skill_id, args, strategic_goal)
        self._check_cancelled(generation)
        if mode == "none":
            # ALIGN selects a skill in the shared LA protocol; this ablation
            # performs no alignment controller, grounding, or alignment VA call.
            evidence = "Alignment skipped by direct-VLA baseline; no alignment success measured."
            self._store_transition("ALIGN", {
                "status": "UNKNOWN", "transition": "READY_TO_MANIPULATE",
                "visual_evidence": evidence, "alignment_skipped": True,
            })
            self.recorder.write("gate", generation=generation, skill_id=skill_id,
                                gate="align", enabled=False, applied="allow",
                                va_recommendation=None, controller_aligned=False,
                                controller_state="alignment_skipped", reason="direct_vla")
            entry = AgentHistoryEntry(skill_id, "ALIGN", self.global_target,
                                      "alignment_skipped", "UNKNOWN", evidence)
        else:
            entry = super()._align(generation, skill_id, args, strategic_goal)
        # Track a fresh preparation for this operation, including the deliberate
        # geometric timeout/loss fallback. Its actual outcome stays in evidence.
        # A new NAV/ALIGN invalidates it; it can be consumed only once.
        operation = self._operation
        operation.alignment_skill_id = skill_id
        operation.alignment_segment_id = self._segment_id
        operation.aligned_pose = self._fastlio_pose()
        operation.aligned_yaw = self._sonic_yaw()
        operation.aligned_at = self.monotonic()
        return entry

    def _start_manipulation(self, generation, skill_id):
        operation = self._check_operation_alignment(generation)
        spec = self.skills[self._active_vla_skill]
        self.recorder.write(
            "vla_operation", generation=generation, skill_id=skill_id,
            operation_id=operation.operation_id, alignment_skill_id=operation.alignment_skill_id,
            logical_skill=self._active_vla_skill, backend=spec["backend"],
            vla_prompt=self.manipulation_prompt, vla_trained_prompt=self.vla_trained_prompt,
            geometry_target=spec.get("geometry_target"), alignment=self.experiment["alignment"],
        )
        if self.experiment["alignment"] in {"dual", "unilm_nav"}:
            return super()._start_manipulation(generation, skill_id)
        # The baseline gateway protocol intentionally has no BasePose success
        # ticket (direct VLA, or geometric timeout/loss). Keep the operation's
        # single-use identity locally, without fabricating a controller ticket.
        self._operation = None
        operation.consumed = True
        try:
            return super()._start_manipulation(generation, skill_id)
        finally:
            self._operation = operation
