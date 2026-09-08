"""Gateway-owned trial clock and bounded NaVILA actions, independent of models."""

from __future__ import annotations

import math
import time

from gear_sonic.runtime.protocol.navigation import build_navigation_message

from .recording import Recorder


class ExperimentSupervisor:
    def __init__(self, gateway, *, clock=time.monotonic):
        self.gateway = gateway
        self.recorder = Recorder(gateway.profile)
        self.config = self.recorder.config
        self.clock = clock
        self.active = None
        self.vla_identity = None
        self.vla_deadline = None
        self.step = None
        self.last_step_publish = 0.0

    def start(self, generation):
        if not self.config:
            return
        if self.active:
            old_generation = self.active[0]
            self.recorder.write("intervention", generation=old_generation, command="restart_with_n")
            self.finish(old_generation, "cancelled", "restarted_with_n")
        self.active = (generation, self.clock() + self.config["task"]["total_timeout_s"])
        self.vla_identity = self.vla_deadline = self.step = None
        self.recorder.write(
            "trial_start",
            generation=generation,
            task_id=self.config["task_id"],
            method=self.config["method"],
            condition=self.config["condition"],
        )
        from .snapshots import record_pose_async

        record_pose_async(self.gateway.profile, self.recorder, generation, "task_start")

    def finish(self, generation, state, reason, **fields):
        if self.active and self.active[0] == generation:
            self.recorder.write("trial_end", generation=generation, state=state, reason=reason, **fields)
            self.active = self.step = self.vla_deadline = None

    def confirm_success(self, generation, *, completion_time_s, completed_wall_time_ns):
        """Persist the operator's physical success and time for result aggregation."""
        if not self.active or self.active[0] != generation:
            return
        timing = dict(completion_time_s=completion_time_s, completed_wall_time_ns=completed_wall_time_ns)
        values = dict(success=True, progress=len(self.config["task"]["milestones"]), **timing)
        if self.config["entry_stage"] == "navigation":
            values["nav_success"] = True
        self.recorder.write(
            "annotation", generation=generation, source="operator_console", values=values,
        )
        self.finish(generation, "reached", "operator_success", **timing)

    def abort(self, state, reason):
        if not self.active:
            return
        generation = self.active[0]
        self.finish(generation, state, reason)
        action = self.gateway.navigation.handle_key(" ", now=self.clock(), cancel_reason=reason)
        self.gateway.publish_navigation_action(action, source="experiment_deadline")

    def first_action(self, parameters):
        identity = (int(parameters["generation"]), int(parameters["skill_id"]))
        if not self.active or identity[0] != self.active[0] or self.vla_identity == identity:
            return
        nav = self.gateway.navigation
        if nav.mode != "lavira_manipulate" or identity[1] != nav.skill_id:
            return
        self.vla_identity = identity
        duration = self.config["task"][
            "vla_timeout_s" if self.config["gates"]["completion"] else "fixed_manipulation_s"
        ]
        started = (
            float(parameters["action_monotonic_ns"]) / 1e9 if "action_monotonic_ns" in parameters else self.clock()
        )
        if (
            not math.isfinite(started)
            or started > self.clock()
            or started < self.active[1] - self.config["task"]["total_timeout_s"]
        ):
            self.vla_identity = None
            raise ValueError("VLA action timestamp is outside the active trial")
        self.vla_deadline = started + duration

    def begin_step(self, parameters):
        nav = self.gateway.navigation
        generation, skill, segment = (int(parameters[k]) for k in ("generation", "skill_id", "segment_id"))
        if not self.active or generation != self.active[0] or nav.mode != "lavira_pending":
            raise ValueError("NaVILA step has no current pending navigation owner")
        if (skill, segment) <= (nav.skill_id, nav.segment_id):
            raise ValueError("Stale NaVILA step")
        velocity = tuple(float(x) for x in parameters["velocity"])
        duration = float(parameters["duration_s"])
        if len(velocity) != 3 or not all(math.isfinite(v) for v in (*velocity, duration)):
            raise ValueError("Invalid NaVILA velocity")
        vx, vy, wz = velocity
        if not 0 < duration <= 30 or abs(vx) > 0.3 or vy != 0 or abs(wz) > 0.3 or (vx and wz):
            raise ValueError("NaVILA step exceeds experiment motion envelope")
        nav.mode, nav.skill_id, nav.segment_id = "lavira_nav", skill, segment
        self.step = (generation, skill, segment, velocity, self.clock() + duration)
        self.last_step_publish = -math.inf

    def tick(self):
        if not self.active:
            return
        now = self.clock()
        if now >= self.active[1]:
            self.abort("failed", "total_timeout")
            return
        if self.vla_deadline is not None and now >= self.vla_deadline:
            disabled = not self.config["gates"]["completion"]
            if disabled:
                self.recorder.write(
                    "gate",
                    generation=self.active[0],
                    gate="completion",
                    enabled=False,
                    applied="allow",
                    reason="fixed_duration",
                    va_recommendation=None,
                    check_id="fixed_duration",
                    skill_id=self.vla_identity[1],
                )
            self.abort(
                "completion_candidate" if disabled else "failed", "fixed_duration" if disabled else "vla_timeout"
            )
            return
        if self.step is not None:
            generation, skill, segment, velocity, deadline = self.step
            if now >= deadline:
                self.gateway._send_navigation_stop(generation=generation, skill_id=skill, segment_id=segment)
                self.gateway.navigation.mode = "lavira_pending"
                self.gateway.dispatch_navigation_event(
                    "navigation_status",
                    dict(
                        generation=generation,
                        skill_id=skill,
                        segment_id=segment,
                        state="reached",
                        reason="navila_step_completed",
                    ),
                )
                self.step = None
            elif now - self.last_step_publish >= 0.05:
                self.gateway.navigation_pub.send_string(
                    build_navigation_message(
                        mode="manual_velocity",
                        generation=generation,
                        skill_id=skill,
                        segment_id=segment,
                        velocity=velocity,
                        source="navila",
                    )
                )
                self.last_step_publish = now
