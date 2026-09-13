"""Record NaVILA velocities after the executor sends each Sonic packet."""

from dataclasses import replace
import time

from gear_sonic.runtime.protocol.array_message import unpack_array_message

from .recording import Recorder


class NavilaVelocityRecorder:
    def __init__(self, profile):
        self.recorder = Recorder(profile)
        self.enabled = (
            self.recorder.path is not None and self.recorder.config.get("navigation_backend") == "navila"
        )
        self._active = None
        self._last_key = None
        self._last_sample_s = -float("inf")

    def record_sent(self, decision, *, mode, now=None):
        if not self.enabled:
            return
        if decision.source == "navila":
            self._active = decision
        elif self._active is None:
            return
        owner = self._active
        now = time.monotonic() if now is None else now
        key = (
            decision.generation, decision.skill_id, decision.segment_id,
            mode, decision.source, decision.requested_velocity, decision.velocity, decision.reason,
        )
        # Preserve every change, plus a 2 Hz heartbeat during a steady action.
        if key != self._last_key or now - self._last_sample_s >= 0.5:
            packet = unpack_array_message(decision.message, expected_topic="planner").fields
            self.recorder.write(
                "navila_velocity",
                generation=owner.generation,
                skill_id=owner.skill_id,
                segment_id=owner.segment_id,
                component="planner_executor",
                sent_monotonic_s=now,
                mode=mode,
                source=decision.source,
                requested_velocity=list(decision.requested_velocity),
                velocity=list(decision.velocity),
                reason=decision.reason,
                executor_identity=dict(
                    generation=decision.generation,
                    skill_id=decision.skill_id,
                    segment_id=decision.segment_id,
                ),
                sonic=dict(
                    mode=int(packet["mode"][0]),
                    speed_m_s=float(packet["speed"][0]),
                    movement_direction=packet["movement"].tolist(),
                    facing_direction=packet["facing"].tolist(),
                ),
            )
            self._last_key, self._last_sample_s = key, now
        if decision.source != "navila":
            # Attribute the closing output to the preceding NaVILA segment,
            # including an operator stop that advances the generation.
            self._active = None
            self._last_key = None

    def record_shutdown(self, message):
        if self._active is not None:
            self.record_sent(
                replace(
                    self._active,
                    source="stop",
                    requested_velocity=(0.0, 0.0, 0.0),
                    velocity=(0.0, 0.0, 0.0),
                    reason="executor_shutdown",
                    message=message,
                ),
                mode="stop",
            )
