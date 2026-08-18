#!/usr/bin/env python3
"""Bridge the upper-layer YOLOE visual servo into ControlGateway intents."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Mapping

import zmq

from gear_sonic.runtime.control_client import (
    ControlGatewayIntentClient,
    ControlGatewaySubscriber,
)
from gear_sonic.utils.inference.base_pose_sensor import SensorGatewayBasePoseCamera
from gear_sonic.utils.inference.base_pose_visual_servo import (
    RawServoRuntime,
    run_raw_servo_worker,
    validate_raw_servo_dependencies,
)


class GatewayRawServoAdapter:
    """Give RawServoRuntime explicit ControlGateway generations and typed output."""

    def __init__(
        self,
        config: Any,
        *,
        submit_intent: Callable[[str, Mapping[str, object]], None],
        logger: Callable[[str], None] = print,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.submit_intent = submit_intent
        self.logger = logger
        self.monotonic = monotonic
        self.gateway_generation = 0
        self._publish_enabled = False
        self._terminal_reported = True
        self.runtime = RawServoRuntime(
            config,
            publish=self._publish,
            logger=logger,
            monotonic=monotonic,
        )

    def _publish(self, message: str) -> None:
        if not self._publish_enabled:
            return
        payload = json.loads(message)
        velocity = payload.get("velocity")
        if not isinstance(velocity, Mapping):
            raise ValueError("YOLOE servo output has no velocity object")
        command = [float(velocity[name]) for name in ("vx", "vy", "wz")]
        self.submit_intent(
            "base_pose_velocity",
            {
                "generation": self.gateway_generation,
                "velocity": command,
                "action": str(payload.get("action", "visual_servo")),
                "motion_profile": "yoloe_servo",
                "camera_stream": str(payload.get("camera_stream", "")),
            },
        )

    def start(self, generation: int, *, now: float | None = None) -> bool:
        timestamp = self.monotonic() if now is None else float(now)
        if self.runtime.phase != "idle" or generation <= self.gateway_generation:
            return False
        self.gateway_generation = int(generation)
        # RawServoRuntime historically increments an internal keyboard generation.
        # Rebase it so worker events carry the ControlGateway generation verbatim.
        self.runtime.generation = self.gateway_generation - 1
        self._publish_enabled = True
        self._terminal_reported = False
        return self.runtime.handle_key("n", now=timestamp) == "started"

    def cancel(self, generation: int, reason: str, *, now: float | None = None) -> None:
        timestamp = self.monotonic() if now is None else float(now)
        self.gateway_generation = max(self.gateway_generation, int(generation))
        self._publish_enabled = False
        self.runtime.cancel(reason, timestamp)
        self.runtime.generation = self.gateway_generation
        self._terminal_reported = True

    def tick(self, *, now: float | None = None) -> None:
        timestamp = self.monotonic() if now is None else float(now)
        was_active = self.runtime.phase != "idle"
        self.runtime.poll_events()
        self.runtime.publish_due(timestamp)
        if was_active and self.runtime.phase == "idle" and not self._terminal_reported:
            reason = self.runtime.controller.terminal_reason or "visual_servo_finished"
            state = "reached" if reason == "aligned" else "failed"
            self.submit_intent(
                "base_pose_status",
                {
                    "generation": self.gateway_generation,
                    "state": state,
                    "reason": reason,
                },
            )
            self._terminal_reported = True
            self._publish_enabled = False

    def shutdown(self) -> None:
        self._publish_enabled = False
        self.runtime.shutdown()


def run_base_pose_yolo_agent(config: Any) -> None:
    """Run single-camera YOLOE alignment without owning the SONIC output socket."""

    validate_raw_servo_dependencies(config)
    context = zmq.Context.instance()
    intent = ControlGatewayIntentClient(
        config.control_gateway_intent_endpoint,
        source="base_pose_agent",
        context=context,
        ttl_ms=max(100, int(3000.0 / config.planner_hz)),
    )
    adapter = GatewayRawServoAdapter(
        config,
        submit_intent=lambda name, parameters: intent.send(name, parameters),
    )
    control = ControlGatewaySubscriber(
        config.control_gateway_endpoint,
        context=context,
        accepted_names={"start_base_pose", "cancel_navigation"},
    )
    camera = SensorGatewayBasePoseCamera(
        config.sensor_gateway_endpoint,
        camera_stream=config.camera_stream,
        depth_stream=config.depth_stream,
        require_depth=True,
        timeout_ms=config.camera_timeout_ms,
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        max_skew_ms=config.sensor_gateway_max_skew_ms,
    )
    worker = threading.Thread(
        target=run_raw_servo_worker,
        args=(
            config,
            adapter.runtime.requests,
            adapter.runtime.events,
            adapter.runtime.gate,
            adapter.runtime.stop_event,
        ),
        kwargs={
            "observation_events": adapter.runtime.observation_events,
            "diagnostics": adapter.runtime.diagnostics,
            "camera_factory": lambda: camera,
            "table_required": lambda: adapter.runtime.controller.table_required,
        },
        name="base-pose-yoloe",
        daemon=True,
    )
    worker.start()
    print(
        f"[BasePose/YOLOE] waiting for B; stream={config.camera_stream} "
        f"depth={config.depth_stream} task={config.task!r}"
    )
    try:
        while True:
            command = control.read_command()
            if command is not None:
                generation = int(command.parameters.get("generation", -1))
                if command.name == "start_base_pose":
                    adapter.start(generation)
                else:
                    adapter.cancel(generation, "operator_stop")
            adapter.tick()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        adapter.shutdown()
        worker.join(timeout=2.0)
        adapter.runtime.flush_diagnostics()
        control.close()
        intent.close()

