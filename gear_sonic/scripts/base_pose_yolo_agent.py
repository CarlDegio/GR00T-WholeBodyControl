#!/usr/bin/env python3
"""Bridge the upper-layer YOLOE visual servo into ControlGateway intents."""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Callable, Mapping

import zmq

from gear_sonic.runtime.control_client import (
    ControlGatewayIntentClient,
    ControlGatewaySubscriber,
)
from gear_sonic.utils.inference.base_pose_dual_visual_servo import (
    run_dual_raw_servo_worker,
)
from gear_sonic.utils.inference.base_pose_sensor import (
    SensorGatewayBasePoseCamera,
    SensorGatewayDualBasePoseCamera,
)
from gear_sonic.utils.inference.base_pose_visual_servo import (
    RawServoRuntime,
    run_raw_servo_worker,
    validate_raw_servo_dependencies,
)
from gear_sonic.utils.teleop.sonic_orientation_telemetry import (
    LatestOrientationTelemetry,
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
        orientation_provider: (
            Callable[[float], Mapping[str, Any] | None] | None
        ) = None,
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
            orientation_provider=orientation_provider,
        )

    def _publish(self, message: str) -> None:
        if not self._publish_enabled:
            return
        payload = json.loads(message)
        velocity = payload.get("velocity")
        if not isinstance(velocity, Mapping):
            raise ValueError("YOLOE servo output has no velocity object")
        command = [float(velocity[name]) for name in ("vx", "vy", "wz")]
        parameters: dict[str, object] = {
            "generation": self.gateway_generation,
            "velocity": command,
            "action": str(payload.get("action", "visual_servo")),
            "motion_profile": "yoloe_servo",
            "camera_stream": str(payload.get("camera_stream", "")),
        }
        viewer_overlay = payload.get("viewer_overlay")
        if viewer_overlay is not None:
            if not isinstance(viewer_overlay, Mapping):
                raise ValueError("YOLOE servo viewer_overlay must be an object")
            parameters["viewer_overlay"] = dict(viewer_overlay)
        self.submit_intent(
            "base_pose_velocity",
            parameters,
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


def raw_servo_worker_for_mode(mode: str) -> Callable[..., None]:
    if mode == "raw_yoloe_servo":
        return run_raw_servo_worker
    if mode == "dual_raw_yoloe_servo":
        return run_dual_raw_servo_worker
    raise ValueError(f"unsupported BasePose YOLOE mode: {mode}")


def run_base_pose_yolo_agent(config: Any) -> None:
    """Run single- or dual-camera YOLOE without owning the SONIC socket."""

    worker_target = raw_servo_worker_for_mode(config.mode)
    validate_raw_servo_dependencies(config)
    context = zmq.Context.instance()
    intent = ControlGatewayIntentClient(
        config.control_gateway_intent_endpoint,
        source="base_pose_agent",
        context=context,
        ttl_ms=max(100, int(3000.0 / config.planner_hz)),
        latest_only=True,
    )
    orientation_socket = None
    orientation_provider = None
    if config.raw_orientation_telemetry_source:
        orientation_socket = context.socket(zmq.SUB)
        orientation_socket.setsockopt(zmq.SUBSCRIBE, b"")
        orientation_socket.setsockopt(zmq.CONFLATE, 1)
        orientation_socket.setsockopt(zmq.LINGER, 0)
        orientation_socket.connect(config.raw_orientation_telemetry_source)
        latest_orientation = LatestOrientationTelemetry()
        last_warning_at = -math.inf

        def read_orientation(now: float) -> dict[str, float | None] | None:
            nonlocal last_warning_at
            assert orientation_socket is not None
            while True:
                try:
                    raw = orientation_socket.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    latest_orientation.update(raw)
                except ValueError as exc:
                    if now - last_warning_at >= 1.0:
                        print(
                            "[BasePose/YOLOE] ignored orientation telemetry: "
                            f"{exc}",
                            flush=True,
                        )
                        last_warning_at = now
            return latest_orientation.diagnostics(now)

        orientation_provider = read_orientation
    adapter = GatewayRawServoAdapter(
        config,
        submit_intent=lambda name, parameters: intent.send(name, parameters),
        orientation_provider=orientation_provider,
    )
    control = ControlGatewaySubscriber(
        config.control_gateway_endpoint,
        context=context,
        accepted_names={"start_base_pose", "cancel_navigation"},
    )
    dual_mode = config.mode == "dual_raw_yoloe_servo"
    if dual_mode:
        camera = SensorGatewayDualBasePoseCamera(
            config.sensor_gateway_endpoint,
            stream_depths={
                config.dual_head_camera_stream: config.dual_head_depth_stream,
                config.dual_chest_camera_stream: config.dual_chest_depth_stream,
            },
            timeout_ms=config.camera_timeout_ms,
            request_timeout_ms=config.sensor_gateway_request_timeout_ms,
            max_age_ms=config.sensor_gateway_max_age_ms,
            max_skew_ms=config.sensor_gateway_max_skew_ms,
            buffer_size=config.dual_rgbd_buffer_size,
            poll_hz=config.dual_rgbd_poll_hz,
        )
        stream_summary = (
            f"head={config.dual_head_camera_stream}/"
            f"{config.dual_head_depth_stream} "
            f"chest={config.dual_chest_camera_stream}/"
            f"{config.dual_chest_depth_stream}"
        )
    else:
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
        stream_summary = f"stream={config.camera_stream}/{config.depth_stream}"
    worker_kwargs: dict[str, Any] = {
        "observation_events": adapter.runtime.observation_events,
        "diagnostics": adapter.runtime.diagnostics,
        "camera_factory": lambda: camera,
        "table_required": lambda: adapter.runtime.controller.table_required,
    }
    if dual_mode:
        worker_kwargs["handoff_hold_event"] = adapter.runtime.handoff_hold_ready
        worker_kwargs["position_fallback_allowed"] = (
            lambda: adapter.runtime.controller.position_fallback_allowed
        )
    worker = threading.Thread(
        target=worker_target,
        args=(
            config,
            adapter.runtime.requests,
            adapter.runtime.events,
            adapter.runtime.gate,
            adapter.runtime.stop_event,
        ),
        kwargs=worker_kwargs,
        name="base-pose-dual-yoloe" if dual_mode else "base-pose-yoloe",
        daemon=True,
    )
    worker.start()
    print(
        f"[BasePose/YOLOE] waiting for B; mode={config.mode} "
        f"{stream_summary} task={config.task!r}"
    )
    try:
        while True:
            command = control.read_command()
            if command is not None:
                generation = int(command.parameters.get("generation", -1))
                if command.name == "start_base_pose":
                    begin_generation = getattr(camera, "begin_generation", None)
                    if callable(begin_generation):
                        begin_generation(generation)
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
        if orientation_socket is not None:
            orientation_socket.close(0)
