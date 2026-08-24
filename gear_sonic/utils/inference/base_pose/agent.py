#!/usr/bin/env python3
"""Run Gateway-controlled YOLOE BasePose with dual RGB-D cameras."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Any, Callable, Mapping

import zmq

from gear_sonic.camera.calibration import DEFAULT_CAMERA_INTRINSICS_PATH
from gear_sonic.runtime.profile import (
    load_component_config,
    parse_component_config,
)
from gear_sonic.runtime.gateway.control_client import (
    ControlGatewayIntentClient,
    ControlGatewaySubscriber,
)
from gear_sonic.runtime.inference_service import InferenceServiceContext
from gear_sonic.runtime.telemetry import BASE_POSE_TIMING_SEGMENTS
from gear_sonic.runtime.zmq_sockets import connect_subscriber
from gear_sonic.utils.inference.base_pose.dual_servo import (
    run_dual_raw_servo_worker,
)
from gear_sonic.utils.inference.base_pose.sensor import (
    SensorGatewayDualBasePoseCamera,
)
from gear_sonic.utils.inference.base_pose.servo import (
    RawServoRuntime,
    validate_raw_servo_dependencies,
)
from gear_sonic.utils.teleop.sonic_orientation_telemetry import (
    LatestOrientationTelemetry,
)

LOGGER = logging.getLogger("sonic.base_pose")


@dataclass
class BasePoseAgentConfig:
    task: str
    profile: str = ""
    overlay: tuple[str, ...] = ()
    target_prompt: str = "bluebasket"
    surface_prompt: str = "desk"
    planner_hz: float = 20.0
    final_stop_count: int = 3

    sensor_gateway_request_timeout_ms: int = 100
    sensor_gateway_max_age_ms: float = 1000.0
    sensor_gateway_max_skew_ms: float = 5.0

    camera_timeout_ms: int = 15000
    camera_intrinsics_path: str = str(DEFAULT_CAMERA_INTRINSICS_PATH)
    camera_pitch_deg: float = -38.0
    camera_roll_deg: float = 0.0
    camera_yaw_deg: float = 0.0
    camera_forward_offset_m: float = 0.0
    camera_lateral_offset_m: float = 0.0

    dual_head_camera_stream: str = "ego_view"
    dual_head_depth_stream: str = "camera/ego_view_depth"
    dual_chest_camera_stream: str = "chest_view"
    dual_chest_depth_stream: str = "camera/chest_view_depth"
    dual_chest_camera_pitch_deg: float = -3.0
    dual_chest_camera_roll_deg: float = 0.0
    dual_chest_camera_yaw_deg: float = 0.0
    dual_chest_camera_forward_offset_m: float = 0.0
    dual_chest_camera_lateral_offset_m: float = 0.0
    dual_match_tolerance_frames: int = 30
    dual_head_reacquire_frames: int = 1
    dual_head_release_missing_frames: int = 3
    dual_rgbd_buffer_size: int = 8
    dual_rgbd_poll_hz: float = 60.0

    output_root: str = "outputs/base_pose_adjustment"
    raw_yoloe_model_path: str = "tools/yoloe26m/weights/yoloe-26m-seg.pt"
    raw_yoloe_device: str = "0"
    raw_yoloe_confidence: float = 0.25
    raw_yoloe_imgsz: int = 640
    raw_head_target_distance_m: float = 1.00
    raw_chest_target_distance_m: float = 0.80
    raw_forward_tolerance_m: float = 0.10
    raw_lateral_tolerance_m: float = 0.10
    raw_min_linear_speed_m_s: float = 0.40
    raw_max_lateral_speed_m_s: float = 0.40
    raw_min_yaw_speed_rad_s: float = 0.10
    raw_yaw_tolerance_deg: float = 8.0
    raw_yaw_coarse_speed_rad_s: float = 0.30
    raw_yaw_trim_speed_rad_s: float = 0.20
    raw_forward_recenter_yaw_speed_rad_s: float = 0.30
    raw_horizontal_guard_fraction: float = 0.25
    raw_horizontal_recovery_fraction: float = 0.30
    raw_camera_stale_s: float = 0.4
    raw_max_run_s: float = 180.0
    raw_post_stop_sample_frames: int = 30
    raw_post_stop_deviation_frames: int = 10

    def __post_init__(self) -> None:
        self.target_prompt = str(self.target_prompt).strip()
        if not self.target_prompt:
            raise ValueError("target_prompt must be non-empty")
        self.surface_prompt = str(self.surface_prompt).strip()
        if not self.surface_prompt:
            raise ValueError("surface_prompt must be non-empty")
        for value, name in (
            (self.dual_match_tolerance_frames, "dual_match_tolerance_frames"),
            (self.dual_head_reacquire_frames, "dual_head_reacquire_frames"),
            (
                self.dual_head_release_missing_frames,
                "dual_head_release_missing_frames",
            ),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for value, name in (
            (self.raw_head_target_distance_m, "raw_head_target_distance_m"),
            (
                self.raw_chest_target_distance_m,
                "raw_chest_target_distance_m",
            ),
            (self.raw_forward_tolerance_m, "raw_forward_tolerance_m"),
            (self.raw_lateral_tolerance_m, "raw_lateral_tolerance_m"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.raw_post_stop_sample_frames < 0:
            raise ValueError("raw_post_stop_sample_frames must be non-negative")
        if not (
            self.raw_post_stop_sample_frames == 0
            or 0 < self.raw_post_stop_deviation_frames
            <= self.raw_post_stop_sample_frames
        ):
            raise ValueError(
                "raw_post_stop_deviation_frames must be in "
                "[1, raw_post_stop_sample_frames]"
            )


def load_base_pose_config(
    profile: str = "",
    overlays: tuple[str, ...] = (),
) -> BasePoseAgentConfig:
    return load_component_config(
        BasePoseAgentConfig,
        "base_pose",
        profile or None,
        overlays=overlays,
    )


def parse_base_pose_config(args: list[str] | None = None) -> BasePoseAgentConfig:
    return parse_component_config(BasePoseAgentConfig, "base_pose", args)


class GatewayRawServoAdapter:
    """Give RawServoRuntime explicit ControlGateway generations and typed output."""

    def __init__(
        self,
        config: Any,
        *,
        submit_intent: Callable[[str, Mapping[str, object]], None],
        logger: Callable[[str], None] = print,
        report_event: Callable[..., None] | None = None,
        report_metrics: Callable[..., None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        orientation_provider: (
            Callable[[float], Mapping[str, Any] | None] | None
        ) = None,
    ) -> None:
        self.submit_intent = submit_intent
        self.logger = logger
        self.report_event = report_event
        self.report_metrics = report_metrics
        self.monotonic = monotonic
        self._publish_enabled = False
        self._terminal_reported = True
        self.task_generation = 0
        self.skill_id = 0
        self.segment_id = 0
        self.runtime = RawServoRuntime(
            config,
            publish=self._publish,
            logger=self._log,
            monotonic=monotonic,
            metrics=report_metrics,
            orientation_provider=orientation_provider,
        )

    def _log(self, message: str) -> None:
        self.logger(message)
        if self.report_event is None or "[RawServo] WARNING" not in message:
            return
        code = (
            "DIAGNOSTIC_WRITE_FAILED"
            if "diagnostic" in message
            else "ORIENTATION_INVALID"
            if "orientation telemetry" in message
            else "RUNTIME_WARNING"
        )
        self.report_event(logging.WARNING, code, message)

    def _publish(self, payload: Mapping[str, Any]) -> None:
        if not self._publish_enabled:
            return
        velocity = payload["velocity"]
        command = [float(velocity[name]) for name in ("vx", "vy", "wz")]
        parameters: dict[str, object] = {
            "generation": self.task_generation,
            "skill_id": self.skill_id,
            "segment_id": self.segment_id,
            "velocity": command,
            "action": str(payload.get("action", "visual_servo")),
            "motion_profile": "yoloe_servo",
            "camera_stream": str(payload.get("camera_stream", "")),
        }
        viewer_overlay = payload.get("viewer_overlay")
        if viewer_overlay is not None:
            parameters["viewer_overlay"] = dict(viewer_overlay)
        self.submit_intent(
            "base_pose_velocity",
            parameters,
        )

    def _retire_invocation(self, generation: int) -> None:
        """Return a completed invocation to the public generation axis."""

        self.task_generation = int(generation)
        self.runtime.generation = self.task_generation
        self.skill_id = 0
        self.segment_id = 0

    def _reject_start(
        self,
        *,
        generation: int,
        skill_id: int,
        segment_id: int,
        reason: str,
        clear_gateway_owner: bool,
    ) -> None:
        message = (
            "[BasePose/YOLOE] start rejected "
            f"generation={generation} skill_id={skill_id} "
            f"segment_id={segment_id} reason={reason}"
        )
        self.logger(message)
        if self.report_event is not None:
            self.report_event(
                logging.WARNING,
                "START_REJECTED",
                message,
                generation=generation,
                skill_id=skill_id,
                segment_id=segment_id,
                reason=reason,
            )
        if not clear_gateway_owner:
            return
        status: dict[str, object] = {
            "generation": generation,
            "state": "failed",
            "reason": reason,
        }
        if skill_id:
            status.update(skill_id=skill_id, segment_id=segment_id)
        self.submit_intent("base_pose_status", status)

    def start(
        self,
        generation: int,
        *,
        skill_id: int = 0,
        segment_id: int = 0,
        target: object | None = None,
        surface: object | None = None,
        reference_bbox: object | None = None,
        now: float | None = None,
    ) -> bool:
        timestamp = self.monotonic() if now is None else float(now)
        requested_generation = int(generation)
        requested_skill_id = int(skill_id)
        runtime_generation = (
            requested_generation
            if requested_skill_id == 0
            else requested_generation * 1_000_000 + requested_skill_id
        )
        requested_segment_id = int(segment_id)
        if self.runtime.phase != "idle":
            self._reject_start(
                generation=requested_generation,
                skill_id=requested_skill_id,
                segment_id=requested_segment_id,
                reason=f"runtime_busy:{self.runtime.phase}",
                clear_gateway_owner=False,
            )
            return False
        if runtime_generation <= self.runtime.generation:
            self._reject_start(
                generation=requested_generation,
                skill_id=requested_skill_id,
                segment_id=requested_segment_id,
                reason=(
                    "stale_generation:"
                    f"{runtime_generation}<={self.runtime.generation}"
                ),
                clear_gateway_owner=True,
            )
            return False
        if target is not None:
            normalized_target = str(target).strip()
            if not normalized_target:
                self._reject_start(
                    generation=requested_generation,
                    skill_id=requested_skill_id,
                    segment_id=requested_segment_id,
                    reason="empty_target",
                    clear_gateway_owner=True,
                )
                return False
            self.runtime.config.target_prompt = normalized_target
        if surface is not None:
            normalized_surface = str(surface).strip()
            if not normalized_surface:
                self._reject_start(
                    generation=requested_generation,
                    skill_id=requested_skill_id,
                    segment_id=requested_segment_id,
                    reason="empty_surface",
                    clear_gateway_owner=True,
                )
                return False
            self.runtime.config.surface_prompt = normalized_surface
        self.task_generation = requested_generation
        self.skill_id = requested_skill_id
        self.segment_id = requested_segment_id
        self.runtime.reference_bbox = reference_bbox
        self._publish_enabled = True
        self._terminal_reported = False
        started = self.runtime.start(runtime_generation, now=timestamp)
        if not started:
            self._publish_enabled = False
            self._terminal_reported = True
            self._reject_start(
                generation=requested_generation,
                skill_id=requested_skill_id,
                segment_id=requested_segment_id,
                reason="runtime_start_failed",
                clear_gateway_owner=self.runtime.phase == "idle",
            )
            if self.runtime.phase == "idle":
                self._retire_invocation(requested_generation)
        elif self.report_metrics is not None:
            self.report_metrics({}, activate=True)
        return started

    def cancel(
        self,
        generation: int,
        reason: str,
        *,
        now: float | None = None,
    ) -> bool:
        timestamp = self.monotonic() if now is None else float(now)
        requested_generation = int(generation)
        if requested_generation < self.task_generation:
            self.logger(
                "[BasePose/YOLOE] ignored stale cancel "
                f"generation={requested_generation} "
                f"active_generation={self.task_generation}"
            )
            return False
        self._publish_enabled = False
        if self.runtime.phase == "idle":
            # Global navigation cancellation is also delivered while BasePose
            # is inactive.  Keep generations aligned without reporting a fake
            # operator stop or perturbing the runtime generation twice.
            self._retire_invocation(requested_generation)
            self._terminal_reported = True
            return False
        self.runtime.cancel(
            reason,
            timestamp,
            generation=max(
                self.runtime.generation,
                (
                    requested_generation
                    if self.skill_id == 0
                    else requested_generation * 1_000_000 + self.skill_id
                ),
            ),
        )
        self._retire_invocation(requested_generation)
        self._terminal_reported = True
        return True

    def tick(self, *, now: float | None = None) -> None:
        timestamp = self.monotonic() if now is None else float(now)
        was_active = self.runtime.phase != "idle"
        was_soft_stale = self.runtime.soft_stale
        self.runtime.poll_events()
        self.runtime.publish_due(timestamp)
        if self.report_event is not None:
            if not was_soft_stale and self.runtime.soft_stale:
                self.report_event(
                    logging.WARNING,
                    "CAMERA_STALE",
                    "BasePose camera stream is stale; holding position",
                    generation=self.task_generation,
                    skill_id=self.skill_id,
                )
            elif (
                was_soft_stale
                and not self.runtime.soft_stale
                and self.runtime.phase != "idle"
            ):
                self.report_event(
                    logging.INFO,
                    "CAMERA_RECOVERED",
                    "BasePose camera stream recovered",
                    generation=self.task_generation,
                    skill_id=self.skill_id,
                )
        if (
            was_active
            and self.runtime.phase == "idle"
            and not self._terminal_reported
        ):
            reason = self.runtime.controller.terminal_reason or "visual_servo_finished"
            state = "reached" if reason == "aligned" else "failed"
            status: dict[str, object] = {
                "generation": self.task_generation,
                "state": state,
                "reason": reason,
            }
            if self.skill_id:
                status.update(
                    skill_id=self.skill_id,
                    segment_id=self.segment_id,
                )
            self.submit_intent("base_pose_status", status)
            if self.skill_id:
                # The worker uses a composite identity only while an LA ALIGN
                # invocation is active. Collapse back to the task generation
                # after draining that invocation so a later standalone B start
                # remains monotonic on the public generation axis.
                self._retire_invocation(self.task_generation)
            self._terminal_reported = True
            self._publish_enabled = False

    def shutdown(self) -> None:
        self._publish_enabled = False
        self.runtime.shutdown()


def run_base_pose_yolo_agent(config: Any) -> None:
    """Run dual-camera YOLOE without owning the SONIC socket."""

    validate_raw_servo_dependencies(config)
    service = InferenceServiceContext("base_pose", config)
    profile = service.profile
    context = zmq.Context.instance()

    def report_event(
        level: int,
        code: str,
        message: str,
        *,
        write_log: bool = True,
        **fields: object,
    ) -> None:
        service.event(
            level,
            code,
            message,
            queued=True,
            write_log=write_log,
            **fields,
        )

    flush_events = service.flush_events

    def log_runtime(message: str) -> None:
        level = (
            logging.ERROR
            if "FAILURE" in message
            else logging.WARNING
            if "WARNING" in message
            else logging.INFO
        )
        LOGGER.log(level, message)

    def send_metrics(values: Mapping[str, float], *, activate: bool = False) -> None:
        service.publish_metrics(
            values,
            allowed_names=BASE_POSE_TIMING_SEGMENTS,
            activate=activate,
        )

    intent = ControlGatewayIntentClient(
        profile.endpoint_uri("control_gateway_intent"),
        source="base_pose_agent",
        context=context,
        ttl_ms=max(100, int(3000.0 / config.planner_hz)),
        latest_only=True,
    )
    orientation_socket = None
    orientation_provider = None
    orientation_endpoint = profile.endpoint_uri("orientation_telemetry")
    if orientation_endpoint:
        orientation_socket = connect_subscriber(
            context, orientation_endpoint, conflate=True, linger_ms=0,
        )
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
                        report_event(
                            logging.WARNING,
                            "ORIENTATION_INVALID",
                            "Ignored invalid orientation telemetry",
                            error=str(exc),
                        )
                        last_warning_at = now
            return latest_orientation.diagnostics(now)

        orientation_provider = read_orientation
    adapter = GatewayRawServoAdapter(
        config,
        submit_intent=lambda name, parameters: intent.send(name, parameters),
        logger=log_runtime,
        report_event=lambda level, code, message, **fields: report_event(
            level,
            code,
            message,
            write_log=False,
            **fields,
        ),
        report_metrics=send_metrics,
        orientation_provider=orientation_provider,
    )
    control = ControlGatewaySubscriber(
        profile.endpoint_uri("control_gateway_dispatch"),
        context=context,
        accepted_names={"start_base_pose", "cancel_navigation"},
    )
    camera = SensorGatewayDualBasePoseCamera(
        profile.endpoint_uri("sensor_gateway_metadata"),
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
    worker_kwargs: dict[str, Any] = {
        "observation_events": adapter.runtime.observation_events,
        "diagnostics": adapter.runtime.diagnostics,
        "camera_factory": lambda: camera,
        "table_required": lambda: adapter.runtime.controller.table_required,
        "position_fallback_allowed": (
            lambda: adapter.runtime.controller.position_fallback_allowed
        ),
    }
    worker = threading.Thread(
        target=run_dual_raw_servo_worker,
        args=(
            config,
            adapter.runtime.requests,
            adapter.runtime.events,
            adapter.runtime.gate,
            adapter.runtime.stop_event,
        ),
        kwargs=worker_kwargs,
        name="base-pose-dual-yoloe",
        daemon=True,
    )
    worker.start()
    report_event(
        logging.INFO,
        "READY",
        "BasePose is ready and waiting for a start command",
        streams=stream_summary,
        task=config.task,
    )
    flush_events()
    try:
        while True:
            command = control.read_command()
            if command is not None:
                generation = int(command.parameters.get("generation", -1))
                if command.name == "start_base_pose":
                    camera.begin_generation(generation)
                    adapter.start(
                        generation,
                        skill_id=int(command.parameters.get("skill_id", 0)),
                        segment_id=int(command.parameters.get("segment_id", 0)),
                        target=command.parameters.get("target"),
                        surface=command.parameters.get("surface"),
                        reference_bbox=command.parameters.get("reference_bbox"),
                    )
                else:
                    reason = str(
                        command.parameters.get("reason") or "operator_stop"
                    )
                    adapter.cancel(generation, reason)
            adapter.tick()
            flush_events()
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
        report_event(logging.INFO, "STOPPED", "BasePose stopped")
        service.close()


def main(config: BasePoseAgentConfig) -> None:
    run_base_pose_yolo_agent(config)


if __name__ == "__main__":
    main(parse_base_pose_config())
