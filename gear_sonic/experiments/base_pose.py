"""Head-only perception for BasePose and the single-object geometric baseline."""

from __future__ import annotations

import math
import queue
import threading
import time

from gear_sonic.camera.calibration import load_camera_intrinsics
from gear_sonic.runtime.gateway.control_client import ControlGatewayIntentClient, ControlGatewaySubscriber
from gear_sonic.runtime.inference_service import InferenceServiceContext
from gear_sonic.utils.inference.base_pose.sensor import SensorGatewayDualBasePoseCamera
from gear_sonic.utils.inference.base_pose.servo import (
    RawServoCalibration,
    RawServoEvent,
    YoloePersistentTracker,
    _observation,
    _prompts_match,
    _publish_worker_event,
    _resolve_target,
)


class NullDiagnostics:
    def __init__(self, logger=print):
        self.logger = logger

    def submit_frame(self, *args, **kwargs):
        pass

    def submit_decision(self, *args, **kwargs):
        pass

    def submit_camera_images(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


def head_calibration(config):
    head = load_camera_intrinsics(config.camera_intrinsics_path)[config.dual_head_camera_stream]
    return RawServoCalibration(
        width=head.width,
        height=head.height,
        fx=head.fx,
        fy=head.fy,
        cx=head.cx,
        cy=head.cy,
        camera_pitch_deg=config.camera_pitch_deg,
        camera_roll_deg=config.camera_roll_deg,
        camera_yaw_deg=config.camera_yaw_deg,
        camera_forward_offset_m=config.camera_forward_offset_m,
        camera_lateral_offset_m=config.camera_lateral_offset_m,
    )


def tracker_for(config, target, yaw):
    tracker = YoloePersistentTracker(
        config.raw_yoloe_model_path,
        confidence=config.raw_yoloe_confidence,
        imgsz=config.raw_yoloe_imgsz,
        device=config.raw_yoloe_device,
        yaw_align_target_prompt=yaw,
    )
    tracker.start_all_text(target_prompt=target)
    return tracker


def run_head_worker(
    config,
    requests,
    events,
    gate,
    stop_event,
    *,
    observation_events=None,
    diagnostics=None,
    camera_factory,
    **unused,
):
    camera = camera_factory()
    try:
        while not stop_event.is_set():
            try:
                generation = requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if generation is None:
                return
            try:
                calibration = head_calibration(config)
                target, yaw = config.target_prompt, config.yaw_align_target_prompt
                tracker = tracker_for(config, target, yaw)
                camera.begin_generation(generation)
                details = {
                    "attempt_id": 1,
                    "live_stream": config.dual_head_camera_stream,
                    "control_source_stream": config.dual_head_camera_stream,
                }
                _publish_worker_event(
                    events, observation_events, None, RawServoEvent(generation, "detecting", details=details)
                )
                initialized = False
                target_id = yaw_id = -1
                while gate.is_active(generation) and not stop_event.is_set():
                    try:
                        snapshot = camera.capture_stream(config.dual_head_camera_stream, timeout_ms=200)
                    except Exception:
                        continue  # Existing runtime stale-camera watchdog holds motion.
                    instances = tracker.track(snapshot.rgb)
                    target_instance, _, _ = _resolve_target(instances, target_id)
                    yaw_instance = target_instance
                    if not _prompts_match(target, yaw):
                        yaw_instance, _, _ = _resolve_target(instances, yaw_id, 1)
                    if target_instance is None:
                        continue
                    target_id = target_instance.track_id
                    if yaw_instance is not None:
                        yaw_id = yaw_instance.track_id
                    try:
                        observation = _observation(
                            snapshot,
                            target_instance,
                            yaw_instance,
                            calibration,
                            exclude_target_from_yaw_align_edge=not _prompts_match(target, yaw),
                        )
                    except (ValueError, RuntimeError):
                        continue
                    _publish_worker_event(
                        events,
                        observation_events,
                        None,
                        RawServoEvent(
                            generation,
                            "observation" if initialized else "initialized",
                            observation=observation,
                            details=details,
                        ),
                    )
                    initialized = True
            except Exception as exc:
                _publish_worker_event(
                    events, observation_events, None, RawServoEvent(generation, "error", error=str(exc), hard=True)
                )
    finally:
        camera.close()


class GeometricController:
    def __init__(self, config, started):
        self.config, self.started = config, started
        self.phase = "coarse"
        self.stable = 0
        self.last_stamp = None
        self.reason = None
        self.error_m = self.error_deg = None

    def update(self, forward, right, stamp, now):
        c = self.config
        zero = (0.0, 0.0, 0.0)
        if self.reason:
            return zero
        if now - self.started >= c["timeout_s"]:
            self.reason = "geometric_timeout"
            return zero
        if forward is None or right is None or not all(math.isfinite(x) for x in (forward, right)) or forward <= 0:
            self.stable = 0
            return zero
        bearing = math.atan2(-right, forward)
        self.error_m = forward - c["distance_m"]
        self.error_deg = math.degrees(bearing)
        new_frame = stamp != self.last_stamp
        self.last_stamp = stamp
        angle_ok = abs(self.error_deg) <= c["angle_tolerance_deg"] + 1e-12
        distance_ok = abs(self.error_m) <= c["longitudinal_tolerance_m"] + 1e-12
        if self.phase == "coarse":
            if not angle_ok:
                return (0.0, 0.0, math.copysign(c["coarse_wz"], bearing))
            self.phase = "approach"
        if self.phase == "approach":
            if self.error_m > c["longitudinal_tolerance_m"] + 1e-12:
                return (c["vx"], 0.0, 0.0)
            if self.error_m < -c["longitudinal_tolerance_m"] - 1e-12:
                return (-c["vx"], 0.0, 0.0)
            self.phase = "fine"
        if not angle_ok:
            self.stable = 0
            return (0.0, 0.0, math.copysign(c["fine_wz"], bearing))
        if not distance_ok:
            self.phase, self.stable = "approach", 0
            return zero
        if new_frame:
            self.stable += 1
        if self.stable >= c["stable_frames"]:
            self.reason = "aligned"
        return zero


def run_geometric_service(config, profile):
    service = InferenceServiceContext("base_pose", config)
    geometry = dict(profile.components["experiment"]["geometric"])
    control = ControlGatewaySubscriber(
        profile.endpoint_uri("control_gateway_dispatch"), accepted_names={"start_base_pose", "cancel_navigation"}
    )
    intent = ControlGatewayIntentClient(profile.endpoint_uri("control_gateway_intent"), source="base_pose_agent")
    camera = SensorGatewayDualBasePoseCamera(
        profile.endpoint_uri("sensor_gateway_metadata"),
        stream_depths={config.dual_head_camera_stream: config.dual_head_depth_stream},
        allow_single=True,
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        max_skew_ms=config.sensor_gateway_max_skew_ms,
    )
    lock = threading.Lock()
    active = None
    sample = (None, None, None, 0.0)
    worker_error = None
    stopped = threading.Event()

    def perceive(identity, target):
        nonlocal sample, worker_error
        try:
            calibration = head_calibration(config)
            tracker = tracker_for(config, target, target)
            track_id = -1
            while not stopped.is_set() and active == identity:
                try:
                    snap = camera.capture_stream(config.dual_head_camera_stream, timeout_ms=200)
                    instance, _, _ = _resolve_target(tracker.track(snap.rgb), track_id)
                    if instance is None:
                        value = (None, None, snap.timestamp, time.monotonic())
                    else:
                        track_id = instance.track_id
                        obs = _observation(snap, instance, None, calibration, include_yaw_align_geometry=False)
                        value = (obs.target.forward_m, obs.target.right_m, snap.timestamp, time.monotonic())
                except (TimeoutError, ValueError, RuntimeError):
                    continue
                with lock:
                    if active == identity:
                        sample = value
        except Exception as exc:
            with lock:
                if active == identity:
                    worker_error = str(exc)

    def publish(identity, velocity, action):
        intent.send(
            "base_pose_velocity",
            dict(
                generation=identity[0],
                skill_id=identity[1],
                segment_id=identity[2],
                velocity=list(velocity),
                action=action,
                motion_profile="yoloe_servo",
            ),
        )

    try:
        while True:
            command = control.read_command()
            if command is not None:
                p = command.parameters
                if command.name == "cancel_navigation":
                    active = None
                elif active is None:
                    active = tuple(int(p.get(k, 0)) for k in ("generation", "skill_id", "segment_id"))
                    sample, worker_error = (None, None, None, 0.0), None
                    controller = GeometricController(geometry, time.monotonic())
                    camera.begin_generation(active[0])
                    publish(active, (0.0, 0.0, 0.0), "hold")
                    # The operator's b key carries only the control identity.
                    # Use the same configured target as the experiment agent.
                    target = str(p.get("target") or geometry["target"]).strip()
                    threading.Thread(target=perceive, args=(active, target), daemon=True).start()
            if active is not None:
                now = time.monotonic()
                with lock:
                    forward, right, stamp, received = sample
                    error = worker_error
                if now - received > config.raw_camera_stale_s:
                    forward = right = None
                velocity = controller.update(forward, right, stamp, now)
                reason = error or controller.reason
                # Model startup and missing observations can take seconds.
                # A zero hold must not acquire the gateway's 350 ms motion lease.
                action = "stop" if reason else ("visual_servo" if any(velocity) else "hold")
                publish(active, (0.0, 0.0, 0.0) if reason else velocity, action)
                if reason:
                    intent.send(
                        "base_pose_status",
                        dict(
                            generation=active[0],
                            skill_id=active[1],
                            segment_id=active[2],
                            state="reached" if reason == "aligned" else "failed",
                            reason=reason,
                            longitudinal_error_m=controller.error_m,
                            bearing_error_deg=controller.error_deg,
                        ),
                    )
                    active = None
            time.sleep(1.0 / config.planner_hz)
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        if active:
            publish(active, (0.0, 0.0, 0.0), "stop")
        control.close()
        intent.close()
        camera.close()
        service.close()
