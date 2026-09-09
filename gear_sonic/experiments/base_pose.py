"""Head-only BasePose and camera failover for the single-object geometric baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import queue
import threading
import time

from gear_sonic.camera.calibration import load_camera_intrinsics
from gear_sonic.runtime.gateway.control_client import ControlGatewayIntentClient, ControlGatewaySubscriber
from gear_sonic.runtime.inference_service import InferenceServiceContext
from gear_sonic.utils.inference.base_pose.dual_servo import dual_calibrations_from_config
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
        self.camera_stream = None

    def select_camera(self, stream):
        if stream != self.camera_stream:
            self.camera_stream = stream
            self.phase, self.stable, self.last_stamp = "coarse", 0, None
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


@dataclass(frozen=True)
class GeometricSample:
    forward: float | None = None
    right: float | None = None
    stamp: tuple[str, float] | None = None
    received: float = 0.0
    camera_stream: str | None = None
    both_lost_frames: int = 0
    reason: str | None = None
    views: dict = field(default_factory=dict)


class GeometricPerception:
    """Prefer head initially, then keep the selected camera until its target is lost."""

    def __init__(self, config, camera, target, *, both_lost_frames=20):
        self.config, self.camera = config, camera
        self.streams = (config.dual_head_camera_stream, config.dual_chest_camera_stream)
        self.calibrations = dual_calibrations_from_config(config)
        self.tracker = tracker_for(config, target, target)
        self.camera_stream = self.tracking_stream = None
        self.track_id = -1
        self.last_stamps = {}
        self.both_lost_frames = 0
        self.loss_limit = int(both_lost_frames)

    def _observe(self, stream):
        try:
            snap = self.camera.capture_stream(stream, timeout_ms=200)
        except (TimeoutError, ValueError, RuntimeError) as exc:
            return None, dict(fresh=False, target_visible=None, usable=False, error=str(exc))
        stamp = float(snap.timestamp)
        if not math.isfinite(stamp) or stamp <= self.last_stamps.get(stream, -math.inf):
            return None, dict(fresh=False, target_visible=None, usable=False, reason="no_new_frame")
        self.last_stamps[stream] = stamp
        # One model is shared, but BoT-SORT IDs must never cross camera views.
        if self.tracking_stream != stream:
            self.tracker.reset_tracking()
            self.track_id, self.tracking_stream = -1, stream
        instance, _, _ = _resolve_target(self.tracker.track(snap.rgb), self.track_id)
        details = dict(fresh=True, timestamp=stamp, target_visible=instance is not None, usable=False)
        if instance is None:
            return None, details
        self.track_id = instance.track_id
        details.update(track_id=instance.track_id, bbox=list(instance.bbox_xyxy), confidence=instance.confidence)
        try:
            obs = _observation(
                snap, instance, None, self.calibrations[stream], include_yaw_align_geometry=False
            )
            forward, right = obs.target.forward_m, obs.target.right_m
            if forward <= 0 or not all(math.isfinite(x) for x in (forward, right)):
                raise ValueError("invalid target position")
        except (ValueError, RuntimeError) as exc:
            details["error"] = str(exc)
            return None, details
        details.update(usable=True, forward_m=forward, right_m=right)
        return (forward, right, stamp), details

    def step(self, *, hold=lambda: None):
        preferred = self.camera_stream or self.streams[0]
        order = (preferred, next(stream for stream in self.streams if stream != preferred))
        views = {}
        for stream in order:
            coordinates, views[stream] = self._observe(stream)
            if coordinates is not None:
                self.camera_stream = stream
                self.both_lost_frames = 0
                forward, right, stamp = coordinates
                return GeometricSample(
                    forward, right, (stream, stamp), time.monotonic(), stream, views=views
                )
            # Stop using the old velocity while checking the alternative camera.
            if stream == preferred:
                hold()
        # Count a round only when BOTH cameras supplied distinct new frames.
        # Camera timeouts and repeated cached images cannot satisfy the 20-frame exit.
        if all(view["fresh"] for view in views.values()):
            self.both_lost_frames += 1
        return GeometricSample(
            received=time.monotonic(), both_lost_frames=self.both_lost_frames, views=views,
            reason="geometric_target_lost" if self.both_lost_frames >= self.loss_limit else None,
        )


def run_geometric_service(config, profile):
    service = InferenceServiceContext("base_pose", config)
    geometry = dict(profile.components["experiment"]["geometric"])
    control = ControlGatewaySubscriber(
        profile.endpoint_uri("control_gateway_dispatch"), accepted_names={"start_base_pose", "cancel_navigation"}
    )
    intent = ControlGatewayIntentClient(profile.endpoint_uri("control_gateway_intent"), source="base_pose_agent")
    camera = SensorGatewayDualBasePoseCamera(
        profile.endpoint_uri("sensor_gateway_metadata"),
        stream_depths={
            config.dual_head_camera_stream: config.dual_head_depth_stream,
            config.dual_chest_camera_stream: config.dual_chest_depth_stream,
        },
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        max_skew_ms=config.sensor_gateway_max_skew_ms,
    )
    lock = threading.Lock()
    active = None
    sample = GeometricSample()
    worker_error = None
    stopped = threading.Event()

    def report(identity, code, message, **fields):
        service.event(
            logging.INFO, code, message, queued=True,
            generation=identity[0], skill_id=identity[1], segment_id=identity[2], **fields,
        )

    def update_sample(identity, value):
        nonlocal sample
        with lock:
            if active == identity:
                sample = value

    def perceive(identity, target):
        nonlocal worker_error
        try:
            perception = GeometricPerception(
                config, camera, target, both_lost_frames=geometry.get("both_lost_frames", 20)
            )
            if stopped.is_set() or active != identity:
                return
            report(identity, "GEOMETRIC_PERCEPTION_READY", "Geometric YOLOE ready", target=target)
            previous_stream, previous_loss = None, 0
            while not stopped.is_set() and active == identity:
                value = perception.step(hold=lambda: update_sample(identity, GeometricSample()))
                if active != identity:
                    return
                update_sample(identity, value)
                if value.camera_stream and (value.camera_stream != previous_stream or previous_loss):
                    report(
                        identity, "GEOMETRIC_CAMERA_SELECTED", "Geometric target acquired",
                        previous_camera=previous_stream, camera_stream=value.camera_stream, views=value.views,
                    )
                    previous_stream = value.camera_stream
                if value.both_lost_frames != previous_loss:
                    report(
                        identity, "GEOMETRIC_TARGET_LOST", "Geometric dual-camera loss count updated",
                        both_lost_frames=value.both_lost_frames, views=value.views, reason=value.reason,
                    )
                previous_loss = value.both_lost_frames
                if value.reason:
                    return
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
                    sample, worker_error = GeometricSample(), None
                    controller = GeometricController(geometry, time.monotonic())
                    motion_logged = False
                    camera.begin_generation(active[0])
                    publish(active, (0.0, 0.0, 0.0), "hold")
                    # The operator's b key carries only the control identity.
                    # Use the same configured target as the experiment agent.
                    target = str(p.get("target") or geometry["target"]).strip()
                    report(active, "GEOMETRIC_STARTED", "Geometric alignment started", target=target)
                    threading.Thread(target=perceive, args=(active, target), daemon=True).start()
            if active is not None:
                now = time.monotonic()
                with lock:
                    value = sample
                    error = worker_error
                forward, right = value.forward, value.right
                if now - value.received > config.raw_camera_stale_s:
                    forward = right = None
                if forward is not None:
                    controller.select_camera(value.camera_stream)
                velocity = controller.update(forward, right, value.stamp, now)
                reason = error or value.reason or controller.reason
                # Model startup and missing observations can take seconds.
                # A zero hold must not acquire the gateway's 350 ms motion lease.
                action = "stop" if reason else ("visual_servo" if any(velocity) else "hold")
                publish(active, (0.0, 0.0, 0.0) if reason else velocity, action)
                if action == "visual_servo" and not motion_logged:
                    motion_logged = True
                    report(
                        active, "GEOMETRIC_MOTION_STARTED", "First geometric motion command sent",
                        camera_stream=value.camera_stream, velocity=list(velocity), views=value.views,
                    )
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
                            camera_stream=controller.camera_stream,
                            both_lost_frames=value.both_lost_frames,
                        ),
                    )
                    report(
                        active, "GEOMETRIC_FINISHED", "Geometric alignment stopped",
                        reason=reason, camera_stream=controller.camera_stream,
                        both_lost_frames=value.both_lost_frames,
                    )
                    active = None
            service.flush_events()
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
