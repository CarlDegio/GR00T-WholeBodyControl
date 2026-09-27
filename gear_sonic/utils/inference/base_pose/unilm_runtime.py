"""Gateway-owned UniLM runtime shared by agent and operator-key entrypoints."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np
import zmq

from gear_sonic.runtime.gateway.control_client import ControlGatewayIntentClient, ControlGatewaySubscriber
from gear_sonic.runtime.gateway.sensor_client import SensorGatewayClient
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest, TimestampBasis
from gear_sonic.runtime.inference_service import InferenceServiceContext
from gear_sonic.runtime.profile import load_component_config
from gear_sonic.runtime.zmq_sockets import connect_subscriber
from gear_sonic.utils.inference.base_pose.sensor import _decode_rgbd_snapshot
from gear_sonic.utils.inference.base_pose.unilm import (
    Observation,
    PoseTracker,
    UniLMClient,
    UniLMConfig,
    UniLMError,
    camera_calibration,
    pose_from_odometry,
    robot_geometry,
)
from gear_sonic.utils.teleop.sonic_orientation_telemetry import decode_orientation_telemetry


def identity(parameters) -> tuple[int, int, int]:
    return tuple(int(parameters.get(k, 0)) for k in ("generation", "skill_id", "segment_id"))


class ObservationMemory:
    """Keep each reached NavDP segment's own frames until the task resets."""

    def __init__(self, size=5):
        self.size = size
        self.active = None
        self.frames = deque(maxlen=size)
        self.frozen = {}

    def start(self, key, seed=None):
        self.active = key
        self.frames.clear()
        if seed is not None:
            self.frames.append(seed)

    def observe(self, observation):
        if self.active is not None and (
            not self.frames or observation.frame.timestamp > self.frames[-1].frame.timestamp
        ):
            self.frames.append(observation)

    def finish(self, key, reached):
        if key != self.active:
            return
        if reached:
            self.frozen[key] = tuple(self.frames)
        self.active = None

    def history_through(self, source_key):
        """Combine saved segments up to the handoff, never accepting an unknown source."""
        if source_key not in self.frozen:
            return ()
        frames = {}
        for key, observations in self.frozen.items():
            if key[0] == source_key[0]:
                for observation in observations:
                    frames.setdefault(observation.frame.timestamp, observation)
            if key == source_key:
                break
        return tuple(frames[timestamp] for timestamp in sorted(frames))

    def clear(self):
        self.active = None
        self.frames.clear()
        self.frozen.clear()


class CameraMonitor:
    """One sensor thread owns its socket; motion tick never waits for RGB/API."""

    def __init__(self, profile, base, config, imu_to_base):
        self.profile, self.base, self.config = profile, base, config
        self.camera = camera_calibration(config, base)
        self.imu_to_base = imu_to_base
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.pose = None
        self.pose_received = 0.0
        self.observations = queue.Queue(maxsize=20)
        self.last_error = "awaiting_sensor_data"
        self.thread = threading.Thread(target=self._run, name=f"unilm-{config.camera}-odometry", daemon=True)

    def current(self):
        with self.lock:
            if self.pose is None or time.monotonic() - self.pose_received > self.config.odometry_timeout_s:
                raise UniLMError(f"odometry_unavailable:{self.last_error}")
            return self.pose, self.pose_received

    def _run(self):
        client = SensorGatewayClient(
            self.profile.endpoint_uri("sensor_gateway_metadata"),
            request_timeout_ms=self.base.sensor_gateway_request_timeout_ms,
        )
        next_image = 0.0
        last_image = None
        rgb = f"camera/{self.camera.camera_stream}"
        depth = self.camera.depth_stream
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                try:
                    odom = client.read_snapshot(
                        SnapshotRequest(("ros/odometry",), self.config.odometry_timeout_s * 1000, 0), retries=0
                    )
                    metadata = odom.snapshot.frames["ros/odometry"]
                    if metadata.attributes.get("child_frame_id") != "body":
                        raise UniLMError("expected FAST-LIO body (Livox IMU) odometry frame")
                    pose = pose_from_odometry(odom.arrays["ros/odometry"], self.imu_to_base)
                    with self.lock:
                        self.pose, self.pose_received = pose, metadata.metadata.timestamp_ns * 1e-9
                    if started >= next_image:
                        next_image = started + 1 / self.config.sample_hz
                        snapshot = client.read_snapshot(
                            SnapshotRequest(
                                (rgb, depth),
                                self.base.sensor_gateway_max_age_ms,
                                self.base.sensor_gateway_max_skew_ms,
                                timestamp_basis=TimestampBasis.SOURCE,
                            ),
                            retries=0,
                        )
                        frame = _decode_rgbd_snapshot(
                            snapshot,
                            camera_stream=self.camera.camera_stream,
                            rgb_stream=rgb,
                            depth_stream=depth,
                            require_depth=True,
                        )
                        if last_image is not None and frame.timestamp <= last_image:
                            continue
                        paired = client.read_snapshot(
                            SnapshotRequest(
                                ("ros/odometry",),
                                self.base.sensor_gateway_max_age_ms,
                                self.config.pose_skew_s * 1000,
                                anchor_timestamp_ns=int(frame.timestamp * 1e9),
                                timestamp_basis=TimestampBasis.SOURCE,
                            ),
                            retries=0,
                        )
                        paired_meta = paired.snapshot.frames["ros/odometry"]
                        if abs(paired_meta.source_timestamp_ns * 1e-9 - frame.timestamp) > self.config.pose_skew_s:
                            raise UniLMError("camera_odometry_timestamp_skew")
                        observed_pose = pose_from_odometry(paired.arrays["ros/odometry"], self.imu_to_base)
                        observation = Observation(frame, observed_pose)
                        last_image = frame.timestamp
                        if self.observations.full():
                            self.observations.get_nowait()
                        self.observations.put_nowait(observation)
                except Exception as exc:
                    with self.lock:
                        self.last_error = str(exc)
                self.stop.wait(max(0.005, 0.05 - (time.monotonic() - started)))
        finally:
            client.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


class UniLMRuntime:
    """Single-threaded command ownership, with cancellable inference workers."""

    def __init__(self, config, base, *, send, infer, current_pose, geometry, report):
        self.config, self.base, self.send = config, base, send
        self.camera = camera_calibration(config, base)
        self.infer, self.current_pose, self.geometry, self.report = infer, current_pose, geometry, report
        self.memory = ObservationMemory(config.history_size)
        self.latest = None
        self.active = None
        self.cancel = threading.Event()
        self.results = queue.SimpleQueue()
        self.tracker = None
        self.collecting = None
        self.orientation = None
        self.seen_generation = -1
        self.seen_starts = set()
        self.directory = None
        self.motion_log = None
        self.last_tracking = None

    def velocity(self, velocity=(0.0, 0.0, 0.0), action="hold"):
        if self.active is not None:
            self.send(
                "base_pose_velocity",
                {
                    **self.active,
                    "velocity": list(velocity),
                    "action": action,
                    "motion_profile": "yoloe_servo",
                    "camera_stream": self.camera.camera_stream,
                },
            )

    def terminal(self, state, reason):
        if self.active is None:
            return
        self.cancel.set()
        self.velocity(action="stop")
        result = {
            **self.active,
            "state": state,
            "reason": reason,
            "backend": "unilm_nav",
            "elapsed_s": time.monotonic() - self.started,
        }
        if self.tracker is not None:
            result["goal_world"] = asdict(self.tracker.goal)
        if self.last_tracking is not None:
            result["last_tracking"] = self.last_tracking
        if self.directory is not None:
            result["artifacts"] = str(self.directory)
            try:
                (self.directory / "result.json").write_text(json.dumps(result, indent=2))
            except OSError as exc:
                result["log_error"] = str(exc)
        self.send("base_pose_status", result)
        self.report("UNILM_COMPLETED" if state == "reached" else "UNILM_FAILED", reason, **result)
        if self.motion_log:
            self.motion_log.close()
        self.motion_log = None
        self.active = self.tracker = self.collecting = None

    def command(self, name, parameters, task):
        key = identity(parameters)
        if key[0] < self.seen_generation:
            return
        if key[0] > self.seen_generation:
            self.cancel.set()
            if self.active:
                self.terminal("stopped", "generation_changed")
            self.memory.clear()
            self.latest = None
            self.seen_starts.clear()
            self.seen_generation = key[0]
        if name in {"start_navigation", "cancel_navigation"}:
            self.terminal("stopped", "operator_cancelled")
            self.memory.clear()
        elif name == "navigation_goal":
            seed = self.latest
            if (
                seed is not None
                and time.time() - seed.frame.timestamp > self.base.sensor_gateway_max_age_ms / 1000
            ):
                seed = None
            self.memory.start(key, seed)
        elif name == "navigation_status":
            if parameters.get("state") in {"reached", "failed", "stopped"}:
                self.memory.finish(key, parameters.get("state") == "reached")
        elif name == "start_base_pose":
            if self.active is not None or key in self.seen_starts:
                return
            self.seen_starts.add(key)
            self.active = dict(zip(("generation", "skill_id", "segment_id"), key))
            self.task = str(parameters.get("task", task)).strip()
            self.cancel = threading.Event()
            self.last_tracking = None
            self.started = time.monotonic()
            self.deadline = self.started + self.config.timeout_s
            self.directory = Path(self.config.output_root) / f"{time.time_ns()}_g{key[0]}_s{key[1]}"
            self.velocity()
            try:
                self.directory.mkdir(parents=True, exist_ok=False)
                self.motion_log = (self.directory / "motion.jsonl").open("w")
                (self.directory / "config.json").write_text(
                    json.dumps(
                        {
                            "unilm": asdict(self.config),
                            "task": self.task,
                            **asdict(self.camera),
                            "start_parameters": parameters,
                            "source": parameters.get("observation_source", "current"),
                        },
                        indent=2,
                    )
                )
            except OSError as exc:
                self.terminal("failed", f"artifact_write_failed:{exc}")
                return
            if not self.task:
                self.terminal("failed", "manipulation_task_missing")
                return
            source = parameters.get("observation_source", "current")
            if source == "navigation":
                try:
                    source_key = (
                        key[0],
                        int(parameters["source_nav_skill_id"]),
                        int(parameters["source_nav_segment_id"]),
                    )
                except (KeyError, ValueError, TypeError):
                    self.terminal("failed", "invalid_navigation_source_identity")
                    return
                frames = self.memory.history_through(source_key)
                if not frames:
                    self.terminal("failed", "navigation_history_missing")
                elif len(frames) < self.config.history_size:
                    self.terminal("failed", "navigation_history_incomplete")
                else:
                    self._infer(frames)
            elif source == "current":
                self.collecting = []
                self.collect_after = time.time()
            else:
                self.terminal("failed", "invalid_observation_source")

    def observe(self, observation):
        if self.latest is not None and observation.frame.timestamp <= self.latest.frame.timestamp:
            return
        self.latest = observation
        self.memory.observe(observation)
        if self.active is not None and self.collecting is not None:
            if observation.frame.timestamp <= self.collect_after:
                return
            if self.collecting and observation.frame.timestamp <= self.collecting[-1].frame.timestamp:
                return
            self.collecting.append(observation)
            if len(self.collecting) == self.config.history_size:
                frames = tuple(self.collecting)
                self.collecting = None
                self._infer(frames)

    def _infer(self, frames):
        key, cancel, directory, deadline, task = (
            identity(self.active),
            self.cancel,
            self.directory,
            self.deadline,
            self.task,
        )

        def work():
            try:
                for i, obs in enumerate(frames):
                    if cancel.is_set():
                        return
                    np.savez_compressed(
                        directory / f"observation_{i}.npz",
                        rgb=obs.frame.rgb,
                        depth=obs.frame.depth_raw,
                        pose=[obs.pose.x, obs.pose.y, obs.pose.yaw],
                        timestamp=obs.frame.timestamp,
                        intrinsics=[obs.frame.fx, obs.frame.fy, obs.frame.cx, obs.frame.cy],
                        depth_scale_m=obs.frame.depth_scale_m,
                    )
                goal, details = self.infer(
                    frames, task, self.geometry, self.base, deadline=deadline, cancel=cancel, directory=directory
                )
                self.results.put((key, cancel, goal, None))
            except Exception as exc:
                self.results.put((key, cancel, None, str(exc)))

        threading.Thread(target=work, name="unilm-inference", daemon=True).start()

    def tick(self):
        while not self.results.empty():
            key, cancel, goal, error = self.results.get()
            if self.active is None or key != identity(self.active) or cancel is not self.cancel or cancel.is_set():
                continue
            if error:
                self.terminal("failed", error)
            else:
                try:
                    pose, _ = self.current_pose()
                    if math.hypot(goal.x - pose.x, goal.y - pose.y) > self.config.max_goal_distance_m:
                        raise UniLMError("predicted_pose_outside_near_field")
                    self.tracker = PoseTracker(goal, self.config)
                except Exception as exc:
                    self.terminal("failed", str(exc))
        if self.active is None:
            return
        now = time.monotonic()
        if now >= self.deadline:
            self.terminal("failed", "unilm_timeout")
            return
        if self.tracker is None:
            self.velocity()
            return
        try:
            pose, received = self.current_pose()
            o = self.orientation
            if (
                o is None
                or o.heading_lag_rad is None
                or o.state_age_s is None
                or o.state_age_s + max(0, now - o.emitted_at_monotonic_s) > self.config.odometry_timeout_s
            ):
                raise UniLMError("sonic_orientation_stale_or_unavailable")
            if (
                not self.tracker.position_acquired
                and math.hypot(self.tracker.goal.x - pose.x, self.tracker.goal.y - pose.y)
                > self.config.max_goal_distance_m
            ):
                raise UniLMError("tracking_pose_outside_near_field")
            velocity, reached, diagnostics = self.tracker.update(pose, now, o.heading_lag_rad)
            self.last_tracking = {
                "t": now - self.started,
                "pose": asdict(pose),
                "velocity": velocity,
                **diagnostics,
            }
            self.motion_log.write(json.dumps(self.last_tracking) + "\n")
            if reached:
                self.terminal("reached", "aligned")
            else:
                self.velocity(velocity, "visual_servo")
        except Exception as exc:
            self.terminal("failed", str(exc))


def run(base):
    service = InferenceServiceContext("base_pose", base, enable_metrics=False)
    profile = service.profile
    config = load_component_config(UniLMConfig, "unilm_nav", base.profile or None, overlays=base.overlay)
    geometry, imu_to_base = robot_geometry(config.urdf_path, config.camera_height_m)
    client = UniLMClient(config)
    context = zmq.Context()
    sender = ControlGatewayIntentClient(
        profile.endpoint_uri("control_gateway_intent"), source="base_pose_agent", context=context, ttl_ms=300
    )
    control = ControlGatewaySubscriber(
        profile.endpoint_uri("control_gateway_dispatch"),
        context=context,
        accepted_names={
            "start_navigation",
            "navigation_goal",
            "navigation_status",
            "start_base_pose",
            "cancel_navigation",
        },
    )
    orientation = connect_subscriber(
        context, profile.endpoint_uri("orientation_telemetry"), conflate=True, linger_ms=0
    )
    monitor = CameraMonitor(profile, base, config, imu_to_base)
    runtime = UniLMRuntime(
        config,
        base,
        send=sender.send,
        infer=client.infer,
        current_pose=monitor.current,
        geometry=geometry,
        report=lambda code, message, **fields: service.event(
            logging.INFO if code == "UNILM_COMPLETED" else logging.WARNING, code, message, **fields
        ),
    )
    task = str(profile.component("vla")["vla_prompt"])
    monitor.thread.start()
    service.event(
        logging.INFO,
        "READY",
        "UniLM-Nav ready; B=standalone, N=complete agent",
        model=config.model,
        camera_height_m=config.camera_height_m,
        **asdict(runtime.camera),
    )
    try:
        while True:
            start = time.monotonic()
            while True:
                command = control.read_command()
                if command is None:
                    break
                runtime.command(command.name, command.parameters, task)
            while not monitor.observations.empty():
                runtime.observe(monitor.observations.get())
            try:
                while True:
                    raw = orientation.recv(zmq.NOBLOCK)
                    runtime.orientation = decode_orientation_telemetry(raw)
            except zmq.Again:
                pass
            except ValueError:
                runtime.orientation = None
            runtime.tick()
            time.sleep(max(0.001, 1 / base.planner_hz - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        runtime.terminal("stopped", "service_shutdown")
        monitor.close()
        control.close()
        orientation.close(0)
        sender.close()
        context.term()
        service.close()


def check_ready(profile_path: str, timeout: float) -> bool:
    from gear_sonic.runtime.profile import load_runtime_profile
    from gear_sonic.utils.inference.base_pose.agent import load_base_pose_config

    profile = load_runtime_profile(profile_path)
    base = load_base_pose_config(profile_path)
    config = load_component_config(UniLMConfig, "unilm_nav", profile_path)
    _, transform = robot_geometry(config.urdf_path, config.camera_height_m)
    monitor = CameraMonitor(profile, base, config, transform)
    monitor.thread.start()
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                monitor.current()
                if not monitor.observations.empty():
                    print(f"UniLM {monitor.camera.camera_stream} RGB-D and synchronized odometry: READY")
                    return True
            except UniLMError:
                pass
            time.sleep(0.1)
        print(f"UniLM sensors unavailable: {monitor.last_error}")
        return False
    finally:
        monitor.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read-only UniLM sensor readiness check")
    parser.add_argument("--check-ready", action="store_true", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    raise SystemExit(0 if check_ready(args.profile, args.timeout) else 1)
