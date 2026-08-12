#!/usr/bin/env python3
"""Run FAST-LIO and reset it after a clearly invalid odometry excursion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
import signal
import subprocess
import threading
import time
from typing import Any

from gear_sonic.runtime.config import load_runtime_profile
from gear_sonic.runtime.control_client import ControlGatewayIntentClient
from gear_sonic.runtime.slam_recovery import (
    MaliciousDriftDetector,
    OdometrySample,
    SlamRecoveryLimits,
)


@dataclass(frozen=True)
class FastLioSupervisorSettings:
    profile_name: str
    config_file: str
    odometry_topic: str
    control_gateway_endpoint: str
    recovery_limits: SlamRecoveryLimits


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="", help="Runtime YAML profile")
    parser.add_argument("--config-file", required=True, help="FAST-LIO config file")
    parser.add_argument("--control-gateway-endpoint", default="")
    return parser


def resolve_settings(args: argparse.Namespace) -> FastLioSupervisorSettings:
    profile = load_runtime_profile(args.profile or None)
    control_gateway_endpoint = (
        str(args.control_gateway_endpoint)
        if args.control_gateway_endpoint
        else profile.endpoint_uri("control_gateway_intent")
    )
    if not control_gateway_endpoint.startswith("tcp://"):
        raise ValueError("control_gateway_endpoint must use tcp://")
    return FastLioSupervisorSettings(
        profile_name=profile.name,
        config_file=str(args.config_file),
        odometry_topic=str(profile.ros_topics["odometry"]),
        control_gateway_endpoint=control_gateway_endpoint,
        recovery_limits=SlamRecoveryLimits.from_mapping(
            profile.component("slam_recovery")
        ),
    )


def build_fastlio_launch_argv(config_file: str) -> list[str]:
    return [
        "ros2",
        "launch",
        "fast_lio",
        "mapping.launch.py",
        f"config_file:={config_file}",
        "rviz:=false",
    ]


def _start_fastlio(config_file: str) -> subprocess.Popen:
    command = build_fastlio_launch_argv(config_file)
    print(f"[SLAM recovery] starting: {' '.join(command)}", flush=True)
    return subprocess.Popen(command, start_new_session=True)


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process: subprocess.Popen,
    process_group_id: int,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        # Reap the launch parent when it exits, but keep the group as the
        # ownership/liveness boundary because mapping descendants may remain.
        process.poll()
        if not _process_group_exists(process_group_id):
            return True
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            return False
        time.sleep(min(0.05, remaining_s))


def _signal_process_group(
    process_group_id: int,
    signum: signal.Signals,
) -> None:
    os.killpg(process_group_id, signum)


def _stop_fastlio(process: subprocess.Popen) -> None:
    process_group_id = process.pid
    if not _process_group_exists(process_group_id):
        process.poll()
        return

    for signum, timeout_s in (
        (signal.SIGINT, 5.0),
        (signal.SIGTERM, 2.0),
        (signal.SIGKILL, 1.0),
    ):
        try:
            _signal_process_group(process_group_id, signum)
        except ProcessLookupError:
            process.poll()
            return
        if _wait_for_process_group_exit(
            process,
            process_group_id,
            timeout_s,
        ):
            return

    raise RuntimeError(
        f"FAST-LIO process group {process_group_id} survived SIGKILL"
    )


def _cancel_navigation_and_stop_fastlio(
    control: ControlGatewayIntentClient,
    process: subprocess.Popen,
    reason: str,
) -> None:
    print(
        "[SLAM recovery] MALICIOUS DRIFT detected: "
        f"{reason}; cancelling navigation and commanding zero velocity",
        flush=True,
    )
    control.send("navigation_key", {"key": " "})
    # Let ControlGateway dispatch the typed stop before tearing down SLAM.
    time.sleep(0.1)
    _stop_fastlio(process)


def _quaternion_yaw(orientation: Any) -> float:
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    w = float(orientation.w)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def odometry_sample(message: Any, *, fallback_timestamp_s: float) -> OdometrySample:
    stamp = message.header.stamp
    timestamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    if timestamp_s <= 0.0:
        timestamp_s = float(fallback_timestamp_s)
    pose = message.pose.pose
    twist = message.twist.twist
    return OdometrySample(
        timestamp_s=timestamp_s,
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw=_quaternion_yaw(pose.orientation),
        velocity_x=float(twist.linear.x),
        velocity_y=float(twist.linear.y),
        yaw_rate=float(twist.angular.z),
    )


def run_fastlio_supervisor(settings: FastLioSupervisorSettings) -> None:
    # Lazy ROS imports keep configuration and pure detector tests usable off-robot.
    import rclpy
    from nav_msgs.msg import Odometry

    rclpy.init(args=None)
    node = rclpy.create_node("sonic_fastlio_supervisor")
    detector = MaliciousDriftDetector(settings.recovery_limits)
    stop = threading.Event()
    pending_reason: str | None = None
    child: subprocess.Popen | None = None
    control = ControlGatewayIntentClient(
        settings.control_gateway_endpoint,
        source="slam_recovery",
    )

    def request_stop(_signum=None, _frame=None) -> None:
        stop.set()

    def on_odometry(message: Any) -> None:
        nonlocal pending_reason
        if pending_reason is not None:
            return
        now_s = time.monotonic()
        pending_reason = detector.observe(
            odometry_sample(message, fallback_timestamp_s=now_s),
            monotonic_s=now_s,
        )

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    subscription = node.create_subscription(
        Odometry,
        settings.odometry_topic,
        on_odometry,
        20,
    )
    try:
        print(
            f"[SLAM recovery] profile={settings.profile_name} "
            f"odometry={settings.odometry_topic}",
            flush=True,
        )
        child = _start_fastlio(settings.config_file)
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
            if child.poll() is not None:
                raise RuntimeError(
                    f"FAST-LIO exited unexpectedly with status {child.returncode}"
                )
            if pending_reason is None:
                continue

            reason = pending_reason
            _cancel_navigation_and_stop_fastlio(control, child, reason)
            if stop.is_set():
                break
            child = _start_fastlio(settings.config_file)
            detector.reset()
            pending_reason = None
            print(
                "[SLAM recovery] FAST-LIO restarted with an empty in-memory map; "
                "the cancelled navigation task will not resume automatically",
                flush=True,
            )
    finally:
        del subscription
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if child is not None:
            _stop_fastlio(child)
        control.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    settings = resolve_settings(build_argument_parser().parse_args())
    run_fastlio_supervisor(settings)


if __name__ == "__main__":
    main()
