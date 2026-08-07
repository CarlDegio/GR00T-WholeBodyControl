#!/usr/bin/env python3
"""Block navigation stages until real ROS samples and TCP services are ready."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import subprocess
import sys
import time
from typing import Literal


@dataclass
class ReadinessConfig:
    stage: Literal["lidar", "navigation"]
    timeout: float = 30.0
    camera_host: str = "192.168.123.164"
    camera_port: int = 5555
    navdp_host: str = "127.0.0.1"
    navdp_port: int = 19999


def wait_for_topic_sample(topic: str, timeout: float) -> bool:
    print(f"[Readiness] waiting for a real {topic} sample ({timeout:.1f}s max) ...", flush=True)
    deadline = time.monotonic() + timeout
    command = [
        "ros2", "topic", "echo", topic, "--once",
        "--qos-profile", "sensor_data",
        "--qos-reliability", "best_effort",
    ]
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(0.1, min(2.0, remaining)),
            )
        except subprocess.TimeoutExpired:
            result = None
        except FileNotFoundError:
            return False
        if result is not None and result.returncode == 0:
            print(f"[Readiness] {topic}: READY", flush=True)
            return True
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    return False


def wait_for_tcp(host: str, port: int, timeout: float) -> bool:
    print(f"[Readiness] waiting for {host}:{port} ({timeout:.1f}s max) ...", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                print(f"[Readiness] {host}:{port}: READY", flush=True)
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _remaining(deadline: float) -> float:
    return max(0.1, deadline - time.monotonic())


def wait_for_lidar(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    return wait_for_topic_sample("/livox/lidar", _remaining(deadline)) and wait_for_topic_sample(
        "/livox/imu", _remaining(deadline)
    )


def wait_for_navigation(
    *,
    timeout: float,
    camera_host: str,
    camera_port: int,
    navdp_host: str,
    navdp_port: int,
) -> bool:
    deadline = time.monotonic() + timeout
    checks = (
        lambda: wait_for_topic_sample("/Odometry_loc", _remaining(deadline)),
        lambda: wait_for_topic_sample("/cloud_registered_1", _remaining(deadline)),
        lambda: wait_for_tcp(camera_host, camera_port, _remaining(deadline)),
        lambda: wait_for_tcp(navdp_host, navdp_port, _remaining(deadline)),
    )
    return all(check() for check in checks)


def main(config: ReadinessConfig) -> None:
    if config.timeout <= 0:
        raise ValueError("timeout must be positive")
    if config.stage == "lidar":
        ready = wait_for_lidar(config.timeout)
    else:
        ready = wait_for_navigation(
            timeout=config.timeout,
            camera_host=config.camera_host,
            camera_port=config.camera_port,
            navdp_host=config.navdp_host,
            navdp_port=config.navdp_port,
        )
    if not ready:
        print(f"[Readiness] ERROR: {config.stage} did not become ready", file=sys.stderr)
        raise SystemExit(1)
    print(f"[Readiness] {config.stage}: ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(ReadinessConfig))
