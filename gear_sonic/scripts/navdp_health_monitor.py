#!/usr/bin/env python3
"""Compact readiness monitor for the one-window NavDP navigation stack."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import subprocess
import time

from gear_sonic.runtime.config import load_runtime_profile

_DEFAULT_PROFILE = load_runtime_profile()
_CAMERA_ENDPOINT = _DEFAULT_PROFILE.endpoint("camera_server")
_NAVDP_ENDPOINT = _DEFAULT_PROFILE.endpoint("xnavdp_http")


@dataclass
class HealthConfig:
    camera_host: str = _CAMERA_ENDPOINT.host
    camera_port: int = _CAMERA_ENDPOINT.port
    navdp_host: str = _NAVDP_ENDPOINT.host
    navdp_port: int = _NAVDP_ENDPOINT.port
    interval_s: float = 1.0


def reachable(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ros_topics() -> set[str]:
    try:
        result = subprocess.run(
            ["ros2", "topic", "list"], capture_output=True, text=True, timeout=2.0
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()
    return set(result.stdout.splitlines()) if result.returncode == 0 else set()


def main(config: HealthConfig) -> None:
    while True:
        topics = ros_topics()
        states = {
            "camera": reachable(config.camera_host, config.camera_port),
            "navdp": reachable(config.navdp_host, config.navdp_port),
            "lidar": _DEFAULT_PROFILE.ros_topics["lidar"] in topics,
            "imu": _DEFAULT_PROFILE.ros_topics["lidar_imu"] in topics,
            "odom": _DEFAULT_PROFILE.ros_topics["odometry"] in topics,
            "cloud": _DEFAULT_PROFILE.ros_topics["registered_cloud"] in topics,
        }
        summary = " ".join(f"{name}={'OK' if ready else '--'}" for name, ready in states.items())
        print(f"\r[Health] {summary}", end="", flush=True)
        time.sleep(config.interval_s)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(HealthConfig))
