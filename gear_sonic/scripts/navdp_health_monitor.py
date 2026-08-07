#!/usr/bin/env python3
"""Compact readiness monitor for the one-window NavDP navigation stack."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import subprocess
import time


@dataclass
class HealthConfig:
    camera_host: str = "192.168.123.164"
    camera_port: int = 5555
    navdp_host: str = "127.0.0.1"
    navdp_port: int = 19999
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
            "lidar": "/livox/lidar" in topics,
            "imu": "/livox/imu" in topics,
            "odom": "/Odometry_loc" in topics,
            "cloud": "/cloud_registered_1" in topics,
        }
        summary = " ".join(f"{name}={'OK' if ready else '--'}" for name, ready in states.items())
        print(f"\r[Health] {summary}", end="", flush=True)
        time.sleep(config.interval_s)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(HealthConfig))
