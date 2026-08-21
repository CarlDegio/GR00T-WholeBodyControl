#!/usr/bin/env python3
"""Block navigation stages until real ROS samples and TCP services are ready."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import subprocess
import sys
import time
from typing import Literal

from gear_sonic.runtime.gateway.sensor_client import (
    SensorGatewayClient,
    SensorGatewayClientError,
)
from gear_sonic.runtime.profile import load_runtime_profile


@dataclass
class ReadinessConfig:
    stage: Literal["lidar", "navigation"]
    profile: str = ""
    overlay: tuple[str, ...] = ()
    timeout: float = 30.0
    require_sensor_gateway: bool = False


@dataclass(frozen=True)
class ReadinessSettings:
    camera_host: str
    camera_port: int
    navdp_host: str
    navdp_port: int
    sensor_gateway_host: str
    sensor_gateway_port: int
    lidar_topic: str
    lidar_imu_topic: str
    odometry_topic: str
    registered_cloud_topic: str


def resolve_readiness_settings(config: ReadinessConfig) -> ReadinessSettings:
    profile = load_runtime_profile(config.profile or None, overlays=config.overlay)
    camera = profile.endpoint("camera_server")
    navdp = profile.endpoint("xnavdp_http")
    sensor_gateway = profile.endpoint("sensor_gateway_metadata")
    return ReadinessSettings(
        camera_host=camera.host,
        camera_port=camera.port,
        navdp_host=navdp.host,
        navdp_port=navdp.port,
        sensor_gateway_host=sensor_gateway.host,
        sensor_gateway_port=sensor_gateway.port,
        lidar_topic=str(profile.ros_topics["lidar"]),
        lidar_imu_topic=str(profile.ros_topics["lidar_imu"]),
        odometry_topic=str(profile.ros_topics["odometry"]),
        registered_cloud_topic=str(profile.ros_topics["registered_cloud"]),
    )


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


def wait_for_sensor_gateway(host: str, port: int, timeout: float) -> bool:
    endpoint = f"tcp://{host}:{int(port)}"
    print(
        f"[Readiness] waiting for SensorGateway {endpoint} ({timeout:.1f}s max) ...",
        flush=True,
    )
    deadline = time.monotonic() + timeout
    client = SensorGatewayClient(endpoint, request_timeout_ms=250)
    try:
        while time.monotonic() < deadline:
            try:
                if client.ping():
                    print(f"[Readiness] SensorGateway {endpoint}: READY", flush=True)
                    return True
            except SensorGatewayClientError:
                pass
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        return False
    finally:
        client.close()


def wait_for_lidar(
    timeout: float,
    lidar_topic: str = "",
    lidar_imu_topic: str = "",
) -> bool:
    if not lidar_topic or not lidar_imu_topic:
        profile = load_runtime_profile()
        lidar_topic = lidar_topic or str(profile.ros_topics["lidar"])
        lidar_imu_topic = lidar_imu_topic or str(profile.ros_topics["lidar_imu"])
    deadline = time.monotonic() + timeout
    return wait_for_topic_sample(
        lidar_topic, _remaining(deadline)
    ) and wait_for_topic_sample(
        lidar_imu_topic, _remaining(deadline)
    )


def wait_for_navigation(
    *,
    timeout: float,
    camera_host: str,
    camera_port: int,
    navdp_host: str,
    navdp_port: int,
    require_sensor_gateway: bool = False,
    sensor_gateway_host: str = "",
    sensor_gateway_port: int = 0,
    odometry_topic: str = "",
    registered_cloud_topic: str = "",
) -> bool:
    if not odometry_topic or not registered_cloud_topic or (
        require_sensor_gateway and (not sensor_gateway_host or not sensor_gateway_port)
    ):
        profile = load_runtime_profile()
        sensor_gateway = profile.endpoint("sensor_gateway_metadata")
        sensor_gateway_host = sensor_gateway_host or sensor_gateway.host
        sensor_gateway_port = sensor_gateway_port or sensor_gateway.port
        odometry_topic = odometry_topic or str(profile.ros_topics["odometry"])
        registered_cloud_topic = registered_cloud_topic or str(
            profile.ros_topics["registered_cloud"]
        )
    deadline = time.monotonic() + timeout
    checks = [
        lambda: wait_for_topic_sample(odometry_topic, _remaining(deadline)),
        lambda: wait_for_topic_sample(registered_cloud_topic, _remaining(deadline)),
        lambda: wait_for_tcp(camera_host, camera_port, _remaining(deadline)),
        lambda: wait_for_tcp(navdp_host, navdp_port, _remaining(deadline)),
    ]
    if require_sensor_gateway:
        checks.append(
            lambda: wait_for_sensor_gateway(
                sensor_gateway_host,
                sensor_gateway_port,
                _remaining(deadline),
            )
        )
    return all(check() for check in checks)


def main(config: ReadinessConfig) -> None:
    if config.timeout <= 0:
        raise ValueError("timeout must be positive")
    settings = resolve_readiness_settings(config)
    if config.stage == "lidar":
        ready = wait_for_lidar(
            config.timeout,
            settings.lidar_topic,
            settings.lidar_imu_topic,
        )
    else:
        ready = wait_for_navigation(
            timeout=config.timeout,
            camera_host=settings.camera_host,
            camera_port=settings.camera_port,
            navdp_host=settings.navdp_host,
            navdp_port=settings.navdp_port,
            require_sensor_gateway=config.require_sensor_gateway,
            sensor_gateway_host=settings.sensor_gateway_host,
            sensor_gateway_port=settings.sensor_gateway_port,
            odometry_topic=settings.odometry_topic,
            registered_cloud_topic=settings.registered_cloud_topic,
        )
    if not ready:
        print(f"[Readiness] ERROR: {config.stage} did not become ready", file=sys.stderr)
        raise SystemExit(1)
    print(f"[Readiness] {config.stage}: ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(ReadinessConfig))
