from __future__ import annotations

import subprocess

from gear_sonic.scripts import navdp_readiness_gate


def test_topic_gate_requires_a_real_message(monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(navdp_readiness_gate.subprocess, "run", run)

    assert navdp_readiness_gate.wait_for_topic_sample("/livox/lidar", 30.0)
    assert calls == [[
        "ros2", "topic", "echo", "/livox/lidar", "--once",
        "--qos-profile", "sensor_data",
        "--qos-reliability", "best_effort",
    ]]


def test_topic_gate_fails_closed_on_timeout(monkeypatch) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ros2 topic echo", 30.0)

    monkeypatch.setattr(navdp_readiness_gate.subprocess, "run", timeout)
    ticks = iter((0.0, 31.0))
    monkeypatch.setattr(navdp_readiness_gate.time, "monotonic", lambda: next(ticks))

    assert not navdp_readiness_gate.wait_for_topic_sample("/livox/imu", 30.0)


def test_topic_gate_retries_until_dds_discovers_publisher(monkeypatch) -> None:
    attempts = iter((1, 0))
    calls = 0

    def run(command, **_kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, next(attempts))

    monkeypatch.setattr(navdp_readiness_gate.subprocess, "run", run)
    monkeypatch.setattr(navdp_readiness_gate.time, "sleep", lambda _seconds: None)

    assert navdp_readiness_gate.wait_for_topic_sample("/livox/lidar", 30.0)
    assert calls == 2


def test_lidar_stage_waits_for_lidar_and_imu(monkeypatch) -> None:
    topics: list[tuple[str, float]] = []
    monkeypatch.setattr(
        navdp_readiness_gate,
        "wait_for_topic_sample",
        lambda topic, timeout: topics.append((topic, timeout)) or True,
    )

    assert navdp_readiness_gate.wait_for_lidar(30.0)
    assert [topic for topic, _timeout in topics] == ["/livox/lidar", "/livox/imu"]
    assert all(0.0 < timeout <= 30.0 for _topic, timeout in topics)


def test_navigation_stage_checks_odom_cloud_camera_and_server(monkeypatch) -> None:
    topics: list[str] = []
    endpoints: list[tuple[str, int]] = []
    monkeypatch.setattr(
        navdp_readiness_gate,
        "wait_for_topic_sample",
        lambda topic, _timeout: topics.append(topic) or True,
    )
    monkeypatch.setattr(
        navdp_readiness_gate,
        "wait_for_tcp",
        lambda host, port, _timeout: endpoints.append((host, port)) or True,
    )

    assert navdp_readiness_gate.wait_for_navigation(
        timeout=60.0,
        camera_host="192.168.123.164",
        camera_port=5555,
        navdp_host="127.0.0.1",
        navdp_port=19999,
    )
    assert topics == ["/Odometry_loc", "/cloud_registered_1"]
    assert endpoints == [("192.168.123.164", 5555), ("127.0.0.1", 19999)]


def test_gateway_navigation_stage_requires_a_real_gateway_ping(monkeypatch) -> None:
    gateway_checks: list[tuple[str, int]] = []
    monkeypatch.setattr(
        navdp_readiness_gate,
        "wait_for_topic_sample",
        lambda _topic, _timeout: True,
    )
    monkeypatch.setattr(
        navdp_readiness_gate,
        "wait_for_tcp",
        lambda _host, _port, _timeout: True,
    )
    monkeypatch.setattr(
        navdp_readiness_gate,
        "wait_for_sensor_gateway",
        lambda host, port, _timeout: gateway_checks.append((host, port)) or True,
    )

    assert navdp_readiness_gate.wait_for_navigation(
        timeout=60.0,
        camera_host="192.168.123.164",
        camera_port=5555,
        navdp_host="127.0.0.1",
        navdp_port=19999,
        require_sensor_gateway=True,
        sensor_gateway_port=5560,
    )
    assert gateway_checks == [("127.0.0.1", 5560)]
