from __future__ import annotations

import subprocess

from gear_sonic.scripts import navdp_health_monitor


def test_ros_topic_timeout_is_reported_as_not_ready(monkeypatch) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ros2 topic list", 1.0)

    monkeypatch.setattr(navdp_health_monitor.subprocess, "run", timeout)

    assert navdp_health_monitor.ros_topics() == set()
