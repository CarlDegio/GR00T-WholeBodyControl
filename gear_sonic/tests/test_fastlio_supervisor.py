from __future__ import annotations

from types import SimpleNamespace

from gear_sonic.scripts import run_fastlio_supervisor


def test_settings_come_from_the_unified_runtime_profile() -> None:
    parser = run_fastlio_supervisor.build_argument_parser()
    settings = run_fastlio_supervisor.resolve_settings(
        parser.parse_args(["--config-file", "mid360.yaml"])
    )

    assert settings.profile_name == "agent_full_current"
    assert settings.odometry_topic == "/Odometry_loc"
    assert settings.control_gateway_endpoint == "tcp://127.0.0.1:5561"
    assert settings.recovery_limits.max_planar_speed_m_s == 1.5
    assert settings.recovery_limits.consecutive_samples == 3


def test_fastlio_launch_uses_the_existing_mapping_contract() -> None:
    assert run_fastlio_supervisor.build_fastlio_launch_argv("mid360.yaml") == [
        "ros2",
        "launch",
        "fast_lio",
        "mapping.launch.py",
        "config_file:=mid360.yaml",
        "rviz:=false",
    ]


def test_odometry_message_is_reduced_without_ros_imports() -> None:
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=500_000_000)),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=1.0, y=2.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.3, y=0.0),
                angular=SimpleNamespace(z=0.2),
            )
        ),
    )

    sample = run_fastlio_supervisor.odometry_sample(
        message,
        fallback_timestamp_s=99.0,
    )

    assert sample.timestamp_s == 12.5
    assert sample.x == 1.0
    assert sample.y == 2.0
    assert sample.yaw == 0.0
    assert sample.velocity_x == 0.3
    assert sample.yaw_rate == 0.2


def test_recovery_sends_typed_stop_before_stopping_fastlio(monkeypatch) -> None:
    events: list[object] = []

    class FakeControl:
        def send(self, name: str, parameters: dict[str, object]) -> None:
            events.append(("send", name, parameters))

    process = object()
    monkeypatch.setattr(
        run_fastlio_supervisor.time,
        "sleep",
        lambda seconds: events.append(("sleep", seconds)),
    )
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_stop_fastlio",
        lambda stopped: events.append(("stop", stopped)),
    )

    run_fastlio_supervisor._cancel_navigation_and_stop_fastlio(
        FakeControl(),
        process,
        "planar_speed:3.200m/s",
    )

    assert events == [
        ("send", "navigation_key", {"key": " "}),
        ("sleep", 0.1),
        ("stop", process),
    ]
