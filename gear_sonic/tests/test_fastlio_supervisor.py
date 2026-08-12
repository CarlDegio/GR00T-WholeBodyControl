from __future__ import annotations

import signal
from types import SimpleNamespace

import pytest

from gear_sonic.scripts import run_fastlio_supervisor


class _ExitedLaunchProcess:
    pid = 4242

    def __init__(self) -> None:
        self.poll_count = 0

    def poll(self) -> int:
        self.poll_count += 1
        return 0


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


def test_group_wait_does_not_treat_exited_launch_parent_as_cleanup_complete(
    monkeypatch,
) -> None:
    process = _ExitedLaunchProcess()
    group_states = iter([True, False])
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_process_group_exists",
        lambda _process_group_id: next(group_states),
    )
    monkeypatch.setattr(run_fastlio_supervisor.time, "sleep", lambda _seconds: None)

    assert run_fastlio_supervisor._wait_for_process_group_exit(
        process,
        process.pid,
        timeout_s=1.0,
    )
    assert process.poll_count == 2


def test_stop_escalates_when_parent_exits_but_process_group_survives_sigint(
    monkeypatch,
) -> None:
    process = _ExitedLaunchProcess()
    signals: list[signal.Signals] = []
    wait_results = iter([False, True])
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_process_group_exists",
        lambda _process_group_id: True,
    )
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_signal_process_group",
        lambda _process_group_id, signum: signals.append(signum),
    )
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_wait_for_process_group_exit",
        lambda _process, _process_group_id, _timeout_s: next(wait_results),
    )

    run_fastlio_supervisor._stop_fastlio(process)

    assert signals == [signal.SIGINT, signal.SIGTERM]


def test_stop_is_noop_when_process_group_is_already_gone(monkeypatch) -> None:
    process = _ExitedLaunchProcess()
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_process_group_exists",
        lambda _process_group_id: False,
    )
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_signal_process_group",
        lambda *_args: pytest.fail("dead process group must not be signalled"),
    )

    run_fastlio_supervisor._stop_fastlio(process)

    assert process.poll_count == 1


def test_stop_raises_when_process_group_survives_sigkill(monkeypatch) -> None:
    process = _ExitedLaunchProcess()
    signals: list[signal.Signals] = []
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_process_group_exists",
        lambda _process_group_id: True,
    )
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_signal_process_group",
        lambda _process_group_id, signum: signals.append(signum),
    )
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_wait_for_process_group_exit",
        lambda _process, _process_group_id, _timeout_s: False,
    )

    with pytest.raises(RuntimeError, match="survived SIGKILL"):
        run_fastlio_supervisor._stop_fastlio(process)

    assert signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
