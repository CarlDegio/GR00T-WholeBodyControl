from __future__ import annotations

import json
from pathlib import Path

from gear_sonic.runtime.gateway.services import sensor as run_sensor_gateway


def test_script_imports_without_ros2_and_resolves_json_defaults() -> None:
    parser = run_sensor_gateway.build_argument_parser()
    args = parser.parse_args(["--no-enable-ros"])

    settings = run_sensor_gateway.resolve_sensor_gateway_settings(args)

    assert settings.camera_endpoint == "tcp://192.168.123.164:5555"
    assert settings.cpp_state_endpoint == "tcp://127.0.0.1:5557"
    assert settings.rpc_bind_endpoint == "tcp://127.0.0.1:5560"
    assert settings.runtime_metrics_bind_endpoint == "tcp://127.0.0.1:5567"
    assert settings.runtime_metrics_window_size == 100
    assert settings.ros_topics["odometry"] == "/Odometry_loc"
    assert settings.slot_count == 8
    assert settings.history_size == 64
    assert settings.loop_hz == 200.0
    assert settings.expected_hz["odometry"] == 10.0
    assert not settings.enable_ros
    assert not settings.enable_rgb_preview

    assert parser.parse_args(["--no-enable-runtime-metrics"]).enable_runtime_metrics is False
    assert parser.parse_args(["--no-enable-vla-timing"]).enable_runtime_metrics is False
    assert "ACTIVE TASK: waiting" in run_sensor_gateway._health_dashboard_text(
        {"streams": {}}, settings
    )


def test_profile_overlay_is_the_final_endpoint_configuration_layer(tmp_path: Path) -> None:
    overlay = tmp_path / "sensor_gateway.json"
    overlay.write_text(
        json.dumps(
            {
                "endpoints": {
                    "camera_server": {"port": 6001},
                    "cpp_state": {"host": "10.0.0.2", "port": 6002},
                    "sensor_gateway_metadata": {"host": "*", "port": 6003},
                }
            }
        ),
        encoding="utf-8",
    )
    parser = run_sensor_gateway.build_argument_parser()
    args = parser.parse_args(
        [
            "--overlay",
            str(overlay),
            "--no-enable-camera",
            "--no-enable-cpp-state",
            "--enable-rgb-preview",
        ]
    )

    settings = run_sensor_gateway.resolve_sensor_gateway_settings(args)

    assert settings.camera_endpoint == "tcp://192.168.123.164:6001"
    assert settings.cpp_state_endpoint == "tcp://10.0.0.2:6002"
    assert settings.rpc_bind_endpoint == "tcp://*:6003"
    assert not settings.enable_camera
    assert not settings.enable_cpp_state
    assert settings.enable_rgb_preview
    assert "--camera-port" not in parser._option_string_actions
    assert "--rpc-port" not in parser._option_string_actions


def test_sensor_gateway_script_has_no_control_endpoint_or_output_path() -> None:
    source = Path(run_sensor_gateway.__file__).read_text(encoding="utf-8")

    assert "cpp_command" not in source
    assert "5556" not in source
    assert "zmq.PUB" not in source
    assert "zmq.PUSH" not in source


def test_health_dashboard_lists_ports_topics_and_source_age() -> None:
    parser = run_sensor_gateway.build_argument_parser()
    settings = run_sensor_gateway.resolve_sensor_gateway_settings(
        parser.parse_args(["--no-enable-ros"])
    )
    payload = {
        "streams": {
            "source/camera_server": {
                "state": "healthy",
                "rate_hz": 20.0,
                "last_message_age_ms": 12.5,
                "latency_ms": None,
            },
            "source/cpp_state": {
                "state": "stale",
                "rate_hz": 18.0,
                "last_message_age_ms": 80.0,
                "latency_ms": 3.0,
            },
        }
    }

    timing = {
        "sample_count": 4,
        "window_size": 100,
        "last_sample_age_ms": 8.0,
        "values": {
            "policy_roundtrip": {
                "last": 101.0,
                "mean": 99.0,
                "p50": 98.0,
                "p95": 120.0,
            }
        },
    }
    text = run_sensor_gateway._health_dashboard_text(
        payload, settings, "vla", timing
    )

    assert "tcp://192.168.123.164:5555" in text
    assert "tcp://127.0.0.1:5557" in text
    assert "tcp://127.0.0.1:5560" in text
    assert "/Odometry_loc" in text
    assert "12.5ms" in text
    assert "3.0ms" in text
    assert "ACTIVE TASK: VLA" in text
    assert "policy_roundtrip" in text
    assert "action_ready" in text
    assert "120.0" in text


def test_health_dashboard_switches_to_lavira_without_showing_vla_segments() -> None:
    parser = run_sensor_gateway.build_argument_parser()
    settings = run_sensor_gateway.resolve_sensor_gateway_settings(
        parser.parse_args(["--no-enable-ros"])
    )
    text = run_sensor_gateway._health_dashboard_text(
        {"streams": {}},
        settings,
        "lavira",
        {"sample_count": 1, "window_size": 100, "values": {}},
    )

    assert "ACTIVE TASK: LaViRA / ObjectNav" in text
    assert "api_inference" in text
    assert "camera_rgbd" in text
    assert "policy_roundtrip" not in text


def test_health_dashboard_shows_base_pose_control_timing() -> None:
    parser = run_sensor_gateway.build_argument_parser()
    settings = run_sensor_gateway.resolve_sensor_gateway_settings(
        parser.parse_args(["--no-enable-ros"])
    )
    text = run_sensor_gateway._health_dashboard_text(
        {"streams": {}},
        settings,
        "base_pose",
        {"sample_count": 1, "window_size": 100, "values": {}},
    )

    assert "ACTIVE TASK: BASE_POSE" in text
    assert "worker_to_control" in text
    assert "control_update" in text
    assert "policy_roundtrip" not in text


def test_source_state_changes_only_reports_failures_and_recovery() -> None:
    previous: dict[str, str] = {}

    def changes(state: str, error: str = ""):
        return run_sensor_gateway._source_state_changes(
            {
                "streams": {
                    "source/camera_server": {
                        "state": state,
                        "last_error": error,
                    },
                    "camera/ego_view": {"state": "down"},
                }
            },
            previous,
        )

    assert changes("waiting") == []
    assert changes("healthy") == []
    assert changes("stale") == [
        ("source/camera_server", "stale", "healthy", "")
    ]
    assert changes("stale") == []
    assert changes("down", "camera disconnected") == [
        ("source/camera_server", "down", "stale", "camera disconnected")
    ]
    assert changes("down", "camera disconnected") == []
    assert changes("healthy") == [
        ("source/camera_server", "healthy", "down", "")
    ]
    assert changes("idle") == []


def test_noninteractive_health_display_does_not_print_periodic_summaries(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(run_sensor_gateway.sys.stdout, "isatty", lambda: False)
    display = run_sensor_gateway._HealthDisplay()

    display.render(
        {"streams": {"source/camera_server": {"state": "healthy"}}},
        run_sensor_gateway.resolve_sensor_gateway_settings(
            run_sensor_gateway.build_argument_parser().parse_args(
                ["--no-enable-ros"]
            )
        ),
    )

    assert capsys.readouterr().out == ""
