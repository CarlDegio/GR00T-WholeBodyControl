from __future__ import annotations

from pathlib import Path

from gear_sonic.scripts import run_sensor_gateway


def test_script_imports_without_ros2_and_resolves_json_defaults() -> None:
    parser = run_sensor_gateway.build_argument_parser()
    args = parser.parse_args(
        [
            "--camera-host",
            "192.168.123.164",
            "--no-enable-ros",
        ]
    )

    settings = run_sensor_gateway.resolve_sensor_gateway_settings(args)

    assert settings.camera_endpoint == "tcp://192.168.123.164:5555"
    assert settings.cpp_state_endpoint == "tcp://127.0.0.1:5557"
    assert settings.rpc_bind_endpoint == "tcp://127.0.0.1:5560"
    assert settings.vla_timing_bind_endpoint == "tcp://127.0.0.1:5567"
    assert settings.ros_topics["odometry"] == "/Odometry_loc"
    assert settings.slot_count == 8
    assert settings.history_size == 64
    assert settings.loop_hz == 200.0
    assert settings.expected_hz["odometry"] == 10.0
    assert not settings.enable_ros


def test_cli_overrides_are_the_final_configuration_layer() -> None:
    parser = run_sensor_gateway.build_argument_parser()
    args = parser.parse_args(
        [
            "--camera-port",
            "6001",
            "--cpp-state-host",
            "10.0.0.2",
            "--cpp-state-port",
            "6002",
            "--rpc-bind-host",
            "*",
            "--rpc-port",
            "6003",
            "--no-enable-camera",
            "--no-enable-cpp-state",
        ]
    )

    settings = run_sensor_gateway.resolve_sensor_gateway_settings(args)

    assert settings.camera_endpoint == "tcp://192.168.123.164:6001"
    assert settings.cpp_state_endpoint == "tcp://10.0.0.2:6002"
    assert settings.rpc_bind_endpoint == "tcp://*:6003"
    assert not settings.enable_camera
    assert not settings.enable_cpp_state


def test_sensor_gateway_script_has_no_control_endpoint_or_output_path() -> None:
    source = Path(run_sensor_gateway.__file__).read_text(encoding="utf-8")

    assert not run_sensor_gateway.CONTROL_OUTPUTS_ENABLED
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
        "segments_ms": {
            "policy_roundtrip": {
                "last": 101.0,
                "mean": 99.0,
                "p50": 98.0,
                "p95": 120.0,
            }
        },
    }
    text = run_sensor_gateway._health_dashboard_text(payload, settings, timing)

    assert "tcp://192.168.123.164:5555" in text
    assert "tcp://127.0.0.1:5557" in text
    assert "tcp://127.0.0.1:5560" in text
    assert "/Odometry_loc" in text
    assert "12.5ms" in text
    assert "3.0ms" in text
    assert "VLA segmented latency" in text
    assert "policy_roundtrip" in text
    assert "120.0" in text
