from __future__ import annotations

from pathlib import Path
import time

import msgpack
import numpy as np
import zmq

from gear_sonic.scripts import run_sensor_gateway_shadow


def test_shadow_script_imports_without_ros_and_uses_json_defaults() -> None:
    args = run_sensor_gateway_shadow.build_argument_parser().parse_args(
        [
            "--camera-host",
            "192.168.123.164",
            "--no-enable-ros",
        ]
    )

    settings = run_sensor_gateway_shadow.resolve_shadow_settings(args)

    assert settings.gateway_endpoint == "tcp://127.0.0.1:5560"
    assert settings.camera_endpoint == "tcp://192.168.123.164:5555"
    assert settings.cpp_state_endpoint == "tcp://127.0.0.1:5557"
    assert settings.sample_hz == 2.0
    assert settings.max_source_skew_ms == 1.0
    assert settings.startup_timeout_s == 30.0
    assert settings.camera_streams == (
        "ego_view",
        "ego_view_depth",
        "chest_view",
        "left_wrist",
        "right_wrist",
    )
    assert settings.enable_cpp_state
    assert not settings.enable_ros


def test_shadow_cli_overrides_sampling_and_gateway_address() -> None:
    args = run_sensor_gateway_shadow.build_argument_parser().parse_args(
        [
            "--gateway-host",
            "localhost",
            "--gateway-port",
            "6000",
            "--sample-hz",
            "5",
            "--duration-s",
            "10",
        ]
    )

    settings = run_sensor_gateway_shadow.resolve_shadow_settings(args)

    assert settings.gateway_endpoint == "tcp://localhost:6000"
    assert settings.sample_hz == 5.0
    assert settings.duration_s == 10.0


def test_shadow_script_has_no_control_transport() -> None:
    source = Path(run_sensor_gateway_shadow.__file__).read_text(encoding="utf-8")

    assert not run_sensor_gateway_shadow.CONTROL_OUTPUTS_ENABLED
    assert "cpp_command" not in source
    assert "5556" not in source
    assert "zmq.PUB" not in source
    assert "zmq.PUSH" not in source


def test_cpp_state_shadow_preserves_raw_payload_and_source_timestamp() -> None:
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("inproc://shadow-cpp-state")
    ingress = run_sensor_gateway_shadow.DirectCppStateIngress(
        context,
        "inproc://shadow-cpp-state",
    )
    state = {
        "ros_timestamp": 123.5,
        "base_quat": [1.0, 0.0, 0.0, 0.0],
    }
    payload = msgpack.packb(state, use_bin_type=True)
    try:
        deadline = time.monotonic() + 1.0
        while ingress.latest() is None and time.monotonic() < deadline:
            publisher.send(b"g1_debug" + payload)
            time.sleep(0.01)
            ingress.poll()

        sample = ingress.latest()
        assert sample is not None
        raw, source_ns = sample
        np.testing.assert_array_equal(raw, np.frombuffer(payload, dtype=np.uint8))
        assert source_ns == 123_500_000_000
    finally:
        ingress.close()
        publisher.close(linger=0)
        context.term()
