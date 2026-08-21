from __future__ import annotations

import json

import pytest

from gear_sonic.runtime.protocol import (
    MessageMetadata,
    OperatorCommand,
    SharedMemoryFrame,
)
from gear_sonic.runtime.profile import (
    ENDPOINT_SCHEMES,
    load_runtime_profile,
)


def test_current_inference_profile_has_unique_ports() -> None:
    profile = load_runtime_profile()
    ports = [address.port for address in profile.endpoints.values()]

    assert len(ports) == len(set(ports))
    assert set(profile.endpoints) == set(ENDPOINT_SCHEMES)
    assert profile.endpoint("policy_server").port == 29999
    assert profile.endpoint("navigation_runtime_status").port == 5570
    assert profile.endpoint("xnavdp_http").port == 19999


def test_endpoint_uri_and_reserved_gateway_channels() -> None:
    profile = load_runtime_profile()

    assert profile.endpoint_uri("planner_relay") == "tcp://127.0.0.1:5563"
    assert profile.endpoint_uri("xnavdp_http") == "http://127.0.0.1:19999"
    assert profile.endpoint("sensor_gateway_visualization_ingress").port == 5566


def test_message_metadata_expiry_uses_explicit_clock_sample() -> None:
    metadata = MessageMetadata(
        source="operator_console",
        sequence=7,
        timestamp_ns=1_000_000_000,
        ttl_ms=100,
        generation=2,
    )

    assert metadata.age_ms(1_080_000_000) == 80.0
    assert not metadata.is_expired(1_100_000_000)
    assert metadata.is_expired(1_100_000_001)
    assert MessageMetadata.from_dict(metadata.to_dict()) == metadata


def test_operator_command_json_round_trip_preserves_identity_and_ttl() -> None:
    command = OperatorCommand(
        metadata=MessageMetadata(
            source="sonicctl",
            sequence=3,
            timestamp_ns=123456,
            ttl_ms=250,
            generation=9,
        ),
        command_id="console-3",
        name="manual_velocity",
        parameters={"vx": 0.3, "vy": 0.0, "wz": -0.2},
    )

    encoded = command.to_json()
    decoded = OperatorCommand.from_json(encoded)

    assert decoded == command
    assert json.loads(encoded)["version"] == 1
    assert decoded.parameters == {"vx": 0.3, "vy": 0.0, "wz": -0.2}


def test_contracts_reject_bad_versions() -> None:
    command = {
        "type": "sonic.operator_command",
        "version": 99,
        "metadata": {},
        "command_id": "bad",
        "name": "stop",
    }

    with pytest.raises(ValueError, match="unsupported operator command"):
        OperatorCommand.from_dict(command)


def test_shared_memory_frame_round_trip() -> None:
    metadata = MessageMetadata(
        source="control_gateway",
        sequence=5,
        timestamp_ns=900,
        ttl_ms=500,
    )
    frame = SharedMemoryFrame(
        metadata=metadata,
        stream="chest_view_depth",
        shared_memory="sonic_sensor_0",
        shape=(480, 640),
        dtype="uint16",
        offset_bytes=0,
        size_bytes=480 * 640 * 2,
        source_timestamp_ns=850,
        source_clock="camera_unix",
    )

    assert SharedMemoryFrame.from_dict(frame.to_dict()) == frame
