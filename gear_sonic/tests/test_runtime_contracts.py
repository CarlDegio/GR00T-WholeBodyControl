from __future__ import annotations

import json

import pytest

from gear_sonic.runtime.contracts import (
    CommandAck,
    ControlGatewayHealth,
    MessageMetadata,
    OperatorCommand,
    SharedMemoryFrame,
    validate_finite_velocity,
)
from gear_sonic.runtime.endpoints import ENDPOINTS, ROS_TOPICS, Transport, get_endpoint


def test_current_inference_profile_has_unique_ports() -> None:
    ports = [endpoint.port for endpoint in ENDPOINTS.values()]

    assert len(ports) == len(set(ports))
    assert get_endpoint("policy_server").port == 29999
    assert get_endpoint("camera_server").port == 5555
    assert get_endpoint("cpp_command").port == 5556
    assert get_endpoint("cpp_state").port == 5557
    assert get_endpoint("navigation_command").port == 5558
    assert get_endpoint("navigation_status").port == 5559
    assert get_endpoint("planner_relay").port == 5563
    assert get_endpoint("depth_anything").port == 5564
    assert get_endpoint("navigation_runtime_status").port == 5570
    assert get_endpoint("xnavdp_http").port == 19999


def test_endpoint_uri_and_reserved_gateway_channels() -> None:
    assert get_endpoint("planner_relay").uri() == "tcp://127.0.0.1:5563"
    assert get_endpoint("xnavdp_http").uri("localhost") == "http://localhost:19999"
    assert get_endpoint("xnavdp_http").transport is Transport.HTTP
    assert get_endpoint("sensor_gateway_metadata").active
    assert get_endpoint("control_gateway_intent").active
    assert get_endpoint("control_gateway_status").active
    assert get_endpoint("control_gateway_dispatch").active
    assert get_endpoint("sensor_gateway_visualization_ingress").port == 5566
    assert get_endpoint("navigation_runtime_status").active


def test_ros_topic_inventory_uses_absolute_names() -> None:
    assert set(ROS_TOPICS) == {
        "/livox/lidar",
        "/livox/imu",
        "/Odometry_loc",
        "/cloud_registered_1",
    }


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
    assert validate_finite_velocity(decoded.parameters) == (0.3, 0.0, -0.2)


def test_contracts_reject_bad_versions_and_non_finite_velocity() -> None:
    command = {
        "type": "sonic.operator_command",
        "version": 99,
        "metadata": {},
        "command_id": "bad",
        "name": "stop",
    }

    with pytest.raises(ValueError, match="unsupported operator command"):
        OperatorCommand.from_dict(command)
    with pytest.raises(ValueError, match="finite"):
        validate_finite_velocity({"vx": float("nan"), "vy": 0.0, "wz": 0.0})


def test_ack_and_shared_memory_frame_round_trip() -> None:
    metadata = MessageMetadata(
        source="control_gateway",
        sequence=5,
        timestamp_ns=900,
        ttl_ms=500,
    )
    ack = CommandAck(
        metadata=metadata,
        command_id="console-3",
        accepted=True,
        system_state="active",
        control_mode="planner_manual",
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

    assert CommandAck.from_dict(ack.to_dict()) == ack
    assert SharedMemoryFrame.from_dict(frame.to_dict()) == frame


def test_control_gateway_health_round_trip() -> None:
    health = ControlGatewayHealth(
        metadata=MessageMetadata(
            source="control_gateway",
            sequence=4,
            timestamp_ns=123,
            ttl_ms=1000,
        ),
        state="ready",
        accepted_commands=8,
        rejected_commands=1,
        last_command_id="gui-7",
    )

    assert ControlGatewayHealth.from_dict(health.to_dict()) == health
