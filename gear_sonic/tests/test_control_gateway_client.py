from __future__ import annotations

from gear_sonic.runtime.protocol import MessageMetadata, OperatorCommand
from gear_sonic.runtime.gateway.control_client import ControlGatewaySubscriber


def _command(sequence: int, timestamp_ns: int, name: str) -> OperatorCommand:
    return OperatorCommand(
        metadata=MessageMetadata(
            source="test_console",
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            ttl_ms=100,
        ),
        command_id=f"test-{sequence}",
        name=name,
        parameters={},
    )


def test_gateway_subscriber_decodes_typed_command_and_rejects_replays() -> None:
    subscriber = object.__new__(ControlGatewaySubscriber)
    subscriber._monotonic_ns = lambda: 1_050_000_000
    subscriber._last_sequence_by_source = {}

    command = _command(7, 1_000_000_000, "select_planner_mode")

    assert subscriber.decode_command(command.to_json()) == command
    assert subscriber.decode_command(command.to_json()) is None


def test_gateway_subscriber_rejects_expired_command() -> None:
    subscriber = object.__new__(ControlGatewaySubscriber)
    subscriber._monotonic_ns = lambda: 1_100_000_001
    subscriber._last_sequence_by_source = {}

    command = _command(1, 1_000_000_000, "select_pose_mode")
    assert subscriber.decode_command(command.to_json()) is None
