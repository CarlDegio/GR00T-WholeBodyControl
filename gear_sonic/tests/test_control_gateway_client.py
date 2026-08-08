from __future__ import annotations

from gear_sonic.runtime.contracts import MessageMetadata, OperatorCommand
from gear_sonic.runtime.control_client import ControlGatewaySubscriber


def _command(sequence: int, timestamp_ns: int, legacy_message: str) -> OperatorCommand:
    return OperatorCommand(
        metadata=MessageMetadata(
            source="test_console",
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            ttl_ms=100,
        ),
        command_id=f"test-{sequence}",
        name="legacy_passthrough",
        parameters={"legacy_message": legacy_message},
    )


def test_gateway_subscriber_adapter_preserves_message_and_rejects_replays() -> None:
    subscriber = object.__new__(ControlGatewaySubscriber)
    subscriber._monotonic_ns = lambda: 1_050_000_000
    subscriber._last_sequence_by_source = {}

    command = _command(7, 1_000_000_000, "o")

    assert subscriber.decode(command.to_json()) == "o"
    assert subscriber.decode(command.to_json()) is None


def test_gateway_subscriber_adapter_rejects_expired_intent() -> None:
    subscriber = object.__new__(ControlGatewaySubscriber)
    subscriber._monotonic_ns = lambda: 1_100_000_001
    subscriber._last_sequence_by_source = {}

    assert subscriber.decode(_command(1, 1_000_000_000, "i").to_json()) is None
