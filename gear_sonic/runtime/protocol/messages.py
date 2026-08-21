"""Versioned, transport-neutral contracts for the planned runtime gateways."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Mapping, Sequence

CONTRACT_VERSION = 1


def _require_nonempty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty")


@dataclass(frozen=True)
class MessageMetadata:
    """Ordering and lifetime using the gateway host's monotonic clock."""

    source: str
    sequence: int
    timestamp_ns: int
    ttl_ms: int
    generation: int = 0

    def __post_init__(self) -> None:
        _require_nonempty(self.source, "source")
        if self.sequence < 0:
            raise ValueError("sequence cannot be negative")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns cannot be negative")
        if self.ttl_ms < 0:
            raise ValueError("ttl_ms cannot be negative")
        if self.generation < 0:
            raise ValueError("generation cannot be negative")

    @classmethod
    def now(
        cls,
        *,
        source: str,
        sequence: int,
        ttl_ms: int,
        generation: int = 0,
    ) -> "MessageMetadata":
        return cls(
            source=source,
            sequence=sequence,
            timestamp_ns=time.monotonic_ns(),
            ttl_ms=ttl_ms,
            generation=generation,
        )

    def age_ms(self, now_ns: int) -> float:
        return max(0, int(now_ns) - self.timestamp_ns) / 1_000_000.0

    def is_expired(self, now_ns: int) -> bool:
        return self.age_ms(now_ns) > self.ttl_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "ttl_ms": self.ttl_ms,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MessageMetadata":
        return cls(
            source=str(payload["source"]),
            sequence=int(payload["sequence"]),
            timestamp_ns=int(payload["timestamp_ns"]),
            ttl_ms=int(payload["ttl_ms"]),
            generation=int(payload.get("generation", 0)),
        )


@dataclass(frozen=True)
class OperatorCommand:
    """Structured user intent; it is not itself a low-level robot command."""

    metadata: MessageMetadata
    command_id: str
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA = "sonic.operator_command"

    def __post_init__(self) -> None:
        _require_nonempty(self.command_id, "command_id")
        _require_nonempty(self.name, "name")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.SCHEMA,
            "version": CONTRACT_VERSION,
            "metadata": self.metadata.to_dict(),
            "command_id": self.command_id,
            "name": self.name,
            "parameters": dict(self.parameters),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorCommand":
        if payload.get("type") != cls.SCHEMA or int(payload.get("version", -1)) != CONTRACT_VERSION:
            raise ValueError("unsupported operator command contract")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("operator command parameters must be a mapping")
        return cls(
            metadata=MessageMetadata.from_dict(payload["metadata"]),
            command_id=str(payload["command_id"]),
            name=str(payload["name"]),
            parameters=dict(parameters),
        )

    @classmethod
    def from_json(cls, message: str | bytes) -> "OperatorCommand":
        return cls.from_dict(json.loads(message))


@dataclass(frozen=True)
class SharedMemoryFrame:
    """Metadata reference to a high-bandwidth frame stored outside ZMQ."""

    metadata: MessageMetadata
    stream: str
    shared_memory: str
    shape: tuple[int, ...]
    dtype: str
    offset_bytes: int
    size_bytes: int
    source_timestamp_ns: int
    source_clock: str = "unknown"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA = "sonic.shared_memory_frame"

    def __post_init__(self) -> None:
        _require_nonempty(self.stream, "stream")
        _require_nonempty(self.shared_memory, "shared_memory")
        _require_nonempty(self.dtype, "dtype")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("shape must contain positive dimensions")
        if self.offset_bytes < 0 or self.size_bytes <= 0:
            raise ValueError("shared-memory byte range is invalid")
        if self.source_timestamp_ns < 0:
            raise ValueError("source_timestamp_ns cannot be negative")
        _require_nonempty(self.source_clock, "source_clock")
        if not isinstance(self.attributes, Mapping):
            raise ValueError("shared-memory frame attributes must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.SCHEMA,
            "version": CONTRACT_VERSION,
            "metadata": self.metadata.to_dict(),
            "stream": self.stream,
            "shared_memory": self.shared_memory,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "offset_bytes": self.offset_bytes,
            "size_bytes": self.size_bytes,
            "source_timestamp_ns": self.source_timestamp_ns,
            "source_clock": self.source_clock,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SharedMemoryFrame":
        if payload.get("type") != cls.SCHEMA or int(payload.get("version", -1)) != CONTRACT_VERSION:
            raise ValueError("unsupported shared-memory frame contract")
        raw_shape = payload["shape"]
        if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
            raise ValueError("shared-memory frame shape must be a sequence")
        return cls(
            metadata=MessageMetadata.from_dict(payload["metadata"]),
            stream=str(payload["stream"]),
            shared_memory=str(payload["shared_memory"]),
            shape=tuple(int(dimension) for dimension in raw_shape),
            dtype=str(payload["dtype"]),
            offset_bytes=int(payload["offset_bytes"]),
            size_bytes=int(payload["size_bytes"]),
            source_timestamp_ns=int(payload["source_timestamp_ns"]),
            source_clock=str(payload.get("source_clock", "unknown")),
            attributes=dict(payload.get("attributes", {})),
        )
