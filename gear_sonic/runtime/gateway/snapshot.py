"""Time-aligned sensor snapshot selection over shared-memory frame metadata."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from gear_sonic.runtime.protocol import SharedMemoryFrame


class TimestampBasis(str, Enum):
    RECEIVE = "receive"
    SOURCE = "source"


@dataclass(frozen=True)
class SnapshotRequest:
    streams: tuple[str, ...]
    max_age_ms: float
    max_skew_ms: float
    anchor_timestamp_ns: int | None = None
    timestamp_basis: TimestampBasis = TimestampBasis.RECEIVE

    def __post_init__(self) -> None:
        if not self.streams or any(not stream for stream in self.streams):
            raise ValueError("snapshot request requires non-empty stream names")
        if len(set(self.streams)) != len(self.streams):
            raise ValueError("snapshot request stream names must be unique")
        if self.max_age_ms < 0.0 or self.max_skew_ms < 0.0:
            raise ValueError("snapshot age and skew limits cannot be negative")
        if self.anchor_timestamp_ns is not None and self.anchor_timestamp_ns < 0:
            raise ValueError("snapshot anchor cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sonic.snapshot_request",
            "version": 1,
            "streams": list(self.streams),
            "max_age_ms": self.max_age_ms,
            "max_skew_ms": self.max_skew_ms,
            "anchor_timestamp_ns": self.anchor_timestamp_ns,
            "timestamp_basis": self.timestamp_basis.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SnapshotRequest":
        if payload.get("type") != "sonic.snapshot_request" or int(payload.get("version", -1)) != 1:
            raise ValueError("unsupported snapshot request")
        return cls(
            streams=tuple(str(stream) for stream in payload["streams"]),
            max_age_ms=float(payload["max_age_ms"]),
            max_skew_ms=float(payload["max_skew_ms"]),
            anchor_timestamp_ns=(
                None
                if payload.get("anchor_timestamp_ns") is None
                else int(payload["anchor_timestamp_ns"])
            ),
            timestamp_basis=TimestampBasis(str(payload.get("timestamp_basis", "receive"))),
        )


@dataclass(frozen=True)
class SensorSnapshot:
    complete: bool
    reason: str
    anchor_timestamp_ns: int | None
    timestamp_basis: TimestampBasis
    frames: Mapping[str, SharedMemoryFrame]
    skew_ms: float | None
    ages_ms: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sonic.sensor_snapshot",
            "version": 1,
            "complete": self.complete,
            "reason": self.reason,
            "anchor_timestamp_ns": self.anchor_timestamp_ns,
            "timestamp_basis": self.timestamp_basis.value,
            "frames": {name: frame.to_dict() for name, frame in self.frames.items()},
            "skew_ms": self.skew_ms,
            "ages_ms": dict(self.ages_ms),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SensorSnapshot":
        if payload.get("type") != "sonic.sensor_snapshot" or int(payload.get("version", -1)) != 1:
            raise ValueError("unsupported sensor snapshot")
        raw_frames = payload.get("frames", {})
        raw_ages = payload.get("ages_ms", {})
        if not isinstance(raw_frames, Mapping) or not isinstance(raw_ages, Mapping):
            raise ValueError("sensor snapshot frames and ages must be mappings")
        return cls(
            complete=bool(payload["complete"]),
            reason=str(payload.get("reason", "")),
            anchor_timestamp_ns=(
                None
                if payload.get("anchor_timestamp_ns") is None
                else int(payload["anchor_timestamp_ns"])
            ),
            timestamp_basis=TimestampBasis(str(payload["timestamp_basis"])),
            frames=MappingProxyType(
                {
                    str(name): SharedMemoryFrame.from_dict(frame)
                    for name, frame in raw_frames.items()
                }
            ),
            skew_ms=None if payload.get("skew_ms") is None else float(payload["skew_ms"]),
            ages_ms=MappingProxyType(
                {str(name): float(age) for name, age in raw_ages.items()}
            ),
        )


class SensorSnapshotStore:
    """Bounded frame history and deterministic synchronized snapshot selection."""

    def __init__(self, *, history_size: int = 64) -> None:
        if history_size < 2:
            raise ValueError("history_size must be at least two")
        self.history_size = int(history_size)
        self._frames: dict[str, deque[SharedMemoryFrame]] = {}
        self._history_sizes: dict[str, int] = {}

    def configure_stream(self, stream: str, *, history_size: int) -> None:
        if history_size < 2:
            raise ValueError("stream history_size must be at least two")
        limit = min(int(history_size), self.history_size)
        self._history_sizes[stream] = limit
        history = self._frames.get(stream)
        if history is not None and history.maxlen != limit:
            self._frames[stream] = deque(history, maxlen=limit)

    def add(self, frame: SharedMemoryFrame) -> None:
        history = self._frames.setdefault(
            frame.stream,
            deque(maxlen=self._history_sizes.get(frame.stream, self.history_size)),
        )
        if history and frame.metadata.timestamp_ns < history[-1].metadata.timestamp_ns:
            raise ValueError(f"receive timestamps moved backwards for stream {frame.stream!r}")
        history.append(frame)

    def discard_stream(self, stream: str) -> None:
        self._frames.pop(stream, None)

    @staticmethod
    def _timestamp(frame: SharedMemoryFrame, basis: TimestampBasis) -> int:
        if basis is TimestampBasis.RECEIVE:
            return frame.metadata.timestamp_ns
        return frame.source_timestamp_ns

    def select(self, request: SnapshotRequest, *, now_ns: int) -> SensorSnapshot:
        missing = [stream for stream in request.streams if not self._frames.get(stream)]
        if missing:
            return self._incomplete(
                request,
                reason=f"missing streams: {', '.join(missing)}",
            )

        anchor = request.anchor_timestamp_ns
        if anchor is None:
            anchor = min(
                self._timestamp(self._frames[stream][-1], request.timestamp_basis)
                for stream in request.streams
            )

        selected: dict[str, SharedMemoryFrame] = {}
        for stream in request.streams:
            selected[stream] = min(
                self._frames[stream],
                key=lambda frame: abs(self._timestamp(frame, request.timestamp_basis) - anchor),
            )

        if request.timestamp_basis is TimestampBasis.SOURCE:
            clocks = {frame.source_clock for frame in selected.values()}
            timestamps_valid = all(
                frame.source_timestamp_ns > 0 for frame in selected.values()
            )
            if "unknown" in clocks or len(clocks) != 1 or not timestamps_valid:
                return self._incomplete(
                    request,
                    reason="source clocks are unknown or incompatible",
                )

        ages_ms = {
            stream: max(0, int(now_ns) - frame.metadata.timestamp_ns) / 1_000_000.0
            for stream, frame in selected.items()
        }
        stale = [
            stream for stream, age_ms in ages_ms.items() if age_ms > request.max_age_ms
        ]
        timestamps = [
            self._timestamp(frame, request.timestamp_basis) for frame in selected.values()
        ]
        skew_timestamps = timestamps
        if request.anchor_timestamp_ns is not None:
            skew_timestamps = [*timestamps, anchor]
        skew_ms = (max(skew_timestamps) - min(skew_timestamps)) / 1_000_000.0
        if stale:
            return SensorSnapshot(
                complete=False,
                reason=f"stale streams: {', '.join(stale)}",
                anchor_timestamp_ns=anchor,
                timestamp_basis=request.timestamp_basis,
                frames=MappingProxyType(selected),
                skew_ms=skew_ms,
                ages_ms=MappingProxyType(ages_ms),
            )
        if skew_ms > request.max_skew_ms:
            return SensorSnapshot(
                complete=False,
                reason=f"snapshot skew {skew_ms:.3f}ms exceeds limit",
                anchor_timestamp_ns=anchor,
                timestamp_basis=request.timestamp_basis,
                frames=MappingProxyType(selected),
                skew_ms=skew_ms,
                ages_ms=MappingProxyType(ages_ms),
            )
        return SensorSnapshot(
            complete=True,
            reason="",
            anchor_timestamp_ns=anchor,
            timestamp_basis=request.timestamp_basis,
            frames=MappingProxyType(selected),
            skew_ms=skew_ms,
            ages_ms=MappingProxyType(ages_ms),
        )

    @staticmethod
    def _incomplete(request: SnapshotRequest, *, reason: str) -> SensorSnapshot:
        return SensorSnapshot(
            complete=False,
            reason=reason,
            anchor_timestamp_ns=request.anchor_timestamp_ns,
            timestamp_basis=request.timestamp_basis,
            frames=MappingProxyType({}),
            skew_ms=None,
            ages_ms=MappingProxyType({}),
        )
