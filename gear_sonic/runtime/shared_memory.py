"""Race-detecting shared-memory ring buffers for high-bandwidth sensor arrays."""

from __future__ import annotations

from multiprocessing import resource_tracker, shared_memory
import struct
import threading
import uuid

import numpy as np

from gear_sonic.runtime.contracts import MessageMetadata, SharedMemoryFrame

_TOKEN_SIZE = struct.calcsize("<Q")
_OWNED_SHARED_MEMORY_NAMES: set[str] = set()


class FrameOverwrittenError(RuntimeError):
    """The requested ring slot was reused while or before it was read."""


class SharedMemoryRing:
    """Single-producer fixed-capacity ring with a seqlock token per slot."""

    def __init__(
        self,
        stream: str,
        *,
        slot_count: int,
        slot_size_bytes: int,
        name: str | None = None,
        source: str = "sensor_gateway",
        ttl_ms: int = 1000,
        initial_sequence: int = 0,
    ) -> None:
        if not stream:
            raise ValueError("stream cannot be empty")
        if slot_count < 2:
            raise ValueError("slot_count must be at least two")
        if slot_size_bytes <= 0:
            raise ValueError("slot_size_bytes must be positive")
        if ttl_ms < 0:
            raise ValueError("ttl_ms cannot be negative")
        if initial_sequence < 0:
            raise ValueError("initial_sequence cannot be negative")

        self.stream = stream
        self.slot_count = int(slot_count)
        self.slot_size_bytes = int(slot_size_bytes)
        self.source = source
        self.ttl_ms = int(ttl_ms)
        self.slot_stride_bytes = _TOKEN_SIZE + self.slot_size_bytes
        safe_stream = "".join(
            character if character.isalnum() or character in "-_." else "-"
            for character in stream
        )
        shared_name = name or f"sonic-{safe_stream}-{uuid.uuid4().hex[:12]}"
        self._memory = shared_memory.SharedMemory(
            name=shared_name,
            create=True,
            size=self.slot_count * self.slot_stride_bytes,
        )
        _OWNED_SHARED_MEMORY_NAMES.add(self._memory.name)
        self._sequence = int(initial_sequence)
        self._lock = threading.Lock()
        self._closed = False
        self._unlinked = False

    @property
    def name(self) -> str:
        return self._memory.name

    @property
    def next_sequence(self) -> int:
        return self._sequence

    def write(
        self,
        array: np.ndarray,
        *,
        received_ns: int,
        source_timestamp_ns: int = 0,
        source_clock: str = "unknown",
        generation: int = 0,
        attributes: dict[str, object] | None = None,
    ) -> SharedMemoryFrame:
        values = np.ascontiguousarray(array)
        if values.nbytes > self.slot_size_bytes:
            raise ValueError(
                f"frame requires {values.nbytes} bytes but slot capacity is "
                f"{self.slot_size_bytes} bytes"
            )
        if self._closed:
            raise RuntimeError("shared-memory ring is closed")

        with self._lock:
            sequence = self._sequence
            slot = sequence % self.slot_count
            slot_offset = slot * self.slot_stride_bytes
            data_offset = slot_offset + _TOKEN_SIZE
            writing_token = sequence * 2 + 1
            ready_token = writing_token + 1
            struct.pack_into("<Q", self._memory.buf, slot_offset, writing_token)
            self._memory.buf[data_offset : data_offset + values.nbytes] = values.tobytes(
                order="C"
            )
            struct.pack_into("<Q", self._memory.buf, slot_offset, ready_token)
            self._sequence += 1

        return SharedMemoryFrame(
            metadata=MessageMetadata(
                source=self.source,
                sequence=sequence,
                timestamp_ns=int(received_ns),
                ttl_ms=self.ttl_ms,
                generation=int(generation),
            ),
            stream=self.stream,
            shared_memory=self.name,
            shape=tuple(int(dimension) for dimension in values.shape),
            dtype=values.dtype.str,
            offset_bytes=data_offset,
            size_bytes=values.nbytes,
            source_timestamp_ns=int(source_timestamp_ns),
            source_clock=source_clock,
            attributes={} if attributes is None else dict(attributes),
        )

    def close(self) -> None:
        if not self._closed:
            self._memory.close()
            self._closed = True

    def unlink(self) -> None:
        if not self._unlinked:
            try:
                self._memory.unlink()
            except FileNotFoundError:
                # A crashed legacy reader may already have unlinked the name.
                # The producer's existing mapping remains valid until close().
                resource_tracker.unregister(self._memory._name, "shared_memory")
            _OWNED_SHARED_MEMORY_NAMES.discard(self.name)
            self._unlinked = True

    def shutdown(self) -> None:
        self.close()
        self.unlink()

    def __enter__(self) -> "SharedMemoryRing":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.shutdown()


def read_shared_memory_frame(frame: SharedMemoryFrame) -> np.ndarray:
    """Copy one frame and reject a slot that was concurrently overwritten."""

    memory = shared_memory.SharedMemory(name=frame.shared_memory, create=False)
    if frame.shared_memory not in _OWNED_SHARED_MEMORY_NAMES:
        # Python <=3.12 registers every SharedMemory attachment for unlink on
        # process exit. Readers do not own gateway memory, so unregister this
        # attachment while retaining ordinary close() semantics.
        resource_tracker.unregister(memory._name, "shared_memory")
    try:
        token_offset = frame.offset_bytes - _TOKEN_SIZE
        if token_offset < 0:
            raise ValueError("shared-memory frame offset is smaller than its token header")
        expected_token = frame.metadata.sequence * 2 + 2
        token_before = struct.unpack_from("<Q", memory.buf, token_offset)[0]
        if token_before != expected_token:
            raise FrameOverwrittenError(
                f"frame {frame.stream}:{frame.metadata.sequence} was overwritten"
            )
        dtype = np.dtype(frame.dtype)
        expected_size = int(np.prod(frame.shape, dtype=np.int64)) * dtype.itemsize
        if expected_size != frame.size_bytes:
            raise ValueError(
                f"frame metadata requires {expected_size} bytes, got {frame.size_bytes}"
            )
        end = frame.offset_bytes + frame.size_bytes
        if end > len(memory.buf):
            raise ValueError("shared-memory frame exceeds its backing allocation")
        values = np.ndarray(
            frame.shape,
            dtype=dtype,
            buffer=memory.buf,
            offset=frame.offset_bytes,
        ).copy()
        token_after = struct.unpack_from("<Q", memory.buf, token_offset)[0]
        if token_after != expected_token:
            raise FrameOverwrittenError(
                f"frame {frame.stream}:{frame.metadata.sequence} changed while reading"
            )
        return values
    finally:
        memory.close()
