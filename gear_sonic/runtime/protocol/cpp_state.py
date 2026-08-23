"""Shared decoder for untouched C++ g1_debug msgpack snapshots."""

from __future__ import annotations

from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np


def _convert_lists_to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _convert_lists_to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def decode_cpp_state_payload(payload: bytes) -> dict[str, Any]:
    """Decode one untouched C++ msgpack state payload."""
    decoded = msgpack.unpackb(payload, raw=False, object_hook=mnp.decode)
    if not isinstance(decoded, dict):
        raise ValueError("C++ state payload must decode to a mapping")
    return _convert_lists_to_numpy(decoded)


def decode_cpp_state_array(values: np.ndarray) -> dict[str, Any]:
    """Decode a C++ state payload materialized from SensorGateway."""
    payload = np.asarray(values, dtype=np.uint8).reshape(-1).tobytes()
    return decode_cpp_state_payload(payload)
