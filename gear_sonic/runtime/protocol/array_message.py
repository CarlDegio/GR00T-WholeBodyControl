"""Typed NumPy-array messages carried on prefixed ZMQ topics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

import numpy as np


HEADER_SIZE = 1280
_WIRE_DTYPES = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("?"),
}
_DTYPE_NAMES = {
    ("f", 4): "f32",
    ("f", 8): "f64",
    ("i", 4): "i32",
    ("i", 8): "i64",
    ("u", 1): "u8",
    ("b", 1): "bool",
}


@dataclass(frozen=True)
class DecodedArrayMessage:
    version: int
    endian: str
    count: int
    fields: dict[str, np.ndarray]


def pack_array_message(
    topic: str,
    fields: Mapping[str, np.ndarray],
    *,
    version: int,
    count: int = 1,
) -> bytes:
    """Pack ordered NumPy fields behind a topic and fixed-size JSON header."""
    topic_bytes = str(topic).encode("utf-8")
    if not topic_bytes:
        raise ValueError("array-message topic cannot be empty")

    descriptors = []
    payload = []
    for name, value in fields.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"array-message field {name!r} must be a NumPy array")
        dtype_name = _DTYPE_NAMES.get((value.dtype.kind, value.dtype.itemsize), "f32")
        array = np.asarray(value, dtype=_WIRE_DTYPES[dtype_name], order="C")
        descriptors.append(
            {"name": str(name), "dtype": dtype_name, "shape": list(array.shape)}
        )
        payload.append(array.tobytes())

    header = json.dumps(
        {
            "v": int(version),
            "endian": "le",
            "count": int(count),
            "fields": descriptors,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    if len(header) > HEADER_SIZE:
        raise ValueError(f"array-message header exceeds {HEADER_SIZE} bytes")
    return topic_bytes + header.ljust(HEADER_SIZE, b"\x00") + b"".join(payload)


def unpack_array_message(
    message: bytes,
    *,
    expected_topic: str,
) -> DecodedArrayMessage:
    """Validate and decode one fixed-header NumPy-array message."""
    topic = str(expected_topic).encode("utf-8")
    if not message.startswith(topic):
        raise ValueError(f"message does not start with expected topic {expected_topic!r}")
    payload_start = len(topic) + HEADER_SIZE
    if len(message) < payload_start:
        raise ValueError("array message is shorter than its header")

    raw_header = message[len(topic):payload_start].split(b"\x00", 1)[0]
    header = json.loads(raw_header.decode("utf-8"))
    endian = str(header.get("endian", "le"))
    if endian not in {"le", "be"}:
        raise ValueError(f"unsupported array-message endian {endian!r}")
    byte_order = "<" if endian == "le" else ">"

    fields: dict[str, np.ndarray] = {}
    offset = payload_start
    for descriptor in header.get("fields", []):
        name = str(descriptor["name"])
        dtype_name = str(descriptor["dtype"])
        base_dtype = _WIRE_DTYPES.get(dtype_name)
        if base_dtype is None:
            raise ValueError(f"unsupported array-message dtype {dtype_name!r}")
        dtype = base_dtype.newbyteorder(byte_order)
        shape = tuple(int(dimension) for dimension in descriptor["shape"])
        if any(dimension < 0 for dimension in shape):
            raise ValueError(f"array-message field {name!r} has a negative shape")
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        end = offset + size
        if end > len(message):
            raise ValueError(f"array-message field {name!r} exceeds payload")
        fields[name] = (
            np.frombuffer(message[offset:end], dtype=dtype)
            .reshape(shape)
            .astype(dtype.newbyteorder("="), copy=True)
        )
        offset = end

    return DecodedArrayMessage(
        version=int(header.get("v", 1)),
        endian=endian,
        count=int(header.get("count", 1)),
        fields=fields,
    )
