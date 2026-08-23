"""Neutral RGB-D materialization shared by inference consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class MaterializedRGBD:
    rgb: np.ndarray
    depth_raw: np.ndarray | None
    camera_info: Mapping[str, Any]
    source_timestamp_ns: int
    depth_source: str | None


def materialize_rgbd(
    snapshot: Any,
    *,
    rgb_stream: str,
    depth_stream: str | None = None,
    require_depth: bool = True,
    prefer_depth_info: bool = True,
    prefer_depth_timestamp: bool = True,
    validate_dtypes: bool = True,
    error_type: type[Exception] = ValueError,
    rgb_label: str = "Gateway RGB",
    depth_label: str = "Gateway depth",
    mismatch_message: str = "Gateway RGB and depth shapes do not match",
) -> MaterializedRGBD:
    """Copy and structurally validate arrays from a gateway snapshot.

    Alignment, calibration, ownership, generation, and unit checks remain in
    the inference service that owns those business contracts.
    """
    rgb_frame = snapshot.snapshot.frames[rgb_stream]
    rgb = np.asarray(snapshot.arrays[rgb_stream])
    invalid_rgb = rgb.ndim != 3 or rgb.shape[2] != 3
    if validate_dtypes:
        invalid_rgb = invalid_rgb or rgb.dtype != np.uint8
    if invalid_rgb:
        expected = "HxWx3 uint8" if validate_dtypes else "HxWx3"
        raise error_type(f"{rgb_label} must be {expected}, got {rgb.shape} {rgb.dtype}")

    info = dict(rgb_frame.attributes.get("camera_info", {}))
    timestamp_ns = int(rgb_frame.source_timestamp_ns)
    depth_raw: np.ndarray | None = None
    depth_source: str | None = None
    if require_depth:
        if depth_stream is None:
            raise ValueError("depth_stream is required when require_depth is true")
        depth_frame = snapshot.snapshot.frames[depth_stream]
        depth_raw = np.asarray(snapshot.arrays[depth_stream])
        invalid_depth = depth_raw.ndim != 2
        if validate_dtypes:
            invalid_depth = invalid_depth or depth_raw.dtype != np.uint16
        if invalid_depth:
            expected = "HxW uint16" if validate_dtypes else "HxW"
            raise error_type(
                f"{depth_label} must be {expected}, got "
                f"{depth_raw.shape} {depth_raw.dtype}"
            )
        if rgb.shape[:2] != depth_raw.shape:
            raise error_type(mismatch_message)
        depth_info = dict(depth_frame.attributes.get("camera_info", {}))
        if prefer_depth_info and depth_info:
            info = depth_info
        depth_source = str(
            depth_frame.attributes.get("depth_source")
            or info.get("depth_source", "")
        ) or None
        depth_timestamp_ns = int(depth_frame.source_timestamp_ns)
        if prefer_depth_timestamp:
            timestamp_ns = depth_timestamp_ns or timestamp_ns
        else:
            timestamp_ns = timestamp_ns or depth_timestamp_ns

    return MaterializedRGBD(
        rgb=rgb.copy(),
        depth_raw=None if depth_raw is None else depth_raw.copy(),
        camera_info=info,
        source_timestamp_ns=timestamp_ns,
        depth_source=depth_source,
    )
