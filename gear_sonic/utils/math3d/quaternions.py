"""Quaternion helpers with explicit component-order contracts."""

from __future__ import annotations

import math
from typing import Any


_TWO_PI = 2.0 * math.pi


def _normalized_components(
    quaternion: Any, *, min_norm: float
) -> tuple[float, float, float, float]:
    if isinstance(quaternion, (str, bytes)):
        raise ValueError("quaternion must contain four finite values")
    try:
        values = tuple(float(component) for component in quaternion)
    except (TypeError, ValueError) as exc:
        raise ValueError("quaternion must contain four finite values") from exc
    if len(values) != 4 or not all(math.isfinite(component) for component in values):
        raise ValueError("quaternion must contain four finite values")
    norm = math.sqrt(sum(component * component for component in values))
    if norm <= min_norm:
        raise ValueError("quaternion norm must be positive")
    return tuple(component / norm for component in values)


def _yaw_from_wxyz_components(w: float, x: float, y: float, z: float) -> float:
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return math.remainder(yaw, _TWO_PI)


def yaw_from_quaternion_wxyz(
    quaternion: Any, *, min_norm: float = 1.0e-12
) -> float:
    """Return wrapped yaw from a scalar-first ``(w, x, y, z)`` quaternion."""
    w, x, y, z = _normalized_components(quaternion, min_norm=min_norm)
    return _yaw_from_wxyz_components(w, x, y, z)


def yaw_from_quaternion_xyzw(
    quaternion: Any, *, min_norm: float = 1.0e-12
) -> float:
    """Return wrapped yaw from a scalar-last ``(x, y, z, w)`` quaternion."""
    x, y, z, w = _normalized_components(quaternion, min_norm=min_norm)
    return _yaw_from_wxyz_components(w, x, y, z)
