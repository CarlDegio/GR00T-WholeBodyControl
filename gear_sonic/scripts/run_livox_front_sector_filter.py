#!/usr/bin/env python3
"""Filter the robot-forward sector from Livox CustomMsg point clouds."""

from __future__ import annotations

import math
from typing import Any, Sequence


def _validate_sector_degrees(sector_degrees: float) -> float:
    value = float(sector_degrees)
    if not math.isfinite(value) or not 0.0 < value < 180.0:
        raise ValueError("sector_degrees must be finite and between 0 and 180")
    return value


def is_inside_forward_sector(
    point: Any,
    sector_degrees: float = 90.0,
) -> bool:
    sector = _validate_sector_degrees(sector_degrees)
    x = float(point.x)
    y = float(point.y)
    if x <= 0.0:
        return False
    if sector == 90.0:
        return abs(y) <= x
    return abs(math.degrees(math.atan2(y, x))) <= sector / 2.0


def retain_points_outside_forward_sector(
    points: Sequence[Any],
    sector_degrees: float = 90.0,
) -> list[Any]:
    sector = _validate_sector_degrees(sector_degrees)
    return [
        point
        for point in points
        if not is_inside_forward_sector(point, sector)
    ]
