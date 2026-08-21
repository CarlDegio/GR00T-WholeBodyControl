"""Shared last-mile safety filters for every planner velocity source."""

from __future__ import annotations

import numpy as np


def depth_requires_stop(
    depth_m: np.ndarray,
    *,
    stop_distance_m: float = 0.10,
    min_area_pixels: int = 2000,
) -> bool:
    mask = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m < stop_distance_m)
    try:
        import cv2

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        return bool(
            count > 1 and int(stats[1:, cv2.CC_STAT_AREA].max()) > min_area_pixels
        )
    except ImportError:
        return int(mask.sum()) > min_area_pixels
