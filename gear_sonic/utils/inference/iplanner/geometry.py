"""LaViRA G1's bottom-center waypoint and 30th-percentile depth selection."""

from __future__ import annotations

import math

import numpy as np

from gear_sonic.utils.inference.lavira.geometry import (
    FRAME_COUNT,
    POLICY_FRAME_INDEX,
    ObjectNavGeometryError,
    _validated_bbox,
)


def build_waypoint_geometry(policy, frames, *, max_direct_travel=15.0):
    if len(frames) != FRAME_COUNT:
        raise ObjectNavGeometryError("expected the existing aligned camera burst")
    # Use the depth paired with the image VA actually saw. Camera capture and
    # synchronization remain owned by the existing Groot camera implementation.
    depth_mm, fx, cx = frames[POLICY_FRAME_INDEX]
    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or not depth.size or not math.isfinite(fx) or fx <= 0:
        raise ObjectNavGeometryError("invalid waypoint depth or camera intrinsics")
    if not math.isfinite(cx):
        raise ObjectNavGeometryError("invalid camera principal point")
    x1, _, x2, y2 = _validated_bbox(policy)
    height, width = depth.shape
    u = min(int((x1 + x2) * 0.5 * width / 1000.0), width - 1)
    v = min(int(y2 * height / 1000.0), height - 1)
    patch = depth[max(0, v - 3):v + 4, max(0, u - 3):u + 4].astype(float) / 1000.0
    valid = patch[np.isfinite(patch) & (patch > 0.1)]
    if not valid.size:
        raise ObjectNavGeometryError("LaViRA waypoint has no valid aligned depth")
    forward = float(np.percentile(valid, 30))
    left = -(u - cx) * forward / fx
    if math.hypot(forward, left) > max_direct_travel:
        raise ObjectNavGeometryError("LaViRA waypoint exceeds the navigation travel limit")
    return {
        "goal_x": forward, "goal_y": left,
        "waypoint_pixel": [u, v], "depth_percentile": 30,
        "depth_m": forward, "policy_rgb_frame_index": POLICY_FRAME_INDEX + 1,
    }
