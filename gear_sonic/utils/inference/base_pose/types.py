"""RGB-D contracts shared by the hardware runtime and simulation adapters.

Importing these types does not initialize a camera, transport, or robot runtime.
The hardware calibration still requires uint16 depth; metric simulation depth is
validated by the simulation calibration implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BasePoseCameraError(RuntimeError):
    """Raised when a base-pose observation is unavailable or malformed."""


@dataclass(frozen=True)
class AlignedRGBDSnapshot:
    rgb: np.ndarray
    depth_raw: np.ndarray | None
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m: float | None
    depth_aligned_to: str | None
    depth_source: str | None
    timestamp: float
