"""Shared orientation-derived vectors with explicit quaternion contracts."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_projected_gravity(base_quat: np.ndarray) -> np.ndarray:
    """Compute body-frame gravity from a scalar-first ``wxyz`` quaternion."""
    base_quat = np.asarray(base_quat, dtype=np.float64)
    if base_quat.shape != (4,):
        raise ValueError(f"base_quat must have shape (4,), got {base_quat.shape}")

    gravity_vec_world = np.array([0.0, 0.0, -1.0])
    base_rotation = R.from_quat(base_quat, scalar_first=True)
    projected_gravity = base_rotation.inv().apply(gravity_vec_world)

    return projected_gravity.astype(np.float32)
