"""Shared scalar and array geometry primitives."""

from gear_sonic.utils.math3d.orientation import compute_projected_gravity
from gear_sonic.utils.math3d.quaternions import (
    yaw_from_quaternion_wxyz,
    yaw_from_quaternion_xyzw,
)

__all__ = [
    "compute_projected_gravity",
    "yaw_from_quaternion_wxyz",
    "yaw_from_quaternion_xyzw",
]
