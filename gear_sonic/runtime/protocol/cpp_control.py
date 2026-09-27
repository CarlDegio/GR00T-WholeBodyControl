"""C++ deploy command and planner topic contracts."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gear_sonic.runtime.protocol.array_message import pack_array_message


def _vector(values: Sequence[float], *, name: str, size: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).reshape(-1)
    if size is not None and result.size != size:
        raise ValueError(f"{name} must have length {size}")
    return result


def build_command_message(start: bool, stop: bool, planner: bool, *, request_id: int = 0,
                          hold_current_upper_body: bool = False) -> bytes:
    if not isinstance(request_id, (int, np.integer)) or not 0 <= request_id < 2**53:
        raise ValueError("C++ request_id must be an exactly representable nonnegative integer")
    extra = {}
    if request_id:
        extra["control_request_id"] = np.asarray([request_id], dtype=np.float64)
        extra["hold_current_upper_body"] = np.asarray([hold_current_upper_body], dtype=np.uint8)
    return pack_array_message(
        "command",
        {
            **extra,
            "start": np.asarray([start], dtype=np.uint8),
            "stop": np.asarray([stop], dtype=np.uint8),
            "planner": np.asarray([planner], dtype=np.uint8),
        },
        version=1,
    )


def build_gripper_command(left_closed: bool, right_closed: bool) -> bytes:
    """Latch manual Dex1 targets without starting or changing body control."""
    return pack_array_message(
        "command",
        {"manual_grippers": np.asarray([left_closed, right_closed], dtype=np.uint8)},
        version=1,
    )


def build_gripper_release_command() -> bytes:
    """Release demo manual targets so the POSE hand stream controls the grippers."""
    return pack_array_message(
        "command",
        {"release_manual_grippers": np.asarray([1], dtype=np.uint8)},
        version=1,
    )


def build_planner_message(
    mode: int,
    movement: Sequence[float],
    facing: Sequence[float],
    speed: float = -1.0,
    height: float = -1.0,
    upper_body_position: Sequence[float] | None = None,
    upper_body_velocity: Sequence[float] | None = None,
    left_hand_position: Sequence[float] | None = None,
    right_hand_position: Sequence[float] | None = None,
    vr_3pt_position: Sequence[float] | None = None,
    vr_3pt_orientation: Sequence[float] | None = None,
    pose_entry_transition: bool = False,
) -> bytes:
    fields = {
        "mode": np.asarray([mode], dtype=np.int32),
        "movement": _vector(movement, name="movement", size=3),
        "facing": _vector(facing, name="facing", size=3),
        "speed": np.asarray([speed], dtype=np.float32),
        "height": np.asarray([height], dtype=np.float32),
    }
    optional = (
        ("upper_body_position", upper_body_position),
        ("upper_body_velocity", upper_body_velocity),
        ("left_hand_joints", left_hand_position),
        ("right_hand_joints", right_hand_position),
        ("vr_position", vr_3pt_position),
        ("vr_orientation", vr_3pt_orientation),
    )
    for name, values in optional:
        if values is not None:
            fields[name] = _vector(values, name=name)
    if pose_entry_transition:
        if upper_body_position is None or upper_body_velocity is None:
            raise ValueError("POSE entry requires upper-body position and velocity targets")
        if mode != 0 or np.any(fields["movement"] != 0):
            raise ValueError("POSE entry requires stationary planner commands")
        fields["pose_entry_transition"] = np.asarray([1], dtype=np.uint8)
    return pack_array_message("planner", fields, version=1)
