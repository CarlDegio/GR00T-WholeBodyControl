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


def build_command_message(start: bool, stop: bool, planner: bool) -> bytes:
    return pack_array_message(
        "command",
        {
            "start": np.asarray([start], dtype=np.uint8),
            "stop": np.asarray([stop], dtype=np.uint8),
            "planner": np.asarray([planner], dtype=np.uint8),
        },
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
    return pack_array_message("planner", fields, version=1)
