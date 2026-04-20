"""Builders for ZMQ wire-format messages on the 'command', 'planner', and 'pose' topics.

Message layout: [topic_bytes][1024-byte JSON header][packed binary payload].
The header describes field names, dtypes, and shapes so the receiver can
deserialize without out-of-band schema knowledge.
"""

import json
import struct
from typing import Sequence

import numpy as np

HEADER_SIZE = 1280


def _build_header(fields: list, version: int = 1, count: int = 1) -> bytes:
    header = {
        "v": version,
        "endian": "le",
        "count": count,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def build_command_message(
    start: bool, stop: bool, planner: bool, delta_heading: float | None = None
) -> bytes:
    """
    Assemble a 'command' topic message:
      - start: u8 (1=start control)
      - stop: u8 (1=stop control)
      - planner: u8 (1=planner mode, 0=streamed motion)
      - delta_heading: f32 (optional, yaw relative to heading command in radians)
    Returns: bytes ready to send via socket.send()
    """
    fields = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = b"".join(
        (
            struct.pack("B", 1 if start else 0),
            struct.pack("B", 1 if stop else 0),
            struct.pack("B", 1 if planner else 0),
        )
    )

    if delta_heading is not None:
        # Append delta_heading field to header and payload
        fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        payload += struct.pack("<f", float(delta_heading))

    header = _build_header(fields, version=1, count=1)

    return b"command" + header + payload


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
    vr_3pt_compliance: Sequence[float] | None = None,
) -> bytes:
    """
    Assemble a 'planner' topic message:
      - mode: i32 (LocomotionMode enum)
      - movement: f32[3] (x,y,z)
      - facing: f32[3] (x,y,z)
      - speed: f32 (optional, -1 for default)
      - height: f32 (optional, -1 for default)
    Returns: bytes ready to send via socket.send()
    """
    if len(movement) != 3:
        raise ValueError("movement must have length 3")
    if len(facing) != 3:
        raise ValueError("facing must have length 3")

    fields = [
        {"name": "mode", "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing", "dtype": "f32", "shape": [3]},
        {"name": "speed", "dtype": "f32", "shape": [1]},
        {"name": "height", "dtype": "f32", "shape": [1]},
    ]

    payload = b"".join(
        (
            struct.pack("<i", int(mode)),
            struct.pack("<fff", float(movement[0]), float(movement[1]), float(movement[2])),
            struct.pack("<fff", float(facing[0]), float(facing[1]), float(facing[2])),
            struct.pack("<f", float(speed)),
            struct.pack("<f", float(height)),
        )
    )

    # Add upper body position and velocity to payload, optionally
    if upper_body_position is not None:
        fields.append(
            {"name": "upper_body_position", "dtype": "f32", "shape": [len(upper_body_position)]}
        )
        for value in upper_body_position:
            payload += struct.pack("<f", float(value))

    if upper_body_velocity is not None:
        fields.append(
            {"name": "upper_body_velocity", "dtype": "f32", "shape": [len(upper_body_velocity)]}
        )
        for value in upper_body_velocity:
            payload += struct.pack("<f", float(value))

    if left_hand_position is not None:
        fields.append(
            {"name": "left_hand_joints", "dtype": "f32", "shape": [len(left_hand_position)]}
        )
        for value in left_hand_position:
            payload += struct.pack("<f", float(value))

    if right_hand_position is not None:
        fields.append(
            {"name": "right_hand_joints", "dtype": "f32", "shape": [len(right_hand_position)]}
        )
        for value in right_hand_position:
            payload += struct.pack("<f", float(value))

    if vr_3pt_position is not None:
        fields.append({"name": "vr_position", "dtype": "f32", "shape": [len(vr_3pt_position)]})
        for value in vr_3pt_position:
            payload += struct.pack("<f", float(value))

    if vr_3pt_orientation is not None:
        fields.append(
            {"name": "vr_orientation", "dtype": "f32", "shape": [len(vr_3pt_orientation)]}
        )
        for value in vr_3pt_orientation:
            payload += struct.pack("<f", float(value))

    if vr_3pt_compliance is not None:
        fields.append({"name": "vr_compliance", "dtype": "f32", "shape": [len(vr_3pt_compliance)]})
        for value in vr_3pt_compliance:
            payload += struct.pack("<f", float(value))

    header = _build_header(fields, version=1, count=1)

    return b"planner" + header + payload


def pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 3) -> bytes:
    """
    Pack pose/action data into ZMQ message format:
    [topic_prefix][1024-byte JSON header][concatenated binary fields]

    This is a general-purpose function for packing numpy arrays into ZMQ messages.
    Supports protocol versions 3 and 4.

    Args:
        pose_data: Dictionary containing numpy arrays to send
        topic: Topic prefix string (default: "pose")
        version: Protocol version (default: 3). Version 4 includes "count" field.

    Returns:
        Packed message as bytes

    Example:
        >>> data = {
        ...     "token_state": np.array([1.0, 2.0], dtype=np.float32),
        ...     "frame_index": np.array([0], dtype=np.int64)
        ... }
        >>> msg = pack_pose_message(data, topic="pose", version=4)
    """
    # Build fields list from pose_data
    fields = []
    binary_data = []

    for key, value in pose_data.items():
        if isinstance(value, np.ndarray):
            # Determine dtype string
            if value.dtype == np.float32:
                dtype_str = "f32"
            elif value.dtype == np.float64:
                dtype_str = "f64"
            elif value.dtype == np.int32:
                dtype_str = "i32"
            elif value.dtype == np.int64:
                dtype_str = "i64"
            elif value.dtype == np.uint8:
                dtype_str = "u8"
            elif value.dtype == bool:
                dtype_str = "bool"
            else:
                # Default to f32, cast if needed
                dtype_str = "f32"
                value = value.astype(np.float32)

            fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})

            # Ensure contiguous and little-endian
            if not value.flags["C_CONTIGUOUS"]:
                value = np.ascontiguousarray(value)
            if value.dtype.byteorder == ">":
                value = value.astype(value.dtype.newbyteorder("<"))

            binary_data.append(value.tobytes())

    # Build header using common utility
    header_bytes = _build_header(fields, version=version, count=1)

    # Pack message: [topic][1024-byte header][binary data]
    topic_bytes = topic.encode("utf-8")
    data_bytes = b"".join(binary_data)

    packed_message = topic_bytes + header_bytes + data_bytes
    return packed_message


def pack_pose_v1_message(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_quat_w: np.ndarray,
    frame_index: np.ndarray,
    *,
    catch_up: bool = False,
    left_hand_joints: np.ndarray | None = None,
    right_hand_joints: np.ndarray | None = None,
    topic: str = "pose",
) -> bytes:
    """
    Pack a Protocol v1 joint-motion message for the `pose` topic.

    Args:
        joint_pos: `[N, 29]` joint positions in IsaacLab order.
        joint_vel: `[N, 29]` joint velocities in IsaacLab order.
        body_quat_w: `[N, 4]` body quaternion(s) in wxyz order.
        frame_index: `[N]` monotonically increasing frame indices.
        catch_up: Whether deploy-side catch-up reset is enabled.
        left_hand_joints: Optional left-hand 7-DOF joint vector.
        right_hand_joints: Optional right-hand 7-DOF joint vector.
        topic: Topic prefix string.

    Returns:
        Packed single-part ZMQ payload.
    """
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    joint_vel = np.asarray(joint_vel, dtype=np.float32)
    body_quat_w = np.asarray(body_quat_w, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)
    catch_up_value = np.asarray([catch_up], dtype=np.uint8)

    if joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must have shape [N, 29], got {joint_pos.shape}")
    if joint_vel.ndim != 2:
        raise ValueError(f"joint_vel must have shape [N, 29], got {joint_vel.shape}")
    if joint_pos.shape != joint_vel.shape:
        raise ValueError(
            f"joint_pos and joint_vel must match, got {joint_pos.shape} and {joint_vel.shape}"
        )
    if joint_pos.shape[1] != 29:
        raise ValueError(f"Protocol v1 expects 29 body joints, got {joint_pos.shape[1]}")
    if body_quat_w.shape != (joint_pos.shape[0], 4):
        raise ValueError(
            f"body_quat_w must have shape {(joint_pos.shape[0], 4)}, got {body_quat_w.shape}"
        )
    if frame_index.shape != (joint_pos.shape[0],):
        raise ValueError(
            f"frame_index must have shape {(joint_pos.shape[0],)}, got {frame_index.shape}"
        )

    pose_data = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_quat_w": body_quat_w,
        "frame_index": frame_index,
        "catch_up": catch_up_value,
    }
    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32).reshape(-1)
        if left_hand_joints.shape != (7,):
            raise ValueError(
                f"left_hand_joints must have shape (7,), got {left_hand_joints.shape}"
            )
        pose_data["left_hand_joints"] = left_hand_joints
    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32).reshape(-1)
        if right_hand_joints.shape != (7,):
            raise ValueError(
                f"right_hand_joints must have shape (7,), got {right_hand_joints.shape}"
            )
        pose_data["right_hand_joints"] = right_hand_joints
    return pack_pose_message(pose_data, topic=topic, version=1)
