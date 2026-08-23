"""Pose and VLA action contracts for the C++ deploy ZMQ input."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from gear_sonic.runtime.protocol.array_message import pack_array_message


def pack_pose_message(
    pose_data: Mapping[str, np.ndarray],
    topic: str = "pose",
    version: int = 3,
) -> bytes:
    return pack_array_message(topic, pose_data, version=version)


def _hand_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {result.shape}")
    return result


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
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    joint_vel = np.asarray(joint_vel, dtype=np.float32)
    body_quat_w = np.asarray(body_quat_w, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)
    if joint_pos.ndim != 2 or joint_pos.shape[1:] != (29,):
        raise ValueError(f"joint_pos must have shape [N, 29], got {joint_pos.shape}")
    if joint_vel.shape != joint_pos.shape:
        raise ValueError(
            f"joint_vel must have shape {joint_pos.shape}, got {joint_vel.shape}"
        )
    if body_quat_w.shape != (joint_pos.shape[0], 4):
        raise ValueError(
            f"body_quat_w must have shape {(joint_pos.shape[0], 4)}, got {body_quat_w.shape}"
        )
    if frame_index.shape != (joint_pos.shape[0],):
        raise ValueError(
            f"frame_index must have shape {(joint_pos.shape[0],)}, got {frame_index.shape}"
        )

    fields = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_quat_w": body_quat_w,
        "frame_index": frame_index,
        "catch_up": np.asarray([catch_up], dtype=np.uint8),
    }
    if left_hand_joints is not None:
        fields["left_hand_joints"] = _hand_vector(
            left_hand_joints, name="left_hand_joints"
        )
    if right_hand_joints is not None:
        fields["right_hand_joints"] = _hand_vector(
            right_hand_joints, name="right_hand_joints"
        )
    return pack_array_message(topic, fields, version=1)


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray | None = None,
    right_hand_joints: np.ndarray | None = None,
) -> bytes:
    motion_token = np.asarray(motion_token, dtype=np.float32)
    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    frame_index = np.asarray(frame_index, dtype=np.int64)
    if frame_index.ndim == 0:
        frame_index = frame_index.reshape(1)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    fields = {"token_state": motion_token, "frame_index": frame_index}
    for name, value in (
        ("left_hand_joints", left_hand_joints),
        ("right_hand_joints", right_hand_joints),
    ):
        if value is None:
            continue
        hand = np.asarray(value, dtype=np.float32)
        if hand.ndim == 1:
            hand = _hand_vector(hand, name=name).reshape(1, 7)
        fields[name] = hand
    return pack_array_message("pose", fields, version=4)
