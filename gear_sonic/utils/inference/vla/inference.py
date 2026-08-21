"""Utility functions for VLA inference.

Includes action processing, observation preparation, latency compensation,
and inference scheduling logic.
"""

import numpy as np

from gear_sonic.data.robot_model.robot_model import RobotModel


def prepare_observation_for_eval(robot_model: RobotModel, obs: dict) -> dict:
    """Split whole-body ``q`` into per-joint-group state keys for the policy.

    Populates ``obs["state"]`` with ``left_arm``, ``right_arm``, ``waist``,
    ``left_leg``, ``right_leg``, ``left_hand``, ``right_hand`` sub-keys
    using the nested dictionary format expected by the OpenPI policy bridge.

    Args:
        robot_model: RobotModel instance.
        obs: Observation dict containing ``"q"`` key and a ``"state"`` sub-dict.

    Returns:
        Modified observation dict with ``obs["state"]`` populated.
    """
    assert "q" in obs, "q is not in the observation"

    whole_q = obs["q"]
    assert whole_q.shape[-1] == robot_model.num_joints, "q has wrong shape"

    if "state" not in obs:
        obs["state"] = {}

    obs["state"]["left_arm"] = whole_q[..., robot_model.get_joint_group_indices("left_arm")]
    obs["state"]["right_arm"] = whole_q[..., robot_model.get_joint_group_indices("right_arm")]
    obs["state"]["waist"] = whole_q[..., robot_model.get_joint_group_indices("waist")]
    obs["state"]["left_leg"] = whole_q[..., robot_model.get_joint_group_indices("left_leg")]
    obs["state"]["right_leg"] = whole_q[..., robot_model.get_joint_group_indices("right_leg")]
    obs["state"]["left_hand"] = whole_q[..., robot_model.get_joint_group_indices("left_hand")]
    obs["state"]["right_hand"] = whole_q[..., robot_model.get_joint_group_indices("right_hand")]

    return obs


def calculate_latency_compensated_index(
    inference_delay: float, control_freq: float, action_horizon: int
) -> int:
    """Calculate the starting action index compensating for inference latency.

    When inference completes, some time has elapsed, so we skip the first few
    actions that are now "stale" and start from a later index in the chunk.

    Args:
        inference_delay: Time elapsed since inference started (seconds).
        control_freq: Control loop frequency (Hz), e.g. 20.
        action_horizon: Total number of actions in the chunk, e.g. 16.

    Returns:
        Starting index (0 to action_horizon-1) for the action chunk.
    """
    raw_index = np.round(inference_delay * control_freq)
    return int(np.clip(raw_index, 0, action_horizon - 1))
