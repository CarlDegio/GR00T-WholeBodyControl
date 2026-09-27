import numpy as np
import pytest

from gear_sonic.runtime.protocol.cpp_control import build_planner_message
from gear_sonic.runtime.protocol.array_message import unpack_array_message
from gear_sonic.utils.inference.vla.poses import (
    SONIC_STAND_UPPER_BODY_RAD, VLA_INITIAL_UPPER_BODY_RAD, sample_upper_body_transition,
    measured_upper_body_from_state, UPPER_BODY_MUJOCO_INDICES,
)


def test_entry_uses_absolute_measured_joint_angles_in_planner_order():
    measured = np.arange(29, dtype=np.float32) * .01
    state = {"body_q": np.full(29, -99), "body_q_measured": measured}
    actual = measured_upper_body_from_state(state)
    np.testing.assert_array_equal(actual, measured[UPPER_BODY_MUJOCO_INDICES])
    with pytest.raises(ValueError, match="body_q_measured"):
        measured_upper_body_from_state({"body_q": measured})
    with pytest.raises(ValueError, match="finite"):
        measured_upper_body_from_state({"body_q_measured": [float('nan')]*29})


def test_pose_entry_moves_from_planner_stance_with_zero_endpoint_velocity():
    start = np.asarray(SONIC_STAND_UPPER_BODY_RAD)
    end = np.asarray(VLA_INITIAL_UPPER_BODY_RAD)
    first, first_v = sample_upper_body_transition(start, end, 0, 3)
    last, last_v = sample_upper_body_transition(start, end, 3, 3)
    np.testing.assert_allclose(first, start)
    np.testing.assert_allclose(last, end, atol=1e-7)
    np.testing.assert_array_equal(first_v, 0)
    np.testing.assert_array_equal(last_v, 0)
    q, dq = sample_upper_body_transition(start, end, 1.5, 3)
    np.testing.assert_allclose(q, (start+end)/2, atol=1e-7)
    h = .01
    plus, _ = sample_upper_body_transition(start, end, 1.5+h, 3)
    minus, _ = sample_upper_body_transition(start, end, 1.5-h, 3)
    np.testing.assert_allclose(dq, (plus-minus)/(2*h), atol=1e-5)


def test_transition_flag_is_explicit_and_round_trips():
    args = dict(mode=0, movement=[0,0,0], facing=[1,0,0],
                upper_body_position=SONIC_STAND_UPPER_BODY_RAD, upper_body_velocity=[0]*17)
    normal = unpack_array_message(build_planner_message(**args), expected_topic="planner").fields
    assert "pose_entry_transition" not in normal
    entry = unpack_array_message(build_planner_message(**args, pose_entry_transition=True), expected_topic="planner").fields
    assert entry["pose_entry_transition"].dtype == np.uint8
    assert entry["pose_entry_transition"].tolist() == [1]
    np.testing.assert_allclose(entry["upper_body_position"], SONIC_STAND_UPPER_BODY_RAD)


def test_transition_cannot_be_used_for_locomotion_or_without_targets():
    with pytest.raises(ValueError):
        build_planner_message(0,[0,0,0],[1,0,0],pose_entry_transition=True)
    with pytest.raises(ValueError):
        build_planner_message(1,[1,0,0],[1,0,0],upper_body_position=[0]*17,
                              upper_body_velocity=[0]*17,pose_entry_transition=True)
