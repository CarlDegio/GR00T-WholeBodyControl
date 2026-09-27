"""Manual gripper routing/state tests, without robot sockets."""
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.runtime.gateway.control import (
    ControlGatewayCore, OperatorConsoleRouter, NAVIGATION_KEYS,
)
from gear_sonic.runtime.protocol.cpp_control import build_gripper_command
from gear_sonic.runtime.protocol.array_message import unpack_array_message
from gear_sonic.utils.inference.vla.runtime import (
    _VlaRuntimeState, _VlaCommandHandler, _publish_cached_action, _publish_manual_grippers,
)


@pytest.mark.parametrize('mode', ['POSE', 'PLANNER'])
def test_new_key_is_available_in_both_modes(mode):
    router = OperatorConsoleRouter()
    router.control_mode = mode
    assert 'j' not in NAVIGATION_KEYS
    assert router.accept_line('j', core=ControlGatewayCore()).command.name == 'toggle_grippers'


@pytest.mark.parametrize('left,right', [(False, False), (True, False), (False, True), (True, True)])
def test_switching_modes_preserves_each_hand(left, right):
    state = _VlaRuntimeState(cpp_mode='PLANNER', cpp_loop_running=True,
                            initial_pose_left_hand_closed=left,
                            initial_pose_right_hand_closed=right)
    def switch(*, start, planner):
        state.cpp_mode = 'PLANNER' if planner else 'POSE'
        return True
    handler = SimpleNamespace(state=state, invalidate_inference=lambda *_: None,
                              publish_initial_pose=lambda: True,
                              send_cpp_control_command=switch,
                              activate_vla_metrics=lambda: None, record_event=lambda *a: None)
    for method in (_VlaCommandHandler._select_pose_mode, _VlaCommandHandler._select_planner_mode):
        method(handler, None)
        assert (state.initial_pose_left_hand_closed, state.initial_pose_right_hand_closed) == (left, right)


def test_default_open_and_manual_toggle_does_not_need_active_policy():
    state = _VlaRuntimeState()
    handler = SimpleNamespace(state=state)
    assert state.pause_loop
    for expected in (True, False):
        _VlaCommandHandler._toggle_grippers(handler, None)
        packet = unpack_array_message(build_gripper_command(
            state.initial_pose_left_hand_closed, state.initial_pose_right_hand_closed
        ), expected_topic='command')
        assert packet.fields['manual_grippers'].tolist() == [int(expected)] * 2
        assert set(packet.fields) == {'manual_grippers'}  # Cannot start or switch body control.
    _VlaCommandHandler._toggle_left_hand(handler, None)
    assert state.initial_pose_left_hand_closed and not state.initial_pose_right_hand_closed
    _VlaCommandHandler._toggle_grippers(handler, None)
    assert state.initial_pose_left_hand_closed and state.initial_pose_right_hand_closed


def _policy_state():
    return _VlaRuntimeState(
        cpp_mode='POSE', cpp_loop_running=True, pause_loop=False,
        cached_action_chunk={
            'motion_token': np.zeros((1, 2, 64), dtype=np.float32),
            'left_hand_joints': np.array([[[0.83, 0, 0, 0, 0, 0, 0]] * 2]),
            'right_hand_joints': np.array([[[0.12, 0, 0, 0, 0, 0, 0]] * 2]),
        },
    )


def test_policy_handover_pause_and_manual_takeover():
    state = _policy_state()
    packets = []
    socket = SimpleNamespace(send=packets.append)
    config = SimpleNamespace(action_horizon=2)
    _publish_manual_grippers(state, socket)
    assert unpack_array_message(packets.pop(), expected_topic='command').fields['manual_grippers'].tolist() == [0, 0]

    assert _publish_cached_action(state, config=config, zmq_socket=socket)
    pose = unpack_array_message(packets[0], expected_topic='pose').fields
    np.testing.assert_allclose(pose['left_hand_joints'][0, 0], 0.83)
    release = unpack_array_message(packets[1], expected_topic='command').fields
    assert set(release) == {'release_manual_grippers'}
    assert release['release_manual_grippers'].tolist() == [1]
    assert state.grippers_follow_policy
    assert state.initial_pose_left_hand_closed and not state.initial_pose_right_hand_closed

    packets.clear()
    _publish_manual_grippers(state, socket)
    assert not packets

    def invalidate(_reason):
        state.cached_action_chunk = None
    handler = SimpleNamespace(state=state, invalidate_inference=invalidate,
                              record_event=lambda *a, **kw: None, activate_vla_metrics=lambda: None)
    _VlaCommandHandler._toggle_policy_pause(handler, None)
    assert state.pause_loop and state.grippers_follow_policy
    _publish_manual_grippers(state, socket)
    assert not packets  # Pause must not snap back to old manual targets.

    _VlaCommandHandler._toggle_right_hand(handler, None)
    assert not state.grippers_follow_policy
    _publish_manual_grippers(state, socket)
    assert unpack_array_message(packets.pop(), expected_topic='command').fields['manual_grippers'].tolist() == [1, 1]

    _VlaCommandHandler._toggle_policy_pause(handler, None)
    assert not state.pause_loop and not state.grippers_follow_policy
    assert not _publish_cached_action(state, config=config, zmq_socket=socket)
    assert not packets  # Resume alone cannot release the latch before fresh actions.
    state.cached_action_chunk = _policy_state().cached_action_chunk
    assert _publish_cached_action(state, config=config, zmq_socket=socket)
    assert state.grippers_follow_policy
    assert unpack_array_message(packets[-1], expected_topic='command').fields['release_manual_grippers'].tolist() == [1]


@pytest.mark.parametrize('method', ['_toggle_grippers', '_toggle_left_hand', '_toggle_right_hand'])
def test_manual_keys_cannot_override_running_vla(method):
    state = _policy_state()
    state.grippers_follow_policy = True
    events = []
    handler = SimpleNamespace(state=state, record_event=lambda *a, **kw: events.append(a))
    getattr(_VlaCommandHandler, method)(handler, None)
    assert state.grippers_follow_policy
    assert not state.initial_pose_left_hand_closed and not state.initial_pose_right_hand_closed
    assert events[-1][1] == 'MANUAL_GRIPPERS_BLOCKED'


def test_paused_stream_hold_does_not_reclaim_manual_grippers():
    state = _policy_state()
    state.pause_loop = True
    state.grippers_follow_policy = False
    packets = []
    _publish_cached_action(state, config=SimpleNamespace(action_horizon=2),
                           zmq_socket=SimpleNamespace(send=packets.append))
    assert len(packets) == 1 and packets[0].startswith(b'pose')
    assert not state.grippers_follow_policy
