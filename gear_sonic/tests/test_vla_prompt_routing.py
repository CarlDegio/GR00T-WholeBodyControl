"""Keep agent task context separate from the trained VLA policy instruction."""

import json
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gear_sonic.utils.inference.lavira import service as lavira_service
from gear_sonic.utils.inference.lavira.agent import language_action_prompt
from gear_sonic.utils.inference.vla.runtime import _VlaCommandHandler, _VlaRuntimeState
from gear_sonic.utils.inference.vla.service import (
    load_inference_config,
    prepare_observation_from_sensors,
)


@pytest.mark.parametrize("prompt", ["", "Put the bottle into the blue basket."])
def test_agent_factory_reads_general_prompt_from_shared_profile(tmp_path, monkeypatch, prompt):
    overlay = tmp_path / "task.json"
    overlay.write_text(json.dumps({"components": {"vla": {"vla_prompt": prompt}}}))
    config = lavira_service.load_lavira_config(overlays=(str(overlay),))
    vla_config = load_inference_config(overlays=(str(overlay),))
    monkeypatch.setattr(lavira_service, "SensorGatewayRGBDCamera", Mock())
    monkeypatch.setattr(lavira_service, "LaViRAClient", Mock())
    runtime = lavira_service.LaviraPlannerRuntime(config, submit_intent=Mock())
    agent = lavira_service._agent(config, "unused", runtime)

    assert agent.manipulation_prompt == prompt
    assert vla_config.vla_prompt == prompt
    assert vla_config.vla_trained_prompt == "collect the pill bottle"
    context = language_action_prompt(
        agent.mission, agent.navigation_mode, agent.global_target, 1, "", "single", None,
        manipulation_prompt=agent.manipulation_prompt,
    )
    assert f"**MANIPULATION TASK AFTER NAVIGATION**:\n{json.dumps(prompt)}\n" in context
    assert vla_config.vla_trained_prompt not in context


def command_handler(trained_prompt):
    state = _VlaRuntimeState(cpp_mode="POSE", cpp_loop_running=True)
    handler = _VlaCommandHandler(
        state,
        control_listener=Mock(),
        language_prompt_ref=[trained_prompt],
        inference_failures=queue.Queue(),
        inference_failed_event=threading.Event(),
        vla_safety_gate=Mock(reason=Mock(return_value="clear")),
        vla_safety_monitor=Mock(),
        task_status_intent=Mock(),
        policy=Mock(ping=Mock(return_value=True)),
        record_event=Mock(),
        invalidate_inference=Mock(),
        publish_initial_pose=Mock(return_value=True),
        send_cpp_control_command=Mock(return_value=True),
        activate_vla_metrics=Mock(),
        fail_active_task=Mock(),
        publish_task_status=Mock(),
    )
    return handler, state


def policy_observation(prompt):
    cameras = ("ego_view", "chest_view", "left_wrist", "right_wrist")
    gateway = Mock()
    gateway.read_camera.return_value = {
        "images": {name: b"camera-jpeg" for name in cameras},
        "image_shapes": {name: (8, 8, 3) for name in cameras},
        "timestamps": {name: [0.0] for name in cameras},
    }
    gateway.read_state.return_value = {
        "body_q": [], "left_hand_q": [], "right_hand_q": [],
        "base_quat": [1.0, 0.0, 0.0, 0.0],
    }
    robot = Mock(num_joints=1)
    robot.get_configuration_from_actuated_joints.return_value = [0.0]
    robot.get_joint_group_indices.return_value = [0]
    observation, _ = prepare_observation_from_sensors(gateway, robot, prompt)
    return observation


@pytest.mark.parametrize("task", ["", "Put the bottle into the blue basket."])
@pytest.mark.parametrize("legacy_context", [False, True])
def test_vla_start_and_resume_keep_trained_policy_prompt(task, legacy_context):
    trained_prompt = load_inference_config().vla_trained_prompt
    handler, state = command_handler(trained_prompt)
    parameters = {"generation": 1, "skill_id": 2, "window_id": 0, "task": task}
    if legacy_context:
        parameters["handoff_context"] = "agent-generated handoff text"

    handler.control_listener.read_command.return_value = SimpleNamespace(
        name="start_vla_task", parameters=parameters,
    )
    handler.handle_next()
    assert state.task_active and not state.pause_loop
    handler.publish_task_status.assert_called_once_with("active", "started", window_id=0)
    assert handler.language_prompt_ref == [trained_prompt]

    state.cached_action_chunk = {"terminal": [0.0]}
    for name in ("hold_vla_task", "resume_vla_task"):
        handler.control_listener.read_command.return_value = SimpleNamespace(
            name=name, parameters={**parameters, "window_id": 1},
        )
        handler.handle_next()
        assert state.pause_loop == (name == "hold_vla_task")
        assert handler.language_prompt_ref == [trained_prompt]

    observation = policy_observation(handler.language_prompt_ref[0])
    assert observation["language"]["annotation.human.task_description"] == [[trained_prompt]]
    handler.record_event.assert_not_called()
    handler.fail_active_task.assert_not_called()


def test_vla_does_not_use_agent_context_when_trained_prompt_is_missing():
    handler, state = command_handler("")
    handler.control_listener.read_command.return_value = SimpleNamespace(
        name="start_vla_task",
        parameters={
            "generation": 1, "skill_id": 2, "task": "general task",
            "handoff_context": "must not become the policy prompt",
        },
    )
    handler.handle_next()
    assert not state.task_active
    assert handler.language_prompt_ref == [""]
    handler.policy.ping.assert_not_called()
    assert handler.record_event.call_args.args[1:] == (
        "VLA_TASK_REJECTED", "Missing VLA trained prompt",
    )
