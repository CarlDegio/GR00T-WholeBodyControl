"""Explicit state and single-cycle stages for the VLA service."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np

from gear_sonic.runtime.protocol.pose import pack_latent_action_message
from gear_sonic.runtime.telemetry import VLA_TIMING_SEGMENTS
from gear_sonic.utils.inference.vla.inference import calculate_latency_compensated_index

if TYPE_CHECKING:
    from gear_sonic.runtime.inference_service import InferenceServiceContext
    from gear_sonic.utils.inference.vla.service import InferenceConfig

LOGGER = logging.getLogger("sonic.vla")


def _current_vla_safety_reason(
    vla_safety_gate,
    vla_safety_monitor,
    *,
    monotonic=None,
) -> str:
    """Evaluate safety against snapshots captured before the current time.

    The monitor updates on another thread. Reading ``now`` before copying its
    orientation can make a newly received sample appear to come from the
    future, which the fail-closed gate correctly rejects. Capture both
    snapshots first so their timestamps cannot be newer than the comparison
    time used for this evaluation.
    """

    safety = vla_safety_monitor.snapshot()
    orientation = vla_safety_monitor.orientation_snapshot()
    clock = time.monotonic if monotonic is None else monotonic
    return vla_safety_gate.reason(
        now=clock(),
        safety=safety,
        robot_state_timestamp_s=orientation.received_at_s,
    )


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


def _pose_policy_is_active(cpp_loop_running: bool, cpp_mode: str, pause_loop: bool) -> bool:
    """Return whether a fresh VLA inference/action is allowed to run."""
    return cpp_loop_running and cpp_mode == "POSE" and not pause_loop


def _should_schedule_vla_inference(
    *,
    cpp_mode: str,
    worker_is_busy: bool,
    request_queue_is_empty: bool,
    time_since_request: float,
    inference_interval: float,
) -> bool:
    """Capture POSE observations even when action publication is paused."""

    return (
        cpp_mode == "POSE"
        and not worker_is_busy
        and request_queue_is_empty
        and time_since_request >= inference_interval
    )


@dataclass
class _VlaRuntimeState:
    pause_loop: bool = True
    cpp_loop_running: bool = False
    cpp_mode: str = "OFF"
    initial_pose_left_hand_closed: bool = False
    initial_pose_right_hand_closed: bool = False
    cached_action_chunk: Any = None
    action_chunk_index: int = 0
    last_inference_request_time: float = 0.0
    inference_generation: int = 0
    active_generation: int = -1
    zmq_frame_counter: int = 0
    task_generation: int = -1
    task_skill_id: int = 0
    task_window_id: int = 0
    task_active: bool = False
    # A LaViRA visual postcheck pauses fresh policy actions, but the C++ POSE
    # input still requires a continuous stream.  While this flag is set the
    # service republishes the terminal action from the current policy chunk.
    task_stream_hold_active: bool = False


def _invalidate_inference_preserving_action(
    state: _VlaRuntimeState,
    invalidate_inference,
    reason: str,
) -> None:
    """Invalidate stale inference work without opening a C++ action-stream gap."""

    cached_action = state.cached_action_chunk
    action_index = state.action_chunk_index
    invalidate_inference(reason)
    state.cached_action_chunk = cached_action
    state.action_chunk_index = action_index


def _stream_hold_is_active(state: _VlaRuntimeState) -> bool:
    """Return whether the current LaViRA hold has a safe action to republish."""

    return (
        state.task_active
        and state.task_stream_hold_active
        and state.pause_loop
        and state.cpp_loop_running
        and state.cpp_mode == "POSE"
        and state.cached_action_chunk is not None
    )


class _VlaCommandHandler:
    """One-to-one command dispatch over the existing VLA runtime state."""

    TASK_COMMANDS = {
        "start_vla_task",
        "hold_vla_task",
        "resume_vla_task",
        "stop_vla_task",
        "cancel_navigation",
    }

    def __init__(
        self,
        state: _VlaRuntimeState,
        *,
        control_listener,
        language_prompt_ref: list[str],
        inference_failures: queue.Queue,
        inference_failed_event: threading.Event,
        vla_safety_gate,
        vla_safety_monitor,
        task_status_intent,
        policy,
        record_event,
        invalidate_inference,
        publish_initial_pose,
        send_cpp_control_command,
        activate_vla_metrics,
        fail_active_task,
        publish_task_status,
    ) -> None:
        self.state = state
        self.control_listener = control_listener
        self.language_prompt_ref = language_prompt_ref
        self.inference_failures = inference_failures
        self.inference_failed_event = inference_failed_event
        self.vla_safety_gate = vla_safety_gate
        self.vla_safety_monitor = vla_safety_monitor
        self.task_status_intent = task_status_intent
        self.policy = policy
        self.record_event = record_event
        self.invalidate_inference = invalidate_inference
        self.publish_initial_pose = publish_initial_pose
        self.send_cpp_control_command = send_cpp_control_command
        self.activate_vla_metrics = activate_vla_metrics
        self.fail_active_task = fail_active_task
        self.publish_task_status = publish_task_status
        self.handlers = {
            "set_prompt": self._set_prompt,
            "select_pose_mode": self._select_pose_mode,
            "select_planner_mode": self._select_planner_mode,
            "toggle_policy_pause": self._toggle_policy_pause,
            "toggle_control_loop": self._toggle_control_loop,
            "toggle_left_hand_initial_pose": self._toggle_left_hand,
            "toggle_right_hand_initial_pose": self._toggle_right_hand,
        }

    def handle_next(self) -> None:
        command = self.control_listener.read_command()
        if command is None:
            return
        if command.name in self.TASK_COMMANDS:
            self._handle_task_command(command)
            return
        handler = self.handlers.get(command.name)
        if handler is not None:
            handler(command)

    def _handle_task_command(self, command) -> None:
        generation = int(command.parameters.get("generation", -1))
        skill_id = int(command.parameters.get("skill_id", 0))
        window_id = int(command.parameters.get("window_id", 0))
        identity = (generation, skill_id)
        active_identity = (self.state.task_generation, self.state.task_skill_id)
        if command.name == "cancel_navigation":
            self._cancel_navigation(generation)
            return
        if command.name == "start_vla_task":
            self._start_task(command, identity, active_identity, window_id)
            return
        self._update_active_task(command, identity, active_identity, window_id)

    def _cancel_navigation(self, generation: int) -> None:
        if generation < self.state.task_generation:
            return
        self.state.task_generation = generation
        self.state.task_skill_id = 0
        self.state.task_window_id = 0
        self.state.task_active = False
        self.state.task_stream_hold_active = False
        self.state.pause_loop = True
        self.invalidate_inference("navigation cancelled")
        if self.state.cpp_mode == "POSE":
            self.send_cpp_control_command(start=True, planner=True)

    def _start_task(self, command, identity, active_identity, window_id: int) -> None:
        if identity < active_identity or identity == active_identity:
            return
        prompt = command.parameters.get("handoff_context")
        original_task = command.parameters.get("task")
        if not isinstance(prompt, str) or not prompt.strip():
            self.record_event(
                logging.ERROR,
                "VLA_TASK_REJECTED",
                "Missing VLA handoff context",
            )
            return
        if not isinstance(original_task, str) or not original_task.strip():
            self.record_event(
                logging.ERROR,
                "VLA_TASK_REJECTED",
                "Missing original task",
            )
            return
        generation, skill_id = identity
        safety_reason = _current_vla_safety_reason(
            self.vla_safety_gate,
            self.vla_safety_monitor,
        )
        if safety_reason != "clear":
            self.record_event(
                logging.ERROR,
                "VLA_SAFETY_BLOCKED",
                "VLA task start rejected by the shared safety gate",
                reason=safety_reason,
                generation=generation,
                skill_id=skill_id,
            )
            self.task_status_intent.send(
                "vla_task_status",
                {
                    "generation": generation,
                    "skill_id": skill_id,
                    "segment_id": window_id,
                    "state": "failed",
                    "reason": safety_reason,
                },
            )
            return
        self.state.task_generation, self.state.task_skill_id = identity
        self.state.task_window_id = 0
        self.state.task_active = True
        self.state.task_stream_hold_active = False
        while not self.inference_failures.empty():
            try:
                self.inference_failures.get_nowait()
            except queue.Empty:
                break
        self.inference_failed_event.clear()
        if not self.policy.ping(timeout_ms=1000):
            self.fail_active_task(
                "vla_policy_unreachable",
                "PolicyServer was unreachable when MANIPULATE started",
            )
            return
        self.language_prompt_ref[0] = prompt
        self.state.zmq_frame_counter = 0
        self.invalidate_inference("VLA task started")
        if self.state.cpp_mode == "PLANNER":
            if not self.publish_initial_pose():
                self.fail_active_task(
                    "vla_initial_pose_failed",
                    "VLA could not publish its initial pose",
                )
                return
            if not self.send_cpp_control_command(start=True, planner=False):
                self.fail_active_task(
                    "vla_pose_mode_failed",
                    "VLA could not enter POSE mode",
                )
                return
        elif self.state.cpp_mode == "OFF":
            if not self.send_cpp_control_command(start=True, planner=False):
                self.fail_active_task(
                    "vla_pose_mode_failed",
                    "VLA could not start POSE mode",
                )
                return
        self.state.pause_loop = False
        self.activate_vla_metrics()
        self.publish_task_status("active", "started", window_id=0)

    def _update_active_task(
        self,
        command,
        identity,
        active_identity,
        window_id: int,
    ) -> None:
        if (
            identity != active_identity
            or not self.state.task_active
            or window_id < self.state.task_window_id
        ):
            return
        self.state.task_window_id = window_id
        if command.name == "hold_vla_task":
            if self.state.pause_loop:
                return
            self.state.pause_loop = True
            self.state.task_stream_hold_active = (
                self.state.cached_action_chunk is not None
            )
            _invalidate_inference_preserving_action(
                self.state,
                self.invalidate_inference,
                f"VLA window {window_id} held with streamed terminal action",
            )
            if not self.state.task_stream_hold_active:
                self.record_event(
                    logging.WARNING,
                    "VLA_STREAM_HOLD_UNAVAILABLE",
                    "VLA postcheck hold began before the first policy action",
                    generation=self.state.task_generation,
                    skill_id=self.state.task_skill_id,
                    window_id=window_id,
                )
            return
        if command.name == "resume_vla_task":
            prompt = command.parameters.get("handoff_context")
            if isinstance(prompt, str) and prompt.strip():
                self.language_prompt_ref[0] = prompt
            if not self.state.pause_loop:
                return
            self.state.pause_loop = False
            self.state.task_stream_hold_active = False
            _invalidate_inference_preserving_action(
                self.state,
                self.invalidate_inference,
                f"VLA window {window_id} resumed from streamed hold",
            )
            self.activate_vla_metrics()
            return
        self.state.task_active = False
        self.state.task_stream_hold_active = False
        self.state.pause_loop = True
        self.invalidate_inference(f"VLA task stopped at window {window_id}")
        if self.state.cpp_mode == "POSE":
            self.send_cpp_control_command(start=True, planner=True)

    def _set_prompt(self, command) -> None:
        new_prompt = command.parameters.get("prompt")
        if isinstance(new_prompt, str) and new_prompt:
            old_prompt = self.language_prompt_ref[0]
            self.language_prompt_ref[0] = new_prompt
            self.invalidate_inference("prompt changed")
            LOGGER.info("Inference prompt changed: %r -> %r", old_prompt, new_prompt)
        else:
            self.record_event(
                logging.WARNING,
                "INVALID_PROMPT",
                "Empty inference prompt ignored",
            )

    def _select_pose_mode(self, _command) -> None:
        self.state.zmq_frame_counter = 0
        self.invalidate_inference("entering POSE mode")
        if self.state.cpp_mode == "PLANNER":
            if not self.publish_initial_pose():
                return
            self.send_cpp_control_command(start=True, planner=False)
        if self.state.cpp_mode == "POSE":
            self.activate_vla_metrics()

    def _select_planner_mode(self, _command) -> None:
        self.state.zmq_frame_counter = 0
        self.invalidate_inference("entering PLANNER mode")
        if self.state.cpp_mode == "POSE":
            self.send_cpp_control_command(start=True, planner=True)
        if self.state.cpp_mode == "PLANNER":
            self.record_event(
                logging.WARNING,
                "PLANNER_BLOCKED",
                "Planner mode active; remote VLA inference is suspended",
            )

    def _toggle_policy_pause(self, _command) -> None:
        if self.state.cpp_mode == "PLANNER":
            self.record_event(
                logging.WARNING,
                "PLANNER_BLOCKED",
                "Switch to POSE mode before resuming remote VLA inference",
            )
            return
        self.state.pause_loop = not self.state.pause_loop
        self.state.task_stream_hold_active = False
        self.invalidate_inference(
            "POSE policy resumed"
            if not self.state.pause_loop
            else "POSE policy paused"
        )
        if self.state.pause_loop:
            self.record_event(
                logging.INFO,
                "INFERENCE_PAUSED",
                "VLA action publication paused; remote inference remains warm",
                remote_requests_continue=True,
            )
        elif self.state.cpp_mode == "POSE":
            self.activate_vla_metrics()

    def _toggle_control_loop(self, _command) -> None:
        self.invalidate_inference("C++ control toggled")
        if self.state.cpp_loop_running:
            current_planner = self.state.cpp_mode == "PLANNER"
            if self.send_cpp_control_command(
                start=False,
                planner=current_planner,
            ):
                self.record_event(
                    logging.INFO,
                    "INFERENCE_PAUSED",
                    "Remote VLA inference paused because the control loop stopped",
                    remote_requests_continue=False,
                )
        elif self.send_cpp_control_command(start=True, planner=True):
            self.record_event(
                logging.WARNING,
                "PLANNER_BLOCKED",
                "Planner mode active; remote VLA inference is suspended",
            )

    def _toggle_left_hand(self, _command) -> None:
        self.state.initial_pose_left_hand_closed = (
            not self.state.initial_pose_left_hand_closed
        )

    def _toggle_right_hand(self, _command) -> None:
        self.state.initial_pose_right_hand_closed = (
            not self.state.initial_pose_right_hand_closed
        )


def _consume_task_failure(
    state: _VlaRuntimeState,
    inference_failures: queue.SimpleQueue,
    fail_active_task,
) -> bool:
    if not state.task_active:
        return False
    try:
        inference_failure = inference_failures.get_nowait()
    except queue.Empty:
        return False
    if inference_failure is None:
        return False
    fail_active_task(
        f"vla_inference_failed:{inference_failure}",
        "VLA inference failed; the task will not be retried",
    )
    return True


def _enforce_active_task_safety(
    state: _VlaRuntimeState,
    *,
    vla_safety_gate,
    vla_safety_monitor,
    invalidate_inference,
    send_cpp_control_command,
    task_status_intent,
    record_event,
) -> None:
    if (
        not state.task_active
        or state.cpp_mode != "POSE"
        or (state.pause_loop and not state.task_stream_hold_active)
    ):
        return
    safety_reason = _current_vla_safety_reason(
        vla_safety_gate,
        vla_safety_monitor,
    )
    if safety_reason == "clear":
        return
    state.task_active = False
    state.task_stream_hold_active = False
    state.pause_loop = True
    invalidate_inference(f"VLA safety blocked: {safety_reason}")
    send_cpp_control_command(start=True, planner=True)
    task_status_intent.send(
        "vla_task_status",
        {
            "generation": state.task_generation,
            "skill_id": state.task_skill_id,
            "segment_id": state.task_window_id,
            "state": "failed",
            "reason": safety_reason,
        },
    )
    record_event(
        logging.ERROR,
        "VLA_SAFETY_BLOCKED",
        "VLA POSE action stopped by the shared safety gate",
        reason=safety_reason,
        generation=state.task_generation,
        skill_id=state.task_skill_id,
    )


def _consume_vla_result(
    state: _VlaRuntimeState,
    result_queue: queue.Queue,
    *,
    service: InferenceServiceContext,
    config: InferenceConfig,
    language_prompt: str,
    record_event,
) -> None:
    try:
        (
            result_generation,
            processed_action,
            inference_start_time,
            timing_ms,
        ) = result_queue.get_nowait()
    except queue.Empty:
        return
    inference_delay = time.monotonic() - inference_start_time
    timing_ms["action_ready"] = inference_delay * 1000.0
    if result_generation != state.inference_generation or not _pose_policy_is_active(
        state.cpp_loop_running,
        state.cpp_mode,
        state.pause_loop,
    ):
        return
    service.publish_metrics(
        timing_ms,
        allowed_names=VLA_TIMING_SEGMENTS,
    )
    state.action_chunk_index = calculate_latency_compensated_index(
        inference_delay,
        config.action_publish_rate,
        config.action_horizon,
    )
    state.cached_action_chunk = processed_action
    if state.active_generation == state.inference_generation:
        return
    record_event(
        logging.INFO,
        "INFERENCE_ACTIVE",
        "Remote VLA inference returned its first active result",
        generation=state.inference_generation,
        prompt=language_prompt,
        action_ready_ms=timing_ms["action_ready"],
    )
    state.active_generation = state.inference_generation


def _schedule_vla_inference(
    state: _VlaRuntimeState,
    *,
    now: float,
    inference_interval: float,
    inference_queue: queue.Queue,
    inference_busy_event: threading.Event,
    inference_failed_event: threading.Event,
) -> None:
    should_start = (
        not inference_failed_event.is_set()
        and _should_schedule_vla_inference(
            cpp_mode=state.cpp_mode,
            worker_is_busy=inference_busy_event.is_set(),
            request_queue_is_empty=inference_queue.empty(),
            time_since_request=(now - state.last_inference_request_time),
            inference_interval=inference_interval,
        )
    )
    if not should_start:
        return
    try:
        inference_queue.put_nowait(state.inference_generation)
        state.last_inference_request_time = now
    except queue.Full:
        pass


def _publish_cached_action(
    state: _VlaRuntimeState,
    *,
    config: InferenceConfig,
    zmq_socket,
) -> None:
    processed_action = state.cached_action_chunk
    if processed_action:
        motion_token = np.asarray(
            get_action_field(processed_action, "motion_token"),
            dtype=np.float32,
        )
        left_hand_joints = np.asarray(
            get_action_field(processed_action, "left_hand_joints"),
            dtype=np.float32,
        )
        right_hand_joints = np.asarray(
            get_action_field(processed_action, "right_hand_joints"),
            dtype=np.float32,
        )

        # Action arrays arrive as (B, T, D); remove the batch dimension.
        if motion_token.ndim == 3:
            motion_token = motion_token[0]
        if left_hand_joints.ndim == 3:
            left_hand_joints = left_hand_joints[0]
        if right_hand_joints.ndim == 3:
            right_hand_joints = right_hand_joints[0]

        horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
        current_idx = min(state.action_chunk_index, horizon - 1)
        if motion_token.ndim == 2:
            motion_token = motion_token[current_idx]
        if left_hand_joints.ndim == 2:
            left_hand_joints = left_hand_joints[current_idx]
        if right_hand_joints.ndim == 2:
            right_hand_joints = right_hand_joints[current_idx]

        frame_index = np.array([state.zmq_frame_counter], dtype=np.int64)
        state.zmq_frame_counter += 1
        zmq_socket.send(
            pack_latent_action_message(
                motion_token,
                frame_index,
                left_hand_joints=left_hand_joints,
                right_hand_joints=right_hand_joints,
            )
        )

    state.action_chunk_index = min(
        state.action_chunk_index + 1,
        config.action_horizon - 1,
    )
