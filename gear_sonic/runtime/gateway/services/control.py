#!/usr/bin/env python3
"""Route typed operator intents to structured control consumers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum, auto
import json
import logging
import time
from typing import Callable, Mapping

import zmq

from gear_sonic.experiments.supervisor import ExperimentSupervisor
from gear_sonic.runtime.gateway.control import (
    BASE_POSE_RUNTIME_STATUS_COMMAND,
    ControlGatewayCore,
    ControlGatewayRouter,
    NavigationControlAction,
    NavigationControlState,
)
from gear_sonic.runtime.gateway.event_display import EventPaneDisplay
from gear_sonic.runtime.profile import RuntimeProfile, load_runtime_profile
from gear_sonic.runtime.protocol import OperatorCommand, build_navigation_message
from gear_sonic.runtime.telemetry import (
    build_event,
    configure_file_logging,
    emit_event,
)
from gear_sonic.runtime.zmq_sockets import (
    bind_publisher,
    bind_pull,
    connect_subscriber,
)


def build_base_pose_runtime_status(
    action: NavigationControlAction,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    """Build viewer telemetry without feeding overlays into motion validation."""

    base_pose_action = str(parameters.get("action", "visual_servo"))
    state = {
        "hold": "inference",
        "visual_servo": "motion",
        "stop": "stopping",
    }.get(base_pose_action, "motion")
    status: dict[str, object] = {
        "generation": action.generation,
        "skill_id": action.skill_id,
        "segment_id": action.segment_id,
        "state": state,
        "velocity": list(action.velocity or (0.0, 0.0, 0.0)),
        "action": base_pose_action,
        "camera_stream": str(parameters.get("camera_stream", "")),
    }
    viewer_overlay = parameters.get("viewer_overlay")
    if viewer_overlay is not None:
        if not isinstance(viewer_overlay, Mapping):
            raise ValueError("base-pose viewer_overlay must be an object")
        status["viewer_overlay"] = dict(viewer_overlay)
    return status


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="", help="Base runtime YAML profile")
    parser.add_argument("--overlay", action="append", default=[])
    return parser


class DispatchDisposition(Enum):
    """Describe whether a command handler consumed the original command."""

    HANDLED = auto()
    FORWARD_ORIGINAL = auto()


@dataclass(frozen=True)
class PendingVlaCompletion:
    generation: int
    skill_id: int
    deadline: float
    parameters: dict[str, object]


class ControlGatewayRuntime:
    """Own Control Gateway transport and route one event at a time."""

    def __init__(self, profile: RuntimeProfile) -> None:
        self.profile = profile
        self.logger = configure_file_logging("control_gateway")
        self.display = EventPaneDisplay()
        self.context = zmq.Context()
        self.intent_pull = bind_pull(
            self.context,
            profile.endpoint_uri("control_gateway_intent"),
            high_water_mark=100,
            linger_ms=0,
        )
        self.event_pull = bind_pull(
            self.context,
            profile.endpoint_uri("runtime_event_ingress"),
            high_water_mark=100,
            linger_ms=0,
        )
        self.dispatch_pub = bind_publisher(
            self.context,
            profile.endpoint_uri("control_gateway_dispatch"),
            high_water_mark=100,
            linger_ms=0,
        )
        self.navigation_pub = bind_publisher(
            self.context,
            profile.endpoint_uri("navigation_command"),
            high_water_mark=100,
            linger_ms=0,
        )
        self.navigation_status_sub = connect_subscriber(
            self.context,
            profile.endpoint_uri("navigation_status"),
            high_water_mark=100,
            linger_ms=0,
        )
        self.router = ControlGatewayRouter()
        self.navigation = NavigationControlState()
        self.gateway_events = ControlGatewayCore(
            source="control_gateway_navigation"
        )
        self.experiment = ExperimentSupervisor(self)
        self.pending_vla_completion: PendingVlaCompletion | None = None
        self.command_handlers: dict[
            str, Callable[[OperatorCommand], DispatchDisposition]
        ] = {
            "navigation_key": self._handle_navigation_key,
            "complete_agent_success": self._handle_complete_agent_result,
            "complete_agent_failure": self._handle_complete_agent_result,
            "lavira_depth_request": self._handle_lavira_depth_request,
            "lavira_rgbd_captured": self._handle_lavira_rgbd_captured,
            "navigation_goal": self._handle_navigation_goal,
            "navigation_heading_goal": self._handle_navigation_heading_goal,
            "navigation_agent_status": self._handle_navigation_agent_status,
            "start_base_pose": self._handle_start_base_pose,
            "base_pose_velocity": self._handle_base_pose_velocity,
            "base_pose_status": self._handle_base_pose_status,
            "select_pose_mode": self._handle_select_pose_mode,
            "experiment_vla_started": self._handle_experiment_vla_started,
            "experiment_nav_step": self._handle_experiment_nav_step,
            "experiment_perturbation": self._handle_experiment_perturbation,
        }
        for name in (
            "start_vla_task",
            "hold_vla_task",
            "resume_vla_task",
            "stop_vla_task",
        ):
            self.command_handlers[name] = self._handle_vla_task

    def show_event(
        self,
        level: int,
        code: str,
        message: str,
        **fields: object,
    ) -> None:
        payload = build_event(
            "control_gateway", level, code, message, **fields,
        )
        emit_event(payload, logger=self.logger)
        self.display.accept(payload)

    def show_command(
        self,
        command: OperatorCommand,
        code: str,
        message: str | None = None,
        level: int = logging.INFO,
        **fields: object,
    ) -> None:
        details = {
            "source": command.metadata.source,
            "generation": command.parameters.get("generation"),
            "reason": command.parameters.get("reason"),
            **fields,
        }
        self.show_event(level, code, message or command.name, **details)

    def dispatch_navigation_event(
        self,
        name: str,
        parameters: dict[str, object],
    ) -> None:
        event = self.gateway_events.accept_command(
            name,
            parameters=parameters,
        )
        self.dispatch_pub.send_string(event.command.to_json())

    def publish_navigation_action(
        self,
        action: NavigationControlAction,
        *,
        source: str,
        input_key: str | None = None,
        agent_parameters: Mapping[str, object] | None = None,
    ) -> bool:
        if agent_parameters and "surface" in agent_parameters:
            raise ValueError(
                "legacy BasePose surface field is not supported; use "
                "yaw_align_target"
            )
        if action.mode == "ignored":
            self.show_event(
                logging.WARNING,
                "CONTROL_IGNORED",
                action.reason or "ignored",
                source=source,
                key=input_key,
                generation=action.generation,
            )
            return False
        pending = self.pending_vla_completion
        if pending is not None and action.generation != pending.generation:
            self.pending_vla_completion = None
            self.experiment.recorder.write(
                "vla_completion_delay_ended", generation=pending.generation,
                skill_id=pending.skill_id, reason=action.reason or "superseded",
            )
        self.navigation_pub.send_string(
            build_navigation_message(
                mode=action.mode,
                generation=action.generation,
                skill_id=action.skill_id,
                segment_id=action.segment_id,
                velocity=action.velocity,
                source=source,
            )
        )
        if action.agent_event is not None:
            event_parameters: dict[str, object] = {
                "generation": action.generation,
                "skill_id": action.skill_id,
                "segment_id": action.segment_id,
            }
            if action.reason:
                event_parameters["reason"] = action.reason
            if action.agent_event == "start_base_pose" and agent_parameters:
                for name in ("target", "yaw_align_target"):
                    if name in agent_parameters:
                        event_parameters[name] = agent_parameters[name]
            self.dispatch_navigation_event(action.agent_event, event_parameters)
            self.show_event(
                logging.INFO,
                action.agent_event.upper(),
                action.mode,
                source=source,
                generation=action.generation,
                reason=action.reason or None,
            )
        return True

    def run(self) -> None:
        self.show_event(
            logging.INFO,
            "READY",
            "ControlGateway ready",
            profile=self.profile.name,
        )
        poller = zmq.Poller()
        poller.register(self.intent_pull, zmq.POLLIN)
        poller.register(self.event_pull, zmq.POLLIN)
        poller.register(self.navigation_status_sub, zmq.POLLIN)
        while True:
            events = dict(poller.poll(100))
            if self.intent_pull in events:
                self._receive_intent()
            if self.navigation_status_sub in events:
                self._receive_navigation_status()
            if self.event_pull in events:
                self._receive_runtime_event()
            self._handle_timeout()

    def close(self) -> None:
        self.display.close()
        for socket in (
            self.intent_pull,
            self.event_pull,
            self.dispatch_pub,
            self.navigation_pub,
            self.navigation_status_sub,
        ):
            socket.close()
        self.context.term()

    def _receive_intent(self) -> None:
        payload = self.intent_pull.recv()
        try:
            command = OperatorCommand.from_json(payload)
            routed = self.router.route(command)
            if not routed.accepted:
                self.show_command(
                    command,
                    "CONTROL_REJECTED",
                    level=logging.WARNING,
                    reason=routed.reason,
                    command_id=command.command_id,
                )
                return
            disposition = self._dispatch_command(command)
            if disposition is DispatchDisposition.FORWARD_ORIGINAL:
                self.dispatch_pub.send_string(command.to_json())
        except (KeyError, TypeError, ValueError) as exc:
            self.show_event(logging.WARNING, "INVALID_INTENT", str(exc))

    def _dispatch_command(
        self,
        command: OperatorCommand,
    ) -> DispatchDisposition:
        if (
            self.pending_vla_completion is not None
            and command.metadata.source == "operator_console"
            and command.name in {
                "toggle_control_loop", "select_planner_mode", "toggle_policy_pause",
                "toggle_left_hand_initial_pose", "toggle_right_hand_initial_pose", "set_prompt",
            }
        ):
            action = self.navigation.handle_key(" ", now=time.monotonic(), cancel_reason=command.name)
            self.publish_navigation_action(action, source=command.metadata.source)
        if self.experiment.active and command.metadata.source == "operator_console" and command.name in {
            "toggle_control_loop", "select_pose_mode", "select_planner_mode", "toggle_policy_pause",
            "toggle_left_hand_initial_pose", "toggle_right_hand_initial_pose", "set_prompt",
        }:
            self.experiment.recorder.write(
                "intervention", generation=self.experiment.active[0], command=command.name,
            )
        handler = self.command_handlers.get(command.name)
        if handler is None:
            self.show_command(command, "CONTROL_COMMAND")
            return DispatchDisposition.FORWARD_ORIGINAL
        return handler(command)

    def _receive_navigation_status(self) -> None:
        try:
            payload = self.navigation_status_sub.recv_json()
            if not self.navigation.accept_status(
                payload,
                owner="lavira",
                agent_final=False,
            ):
                return
            self.dispatch_navigation_event("navigation_status", dict(payload))
            state = str(payload.get("state", "unknown"))
            level = logging.ERROR if state == "failed" else logging.INFO
            self.show_event(
                level,
                "NAVIGATION_STATUS",
                state,
                source="navdp",
                generation=payload.get("generation"),
                reason=payload.get("reason"),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.show_event(
                logging.WARNING,
                "INVALID_NAVIGATION_STATUS",
                str(exc),
            )

    def _receive_runtime_event(self) -> None:
        try:
            self.display.accept(self.event_pull.recv_json())
        except (
            AttributeError,
            KeyError,
            OSError,
            OverflowError,
            TypeError,
            ValueError,
        ) as exc:
            self.show_event(logging.WARNING, "INVALID_RUNTIME_EVENT", str(exc))

    def _handle_timeout(self) -> None:
        pending = self.pending_vla_completion
        if pending is not None:
            if (pending.generation, pending.skill_id) != (self.navigation.generation, self.navigation.skill_id):
                self.pending_vla_completion = None
            elif time.monotonic() >= pending.deadline:
                self.pending_vla_completion = None
                self.experiment.recorder.write(
                    "vla_command", generation=pending.generation,
                    command="stop_vla_task", skill_id=pending.skill_id,
                    window_id=pending.parameters.get("window_id"),
                )
                self.dispatch_navigation_event("stop_vla_task", pending.parameters)
                action = self.navigation.handle_key(
                    " ", now=time.monotonic(), cancel_reason="post_completion_delay_elapsed",
                )
                self.publish_navigation_action(action, source="va_completion")
                self.experiment.recorder.write(
                    "vla_completion_delay_ended", generation=pending.generation,
                    skill_id=pending.skill_id, reason="elapsed",
                )
                self.show_event(
                    logging.INFO, "VLA_COMPLETION_DELAY_ENDED",
                    "Post-completion VLA execution finished; returning to PLANNER",
                    generation=pending.generation, skill_id=pending.skill_id,
                )
        self.experiment.tick()
        timeout_action = self.navigation.tick(now=time.monotonic())
        if timeout_action is not None:
            if timeout_action.agent_event == 'cancel_navigation' and self.experiment.active:
                self.experiment.finish(self.experiment.active[0], 'failed', timeout_action.reason)
            self.publish_navigation_action(
                timeout_action,
                source="ControlGateway manual-hold timer",
            )

    @staticmethod
    def _require_source(
        command: OperatorCommand,
        expected: str,
        message: str,
    ) -> None:
        if command.metadata.source != expected:
            raise ValueError(message)

    @staticmethod
    def _action_identity(action: NavigationControlAction) -> dict[str, object]:
        return {
            "generation": action.generation,
            "skill_id": action.skill_id,
            "segment_id": action.segment_id,
        }

    def _send_navigation_stop(
        self,
        *,
        generation: int,
        skill_id: int,
        segment_id: int,
    ) -> None:
        self.navigation_pub.send_string(
            build_navigation_message(
                mode="stop",
                generation=generation,
                skill_id=skill_id,
                segment_id=segment_id,
            )
        )

    def _handle_navigation_key(
        self,
        command: OperatorCommand,
    ) -> DispatchDisposition:
        key = command.parameters.get("key")
        if not isinstance(key, str) or len(key) != 1:
            raise ValueError("navigation_key requires one string key")
        if key.lower() == "n" and self.pending_vla_completion is not None:
            cancel = self.navigation.handle_key(" ", now=time.monotonic(), cancel_reason="restarted_with_n")
            self.publish_navigation_action(cancel, source="experiment_restart")
        if key.lower() == "n" and self.experiment.active:
            previous = self.experiment.active[0]
            self.experiment.recorder.write("intervention", generation=previous, command="restart_with_n")
            self.experiment.finish(previous, "cancelled", "restarted_with_n")
            cancel = self.navigation.handle_key(" ", now=time.monotonic(), cancel_reason="restarted_with_n")
            self.publish_navigation_action(cancel, source="experiment_restart")
        old_generation = self.navigation.generation
        action = self.navigation.handle_key(
            key,
            now=time.monotonic(),
            cancel_reason=str(command.parameters.get("reason", "")),
        )
        if action.agent_event == "start_navigation":
            self.experiment.start(action.generation)
        elif action.agent_event == "cancel_navigation":
            self.experiment.recorder.write("intervention", generation=old_generation, command="operator_cancel")
            self.experiment.finish(old_generation, "cancelled", "operator_cancel")
        self.publish_navigation_action(
            action,
            source=command.metadata.source,
            input_key="Space" if key == " " else key.upper(),
            agent_parameters=command.parameters,
        )
        return DispatchDisposition.HANDLED

    def _handle_complete_agent_result(self, command: OperatorCommand) -> DispatchDisposition:
        self._require_source(command, "operator_console", "Agent result requires the operator console")
        success = command.name == "complete_agent_success"
        outcome = "success" if success else "failure"
        reason = f"operator_{outcome}"
        code = "TASK_COMPLETED" if success else "TASK_FAILED"
        nav = self.navigation
        if not nav.lavira_task_active or nav.task_started_at is None:
            self.show_command(
                command, "CONTROL_IGNORED", f"No active agent task to mark as {outcome}",
                level=logging.WARNING, reason="no_active_agent",
            )
            return DispatchDisposition.HANDLED
        # Use the CLI key event, so transport and stop/recording latency do not
        # extend the task. An old key must not terminate a newer task.
        completed_ns = command.metadata.timestamp_ns
        completed_at = completed_ns / 1e9
        if completed_at < nav.task_started_at:
            self.show_command(
                command, "CONTROL_IGNORED", "Agent result key predates the active task",
                level=logging.WARNING, reason="result_before_task_start",
            )
            return DispatchDisposition.HANDLED
        now = time.monotonic()
        generation = nav.generation
        timing = dict(
            completion_time_s=completed_at - nav.task_started_at,
            completed_monotonic_ns=completed_ns,
            completed_wall_time_ns=time.time_ns() - max(0, time.monotonic_ns() - completed_ns),
        )
        result = dict(
            generation=generation, skill_id=nav.skill_id, segment_id=max(0, nav.segment_id),
            state="reached" if success else "failed", reason=reason, **timing,
        )
        # Invalidate in-flight model/controller replies before releasing ownership.
        # VLA handles this cancellation by selecting PLANNER with C++ still running.
        action = nav.handle_key(" ", now=now, cancel_reason=reason)
        self.publish_navigation_action(action, source=command.metadata.source, input_key="G" if success else "H")
        self.experiment.confirm_result(generation, success=success, **timing)
        self.experiment.recorder.runtime("control_gateway", code, **result)
        self.show_event(
            logging.INFO if success else logging.WARNING, code,
            f"Operator confirmed {outcome} at {timing['completion_time_s']:.2f}s; returning to PLANNER",
            source=command.metadata.source, **result,
        )
        return DispatchDisposition.HANDLED

    def _handle_lavira_depth_request(
        self,
        command: OperatorCommand,
    ) -> DispatchDisposition:
        self._require_source(
            command,
            "lavira_agent",
            "LaViRA depth request requires lavira_agent source",
        )
        if not self.navigation.accept_lavira_depth_request(command.parameters):
            raise ValueError("stale LaViRA depth request")
        self.dispatch_navigation_event(
            "lavira_depth_request", dict(command.parameters)
        )
        self.show_command(
            command,
            "LAVIRA_DEPTH_REQUEST",
            "LaViRA depth lease acquired",
        )
        return DispatchDisposition.HANDLED

    def _handle_lavira_rgbd_captured(
        self,
        command: OperatorCommand,
    ) -> DispatchDisposition:
        self._require_source(
            command,
            "lavira_agent",
            "LaViRA RGB-D completion requires lavira_agent source",
        )
        if not self.navigation.accept_lavira_rgbd_captured(command.parameters):
            raise ValueError("stale LaViRA RGB-D completion")
        self.dispatch_navigation_event(
            "lavira_rgbd_captured", dict(command.parameters)
        )
        self.show_command(
            command,
            "LAVIRA_RGBD_CAPTURED",
            "LaViRA RGB-D captured",
        )
        return DispatchDisposition.HANDLED

    def _handle_navigation_goal(
        self,
        command: OperatorCommand,
    ) -> DispatchDisposition:
        action = self.navigation.accept_goal(command.parameters)
        goal = command.parameters.get("goal_base")
        if not isinstance(goal, (list, tuple)) or len(goal) != 2:
            raise ValueError("navigation_goal requires goal_base [x, y]")
        self.navigation_pub.send_string(
            build_navigation_message(
                mode=action.mode,
                generation=action.generation,
                skill_id=action.skill_id,
                segment_id=action.segment_id,
                goal_base=goal,
                target=str(command.parameters.get("target", "")),
                target_type=str(command.parameters.get("target_type", "")),
                confidence=float(command.parameters.get("confidence", 0.0)),
            )
        )
        self.dispatch_navigation_event(
            "navigation_goal",
            self._action_identity(action),
        )
        self.show_command(
            command,
            "NAVIGATION_GOAL",
            "navigation goal accepted",
            goal_base=goal,
            target=command.parameters.get("target"),
            confidence=command.parameters.get("confidence"),
        )
        return DispatchDisposition.HANDLED

    def _handle_navigation_heading_goal(
        self,
        command: OperatorCommand,
    ) -> DispatchDisposition:
        self._require_source(
            command,
            "lavira_agent",
            "heading goal requires lavira_agent source",
        )
        action = self.navigation.accept_heading_goal(command.parameters)
        heading_delta_rad = float(command.parameters["heading_delta_rad"])
        heading_turn_direction = (
            None
            if command.parameters.get("heading_turn_direction") is None
            else str(command.parameters["heading_turn_direction"])
        )
        heading_max_angular_speed_rad_s = (
            None
            if command.parameters.get("heading_max_angular_speed_rad_s") is None
            else float(command.parameters["heading_max_angular_speed_rad_s"])
        )
        heading_max_duration_s = (
            None
            if command.parameters.get("heading_max_duration_s") is None
            else float(command.parameters["heading_max_duration_s"])
        )
        self.navigation_pub.send_string(
            build_navigation_message(
                mode=action.mode,
                generation=action.generation,
                skill_id=action.skill_id,
                segment_id=action.segment_id,
                heading_delta_rad=heading_delta_rad,
                heading_turn_direction=heading_turn_direction,
                heading_max_angular_speed_rad_s=(
                    heading_max_angular_speed_rad_s
                ),
                heading_max_duration_s=heading_max_duration_s,
            )
        )
        self.dispatch_navigation_event(
            "navigation_heading_goal",
            self._action_identity(action),
        )
        self.show_command(
            command,
            "NAVIGATION_HEADING_GOAL",
            "heading goal accepted",
            heading_delta_rad=heading_delta_rad,
            heading_turn_direction=heading_turn_direction,
            heading_max_angular_speed_rad_s=heading_max_angular_speed_rad_s,
            heading_max_duration_s=heading_max_duration_s,
        )
        return DispatchDisposition.HANDLED

    def _handle_navigation_agent_status(
        self,
        command: OperatorCommand,
    ) -> DispatchDisposition:
        self._require_source(
            command,
            "lavira_agent",
            "navigation status requires lavira_agent source",
        )
        pending = self.pending_vla_completion
        defer_stop = (
            pending is not None
            and pending.generation == int(command.parameters.get("generation", -1))
            and pending.skill_id == int(command.parameters.get("skill_id", 0))
            and command.parameters.get("state") == "reached"
            and command.parameters.get("reason") == "manipulation_completed"
        )
        if not self.navigation.accept_status(command.parameters, owner="lavira", defer_vla_stop=defer_stop):
            raise ValueError("stale navigation agent status")
        generation = int(command.parameters["generation"])
        self.experiment.finish(generation, str(command.parameters.get("state", "failed")),
                               str(command.parameters.get("reason", "")))
        if not defer_stop and (self.experiment.config or pending is not None):
            # A model exception can leave a VLA or BasePose worker active.
            # Invalidate that generation on every experimental terminal result.
            action = self.navigation.handle_key(" ", now=time.monotonic(), cancel_reason="experiment_terminal")
            self.publish_navigation_action(action, source="experiment_terminal")
        if not defer_stop:
            self._send_navigation_stop(
                generation=generation,
                skill_id=int(command.parameters.get("skill_id", 0)),
                segment_id=int(command.parameters.get("segment_id", 0)),
            )
        self.dispatch_navigation_event(
            "navigation_status", dict(command.parameters)
        )
        state = str(command.parameters.get("state", "failed"))
        level = logging.ERROR if state == "failed" else logging.INFO
        self.show_command(command, "NAVIGATION_STATUS", state, level)
        return DispatchDisposition.HANDLED

    def _handle_experiment_vla_started(self, command):
        self._require_source(command, "vla_service", "VLA first-action event requires vla_service")
        self.experiment.first_action(command.parameters)
        return DispatchDisposition.HANDLED

    def _handle_experiment_nav_step(self, command):
        self._require_source(command, "lavira_agent", "NaVILA actions require lavira_agent")
        self.experiment.begin_step(command.parameters)
        return DispatchDisposition.HANDLED

    def _handle_experiment_perturbation(self, command):
        self._require_source(command, "operator_console", "Disturbance markers require the operator console")
        from gear_sonic.experiments.recording import read_events
        if not self.experiment.active:
            raise ValueError("No active experiment trial")
        protocol = self.experiment.config["condition"].get("perturbation", {})
        recorder = self.experiment.recorder
        generation = self.experiment.active[0]
        events = [e for e in read_events(recorder.path) if e.get("trial_id") == recorder.trial_id(generation)]
        if not protocol.get("enabled") or not any(e["type"] == "perturbation_cue" for e in events):
            raise ValueError("Wait for this trial's configured perturbation cue")
        if any(e["type"] == "perturbation" for e in events):
            raise ValueError("This trial's disturbance is already marked")
        recorder.write("perturbation", generation=generation, protocol=protocol,
                       gate=self.experiment.config["gate_under_test"])
        return DispatchDisposition.HANDLED

    def _handle_start_base_pose(
        self,
        command: OperatorCommand,
    ) -> DispatchDisposition:
        if "surface" in command.parameters:
            raise ValueError(
                "legacy BasePose surface field is not supported; use "
                "yaw_align_target"
            )
        if command.metadata.source != "lavira_agent":
            # Operator-key BasePose start has already been normalized by
            # NavigationControlState and must still reach the consumer.
            return DispatchDisposition.FORWARD_ORIGINAL
        for name in ("target", "yaw_align_target"):
            value = command.parameters.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"LaViRA BasePose start requires non-empty {name}"
                )
        action = self.navigation.accept_base_pose_start(command.parameters)
        self._send_navigation_stop(
            generation=action.generation,
            skill_id=action.skill_id,
            segment_id=action.segment_id,
        )
        self.dispatch_navigation_event(
            "start_base_pose", dict(command.parameters)
        )
        return DispatchDisposition.HANDLED

    def _handle_vla_task(
        self,
        command: OperatorCommand,
    ) -> DispatchDisposition:
        self._require_source(
            command,
            "lavira_agent",
            "VLA task command requires lavira_agent source",
        )
        if not self.navigation.accept_vla_command(
            command.name, command.parameters
        ):
            raise ValueError("stale or unexpected VLA task command")
        if command.name == "stop_vla_task" and command.parameters.get("reason") == "postcondition_satisfied":
            # Close the task clock at VA completion, while the independent VLA
            # service keeps inferring and publishing for five more seconds.
            # Keep navigation ownership until the timer or an operator cancels
            # it; terminal-result cleanup must not cut this interval short.
            duration_s = 5.0
            generation, skill_id = self.navigation.generation, self.navigation.skill_id
            self.pending_vla_completion = PendingVlaCompletion(
                generation, skill_id, time.monotonic() + duration_s, dict(command.parameters),
            )
            self.navigation.accept_status(
                {**command.parameters, "state": "reached"}, owner="lavira", defer_vla_stop=True,
            )
            self.experiment.finish(generation, "reached", "manipulation_completed")
            self.experiment.recorder.write(
                "vla_completion_delay_started", generation=generation, skill_id=skill_id,
                window_id=command.parameters.get("window_id"), duration_s=duration_s,
            )
            self.show_command(
                command, "VLA_COMPLETION_DELAY_STARTED",
                "VA confirmed completion; continuing VLA for 5 seconds before PLANNER",
                duration_s=duration_s,
            )
            return DispatchDisposition.HANDLED
        self.experiment.recorder.write("vla_command", generation=self.navigation.generation,
            command=command.name, skill_id=self.navigation.skill_id,
            window_id=command.parameters.get("window_id"))
        if command.name == "start_vla_task":
            self._send_navigation_stop(
                generation=self.navigation.generation,
                skill_id=self.navigation.skill_id,
                segment_id=self.navigation.segment_id,
            )
        self.dispatch_navigation_event(command.name, dict(command.parameters))
        return DispatchDisposition.HANDLED

    def _handle_base_pose_velocity(
        self,
        command: OperatorCommand,
    ) -> DispatchDisposition:
        self._require_source(
            command,
            "base_pose_agent",
            "base-pose velocity requires base_pose_agent source",
        )
        action = self.navigation.accept_base_pose_velocity(
            command.parameters,
            now=time.monotonic(),
        )
        self.publish_navigation_action(
            action,
            source=command.metadata.source,
        )
        self.dispatch_navigation_event(
            BASE_POSE_RUNTIME_STATUS_COMMAND,
            build_base_pose_runtime_status(action, command.parameters),
        )
        return DispatchDisposition.HANDLED

    def _handle_base_pose_status(
        self,
        command: OperatorCommand,
    ) -> DispatchDisposition:
        self._require_source(
            command,
            "base_pose_agent",
            "base-pose status requires base_pose_agent source",
        )
        if not self.navigation.accept_status(
            command.parameters,
            owner="base_pose",
            agent_final=not self.navigation.lavira_task_active,
        ):
            raise ValueError("stale base-pose status")
        generation = int(command.parameters["generation"])
        skill_id = int(command.parameters.get("skill_id", 0))
        segment_id = int(command.parameters.get("segment_id", 0))
        self._send_navigation_stop(
            generation=generation,
            skill_id=skill_id,
            segment_id=segment_id,
        )
        self.dispatch_navigation_event(
            BASE_POSE_RUNTIME_STATUS_COMMAND,
            {
                "generation": generation,
                "skill_id": skill_id,
                "segment_id": segment_id,
                "state": str(command.parameters.get("state", "failed")),
                "reason": str(command.parameters.get("reason", "")),
                "velocity": [0.0, 0.0, 0.0],
            },
        )
        if self.navigation.lavira_task_active:
            self.dispatch_navigation_event(
                "base_pose_status", dict(command.parameters)
            )
        state = str(command.parameters.get("state", "failed"))
        level = logging.ERROR if state == "failed" else logging.INFO
        self.show_command(command, "BASE_POSE_STATUS", state, level)
        return DispatchDisposition.HANDLED

    def _handle_select_pose_mode(
        self,
        command: OperatorCommand,
    ) -> DispatchDisposition:
        self.publish_navigation_action(
            self.navigation.handle_key(
                " ",
                now=time.monotonic(),
                cancel_reason="select_pose_mode",
            ),
            source=command.metadata.source,
            input_key="POSE mode",
        )
        self.show_command(command, "POSE_MODE", "POSE mode selected")
        return DispatchDisposition.FORWARD_ORIGINAL


def run_control_gateway(profile: RuntimeProfile) -> None:
    runtime = ControlGatewayRuntime(profile)
    try:
        runtime.run()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()


def main() -> None:
    args = build_argument_parser().parse_args()
    run_control_gateway(
        load_runtime_profile(args.profile or None, overlays=tuple(args.overlay))
    )


if __name__ == "__main__":
    main()
