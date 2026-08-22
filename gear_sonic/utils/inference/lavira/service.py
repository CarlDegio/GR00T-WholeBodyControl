#!/usr/bin/env python3
"""Unified LaViRA manipulation agent behind the ControlGateway."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
import queue
import threading
import time
from typing import Any, Callable, Literal, Mapping

import zmq

from gear_sonic.runtime.gateway.control_client import (
    ControlGatewayIntentClient,
    ControlGatewaySubscriber,
)
from gear_sonic.runtime.profile import (
    RuntimeProfileSelection,
    load_component_config,
    load_runtime_profile,
)
from gear_sonic.runtime.telemetry import (
    build_event,
    configure_file_logging,
    emit_event,
    open_telemetry_publisher,
)
from gear_sonic.utils.inference.lavira.agent import (
    DEFAULT_LA_MODEL,
    DEFAULT_VA_MODEL,
    LaViRAAgent,
    LaViRAAgentCancelled,
    LaViRAClient,
    LaViRATaskResult,
)
from gear_sonic.utils.inference.lavira.object_nav import (
    ObjectNavResult,
    SensorGatewayRGBDCamera,
)

LOGGER = logging.getLogger("sonic.lavira")
EventReporter = Callable[..., None]


class LaviraRuntimeEventHandler(logging.Handler):
    """Promote every otherwise-unreported LaViRA warning/error to telemetry."""

    def __init__(self, pending_events: queue.SimpleQueue[dict[str, object]]) -> None:
        super().__init__(level=logging.WARNING)
        self.pending_events = pending_events

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "runtime_event_emitted", False):
            return
        try:
            fields: dict[str, object] = {"logger": record.name}
            if record.exc_info:
                fields["exception"] = logging.Formatter().formatException(
                    record.exc_info
                )
            self.pending_events.put(
                build_event(
                    "lavira",
                    record.levelno,
                    f"LOG_{record.levelname}",
                    record.getMessage(),
                    **fields,
                )
            )
        except Exception:
            self.handleError(record)


@dataclass
class LaviraPlannerConfig:
    mission: str
    global_target: str
    navigation_mode: Literal["vln", "object_nav"] = "object_nav"
    max_steps: int = 20
    history_size: int = 5
    la_model: str = DEFAULT_LA_MODEL
    la_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    la_enable_thinking: bool = True
    la_timeout_seconds: float = 180.0
    va_model: str = DEFAULT_VA_MODEL
    va_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    va_enable_thinking: bool = False
    va_timeout_seconds: float = 180.0
    camera_timeout_ms: int = 15000
    sensor_gateway_request_timeout_ms: int = 100
    sensor_gateway_max_age_ms: float = 1000.0
    sensor_gateway_max_skew_ms: float = 5.0
    min_confidence: float = 0.6
    nav_handoff_min_depth_m: float = 0.3
    nav_handoff_max_depth_m: float = 3.0
    alignment_head_camera_stream: str = "ego_view"
    segment_timeout_seconds: float = 180.0
    manipulation_window_seconds: float = 5.0
    manipulation_max_windows: int = 12
    manipulation_timeout_seconds: float = 180.0
    vla_start_timeout_seconds: float = 3.0
    profile: str = ""
    overlay: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.navigation_mode not in {"vln", "object_nav"}:
            raise ValueError("components.lavira.navigation_mode must be vln or object_nav")
        if not self.mission.strip():
            raise ValueError("components.lavira.mission is required")
        if not self.global_target.strip():
            raise ValueError("components.lavira.global_target is required")
        if self.max_steps <= 0 or self.history_size <= 0:
            raise ValueError("LaViRA max_steps and history_size must be positive")
        if (
            not math.isfinite(float(self.nav_handoff_min_depth_m))
            or not math.isfinite(float(self.nav_handoff_max_depth_m))
            or self.nav_handoff_min_depth_m <= 0.0
            or self.nav_handoff_max_depth_m <= self.nav_handoff_min_depth_m
            or self.nav_handoff_max_depth_m > 8.0
        ):
            raise ValueError("LaViRA NAV handoff depth range is invalid")
        if not self.alignment_head_camera_stream.strip():
            raise ValueError("LaViRA ALIGN handoff head camera stream is required")
        if (
            self.manipulation_window_seconds <= 0
            or self.manipulation_max_windows <= 0
            or self.manipulation_timeout_seconds <= 0
            or self.vla_start_timeout_seconds <= 0
        ):
            raise ValueError("LaViRA manipulation limits must be positive")


def load_lavira_config(
    profile: str = "", overlays: tuple[str, ...] = ()
) -> LaviraPlannerConfig:
    return load_component_config(
        LaviraPlannerConfig,
        "lavira",
        profile or None,
        overlays=overlays,
    )


def parse_lavira_config(args: list[str] | None = None) -> LaviraPlannerConfig:
    import tyro

    selection = tyro.cli(RuntimeProfileSelection, args=args)
    return load_lavira_config(selection.profile, selection.overlay)


@dataclass(frozen=True)
class WorkerResult:
    generation: int
    result: LaViRATaskResult | ObjectNavResult | None
    error: str | None


class LaviraPlannerRuntime:
    """Generation gate plus segment-scoped NavDP status mailbox."""

    def __init__(
        self,
        config: LaviraPlannerConfig,
        *,
        submit_intent: Callable[[str, Mapping[str, object]], None],
        report_event: EventReporter | None = None,
    ) -> None:
        self.config = config
        self.submit_intent = submit_intent
        self.report_event = report_event
        self.generation = 0
        self.state = "listen_wasd"
        self.pending_generation: int | None = None
        self.requests: queue.Queue[int | None] = queue.Queue(maxsize=1)
        self.results: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self._condition = threading.Condition()
        self._terminal_status: dict[tuple[int, int, int], dict[str, Any]] = {}

    def _event(
        self,
        level: int,
        code: str,
        message: str,
        **fields: object,
    ) -> None:
        if self.report_event is None:
            return
        try:
            self.report_event(level, code, message, **fields)
        except Exception:
            LOGGER.exception(
                "LaViRA runtime event reporter failed code=%s",
                code,
            )

    def report_todo(self, generation: int, step: int, todo_list: str) -> None:
        """Publish TODO state for the persistent ControlGateway Events pane."""

        self._event(
            logging.INFO,
            "TODO_UPDATED",
            "LaViRA TODO updated",
            generation=generation,
            step=step,
            todo_list=todo_list,
        )

    @property
    def phase(self) -> str:
        return self.state

    @staticmethod
    def _discard_queued(queue_: queue.Queue) -> None:
        try:
            queue_.get_nowait()
        except queue.Empty:
            pass

    @classmethod
    def _put_latest(cls, queue_: queue.Queue, item: Any) -> None:
        cls._discard_queued(queue_)
        queue_.put_nowait(item)

    def is_cancelled(self, generation: int) -> bool:
        return self.stop_event.is_set() or generation != self.generation

    def publish_worker_result(self, item: WorkerResult) -> None:
        self._put_latest(self.results, item)

    def cancel(self, generation: int, reason: str) -> None:
        if generation < self.generation:
            self._event(
                logging.WARNING,
                "STALE_CANCEL_IGNORED",
                "Stale LaViRA cancellation was ignored",
                generation=generation,
                active_generation=self.generation,
                reason=reason,
            )
            return
        with self._condition:
            self.generation = generation
            self.pending_generation = None
            self.state = "listen_wasd"
            self._terminal_status.clear()
            self._discard_queued(self.requests)
            self._condition.notify_all()
        LOGGER.info("LISTEN_WASD reason=%s", reason)
        self._event(
            logging.WARNING,
            "TASK_CANCEL_REQUESTED",
            "LaViRA task cancellation accepted",
            generation=generation,
            reason=reason,
        )

    def start_navigation(self, generation: int) -> bool:
        with self._condition:
            if generation <= self.generation or self.pending_generation is not None:
                self._event(
                    logging.WARNING,
                    "TASK_START_REJECTED",
                    "Duplicate or busy LaViRA task start rejected",
                    generation=generation,
                    active_generation=self.generation,
                    pending_generation=self.pending_generation,
                )
                return False
            self.generation = generation
            self.pending_generation = generation
            self.state = "nav"
            self._terminal_status.clear()
            self._put_latest(self.requests, generation)
            self._condition.notify_all()
        LOGGER.info(
            "NAV generation=%s navigation_mode=%s mission=%s",
            generation,
            self.config.navigation_mode,
            self.config.mission,
        )
        self._event(
            logging.INFO,
            "TASK_ACCEPTED",
            "LaViRA task accepted",
            generation=generation,
            navigation_mode=self.config.navigation_mode,
            mission=self.config.mission,
            global_target=self.config.global_target,
        )
        return True

    def accept_status(
        self,
        message: str | bytes | Mapping[str, Any],
        *,
        channel: str = "",
    ) -> bool:
        payload = json.loads(message) if isinstance(message, (str, bytes)) else dict(message)
        generation = int(payload.get("generation", -1))
        skill_id = int(payload.get("skill_id", 0))
        segment_id = int(payload.get("segment_id", 0))
        if generation != self.generation or self.pending_generation != generation:
            self._event(
                logging.WARNING,
                "STALE_STATUS_IGNORED",
                "Stale LaViRA controller status ignored",
                generation=generation,
                active_generation=self.generation,
                skill_id=skill_id,
                segment_id=segment_id,
                state=payload.get("state"),
            )
            return False
        state = str(payload.get("state", ""))
        self._event(
            logging.ERROR if state == "failed" else logging.INFO,
            "CONTROLLER_STATUS",
            f"Controller reported {state or 'unknown'}",
            generation=generation,
            skill_id=skill_id,
            segment_id=segment_id,
            state=state,
            reason=payload.get("reason"),
            status_type=payload.get("type"),
            channel=channel,
        )
        if state in {"reached", "failed", "stopped"} or (
            channel == "vla_task_status" and state == "active"
        ):
            with self._condition:
                self._terminal_status[(generation, skill_id, segment_id)] = payload
                self._condition.notify_all()
        return True

    def wait_status(
        self, generation: int, *identity_and_timeout: float
    ) -> Mapping[str, Any]:
        if len(identity_and_timeout) == 2:
            skill_id = 0
            segment_id, timeout_s = identity_and_timeout
        elif len(identity_and_timeout) == 3:
            skill_id, segment_id, timeout_s = identity_and_timeout
        else:
            raise TypeError(
                "wait_status expects generation, [skill_id,] segment_id, timeout"
            )
        deadline = time.monotonic() + float(timeout_s)
        key = (int(generation), int(skill_id), int(segment_id))
        with self._condition:
            while True:
                if self.is_cancelled(generation):
                    raise LaViRAAgentCancelled("operator_cancelled")
                status = self._terminal_status.pop(key, None)
                if status is not None:
                    return status
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"segment_{segment_id}_timeout")
                self._condition.wait(timeout=min(0.05, remaining))

    def pop_failure(
        self, generation: int, skill_id: int
    ) -> Mapping[str, Any] | None:
        """Return an asynchronous BasePose/VLA safety failure, if present."""
        with self._condition:
            for key, status in tuple(self._terminal_status.items()):
                if (
                    key[:2] == (int(generation), int(skill_id))
                    and status.get("state") in {"failed", "stopped"}
                ):
                    return self._terminal_status.pop(key)
        return None

    def tick(self) -> None:
        while True:
            try:
                item = self.results.get_nowait()
            except queue.Empty:
                return
            if (
                item.generation != self.generation
                or item.generation != self.pending_generation
            ):
                self._event(
                    logging.WARNING,
                    "STALE_WORKER_RESULT_IGNORED",
                    "Stale LaViRA worker result ignored",
                    generation=item.generation,
                    active_generation=self.generation,
                    pending_generation=self.pending_generation,
                )
                continue
            self.pending_generation = None
            self.state = "listen_wasd"
            if isinstance(item.result, ObjectNavResult):
                if item.result.outcome == "NAVIGATE":
                    policy = item.result.policy
                    self.submit_intent(
                        "navigation_goal",
                        {
                            "generation": item.generation,
                            "skill_id": 0,
                            "segment_id": 0,
                            "goal_base": result_to_goal(item.result),
                            "target": str(policy.get("target", self.config.global_target)),
                            "target_type": str(policy.get("target_type", "global_target")),
                            "confidence": float(policy.get("confidence", 0.0)),
                        },
                    )
                return
            result = item.result or LaViRATaskResult(
                item.generation,
                "failed",
                item.error or "agent_failed",
                0,
                0,
                0,
                self.config.navigation_mode,
            )
            parameters: dict[str, object] = asdict(result)
            self.submit_intent("navigation_agent_status", parameters)
            LOGGER.info(
                "TASK_RESULT %s",
                json.dumps(parameters, ensure_ascii=False, sort_keys=True),
            )
            self._event(
                logging.ERROR if result.state == "failed" else logging.INFO,
                "TASK_RESULT_PUBLISHED",
                "LaViRA task result published to ControlGateway",
                **parameters,
            )

    def shutdown(self) -> None:
        self.stop_event.set()
        with self._condition:
            self._condition.notify_all()
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass


def run_agent_worker(
    factory: Callable[[], LaViRAAgent], runtime: LaviraPlannerRuntime
) -> None:
    agent: LaViRAAgent | None = None
    while not runtime.stop_event.is_set():
        generation = runtime.requests.get()
        if generation is None:
            break
        try:
            agent = agent or factory()
            item = WorkerResult(generation, agent.run(generation), None)
        except Exception as exc:
            LOGGER.exception(
                "LaViRA worker failed",
                extra={"runtime_event_emitted": True},
            )
            runtime._event(
                logging.ERROR,
                "WORKER_FAILED",
                "LaViRA worker failed",
                generation=generation,
                error=str(exc),
            )
            item = WorkerResult(generation, None, str(exc))
        runtime.publish_worker_result(item)
    camera = getattr(agent, "camera", None)
    if camera is not None and hasattr(camera, "close"):
        camera.close()


def _agent(
    config: LaviraPlannerConfig,
    sensor_gateway: str,
    runtime: LaviraPlannerRuntime,
) -> LaViRAAgent:
    camera = SensorGatewayRGBDCamera(
        sensor_gateway,
        timeout_ms=config.camera_timeout_ms,
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        max_skew_ms=config.sensor_gateway_max_skew_ms,
    )
    client = LaViRAClient(
        la_base_url=config.la_base_url,
        va_base_url=config.va_base_url,
        la_model=config.la_model,
        va_model=config.va_model,
        la_enable_thinking=config.la_enable_thinking,
        va_enable_thinking=config.va_enable_thinking,
        la_timeout_seconds=config.la_timeout_seconds,
        va_timeout_seconds=config.va_timeout_seconds,
    )
    return LaViRAAgent(
        navigation_mode=config.navigation_mode,
        mission=config.mission,
        global_target=config.global_target,
        max_steps=config.max_steps,
        history_size=config.history_size,
        min_confidence=config.min_confidence,
        segment_timeout_seconds=config.segment_timeout_seconds,
        camera=camera,
        client=client,
        submit_intent=runtime.submit_intent,
        wait_status=runtime.wait_status,
        cancelled=runtime.is_cancelled,
        manipulation_window_seconds=config.manipulation_window_seconds,
        manipulation_max_windows=config.manipulation_max_windows,
        manipulation_timeout_seconds=config.manipulation_timeout_seconds,
        vla_start_timeout_seconds=config.vla_start_timeout_seconds,
        poll_failure=runtime.pop_failure,
        nav_handoff_min_depth_m=config.nav_handoff_min_depth_m,
        nav_handoff_max_depth_m=config.nav_handoff_max_depth_m,
        alignment_head_camera_stream=config.alignment_head_camera_stream,
        report_event=runtime.report_event,
        report_todo=runtime.report_todo,
    )


def main(config: LaviraPlannerConfig) -> None:
    configure_file_logging("lavira")
    profile = load_runtime_profile(config.profile or None, overlays=config.overlay)
    event_socket = open_telemetry_publisher(
        profile.endpoint_uri("runtime_event_ingress")
    )
    metrics_socket = open_telemetry_publisher(
        profile.endpoint_uri("runtime_metrics_ingress")
    )
    pending_events: queue.SimpleQueue[dict[str, object]] = queue.SimpleQueue()
    forwarding_handler = LaviraRuntimeEventHandler(pending_events)
    LOGGER.addHandler(forwarding_handler)

    def report_event(
        level: int,
        code: str,
        message: str,
        **fields: object,
    ) -> None:
        payload = build_event("lavira", level, code, message, **fields)
        LOGGER.log(
            level,
            "%s | %s | %s",
            code,
            message,
            payload["fields"],
            extra={"runtime_event_emitted": True},
        )
        pending_events.put(payload)

    def flush_events() -> None:
        while True:
            try:
                emit_event(pending_events.get_nowait(), socket=event_socket)
            except queue.Empty:
                return

    context = zmq.Context.instance()
    intent = ControlGatewayIntentClient(
        profile.endpoint_uri("control_gateway_intent"),
        source="lavira_agent",
        context=context,
    )
    runtime = LaviraPlannerRuntime(
        config,
        submit_intent=lambda name, parameters: intent.send(name, parameters),
        report_event=report_event,
    )
    control_gateway = ControlGatewaySubscriber(
        profile.endpoint_uri("control_gateway_dispatch"),
        context=context,
        accepted_names={
            "start_navigation",
            "cancel_navigation",
            "navigation_status",
            "base_pose_status",
            "vla_task_status",
        },
    )
    worker = threading.Thread(
        target=run_agent_worker,
        args=(
            lambda: _agent(
                config,
                profile.endpoint_uri("sensor_gateway_metadata"),
                runtime,
            ),
            runtime,
        ),
        daemon=True,
    )
    worker.start()
    report_event(
        logging.INFO,
        "READY",
        "LaViRA ready and waiting for ControlGateway commands",
        navigation_mode=config.navigation_mode,
        global_target=config.global_target,
    )
    try:
        while True:
            command = control_gateway.read_command()
            if command is not None:
                generation = int(command.parameters.get("generation", -1))
                if command.name == "start_navigation":
                    runtime.start_navigation(generation)
                elif command.name == "cancel_navigation":
                    runtime.cancel(generation, "operator_stop")
                elif command.name in {
                    "navigation_status", "base_pose_status", "vla_task_status"
                }:
                    runtime.accept_status(
                        command.parameters,
                        channel=command.name,
                    )
            runtime.tick()
            flush_events()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        report_event(logging.INFO, "STOPPED", "LaViRA service stopped")
        runtime.shutdown()
        worker.join(timeout=1.0)
        flush_events()
        control_gateway.close()
        intent.close()
        event_socket.close(linger=0)
        metrics_socket.close(linger=0)
        LOGGER.removeHandler(forwarding_handler)


def result_to_goal(result: ObjectNavResult) -> tuple[float, float]:
    """Compatibility helper for callers of the retired single-cycle runner."""
    distance = float(result.geometry["mean_range"])
    angle = math.radians(float(result.geometry["angle_deg"]))
    return distance * math.cos(angle), distance * math.sin(angle)


def run_inference_worker(factory: Callable[[], Any], runtime: LaviraPlannerRuntime) -> None:
    """Compatibility worker; production uses :func:`run_agent_worker`."""
    runner = None
    try:
        while not runtime.stop_event.is_set():
            generation = runtime.requests.get()
            if generation is None:
                break
            released = False

            def release_depth() -> None:
                nonlocal released
                if not released:
                    released = True
                    runtime.submit_intent(
                        "lavira_rgbd_captured",
                        {"generation": generation},
                    )

            try:
                runner = runner or factory()
                item = WorkerResult(
                    generation,
                    runner.run_once(rgbd_capture_complete=release_depth),
                    None,
                )
            except Exception as exc:
                item = WorkerResult(generation, None, str(exc))
            finally:
                release_depth()
            runtime.publish_worker_result(item)
    finally:
        if runner is not None and hasattr(runner, "close"):
            runner.close()


if __name__ == "__main__":
    main(parse_lavira_config())
