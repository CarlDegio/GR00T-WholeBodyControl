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
from gear_sonic.runtime.inference_service import InferenceServiceContext
from gear_sonic.runtime.queues import discard_queued, replace_latest
from gear_sonic.runtime.profile import (
    load_component_config,
    parse_component_config,
)
from gear_sonic.runtime.telemetry import build_event
from gear_sonic.utils.inference.lavira.agent import (
    DEFAULT_LA_MODEL,
    DEFAULT_VA_MODEL,
    LaViRAAgent,
    LaViRAAgentCancelled,
    LaViRAClient,
    LaViRATaskResult,
)
from gear_sonic.utils.inference.lavira.camera import SensorGatewayRGBDCamera
from gear_sonic.utils.inference.lavira.geometry import MAX_DIRECT_TRAVEL

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
    alignment_prompt: str
    global_target: str = ""
    navigation_mode: Literal["vln", "object_nav"] = "object_nav"
    manipulation_prompt: str = ""
    max_steps: int = 20
    history_size: int = 5
    la_model: str = DEFAULT_LA_MODEL
    la_base_url: str = "https://ws-6yzgj1m087a053ip.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    la_enable_thinking: bool = False
    la_timeout_seconds: float = 180.0
    va_model: str = DEFAULT_VA_MODEL
    va_base_url: str = "https://ws-6yzgj1m087a053ip.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
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
    heading_settle_seconds: float = 1.0
    heading_settle_samples: int = 30
    heading_settle_bad_sample_threshold: int = 12
    heading_settle_tolerance_rad: float = math.radians(5.0)
    heading_correction_speed_rad_s: float = 0.2
    heading_correction_timeout_seconds: float = 10.0
    manipulation_window_seconds: float = 5.0
    manipulation_max_windows: int = 36
    manipulation_timeout_seconds: float = 180.0
    vla_start_timeout_seconds: float = 25.0
    profile: str = ""
    overlay: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.navigation_mode not in {"vln", "object_nav"}:
            raise ValueError("components.lavira.navigation_mode must be vln or object_nav")
        if not self.mission.strip():
            raise ValueError("components.lavira.mission is required")
        self.global_target = self.global_target.strip()
        if not self.manipulation_prompt.strip():
            self.manipulation_prompt = self.mission
        self.alignment_prompt = self.alignment_prompt.strip()
        if not self.alignment_prompt:
            raise ValueError("components.lavira.alignment_prompt is required")
        if self.max_steps <= 0 or self.history_size <= 0:
            raise ValueError("LaViRA max_steps and history_size must be positive")
        if (
            not math.isfinite(float(self.nav_handoff_min_depth_m))
            or not math.isfinite(float(self.nav_handoff_max_depth_m))
            or self.nav_handoff_min_depth_m <= 0.0
            or self.nav_handoff_max_depth_m <= self.nav_handoff_min_depth_m
            or self.nav_handoff_max_depth_m > MAX_DIRECT_TRAVEL
        ):
            raise ValueError("LaViRA NAV handoff depth range is invalid")
        if not self.alignment_head_camera_stream.strip():
            raise ValueError("LaViRA ALIGN handoff head camera stream is required")
        if (
            self.manipulation_window_seconds <= 0
            or self.manipulation_max_windows <= 0
            or self.manipulation_timeout_seconds <= 0
            or self.vla_start_timeout_seconds <= 0
            or not math.isfinite(float(self.heading_settle_seconds))
            or self.heading_settle_seconds < 0
            or self.heading_settle_samples <= 0
            or self.heading_settle_bad_sample_threshold <= 0
            or self.heading_settle_bad_sample_threshold > self.heading_settle_samples
            or not math.isfinite(float(self.heading_settle_tolerance_rad))
            or self.heading_settle_tolerance_rad <= 0.0
            or self.heading_settle_tolerance_rad > math.pi
            or not math.isfinite(float(self.heading_correction_speed_rad_s))
            or self.heading_correction_speed_rad_s <= 0.0
            or not math.isfinite(float(self.heading_correction_timeout_seconds))
            or self.heading_correction_timeout_seconds <= 0.0
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
    return parse_component_config(LaviraPlannerConfig, "lavira", args)


@dataclass(frozen=True)
class WorkerResult:
    generation: int
    result: LaViRATaskResult | None
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

    def report_agent_event(self, level: int, code: str, message: str, **fields: object) -> None:
        """Keep late worker events from replacing the operator's terminal result."""
        generation = fields.get("generation")
        if generation is not None and self.is_cancelled(int(generation)):
            return
        self._event(level, code, message, **fields)

    def report_todo(self, generation: int, step: int, todo_list: str) -> None:
        """Publish TODO state for the persistent ControlGateway Events pane."""

        self.report_agent_event(
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

    def is_cancelled(self, generation: int) -> bool:
        return self.stop_event.is_set() or generation != self.generation

    def publish_worker_result(self, item: WorkerResult) -> None:
        with self._condition:
            if getattr(self, "experiment", None) and self.is_cancelled(item.generation):
                return
            replace_latest(self.results, item)

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
            discard_queued(self.requests)
            self._condition.notify_all()
        LOGGER.info("LISTEN_WASD reason=%s", reason)
        if reason == "operator_success":
            self._event(
                logging.INFO, "TASK_WORKER_RELEASED",
                "LaViRA worker released after operator-confirmed success",
                generation=generation, reason=reason,
            )
            return
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
            replace_latest(self.requests, generation)
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
            **({k: payload[k] for k in ('errors', 'control_source_stream', 'longitudinal_error_m', 'bearing_error_deg')
                if k in payload} if getattr(self, 'experiment', None) else {}),
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
    if getattr(runtime, "experiment", None):
        def run_trial(generation):
            trial_agent = None
            try:
                if runtime.is_cancelled(generation):
                    return
                trial_agent = factory()
                item = WorkerResult(generation, trial_agent.run(generation), None)
            except Exception as exc:
                item = WorkerResult(generation, None, str(exc))
            finally:
                if trial_agent is not None:
                    trial_agent.camera.close()
            runtime.publish_worker_result(item)
        while not runtime.stop_event.is_set():
            generation = runtime.requests.get()
            if generation is None:
                break
            # Each trial owns its model clients and camera cursor. A cancelled
            # request may finish late without delaying or changing the next trial.
            threading.Thread(target=run_trial, args=(generation,), daemon=True,
                             name=f"experiment-agent-{generation}").start()
        return
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
    from gear_sonic.runtime.profile import load_runtime_profile
    from gear_sonic.experiments.config import settings
    from gear_sonic.experiments.agent import ExperimentAgent
    experiment = settings(load_runtime_profile(config.profile or None, overlays=config.overlay))
    agent_type = ExperimentAgent if experiment else LaViRAAgent
    extra = {"experiment": experiment} if experiment else {}
    if experiment:
        client.save_request_context = False
    return agent_type(
        **extra,
        navigation_mode=config.navigation_mode,
        mission=config.mission,
        global_target=config.global_target,
        manipulation_prompt=config.manipulation_prompt,
        alignment_prompt=config.alignment_prompt,
        max_steps=config.max_steps,
        history_size=config.history_size,
        min_confidence=config.min_confidence,
        segment_timeout_seconds=config.segment_timeout_seconds,
        heading_settle_seconds=config.heading_settle_seconds,
        heading_settle_samples=config.heading_settle_samples,
        heading_settle_bad_sample_threshold=(
            config.heading_settle_bad_sample_threshold
        ),
        heading_settle_tolerance_rad=config.heading_settle_tolerance_rad,
        heading_correction_speed_rad_s=(
            config.heading_correction_speed_rad_s
        ),
        heading_correction_timeout_seconds=(
            config.heading_correction_timeout_seconds
        ),
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
        report_event=runtime.report_agent_event,
        report_todo=runtime.report_todo,
    )


def main(config: LaviraPlannerConfig) -> None:
    service = InferenceServiceContext(
        "lavira",
        config,
        enable_metrics=False,
    )
    profile = service.profile
    forwarding_handler = LaviraRuntimeEventHandler(service.pending_events)
    LOGGER.addHandler(forwarding_handler)

    def report_event(
        level: int,
        code: str,
        message: str,
        **fields: object,
    ) -> None:
        service.experiment.runtime("lavira", code, **fields)
        payload = build_event("lavira", level, code, message, **fields)
        LOGGER.log(
            level,
            "%s | %s | %s",
            code,
            message,
            payload["fields"],
            extra={"runtime_event_emitted": True},
        )
        service.pending_events.put(payload)

    flush_events = service.flush_events

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
    runtime.experiment = service.experiment.config
    experiment_intents = queue.Queue()
    if runtime.experiment:
        # Keep the ZMQ socket and command sequence owned by the service thread.
        runtime.submit_intent = lambda name, parameters: experiment_intents.put((name, dict(parameters)))
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
                    runtime.cancel(generation, str(command.parameters.get("reason") or "operator_stop"))
                elif command.name in {
                    "navigation_status", "base_pose_status", "vla_task_status"
                }:
                    runtime.accept_status(
                        command.parameters,
                        channel=command.name,
                    )
            while not experiment_intents.empty():
                name, parameters = experiment_intents.get_nowait()
                if not runtime.is_cancelled(int(parameters.get("generation", -1))):
                    intent.send(name, parameters)
            runtime.tick()
            flush_events()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        report_event(logging.INFO, "STOPPED", "LaViRA service stopped")
        runtime.shutdown()
        worker.join(timeout=1.0)
        control_gateway.close()
        intent.close()
        LOGGER.removeHandler(forwarding_handler)
        service.close()
if __name__ == "__main__":
    main(parse_lavira_config())
