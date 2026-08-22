#!/usr/bin/env python3
"""Uni-LaViRA three-task agent behind the unified ControlGateway."""

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
from gear_sonic.runtime.telemetry import configure_file_logging, open_telemetry_publisher
from gear_sonic.utils.inference.lavira.agent import (
    DEFAULT_MODEL,
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


@dataclass
class LaviraPlannerConfig:
    mission: str
    global_target: str
    task_type: Literal["vln", "object_nav", "eqa"] = "object_nav"
    question: str = ""
    max_steps: int = 20
    history_size: int = 5
    la_model: str = DEFAULT_MODEL
    la_base_url: str = "http://127.0.0.1:8000/v1"
    la_timeout_seconds: float = 180.0
    va_model: str = DEFAULT_MODEL
    va_base_url: str = "http://127.0.0.1:8001/v1"
    va_timeout_seconds: float = 180.0
    camera_timeout_ms: int = 15000
    sensor_gateway_request_timeout_ms: int = 100
    sensor_gateway_max_age_ms: float = 1000.0
    sensor_gateway_max_skew_ms: float = 5.0
    min_confidence: float = 0.6
    segment_timeout_seconds: float = 180.0
    profile: str = ""
    overlay: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task_type not in {"vln", "object_nav", "eqa"}:
            raise ValueError("components.lavira.task_type must be vln, object_nav, or eqa")
        if not self.mission.strip():
            raise ValueError("components.lavira.mission is required")
        if not self.global_target.strip():
            raise ValueError("components.lavira.global_target is required")
        if self.task_type == "eqa" and not self.question.strip():
            raise ValueError("components.lavira.question is required for EQA")
        if self.max_steps <= 0 or self.history_size <= 0:
            raise ValueError("LaViRA max_steps and history_size must be positive")


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
    ) -> None:
        self.config = config
        self.submit_intent = submit_intent
        self.generation = 0
        self.state = "listen_wasd"
        self.pending_generation: int | None = None
        self.requests: queue.Queue[int | None] = queue.Queue(maxsize=1)
        self.results: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self._condition = threading.Condition()
        self._terminal_status: dict[tuple[int, int], dict[str, Any]] = {}

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
            return
        with self._condition:
            self.generation = generation
            self.pending_generation = None
            self.state = "listen_wasd"
            self._terminal_status.clear()
            self._discard_queued(self.requests)
            self._condition.notify_all()
        LOGGER.info("LISTEN_WASD reason=%s", reason)

    def start_navigation(self, generation: int) -> bool:
        with self._condition:
            if generation <= self.generation or self.pending_generation is not None:
                return False
            self.generation = generation
            self.pending_generation = generation
            self.state = "nav"
            self._terminal_status.clear()
            self._put_latest(self.requests, generation)
            self._condition.notify_all()
        LOGGER.info(
            "NAV generation=%s task_type=%s mission=%s",
            generation,
            self.config.task_type,
            self.config.mission,
        )
        return True

    def accept_status(self, message: str | bytes | Mapping[str, Any]) -> bool:
        payload = json.loads(message) if isinstance(message, (str, bytes)) else dict(message)
        generation = int(payload.get("generation", -1))
        segment_id = int(payload.get("segment_id", 0))
        if generation != self.generation or self.pending_generation != generation:
            return False
        state = str(payload.get("state", ""))
        if state in {"reached", "failed", "stopped"}:
            with self._condition:
                self._terminal_status[(generation, segment_id)] = payload
                self._condition.notify_all()
        return True

    def wait_status(
        self, generation: int, segment_id: int, timeout_s: float
    ) -> Mapping[str, Any]:
        deadline = time.monotonic() + float(timeout_s)
        key = (int(generation), int(segment_id))
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
                self.config.task_type,
            )
            parameters: dict[str, object] = asdict(result)
            self.submit_intent("navigation_agent_status", parameters)
            LOGGER.info(
                "TASK_RESULT %s",
                json.dumps(parameters, ensure_ascii=False, sort_keys=True),
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
            LOGGER.exception("LaViRA worker failed")
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
        la_timeout_seconds=config.la_timeout_seconds,
        va_timeout_seconds=config.va_timeout_seconds,
    )
    return LaViRAAgent(
        task_type=config.task_type,
        mission=config.mission,
        global_target=config.global_target,
        question=config.question,
        max_steps=config.max_steps,
        history_size=config.history_size,
        min_confidence=config.min_confidence,
        segment_timeout_seconds=config.segment_timeout_seconds,
        camera=camera,
        client=client,
        submit_intent=runtime.submit_intent,
        wait_status=runtime.wait_status,
        cancelled=runtime.is_cancelled,
    )


def main(config: LaviraPlannerConfig) -> None:
    configure_file_logging("lavira")
    profile = load_runtime_profile(config.profile or None, overlays=config.overlay)
    metrics_socket = open_telemetry_publisher(
        profile.endpoint_uri("runtime_metrics_ingress")
    )
    context = zmq.Context.instance()
    intent = ControlGatewayIntentClient(
        profile.endpoint_uri("control_gateway_intent"),
        source="lavira_agent",
        context=context,
    )
    runtime = LaviraPlannerRuntime(
        config,
        submit_intent=lambda name, parameters: intent.send(name, parameters),
    )
    control_gateway = ControlGatewaySubscriber(
        profile.endpoint_uri("control_gateway_dispatch"),
        context=context,
        accepted_names={"start_navigation", "cancel_navigation", "navigation_status"},
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
    LOGGER.info("READY waiting for ControlGateway commands")
    try:
        while True:
            command = control_gateway.read_command()
            if command is not None:
                generation = int(command.parameters.get("generation", -1))
                if command.name == "start_navigation":
                    runtime.start_navigation(generation)
                elif command.name == "cancel_navigation":
                    runtime.cancel(generation, "operator_stop")
                elif command.name == "navigation_status":
                    runtime.accept_status(command.parameters)
            runtime.tick()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.shutdown()
        worker.join(timeout=1.0)
        control_gateway.close()
        intent.close()
        metrics_socket.close(linger=0)
        LOGGER.info("STOPPED")


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
