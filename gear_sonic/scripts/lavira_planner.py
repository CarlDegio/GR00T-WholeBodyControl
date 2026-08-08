#!/usr/bin/env python3
"""LaViRA semantic target selector behind the unified ControlGateway."""

from __future__ import annotations

from dataclasses import dataclass
import math
import queue
import threading
import time
from typing import Any, Callable, Literal, Mapping

import zmq

from gear_sonic.runtime.control_client import (
    ControlGatewayIntentClient,
    ControlGatewaySubscriber,
)
from gear_sonic.utils.inference.object_nav import (
    ObjectNavConfig,
    ObjectNavResult,
    ObjectNavRunner,
    SensorGatewayRGBDCamera,
)


@dataclass
class LaviraPlannerConfig:
    mission: str
    global_target: str
    model: str = "gpt-5.6-luna"
    vision_backend: Literal["codex", "qwenvl"] = "codex"
    qwenvl_model: str = "qwen3-vl-32b-instruct"
    qwenvl_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    warmup: bool = True
    debug: bool = False
    camera_timeout_ms: int = 15000
    sensor_gateway_endpoint: str = "tcp://127.0.0.1:5560"
    sensor_gateway_request_timeout_ms: int = 100
    sensor_gateway_max_age_ms: float = 1000.0
    sensor_gateway_max_skew_ms: float = 5.0
    codex_timeout_seconds: float = 180.0
    min_confidence: float = 0.6
    output_root: str = "outputs/object_nav"
    control_gateway_endpoint: str = "tcp://127.0.0.1:5565"
    control_gateway_intent_endpoint: str = "tcp://127.0.0.1:5561"


@dataclass(frozen=True)
class WorkerResult:
    generation: int
    result: ObjectNavResult | None
    error: str | None


def result_to_goal(result: ObjectNavResult) -> tuple[float, float]:
    """Use depth-derived range/bearing; x is forward and y is left."""
    distance = float(result.geometry["mean_range"])
    angle = math.radians(float(result.geometry["angle_deg"]))
    return distance * math.cos(angle), distance * math.sin(angle)


class LaviraPlannerRuntime:
    def __init__(
        self,
        config: LaviraPlannerConfig,
        *,
        submit_intent: Callable[[str, Mapping[str, object]], None],
        logger: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.submit_intent = submit_intent
        self.logger = logger
        self.generation = 0
        self.state = "listen_wasd"
        self.pending_generation: int | None = None
        self.requests: queue.Queue[int | None] = queue.Queue(maxsize=1)
        self.results: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

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

    def publish_worker_result(self, item: WorkerResult) -> None:
        self._put_latest(self.results, item)

    def cancel(self, generation: int, reason: str) -> None:
        if generation < self.generation:
            return
        self.generation = generation
        self.pending_generation = None
        self.state = "listen_wasd"
        self._discard_queued(self.requests)
        self.logger(f"[LaViRA] LISTEN_WASD ({reason})")

    def start_navigation(self, generation: int) -> bool:
        if generation <= self.generation or self.pending_generation is not None:
            return False
        self.generation = generation
        self.pending_generation = generation
        self.state = "nav"
        self._put_latest(self.requests, generation)
        self.logger(f"[LaViRA] NAV generation={generation}")
        return True

    def tick(self, now: float) -> None:
        while True:
            try:
                item = self.results.get_nowait()
            except queue.Empty:
                break
            if item.generation != self.generation or item.generation != self.pending_generation:
                continue
            self.pending_generation = None
            if item.error or item.result is None or item.result.outcome != "NAVIGATE":
                reason = item.error or f"lavira_{getattr(item.result, 'outcome', 'failed')}"
                terminal_state = (
                    "stopped"
                    if item.error is None
                    and item.result is not None
                    and item.result.outcome == "STOP"
                    else "failed"
                )
                self.state = "listen_wasd"
                self.submit_intent(
                    "navigation_agent_status",
                    {
                        "generation": item.generation,
                        "state": terminal_state,
                        "reason": reason,
                    },
                )
                self.logger(f"[LaViRA] LISTEN_WASD ({reason})")
                continue
            try:
                goal = result_to_goal(item.result)
                policy = item.result.policy
                self.submit_intent(
                    "navigation_goal",
                    {
                        "generation": item.generation,
                        "goal_base": goal,
                        "target": str(policy.get("target", self.config.global_target)),
                        "target_type": str(policy.get("target_type", "global_target")),
                        "confidence": float(policy.get("confidence", 0.0)),
                    },
                )
            except Exception as exc:
                self.state = "listen_wasd"
                self.submit_intent(
                    "navigation_agent_status",
                    {
                        "generation": item.generation,
                        "state": "failed",
                        "reason": f"invalid_goal: {exc}",
                    },
                )

    def accept_status(self, message: str | bytes | Mapping[str, Any]) -> bool:
        payload = json.loads(message) if isinstance(message, (str, bytes)) else dict(message)
        if int(payload.get("generation", -1)) != self.generation:
            return False
        if payload.get("state") in {"reached", "failed", "stopped"}:
            self.pending_generation = None
            self.state = "listen_wasd"
            self.logger(
                f"[LaViRA] LISTEN_WASD (navdp_{payload.get('state')}: "
                f"{payload.get('reason', '')})"
            )
        return True

    def shutdown(self) -> None:
        self.stop_event.set()
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass


def run_inference_worker(
    factory: Callable[[], ObjectNavRunner], runtime: LaviraPlannerRuntime
) -> None:
    runner: ObjectNavRunner | None = None
    try:
        if runtime.config.warmup and not runtime.stop_event.is_set():
            try:
                runner = factory()
                runtime.logger("[LaViRA] warmup started")
                runner.warmup()
                runtime.logger("[LaViRA] warmup complete")
            except Exception as exc:
                # Warmup is an optimization, not the lifetime of the planner.
                # Keep the worker alive so a later NAV request either retries
                # successfully or returns a visible error to the state machine.
                runtime.logger(f"[LaViRA] warmup failed: {exc}")
                if runner is not None:
                    runner.close()
                runner = None
        while not runtime.stop_event.is_set():
            generation = runtime.requests.get()
            if generation is None:
                break
            try:
                runner = runner or factory()
                item = WorkerResult(generation, runner.run_once(), None)
            except Exception as exc:
                item = WorkerResult(generation, None, str(exc))
            runtime.publish_worker_result(item)
    finally:
        if runner is not None:
            runner.close()


def _runner(config: LaviraPlannerConfig) -> ObjectNavRunner:
    camera = SensorGatewayRGBDCamera(
        config.sensor_gateway_endpoint,
        timeout_ms=config.camera_timeout_ms,
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        max_skew_ms=config.sensor_gateway_max_skew_ms,
    )
    return ObjectNavRunner(ObjectNavConfig(
        mission=config.mission,
        global_target=config.global_target,
        vision_backend=config.vision_backend,
        model=config.model,
        qwenvl_model=config.qwenvl_model,
        qwenvl_base_url=config.qwenvl_base_url,
        camera_timeout_ms=config.camera_timeout_ms,
        codex_timeout_seconds=config.codex_timeout_seconds,
        min_confidence=config.min_confidence,
        output_root=config.output_root,
    ), camera=camera, own_camera=True)


def main(config: LaviraPlannerConfig) -> None:
    context = zmq.Context.instance()
    intent = ControlGatewayIntentClient(
        config.control_gateway_intent_endpoint,
        source="lavira_agent",
        context=context,
    )
    runtime = LaviraPlannerRuntime(
        config,
        submit_intent=lambda name, parameters: intent.send(name, parameters),
    )
    control_gateway = ControlGatewaySubscriber(
        config.control_gateway_endpoint,
        context=context,
        accepted_names={
            "start_navigation",
            "cancel_navigation",
            "navigation_status",
        },
    )
    worker = threading.Thread(target=run_inference_worker, args=(lambda: _runner(config), runtime), daemon=True)
    worker.start()
    print(
        "[LaViRA] waiting for typed start/cancel/status events from ControlGateway"
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
                elif command.name == "navigation_status":
                    runtime.accept_status(command.parameters)
            runtime.tick(time.monotonic())
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.shutdown()
        control_gateway.close()
        intent.close()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(LaviraPlannerConfig))
