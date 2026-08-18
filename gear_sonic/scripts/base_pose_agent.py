#!/usr/bin/env python3
"""Execute Base-Pose plans through the unified ControlGateway."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Callable, Literal, Mapping

import zmq

from gear_sonic.base_pose import (
    BasePoseSequenceController,
)
from gear_sonic.runtime.control_client import (
    ControlGatewayIntentClient,
    ControlGatewaySubscriber,
)
from gear_sonic.utils.inference.base_pose import (
    BasePoseConfig,
    BasePosePlanner,
    BasePoseResult,
)
from gear_sonic.utils.inference.base_pose_sensor import SensorGatewayBasePoseCamera


@dataclass
class BasePoseAgentConfig:
    task: str
    mode: Literal["rgb", "rgbd", "rgb_depth_query"] = "rgb"
    vision_backend: Literal["codex", "qwenvl"] = "codex"
    model: str = "gpt-5.6-sol"
    qwenvl_model: str = "qwen3-vl-plus"
    qwenvl_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    qwenvl_thinking_budget: int = 500
    reasoning_effort: str = "max"
    codex_fast: bool = True
    codex_timeout_seconds: float = 600.0
    camera_stream: str = "ego_view"
    depth_stream: str = "derived/lingbot_depth"
    camera_timeout_ms: int = 15000
    sensor_gateway_endpoint: str = "tcp://127.0.0.1:5560"
    sensor_gateway_request_timeout_ms: int = 100
    sensor_gateway_max_age_ms: float = 1000.0
    sensor_gateway_max_skew_ms: float = 5.0
    control_gateway_endpoint: str = "tcp://127.0.0.1:5565"
    control_gateway_intent_endpoint: str = "tcp://127.0.0.1:5561"
    planner_hz: float = 20.0
    transition_pause: float = 0.5
    rotation_speed: float = 0.4
    translation_speed: float = 0.3
    rotation_scale: float = 1.0
    translation_scale: float = 1.0
    persist_diagnostics: bool = False
    output_root: str = "outputs/base_pose_adjustment"


@dataclass(frozen=True)
class WorkerResult:
    generation: int
    result: BasePoseResult | None
    error: str | None


class BasePoseAgentRuntime:
    """Own inference and sequence timing, but never own the robot output socket."""

    def __init__(
        self,
        config: BasePoseAgentConfig,
        *,
        submit_intent: Callable[[str, Mapping[str, object]], None],
        logger: Callable[[str], None] = print,
    ) -> None:
        if not config.task.strip():
            raise ValueError("base-pose task must be non-empty")
        if config.planner_hz <= 0.0:
            raise ValueError("planner_hz must be positive")
        self.config = config
        self.submit_intent = submit_intent
        self.logger = logger
        self.controller = BasePoseSequenceController(
            rotation_speed=config.rotation_speed,
            translation_speed=config.translation_speed,
            rotation_scale=config.rotation_scale,
            translation_scale=config.translation_scale,
            transition_pause=config.transition_pause,
            stop_duration=1.0 / config.planner_hz,
        )
        self.generation = 0
        self.state = "idle"
        self.pending_generation: int | None = None
        self.next_publish_at = 0.0
        self.requests: queue.Queue[int | None] = queue.Queue(maxsize=1)
        self.results: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

    @staticmethod
    def _discard(queue_: queue.Queue) -> None:
        try:
            queue_.get_nowait()
        except queue.Empty:
            pass

    @classmethod
    def _put_latest(cls, queue_: queue.Queue, item: Any) -> None:
        cls._discard(queue_)
        queue_.put_nowait(item)

    def publish_worker_result(self, item: WorkerResult) -> None:
        self._put_latest(self.results, item)

    def start(self, generation: int) -> bool:
        if generation <= self.generation or self.state != "idle":
            return False
        self.generation = generation
        self.pending_generation = generation
        self.state = "inference"
        self.controller.cancel()
        self._put_latest(self.requests, generation)
        self.logger(f"[BasePose] INFERENCE generation={generation}")
        return True

    def cancel(self, generation: int, reason: str) -> None:
        if generation < self.generation:
            return
        self.generation = generation
        self.pending_generation = None
        self.state = "idle"
        self.controller.cancel()
        self._discard(self.requests)
        self.logger(f"[BasePose] IDLE ({reason})")

    def _terminal(self, state: str, reason: str) -> None:
        self.pending_generation = None
        self.state = "idle"
        self.controller.cancel()
        self.submit_intent(
            "base_pose_status",
            {
                "generation": self.generation,
                "state": state,
                "reason": reason,
            },
        )
        self.logger(f"[BasePose] {state.upper()} ({reason})")

    def _accept_result(self, item: WorkerResult, *, now: float) -> bool:
        if (
            item.generation != self.generation
            or item.generation != self.pending_generation
            or self.state != "inference"
        ):
            return False
        self.pending_generation = None
        if item.error is not None or item.result is None:
            self._terminal("failed", item.error or "missing_result")
            return True
        status = str(item.result.plan["status"])
        if status == "ADJUST":
            if not self.controller.start(item.result.plan, now):
                self._terminal("failed", "empty_adjustment")
            else:
                self.state = "motion"
                self.next_publish_at = float(now)
                self.logger(
                    f"[BasePose] MOTION generation={item.generation} "
                    f"steps={len(item.result.plan['command_sequence'])}"
                )
        elif status == "READY":
            self._terminal("reached", "base_pose_READY")
        else:
            self._terminal("failed", f"base_pose_{status}")
        return True

    def tick(self, now: float) -> None:
        while True:
            try:
                item = self.results.get_nowait()
            except queue.Empty:
                break
            self._accept_result(item, now=now)
        if self.state != "motion" or now + 1.0e-12 < self.next_publish_at:
            return
        was_active = self.controller.active
        action, command = self.controller.step(now)
        self.submit_intent(
            "base_pose_velocity",
            {
                "generation": self.generation,
                "velocity": list(command.velocity),
                "action": action,
            },
        )
        self.next_publish_at = float(now) + 1.0 / self.config.planner_hz
        if was_active and not self.controller.active:
            self._terminal("reached", "base_pose_sequence_completed")

    def shutdown(self) -> None:
        self.stop_event.set()
        self.controller.cancel()
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass


def run_inference_worker(
    planner: BasePosePlanner,
    camera: SensorGatewayBasePoseCamera,
    runtime: BasePoseAgentRuntime,
) -> None:
    try:
        while not runtime.stop_event.is_set():
            generation = runtime.requests.get()
            if generation is None or runtime.stop_event.is_set():
                return
            try:
                result = planner.plan(camera.capture())
                item = WorkerResult(generation, result, None)
            except Exception as exc:
                item = WorkerResult(generation, None, f"{type(exc).__name__}: {exc}")
            if not runtime.stop_event.is_set():
                runtime.publish_worker_result(item)
    finally:
        camera.close()
        planner.close()


def main(config: BasePoseAgentConfig) -> None:
    context = zmq.Context.instance()
    intent = ControlGatewayIntentClient(
        config.control_gateway_intent_endpoint,
        source="base_pose_agent",
        context=context,
        ttl_ms=max(100, int(3000.0 / config.planner_hz)),
    )
    runtime = BasePoseAgentRuntime(
        config,
        submit_intent=lambda name, parameters: intent.send(name, parameters),
    )
    control = ControlGatewaySubscriber(
        config.control_gateway_endpoint,
        context=context,
        accepted_names={"start_base_pose", "cancel_navigation"},
    )
    camera = SensorGatewayBasePoseCamera(
        config.sensor_gateway_endpoint,
        camera_stream=config.camera_stream,
        depth_stream=config.depth_stream,
        require_depth=config.mode != "rgb",
        timeout_ms=config.camera_timeout_ms,
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        max_skew_ms=config.sensor_gateway_max_skew_ms,
    )
    planner = BasePosePlanner(
        BasePoseConfig(
            task=config.task,
            mode=config.mode,
            vision_backend=config.vision_backend,
            model=config.model,
            qwenvl_model=config.qwenvl_model,
            qwenvl_base_url=config.qwenvl_base_url,
            qwenvl_thinking_budget=config.qwenvl_thinking_budget,
            reasoning_effort=config.reasoning_effort,
            codex_fast=config.codex_fast,
            camera_stream=config.camera_stream,
            codex_timeout_seconds=config.codex_timeout_seconds,
            persist_diagnostics=config.persist_diagnostics,
            output_root=config.output_root,
        )
    )
    worker = threading.Thread(
        target=run_inference_worker,
        args=(planner, camera, runtime),
        name="base-pose-inference",
        daemon=True,
    )
    worker.start()
    print(
        f"[BasePose] waiting for B via ControlGateway; mode={config.mode} "
        f"backend={config.vision_backend} task={config.task!r}"
    )
    try:
        while True:
            command = control.read_command()
            if command is not None:
                generation = int(command.parameters.get("generation", -1))
                if command.name == "start_base_pose":
                    runtime.start(generation)
                else:
                    runtime.cancel(generation, "operator_stop")
            runtime.tick(time.monotonic())
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.shutdown()
        worker.join(timeout=1.0)
        control.close()
        intent.close()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(BasePoseAgentConfig))
