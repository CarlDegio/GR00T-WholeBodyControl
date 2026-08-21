#!/usr/bin/env python3
"""LaViRA semantic target selector behind the unified ControlGateway."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import queue
import threading
import time
from typing import Any, Callable, Mapping

import zmq

from gear_sonic.runtime.profile import (
    RuntimeProfileSelection,
    load_component_config,
    load_runtime_profile,
)
from gear_sonic.runtime.telemetry import (
    LAVIRA_TIMING_SEGMENTS,
    configure_file_logging,
    open_telemetry_publisher,
    publish_metrics,
)
from gear_sonic.runtime.gateway.control_client import (
    ControlGatewayIntentClient,
    ControlGatewaySubscriber,
)
from gear_sonic.utils.inference.lavira.object_nav import (
    ObjectNavConfig,
    ObjectNavResult,
    ObjectNavRunner,
    SensorGatewayRGBDCamera,
)


LOGGER = logging.getLogger("sonic.lavira")


@dataclass
class LaviraPlannerConfig:
    mission: str
    global_target: str
    profile: str = ""
    overlay: tuple[str, ...] = ()
    qwenvl_model: str = "qwen3-vl-32b-instruct"
    qwenvl_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    camera_timeout_ms: int = 15000
    sensor_gateway_request_timeout_ms: int = 100
    sensor_gateway_max_age_ms: float = 1000.0
    sensor_gateway_max_skew_ms: float = 5.0
    qwenvl_timeout_seconds: float = 180.0
    min_confidence: float = 0.6


def load_lavira_config(
    profile: str = "",
    overlays: tuple[str, ...] = (),
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
    ) -> None:
        self.config = config
        self.submit_intent = submit_intent
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
        LOGGER.info("LISTEN_WASD reason=%s", reason)

    def start_navigation(self, generation: int) -> bool:
        if generation <= self.generation or self.pending_generation is not None:
            return False
        self.generation = generation
        self.pending_generation = generation
        self.state = "nav"
        self._put_latest(self.requests, generation)
        LOGGER.info("NAV generation=%s", generation)
        return True

    def tick(self) -> dict[str, float] | None:
        accepted_timing_ms: dict[str, float] | None = None
        while True:
            try:
                item = self.results.get_nowait()
            except queue.Empty:
                break
            if item.generation != self.generation or item.generation != self.pending_generation:
                continue
            self.pending_generation = None
            if item.result is not None:
                timing = item.result.geometry.get("timing_s")
                if isinstance(timing, Mapping):
                    accepted_timing_ms = {
                        name: float(value) * 1000.0 for name, value in timing.items()
                    }
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
                LOGGER.info("LISTEN_WASD reason=%s", reason)
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
        return accepted_timing_ms

    def accept_status(self, message: str | bytes | Mapping[str, Any]) -> bool:
        payload = json.loads(message) if isinstance(message, (str, bytes)) else dict(message)
        if int(payload.get("generation", -1)) != self.generation:
            return False
        if payload.get("state") in {"reached", "failed", "stopped"}:
            self.pending_generation = None
            self.state = "listen_wasd"
            LOGGER.info(
                "LISTEN_WASD reason=navdp_%s detail=%s",
                payload.get("state"),
                payload.get("reason", ""),
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
        while not runtime.stop_event.is_set():
            generation = runtime.requests.get()
            if generation is None:
                break
            released = False

            def release_depth() -> None:
                nonlocal released
                if released:
                    return
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
                # Custom/test runners may fail before invoking the camera callback.
                # Releasing twice is prevented above, so every request is fail-closed.
                release_depth()
            runtime.publish_worker_result(item)
    finally:
        if runner is not None:
            runner.close()


def _runner(config: LaviraPlannerConfig, sensor_gateway: str) -> ObjectNavRunner:
    camera = SensorGatewayRGBDCamera(
        sensor_gateway,
        timeout_ms=config.camera_timeout_ms,
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        max_skew_ms=config.sensor_gateway_max_skew_ms,
    )
    return ObjectNavRunner(
        ObjectNavConfig(
            mission=config.mission,
            global_target=config.global_target,
            qwenvl_model=config.qwenvl_model,
            qwenvl_base_url=config.qwenvl_base_url,
            qwenvl_timeout_seconds=config.qwenvl_timeout_seconds,
            min_confidence=config.min_confidence,
        ),
        camera=camera,
        own_camera=True,
    )


def main(config: LaviraPlannerConfig) -> None:
    configure_file_logging("lavira")
    profile = load_runtime_profile(config.profile or None, overlays=config.overlay)
    metrics_socket = open_telemetry_publisher(
        profile.endpoint_uri("runtime_metrics_ingress")
    )

    def send_metrics(values: Mapping[str, float], *, activate: bool = False) -> None:
        publish_metrics(
            metrics_socket,
            "lavira",
            values,
            allowed_names=LAVIRA_TIMING_SEGMENTS,
            activate=activate,
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
        accepted_names={
            "start_navigation",
            "cancel_navigation",
            "navigation_status",
        },
    )
    worker = threading.Thread(
        target=run_inference_worker,
        args=(
            lambda: _runner(
                config, profile.endpoint_uri("sensor_gateway_metadata")
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
                    if runtime.start_navigation(generation):
                        send_metrics({}, activate=True)
                elif command.name == "cancel_navigation":
                    runtime.cancel(generation, "operator_stop")
                elif command.name == "navigation_status":
                    runtime.accept_status(command.parameters)
            timing_ms = runtime.tick()
            if timing_ms:
                send_metrics(timing_ms)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.shutdown()
        control_gateway.close()
        intent.close()
        metrics_socket.close(linger=0)
        LOGGER.info("STOPPED")


if __name__ == "__main__":
    main(parse_lavira_config())
