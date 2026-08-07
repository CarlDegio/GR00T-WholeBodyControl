#!/usr/bin/env python3
"""LaViRA semantic target selector and LISTEN_WASD/NAV keyboard state machine."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import queue
import select
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable, Iterator, Literal, Mapping, TextIO

import zmq

from gear_sonic.scripts.navdp_planner import build_navigation_message
from gear_sonic.utils.inference.object_nav import ObjectNavConfig, ObjectNavResult, ObjectNavRunner


MANUAL = {
    "w": ("forward", (0.3, 0.0, 0.0)),
    "s": ("backward", (-0.3, 0.0, 0.0)),
    "a": ("move_left", (0.0, 0.15, 0.0)),
    "d": ("move_right", (0.0, -0.15, 0.0)),
    "q": ("turn_left", (0.0, 0.0, 0.5)),
    "e": ("turn_right", (0.0, 0.0, -0.5)),
}


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
    host: str = "*"
    port: int = 5558
    status_host: str = "127.0.0.1"
    status_port: int = 5559
    planner_hz: float = 20.0
    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_timeout_ms: int = 15000
    codex_timeout_seconds: float = 180.0
    min_confidence: float = 0.6
    output_root: str = "outputs/object_nav"


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
        publish: Callable[[str], None],
        logger: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.publish = publish
        self.logger = logger
        self.generation = 0
        self.state = "listen_wasd"
        self.pending_generation: int | None = None
        self.requests: queue.Queue[int | None] = queue.Queue(maxsize=1)
        self.results: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.manual_velocity = (0.0, 0.0, 0.0)
        self.manual_deadline = 0.0

    @property
    def phase(self) -> str:
        return self.state

    def _send(self, mode: str, **kwargs: Any) -> None:
        self.publish(build_navigation_message(mode=mode, generation=self.generation, **kwargs))

    def stop(self, reason: str) -> None:
        self.generation += 1
        self.pending_generation = None
        self.state = "listen_wasd"
        self.manual_velocity = (0.0, 0.0, 0.0)
        self._send("stop")
        self.logger(f"[LaViRA] LISTEN_WASD ({reason})")

    def handle_key(self, key: str, *, now: float) -> str:
        normalized = key.lower()
        if normalized == "x":
            self.stop("exit")
            return "exit"
        if key == " ":
            self.stop("operator_stop")
            return "cancelled"
        if normalized == "n":
            if self.state != "listen_wasd" or self.pending_generation is not None:
                return "busy"
            self.generation += 1
            self.pending_generation = self.generation
            self.state = "nav"
            self._send("stop")
            self.requests.put_nowait(self.generation)
            self.logger(f"[LaViRA] NAV generation={self.generation}")
            return "started"
        if normalized in MANUAL:
            if self.state != "listen_wasd":
                return "ignored"
            _, self.manual_velocity = MANUAL[normalized]
            self.manual_deadline = now + 0.55
            self._send("manual_velocity", velocity=self.manual_velocity)
            return "manual"
        return "ignored"

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
                self.stop(item.error or f"lavira_{getattr(item.result, 'outcome', 'failed')}")
                continue
            try:
                goal = result_to_goal(item.result)
                policy = item.result.policy
                self._send(
                    "nav_goal",
                    goal_base=goal,
                    target=str(policy.get("target", self.config.global_target)),
                    target_type=str(policy.get("target_type", "global_target")),
                    confidence=float(policy.get("confidence", 0.0)),
                )
            except Exception as exc:
                self.stop(f"invalid_goal: {exc}")
        if self.state == "listen_wasd" and now >= self.manual_deadline and any(self.manual_velocity):
            self.manual_velocity = (0.0, 0.0, 0.0)
            self._send("manual_velocity", velocity=self.manual_velocity)

    def accept_status(self, message: str | bytes | Mapping[str, Any]) -> bool:
        payload = json.loads(message) if isinstance(message, (str, bytes)) else dict(message)
        if payload.get("type") != "sonic_navigation_status" or int(payload.get("generation", -1)) != self.generation:
            return False
        if payload.get("state") in {"reached", "failed", "stopped"}:
            self.stop(f"navdp_{payload.get('state')}: {payload.get('reason', '')}")
        return True

    def shutdown(self) -> None:
        self.stop_event.set()
        self.stop("shutdown")
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
            runner = factory()
            runtime.logger("[LaViRA] warmup started")
            runner.warmup()
            runtime.logger("[LaViRA] warmup complete")
        while not runtime.stop_event.is_set():
            generation = runtime.requests.get()
            if generation is None:
                break
            try:
                runner = runner or factory()
                item = WorkerResult(generation, runner.run_once(), None)
            except Exception as exc:
                item = WorkerResult(generation, None, str(exc))
            try:
                runtime.results.put_nowait(item)
            except queue.Full:
                pass
    finally:
        if runner is not None:
            runner.close()


def _runner(config: LaviraPlannerConfig) -> ObjectNavRunner:
    return ObjectNavRunner(ObjectNavConfig(
        mission=config.mission,
        global_target=config.global_target,
        vision_backend=config.vision_backend,
        model=config.model,
        qwenvl_model=config.qwenvl_model,
        qwenvl_base_url=config.qwenvl_base_url,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        camera_timeout_ms=config.camera_timeout_ms,
        codex_timeout_seconds=config.codex_timeout_seconds,
        min_confidence=config.min_confidence,
        output_root=config.output_root,
    ))


def read_key_nonblocking(stream: TextIO = sys.stdin) -> str | None:
    readable, _, _ = select.select([stream], [], [], 0.0)
    return stream.read(1) if readable else None


@contextmanager
def cbreak_terminal(stream: TextIO = sys.stdin) -> Iterator[None]:
    if not stream.isatty():
        raise RuntimeError("LaViRA keyboard input requires an interactive TTY/tmux pane")
    descriptor = stream.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def main(config: LaviraPlannerConfig) -> None:
    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.bind(f"tcp://{config.host}:{config.port}")
    status = context.socket(zmq.SUB)
    status.connect(f"tcp://{config.status_host}:{config.status_port}")
    status.setsockopt_string(zmq.SUBSCRIBE, "")
    runtime = LaviraPlannerRuntime(config, publish=publisher.send_string)
    worker = threading.Thread(target=run_inference_worker, args=(lambda: _runner(config), runtime), daemon=True)
    worker.start()
    print("[LaViRA] LISTEN_WASD: W/S/A/D/Q/E | N: NAV | Space: stop | X: exit")
    try:
        with cbreak_terminal():
            while True:
                key = read_key_nonblocking()
                if key is not None and runtime.handle_key(key, now=time.monotonic()) == "exit":
                    break
                while status.poll(0):
                    runtime.accept_status(status.recv())
                runtime.tick(time.monotonic())
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.shutdown()
        publisher.close(0)
        status.close(0)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(LaviraPlannerConfig))
