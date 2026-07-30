#!/usr/bin/env python3
"""Bridge SONIC camera frames and remote NaVILA actions to REASEN."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import re
import select
import signal
import sys
import termios
import threading
import time
from typing import Any
import tty

import cv2
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.scripts.keyboard_planner_thread_server import (
    KeyboardPlannerConfig,
    build_navila_message,
    key_commands,
)


@dataclass
class NavilaPlannerConfig:
    remote_host: str = "124.220.175.102"
    remote_port: int = 29999
    remote_timeout_s: float = 30.0
    api_token: str = ""
    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_name: str = "chest_view"
    instruction: str = "Navigate safely toward the goal."
    test_image: Path | None = None
    image_hz: float = 2.0
    jpeg_quality: int = 80
    show_cv: bool = False
    output_host: str = "*"
    output_port: int = 5558
    keyboard_duration: float = 0.5
    forward_speed: float = 0.5
    backward_speed: float = 0.3
    lateral_speed: float = 0.15
    keyboard_yaw_speed: float = 0.5
    yaw_speed: float = math.pi / 6.0
    stop_duration: float = 0.5


@dataclass
class NavilaPlannerControl:
    """Interactive run state and terminal-log de-duplication."""

    active: bool = False
    _last_reported_action: tuple[str, tuple[float, float, float], float] | None = None
    action_ready_at: float = 0.0

    def toggle(self) -> bool:
        self.active = not self.active
        if self.active:
            self._last_reported_action = None
            self.action_ready_at = 0.0
        return self.active

    def accept_action(
        self,
        action: str,
        velocity: tuple[float, float, float],
        duration: float,
        now: float | None = None,
    ) -> bool:
        signature = (action, velocity, duration)
        changed = signature != self._last_reported_action
        self._last_reported_action = signature
        if action == "stop":
            self.active = False
            self.action_ready_at = 0.0
        else:
            self.action_ready_at = (time.monotonic() if now is None else now) + duration
        return changed

    def can_infer(self, now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        return self.active and current_time >= self.action_ready_at


def build_keyboard_message(key: str, config: NavilaPlannerConfig) -> str | None:
    keyboard_config = KeyboardPlannerConfig(
        duration=config.keyboard_duration,
        forward_speed=config.forward_speed,
        backward_speed=config.backward_speed,
        lateral_speed=config.lateral_speed,
        yaw_speed=config.keyboard_yaw_speed,
    )
    command = key_commands(keyboard_config).get(key.lower())
    if command is None:
        return None
    action, velocity = command
    return build_navila_message(action, velocity, keyboard_config.duration)


def build_policy_data(
    jpeg: bytes,
    sequence: int,
    timestamp_ns: int,
    camera: str,
    instruction: str,
) -> dict[str, Any]:
    return {
        "type": "navila_camera_frame",
        "version": 1,
        "sequence": sequence,
        "timestamp_ns": timestamp_ns,
        "camera": camera,
        "instruction": instruction,
        "encoding": "jpeg",
        "image_jpeg": jpeg,
    }


def parse_remote_text(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    stripped = raw.strip()
    if not stripped:
        raise ValueError("empty NaVILA command")
    if not stripped.startswith("{"):
        return stripped
    payload = json.loads(stripped)
    if payload.get("type") == "navila_reasan_velocity_command":
        return stripped
    for key in ("text", "action", "raw_text", "response", "output"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("NaVILA JSON has no text/action field")


def build_remote_message(action: str, velocity: tuple[float, float, float], duration: float, text: str) -> str:
    payload = json.loads(build_navila_message(action, velocity, duration, text))
    payload["source"] = "navila"
    return json.dumps(payload, separators=(",", ":"))


def navila_text_to_message(text: str, config: NavilaPlannerConfig) -> str:
    if text.lstrip().startswith("{"):
        payload = json.loads(text)
        if payload.get("type") == "navila_reasan_velocity_command":
            return json.dumps(payload, separators=(",", ":"))

    lowered = text.lower()
    if re.search(r"\bstop\b", lowered):
        return build_remote_message("stop", (0.0, 0.0, 0.0), config.stop_duration, text)
    turn_match = re.search(r"\bturn\s+(left|right)\b", lowered)
    if turn_match:
        degrees = next((value for value in (15, 30, 45) if str(value) in lowered), 15)
        direction = 1.0 if turn_match.group(1) == "left" else -1.0
        duration = math.radians(degrees) / config.yaw_speed
        return build_remote_message(
            f"turn_{turn_match.group(1)}", (0.0, 0.0, direction * config.yaw_speed), duration, text
        )
    if re.search(r"\b(move\s+forward|move)\b", lowered):
        centimeters = next((value for value in (25, 50, 75) if str(value) in lowered), 25)
        duration = (centimeters / 100.0) / config.forward_speed
        return build_remote_message("move_forward", (config.forward_speed, 0.0, 0.0), duration, text)
    raise ValueError(f"unsupported NaVILA action: {text!r}")


def _put_latest(target: queue.Queue, item: Any) -> None:
    try:
        target.put_nowait(item)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        target.put_nowait(item)


def _inference_worker_loop(
    policy: Any,
    request_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            generation, data = request_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        busy_event.set()
        try:
            response = policy.call_endpoint("get_action", data)
            _put_latest(result_queue, (generation, response, None))
        except Exception as exc:
            _put_latest(result_queue, (generation, None, str(exc)))
        finally:
            busy_event.clear()


def main(config: NavilaPlannerConfig) -> None:
    if (
        config.image_hz <= 0.0
        or config.remote_timeout_s <= 0.0
        or config.forward_speed <= 0.0
        or config.yaw_speed <= 0.0
    ):
        raise ValueError("image_hz, remote_timeout_s, forward_speed and yaw_speed must be positive")
    if not 1 <= config.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be inside [1,100]")

    from gr00t.policy.server_client import PolicyClient

    policy = PolicyClient(
        host=config.remote_host,
        port=config.remote_port,
        timeout_ms=int(config.remote_timeout_s * 1000),
        api_token=config.api_token or None,
    )
    print(f"[NaVILA] connecting to PolicyServer at {config.remote_host}:{config.remote_port}")
    print(f"[NaVILA] PolicyServer {'reachable' if policy.ping() else 'not reachable yet'}")

    context = zmq.Context.instance()
    output_socket = context.socket(zmq.PUB)
    output_socket.setsockopt(zmq.LINGER, 0)
    output_socket.bind(f"tcp://{config.output_host}:{config.output_port}")
    camera = None if config.test_image is not None else ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )
    test_rgb = None
    if config.test_image is not None:
        test_bgr = cv2.imread(str(config.test_image), cv2.IMREAD_COLOR)
        if test_bgr is None:
            raise FileNotFoundError(f"test image not found or unreadable: {config.test_image}")
        test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)

    running = True
    control = NavilaPlannerControl()

    def stop(_signum=None, _frame=None) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    image_period = 1.0 / config.image_hz
    next_image = time.monotonic()
    image_sequence = 0
    command_count = 0
    last_action = "waiting"
    generation = 0
    request_queue: queue.Queue = queue.Queue(maxsize=1)
    result_queue: queue.Queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()
    inference_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(policy, request_queue, result_queue, inference_stop_event, inference_busy_event),
        daemon=True,
    )
    inference_thread.start()
    window_name = "NaVILA chest_view"
    terminal_settings = termios.tcgetattr(sys.stdin.fileno()) if sys.stdin.isatty() else None
    if config.show_cv:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(
        f"[NaVILA] camera {config.camera_host}:{config.camera_port}/{config.camera_name} "
        f"-> policy tcp://{config.remote_host}:{config.remote_port} "
        f"-> REASEN tcp://{config.output_host}:{config.output_port}"
    )
    print(f"[Planner] KEYBOARD | P NaVILA/keyboard | X quit | instruction={config.instruction!r}")
    print("[Planner] W/S forward/back | A/D lateral | Q/E yaw | Space stop")
    time.sleep(0.3)

    stop_message = build_remote_message(
        "stop", (0.0, 0.0, 0.0), config.stop_duration, "NaVILA planner paused"
    )

    def set_from_key(key: int | str) -> None:
        nonlocal generation, last_action, running
        if isinstance(key, int):
            key = chr(key).lower() if key >= 0 else ""
        else:
            key = key.lower()
        if key == "x":
            running = False
        elif key == "p":
            generation += 1
            active = control.toggle()
            if active:
                print(f"[Planner] NAVILA | instruction={config.instruction!r}")
            else:
                output_socket.send_string(stop_message)
                last_action = "stop"
                print("[Planner] KEYBOARD")
        elif not control.active:
            message = build_keyboard_message(key, config)
            if message is not None:
                output_socket.send_string(message)
                payload = json.loads(message)
                velocity = payload["velocity"]
                last_action = str(payload["action"])
                print(
                    f"\r{last_action:>10}: vx={velocity['vx']:+.2f} "
                    f"vy={velocity['vy']:+.2f} wz={velocity['wz']:+.2f} "
                    f"duration={payload['duration_s']:g}s   ",
                    end="",
                    flush=True,
                )

    try:
        if terminal_settings is not None:
            tty.setcbreak(sys.stdin.fileno())
        while running:
            now = time.monotonic()
            if terminal_settings is not None and select.select([sys.stdin], [], [], 0.0)[0]:
                set_from_key(sys.stdin.read(1))
            if config.show_cv:
                set_from_key(cv2.waitKey(1) & 0xFF)
            if not running:
                break
            if now >= next_image:
                camera_message = None if camera is None else camera.read(blocking=False)
                image = test_rgb if test_rgb is not None else (
                    None if camera_message is None else camera_message.get("images", {}).get(config.camera_name)
                )
                if image is not None:
                    if config.show_cv:
                        display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        cv2.putText(
                            display,
                            f"{'NAVILA' if control.active else 'KEYBOARD'} sent={image_sequence} "
                            f"actions={command_count} {last_action} "
                            f"wait={max(0.0, control.action_ready_at - now):.1f}s",
                            (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(window_name, display)
                    if (
                        control.can_infer(now)
                        and not inference_busy_event.is_set()
                        and request_queue.empty()
                        and result_queue.empty()
                    ):
                        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        success, jpeg = cv2.imencode(
                            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
                        )
                        if not success:
                            next_image = now + image_period
                            continue
                        _put_latest(
                            request_queue,
                            (
                                generation,
                                build_policy_data(
                                    jpeg=jpeg.tobytes(),
                                    sequence=image_sequence,
                                    timestamp_ns=time.time_ns(),
                                    camera=config.camera_name,
                                    instruction=config.instruction,
                                ),
                            ),
                        )
                        image_sequence += 1
                next_image = now + image_period

            try:
                result_generation, response, inference_error = result_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                if result_generation != generation or not control.active:
                    continue
                try:
                    if inference_error is not None:
                        raise RuntimeError(inference_error)
                    text = parse_remote_text(json.dumps(response, separators=(",", ":")))
                    message = navila_text_to_message(text, config)
                    output_socket.send_string(message)
                    payload = json.loads(message)
                    command_count += 1
                    last_action = str(payload["action"])
                    velocity = payload["velocity"]
                    velocity_tuple = (
                        float(velocity["vx"]),
                        float(velocity["vy"]),
                        float(velocity["wz"]),
                    )
                    should_report = control.accept_action(
                        last_action,
                        velocity_tuple,
                        float(payload["duration_s"]),
                        now=time.monotonic(),
                    )
                    if should_report:
                        print(
                            f"[NaVILA] {last_action}: vx={velocity_tuple[0]:+.2f} "
                            f"vy={velocity_tuple[1]:+.2f} wz={velocity_tuple[2]:+.2f} "
                            f"duration={payload['duration_s']:.2f}s"
                        )
                    if not control.active:
                        generation += 1
                        print("[Planner] KEYBOARD by remote stop")
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    print(f"[NaVILA] ignored remote command: {exc}")
            time.sleep(0.002)
    finally:
        if terminal_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_settings)
        stop_message = build_remote_message("stop", (0.0, 0.0, 0.0), config.stop_duration, "bridge stopped")
        for _ in range(3):
            output_socket.send_string(stop_message)
            time.sleep(0.02)
        if camera is not None:
            camera.close()
        if config.show_cv:
            cv2.destroyWindow(window_name)
        inference_stop_event.set()
        inference_thread.join(timeout=1.0)
        output_socket.close()


if __name__ == "__main__":
    main(tyro.cli(NavilaPlannerConfig))
