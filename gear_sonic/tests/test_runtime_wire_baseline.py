from __future__ import annotations

import json
import multiprocessing
import queue
import struct
import time
from pathlib import Path
from typing import Any

import pytest
import zmq

from gear_sonic.scripts.navdp_planner import (
    build_navigation_message,
    decode_navigation_message,
)
from gear_sonic.planner_control import (
    build_planner_velocity_message,
    decode_planner_velocity_message,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message


def _receive_one(
    endpoint: str,
    topic: bytes,
    ready: Any,
    output: Any,
) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, topic)
    socket.connect(endpoint)
    ready.set()
    try:
        if socket.poll(3000, zmq.POLLIN):
            output.put(socket.recv())
        else:
            output.put(None)
    finally:
        socket.close(linger=0)
        context.term()


def _cross_process_receive(payload: bytes, topic: bytes, socket_path: Path) -> bytes:
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    socket_path.unlink(missing_ok=True)
    endpoint = f"ipc://{socket_path}"
    publisher.bind(endpoint)
    process_context = multiprocessing.get_context("spawn")
    ready = process_context.Event()
    output = process_context.Queue()
    process = process_context.Process(
        target=_receive_one,
        args=(endpoint, topic, ready, output),
    )
    process.start()
    try:
        assert ready.wait(timeout=2.0)
        deadline = time.monotonic() + 3.0
        received = None
        while received is None and time.monotonic() < deadline:
            publisher.send(payload)
            try:
                received = output.get(timeout=0.05)
            except queue.Empty:
                continue
        assert received is not None
        return received
    finally:
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        publisher.close(linger=0)
        context.term()
        socket_path.unlink(missing_ok=True)


def test_navigation_json_wire_format_is_frozen() -> None:
    message = build_navigation_message(
        mode="manual_velocity",
        generation=4,
        timestamp=12.5,
        velocity=(0.3, -0.1, 0.5),
    )

    assert json.loads(message) == {
        "type": "sonic_navigation_command",
        "version": 1,
        "generation": 4,
        "mode": "manual_velocity",
        "timestamp": 12.5,
        "velocity": {"vx": 0.3, "vy": -0.1, "wz": 0.5},
    }


def test_planner_binary_wire_format_is_frozen() -> None:
    message = build_planner_message(
        1,
        movement=(1.0, 0.0, 0.0),
        facing=(0.0, 1.0, 0.0),
        speed=0.3,
        height=-1.0,
    )
    header = json.loads(message[7 : 7 + 1280].split(b"\x00", 1)[0])
    payload = message[7 + 1280 :]

    assert message.startswith(b"planner")
    assert header == {
        "v": 1,
        "endian": "le",
        "count": 1,
        "fields": [
            {"name": "mode", "dtype": "i32", "shape": [1]},
            {"name": "movement", "dtype": "f32", "shape": [3]},
            {"name": "facing", "dtype": "f32", "shape": [3]},
            {"name": "speed", "dtype": "f32", "shape": [1]},
            {"name": "height", "dtype": "f32", "shape": [1]},
        ],
    }
    assert payload == struct.pack("<i8f", 1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3, -1.0)


def test_navdp_velocity_wire_format_is_generation_scoped() -> None:
    message = build_planner_velocity_message(
        generation=7,
        source="navdp",
        velocity=(0.3, 0.0, 0.4),
        timestamp=12.5,
        heading_target_rad=0.8,
        heading_reference_rad=0.2,
    )

    assert json.loads(message) == {
        "type": "sonic_planner_velocity",
        "version": 1,
        "generation": 7,
        "timestamp": 12.5,
        "source": "navdp",
        "velocity": {"vx": 0.3, "vy": 0.0, "wz": 0.4},
        "heading": {"target_rad": 0.8, "reference_rad": 0.2},
    }
    assert decode_planner_velocity_message(message).generation == 7


@pytest.mark.parametrize(
    ("payload", "topic"),
    [
        (
            build_navigation_message(
                mode="nav_goal",
                generation=8,
                timestamp=123.0,
                goal_base=(1.2, -0.4),
                target="basket",
                confidence=0.8,
            ).encode(),
            b"",
        ),
        (
            build_planner_message(
                1,
                movement=(1.0, 0.0, 0.0),
                facing=(1.0, 0.0, 0.0),
                speed=0.3,
            ),
            b"planner",
        ),
        (
            build_planner_velocity_message(
                generation=8,
                source="navdp",
                velocity=(0.2, 0.0, -0.1),
                timestamp=123.0,
            ).encode(),
            b"",
        ),
    ],
)
def test_current_packets_survive_cross_process_zmq_unchanged(
    payload: bytes, topic: bytes, tmp_path: Path
) -> None:
    assert _cross_process_receive(payload, topic, tmp_path / "wire.sock") == payload


def test_existing_navigation_timestamp_supports_passive_latency_measurement(tmp_path: Path) -> None:
    sent_at = time.time()
    payload = build_navigation_message(
        mode="stop",
        generation=9,
        timestamp=sent_at,
    ).encode()

    received = _cross_process_receive(payload, b"", tmp_path / "latency.sock")
    command = decode_navigation_message(received)
    observed_latency = time.time() - command.timestamp

    assert 0.0 <= observed_latency < 3.0
