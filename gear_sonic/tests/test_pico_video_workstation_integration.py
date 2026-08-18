from __future__ import annotations

import json
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import threading
import time

import numpy as np
import pytest
import zmq

from gear_sonic.pico_video.bridge import BridgeSettings, PicoVideoBridge
from gear_sonic.pico_video.gateway_source import SensorGatewayVideoSource
from gear_sonic.pico_video.mock_camera import MockCameraPublisher, MockCameraSettings
from gear_sonic.runtime.client import SensorGatewayClient
from gear_sonic.runtime.sensor_gateway import (
    CameraZmqIngress,
    SensorGatewayCore,
    SensorGatewayRpcServer,
)


def _free_tcp_port() -> int:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    return port


def _control_packet(command: bytes, data: bytes) -> bytes:
    body = struct.pack("<I", len(command)) + command
    body += struct.pack("<I", len(data)) + data
    return struct.pack(">I", len(body)) + body


def _open_camera_packet(*, port: int) -> bytes:
    camera = b"ZED"
    encoded_ip = b"127.0.0.1"
    payload = b"\xca\xfe\x01" + struct.pack(
        "<7i", 1280, 480, 30, 4_000_000, 0, 0, port
    )
    payload += bytes((len(camera),)) + camera
    payload += bytes((len(encoded_ip),)) + encoded_ip
    return _control_packet(b"OPEN_CAMERA", payload)


def _read_exact(connection: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        if not chunk:
            raise EOFError("PICO video connection closed")
        result.extend(chunk)
    return bytes(result)


class _VideoReceiver:
    def __init__(self) -> None:
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(5.0)
        self.port = self.listener.getsockname()[1]
        self.connection: socket.socket | None = None

    def receive(self, *, timeout_s: float = 5.0) -> bytes:
        if self.connection is None:
            self.connection, _ = self.listener.accept()
        self.connection.settimeout(timeout_s)
        length = struct.unpack(">I", _read_exact(self.connection, 4))[0]
        return _read_exact(self.connection, length)

    def wait_for_disconnect(self, *, timeout_s: float) -> bool:
        if self.connection is None:
            return False
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.connection.settimeout(max(0.01, deadline - time.monotonic()))
            try:
                if self.connection.recv(64 * 1024) == b"":
                    return True
            except socket.timeout:
                return False
        return False

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self.listener.close()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are not installed",
)
def test_mock_camera_to_sensor_gateway_to_fake_pico_loop(
    tmp_path: Path,
) -> None:
    camera_port = _free_tcp_port()
    mock = MockCameraPublisher(MockCameraSettings(port=camera_port))
    mock_thread = threading.Thread(target=mock.run, name="integration-mock-camera")
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=8, history_size=32, frame_ttl_ms=1000)
    ingress = CameraZmqIngress(
        context,
        f"tcp://127.0.0.1:{camera_port}",
        core,
        expected_hz=30.0,
    )
    ingress_stop = threading.Event()

    def poll_ingress() -> None:
        while not ingress_stop.is_set():
            ingress.poll_once(timeout_ms=20)

    ingress_thread = threading.Thread(target=poll_ingress, name="integration-ingress")
    rpc_endpoint = "inproc://pico-workstation-integration"
    rpc = SensorGatewayRpcServer(context, rpc_endpoint, core)
    client = SensorGatewayClient(
        rpc_endpoint,
        context=context,
        request_timeout_ms=250,
    )
    source = SensorGatewayVideoSource(client, max_age_ms=250.0)
    bridge = PicoVideoBridge(
        BridgeSettings(
            gateway_endpoint=rpc_endpoint,
            control_host="127.0.0.1",
            control_port=0,
            encoder="libx264",
            stats_interval_s=0.0,
        ),
        source=source,
        source_close=client.close,
    )
    bridge_thread = threading.Thread(
        target=bridge.serve_forever,
        name="integration-pico-bridge",
    )
    receiver = _VideoReceiver()
    control: socket.socket | None = None
    access_units: list[bytes] = []
    mock_started = False
    ingress_started = False
    rpc_started = False
    bridge_started = False
    try:
        mock_thread.start()
        mock_started = True
        assert mock.wait_until_ready(timeout_s=2.0)
        ingress_thread.start()
        ingress_started = True
        rpc.start()
        rpc_started = True

        deadline = time.monotonic() + 3.0
        initial_frame = None
        while time.monotonic() < deadline and initial_frame is None:
            initial_frame = source.poll()
            if initial_frame is None:
                time.sleep(0.02)
        assert initial_frame is not None, f"source_status={source.status}"
        health = client.health()
        assert {
            "camera_encoded/ego_view",
            "camera_encoded/left_wrist",
            "camera_encoded/right_wrist",
        } <= set(health["streams"])

        bridge_thread.start()
        bridge_started = True
        assert bridge.wait_until_ready(timeout_s=2.0)
        control = socket.create_connection(bridge.control_address, timeout=2.0)
        open_packet = _open_camera_packet(port=receiver.port)
        control.sendall(open_packet[:3])
        control.sendall(open_packet[3:])
        for _ in range(10):
            access_units.append(receiver.receive())

        mock.stop()
        mock_thread.join(timeout=2.0)
        assert not mock_thread.is_alive()
        for _ in range(30):
            access_units.append(receiver.receive(timeout_s=3.0))
        control.sendall(_control_packet(b"CLOSE_CAMERA", b""))
        assert receiver.wait_for_disconnect(timeout_s=3.0)
    finally:
        if control is not None:
            control.close()
        receiver.close()
        bridge.stop()
        if bridge_started:
            bridge_thread.join(timeout=4.0)
        if mock_started and mock_thread.is_alive():
            mock.stop()
            mock_thread.join(timeout=2.0)
        ingress_stop.set()
        if ingress_started:
            ingress_thread.join(timeout=2.0)
        ingress.close()
        if rpc_started:
            rpc.close()
        core.close()
        context.term()

    assert not bridge_thread.is_alive()
    assert not ingress_thread.is_alive()
    stream_path = tmp_path / "pico-workstation-loop.h264"
    stream_path.write_bytes(b"".join(access_units))
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
            "-of",
            "json",
            str(stream_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(probe.stdout)["streams"][0] == {
        "codec_name": "h264",
        "width": 1280,
        "height": 480,
    }

    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "h264",
            "-i",
            str(stream_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    frame_bytes = 1280 * 480 * 3
    assert len(raw) % frame_bytes == 0
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 480, 1280, 3)
    assert len(frames) >= 40
    eye_difference = np.abs(
        frames.astype(np.int16)[:, :, :640] - frames.astype(np.int16)[:, :, 640:]
    ).mean(axis=(1, 2, 3))
    assert np.all(eye_difference < 3.0)
    red_dominant = (
        (frames[..., 0].mean(axis=(1, 2)) > frames[..., 1].mean(axis=(1, 2)) * 1.5)
        & (frames[..., 0].mean(axis=(1, 2)) > frames[..., 2].mean(axis=(1, 2)) * 1.5)
    )
    assert np.any(~red_dominant[:10])
    live_frames = frames[:10]
    assert any(
        not np.array_equal(live_frames[0], candidate)
        for candidate in live_frames[1:]
    )
    assert red_dominant[-1]
