from __future__ import annotations

from contextlib import contextmanager
import queue
import socket
import struct
import threading
import time
from typing import Iterator

import cv2
import numpy as np

from gear_sonic.pico_video.bridge import (
    BridgeSettings,
    LatestFrameSlot,
    PicoVideoBridge,
)
from gear_sonic.pico_video.encoder import EncoderSettings
from gear_sonic.pico_video.gateway_source import GatewayFrame


TEST_ACCESS_UNIT = b"\x00\x00\x00\x01\x09\xf0\x00\x00\x01\x65\xaa"


def _jpeg() -> bytes:
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    bgr[:, :320] = (0, 0, 255)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok
    return encoded.tobytes()


class RepeatingSource:
    def __init__(self, jpeg: bytes) -> None:
        self.jpeg = jpeg
        self.sequence = 0
        self.status = "READY"

    def poll(self) -> GatewayFrame:
        self.sequence += 1
        return GatewayFrame(
            jpeg=self.jpeg,
            generation=1,
            sequence=self.sequence,
            received_timestamp_ns=time.monotonic_ns(),
            source_timestamp_ns=time.time_ns(),
            source_shape=(480, 640, 3),
        )


class StaleSource:
    status = "SENSOR FRAME STALE"

    def poll(self) -> None:
        return None


class RecordingEncoder:
    def __init__(self, settings: EncoderSettings) -> None:
        self.settings = settings
        self.frames: list[np.ndarray] = []
        self._output: queue.Queue[bytes | None] = queue.Queue()
        self._closed = False

    def write_frame(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())
        self._output.put(TEST_ACCESS_UNIT)

    def iter_access_units(
        self, stop_event: threading.Event | None = None
    ) -> Iterator[bytes]:
        while stop_event is None or not stop_event.is_set():
            try:
                item = self._output.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                return
            yield item

    def finish_input(self) -> None:
        self._output.put(None)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._output.put(None)


class RecordingEncoderFactory:
    def __init__(self) -> None:
        self.instances: list[RecordingEncoder] = []

    def __call__(self, settings: EncoderSettings) -> RecordingEncoder:
        encoder = RecordingEncoder(settings)
        self.instances.append(encoder)
        return encoder


def _control_packet(command: bytes, data: bytes) -> bytes:
    body = struct.pack("<I", len(command)) + command
    body += struct.pack("<I", len(data)) + data
    return struct.pack(">I", len(body)) + body


def _open_camera_packet(*, ip: str, port: int, width: int = 1280) -> bytes:
    camera = b"ZED"
    encoded_ip = ip.encode("utf-8")
    payload = b"\xca\xfe\x01" + struct.pack(
        "<7i", width, 480, 30, 4_000_000, 0, 0, port
    )
    payload += bytes((len(camera),)) + camera
    payload += bytes((len(encoded_ip),)) + encoded_ip
    return _control_packet(b"OPEN_CAMERA", payload)


def _close_camera_packet() -> bytes:
    return _control_packet(b"CLOSE_CAMERA", b"")


def _read_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise EOFError("video connection closed")
        data.extend(chunk)
    return bytes(data)


class FakePicoVideoReceiver:
    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self._listener.settimeout(3.0)
        self.port = self._listener.getsockname()[1]
        self._connection: socket.socket | None = None

    def __enter__(self) -> "FakePicoVideoReceiver":
        return self

    def _accept(self) -> socket.socket:
        if self._connection is None:
            self._connection, _ = self._listener.accept()
            self._connection.settimeout(3.0)
        return self._connection

    def receive_access_unit(self) -> bytes:
        connection = self._accept()
        size = struct.unpack(">I", _read_exact(connection, 4))[0]
        return _read_exact(connection, size)

    def wait_for_disconnect(self, *, timeout_s: float) -> bool:
        connection = self._accept()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            connection.settimeout(max(0.01, deadline - time.monotonic()))
            try:
                if connection.recv(65536) == b"":
                    return True
            except socket.timeout:
                return False
        return False

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        if self._connection is not None:
            self._connection.close()
        self._listener.close()


@contextmanager
def running_bridge(
    *,
    source: RepeatingSource | StaleSource,
    encoder_factory: RecordingEncoderFactory,
) -> Iterator[PicoVideoBridge]:
    bridge = PicoVideoBridge(
        BridgeSettings(
            gateway_endpoint="inproc://unused",
            control_host="127.0.0.1",
            control_port=0,
            encoder="libx264",
        ),
        source=source,
        encoder_factory=encoder_factory,
    )
    thread = threading.Thread(target=bridge.serve_forever, name="test-pico-bridge")
    thread.start()
    assert bridge.wait_until_ready(timeout_s=2.0)
    try:
        yield bridge
    finally:
        bridge.stop()
        thread.join(timeout=3.0)
        assert not thread.is_alive()


def test_open_camera_streams_framed_access_units_to_requested_loopback_port() -> None:
    factory = RecordingEncoderFactory()
    with FakePicoVideoReceiver() as pico, running_bridge(
        source=RepeatingSource(_jpeg()), encoder_factory=factory
    ) as bridge:
        with socket.create_connection(bridge.control_address, timeout=1.0) as control:
            control.sendall(_open_camera_packet(ip="127.0.0.1", port=pico.port))
            access_unit = pico.receive_access_unit()

    assert access_unit == TEST_ACCESS_UNIT
    assert factory.instances[0].settings.width == 1280
    assert factory.instances[0].settings.height == 480


def test_close_camera_stops_video_without_hanging_control_connection() -> None:
    factory = RecordingEncoderFactory()
    with FakePicoVideoReceiver() as pico, running_bridge(
        source=RepeatingSource(_jpeg()), encoder_factory=factory
    ) as bridge:
        with socket.create_connection(bridge.control_address, timeout=1.0) as control:
            control.sendall(_open_camera_packet(ip="127.0.0.1", port=pico.port))
            pico.receive_access_unit()
            control.sendall(_close_camera_packet())
            assert pico.wait_for_disconnect(timeout_s=2.0)


def test_second_open_replaces_first_session() -> None:
    factory = RecordingEncoderFactory()
    with FakePicoVideoReceiver() as first, FakePicoVideoReceiver() as second, running_bridge(
        source=RepeatingSource(_jpeg()), encoder_factory=factory
    ) as bridge:
        with socket.create_connection(bridge.control_address, timeout=1.0) as control:
            control.sendall(_open_camera_packet(ip="127.0.0.1", port=first.port))
            first.receive_access_unit()
            control.sendall(_open_camera_packet(ip="127.0.0.1", port=second.port))
            assert first.wait_for_disconnect(timeout_s=2.0)
            assert second.receive_access_unit() == TEST_ACCESS_UNIT

    assert len(factory.instances) == 2


def test_invalid_open_is_isolated_and_next_valid_open_still_streams() -> None:
    factory = RecordingEncoderFactory()
    with FakePicoVideoReceiver() as pico, running_bridge(
        source=RepeatingSource(_jpeg()), encoder_factory=factory
    ) as bridge:
        with socket.create_connection(bridge.control_address, timeout=1.0) as control:
            control.sendall(
                _open_camera_packet(ip="127.0.0.1", port=pico.port, width=640)
            )
            control.sendall(_open_camera_packet(ip="127.0.0.1", port=pico.port))
            assert pico.receive_access_unit() == TEST_ACCESS_UNIT

    assert len(factory.instances) == 1


def test_unreachable_video_endpoint_and_unknown_command_do_not_kill_control() -> None:
    unavailable = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    unavailable.bind(("127.0.0.1", 0))
    unavailable_port = unavailable.getsockname()[1]
    unavailable.close()
    factory = RecordingEncoderFactory()
    with FakePicoVideoReceiver() as pico, running_bridge(
        source=RepeatingSource(_jpeg()), encoder_factory=factory
    ) as bridge:
        with socket.create_connection(bridge.control_address, timeout=1.0) as control:
            control.sendall(
                _open_camera_packet(ip="127.0.0.1", port=unavailable_port)
            )
            control.sendall(_control_packet(b"UNKNOWN_COMMAND", b"ignored"))
            control.sendall(_open_camera_packet(ip="127.0.0.1", port=pico.port))
            assert pico.receive_access_unit() == TEST_ACCESS_UNIT


def test_control_disconnect_stops_associated_video_session() -> None:
    factory = RecordingEncoderFactory()
    with FakePicoVideoReceiver() as pico, running_bridge(
        source=RepeatingSource(_jpeg()), encoder_factory=factory
    ) as bridge:
        control = socket.create_connection(bridge.control_address, timeout=1.0)
        control.sendall(_open_camera_packet(ip="127.0.0.1", port=pico.port))
        pico.receive_access_unit()
        control.close()
        assert pico.wait_for_disconnect(timeout_s=2.0)


def test_latest_frame_slot_replaces_unconsumed_frame() -> None:
    slot = LatestFrameSlot()
    first = np.zeros((2, 4, 3), dtype=np.uint8)
    second = np.full((2, 4, 3), 255, dtype=np.uint8)

    slot.put(first)
    slot.put(second)

    version, output = slot.get_after(0, timeout_s=0.1)
    assert version == 2
    np.testing.assert_array_equal(output, second)


def test_stale_source_streams_a_fresh_red_status_card() -> None:
    factory = RecordingEncoderFactory()
    with FakePicoVideoReceiver() as pico, running_bridge(
        source=StaleSource(), encoder_factory=factory
    ) as bridge:
        with socket.create_connection(bridge.control_address, timeout=1.0) as control:
            control.sendall(_open_camera_packet(ip="127.0.0.1", port=pico.port))
            pico.receive_access_unit()

    frame = factory.instances[0].frames[0]
    assert frame[..., 0].mean() > frame[..., 1].mean() * 1.5
    assert frame[..., 0].mean() > frame[..., 2].mean() * 1.5
