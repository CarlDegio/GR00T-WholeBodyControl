from __future__ import annotations

import socket
import threading

import msgpack
import numpy as np
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.pico_video.mock_camera import MockCameraPublisher, MockCameraSettings
from gear_sonic.scripts.run_mock_camera_server import build_argument_parser


def _free_tcp_port() -> int:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    return port


def test_mock_camera_publishes_changing_pico_views_in_production_camera_schema() -> None:
    port = _free_tcp_port()
    publisher = MockCameraPublisher(
        MockCameraSettings(port=port, width=640, height=480, fps=30, jpeg_quality=95)
    )
    thread = threading.Thread(target=publisher.run, name="test-mock-camera")
    thread.start()
    assert publisher.wait_until_ready(timeout_s=2.0)
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.setsockopt(zmq.RCVTIMEO, 2000)
    subscriber.connect(f"tcp://127.0.0.1:{port}")
    try:
        first_wire = msgpack.unpackb(subscriber.recv(), raw=False)
        second_wire = msgpack.unpackb(subscriber.recv(), raw=False)
        first = ImageMessageSchema.deserialize(first_wire)
        second = ImageMessageSchema.deserialize(second_wire)
    finally:
        subscriber.close(linger=0)
        context.term()
        publisher.stop()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert first_wire["schema_version"] == 2
    assert isinstance(first_wire["images"]["ego_view"], bytes)
    assert set(first.images) == {"ego_view", "left_wrist", "right_wrist"}
    for name in first.images:
        assert first.images[name].shape == (480, 640, 3)
        assert first.image_shapes[name] == [480, 640, 3]
        assert first.timestamps[name] == first.timestamps["ego_view"]
    assert first.timestamps["ego_view"] < second.timestamps["ego_view"]
    assert not np.array_equal(first.images["ego_view"], second.images["ego_view"])


def test_mock_camera_cli_defaults_match_the_head_camera_profile() -> None:
    args = build_argument_parser().parse_args([])

    assert args.port == 5555
    assert args.width == 640
    assert args.height == 480
    assert args.fps == 30
    assert args.jpeg_quality == 95
