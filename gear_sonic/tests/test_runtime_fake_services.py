from __future__ import annotations

import time

import msgpack
import numpy as np
import pytest
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.runtime.gateway.fakes import (
    FakeCameraServer,
    FakeCppService,
    FakePolicyServer,
    pack_policy_message,
    unpack_policy_message,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message


def _request(socket: zmq.Socket, endpoint: str, data: dict | None = None) -> object:
    request = {"endpoint": endpoint}
    if data is not None:
        request["data"] = data
    socket.send(pack_policy_message(request))
    return unpack_policy_message(socket.recv())


def _publish_until_received(
    publish,
    subscriber: zmq.Socket,
    *,
    timeout_s: float = 1.0,
) -> bytes:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        publish()
        if subscriber.poll(20, zmq.POLLIN):
            return subscriber.recv()
    raise TimeoutError("fake PUB/SUB message was not received")


def test_fake_policy_server_supports_ping_numpy_action_and_reset() -> None:
    context = zmq.Context()
    server = FakePolicyServer(context, "inproc://fake-policy")
    server.start()
    client = context.socket(zmq.REQ)
    client.connect("inproc://fake-policy")
    try:
        assert _request(client, "ping")["status"] == "ok"
        action, info = _request(client, "get_action", {"observation": {}})
        assert action["action"].shape == (1, 1, 29)
        assert action["action"].dtype == np.float32
        assert info == {"source": "fake_policy"}
        assert _request(client, "reset", {"options": None}) == {"status": "ok"}
        assert server.request_count == 3
    finally:
        client.close(linger=0)
        server.stop()
        context.term()


def test_fake_policy_timeout_then_clean_restart() -> None:
    context = zmq.Context()
    endpoint = "inproc://fake-policy-recovery"
    slow_server = FakePolicyServer(context, endpoint, response_delay_s=0.1)
    slow_server.start()
    client = context.socket(zmq.REQ)
    client.setsockopt(zmq.RCVTIMEO, 20)
    client.connect(endpoint)
    try:
        client.send(pack_policy_message({"endpoint": "ping"}))
        with pytest.raises(zmq.Again):
            client.recv()
    finally:
        client.close(linger=0)
        slow_server.stop()

    recovered_server = FakePolicyServer(context, endpoint)
    recovered_server.start()
    recovered_client = context.socket(zmq.REQ)
    recovered_client.setsockopt(zmq.RCVTIMEO, 200)
    recovered_client.connect(endpoint)
    try:
        assert _request(recovered_client, "ping")["status"] == "ok"
    finally:
        recovered_client.close(linger=0)
        recovered_server.stop()
        context.term()


def test_fake_camera_preserves_current_schema_and_timestamps() -> None:
    context = zmq.Context()
    server = FakeCameraServer(context, "inproc://fake-camera")
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.connect("inproc://fake-camera")
    schema = ImageMessageSchema(
        timestamps={"ego_view": 10.25, "ego_view_depth": 10.25},
        images={
            "ego_view": np.zeros((2, 3, 3), dtype=np.uint8),
            "ego_view_depth": np.full((2, 3), 1200, dtype=np.uint16),
        },
        camera_info={"ego_view": {"depth_scale_m": 0.001}},
    )
    serialized = schema.serialize()
    try:
        wire = _publish_until_received(lambda: server.publish(serialized), subscriber)
        decoded = ImageMessageSchema.deserialize(msgpack.unpackb(wire, raw=False))
        assert decoded.timestamps == schema.timestamps
        np.testing.assert_array_equal(decoded.images["ego_view_depth"], schema.images["ego_view_depth"])
        assert decoded.camera_info == schema.camera_info
    finally:
        subscriber.close(linger=0)
        server.close()
        context.term()


def test_fake_cpp_preserves_state_prefix_and_receives_raw_planner_command() -> None:
    context = zmq.Context()
    command_publisher = context.socket(zmq.PUB)
    command_publisher.bind("inproc://fake-cpp-command")
    service = FakeCppService(
        context,
        state_endpoint="inproc://fake-cpp-state",
        command_endpoint="inproc://fake-cpp-command",
    )
    state_subscriber = context.socket(zmq.SUB)
    state_subscriber.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
    state_subscriber.connect("inproc://fake-cpp-state")
    state = {
        "control_loop_type": "cpp",
        "index": 42,
        "ros_timestamp": 123.5,
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "body_q": [0.0] * 29,
    }
    planner = build_planner_message(
        1,
        movement=(1.0, 0.0, 0.0),
        facing=(1.0, 0.0, 0.0),
        speed=0.3,
    )
    try:
        raw_state = _publish_until_received(lambda: service.publish_state(state), state_subscriber)
        assert raw_state.startswith(b"g1_debug")
        assert msgpack.unpackb(raw_state[len(b"g1_debug") :], raw=False) == state

        received_command = _publish_until_received(
            lambda: command_publisher.send(planner),
            service.command_socket,
        )
        assert received_command == planner
    finally:
        state_subscriber.close(linger=0)
        service.close()
        command_publisher.close(linger=0)
        context.term()
