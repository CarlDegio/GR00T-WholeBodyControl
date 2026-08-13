from __future__ import annotations

import base64
from types import MappingProxyType
import threading
import time

import msgpack
import msgpack_numpy as mnp
import numpy as np
import pytest
import zmq

from gear_sonic.runtime.client import MaterializedSnapshot, SensorGatewayClient
from gear_sonic.runtime.contracts import MessageMetadata, SharedMemoryFrame
from gear_sonic.runtime.sensor_gateway import SensorGatewayCore, SensorGatewayRpc
from gear_sonic.runtime.snapshot import SensorSnapshot, TimestampBasis
from gear_sonic.runtime.vla_sensor_gateway import (
    VLA_CAMERA_NAMES,
    VLA_CAMERA_STREAMS,
    VLA_STATE_STREAM,
    VlaSensorGatewayIngress,
    camera_message_from_snapshot,
    decode_cpp_state_array,
)


def _materialized_camera(
    jpeg_payloads: dict[str, bytes],
    *,
    received_ns: int,
    source_ns: int,
    encoding: str = "jpeg_bytes",
    image_shape: tuple[int, int, int] = (3, 4, 3),
) -> MaterializedSnapshot:
    arrays = {
        f"camera_encoded/{name}": np.frombuffer(payload, np.uint8).copy()
        for name, payload in jpeg_payloads.items()
    }
    frames = {
        stream: SharedMemoryFrame(
            metadata=MessageMetadata(
                source="sensor_gateway",
                sequence=index,
                timestamp_ns=received_ns,
                ttl_ms=1000,
            ),
            stream=stream,
            shared_memory=f"fake-{stream}",
            shape=tuple(image.shape),
            dtype=image.dtype.str,
            offset_bytes=8,
            size_bytes=image.nbytes,
            source_timestamp_ns=source_ns + index,
            source_clock="camera_unix",
            attributes={
                "encoding": encoding,
                "image_shape": image_shape,
                "camera_info": {"name": stream},
            },
        )
        for index, (stream, image) in enumerate(arrays.items())
    }
    return MaterializedSnapshot(
        snapshot=SensorSnapshot(
            complete=True,
            reason="",
            anchor_timestamp_ns=received_ns,
            timestamp_basis=TimestampBasis.RECEIVE,
            frames=MappingProxyType(frames),
            skew_ms=0.0,
            ages_ms=MappingProxyType({stream: 0.0 for stream in arrays}),
        ),
        arrays=MappingProxyType(arrays),
        attempts=1,
    )


def test_gateway_recreates_vla_four_camera_message_from_jpeg_bytes() -> None:
    jpeg_payloads = {
        name: b"\xff\xd8" + bytes([index, index + 1]) + b"\xff\xd9"
        for index, name in enumerate(VLA_CAMERA_NAMES)
    }
    message = camera_message_from_snapshot(
        _materialized_camera(
            jpeg_payloads,
            received_ns=time.monotonic_ns(),
            source_ns=123_000_000_000,
        )
    )

    assert VLA_CAMERA_STREAMS == tuple(
        f"camera_encoded/{name}" for name in VLA_CAMERA_NAMES
    )
    assert tuple(message["images"]) == VLA_CAMERA_NAMES
    for index, name in enumerate(VLA_CAMERA_NAMES):
        assert message["images"][name] == jpeg_payloads[name]
        assert message["image_shapes"][name] == (3, 4, 3)
        assert message["timestamps"][name] == pytest.approx(
            (123_000_000_000 + index) * 1.0e-9
        )
        assert message["camera_info"][name] == {
            "name": f"camera_encoded/{name}"
        }


def test_gateway_decodes_base64_camera_payload_to_jpeg_bytes() -> None:
    jpeg_payloads = {
        name: b"\xff\xd8" + bytes([index + 10]) + b"\xff\xd9"
        for index, name in enumerate(VLA_CAMERA_NAMES)
    }
    base64_payloads = {
        name: base64.b64encode(payload) for name, payload in jpeg_payloads.items()
    }

    message = camera_message_from_snapshot(
        _materialized_camera(
            base64_payloads,
            received_ns=time.monotonic_ns(),
            source_ns=123_000_000_000,
            encoding="base64_jpeg",
        )
    )

    assert message["images"] == jpeg_payloads


def test_gateway_decodes_cpp_state_like_the_legacy_subscriber() -> None:
    expected = {
        "body_q": np.arange(29, dtype=np.float32),
        "left_hand_q": [0.1, 0.2],
        "right_hand_q": [0.3, 0.4],
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "ros_timestamp": 123.5,
    }
    payload = msgpack.packb(expected, default=mnp.encode, use_bin_type=True)

    decoded = decode_cpp_state_array(np.frombuffer(payload, dtype=np.uint8))

    np.testing.assert_array_equal(decoded["body_q"], expected["body_q"])
    np.testing.assert_array_equal(decoded["left_hand_q"], expected["left_hand_q"])
    np.testing.assert_array_equal(decoded["right_hand_q"], expected["right_hand_q"])
    np.testing.assert_array_equal(decoded["base_quat"], expected["base_quat"])
    assert decoded["ros_timestamp"] == expected["ros_timestamp"]


class _BlockingClient:
    def read_snapshot(self, *_args, **_kwargs):
        time.sleep(0.2)
        raise RuntimeError("simulated unavailable gateway")


def test_gateway_rpc_never_blocks_the_vla_control_or_inference_threads() -> None:
    ingress = VlaSensorGatewayIngress(
        "tcp://127.0.0.1:5560",
        client=_BlockingClient(),
        poll_hz=50.0,
    )
    try:
        started = time.perf_counter()
        ingress.start()
        assert ingress.read_camera() is None
        assert ingress.read_state(clear=False) is None
        assert time.perf_counter() - started < 0.05
    finally:
        ingress.close()


class _RpcThread:
    def __init__(self, context: zmq.Context, endpoint: str, core: SensorGatewayCore) -> None:
        self.context = context
        self.endpoint = endpoint
        self.core = core
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        rpc = SensorGatewayRpc(self.context, self.endpoint, self.core)
        self.ready.set()
        try:
            while not self.stop.is_set():
                rpc.serve_once(timeout_ms=10)
        finally:
            rpc.close()

    def start(self) -> None:
        self.thread.start()
        assert self.ready.wait(1.0)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(1.0)
        assert not self.thread.is_alive()


def test_gateway_ingress_materializes_every_vla_input_and_clear_semantics() -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=8, history_size=16)
    server = _RpcThread(context, "inproc://vla-gateway-input", core)
    server.start()
    received_ns = time.monotonic_ns()
    source_ns = 123_000_000_000
    jpeg_payloads = {
        name: b"\xff\xd8" + bytes([index, index + 1]) + b"\xff\xd9"
        for index, name in enumerate(VLA_CAMERA_NAMES)
    }
    state = {
        "body_q": np.arange(29, dtype=np.float32),
        "left_hand_q": np.zeros(7, dtype=np.float32),
        "right_hand_q": np.ones(7, dtype=np.float32),
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "ros_timestamp": 123.0,
    }
    for stream, name in zip(VLA_CAMERA_STREAMS, VLA_CAMERA_NAMES, strict=True):
        core.publish_array(
            stream,
            np.frombuffer(jpeg_payloads[name], dtype=np.uint8).copy(),
            received_ns=received_ns,
            source_timestamp_ns=source_ns,
            source_clock="camera_unix",
            attributes={
                "encoding": "jpeg_bytes",
                "image_shape": (3, 4, 3),
                "camera_info": {"camera": name},
            },
        )
    payload = msgpack.packb(state, default=mnp.encode, use_bin_type=True)
    core.publish_array(
        VLA_STATE_STREAM,
        np.frombuffer(payload, dtype=np.uint8).copy(),
        received_ns=received_ns,
        source_timestamp_ns=123_000_000_000,
        source_clock="ros_time",
    )
    client = SensorGatewayClient(
        "inproc://vla-gateway-input",
        context=context,
        request_timeout_ms=100,
    )
    ingress = VlaSensorGatewayIngress(
        "inproc://vla-gateway-input",
        client=client,
        poll_hz=100.0,
    )
    try:
        ingress.start()
        deadline = time.monotonic() + 1.0
        camera = None
        cpp_state = None
        while time.monotonic() < deadline:
            camera = ingress.read_camera()
            cpp_state = ingress.read_state(clear=False)
            if camera is not None and cpp_state is not None:
                break
            time.sleep(0.01)

        assert camera is not None
        assert cpp_state is not None
        for name in VLA_CAMERA_NAMES:
            assert camera["images"][name] == jpeg_payloads[name]
            assert camera["image_shapes"][name] == (3, 4, 3)
        np.testing.assert_array_equal(cpp_state["body_q"], state["body_q"])
        assert ingress.read_state(clear=True) is not None
        assert ingress.read_state(clear=False) is None
        assert ingress.read_camera() is not None
    finally:
        ingress.close()
        client.close()
        server.close()
        core.close()
        context.term()
