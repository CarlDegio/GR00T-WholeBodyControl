from __future__ import annotations

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
from gear_sonic.runtime.data_exporter_sensor_gateway import (
    DATA_EXPORTER_ROBOT_CONFIG_STREAM,
    DATA_EXPORTER_STATE_STREAM,
    DataExporterSensorGatewayIngress,
    data_exporter_camera_message_from_snapshot,
    decode_robot_config_array,
)
from gear_sonic.runtime.sensor_gateway import SensorGatewayCore, SensorGatewayRpc
from gear_sonic.runtime.snapshot import SensorSnapshot, TimestampBasis


class _RecordingPreview:
    def __init__(self) -> None:
        self.start_count = 0
        self.published: list[dict[str, object]] = []
        self.close_count = 0

    def start(self) -> None:
        self.start_count += 1

    def publish(self, images: dict[str, object]) -> None:
        self.published.append(dict(images))

    def close(self) -> None:
        self.close_count += 1


def _materialized_encoded_camera(
    payloads: dict[str, bytes],
    encodings: dict[str, str],
) -> MaterializedSnapshot:
    received_ns = time.monotonic_ns()
    arrays = {
        f"camera_encoded/{name}": np.frombuffer(payload, dtype=np.uint8)
        for name, payload in payloads.items()
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
            shape=tuple(values.shape),
            dtype=values.dtype.str,
            offset_bytes=8,
            size_bytes=values.nbytes,
            source_timestamp_ns=123_000_000_000 + index,
            source_clock="camera_unix",
            attributes={
                "encoding": encodings[stream.removeprefix("camera_encoded/")],
                "camera_info": {"stream": stream},
            },
        )
        for index, (stream, values) in enumerate(arrays.items())
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


def test_encoded_camera_adapter_preserves_bytes_and_base64_wire_values() -> None:
    names = ("ego_view", "left_wrist")
    jpeg = b"\xff\xd8fake-jpeg\xff\xd9"
    base64_jpeg = "LzlqLzRBQVFTa1pKUmdBQkFRQUFBUT09"
    snapshot = _materialized_encoded_camera(
        {"ego_view": jpeg, "left_wrist": base64_jpeg.encode("utf-8")},
        {"ego_view": "jpeg_bytes", "left_wrist": "base64_jpeg"},
    )

    message = data_exporter_camera_message_from_snapshot(
        snapshot,
        names,
        encoded=True,
    )

    assert message["images"]["ego_view"] == jpeg
    assert message["images"]["left_wrist"] == base64_jpeg
    assert message["timestamps"]["ego_view"] == pytest.approx(123.0)
    assert message["camera_info"]["left_wrist"] == {
        "stream": "camera_encoded/left_wrist"
    }


def test_decoded_camera_adapter_preserves_rgb_array_values() -> None:
    received_ns = time.monotonic_ns()
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    stream = "camera/ego_view"
    frame = SharedMemoryFrame(
        metadata=MessageMetadata(
            source="sensor_gateway",
            sequence=0,
            timestamp_ns=received_ns,
            ttl_ms=1000,
        ),
        stream=stream,
        shared_memory="fake-camera",
        shape=image.shape,
        dtype=image.dtype.str,
        offset_bytes=8,
        size_bytes=image.nbytes,
        source_timestamp_ns=123_000_000_000,
        source_clock="camera_unix",
        attributes={"camera_info": {"color_order": "RGB"}},
    )
    snapshot = MaterializedSnapshot(
        snapshot=SensorSnapshot(
            complete=True,
            reason="",
            anchor_timestamp_ns=received_ns,
            timestamp_basis=TimestampBasis.RECEIVE,
            frames=MappingProxyType({stream: frame}),
            skew_ms=0.0,
            ages_ms=MappingProxyType({stream: 0.0}),
        ),
        arrays=MappingProxyType({stream: image}),
        attempts=1,
    )

    message = data_exporter_camera_message_from_snapshot(
        snapshot,
        ("ego_view",),
        encoded=False,
    )

    np.testing.assert_array_equal(message["images"]["ego_view"], image)
    assert message["camera_info"]["ego_view"] == {"color_order": "RGB"}


def test_robot_config_decoder_preserves_legacy_list_types() -> None:
    expected = {
        "robot": "g1",
        "joint_names": ["hip", "knee"],
        "limits": {"lower": [-1.0, -2.0]},
    }
    payload = msgpack.packb(expected, use_bin_type=True)

    decoded = decode_robot_config_array(np.frombuffer(payload, dtype=np.uint8))

    assert decoded == expected
    assert isinstance(decoded["joint_names"], list)
    assert isinstance(decoded["limits"]["lower"], list)


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


def test_data_exporter_gateway_cache_materializes_all_migrated_inputs() -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=8, history_size=16)
    server = _RpcThread(context, "inproc://data-exporter-gateway", core)
    server.start()
    received_ns = time.monotonic_ns()
    jpeg_by_name = {
        "ego_view": b"\xff\xd8ego\xff\xd9",
        "chest_view": b"\xff\xd8chest\xff\xd9",
    }
    for name, jpeg in jpeg_by_name.items():
        core.publish_array(
            f"camera_encoded/{name}",
            np.frombuffer(jpeg, dtype=np.uint8).copy(),
            received_ns=received_ns,
            source_timestamp_ns=123_000_000_000,
            source_clock="camera_unix",
            attributes={
                "encoding": "jpeg_bytes",
                "camera_info": {"camera": name},
            },
        )
    state = {
        "body_q": np.arange(29, dtype=np.float32),
        "left_hand_q": [0.1, 0.2],
        "ros_timestamp": 123.0,
    }
    core.publish_array(
        DATA_EXPORTER_STATE_STREAM,
        np.frombuffer(
            msgpack.packb(state, default=mnp.encode, use_bin_type=True),
            dtype=np.uint8,
        ).copy(),
        received_ns=received_ns,
    )
    robot_config = {"robot": "g1", "joint_names": ["hip", "knee"]}
    core.publish_array(
        DATA_EXPORTER_ROBOT_CONFIG_STREAM,
        np.frombuffer(
            msgpack.packb(robot_config, use_bin_type=True),
            dtype=np.uint8,
        ).copy(),
        received_ns=received_ns,
    )
    client = SensorGatewayClient(
        "inproc://data-exporter-gateway",
        context=context,
        request_timeout_ms=100,
    )
    ingress = DataExporterSensorGatewayIngress(
        "inproc://data-exporter-gateway",
        camera_names=("ego_view", "chest_view"),
        defer_video_encoding=True,
        client=client,
        poll_hz=100.0,
    )
    try:
        ingress.start()
        assert ingress.wait_for_robot_config(1.0) == robot_config
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
        assert camera["images"] == jpeg_by_name
        np.testing.assert_array_equal(cpp_state["body_q"], state["body_q"])
        np.testing.assert_array_equal(cpp_state["left_hand_q"], state["left_hand_q"])
        assert ingress.read_state(clear=True) is not None
        assert ingress.read_state(clear=False) is None
        assert ingress.read_camera() is not None
    finally:
        ingress.close()
        client.close()
        server.close()
        core.close()
        context.term()


def test_ingress_starts_updates_and_idempotently_closes_rgb_preview() -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=4, history_size=8)
    server = _RpcThread(context, "inproc://data-exporter-preview", core)
    server.start()
    jpeg = b"\xff\xd8preview-jpeg\xff\xd9"
    core.publish_array(
        "camera_encoded/ego_view",
        np.frombuffer(jpeg, dtype=np.uint8).copy(),
        received_ns=time.monotonic_ns(),
        source_timestamp_ns=123_000_000_000,
        source_clock="camera_unix",
        attributes={"encoding": "jpeg_bytes", "camera_info": {}},
    )
    client = SensorGatewayClient(
        "inproc://data-exporter-preview",
        context=context,
        request_timeout_ms=100,
    )
    preview = _RecordingPreview()
    ingress = DataExporterSensorGatewayIngress(
        "inproc://data-exporter-preview",
        camera_names=("ego_view",),
        defer_video_encoding=True,
        preview_rgb=True,
        preview_worker=preview,
        client=client,
        poll_hz=100.0,
    )
    try:
        ingress.start()
        deadline = time.monotonic() + 1.0
        while not preview.published and time.monotonic() < deadline:
            time.sleep(0.01)

        assert preview.start_count == 1
        assert preview.published == [{"ego_view": jpeg}]
    finally:
        ingress.close()
        ingress.close()
        client.close()
        server.close()
        core.close()
        context.term()

    assert preview.close_count == 1
