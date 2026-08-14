from __future__ import annotations

import base64
from types import SimpleNamespace
import threading
import time

import msgpack
import cv2
import numpy as np
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.runtime.fakes import FakeCameraServer, FakeCppService
from gear_sonic.runtime.client import SensorGatewayClient
from gear_sonic.runtime.sensor_gateway import (
    CameraZmqIngress,
    CppStateZmqIngress,
    LingBotDepthZmqIngress,
    Ros2SensorIngress,
    SensorGatewayCore,
    SensorGatewayRpc,
    SensorGatewayRpcServer,
    imu_array,
    livox_xyz_array,
    odometry_array,
    pointcloud2_xyz_array,
    ros_stamp_ns,
)
from gear_sonic.runtime.shared_memory import read_shared_memory_frame
from gear_sonic.runtime.snapshot import SnapshotRequest


def _publish_until_ingested(publish, poll, *, timeout_s: float = 1.0) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        publish()
        count = poll()
        if count:
            return count
        time.sleep(0.005)
    raise TimeoutError("SensorGateway ingress did not receive the fake message")


class _RecordingPreview:
    def __init__(self) -> None:
        self.start_count = 0
        self.frames = []
        self.close_count = 0

    def start(self) -> None:
        self.start_count += 1

    def publish(self, images) -> None:
        self.frames.append(dict(images))

    def close(self) -> None:
        self.close_count += 1


def test_camera_ingress_owns_rgb_preview_lifecycle_and_forwards_rgb_only() -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=2, history_size=4)
    camera_server = FakeCameraServer(context, "inproc://gateway-rgb-preview")
    preview = _RecordingPreview()
    ingress = CameraZmqIngress(
        context,
        "inproc://gateway-rgb-preview",
        core,
        preview_rgb=True,
        preview_worker=preview,
    )
    schema = ImageMessageSchema(
        timestamps={"ego_view": 100.0, "chest_view": 100.0, "ego_view_depth": 100.0},
        images={
            "ego_view": np.full((2, 3, 3), 10, dtype=np.uint8),
            "chest_view": np.full((2, 3, 3), 20, dtype=np.uint8),
            "ego_view_depth": np.full((2, 3), 1200, dtype=np.uint16),
        },
        camera_info={},
    )

    try:
        assert _publish_until_ingested(
            lambda: camera_server.publish(schema.serialize()),
            ingress.poll_once,
        ) == 3
        assert preview.start_count == 1
        assert len(preview.frames) == 1
        assert tuple(preview.frames[0]) == ("ego_view", "chest_view")
    finally:
        ingress.close()
        camera_server.close()
        core.close()
        context.term()

    assert preview.close_count == 1


def test_gateway_core_resizes_ring_without_resetting_sequence() -> None:
    core = SensorGatewayCore(slot_count=2, history_size=4, retired_ring_ttl_s=10.0)
    try:
        first = core.publish_array(
            "lidar/points_xyz",
            np.zeros((1, 3), dtype=np.float32),
            received_ns=100,
        )
        second = core.publish_array(
            "lidar/points_xyz",
            np.ones((4, 3), dtype=np.float32),
            received_ns=200,
        )

        assert first.metadata.sequence == 0
        assert second.metadata.sequence == 1
        assert first.shared_memory != second.shared_memory
        np.testing.assert_array_equal(
            read_shared_memory_frame(second),
            np.ones((4, 3), dtype=np.float32),
        )
        health = core.health_payload(now_ns=250)
        assert health["retired_ring_count"] == 1
        assert health["streams"]["lidar/points_xyz"]["message_count"] == 2
        assert health["streams"]["lidar/points_xyz"]["out_of_order_messages"] == 0
    finally:
        core.close()


def test_gateway_reports_enabled_sources_before_and_after_first_message() -> None:
    core = SensorGatewayCore(slot_count=2, history_size=4)
    try:
        core.register_endpoint("source/test", expected_hz=10.0, started_ns=1_000_000_000)
        waiting = core.health_payload(now_ns=1_050_000_000)
        assert waiting["streams"]["source/test"]["state"] == "waiting"

        core.observe_endpoint(
            "source/test",
            expected_hz=10.0,
            received_ns=1_100_000_000,
        )
        healthy = core.health_payload(now_ns=1_150_000_000)
        assert healthy["streams"]["source/test"]["state"] == "healthy"
        assert healthy["streams"]["source/test"]["message_count"] == 1
    finally:
        core.close()


def test_high_rate_stream_retains_every_snapshot_metadata_slot() -> None:
    core = SensorGatewayCore(
        slot_count=8,
        history_size=64,
        frame_ttl_ms=1000,
    )
    try:
        core.publish_array(
            "camera/chest_view",
            np.zeros((2, 2, 3), dtype=np.uint8),
            received_ns=1_000_000_000,
            expected_hz=30.0,
        )
        for index in range(40):
            core.publish_array(
                "ros/livox_imu",
                np.full((10,), index, dtype=np.float64),
                received_ns=1_000_000_000 + index * 5_000_000,
                expected_hz=200.0,
            )

        snapshot = core.select(
            SnapshotRequest(
                streams=("camera/chest_view", "ros/livox_imu"),
                max_age_ms=500.0,
                max_skew_ms=10.0,
                anchor_timestamp_ns=1_000_000_000,
            ),
            now_ns=1_200_000_000,
        )

        assert snapshot.complete
        np.testing.assert_array_equal(
            read_shared_memory_frame(snapshot.frames["ros/livox_imu"]),
            np.zeros((10,), dtype=np.float64),
        )
        health = core.health_payload(now_ns=1_200_000_000)
        assert health["shared_memory"]["camera/chest_view"]["slot_count"] == 32
        assert health["shared_memory"]["ros/livox_imu"]["slot_count"] == 64
    finally:
        core.close()


def test_snapshot_and_health_rpc_return_shared_memory_metadata() -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=2, history_size=4)
    rpc = SensorGatewayRpc(context, "inproc://sensor-gateway-rpc", core)
    client = context.socket(zmq.REQ)
    client.connect("inproc://sensor-gateway-rpc")
    try:
        now_ns = time.monotonic_ns()
        core.publish_array("camera/ego_view", np.zeros((2, 3, 3), np.uint8), received_ns=now_ns)
        core.publish_array("cpp/state_msgpack", np.zeros((8,), np.uint8), received_ns=now_ns)
        request = SnapshotRequest(
            streams=("camera/ego_view", "cpp/state_msgpack"),
            max_age_ms=1000.0,
            max_skew_ms=10.0,
        )
        client.send_json(request.to_dict())
        assert rpc.serve_once(timeout_ms=100)
        snapshot = client.recv_json()
        assert snapshot["complete"]
        assert set(snapshot["frames"]) == {"camera/ego_view", "cpp/state_msgpack"}

        client.send_json({"type": "sonic.sensor_gateway_health_request", "version": 1})
        assert rpc.serve_once(timeout_ms=100)
        health = client.recv_json()
        assert set(health["streams"]) == {"camera/ego_view", "cpp/state_msgpack"}
    finally:
        client.close(linger=0)
        rpc.close()
        core.close()
        context.term()


def test_rpc_server_owns_rep_socket_on_a_dedicated_thread() -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=2, history_size=4)
    server = SensorGatewayRpcServer(context, "inproc://sensor-gateway-thread", core)
    client = SensorGatewayClient(
        "inproc://sensor-gateway-thread",
        context=context,
        request_timeout_ms=100,
    )
    try:
        server.start()
        assert client.ping()
    finally:
        client.close()
        server.close()
        core.close()
        context.term()


def test_large_ring_write_does_not_hold_the_global_metadata_lock() -> None:
    core = SensorGatewayCore(slot_count=2, history_size=4)
    release = threading.Event()
    entered = threading.Event()
    try:
        core.publish_array("camera/ego_view", np.zeros((2, 2, 3), np.uint8))
        ring = core._rings["camera/ego_view"]
        original_write = ring.write

        def slow_write(*args, **kwargs):
            entered.set()
            assert release.wait(1.0)
            return original_write(*args, **kwargs)

        ring.write = slow_write
        publisher = threading.Thread(
            target=lambda: core.publish_array(
                "camera/ego_view",
                np.ones((2, 2, 3), np.uint8),
            )
        )
        publisher.start()
        assert entered.wait(1.0)

        started = time.perf_counter()
        health = core.health_payload()
        assert time.perf_counter() - started < 0.05
        assert "camera/ego_view" in health["streams"]

        release.set()
        publisher.join(1.0)
        assert not publisher.is_alive()
    finally:
        release.set()
        core.close()


def test_camera_and_cpp_ingress_are_read_only_copies_of_current_wires(monkeypatch) -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=2, history_size=4)
    camera_server = FakeCameraServer(context, "inproc://gateway-camera")
    command_publisher = context.socket(zmq.PUB)
    command_publisher.bind("inproc://gateway-unused-command")
    cpp_service = FakeCppService(
        context,
        state_endpoint="inproc://gateway-cpp-state",
        command_endpoint="inproc://gateway-unused-command",
    )
    camera_ingress = CameraZmqIngress(context, "inproc://gateway-camera", core)
    state_ingress = CppStateZmqIngress(context, "inproc://gateway-cpp-state", core)
    rgb_names = ("ego_view", "chest_view", "left_wrist", "right_wrist")
    depth_names = ("ego_view_depth", "chest_view_depth")
    camera_schema = ImageMessageSchema(
        timestamps={name: 100.0 for name in (*rgb_names, *depth_names)},
        images={
            **{
                name: np.full((2, 3, 3), index, dtype=np.uint8)
                for index, name in enumerate(rgb_names, start=1)
            },
            **{
                name: np.full((2, 3), 1200, dtype=np.uint16)
                for name in depth_names
            },
        },
        camera_info={
            "ego_view": {"depth_scale_m": 0.001, "fx": 500.0},
            "chest_view": {"depth_scale_m": 0.001, "fx": 510.0},
        },
    )
    camera_wire = camera_schema.serialize()
    publish_order = []
    original_publish_array = core.publish_array

    def record_publish_order(stream, array, **kwargs):
        publish_order.append(stream)
        return original_publish_array(stream, array, **kwargs)

    monkeypatch.setattr(core, "publish_array", record_publish_order)
    state = {
        "control_loop_type": "cpp",
        "index": 42,
        "ros_timestamp": 101.0,
        "base_quat": [1.0, 0.0, 0.0, 0.0],
    }
    robot_config = {
        "robot": "g1",
        "joint_names": ["left_hip_pitch", "right_hip_pitch"],
        "control_dt": 0.02,
    }
    try:
        assert _publish_until_ingested(
            lambda: camera_server.publish(camera_wire),
            camera_ingress.poll_once,
        ) == 6
        encoded_positions = [
            publish_order.index(f"camera_encoded/{name}") for name in rgb_names
        ]
        decoded_positions = [
            index
            for index, stream in enumerate(publish_order)
            if stream.startswith("camera/")
        ]
        assert max(encoded_positions) < min(decoded_positions)
        assert _publish_until_ingested(
            lambda: cpp_service.publish_state(state),
            state_ingress.poll_once,
        ) == 1
        assert _publish_until_ingested(
            lambda: cpp_service.publish_state(robot_config, topic="robot_config"),
            state_ingress.poll_once,
        ) == 1

        snapshot = core.select(
            SnapshotRequest(
                streams=(
                    "camera/ego_view",
                    "camera_encoded/ego_view",
                    "camera/ego_view_depth",
                    "cpp/state_msgpack",
                    "cpp/robot_config_msgpack",
                ),
                max_age_ms=1000.0,
                max_skew_ms=100.0,
            )
        )
        assert snapshot.complete
        np.testing.assert_array_equal(
            read_shared_memory_frame(snapshot.frames["camera/ego_view"]),
            camera_schema.images["ego_view"],
        )
        encoded_frame = snapshot.frames["camera_encoded/ego_view"]
        assert encoded_frame.attributes["encoding"] == "jpeg_bytes"
        assert encoded_frame.attributes["image_shape"] == [2, 3, 3]
        encoded_rgb = read_shared_memory_frame(encoded_frame).tobytes()
        assert encoded_rgb == camera_wire["images"]["ego_view"]
        assert snapshot.frames["camera/ego_view_depth"].attributes["camera_info"]["fx"] == 500.0
        raw_state = read_shared_memory_frame(snapshot.frames["cpp/state_msgpack"]).tobytes()
        assert msgpack.unpackb(raw_state, raw=False) == state
        raw_config = read_shared_memory_frame(
            snapshot.frames["cpp/robot_config_msgpack"]
        ).tobytes()
        assert msgpack.unpackb(raw_config, raw=False) == robot_config
        assert cpp_service.receive_command(timeout_ms=0) is None

        legacy_wire = {
            **camera_wire,
            "images": {
                "ego_view": base64.b64encode(
                    camera_wire["images"]["ego_view"]
                ).decode("ascii")
            },
            "timestamps": {"ego_view": 100.0},
            "image_shapes": {"ego_view": [2, 3, 3]},
        }
        assert _publish_until_ingested(
            lambda: camera_server.publish(legacy_wire),
            camera_ingress.poll_once,
        ) == 1
        legacy_snapshot = core.select(
            SnapshotRequest(
                streams=("camera_encoded/ego_view",),
                max_age_ms=1000.0,
                max_skew_ms=100.0,
            )
        )
        legacy_frame = legacy_snapshot.frames["camera_encoded/ego_view"]
        assert legacy_frame.attributes["encoding"] == "base64_jpeg"
        assert legacy_frame.attributes["image_shape"] == [2, 3, 3]
    finally:
        state_ingress.close()
        camera_ingress.close()
        cpp_service.close()
        command_publisher.close(linger=0)
        camera_server.close()
        core.close()
        context.term()


def test_lingbot_depth_ingress_publishes_derived_depth_only() -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=2, history_size=4)
    server = FakeCameraServer(context, "inproc://gateway-lingbot")
    ingress = LingBotDepthZmqIngress(
        context,
        "inproc://gateway-lingbot",
        core,
        expected_hz=2.0,
    )
    depth = np.full((2, 3), 1250, dtype=np.uint16)
    schema = ImageMessageSchema(
        timestamps={"chest_view": 123.0, "chest_view_depth": 123.0},
        images={
            "chest_view": np.full((2, 3, 3), 7, dtype=np.uint8),
            "chest_view_depth": depth,
        },
        camera_info={
            "chest_view": {
                "fx": 500.0,
                "fy": 500.0,
                "cx": 1.0,
                "cy": 1.0,
                "width": 3,
                "height": 2,
                "depth_scale_m": 0.001,
                "depth_aligned_to": "chest_view",
            }
        },
    )
    try:
        assert _publish_until_ingested(
            lambda: server.publish(schema.serialize()),
            ingress.poll_once,
        ) == 1
        snapshot = core.select(
            SnapshotRequest(
                streams=("derived/lingbot_depth",),
                max_age_ms=1000.0,
                max_skew_ms=5.0,
            )
        )
        assert snapshot.complete
        frame = snapshot.frames["derived/lingbot_depth"]
        np.testing.assert_array_equal(read_shared_memory_frame(frame), depth)
        assert frame.source_timestamp_ns == 123_000_000_000
        assert frame.source_clock == "camera_unix"
        assert frame.attributes["depth_source"] == "lingbot-depth"
        assert frame.attributes["camera_info"]["depth_aligned_to"] == "chest_view"
    finally:
        ingress.close()
        server.close()
        core.close()
        context.term()


def test_ros_message_conversion_helpers_preserve_documented_axes() -> None:
    vector = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
    quaternion = SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9)
    stamp = SimpleNamespace(sec=12, nanosec=345)
    odometry = SimpleNamespace(
        header=SimpleNamespace(stamp=stamp),
        pose=SimpleNamespace(
            pose=SimpleNamespace(position=vector(1.0, 2.0, 3.0), orientation=quaternion)
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(linear=vector(4.0, 5.0, 6.0), angular=vector(7.0, 8.0, 9.0))
        ),
    )
    imu = SimpleNamespace(
        orientation=quaternion,
        angular_velocity=vector(1.0, 2.0, 3.0),
        linear_acceleration=vector(4.0, 5.0, 6.0),
    )
    lidar = SimpleNamespace(points=[vector(1.0, 2.0, 3.0), vector(4.0, 5.0, 6.0)])

    assert ros_stamp_ns(odometry) == 12_000_000_345
    np.testing.assert_allclose(
        odometry_array(odometry),
        [1, 2, 3, 0.1, 0.2, 0.3, 0.9, 4, 5, 6, 7, 8, 9],
    )
    np.testing.assert_allclose(
        imu_array(imu),
        [0.1, 0.2, 0.3, 0.9, 1, 2, 3, 4, 5, 6],
    )
    np.testing.assert_allclose(livox_xyz_array(lidar), [[1, 2, 3], [4, 5, 6]])


def test_pointcloud2_conversion_supports_structured_ros_arrays() -> None:
    raw = np.asarray(
        [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
    )
    module = SimpleNamespace(read_points=lambda *_args, **_kwargs: raw)

    converted = pointcloud2_xyz_array(object(), module)

    assert converted.dtype == np.float32
    assert converted.flags.c_contiguous
    np.testing.assert_allclose(converted, [[1, 2, 3], [4, 5, 6]])


def test_ros_ingress_callbacks_publish_all_existing_topic_types_without_ros_runtime() -> None:
    vector = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
    quaternion = SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9)
    header = SimpleNamespace(
        stamp=SimpleNamespace(sec=12, nanosec=345),
        frame_id="sensor_frame",
    )
    odometry = SimpleNamespace(
        header=header,
        child_frame_id="body",
        pose=SimpleNamespace(
            pose=SimpleNamespace(position=vector(1.0, 2.0, 3.0), orientation=quaternion)
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(linear=vector(4.0, 5.0, 6.0), angular=vector(7.0, 8.0, 9.0))
        ),
    )
    imu = SimpleNamespace(
        header=header,
        orientation=quaternion,
        angular_velocity=vector(1.0, 2.0, 3.0),
        linear_acceleration=vector(4.0, 5.0, 6.0),
    )
    lidar = SimpleNamespace(header=header, points=[vector(1.0, 2.0, 3.0)])
    cloud_values = np.asarray(
        [(4.0, 5.0, 6.0)],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
    )
    cloud = SimpleNamespace(header=header)

    core = SensorGatewayCore(slot_count=2, history_size=4)
    ingress = object.__new__(Ros2SensorIngress)
    ingress.core = core
    ingress._expected_hz = {
        "source/ros_lidar": 10.0,
        "source/ros_imu": 200.0,
        "source/ros_odometry": 100.0,
        "source/ros_registered_cloud": 10.0,
    }
    ingress._topics = {
        "lidar": "/livox/lidar",
        "imu": "/livox/imu",
        "odometry": "/Odometry_loc",
        "registered_cloud": "/cloud_registered_1",
    }
    ingress._point_cloud2 = SimpleNamespace(
        read_points=lambda *_args, **_kwargs: cloud_values
    )
    try:
        ingress._on_lidar(lidar)
        ingress._on_imu(imu)
        ingress._on_odometry(odometry)
        ingress._on_registered_cloud(cloud)

        streams = (
            "ros/livox_lidar_xyz",
            "ros/livox_imu",
            "ros/odometry",
            "ros/registered_cloud_xyz",
        )
        snapshot = core.select(
            SnapshotRequest(streams=streams, max_age_ms=1000.0, max_skew_ms=100.0)
        )
        assert snapshot.complete
        np.testing.assert_allclose(
            read_shared_memory_frame(snapshot.frames["ros/livox_lidar_xyz"]),
            [[1.0, 2.0, 3.0]],
        )
        np.testing.assert_allclose(
            read_shared_memory_frame(snapshot.frames["ros/registered_cloud_xyz"]),
            [[4.0, 5.0, 6.0]],
        )
        assert snapshot.frames["ros/odometry"].attributes["child_frame_id"] == "body"
        assert snapshot.frames["ros/livox_imu"].source_timestamp_ns == 12_000_000_345

        health = core.health_payload()
        assert health["streams"]["source/ros_lidar"]["state"] == "healthy"
        assert health["streams"]["source/ros_registered_cloud"]["message_count"] == 1
    finally:
        core.close()
