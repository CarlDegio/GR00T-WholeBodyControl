from __future__ import annotations

from types import MappingProxyType
import threading
import time

import numpy as np
import pytest
import zmq

from gear_sonic.navdp import gateway as navdp_gateway
from gear_sonic.runtime.client import MaterializedSnapshot, SensorGatewayClient
from gear_sonic.runtime.contracts import MessageMetadata, SharedMemoryFrame
from gear_sonic.runtime.sensor_gateway import SensorGatewayCore, SensorGatewayRpc
from gear_sonic.runtime.snapshot import SensorSnapshot, TimestampBasis
from gear_sonic.scripts.navdp_planner import (
    NavDPSensorGatewayIngress,
    _SharedSensors,
    _extract_camera_frame,
    _gateway_camera_frame,
    _update_slam_cloud_state,
    _update_odometry_state,
)


def _frame(
    stream: str,
    array: np.ndarray,
    *,
    sequence: int = 0,
    received_ns: int,
    source_ns: int,
    attributes: dict | None = None,
) -> SharedMemoryFrame:
    return SharedMemoryFrame(
        metadata=MessageMetadata(
            source="sensor_gateway",
            sequence=sequence,
            timestamp_ns=received_ns,
            ttl_ms=1000,
        ),
        stream=stream,
        shared_memory=f"fake-{stream}",
        shape=tuple(array.shape),
        dtype=array.dtype.str,
        offset_bytes=8,
        size_bytes=array.nbytes,
        source_timestamp_ns=source_ns,
        source_clock="test_clock",
        attributes={} if attributes is None else attributes,
    )


def _snapshot(
    arrays: dict[str, np.ndarray],
    *,
    received_ns: int,
    source_ns: int,
    attributes: dict[str, dict] | None = None,
) -> MaterializedSnapshot:
    frames = {
        stream: _frame(
            stream,
            array,
            received_ns=received_ns,
            source_ns=source_ns,
            attributes=(attributes or {}).get(stream),
        )
        for stream, array in arrays.items()
    }
    snapshot = SensorSnapshot(
        complete=True,
        reason="",
        anchor_timestamp_ns=received_ns,
        timestamp_basis=TimestampBasis.RECEIVE,
        frames=MappingProxyType(frames),
        skew_ms=0.0,
        ages_ms=MappingProxyType({stream: 0.0 for stream in arrays}),
    )
    return MaterializedSnapshot(
        snapshot=snapshot,
        arrays=MappingProxyType(arrays),
        attempts=1,
    )


def test_gateway_camera_builds_the_same_navdp_rgb_depth_and_intrinsics() -> None:
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    depth = np.arange(12, dtype=np.uint16).reshape(3, 4) + 100
    info = {
        "fx": 300.0,
        "fy": 301.0,
        "cx": 2.0,
        "cy": 1.5,
        "depth_scale_m": 0.001,
    }
    source_ns = 123_000_000_000
    materialized = _snapshot(
        {
            "camera/ego_view": rgb,
            "camera/ego_view_depth": depth,
        },
        received_ns=time.monotonic_ns(),
        source_ns=source_ns,
        attributes={"camera/ego_view": {"camera_info": info}},
    )
    legacy_rgb, legacy_depth, legacy_info = _extract_camera_frame(
        {
            "images": {"ego_view": rgb, "ego_view_depth": depth},
            "camera_info": {"ego_view": info},
        }
    )

    gateway = _gateway_camera_frame(materialized)

    np.testing.assert_array_equal(gateway.rgb, legacy_rgb)
    np.testing.assert_array_equal(gateway.depth_m, legacy_depth)
    assert gateway.camera_info == legacy_info
    assert gateway.source_timestamp_s == pytest.approx(123.0)


def test_navdp_rgb_encoder_passes_quality_95_to_opencv(monkeypatch) -> None:
    """Catch NavDP upload frames being JPEG-encoded below production quality."""
    import cv2

    real_imencode = cv2.imencode
    imencode_calls = []

    def capture_imencode(extension, image, parameters=None):
        imencode_calls.append((extension, parameters))
        if parameters is None:
            return real_imencode(extension, image)
        return real_imencode(extension, image, parameters)

    monkeypatch.setattr(cv2, "imencode", capture_imencode)

    navdp_gateway._encode_navdp_frames(
        np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8), dtype=np.float32)
    )

    assert imencode_calls == [
        (".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
        (".png", None),
    ]


def test_gateway_odometry_updates_internal_pose() -> None:
    values = np.asarray(
        [1.0, -2.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.2],
        dtype=np.float64,
    )
    gateway = _SharedSensors()

    _update_odometry_state(
        gateway,
        values.copy(),
        source_timestamp_s=123.0,
        received_monotonic_s=10.0,
    )

    assert gateway.pose is not None
    assert gateway.pose.x == pytest.approx(1.0)
    assert gateway.pose.y == pytest.approx(-2.0)
    assert gateway.pose.yaw == pytest.approx(0.0)
    assert gateway.pose_time == pytest.approx(10.0)
    assert list(gateway.pose_history) == [(123.0, gateway.pose)]
    np.testing.assert_array_equal(gateway.robot_history, [[1.0, -2.0]])


def test_slam_cloud_resets_cached_map_after_a_stale_gap() -> None:
    sensors = _SharedSensors()
    sensors.pose = navdp_gateway.Pose2D(10.0, 20.0, 0.0)
    sensors.slam_map_xy = np.asarray([[9.0, 20.0]], dtype=np.float32)
    sensors.slam_map_time = 10.0
    sensors.robot_history = np.asarray([[9.0, 20.0], [9.5, 20.0]], dtype=np.float32)

    _update_slam_cloud_state(
        sensors,
        np.asarray([[10.5, 20.5, 0.2]], dtype=np.float32),
        received_monotonic_s=12.0,
        reset_after_s=1.0,
    )

    np.testing.assert_array_equal(sensors.slam_map_xy, [[10.5, 20.5]])
    np.testing.assert_array_equal(sensors.robot_history, [[10.0, 20.0]])


def test_slam_cloud_keeps_accumulating_while_fresh() -> None:
    sensors = _SharedSensors()
    sensors.pose = navdp_gateway.Pose2D(10.0, 20.0, 0.0)
    sensors.slam_map_xy = np.asarray([[9.0, 20.0]], dtype=np.float32)
    sensors.slam_map_time = 10.0

    _update_slam_cloud_state(
        sensors,
        np.asarray([[10.5, 20.5, 0.2]], dtype=np.float32),
        received_monotonic_s=10.5,
        reset_after_s=1.0,
    )

    assert {tuple(point) for point in sensors.slam_map_xy} == {
        (9.0, 20.0),
        (10.5, 20.5),
    }


class _BlockingClient:
    def read_snapshot(self, *_args, **_kwargs):
        time.sleep(0.2)
        raise RuntimeError("simulated unavailable gateway")


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


def test_gateway_rpc_runs_off_the_navdp_control_thread() -> None:
    ingress = NavDPSensorGatewayIngress(
        "tcp://127.0.0.1:5560",
        _SharedSensors(),
        poll_hz=20.0,
        request_timeout_ms=100,
        client=_BlockingClient(),
    )
    try:
        started = time.perf_counter()
        ingress.start()
        assert ingress.poll_camera() is None
        assert time.perf_counter() - started < 0.05
    finally:
        ingress.close()


def test_gateway_ingress_materializes_all_navdp_inputs_from_one_service() -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=8, history_size=16)
    server = _RpcThread(context, "inproc://navdp-gateway-input", core)
    server.start()
    now_ns = time.monotonic_ns()
    source_ns = 123_000_000_000
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    depth = np.full((3, 4), 750, dtype=np.uint16)
    odometry = np.asarray(
        [1.0, 2.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    lidar = np.asarray([[1.0, 0.0, 0.0], [0.8, 0.2, 0.1]], dtype=np.float32)
    cloud = np.asarray([[1.1, 2.1, 0.2], [1.2, 2.2, 0.3]], dtype=np.float32)
    for stream, values, attributes in (
        (
            "camera/ego_view",
            rgb,
            {"camera_info": {"depth_scale_m": 0.001, "fx": 300.0}},
        ),
        ("camera/ego_view_depth", depth, {}),
        ("ros/odometry", odometry, {}),
        ("ros/livox_lidar_xyz", lidar, {}),
        ("ros/registered_cloud_xyz", cloud, {}),
    ):
        core.publish_array(
            stream,
            values,
            received_ns=now_ns,
            source_timestamp_ns=source_ns,
            source_clock="test_clock",
            attributes=attributes,
        )
    client = SensorGatewayClient(
        "inproc://navdp-gateway-input",
        context=context,
        request_timeout_ms=100,
    )
    sensors = _SharedSensors()
    ingress = NavDPSensorGatewayIngress(
        "inproc://navdp-gateway-input",
        sensors,
        poll_hz=50.0,
        request_timeout_ms=100,
        client=client,
    )
    try:
        ingress.start()
        deadline = time.monotonic() + 1.0
        camera = None
        while time.monotonic() < deadline:
            camera = ingress.poll_camera() or camera
            with sensors.lock:
                ready = (
                    sensors.pose is not None
                    and len(sensors.points) == len(lidar)
                    and len(sensors.slam_map_xy) == len(cloud)
                )
            if camera is not None and ready:
                break
            time.sleep(0.01)

        assert camera is not None
        np.testing.assert_array_equal(camera.rgb, rgb)
        np.testing.assert_allclose(camera.depth_m, 0.75)
        assert sensors.pose is not None
        assert sensors.pose.x == pytest.approx(1.0)
        assert sensors.pose.y == pytest.approx(2.0)
        assert sensors.pose.yaw == pytest.approx(0.0)
        assert len(sensors.points) == len(lidar)
        np.testing.assert_array_equal(sensors.slam_map_xy, cloud[:, :2])
    finally:
        ingress.close()
        client.close()
        server.close()
        core.close()
        context.term()
