from __future__ import annotations

import sys
import types

import numpy as np
from pathlib import Path
from types import MappingProxyType


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.scripts.run_lingbot_depth_viewer import (
    LingBotInferenceModeGate,
    LingBotSensorGatewayClient,
    LingBotDepthViewerConfig,
    configure_lingbot_runtime_environment,
    conservative_depth_fusion,
    normalized_intrinsics,
    prepare_depth_meters,
    resolve_local_model_path,
    completed_depth_payload,
    mark_ready,
)
from gear_sonic.runtime.client import MaterializedSnapshot
from gear_sonic.runtime.contracts import MessageMetadata, SharedMemoryFrame
from gear_sonic.runtime.snapshot import SensorSnapshot, TimestampBasis
from gear_sonic.utils.inference.object_nav import (
    ComposedRGBDCamera,
    SensorGatewayRGBDCamera,
)


def test_lingbot_viewer_uses_fixed_ten_meter_range_by_default() -> None:
    assert LingBotDepthViewerConfig().max_depth_m == 10.0


def test_lingbot_reads_chest_rgbd_from_gateway_shared_memory() -> None:
    rgb = np.full((2, 3, 3), 17, dtype=np.uint8)
    depth = np.full((2, 3), 1234, dtype=np.uint16)
    received_ns = 5_000_000_000
    frames = {
        "camera/chest_view": SharedMemoryFrame(
            metadata=MessageMetadata(
                source="sensor_gateway", sequence=1, timestamp_ns=received_ns, ttl_ms=1000
            ),
            stream="camera/chest_view",
            shared_memory="fake-rgb",
            shape=rgb.shape,
            dtype=rgb.dtype.str,
            offset_bytes=8,
            size_bytes=rgb.nbytes,
            source_timestamp_ns=12_000_000_000,
            source_clock="camera_unix",
            attributes={"camera_info": {"fx": 500.0, "depth_scale_m": 0.001}},
        ),
        "camera/chest_view_depth": SharedMemoryFrame(
            metadata=MessageMetadata(
                source="sensor_gateway", sequence=1, timestamp_ns=received_ns, ttl_ms=1000
            ),
            stream="camera/chest_view_depth",
            shared_memory="fake-depth",
            shape=depth.shape,
            dtype=depth.dtype.str,
            offset_bytes=8,
            size_bytes=depth.nbytes,
            source_timestamp_ns=12_000_000_000,
            source_clock="camera_unix",
        ),
    }
    materialized = MaterializedSnapshot(
        snapshot=SensorSnapshot(
            complete=True,
            reason="",
            anchor_timestamp_ns=received_ns,
            timestamp_basis=TimestampBasis.RECEIVE,
            frames=MappingProxyType(frames),
            skew_ms=0.0,
            ages_ms=MappingProxyType({name: 0.0 for name in frames}),
        ),
        arrays=MappingProxyType(
            {"camera/chest_view": rgb, "camera/chest_view_depth": depth}
        ),
        attempts=1,
    )

    class FakeClient:
        def read_snapshot(self, *_args, **_kwargs):
            return materialized

    client = LingBotSensorGatewayClient("inproc://unused", client=FakeClient())
    packet = client.read()

    np.testing.assert_array_equal(packet["images"]["chest_view"], rgb)
    np.testing.assert_array_equal(packet["images"]["chest_view_depth"], depth)
    assert packet["timestamps"]["chest_view"] == 12.0
    assert packet["camera_info"]["chest_view"]["fx"] == 500.0


def test_lavira_reads_raw_rgb_and_lingbot_depth_from_one_source_aligned_snapshot() -> None:
    rgb = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
        dtype=np.uint8,
    )
    depth = np.full((2, 2), 1500, dtype=np.uint16)
    received_ns = 5_000_000_000
    camera_info = {
        "fx": 500.0,
        "fy": 501.0,
        "cx": 1.0,
        "cy": 1.0,
        "width": 2,
        "height": 2,
        "depth_scale_m": 0.001,
        "depth_aligned_to": "chest_view",
    }
    frames = {
        "camera/chest_view": SharedMemoryFrame(
            metadata=MessageMetadata(
                source="sensor_gateway", sequence=1, timestamp_ns=received_ns, ttl_ms=1000
            ),
            stream="camera/chest_view",
            shared_memory="fake-rgb",
            shape=rgb.shape,
            dtype=rgb.dtype.str,
            offset_bytes=8,
            size_bytes=rgb.nbytes,
            source_timestamp_ns=12_000_000_000,
            source_clock="camera_unix",
            attributes={"camera_info": camera_info},
        ),
        "derived/lingbot_depth": SharedMemoryFrame(
            metadata=MessageMetadata(
                source="sensor_gateway", sequence=1, timestamp_ns=received_ns, ttl_ms=1000
            ),
            stream="derived/lingbot_depth",
            shared_memory="fake-lingbot",
            shape=depth.shape,
            dtype=depth.dtype.str,
            offset_bytes=8,
            size_bytes=depth.nbytes,
            source_timestamp_ns=12_000_000_000,
            source_clock="camera_unix",
            attributes={"camera_info": camera_info},
        ),
    }
    materialized = MaterializedSnapshot(
        snapshot=SensorSnapshot(
            complete=True,
            reason="",
            anchor_timestamp_ns=12_000_000_000,
            timestamp_basis=TimestampBasis.SOURCE,
            frames=MappingProxyType(frames),
            skew_ms=0.0,
            ages_ms=MappingProxyType({name: 0.0 for name in frames}),
        ),
        arrays=MappingProxyType(
            {"camera/chest_view": rgb, "derived/lingbot_depth": depth}
        ),
        attempts=1,
    )

    class FakeClient:
        request = None

        def read_snapshot(self, request, **_kwargs):
            self.request = request
            return materialized

    fake = FakeClient()
    camera = SensorGatewayRGBDCamera(
        "inproc://unused",
        client=fake,
        max_age_ms=1000.0,
        max_skew_ms=5.0,
    )
    snapshot = camera.capture_aligned_rgbd()

    assert fake.request.timestamp_basis is TimestampBasis.SOURCE
    assert fake.request.streams == (
        "camera/chest_view",
        "derived/lingbot_depth",
    )
    np.testing.assert_array_equal(snapshot.rgb_bgr, rgb[..., ::-1])
    np.testing.assert_array_equal(snapshot.depth_raw, depth)
    np.testing.assert_allclose(snapshot.depth_mm, 1500.0)
    assert snapshot.timestamp == 12.0


def test_lingbot_gpu_work_is_gated_by_pose_and_planner_mode() -> None:
    gate = LingBotInferenceModeGate()

    assert gate.inference_enabled
    assert gate.accept("select_pose_mode")
    assert not gate.inference_enabled
    assert not gate.accept("unrelated_command")
    assert not gate.inference_enabled
    assert gate.accept("select_planner_mode")
    assert gate.inference_enabled


def test_prepare_depth_meters_marks_out_of_range_values_invalid() -> None:
    raw = np.array([[0, 500, 5000, 65535]], dtype=np.uint16)

    depth = prepare_depth_meters(raw, depth_scale_m=0.001, max_depth_m=5.0)

    np.testing.assert_allclose(depth, [[0.0, 0.5, 5.0, 0.0]])


def test_normalized_intrinsics_scales_rows_by_image_dimensions() -> None:
    matrix = normalized_intrinsics(
        {"fx": 600.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
        width=640,
        height=480,
    )

    np.testing.assert_allclose(
        matrix,
        [[600 / 640, 0.0, 0.5], [0.0, 500 / 480, 0.5], [0.0, 0.0, 1.0]],
    )


def test_conservative_fusion_keeps_nearest_valid_depth() -> None:
    raw = np.array([[0.0, 1.0, 3.0, 0.0]], dtype=np.float32)
    completed = np.array([[2.0, 2.0, 2.0, 0.0]], dtype=np.float32)

    fused = conservative_depth_fusion(raw, completed, max_depth_m=5.0)

    np.testing.assert_allclose(fused, [[2.0, 1.0, 2.0, 0.0]])


def test_model_resolution_never_contacts_huggingface_network() -> None:
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/cache/model.pt"

    result = resolve_local_model_path("org/model", download=fake_download)

    assert result == "/cache/model.pt"
    assert calls == [
        {
            "repo_id": "org/model",
            "repo_type": "model",
            "filename": "model.pt",
            "local_files_only": True,
        }
    ]


def test_lingbot_runtime_keeps_xformers_enabled(monkeypatch) -> None:
    monkeypatch.setenv("XFORMERS_DISABLED", "1")

    configure_lingbot_runtime_environment()

    assert "XFORMERS_DISABLED" not in __import__("os").environ


def test_mark_ready_replaces_stale_marker(tmp_path: Path) -> None:
    marker = tmp_path / "lingbot.ready"
    marker.write_text("stale")

    mark_ready(str(marker))

    assert marker.read_text() == "ready\n"


def test_completed_depth_payload_contains_only_lingbot_depth() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    completed_m = np.array(
        [[0.0, 1.25, 11.0], [2.5, np.inf, 0.1]], dtype=np.float32
    )
    info = {
        "fx": 500.0,
        "fy": 500.0,
        "cx": 1.0,
        "cy": 1.0,
        "width": 3,
        "height": 2,
        "depth_scale_m": 0.001,
        "depth_aligned_to": "chest_view",
    }

    payload = completed_depth_payload(
        rgb, completed_m, info, timestamp=12.5, max_depth_m=10.0
    )

    np.testing.assert_array_equal(
        payload.images["chest_view_depth"],
        [[0, 1250, 0], [2500, 0, 100]],
    )
    assert payload.timestamps == {"chest_view": 12.5, "chest_view_depth": 12.5}
    assert payload.camera_info["chest_view"]["depth_scale_m"] == 0.001

    decoded = ComposedRGBDCamera.decode_payload(payload.serialize())
    np.testing.assert_allclose(
        decoded.depth_mm,
        [[0.0, 1250.0, 0.0], [2500.0, 0.0, 100.0]],
    )
