from __future__ import annotations

from pathlib import Path
import sys
import types
from types import MappingProxyType
from unittest.mock import Mock

import numpy as np


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.runtime.gateway.sensor_client import (
    MaterializedSnapshot,
    SensorGatewayClientError,
)
from gear_sonic.runtime.protocol import MessageMetadata, SharedMemoryFrame
from gear_sonic.runtime.gateway.snapshot import SensorSnapshot, TimestampBasis
from gear_sonic.utils.inference.lavira.depth_service import (
    CAMERA_NAME,
    DEPTH_SOURCE,
    RGB_STREAM,
    DepthAnythingConfig,
    load_depth_anything_config,
    DepthAnythingInferenceGate,
    DepthAnythingSensorGatewayClient,
    depth_anything_status_payload,
    mark_ready,
    metric_depth_payload,
)
from gear_sonic.utils.inference.lavira.object_nav import SensorGatewayRGBDCamera
from gear_sonic.utils.inference.lavira import depth_service


def _materialized(
    arrays: dict[str, np.ndarray],
    attributes: dict[str, dict] | None = None,
) -> MaterializedSnapshot:
    source_timestamp_ns = 12_000_000_000
    received_ns = 5_000_000_000
    attributes = attributes or {}
    frames = {
        stream: SharedMemoryFrame(
            metadata=MessageMetadata(
                source="sensor_gateway",
                sequence=1,
                timestamp_ns=received_ns,
                ttl_ms=1000,
            ),
            stream=stream,
            shared_memory=f"fake-{index}",
            shape=array.shape,
            dtype=array.dtype.str,
            offset_bytes=8,
            size_bytes=array.nbytes,
            source_timestamp_ns=source_timestamp_ns,
            source_clock="camera_unix",
            attributes=attributes.get(stream, {}),
        )
        for index, (stream, array) in enumerate(arrays.items())
    }
    return MaterializedSnapshot(
        snapshot=SensorSnapshot(
            complete=True,
            reason="",
            anchor_timestamp_ns=source_timestamp_ns,
            timestamp_basis=TimestampBasis.SOURCE,
            frames=MappingProxyType(frames),
            skew_ms=0.0,
            ages_ms=MappingProxyType({name: 0.0 for name in frames}),
        ),
        arrays=MappingProxyType(arrays),
        attempts=1,
    )


def test_depth_anything_base_defaults_target_ten_hz() -> None:
    config = load_depth_anything_config()

    assert config.encoder == "vitb"
    assert config.inference_hz == 10.5
    assert config.input_size == 644
    assert config.model_max_depth_m == 20.0
    assert config.use_amp


def test_depth_anything_gate_only_runs_for_lavira_capture() -> None:
    gate = DepthAnythingInferenceGate()
    assert not gate.active

    assert gate.apply("start_navigation", {"generation": 1})
    assert gate.active and gate.owner == "lavira"
    assert gate.apply("lavira_rgbd_captured", {"generation": 1})
    assert not gate.active

    assert not gate.apply("start_base_pose", {"generation": 2})
    assert not gate.active


def test_depth_anything_status_reports_resident_idle_state() -> None:
    gate = DepthAnythingInferenceGate()
    payload = depth_anything_status_payload(gate)

    assert payload["type"] == "sonic.depth_anything_status"
    assert payload["active"] is False
    assert payload["owner"] == "idle"


def test_depth_anything_reads_only_chest_rgb_from_gateway() -> None:
    rgb = np.full((2, 3, 3), 17, dtype=np.uint8)
    materialized = _materialized(
        {RGB_STREAM: rgb},
        {RGB_STREAM: {"camera_info": {"fx": 500.0}}},
    )

    class FakeClient:
        request = None

        def read_snapshot(self, request, **_kwargs):
            self.request = request
            return materialized

    fake = FakeClient()
    client = DepthAnythingSensorGatewayClient("inproc://unused", client=fake)
    packet = client.read()

    assert fake.request.streams == ("camera/chest_view",)
    assert all(not stream.endswith("_depth") for stream in fake.request.streams)
    assert fake.request.timestamp_basis is TimestampBasis.SOURCE
    assert packet is not None
    np.testing.assert_array_equal(packet[0], rgb)
    assert packet[1] == 12.0
    assert packet[2]["fx"] == 500.0


def test_depth_anything_logs_camera_failure_and_recovery_once(
    monkeypatch,
) -> None:
    rgb = np.full((2, 3, 3), 17, dtype=np.uint8)
    materialized = _materialized({RGB_STREAM: rgb})

    class FakeClient:
        responses = iter(
            (
                SensorGatewayClientError("not ready"),
                SensorGatewayClientError("still unavailable"),
                materialized,
            )
        )

        def read_snapshot(self, _request, **_kwargs):
            response = next(self.responses)
            if isinstance(response, Exception):
                raise response
            return response

    logger = Mock()
    monkeypatch.setattr(depth_service, "LOGGER", logger)
    client = DepthAnythingSensorGatewayClient(
        "inproc://unused", client=FakeClient()
    )

    assert client.read() is None
    assert client.read() is None
    assert client.read() is not None

    logger.warning.assert_called_once_with(
        "waiting for chest RGB: %s", "not ready"
    )
    logger.info.assert_called_once_with("chest RGB recovered")


def test_metric_payload_is_direct_uint16_millimetres_with_metre_scale() -> None:
    depth_m = np.array(
        [[0.0, 1.25, 21.0], [2.5, np.inf, 0.1]], dtype=np.float32
    )
    info = {
        "fx": 500.0,
        "fy": 500.0,
        "cx": 1.0,
        "cy": 1.0,
        "width": 3,
        "height": 2,
    }

    payload = metric_depth_payload(
        depth_m,
        info,
        timestamp=12.5,
        publish_max_depth_m=20.0,
        inference_owner="base_pose",
        inference_generation=7,
    )

    np.testing.assert_array_equal(
        payload.images["chest_view_depth"],
        [[0, 1250, 0], [2500, 0, 100]],
    )
    assert payload.timestamps == {"chest_view_depth": 12.5}
    camera_info = payload.camera_info[CAMERA_NAME]
    assert camera_info["depth_scale_m"] == 0.001
    assert camera_info["depth_source"] == DEPTH_SOURCE
    assert camera_info["metric_model_output"] is True
    assert camera_info["uses_raw_depth"] is False
    assert camera_info["inference_owner"] == "base_pose"
    assert camera_info["inference_generation"] == 7


def test_lavira_reads_depth_anything_chest_depth_in_metric_units() -> None:
    rgb = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
        dtype=np.uint8,
    )
    depth = np.full((2, 2), 1500, dtype=np.uint16)
    info = {
        "fx": 500.0,
        "fy": 501.0,
        "cx": 1.0,
        "cy": 1.0,
        "width": 2,
        "height": 2,
        "depth_scale_m": 0.001,
        "depth_aligned_to": "chest_view",
        "depth_source": DEPTH_SOURCE,
        "inference_owner": "lavira",
    }
    depth_stream = "derived/depth_anything/chest_view"
    materialized = _materialized(
        {"camera/chest_view": rgb, depth_stream: depth},
        {
            "camera/chest_view": {"camera_info": info},
            depth_stream: {"camera_info": info, "depth_source": DEPTH_SOURCE},
        },
    )

    class FakeClient:
        request = None

        def read_snapshot(self, request, **_kwargs):
            self.request = request
            return materialized

    fake = FakeClient()
    camera = SensorGatewayRGBDCamera("inproc://unused", client=fake)
    snapshot = camera.capture_aligned_rgbd()

    assert fake.request.streams == ("camera/chest_view", depth_stream)
    np.testing.assert_array_equal(snapshot.rgb_bgr, rgb[..., ::-1])
    np.testing.assert_allclose(snapshot.depth_mm, 1500.0)
    assert snapshot.fx == 500.0
    assert snapshot.cx == 1.0


def test_mark_ready_replaces_stale_marker(tmp_path: Path) -> None:
    marker = tmp_path / "depth-anything.ready"
    marker.write_text("stale")

    mark_ready(str(marker))

    assert marker.read_text() == "ready\n"
