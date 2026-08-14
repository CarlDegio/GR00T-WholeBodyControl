"""Compatibility tests for the RGB-only camera viewer."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.runtime.client import SensorGatewayClientError
import gear_sonic.scripts.launch_data_collection as data_collection_launcher
from gear_sonic.scripts.run_camera_viewer import (
    GatewayCameraClient,
    _gateway_rgb_streams,
    _rgb_camera_names,
    _wait_for_first_camera_frame,
)
from gear_sonic.scripts.launch_data_collection import (
    DataCollectionLaunchConfig,
    build_camera_viewer_command,
    build_pico_video_command,
)
from gear_sonic.scripts.run_depth_camera_viewer import colorize_depth


def _gateway_health(*streams: str) -> dict:
    return {
        "type": "sonic.sensor_gateway_health",
        "version": 1,
        "timestamp_ns": 123,
        "streams": {stream: {"state": "healthy"} for stream in streams},
        "shared_memory": {},
        "retired_ring_count": 0,
    }


class _FakeGatewayClient:
    def __init__(
        self,
        *,
        health: dict | None = None,
        arrays: dict[str, np.ndarray] | None = None,
        health_error: Exception | None = None,
        snapshot_error: Exception | None = None,
    ) -> None:
        self.health_payload = health or _gateway_health()
        self.arrays = arrays or {}
        self.health_error = health_error
        self.snapshot_error = snapshot_error
        self.requests = []
        self.closed = False

    def health(self) -> dict:
        if self.health_error is not None:
            raise self.health_error
        return self.health_payload

    def read_snapshot(self, request, *, retries: int = 2):
        self.requests.append((request, retries))
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return SimpleNamespace(arrays=MappingProxyType(self.arrays))

    def close(self) -> None:
        self.closed = True


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


class _SlowUnavailableCamera:
    def __init__(self, clock: _FakeClock, rpc_duration_s: float) -> None:
        self.clock = clock
        self.rpc_duration_s = rpc_duration_s
        self.read_count = 0

    def read(self, blocking: bool = False):
        assert blocking is False
        self.read_count += 1
        self.clock.now += self.rpc_duration_s
        return None


def test_first_frame_wait_uses_wall_clock_deadline_including_rpc_time() -> None:
    clock = _FakeClock()
    camera = _SlowUnavailableCamera(clock, rpc_duration_s=0.1)

    sample = _wait_for_first_camera_frame(
        camera,
        timeout_s=10.0,
        poll_interval_s=0.1,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert sample is None
    assert abs(clock.now - 10.0) <= 1.0e-9
    assert camera.read_count == 50


def test_gateway_rgb_streams_are_sorted_and_exclude_depth_and_non_camera() -> None:
    health = _gateway_health(
        "camera/right_wrist",
        "camera/ego_view_depth",
        "source/camera_server",
        "camera/ego_view",
        "cpp/state_msgpack",
    )

    assert _gateway_rgb_streams(health) == (
        "camera/ego_view",
        "camera/right_wrist",
    )


def test_gateway_camera_client_materializes_only_rgb_images_by_camera_name() -> None:
    ego_rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    fake = _FakeGatewayClient(
        health=_gateway_health(
            "camera/right_wrist",
            "camera/ego_view",
        ),
        arrays={
            "camera/ego_view": ego_rgb,
            "camera/right_wrist": np.zeros((2, 3, 3), dtype=np.float32),
        },
    )
    camera = GatewayCameraClient("inproc://unused", client=fake)

    message = camera.read()

    assert message is not None
    assert tuple(message["images"]) == ("ego_view",)
    np.testing.assert_array_equal(message["images"]["ego_view"], ego_rgb)
    request, retries = fake.requests[0]
    assert request.streams == ("camera/ego_view", "camera/right_wrist")
    assert request.max_age_ms == 1500.0
    assert request.max_skew_ms == 5.0
    assert retries == 0


def test_gateway_camera_client_treats_gateway_errors_as_missing_frames() -> None:
    health_failure = GatewayCameraClient(
        "inproc://unused",
        client=_FakeGatewayClient(
            health_error=SensorGatewayClientError("health unavailable")
        ),
    )
    snapshot_failure = GatewayCameraClient(
        "inproc://unused",
        client=_FakeGatewayClient(
            health=_gateway_health("camera/ego_view"),
            snapshot_error=SensorGatewayClientError("snapshot unavailable"),
        ),
    )

    assert health_failure.read() is None
    assert snapshot_failure.read() is None


def test_data_collection_launcher_viewer_uses_profile_without_direct_camera() -> None:
    command = build_camera_viewer_command(
        DataCollectionLaunchConfig(runtime_profile="/tmp/runtime profile.yaml"),
        Path("/workspace/sonic"),
    )

    assert command == (
        "cd /workspace/sonic && "
        "source .venv_data_collection/bin/activate && "
        "python gear_sonic/scripts/run_camera_viewer.py "
        "--profile '/tmp/runtime profile.yaml'"
    )


def test_data_collection_launcher_keeps_pico_video_running_by_default() -> None:
    config = DataCollectionLaunchConfig(runtime_profile="/tmp/runtime profile.yaml")

    command = build_pico_video_command(config, Path("/workspace/sonic"))

    assert config.pico_video is True
    assert command == (
        "cd /workspace/sonic && "
        "source .venv_teleop/bin/activate && "
        "python -m gear_sonic.scripts.run_pico_video_bridge "
        "--profile '/tmp/runtime profile.yaml' "
        "--encoder h264_nvenc --stay-alive"
    )


def test_data_collection_launcher_starts_pico_video_in_gateway_pane(
    monkeypatch,
) -> None:
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(data_collection_launcher, "_check_prerequisites", lambda **_: None)
    monkeypatch.setattr(data_collection_launcher, "_kill_existing_session", lambda: None)
    monkeypatch.setattr(data_collection_launcher, "_create_tmux_session", lambda: None)
    monkeypatch.setattr(data_collection_launcher, "_check_pane_alive", lambda _pane: True)
    monkeypatch.setattr(data_collection_launcher, "_send_to_pane", lambda *_a, **_k: None)
    monkeypatch.setattr(data_collection_launcher, "_get_local_ip", lambda: "unknown")
    monkeypatch.setattr(data_collection_launcher.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(data_collection_launcher.subprocess, "run", run)

    data_collection_launcher.main(
        DataCollectionLaunchConfig(runtime_profile="/tmp/runtime profile.yaml")
    )
    repo_root = Path(data_collection_launcher.__file__).resolve().parent.parent.parent
    pico_command = build_pico_video_command(
        DataCollectionLaunchConfig(runtime_profile="/tmp/runtime profile.yaml"),
        repo_root,
    )

    assert [
        "tmux",
        "split-window",
        "-v",
        "-t",
        "sonic_data_collection:gateways.1",
    ] in commands
    assert [
        "tmux",
        "send-keys",
        "-t",
        "sonic_data_collection:gateways.2",
        pico_command,
        "C-m",
    ] in commands


def test_rgb_viewer_ignores_depth_from_schema_v2_message() -> None:
    message = ImageMessageSchema(
        timestamps={"chest_view": 1.0, "chest_view_depth": 1.0},
        images={
            "chest_view": np.zeros((2, 3, 3), dtype=np.uint8),
            "chest_view_depth": np.full((2, 3), 1000, dtype=np.uint16),
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
    decoded = ImageMessageSchema.deserialize(message.serialize())

    assert _rgb_camera_names(decoded.images) == ["chest_view"]


def test_colorize_depth_uses_fixed_range_and_marks_invalid_pixels() -> None:
    depth_mm = np.array([[0, 1000, 5000, 6000]], dtype=np.uint16)

    color, stats = colorize_depth(depth_mm, max_depth_m=5.0)

    assert color.shape == (1, 4, 3)
    assert color.dtype == np.uint8
    assert np.array_equal(color[0, 0], np.zeros(3, dtype=np.uint8))
    assert np.array_equal(color[0, 3], np.zeros(3, dtype=np.uint8))
    assert stats.valid_ratio == 0.5
    assert stats.min_depth_m == 1.0
    assert stats.median_depth_m == 3.0
