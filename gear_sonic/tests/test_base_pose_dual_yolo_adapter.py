from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.runtime.client import SensorGatewayClientError
from gear_sonic.runtime.snapshot import TimestampBasis
from gear_sonic.scripts.base_pose_agent import BasePoseAgentConfig
from gear_sonic.scripts.base_pose_yolo_agent import (
    GatewayRawServoAdapter,
    raw_servo_worker_for_mode,
)
from gear_sonic.utils.inference.base_pose import BasePoseCameraError
from gear_sonic.utils.inference.base_pose_dual_visual_servo import (
    DualCameraFailoverCoordinator,
    DualCameraReference,
    HeadCameraTextMonitor,
    dual_calibrations_from_config,
    run_dual_raw_servo_worker,
)
from gear_sonic.utils.inference.base_pose_sensor import (
    DualBasePoseCapture,
    SensorGatewayDualBasePoseCamera,
)
from gear_sonic.utils.inference.base_pose_visual_servo import (
    GenerationGate,
    RawServoCalibration,
    RawServoEvent,
    ServoPhase,
    TrackedInstance,
    run_raw_servo_worker,
)


HEAD = "ego_view"
CHEST = "chest_view"


def _reference(stream_name: str, marker: int) -> DualCameraReference:
    return DualCameraReference(
        stream_name=stream_name,
        rgb=np.full((4, 6, 3), marker, dtype=np.uint8),
        target_prompt="blue basket",
        target_bbox=(100.0, 100.0, 400.0, 600.0),
        table_bboxes=((50.0, 300.0, 950.0, 900.0),),
        camera_timestamp=float(marker),
    )


def test_dual_mode_is_the_agent_near_aligned_default() -> None:
    config = BasePoseAgentConfig(task="align")

    assert config.mode == "dual_raw_yoloe_servo"
    assert not hasattr(config, "vision_backend")
    assert not hasattr(config, "model")
    assert not hasattr(config, "reasoning_effort")
    assert not hasattr(config, "codex_fast")
    assert not hasattr(config, "codex_timeout_seconds")
    assert config.qwenvl_model == "qwen3-vl-plus"
    assert config.qwenvl_timeout_seconds == pytest.approx(600.0)
    assert config.dual_head_camera_stream == HEAD
    assert config.dual_head_depth_stream == "camera/ego_view_depth"
    assert config.dual_chest_camera_stream == CHEST
    assert (
        config.dual_chest_depth_stream
        == "derived/depth_anything/chest_view"
    )
    assert config.dual_chest_camera_pitch_deg == pytest.approx(-3.0)
    assert config.dual_match_tolerance_frames == 30
    assert config.dual_head_reacquire_frames == 10
    assert config.dual_initialization_grace_s == pytest.approx(30.0)
    assert config.dual_rgbd_buffer_size == 8
    assert config.dual_rgbd_poll_hz == pytest.approx(60.0)


def test_dual_worker_selection_keeps_single_mode_available() -> None:
    assert (
        raw_servo_worker_for_mode("dual_raw_yoloe_servo")
        is run_dual_raw_servo_worker
    )
    assert raw_servo_worker_for_mode("raw_yoloe_servo") is run_raw_servo_worker
    with pytest.raises(ValueError, match="unsupported"):
        raw_servo_worker_for_mode("rgb")


def test_dual_failover_matches_agent_near_camera_order() -> None:
    head = _reference(HEAD, 1)
    chest = _reference(CHEST, 2)
    coordinator = DualCameraFailoverCoordinator(
        (HEAD, CHEST),
        {HEAD: head, CHEST: chest},
    )

    initial = coordinator.start()
    assert initial.live_stream == HEAD
    coordinator.mark_success(initial)

    origin_text = coordinator.advance_after_failure(initial)
    assert origin_text is not None
    assert origin_text.live_stream == HEAD
    assert origin_text.stage == "origin_text"

    origin_qwen = coordinator.advance_after_failure(origin_text)
    assert origin_qwen is not None
    assert origin_qwen.live_stream == HEAD
    assert origin_qwen.stage == "origin_qwen"

    alternate_text = coordinator.advance_after_failure(origin_qwen)
    assert alternate_text is not None
    assert alternate_text.live_stream == CHEST
    assert alternate_text.reference is chest
    assert alternate_text.stage == "alternate_text"

    alternate_qwen = coordinator.advance_after_failure(alternate_text)
    assert alternate_qwen is not None
    assert alternate_qwen.live_stream == CHEST
    assert alternate_qwen.stage == "alternate_qwen"
    assert coordinator.advance_after_failure(alternate_qwen) is None


def test_head_monitor_reuses_initialized_dynamic_target_prompt() -> None:
    target = TrackedInstance(
        1,
        0,
        "red tote returned by qwen",
        0.9,
        (1.0, 0.0, 4.0, 3.0),
        np.ones((4, 6), dtype=np.uint8),
    )
    desk = TrackedInstance(
        2,
        1,
        "desk",
        0.8,
        (0.0, 1.0, 6.0, 4.0),
        np.ones((4, 6), dtype=np.uint8),
    )

    class Tracker:
        def __init__(self) -> None:
            self.prompts: list[tuple[str, str]] = []

        def start_all_text(self, *, target_prompt: str, surface_prompt: str):
            self.prompts.append((target_prompt, surface_prompt))

        def track(self, _rgb):
            return [target, desk]

    tracker = Tracker()
    monitor = HeadCameraTextMonitor(
        tracker,
        "red tote returned by qwen",
        required_target_frames=2,
    )
    monitor.start()
    first = monitor.inspect(_worker_snapshot(HEAD, 1))
    second = monitor.inspect(_worker_snapshot(HEAD, 2))

    assert tracker.prompts == [("red tote returned by qwen", "desk")]
    assert not first.triggered
    assert second.triggered
    assert second.reference is not None
    assert second.reference.kind == "head_monitor"
    assert second.reference.target_prompt == "red tote returned by qwen"


def test_dual_calibration_uses_head_and_chest_mounts() -> None:
    config = BasePoseAgentConfig(task="align")

    calibrations = dual_calibrations_from_config(config)

    assert set(calibrations) == {HEAD, CHEST}
    assert calibrations[HEAD].camera_pitch_deg == pytest.approx(-38.0)
    assert calibrations[CHEST].camera_pitch_deg == pytest.approx(-3.0)
    assert calibrations[HEAD].width == 640
    assert calibrations[CHEST].width == 640


class _FakeDualGatewayClient:
    def __init__(self, stream_name: str) -> None:
        self.stream_name = stream_name
        self.timestamp_ns: int | None = None
        self.generation = 1
        self.lock = threading.Lock()
        self.closed = False

    def publish(self, timestamp_ns: int, *, generation: int = 1) -> None:
        with self.lock:
            self.timestamp_ns = timestamp_ns
            self.generation = generation

    @staticmethod
    def _materialized(
        stream_name: str,
        depth_stream: str,
        timestamp_ns: int,
        generation: int,
    ):
        rgb_stream = f"camera/{stream_name}"
        camera_info = {
            "fx": 100.0,
            "fy": 101.0,
            "cx": 2.5,
            "cy": 1.5,
            "width": 6,
            "height": 4,
            "depth_scale_m": 0.001,
            "depth_aligned_to": stream_name,
        }
        if depth_stream.startswith("derived/depth_anything/"):
            camera_info.update(
                {
                    "depth_source": "depth-anything-v2-metric-hypersim-vitb",
                    "uses_raw_depth": False,
                    "inference_owner": "base_pose",
                    "inference_generation": generation,
                }
            )
        frame = lambda: SimpleNamespace(
            source_timestamp_ns=timestamp_ns,
            attributes={
                "camera_info": camera_info,
                "depth_source": camera_info.get("depth_source", ""),
            },
        )
        return SimpleNamespace(
            snapshot=SimpleNamespace(
                frames={rgb_stream: frame(), depth_stream: frame()}
            ),
            arrays={
                rgb_stream: np.zeros((4, 6, 3), dtype=np.uint8),
                depth_stream: np.full((4, 6), 1000, dtype=np.uint16),
            },
        )

    def read_snapshot(self, request, *, retries: int):
        assert retries == 0
        assert request.timestamp_basis is TimestampBasis.SOURCE
        rgb_stream = request.streams[0]
        stream_name = rgb_stream.removeprefix("camera/")
        assert stream_name == self.stream_name
        with self.lock:
            timestamp_ns = self.timestamp_ns
            generation = self.generation
        if timestamp_ns is None:
            raise SensorGatewayClientError(f"{stream_name} unavailable")
        return self._materialized(
            stream_name,
            request.streams[1],
            timestamp_ns,
            generation,
        )

    def close(self) -> None:
        self.closed = True


def test_dual_gateway_does_not_let_one_camera_block_the_other() -> None:
    clients: dict[str, _FakeDualGatewayClient] = {}

    def client_factory(stream_name: str) -> _FakeDualGatewayClient:
        client = _FakeDualGatewayClient(stream_name)
        clients[stream_name] = client
        return client

    camera = SensorGatewayDualBasePoseCamera(
        "inproc://unused",
        stream_depths={
            HEAD: "camera/ego_view_depth",
            CHEST: "derived/depth_anything/chest_view",
        },
        timeout_ms=60,
        poll_hz=200.0,
        client_factory=client_factory,
    )
    clients[HEAD].publish(1_000_000_000)

    head_only = camera.capture()
    assert set(head_only.snapshots) == {HEAD}
    assert "chest_view unavailable" in head_only.errors[CHEST]
    assert head_only.require(HEAD).depth_aligned_to == HEAD

    # A ready chest frame must remain available after an unrelated head read.
    clients[CHEST].publish(2_000_000_000)
    clients[HEAD].publish(3_000_000_000)
    head = camera.capture_stream(HEAD)
    chest = camera.capture_stream(CHEST)
    assert head.timestamp == pytest.approx(3.0)
    assert chest.timestamp == pytest.approx(2.0)
    assert chest.depth_aligned_to == CHEST

    camera.close()
    assert all(client.closed for client in clients.values())


def test_dual_gateway_rejects_depth_from_an_old_base_pose_generation() -> None:
    clients: dict[str, _FakeDualGatewayClient] = {}

    def client_factory(stream_name: str) -> _FakeDualGatewayClient:
        client = _FakeDualGatewayClient(stream_name)
        clients[stream_name] = client
        return client

    camera = SensorGatewayDualBasePoseCamera(
        "inproc://unused",
        stream_depths={
            HEAD: "camera/ego_view_depth",
            CHEST: "derived/depth_anything/chest_view",
        },
        timeout_ms=60,
        poll_hz=200.0,
        client_factory=client_factory,
    )
    try:
        camera.begin_generation(7)
        clients[CHEST].publish(2_000_000_000, generation=6)
        with pytest.raises(BasePoseCameraError, match="generation 6"):
            camera.capture_stream(CHEST)

        clients[CHEST].publish(3_000_000_000, generation=7)
        chest = camera.capture_stream(CHEST)
        assert chest.timestamp == pytest.approx(3.0)
    finally:
        camera.close()


def test_dual_runtime_uses_camera_specific_standoff_and_initial_phase(tmp_path) -> None:
    config = BasePoseAgentConfig(task="align", output_root=str(tmp_path))
    adapter = GatewayRawServoAdapter(
        config,
        submit_intent=lambda _name, _values: None,
    )

    adapter.runtime.active_camera_stream = HEAD
    adapter.runtime._reset_controller(1.0)
    assert adapter.runtime.controller.target_distance_m == pytest.approx(0.9)
    assert adapter.runtime.controller.phase is ServoPhase.YAW_ALIGN

    adapter.runtime.active_camera_stream = CHEST
    adapter.runtime._reset_controller(2.0)
    assert adapter.runtime.controller.target_distance_m == pytest.approx(0.8)
    assert adapter.runtime.controller.phase is ServoPhase.FORWARD_APPROACH

    assert adapter.runtime._is_vertical_recenter_switch(
        previous_stream=CHEST,
        next_stream=HEAD,
        details={"failover_stage": "head_monitor"},
    )
    adapter.runtime.vertical_recenter_armed = True
    adapter.runtime.active_camera_stream = HEAD
    adapter.runtime._reset_controller(3.0)
    assert adapter.runtime.controller.phase is ServoPhase.VERTICAL_RECENTER


def _worker_snapshot(stream_name: str, marker: int) -> object:
    from gear_sonic.utils.inference.base_pose import AlignedRGBDSnapshot

    return AlignedRGBDSnapshot(
        rgb=np.full((4, 6, 3), marker, dtype=np.uint8),
        depth_raw=np.full((4, 6), 1000, dtype=np.uint16),
        fx=100.0,
        fy=101.0,
        cx=2.5,
        cy=1.5,
        depth_scale_m=0.001,
        depth_aligned_to=stream_name,
        depth_source=None,
        timestamp=float(marker),
    )


class _FakeWorkerCamera:
    def __init__(self) -> None:
        self.count = 0
        self.closed = False

    def capture(self) -> DualBasePoseCapture:
        self.count += 1
        return DualBasePoseCapture(
            snapshots={
                HEAD: _worker_snapshot(HEAD, self.count),
                CHEST: _worker_snapshot(CHEST, self.count),
            },
            errors={},
        )

    def close(self) -> None:
        self.closed = True


def _qwen_reference(
    _config,
    stream_name,
    snapshot,
    _calibration,
    _output_dir,
) -> DualCameraReference:
    return DualCameraReference(
        stream_name=stream_name,
        rgb=snapshot.rgb,
        target_prompt="blue basket",
        target_bbox=(100.0, 100.0, 400.0, 600.0),
        table_bboxes=((50.0, 300.0, 950.0, 900.0),),
        camera_timestamp=snapshot.timestamp,
        kind="qwen",
    )


def test_dual_worker_exhausts_the_agent_near_failover_sequence(tmp_path: Path) -> None:
    class MissingTracker:
        def start(self, _rgb, **_kwargs):
            return {}

        def start_text(self, _rgb, **_kwargs):
            return {}

        def start_all_text(self, **_kwargs):
            return {}

        def track(self, _rgb):
            return []

    config = BasePoseAgentConfig(
        task="align",
        output_root=str(tmp_path),
        dual_match_tolerance_frames=2,
        raw_servo_hz=10000.0,
        raw_reference_update_interval_frames=5,
    )
    requests: queue.Queue[int | None] = queue.Queue()
    requests.put(1)
    requests.put(None)
    events: queue.Queue[RawServoEvent] = queue.Queue()
    gate = GenerationGate()
    gate.activate(1)
    camera = _FakeWorkerCamera()
    calibrations = {
        HEAD: RawServoCalibration(6, 4, 100.0, 101.0, 2.5, 1.5),
        CHEST: RawServoCalibration(
            6,
            4,
            100.0,
            101.0,
            2.5,
            1.5,
            camera_pitch_deg=-3.0,
        ),
    }

    run_dual_raw_servo_worker(
        config,
        requests,
        events,
        gate,
        threading.Event(),
        camera_factory=lambda: camera,
        tracker_factory=MissingTracker,
        qwen_reference_factory=_qwen_reference,
        calibration_factory=lambda _config: calibrations,
        reference_factory=lambda *_args, **_kwargs: (
            {HEAD: _reference(HEAD, 1), CHEST: _reference(CHEST, 2)},
            {},
        ),
    )

    emitted: list[RawServoEvent] = []
    while not events.empty():
        emitted.append(events.get_nowait())
    switching = [event for event in emitted if event.kind == "switching"]
    terminal = [event for event in emitted if event.kind == "error"]
    assert switching, [(event.kind, event.error) for event in emitted]
    assert [event.details["live_stream"] for event in switching] == [
        HEAD,
        HEAD,
        CHEST,
        CHEST,
    ]
    assert len(terminal) == 1
    assert "both-camera Qwen failover exhausted" in (terminal[0].error or "")
    summary_path = next(tmp_path.glob("dual_raw_yoloe_*")) / (
        "initial_reference_summary.json"
    )
    assert json.loads(summary_path.read_text())["selected_initial_stream"] == HEAD
    assert camera.closed
