from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.runtime.gateway.sensor_client import SensorGatewayClientError
from gear_sonic.runtime.gateway.snapshot import TimestampBasis
from gear_sonic.utils.inference.base_pose.agent import (
    BasePoseAgentConfig,
    GatewayRawServoAdapter,
    _dual_worker_kwargs,
)
import gear_sonic.utils.inference.base_pose.dual_servo as dual_servo
from gear_sonic.utils.inference.base_pose.dual_servo import (
    DualCameraFailoverCoordinator,
    HeadCameraTextMonitor,
    _claim_new_stream_snapshot,
    detect_dual_camera_eligibility,
    dual_calibrations_from_config,
    run_dual_raw_servo_worker,
)
from gear_sonic.utils.inference.base_pose.sensor import (
    BasePoseCameraError,
    DualBasePoseCapture,
    SensorGatewayDualBasePoseCamera,
)
import gear_sonic.utils.inference.base_pose.servo as visual_servo
from gear_sonic.utils.inference.base_pose.servo import (
    HEAD_MONITOR_HOLD_EVENT,
    GenerationGate,
    RawServoCalibration,
    RawServoEvent,
    RawServoObservation,
    RawServoRuntime,
    ServoPhase,
    YawAlignGeometry,
    TargetGeometry,
    TrackedInstance,
)
from gear_sonic.utils.inference.base_pose.diagnostics import (
    DetectionFrameData,
)

HEAD = "ego_view"
CHEST = "chest_view"


def test_dual_perception_claims_only_strictly_newer_frames_per_stream() -> None:
    claimed: dict[str, float] = {}

    assert _claim_new_stream_snapshot(
        claimed, CHEST, SimpleNamespace(timestamp=1.0)
    )
    assert not _claim_new_stream_snapshot(
        claimed, CHEST, SimpleNamespace(timestamp=1.0)
    )
    assert not _claim_new_stream_snapshot(
        claimed, CHEST, SimpleNamespace(timestamp=0.5)
    )
    assert _claim_new_stream_snapshot(
        claimed, CHEST, SimpleNamespace(timestamp=2.0)
    )
    assert _claim_new_stream_snapshot(
        claimed, HEAD, SimpleNamespace(timestamp=1.0)
    )


def test_base_pose_config_exposes_only_dual_camera_mode() -> None:
    config = BasePoseAgentConfig(task="align")

    assert not hasattr(config, "mode")
    assert not hasattr(config, "camera_stream")
    assert not hasattr(config, "depth_stream")
    assert not hasattr(config, "vision_backend")
    assert not hasattr(config, "model")
    assert not hasattr(config, "reasoning_effort")
    assert not hasattr(config, "codex_fast")
    assert not hasattr(config, "codex_timeout_seconds")
    assert not hasattr(config, "qwenvl_model")
    assert not hasattr(config, "dual_qwenvl_fallback_model")
    assert config.target_prompt == "bluebasket"
    assert config.yaw_align_target_prompt == "cardboard box"
    assert config.dual_head_camera_stream == HEAD
    assert config.dual_head_depth_stream == "camera/ego_view_depth"
    assert config.dual_chest_camera_stream == CHEST
    assert config.dual_chest_depth_stream == "camera/chest_view_depth"
    assert config.dual_chest_camera_pitch_deg == pytest.approx(-3.0)
    assert config.dual_match_tolerance_frames == 30
    assert config.dual_head_reacquire_frames == 1
    assert config.dual_head_release_missing_frames == 3
    assert config.dual_rgbd_buffer_size == 8
    assert config.dual_rgbd_poll_hz == pytest.approx(60.0)
    assert not hasattr(config, "raw_chest_handoff_distance_m")
    assert config.raw_head_target_distance_m == pytest.approx(1.0)
    assert config.raw_head_approach_cutoff_m == pytest.approx(1.3)
    assert config.raw_chest_target_distance_m == pytest.approx(0.8)
    assert config.raw_chest_approach_cutoff_m == pytest.approx(1.3)
    assert config.raw_forward_tolerance_m == pytest.approx(0.10)
    assert config.raw_lateral_tolerance_m == pytest.approx(0.10)
    assert config.raw_post_stop_sample_frames == 30
    assert config.raw_post_stop_deviation_frames == 10
    assert config.raw_diagnostic_image_interval_frames == 5
    assert config.raw_min_linear_speed_m_s == pytest.approx(0.4)
    assert config.raw_max_lateral_speed_m_s == pytest.approx(0.4)


def test_base_pose_service_worker_callbacks_match_worker_signature() -> None:
    controller = SimpleNamespace(
        yaw_alignment_required=True,
        position_fallback_allowed=False,
    )
    runtime = SimpleNamespace(
        observation_events=object(),
        diagnostics=object(),
        controller=controller,
    )
    camera = object()
    worker_kwargs = _dual_worker_kwargs(
        SimpleNamespace(runtime=runtime), camera,
    )

    inspect.signature(run_dual_raw_servo_worker).bind(
        None, None, None, None, None, **worker_kwargs,
    )
    assert worker_kwargs["camera_factory"]() is camera
    assert worker_kwargs["yaw_alignment_required"]() is True
    assert worker_kwargs["position_fallback_allowed"]() is False
    assert "table_required" not in worker_kwargs


def test_dual_failover_matches_agent_near_camera_order() -> None:
    coordinator = DualCameraFailoverCoordinator(
        (HEAD, CHEST),
        (HEAD, CHEST),
    )

    initial = coordinator.start()
    assert initial.live_stream == HEAD
    coordinator.mark_success(initial)

    origin_text = coordinator.advance_after_failure(initial)
    assert origin_text is not None
    assert origin_text.live_stream == HEAD
    assert origin_text.stage == "origin_text"

    alternate_text = coordinator.advance_after_failure(origin_text)
    assert alternate_text is not None
    assert alternate_text.live_stream == CHEST
    assert alternate_text.stage == "alternate_text"

    assert coordinator.advance_after_failure(alternate_text) is None


def test_normal_chest_failure_can_try_head_then_return_to_chest() -> None:
    coordinator = DualCameraFailoverCoordinator(
        (HEAD, CHEST),
        (CHEST,),
    )

    initial = coordinator.start()
    assert initial.live_stream == CHEST
    coordinator.mark_success(initial)
    head_text = coordinator.advance_after_failure(initial)
    assert head_text is not None
    assert (head_text.live_stream, head_text.stage) == (HEAD, "alternate_text")
    chest_text = coordinator.advance_after_failure(head_text)
    assert chest_text is not None
    assert (chest_text.live_stream, chest_text.stage) == (CHEST, "origin_text")
    assert coordinator.advance_after_failure(chest_text) is None


def test_initial_reference_resets_tracker_for_each_camera() -> None:
    target = TrackedInstance(
        11,
        0,
        0.91,
        (1.0, 0.0, 5.0, 3.0),
        np.ones((4, 6), dtype=np.uint8),
    )
    yaw_align_target = TrackedInstance(
        12,
        1,
        0.88,
        (0.0, 1.0, 6.0, 4.0),
        np.ones((4, 6), dtype=np.uint8),
    )

    class TextTracker:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.markers: list[int] = []
            self.frames_since_reset = 0
            self.reset_count = 0

        def start_all_text(self, *, target_prompt: str):
            self.prompts.append(target_prompt)
            return {
                "class_names": [target_prompt, "yaw_align_target"],
                "prompt_mode": "target_text_yaw_align_target_text",
            }

        def reset_tracking(self) -> None:
            self.frames_since_reset = 0
            self.reset_count += 1

        def track(self, rgb):
            marker = int(rgb[0, 0, 0])
            self.markers.append(marker)
            self.frames_since_reset += 1
            return [target, yaw_align_target] if self.frames_since_reset == 1 else [yaw_align_target]

    tracker = TextTracker()
    config = BasePoseAgentConfig(task="approach the blue basket")
    calibration = RawServoCalibration(
        6,
        4,
        100.0,
        101.0,
        2.5,
        1.5,
        camera_pitch_deg=-3.0,
    )
    logs: list[str] = []
    initial_targets: dict[str, TrackedInstance | None] = {}
    initial_yaw_align_targets: dict[str, TrackedInstance | None] = {}
    eligible_streams, errors = detect_dual_camera_eligibility(
        config,
        {
            HEAD: _worker_snapshot(HEAD, 1),
            CHEST: _worker_snapshot(CHEST, 2),
        },
        {HEAD: calibration, CHEST: calibration},
        tracker=tracker,
        logger=logs.append,
        targets_out=initial_targets,
        yaw_align_targets_out=initial_yaw_align_targets,
    )

    assert tracker.prompts == ["bluebasket"]
    assert tracker.markers == [1, 2]
    assert tracker.reset_count == 2
    assert eligible_streams == {HEAD, CHEST}
    assert errors == {}
    assert initial_targets == {HEAD: target, CHEST: target}
    assert initial_yaw_align_targets == {HEAD: yaw_align_target, CHEST: yaw_align_target}
    assert "YOLOE prompt" in logs[0]
    assert any(f"stream={HEAD} eligible=true" in line for line in logs)
    assert any(f"stream={CHEST} eligible=true" in line for line in logs)


def test_same_target_yaw_align_target_reuses_instance_and_disables_edge_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = TrackedInstance(
        11,
        0,
        0.91,
        (1.0, 0.0, 5.0, 3.0),
        np.ones((4, 6), dtype=np.uint8),
    )

    class Tracker:
        def track(self, _rgb):
            return [target]

    captured: dict[str, object] = {}

    def capture_observation(_snapshot, target_value, yaw_align_target_value, _calibration, **kwargs):
        captured["target"] = target_value
        captured["yaw_align_target"] = yaw_align_target_value
        captured.update(kwargs)
        return SimpleNamespace(yaw_align_geometry=object(), yaw_align_geometry_error=None)

    monkeypatch.setattr(dual_servo, "_observation", capture_observation)
    calibration = RawServoCalibration(6, 4, 100.0, 101.0, 2.5, 1.5)

    selected_target, selected_yaw_align_target, _observation, _, _ = (
        dual_servo._observe_tracked_snapshot(
            _worker_snapshot(HEAD, 1),
            Tracker(),
            calibration,
            target_id=None,
            yaw_align_target_id=None,
            require_yaw_align_geometry=True,
            reuse_target_as_yaw_align_target=True,
        )
    )

    assert selected_target is target
    assert selected_yaw_align_target is target
    assert captured["target"] is target
    assert captured["yaw_align_target"] is target
    assert captured["exclude_target_from_yaw_align_edge"] is False


def test_head_monitor_reacquires_head_stream_directly() -> None:
    coordinator = DualCameraFailoverCoordinator(
        (HEAD, CHEST),
        (CHEST,),
    )

    attempt = coordinator.begin_head_monitor_reacquisition()

    assert attempt.stage == "head_monitor"
    assert attempt.live_stream == HEAD


def test_head_monitor_reuses_initialized_dynamic_target_prompt() -> None:
    target = TrackedInstance(
        1,
        0,
        0.9,
        (1.0, 0.0, 4.0, 3.0),
        np.ones((4, 6), dtype=np.uint8),
    )
    yaw_align_target = TrackedInstance(
        2,
        1,
        0.8,
        (0.0, 1.0, 6.0, 4.0),
        np.ones((4, 6), dtype=np.uint8),
    )

    class Tracker:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def start_all_text(self, *, target_prompt: str):
            self.prompts.append(target_prompt)

        def track(self, _rgb):
            return [target, yaw_align_target]

    tracker = Tracker()
    monitor = HeadCameraTextMonitor(
        tracker,
        "red tote",
        required_target_frames=2,
    )
    monitor.start()
    first = monitor.inspect(_worker_snapshot(HEAD, 1))
    second = monitor.inspect(_worker_snapshot(HEAD, 2))

    assert tracker.prompts == ["red tote"]
    assert not first.triggered
    assert second.triggered


def test_head_monitor_reuses_target_detection_as_yaw_align_target() -> None:
    target = TrackedInstance(
        1,
        0,
        0.9,
        (1.0, 0.0, 4.0, 3.0),
        np.ones((4, 6), dtype=np.uint8),
    )

    class Tracker:
        def start_all_text(self, **_kwargs):
            return None

        def track(self, _rgb):
            return [target]

    monitor = HeadCameraTextMonitor(
        Tracker(),
        "rubbish bin",
        reuse_target_as_yaw_align_target=True,
    )
    monitor.start()
    result = monitor.inspect(_worker_snapshot(HEAD, 1))

    assert result.target is target
    assert result.yaw_align_target is target


def test_head_monitor_holds_through_two_misses_and_releases_on_third() -> None:
    target = TrackedInstance(
        1,
        0,
        0.9,
        (1.0, 0.0, 4.0, 3.0),
        np.ones((4, 6), dtype=np.uint8),
    )

    class Tracker:
        def __init__(self) -> None:
            self.responses = iter(
                ([target], [], [], [target], [], [], [])
            )

        def start_all_text(self, **_kwargs):
            return None

        def track(self, _rgb):
            return next(self.responses)

    monitor = HeadCameraTextMonitor(
        Tracker(),
        "blue basket",
        required_target_frames=5,
        release_missing_frames=3,
    )
    monitor.start()

    monitor.inspect(_worker_snapshot(HEAD, 1))
    assert (monitor.target_streak, monitor.hold_active) == (1, True)
    monitor.inspect(_worker_snapshot(HEAD, 2))
    assert (monitor.missing_streak, monitor.hold_active) == (1, True)
    monitor.inspect(_worker_snapshot(HEAD, 3))
    assert (monitor.missing_streak, monitor.hold_active) == (2, True)
    monitor.inspect(_worker_snapshot(HEAD, 4))
    assert (monitor.target_streak, monitor.missing_streak) == (1, 0)
    assert monitor.hold_active
    monitor.inspect(_worker_snapshot(HEAD, 5))
    assert monitor.hold_active
    monitor.inspect(_worker_snapshot(HEAD, 6))
    assert monitor.hold_active
    monitor.inspect(_worker_snapshot(HEAD, 7))
    assert monitor.missing_streak == 3
    assert not monitor.hold_active


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
    assert head_only.snapshots[HEAD].depth_aligned_to == HEAD

    # A ready chest frame must remain available after an unrelated head read.
    clients[CHEST].publish(2_000_000_000)
    clients[HEAD].publish(3_000_000_000)
    head = camera.capture_stream(HEAD)
    chest_peek = None
    deadline = time.monotonic() + 0.1
    while chest_peek is None and time.monotonic() < deadline:
        chest_peek = camera.peek_stream(CHEST)
        time.sleep(0.002)
    chest = camera.capture_stream(CHEST)
    assert head.timestamp == pytest.approx(3.0)
    assert chest_peek is not None
    assert chest_peek.timestamp == pytest.approx(2.0)
    # Peeking for diagnostics must not consume the inference frame.
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


def test_dual_runtime_uses_chest_approach_mode_and_head_standoff(tmp_path) -> None:
    config = BasePoseAgentConfig(task="align", output_root=str(tmp_path))
    adapter = GatewayRawServoAdapter(
        config,
        submit_intent=lambda _name, _values: None,
    )

    adapter.runtime.active_camera_stream = HEAD
    adapter.runtime._reset_controller(1.0)
    assert adapter.runtime.controller.target_distance_m == pytest.approx(1.0)
    assert adapter.runtime.controller.far_approach_cutoff_m == pytest.approx(1.3)
    assert not adapter.runtime.controller.chest_approach_only
    assert adapter.runtime.controller.phase is ServoPhase.FORWARD_APPROACH

    adapter.runtime.active_camera_stream = CHEST
    adapter.runtime._reset_controller(2.0)
    assert adapter.runtime.controller.target_distance_m == pytest.approx(0.8)
    assert adapter.runtime.controller.far_approach_cutoff_m == pytest.approx(1.3)
    assert adapter.runtime.controller.chest_approach_only
    assert adapter.runtime.controller.phase is ServoPhase.FORWARD_APPROACH

    adapter.runtime.vertical_recenter_armed = True
    adapter.runtime.active_camera_stream = HEAD
    adapter.runtime._reset_controller(3.0)
    assert not adapter.runtime.controller.chest_approach_only
    assert adapter.runtime.controller.phase is ServoPhase.VERTICAL_RECENTER


def test_head_stage_control_source_uses_shared_tolerances_without_reset(
    tmp_path: Path,
) -> None:
    config = BasePoseAgentConfig(
        task="align",
        output_root=str(tmp_path),
        raw_forward_tolerance_m=0.03,
        raw_lateral_tolerance_m=0.04,
        raw_head_approach_cutoff_m=1.4,
        raw_chest_approach_cutoff_m=1.1,
    )
    runtime = RawServoRuntime(
        config,
        publish=lambda _message: None,
        logger=lambda _message: None,
    )
    runtime.active_camera_stream = HEAD
    runtime.control_source_stream = HEAD
    runtime.controller.phase = ServoPhase.TRANSLATE_TARGET
    observation = _handoff_observation(1.2)

    runtime._prepare_control_source(
        {"control_source_stream": CHEST},
        observation,
    )

    assert runtime.active_camera_stream == HEAD
    assert runtime.control_source_stream == CHEST
    assert runtime.controller.target_distance_m == pytest.approx(0.8)
    assert runtime.controller.far_approach_cutoff_m == pytest.approx(1.1)
    assert runtime.controller.phase is ServoPhase.TRANSLATE_TARGET
    assert runtime.controller.forward_tolerance_m == pytest.approx(0.03)
    assert runtime.controller.lateral_tolerance_m == pytest.approx(0.04)

    runtime._prepare_control_source(
        {"control_source_stream": HEAD},
        observation,
    )

    assert runtime.active_camera_stream == HEAD
    assert runtime.control_source_stream == HEAD
    assert runtime.controller.target_distance_m == pytest.approx(1.0)
    assert runtime.controller.far_approach_cutoff_m == pytest.approx(1.4)
    assert runtime.controller.phase is ServoPhase.TRANSLATE_TARGET
    assert runtime.controller.forward_tolerance_m == pytest.approx(0.03)
    assert runtime.controller.lateral_tolerance_m == pytest.approx(0.04)


def test_runtime_preserves_vertical_recenter_across_head_retries_and_resets_chest(
    tmp_path: Path,
) -> None:
    adapter = GatewayRawServoAdapter(
        BasePoseAgentConfig(task="align", output_root=str(tmp_path)),
        submit_intent=lambda _name, _values: None,
    )
    runtime = adapter.runtime
    runtime.generation = 1
    runtime.phase = "aligning"
    runtime.current_attempt_id = 1
    runtime.active_camera_stream = CHEST

    assert runtime.accept_event(
        RawServoEvent(
            1,
            "switching",
            details={
                "attempt_id": 2,
                "live_stream": HEAD,
                "failover_stage": "alternate_text",
            },
        ),
        now=1.0,
    )
    assert runtime.vertical_recenter_armed
    assert runtime.controller.phase is ServoPhase.VERTICAL_RECENTER

    assert runtime.accept_event(
        RawServoEvent(
            1,
            "switching",
            details={
                "attempt_id": 3,
                "live_stream": HEAD,
                "failover_stage": "head_monitor",
            },
        ),
        now=1.1,
    )
    assert runtime.vertical_recenter_armed
    assert runtime.controller.phase is ServoPhase.VERTICAL_RECENTER

    assert runtime.accept_event(
        RawServoEvent(
            1,
            "switching",
            details={
                "attempt_id": 4,
                "live_stream": CHEST,
                "failover_stage": "alternate_text",
            },
        ),
        now=1.2,
    )
    assert not runtime.vertical_recenter_armed
    assert runtime.controller.chest_approach_only
    assert runtime.controller.phase is ServoPhase.FORWARD_APPROACH

    assert runtime.accept_event(
        RawServoEvent(
            1,
            "switching",
            details={
                "attempt_id": 5,
                "live_stream": CHEST,
                "failover_stage": "origin_text",
            },
        ),
        now=1.3,
    )
    assert runtime.controller.chest_approach_only
    assert runtime.controller.phase is ServoPhase.FORWARD_APPROACH


def _worker_snapshot(stream_name: str, marker: int) -> object:
    from gear_sonic.utils.inference.base_pose.sensor import AlignedRGBDSnapshot

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


def test_sampled_camera_image_copies_rgb_mask_and_head_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _worker_snapshot(HEAD, 7)
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:5] = 1
    target = TrackedInstance(
        1,
        0,
        0.9,
        (2.0, 1.0, 5.0, 3.0),
        mask,
    )
    segment = SimpleNamespace(
        endpoint_a=np.array([0.0, 1.0]),
        endpoint_b=np.array([5.0, 1.0]),
    )
    monkeypatch.setattr(
        dual_servo,
        "_rgb_edge_line_segments",
        lambda _edges: [segment],
    )
    yaw_align_geometry = YawAlignGeometry(
        yaw_error_rad=0.0,
        line_length_px=5.0,
        valid_depth_samples=20,
        line_center_px=(2.5, 2.0),
        line_endpoints_px=((0.0, 2.0), (5.0, 2.0)),
    )

    image = dual_servo._camera_image_data(
        HEAD,
        snapshot,
        target=target,
        edge_intersection=np.ones((4, 6), dtype=np.uint8),
        selected_yaw_align_geometry=yaw_align_geometry,
    )
    snapshot.rgb[:] = 0
    mask[:] = 0

    assert np.all(image.rgb == 7)
    assert np.count_nonzero(image.target_mask) == 6
    assert image.candidate_lines_px == (((0.0, 1.0), (5.0, 1.0)),)
    assert image.selected_line_px == ((0.0, 2.0), (5.0, 2.0))


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


def test_dual_worker_exhausts_the_agent_near_failover_sequence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    started_text_prompts: list[str] = []

    class MissingTracker:
        def start_all_text(self, *, target_prompt: str):
            started_text_prompts.append(target_prompt)
            return {}

        def track(self, _rgb):
            return []

    config = BasePoseAgentConfig(
        task="align",
        output_root=str(tmp_path),
        dual_match_tolerance_frames=2,
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
        yaw_alignment_required=lambda: True,
        position_fallback_allowed=lambda: True,
        tracker_factory=MissingTracker,
        calibration_factory=lambda _config: calibrations,
        eligibility_factory=lambda *_args, **_kwargs: (
            {HEAD, CHEST},
            {},
        ),
    )

    emitted: list[RawServoEvent] = []
    while not events.empty():
        emitted.append(events.get_nowait())
    switching = [event for event in emitted if event.kind == "switching"]
    terminal = [event for event in emitted if event.kind == "error"]
    detecting = [event for event in emitted if event.kind == "detecting"]
    assert started_text_prompts
    assert set(started_text_prompts) == {"bluebasket"}
    assert detecting[0].details["prompt_mode"] == "target_text_yaw_align_target_text"
    assert detecting[0].details["target_prompt"] == "bluebasket"
    assert switching, [(event.kind, event.error) for event in emitted]
    assert [event.details["live_stream"] for event in switching] == [
        HEAD,
        CHEST,
    ]
    assert len(terminal) == 1
    assert "both-camera text failover exhausted" in (terminal[0].error or "")
    output_dir = next(tmp_path.glob("dual_raw_yoloe_*"))
    assert not list(output_dir.rglob("*.json"))
    assert "YOLOE initial selection selected=ego_view" in capsys.readouterr().out
    assert camera.closed


def _handoff_observation(forward_m: float) -> RawServoObservation:
    return RawServoObservation(
        target=TargetGeometry(
            forward_m,
            0.0,
            (forward_m, 0.0, 0.4),
            100,
            0.9,
            forward_m,
        ),
        yaw_align_geometry=None,
        camera_timestamp=1.0,
        target_track_id=1,
        yaw_align_target_track_id=None,
    )


def _joint_observation(
    forward_m: float,
    *,
    right_m: float = 0.0,
    yaw_rad: float = 0.0,
) -> RawServoObservation:
    return replace(
        _handoff_observation(forward_m),
        target=TargetGeometry(
            forward_m,
            right_m,
            (forward_m, -right_m, 0.4),
            100,
            0.9,
            forward_m,
        ),
        yaw_align_geometry=YawAlignGeometry(
            yaw_error_rad=yaw_rad,
            line_length_px=100.0,
            valid_depth_samples=20,
            line_center_px=(320.0, 200.0),
        ),
        yaw_align_geometry_camera_stream=HEAD,
    )


def test_runtime_allows_joint_completion_regardless_of_live_stream(
    tmp_path: Path,
) -> None:
    runtime = RawServoRuntime(
        BasePoseAgentConfig(
            task="align",
            output_root=str(tmp_path),
            raw_chest_target_distance_m=0.7,
            raw_forward_tolerance_m=0.07,
            raw_lateral_tolerance_m=0.07,
            raw_post_stop_sample_frames=0,
        ),
        publish=lambda _message: None,
        logger=lambda _message: None,
    )
    assert runtime.start(1, now=0.0)
    details = {
        "attempt_id": 1,
        "live_stream": HEAD,
        "failover_stage": "initial",
        "control_source_stream": CHEST,
        "yaw_source": {"stream": HEAD, "valid": True, "realtime": True},
    }
    assert runtime.accept_event(
        RawServoEvent(1, "detecting", details=details),
        now=0.01,
    )
    for index in range(5):
        assert runtime.accept_event(
            RawServoEvent(
                1,
                "initialized" if index == 0 else "observation",
                observation=_joint_observation(0.7, right_m=0.02),
                details=details,
            ),
            now=0.02 + index * 0.01,
        )

    assert runtime.controller.terminal_reason == "aligned"
    assert runtime.controller.current.velocity == (0.0, 0.0, 0.0)
    assert runtime.controller.target_distance_m == pytest.approx(0.7)
    assert runtime.phase == "idle"


def test_runtime_reuses_yoloe_after_ten_post_stop_deviation_frames(
    tmp_path: Path,
) -> None:
    published: list[dict[str, object]] = []
    logs: list[str] = []
    runtime = RawServoRuntime(
        BasePoseAgentConfig(
            task="align",
            output_root=str(tmp_path),
            raw_chest_target_distance_m=0.7,
            raw_forward_tolerance_m=0.07,
            raw_lateral_tolerance_m=0.07,
            raw_post_stop_sample_frames=30,
            raw_post_stop_deviation_frames=10,
        ),
        publish=published.append,
        logger=logs.append,
    )
    assert runtime.start(1, now=0.0)
    assert runtime.requests.get_nowait() == 1
    details = {
        "attempt_id": 1,
        "live_stream": HEAD,
        "failover_stage": "initial",
        "control_source_stream": CHEST,
        "yaw_source": {"stream": HEAD, "valid": True, "realtime": True},
    }
    assert runtime.accept_event(
        RawServoEvent(1, "detecting", details=details),
        now=0.01,
    )
    for index in range(5):
        assert runtime.accept_event(
            RawServoEvent(
                1,
                "initialized" if index == 0 else "observation",
                observation=_joint_observation(0.7),
                details=details,
            ),
            now=0.02 + index * 0.01,
        )
    assert runtime.controller.phase is ServoPhase.POST_STOP_SAMPLING

    for index in range(10):
        assert runtime.accept_event(
            RawServoEvent(
                1,
                "observation",
                observation=_joint_observation(1.2),
                details=details,
            ),
            now=0.1 + index * 0.01,
        )

    assert runtime.generation == 1
    assert runtime.requests.empty()
    assert runtime.phase == "aligning"
    assert runtime.controller.phase is ServoPhase.TRANSLATE_TARGET
    assert runtime.controller.post_stop_out_of_tolerance_streak == 10
    assert runtime.controller.post_stop_realign_count == 1
    assert runtime.controller.current.vx > 0.0
    assert published[-1]["action"] == "visual_servo"
    assert any("RESUME existing YOLOE navigation" in line for line in logs)


@pytest.mark.parametrize(
    ("missing_frames", "expected_stage"),
    (
        (1, "grace"),
        (9, "grace"),
        (10, "zero_hold"),
        (19, "zero_hold"),
        (20, "switch_state"),
    ),
)
def test_joint_head_yaw_loss_policy_boundaries(
    missing_frames: int,
    expected_stage: str,
) -> None:
    details = dual_servo._joint_head_yaw_loss_details(missing_frames)

    assert details == {
        "missing_frames": missing_frames,
        "grace_frames": 10,
        "zero_hold_frames": 10,
        "switch_frame": 20,
        "stage": expected_stage,
    }


def test_runtime_keeps_bidirectional_chest_control_for_nine_head_yaw_misses(
    tmp_path: Path,
) -> None:
    runtime = RawServoRuntime(
        BasePoseAgentConfig(
            task="align",
            output_root=str(tmp_path),
            raw_chest_target_distance_m=0.7,
            raw_forward_tolerance_m=0.07,
        ),
        publish=lambda _message: None,
        logger=lambda _message: None,
    )
    assert runtime.start(1, now=0.0)
    common = {
        "attempt_id": 1,
        "live_stream": CHEST,
        "failover_stage": "initial",
        "control_source_stream": CHEST,
    }
    live_yaw = {
        **common,
        "yaw_source": {"stream": HEAD, "valid": True, "realtime": True},
    }
    assert runtime.accept_event(
        RawServoEvent(1, "detecting", details=common),
        now=0.01,
    )
    assert runtime.accept_event(
        RawServoEvent(
            1,
            "initialized",
            observation=_joint_observation(0.5),
            details=live_yaw,
        ),
        now=0.02,
    )
    assert runtime.controller.current.vx < 0.0

    for missing_frames in range(1, 10):
        loss = dual_servo._joint_head_yaw_loss_details(missing_frames)
        assert runtime.accept_event(
            RawServoEvent(
                1,
                "observation",
                observation=_handoff_observation(0.5),
                details={
                    **common,
                    "yaw_source": {
                        "stream": HEAD,
                        "valid": False,
                        "realtime": True,
                        "error": "missing tracked yaw_align_target for live head yaw",
                    },
                    "head_yaw_loss": loss,
                },
            ),
            now=0.02 + 0.01 * missing_frames,
        )
        assert runtime.controller.current.vx < 0.0

    for missing_frames in range(10, 20):
        loss = dual_servo._joint_head_yaw_loss_details(missing_frames)
        assert runtime.accept_event(
            RawServoEvent(
                1,
                "observation",
                observation=_handoff_observation(0.5),
                details={
                    **common,
                    "yaw_source": {
                        "stream": HEAD,
                        "valid": False,
                        "realtime": True,
                    },
                    "head_yaw_loss": loss,
                },
            ),
            now=0.02 + 0.01 * missing_frames,
        )
        assert runtime.controller.current.velocity == (0.0, 0.0, 0.0)

    assert runtime.accept_event(
        RawServoEvent(
            1,
            "observation",
            observation=_joint_observation(0.5),
            details=live_yaw,
        ),
        now=0.22,
    )
    assert runtime.controller.current.vx < 0.0


def test_joint_chest_loss_coasts_four_frames_then_holds_zero(
    tmp_path: Path,
) -> None:
    runtime = RawServoRuntime(
        BasePoseAgentConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _message: None,
    )
    assert runtime.start(1, now=0.0)
    common = {
        "attempt_id": 1,
        "live_stream": CHEST,
        "failover_stage": "initial",
        "control_source_stream": CHEST,
    }
    assert runtime.accept_event(
        RawServoEvent(1, "detecting", details=common),
        now=0.01,
    )
    assert runtime.accept_event(
        RawServoEvent(
            1,
            "initialized",
            observation=_handoff_observation(1.2),
            details=common,
        ),
        now=0.02,
    )
    original = runtime.controller.current
    assert original.vx > 0.0

    for missing_frames in range(1, 5):
        assert runtime.accept_event(
            RawServoEvent(
                1,
                "invalid",
                error="missing tracked target",
                details={
                    **common,
                    "chest_target_loss": {
                        "missing_frames": missing_frames,
                        "grace_frames": 5,
                        "zero_hold_frames": 5,
                        "stage": "grace",
                    },
                },
            ),
            now=0.02 + missing_frames * 0.01,
        )
        assert runtime.controller.current.velocity == original.velocity

    assert runtime.accept_event(
        RawServoEvent(
            1,
            "invalid",
            error="missing tracked target",
            details={
                **common,
                "chest_target_loss": {
                    "missing_frames": 5,
                    "grace_frames": 5,
                    "zero_hold_frames": 5,
                    "stage": "zero_hold",
                },
            },
        ),
        now=0.07,
    )
    assert runtime.controller.current.velocity == (0.0, 0.0, 0.0)


def test_runtime_stops_on_first_head_hit_and_resumes_after_three_misses(
    tmp_path: Path,
) -> None:
    published: list[dict[str, object]] = []
    runtime = RawServoRuntime(
        BasePoseAgentConfig(task="align", output_root=str(tmp_path)),
        publish=published.append,
        logger=lambda _message: None,
    )
    assert runtime.start(1, now=0.0)
    common = {
        "attempt_id": 1,
        "live_stream": CHEST,
        "failover_stage": "initial",
    }
    assert runtime.accept_event(
        RawServoEvent(1, "detecting", details=common),
        now=0.01,
    )
    assert runtime.accept_event(
        RawServoEvent(
            1,
            "initialized",
            observation=_handoff_observation(1.2),
            details={
                **common,
                "head_monitor": {
                    "target_streak": 0,
                    "missing_streak": 3,
                    "hold_active": False,
                },
            },
        ),
        now=0.02,
    )
    assert runtime.controller.current.vx > 0.0

    assert runtime.accept_event(
        RawServoEvent(
            1,
            HEAD_MONITOR_HOLD_EVENT,
            details={
                **common,
                "head_monitor": {
                    "target_streak": 1,
                    "missing_streak": 0,
                    "hold_active": True,
                },
            },
        ),
        now=0.03,
    )
    assert runtime.head_monitor_hold_active
    assert published[-1]["velocity"] == {
        "vx": 0.0,
        "vy": 0.0,
        "wz": 0.0,
    }

    for missing_streak in (1, 2):
        assert runtime.accept_event(
            RawServoEvent(
                1,
                "observation",
                observation=_handoff_observation(1.2),
                details={
                    **common,
                    "head_monitor": {
                        "target_streak": 0,
                        "missing_streak": missing_streak,
                        "hold_active": True,
                    },
                },
            ),
            now=0.03 + missing_streak * 0.01,
        )
        assert runtime.head_monitor_hold_active
        assert runtime.controller.current.velocity == (0.0, 0.0, 0.0)

    assert runtime.accept_event(
        RawServoEvent(
            1,
            "observation",
            observation=_handoff_observation(1.2),
            details={
                **common,
                "head_monitor": {
                    "target_streak": 0,
                    "missing_streak": 3,
                    "hold_active": False,
                },
            },
        ),
        now=0.06,
    )
    assert not runtime.head_monitor_hold_active
    assert runtime.controller.current.vx > 0.0


def test_close_chest_distance_does_not_trigger_head_handoff(
    tmp_path: Path,
) -> None:
    from gear_sonic.utils.inference.base_pose.sensor import AlignedRGBDSnapshot

    height, width = 120, 160

    def snapshot(
        stream_name: str,
        marker: int,
        *,
        depth_mm: int = 600,
    ) -> AlignedRGBDSnapshot:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[88:93, 8:152] = 255
        return AlignedRGBDSnapshot(
            rgb=rgb,
            depth_raw=np.full((height, width), depth_mm, dtype=np.uint16),
            fx=100.0,
            fy=101.0,
            cx=79.5,
            cy=59.5,
            depth_scale_m=0.001,
            depth_aligned_to=stream_name,
            depth_source=(
                "depth-anything" if stream_name == CHEST else None
            ),
            timestamp=float(marker),
        )

    class ChestCamera:
        def __init__(self) -> None:
            self.marker = 0
            self.closed = False

        def capture(self) -> DualBasePoseCapture:
            self.marker += 1
            return DualBasePoseCapture(
                snapshots={CHEST: snapshot(CHEST, self.marker)},
                errors={HEAD: "head unavailable during initial grounding"},
            )

        def capture_stream(self, stream_name: str, *, timeout_ms: int):
            assert stream_name in {HEAD, CHEST}
            assert timeout_ms > 0
            self.marker += 1
            return snapshot(stream_name, self.marker)

        def close(self) -> None:
            self.closed = True

    target_mask = np.zeros((height, width), dtype=np.uint8)
    target_mask[20:50, 60:100] = 1
    target = TrackedInstance(
        1,
        0,
        0.9,
        (60.0, 20.0, 100.0, 50.0),
        target_mask,
    )
    yaw_align_target = TrackedInstance(
        2,
        1,
        0.9,
        (0.0, 70.0, float(width), float(height)),
        np.ones((height, width), dtype=np.uint8),
    )

    class Tracker:
        def __init__(self) -> None:
            self.start_calls = 0
            self.track_calls = 0

        def start_all_text(self, **_kwargs):
            self.start_calls += 1
            return {}

        def track(self, _rgb):
            self.track_calls += 1
            return [target]

    class EmptyHeadMonitorTracker:
        def start_all_text(self, **_kwargs):
            return None

        def track(self, _rgb):
            return []

    config = BasePoseAgentConfig(
        task="approach the blue basket",
        output_root=str(tmp_path),
    )
    requests: queue.Queue[int | None] = queue.Queue()
    requests.put(1)
    requests.put(None)
    events: queue.Queue[RawServoEvent] = queue.Queue()
    gate = GenerationGate()
    gate.activate(1)
    camera = ChestCamera()
    tracker = Tracker()
    emitted: list[RawServoEvent] = []
    worker = threading.Thread(
        target=run_dual_raw_servo_worker,
        args=(config, requests, events, gate, threading.Event()),
        kwargs={
            "camera_factory": lambda: camera,
            "tracker_factory": lambda: tracker,
            "head_monitor_tracker_factory": EmptyHeadMonitorTracker,
            "calibration_factory": lambda _config: {
                HEAD: RawServoCalibration(
                    width, height, 100.0, 101.0, 79.5, 59.5
                ),
                CHEST: RawServoCalibration(
                    width,
                    height,
                    100.0,
                    101.0,
                    79.5,
                    59.5,
                    camera_pitch_deg=-3.0,
                ),
            },
            "eligibility_factory": lambda *_args, **_kwargs: (
                {CHEST},
                {HEAD: "head unavailable during initial grounding"},
            ),
            "yaw_alignment_required": lambda: False,
            "position_fallback_allowed": lambda: True,
        },
        daemon=True,
    )
    worker.start()
    while len(
        [
            event
            for event in emitted
            if event.kind in {"initialized", "observation"}
        ]
    ) < 5:
        emitted.append(events.get(timeout=2.0))
    gate.cancel(1)
    worker.join(timeout=2.0)
    while not events.empty():
        emitted.append(events.get_nowait())
    applied = [
        event
        for event in emitted
        if event.kind in {"initialized", "observation"}
    ]
    switching = [event for event in emitted if event.kind == "switching"]
    terminal = [event for event in emitted if event.kind == "error"]

    assert len(applied) >= 5
    assert all(
        event.details["perception_schedule"] == "new_frame_latest_only"
        for event in applied
    )
    assert all(event.observation is not None for event in applied)
    assert all(event.observation.yaw_align_geometry is None for event in applied)
    assert switching == []
    assert terminal == []
    assert tracker.start_calls == 1
    assert not worker.is_alive()
    assert camera.closed


def test_head_monitor_keeps_chest_position_when_live_head_yaw_is_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gear_sonic.utils.inference.base_pose.sensor import AlignedRGBDSnapshot

    height, width = 120, 160

    def snapshot(
        stream_name: str,
        marker: int,
        *,
        depth_mm: int = 600,
    ) -> AlignedRGBDSnapshot:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[88:93, 8:152] = 255
        return AlignedRGBDSnapshot(
            rgb=rgb,
            depth_raw=np.full((height, width), depth_mm, dtype=np.uint16),
            fx=100.0,
            fy=101.0,
            cx=79.5,
            cy=59.5,
            depth_scale_m=0.001,
            depth_aligned_to=stream_name,
            depth_source=None,
            timestamp=float(marker),
        )

    class Camera:
        def __init__(self) -> None:
            self.marker = 0
            self.chest_live_frames = 0
            self.closed = False

        def _next(self, stream_name: str) -> AlignedRGBDSnapshot:
            self.marker += 1
            depth_mm = 600
            if stream_name == CHEST:
                self.chest_live_frames += 1
                depth_mm = 1000 if self.chest_live_frames <= 2 else 600
            return snapshot(stream_name, self.marker, depth_mm=depth_mm)

        def capture(self) -> DualBasePoseCapture:
            self.marker += 1
            return DualBasePoseCapture(
                snapshots={CHEST: snapshot(CHEST, self.marker, depth_mm=1000)},
                errors={HEAD: "initial head grounding unavailable"},
            )

        def capture_stream(self, stream_name: str, *, timeout_ms: int):
            assert timeout_ms > 0
            return self._next(stream_name)

        def poll_stream(self, stream_name: str):
            return self._next(stream_name)

        def close(self) -> None:
            self.closed = True

    target_mask = np.zeros((height, width), dtype=np.uint8)
    target_mask[20:50, 60:100] = 1
    target = TrackedInstance(
        1,
        0,
        0.9,
        (60.0, 20.0, 100.0, 50.0),
        target_mask,
    )
    yaw_align_target = TrackedInstance(
        2,
        1,
        0.9,
        (0.0, 70.0, float(width), float(height)),
        np.ones((height, width), dtype=np.uint8),
    )
    tracking_barrier = threading.Barrier(2)
    geometry_streams: list[str] = []
    original_yaw_align_target_geometry = visual_servo._yaw_align_target_geometry_components

    def counting_yaw_align_target_geometry(snapshot_value, *args, **kwargs):
        geometry_streams.append(snapshot_value.depth_aligned_to)
        return original_yaw_align_target_geometry(snapshot_value, *args, **kwargs)

    monkeypatch.setattr(
        dual_servo,
        "_yaw_align_target_geometry_components",
        counting_yaw_align_target_geometry,
    )
    monkeypatch.setattr(
        visual_servo,
        "_yaw_align_target_geometry_components",
        counting_yaw_align_target_geometry,
    )

    class Tracker:
        def start_all_text(self, **_kwargs):
            return {}

        def track(self, _rgb):
            tracking_barrier.wait(timeout=1.0)
            return [target, yaw_align_target]

    class MonitorTracker:
        def start_all_text(self, **_kwargs):
            return None

        def track(self, _rgb):
            tracking_barrier.wait(timeout=1.0)
            return [target, yaw_align_target]

    config = BasePoseAgentConfig(
        task="approach the blue basket",
        output_root=str(tmp_path),
    )
    requests: queue.Queue[int | None] = queue.Queue()
    requests.put(1)
    events: queue.Queue[RawServoEvent] = queue.Queue()
    gate = GenerationGate()
    gate.activate(1)
    camera = Camera()
    calibration = RawServoCalibration(
        width, height, 100.0, 101.0, 79.5, 59.5
    )
    worker = threading.Thread(
        target=run_dual_raw_servo_worker,
        args=(config, requests, events, gate, threading.Event()),
        kwargs={
            "camera_factory": lambda: camera,
            "tracker_factory": Tracker,
            "head_monitor_tracker_factory": MonitorTracker,
            "calibration_factory": lambda _config: {
                HEAD: calibration,
                CHEST: calibration,
            },
            "eligibility_factory": lambda *_args, **_kwargs: (
                {CHEST},
                {HEAD: "initial head grounding unavailable"},
            ),
            "yaw_alignment_required": lambda: False,
            "position_fallback_allowed": lambda: True,
        },
        daemon=True,
    )
    worker.start()

    emitted: list[RawServoEvent] = []
    observations: list[RawServoEvent] = []
    for _ in range(20):
        event = events.get(timeout=2.0)
        emitted.append(event)
        if event.kind in {"initialized", "observation"}:
            observations.append(event)
        if len(observations) == 3:
            break
    gate.cancel(1)
    requests.put(None)
    worker.join(timeout=2.0)

    assert observations
    assert all(event.details["live_stream"] == CHEST for event in observations)
    assert all(
        event.details["control_source_stream"] == CHEST
        for event in observations
    )
    assert all(event.observation is not None for event in observations)
    assert all(event.observation.yaw_align_geometry is not None for event in observations)
    assert all(
        event.observation.yaw_align_geometry_camera_stream == HEAD
        for event in observations
        if event.observation is not None
    )
    assert all(event.details["yaw_source"]["valid"] for event in observations)
    assert all(event.frame is not None for event in observations)
    assert all(
        event.details["parallel_perception"]["enabled"]
        for event in observations
    )
    assert all(
        event.details["parallel_perception"]["chest_yaw_align_geometry_skipped"]
        for event in observations
    )
    assert geometry_streams
    assert set(geometry_streams) == {HEAD}
    assert all(
        event.details["head_monitor"]["status"] == "joint_tracking"
        for event in observations
    )
    assert all(
        event.details["head_monitor"]["hold_active"] is False
        for event in observations
    )
    hold_events = [
        event for event in emitted if event.kind == HEAD_MONITOR_HOLD_EVENT
    ]
    assert hold_events == []
    assert all(
        "chest_distance_handoff" not in event.details
        for event in emitted
        if event.details is not None
    )
    assert all(event.kind != "switching" for event in emitted)
    assert not worker.is_alive()
    assert camera.closed


def test_worker_switches_state_after_twenty_joint_head_yaw_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = TrackedInstance(
        1,
        0,
        0.9,
        (1.0, 0.0, 5.0, 3.0),
        np.ones((4, 6), dtype=np.uint8),
    )
    yaw_align_target = TrackedInstance(
        2,
        1,
        0.9,
        (0.0, 1.0, 6.0, 4.0),
        np.ones((4, 6), dtype=np.uint8),
    )
    yaw_align_geometry = YawAlignGeometry(
        yaw_error_rad=0.0,
        line_length_px=100.0,
        valid_depth_samples=20,
        line_center_px=(3.0, 2.0),
    )

    def observe_chest(snapshot, *_args, **_kwargs):
        observation = replace(
            _handoff_observation(0.5),
            camera_timestamp=snapshot.timestamp,
            target_bbox_xyxy=target.bbox_xyxy,
            image_width=6,
            image_height=4,
        )
        return target, None, observation, False, False

    monkeypatch.setattr(
        dual_servo,
        "_observe_tracked_snapshot",
        observe_chest,
    )
    monkeypatch.setattr(
        dual_servo,
        "_yaw_align_target_geometry_components",
        lambda *_args, **_kwargs: (
            yaw_align_geometry,
            None,
            np.ones((4, 6), dtype=np.uint8),
            None,
            (),
        ),
    )

    class Camera:
        def __init__(self) -> None:
            self.marker = 0
            self.closed = False

        def capture(self) -> DualBasePoseCapture:
            return DualBasePoseCapture(
                snapshots={CHEST: _worker_snapshot(CHEST, 0)},
                errors={HEAD: "initial head grounding unavailable"},
            )

        def _next(self, stream_name: str):
            self.marker += 1
            return _worker_snapshot(stream_name, self.marker)

        def capture_stream(self, stream_name: str, *, timeout_ms: int):
            assert timeout_ms > 0
            return self._next(stream_name)

        def poll_stream(self, stream_name: str):
            return self._next(stream_name)

        def close(self) -> None:
            self.closed = True

    class ChestTracker:
        def start_all_text(self, **_kwargs):
            return {}

    class HeadMonitorTracker:
        def __init__(self) -> None:
            self.calls = 0

        def start_all_text(self, **_kwargs):
            return {}

        def track(self, _rgb):
            self.calls += 1
            return [target, yaw_align_target] if self.calls == 1 else [target]

    config = BasePoseAgentConfig(
        task="align to the blue basket",
        output_root=str(tmp_path),
    )
    requests: queue.Queue[int | None] = queue.Queue()
    requests.put(1)
    events: queue.Queue[RawServoEvent] = queue.Queue()
    gate = GenerationGate()
    gate.activate(1)
    camera = Camera()
    calibration = RawServoCalibration(6, 4, 100.0, 101.0, 2.5, 1.5)
    worker = threading.Thread(
        target=run_dual_raw_servo_worker,
        args=(config, requests, events, gate, threading.Event()),
        kwargs={
            "camera_factory": lambda: camera,
            "tracker_factory": ChestTracker,
            "head_monitor_tracker_factory": HeadMonitorTracker,
            "calibration_factory": lambda _config: {
                HEAD: calibration,
                CHEST: calibration,
            },
            "eligibility_factory": lambda *_args, **_kwargs: (
                {CHEST},
                {HEAD: "initial head grounding unavailable"},
            ),
            "yaw_alignment_required": lambda: False,
            "position_fallback_allowed": lambda: True,
        },
        daemon=True,
    )
    worker.start()

    emitted: list[RawServoEvent] = []
    switching: RawServoEvent | None = None
    for _ in range(80):
        event = events.get(timeout=3.0)
        emitted.append(event)
        if (
            event.kind == "switching"
            and event.details is not None
            and event.details.get("head_yaw_loss") is not None
        ):
            switching = event
            break
    gate.cancel(1)
    requests.put(None)
    worker.join(timeout=3.0)

    loss_events = [
        event
        for event in emitted
        if event.kind in {"initialized", "observation"}
        and event.details is not None
        and event.details.get("head_yaw_loss") is not None
    ]
    assert [
        event.details["head_yaw_loss"]["missing_frames"]
        for event in loss_events
    ] == list(range(1, 20))
    assert [
        event.details["head_yaw_loss"]["stage"] for event in loss_events
    ] == ["grace"] * 9 + ["zero_hold"] * 10
    assert switching is not None
    assert switching.details is not None
    assert switching.details["live_stream"] == HEAD
    assert switching.details["head_yaw_loss"] == {
        "missing_frames": 20,
        "grace_frames": 10,
        "zero_hold_frames": 10,
        "switch_frame": 20,
        "stage": "switch_state",
    }
    assert all(event.kind != HEAD_MONITOR_HOLD_EVENT for event in emitted)
    assert not worker.is_alive()
    assert camera.closed


def test_head_stage_keeps_chest_basket_until_it_becomes_invalid(
    tmp_path: Path,
) -> None:
    from gear_sonic.utils.inference.base_pose.sensor import AlignedRGBDSnapshot

    height, width = 120, 160

    def snapshot(stream_name: str, marker: int) -> AlignedRGBDSnapshot:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        if stream_name == HEAD:
            rgb[99:102, 5:155] = 255
        return AlignedRGBDSnapshot(
            rgb=rgb,
            depth_raw=np.full((height, width), 1000, dtype=np.uint16),
            fx=100.0,
            fy=101.0,
            cx=79.5,
            cy=59.5,
            depth_scale_m=0.001,
            depth_aligned_to=stream_name,
            depth_source=None,
            timestamp=float(marker),
        )

    class Camera:
        def __init__(self) -> None:
            self.marker = 0
            self.closed = False
            self.capture_streams: list[str] = []

        def capture(self) -> DualBasePoseCapture:
            return DualBasePoseCapture(
                snapshots={HEAD: snapshot(HEAD, 0), CHEST: snapshot(CHEST, 0)},
                errors={},
            )

        def capture_stream(self, stream_name: str, *, timeout_ms: int):
            assert stream_name in {HEAD, CHEST}
            assert timeout_ms > 0
            self.capture_streams.append(stream_name)
            self.marker += 1
            return snapshot(stream_name, self.marker)

        def close(self) -> None:
            self.closed = True

    target_mask = np.zeros((height, width), dtype=np.uint8)
    target_mask[25:65, 55:105] = 1
    chest_target = TrackedInstance(
        7,
        0,
        0.9,
        (55.0, 25.0, 105.0, 65.0),
        target_mask,
    )
    head_target = replace(chest_target, track_id=8)
    yaw_align_target_mask = np.zeros((height, width), dtype=np.uint8)
    yaw_align_target_mask[85:115, 5:155] = 1
    head_yaw_align_target = TrackedInstance(
        9,
        1,
        0.95,
        (5.0, 85.0, 155.0, 115.0),
        yaw_align_target_mask,
    )

    class HeadTracker:
        def __init__(self) -> None:
            self.calls = 0

        def start_all_text(self, **_kwargs):
            return {}

        def track(self, _rgb):
            self.calls += 1
            return (
                [head_yaw_align_target]
                if self.calls == 1
                else [head_target, head_yaw_align_target]
            )

    class ChestFallbackTracker:
        def __init__(self) -> None:
            self.started = False
            self.calls = 0

        def start_all_text(self, **_kwargs):
            self.started = True
            return {}

        def track(self, _rgb):
            assert self.started
            self.calls += 1
            return [chest_target] if self.calls <= 2 else []

    config = BasePoseAgentConfig(
        task="align to the blue basket",
        output_root=str(tmp_path),
    )
    calibration = RawServoCalibration(
        width,
        height,
        100.0,
        101.0,
        79.5,
        59.5,
    )
    requests: queue.Queue[int | None] = queue.Queue()
    requests.put(1)
    events: queue.Queue[RawServoEvent] = queue.Queue()
    gate = GenerationGate()
    gate.activate(1)
    camera = Camera()
    worker = threading.Thread(
        target=run_dual_raw_servo_worker,
        args=(config, requests, events, gate, threading.Event()),
        kwargs={
            "camera_factory": lambda: camera,
            "tracker_factory": HeadTracker,
            "head_monitor_tracker_factory": ChestFallbackTracker,
            "calibration_factory": lambda _config: {
                HEAD: calibration,
                CHEST: calibration,
            },
            "eligibility_factory": lambda *_args, **_kwargs: (
                {HEAD, CHEST},
                {},
            ),
            "yaw_alignment_required": lambda: False,
            "position_fallback_allowed": lambda: True,
        },
        daemon=True,
    )
    worker.start()

    emitted: list[RawServoEvent] = []
    observations: list[RawServoEvent] = []
    for _ in range(20):
        event = events.get(timeout=2.0)
        emitted.append(event)
        if event.kind in {"initialized", "observation"}:
            observations.append(event)
        if len(observations) == 3:
            break
    gate.cancel(1)
    requests.put(None)
    worker.join(timeout=2.0)

    assert [event.details["control_source_stream"] for event in observations] == [
        CHEST,
        CHEST,
        HEAD,
    ]
    assert observations[0].details["basket_source"] == {
        "preferred_stream": HEAD,
        "selected_stream": CHEST,
        "source_switched": True,
        f"{HEAD}_valid": False,
        f"{HEAD}_error": "missing tracked target",
        f"{CHEST}_valid": True,
        "chest_fallback_used": True,
        "without_stage_switch": True,
    }
    assert observations[1].details["basket_source"]["preferred_stream"] == CHEST
    assert observations[1].details["basket_source"]["selected_stream"] == CHEST
    assert not observations[1].details["basket_source"]["source_switched"]
    assert observations[1].details["basket_source"][f"{CHEST}_valid"]
    assert observations[0].details["yaw_source"]["stream"] == HEAD
    assert observations[0].details["yaw_source"]["realtime"]
    assert observations[0].details["yaw_source"]["valid"]
    assert observations[1].details["yaw_source"]["stream"] == HEAD
    assert observations[1].details["yaw_source"]["realtime"]
    assert observations[1].details["yaw_source"]["valid"]
    assert observations[0].observation is not None
    assert observations[0].observation.yaw_align_geometry is not None
    assert observations[0].observation.yaw_align_geometry.line_endpoints_px is not None
    assert observations[0].observation.completed_yaw_align_target_mask is not None
    assert observations[0].observation.yaw_align_geometry_camera_stream == HEAD
    assert observations[1].observation is not None
    assert observations[1].observation.yaw_align_geometry is not None
    assert observations[1].observation.yaw_align_geometry.line_endpoints_px is not None
    assert observations[1].observation.completed_yaw_align_target_mask is not None
    assert observations[1].observation.yaw_align_geometry_camera_stream == HEAD
    loss_events = [
        event
        for event in emitted
        if event.kind == "invalid"
        and event.details.get("chest_target_loss") is not None
    ]
    assert [
        event.details["chest_target_loss"]["missing_frames"]
        for event in loss_events
    ] == list(range(1, 11))
    assert [
        event.details["chest_target_loss"]["stage"]
        for event in loss_events
    ] == ["grace"] * 4 + ["zero_hold"] * 5 + ["switch_to_head"]
    assert observations[2].details["basket_source"]["preferred_stream"] == HEAD
    assert observations[2].details["basket_source"]["selected_stream"] == HEAD
    assert not observations[2].details["basket_source"]["source_switched"]
    assert observations[2].details["basket_source"][f"{HEAD}_valid"]
    assert observations[0].frame is not None
    assert observations[0].frame.camera_stream == CHEST
    assert observations[1].frame is not None
    assert observations[1].frame.camera_stream == CHEST
    assert observations[2].frame is not None
    assert observations[2].frame.camera_stream == HEAD
    assert camera.capture_streams[:7] == [
        HEAD,
        CHEST,
        HEAD,
        CHEST,
        HEAD,
        CHEST,
        HEAD,
    ]
    assert all(event.kind != "switching" for event in emitted)
    assert not worker.is_alive()
    assert camera.closed
