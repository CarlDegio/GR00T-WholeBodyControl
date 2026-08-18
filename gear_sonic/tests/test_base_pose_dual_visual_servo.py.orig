"""Software-only tests for dual-camera raw YOLOE failover."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import queue
import sys
import threading
import time
from typing import Callable

import numpy as np
import pytest

import gear_sonic.utils.inference.base_pose_dual_visual_servo as dual_servo
from gear_sonic.scripts.base_pose_planner import BasePosePlannerConfig
from gear_sonic.scripts.launch_inference import (
    InferenceLaunchConfig,
    build_planner_input_command,
    build_reasan_planner_command,
    uses_base_pose_manual_keyboard,
)
from gear_sonic.utils.inference.base_pose import (
    AlignedRGBDSnapshot,
    BASE_POSE_MODES,
    DualRGBDCapture,
)
from gear_sonic.utils.inference.base_pose_dual_visual_servo import (
    DualCameraFailoverCoordinator,
    DualCameraReference,
    LatestReferenceGate,
    dual_calibrations_from_config,
    ground_dual_raw_servo_references,
    run_dual_raw_servo_worker,
)
from gear_sonic.utils.inference.base_pose_visual_servo import (
    GenerationGate,
    RawServoCalibration,
    RawServoEvent,
    RawServoObservation,
    RawServoRuntime,
    ServoPhase,
    TableGeometry,
    TargetGeometry,
    TrackedInstance,
)
from gear_sonic.utils.inference.base_pose_visual_servo_diagnostics import (
    AsyncFrameDiagnosticsWriter,
    DetectionFrameData,
    FrameDiagnosticsWriter,
)


HEAD = "ego_view"
CHEST = "chest_view"


def reference(
    stream_name: str,
    *,
    kind: str = "initial",
    marker: int = 0,
) -> DualCameraReference:
    return DualCameraReference(
        stream_name=stream_name,
        rgb=np.full((4, 6, 3), marker, dtype=np.uint8),
        target_prompt="blue basket",
        target_bbox=(100.0, 100.0, 300.0, 400.0),
        table_bboxes=((50.0, 300.0, 950.0, 900.0),),
        camera_timestamp=float(marker + 1),
        kind=kind,
    )


def rgbd_snapshot(stream_name: str, *, marker: int = 0) -> AlignedRGBDSnapshot:
    return AlignedRGBDSnapshot(
        rgb=np.full((4, 6, 3), marker, dtype=np.uint8),
        depth_raw=np.full((4, 6), 1000 + marker, dtype=np.uint16),
        fx=100.0,
        fy=101.0,
        cx=2.5,
        cy=1.5,
        depth_scale_m=0.001,
        depth_aligned_to=stream_name,
        depth_source=None,
        timestamp=float(marker + 1),
    )


def target_payload() -> dict[str, object]:
    return {
        "status": "READY",
        "primary_target": {
            "text_prompt": "blue plastic basket",
            "bbox_2d": [300.0, 250.0, 600.0, 700.0],
        },
        "manipulation_anchor": "basket opening",
        "selection_reason": "large stable task target",
        "confidence": 0.9,
        "limitations": "",
    }


def table_payload() -> dict[str, object]:
    return {
        "status": "READY",
        "target": "desk",
        "boxes": [
            {"bbox_2d": [25.0, 100.0, 975.0, 900.0], "confidence": 0.91}
        ],
        "limitations": "",
    }


def test_coordinator_prefers_head_when_both_initial_references_exist() -> None:
    coordinator = DualCameraFailoverCoordinator(
        (HEAD, CHEST),
        {HEAD: reference(HEAD), CHEST: reference(CHEST)},
    )

    attempt = coordinator.start()

    assert attempt.attempt_id == 1
    assert attempt.live_stream == HEAD
    assert attempt.reference.stream_name == HEAD
    assert attempt.reference.kind == "initial"
    assert attempt.stage == "initial"
    assert attempt.origin_stream == HEAD


def test_coordinator_starts_only_camera_with_an_initial_reference() -> None:
    coordinator = DualCameraFailoverCoordinator(
        (HEAD, CHEST),
        {CHEST: reference(CHEST)},
    )

    attempt = coordinator.start()

    assert attempt.live_stream == CHEST
    assert attempt.reference.stream_name == CHEST


def test_coordinator_runs_origin_text_qwen_before_alternate_text_qwen() -> None:
    head_initial = reference(HEAD, marker=1)
    chest_initial = reference(CHEST, marker=2)
    coordinator = DualCameraFailoverCoordinator(
        (HEAD, CHEST),
        {HEAD: head_initial, CHEST: chest_initial},
    )
    first = coordinator.start()
    coordinator.mark_success(first)

    origin_text = coordinator.advance_after_failure(first)
    assert origin_text is not None
    assert origin_text.live_stream == HEAD
    assert origin_text.reference is head_initial
    assert origin_text.stage == "origin_text"
    assert origin_text.origin_stream == HEAD

    origin_qwen = coordinator.advance_after_failure(origin_text)
    assert origin_qwen is not None
    assert origin_qwen.live_stream == HEAD
    assert origin_qwen.reference is head_initial
    assert origin_qwen.stage == "origin_qwen"
    assert origin_qwen.origin_stream == HEAD

    alternate_text = coordinator.advance_after_failure(origin_qwen)
    assert alternate_text is not None
    assert alternate_text.live_stream == CHEST
    assert alternate_text.reference is chest_initial
    assert alternate_text.stage == "alternate_text"
    assert alternate_text.origin_stream == HEAD

    alternate_qwen = coordinator.advance_after_failure(alternate_text)
    assert alternate_qwen is not None
    assert alternate_qwen.live_stream == CHEST
    assert alternate_qwen.reference is chest_initial
    assert alternate_qwen.stage == "alternate_qwen"
    assert alternate_qwen.origin_stream == HEAD

    assert coordinator.advance_after_failure(alternate_qwen) is None


def test_coordinator_uses_cross_camera_prompt_when_alternate_has_no_initial() -> None:
    head_initial = reference(HEAD, marker=1)
    coordinator = DualCameraFailoverCoordinator(
        (HEAD, CHEST),
        {HEAD: head_initial},
    )
    first = coordinator.start()
    coordinator.mark_success(first)

    origin_text = coordinator.advance_after_failure(first)
    assert origin_text is not None
    origin_qwen = coordinator.advance_after_failure(origin_text)
    assert origin_qwen is not None
    alternate_text = coordinator.advance_after_failure(origin_qwen)
    assert alternate_text is not None
    assert alternate_text.live_stream == CHEST
    assert alternate_text.reference is head_initial
    assert alternate_text.reference.stream_name == HEAD
    assert alternate_text.stage == "alternate_text"
    alternate_qwen = coordinator.advance_after_failure(alternate_text)
    assert alternate_qwen is not None
    assert alternate_qwen.live_stream == CHEST
    assert alternate_qwen.reference is head_initial
    assert alternate_qwen.stage == "alternate_qwen"


def test_coordinator_success_resets_cycle_for_repeated_switching() -> None:
    head_initial = reference(HEAD, marker=1)
    chest_initial = reference(CHEST, marker=2)
    coordinator = DualCameraFailoverCoordinator(
        (HEAD, CHEST),
        {HEAD: head_initial, CHEST: chest_initial},
    )
    head = coordinator.start()
    coordinator.mark_success(head)
    origin_text = coordinator.advance_after_failure(head)
    assert origin_text is not None
    origin_qwen = coordinator.advance_after_failure(origin_text)
    assert origin_qwen is not None
    chest = coordinator.advance_after_failure(origin_qwen)
    assert chest is not None and chest.live_stream == CHEST

    coordinator.mark_success(chest)
    chest_text = coordinator.advance_after_failure(chest)

    assert chest_text is not None
    assert chest_text.live_stream == CHEST
    assert chest_text.reference is chest_initial
    assert chest_text.stage == "origin_text"
    assert chest_text.origin_stream == CHEST
    assert chest_text.attempt_id == 5


def test_dual_calibrations_use_each_saved_stream_and_chest_minus_three_pitch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = {
        HEAD: SimpleNamespace(width=640, height=480, fx=400.0, fy=401.0, cx=320.0, cy=240.0),
        CHEST: SimpleNamespace(width=640, height=480, fx=607.0, fy=608.0, cx=313.0, cy=254.0),
    }
    monkeypatch.setattr(dual_servo, "load_camera_intrinsics", lambda _path: saved)
    config = SimpleNamespace(
        camera_intrinsics_path="ignored.json",
        dual_head_camera_stream=HEAD,
        dual_chest_camera_stream=CHEST,
        camera_height_m=1.2,
        camera_pitch_deg=-38.0,
        camera_roll_deg=0.0,
        camera_yaw_deg=0.0,
        camera_forward_offset_m=0.0,
        camera_lateral_offset_m=0.0,
        dual_chest_camera_height_m=1.0,
        dual_chest_camera_pitch_deg=-3.0,
        dual_chest_camera_roll_deg=0.0,
        dual_chest_camera_yaw_deg=0.0,
        dual_chest_camera_forward_offset_m=0.0,
        dual_chest_camera_lateral_offset_m=0.0,
    )

    calibrations = dual_calibrations_from_config(config)

    assert calibrations[HEAD].fx == 400.0
    assert calibrations[HEAD].camera_pitch_deg == -38.0
    assert calibrations[CHEST].fx == 607.0
    assert calibrations[CHEST].camera_height_m == 1.0
    assert calibrations[CHEST].camera_pitch_deg == -3.0


def test_dual_grounding_runs_four_requests_concurrently_and_returns_both(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(4, timeout=2.0)
    calls: list[tuple[str, str]] = []

    class Client:
        def run(self, **kwargs):
            image_path = Path(kwargs["image_paths"][0])
            calls.append((image_path.parent.name, kwargs["schema_filename"]))
            barrier.wait()
            if kwargs["schema_filename"] == "target_schema.json":
                return target_payload()
            return table_payload()

    calibrations = {
        HEAD: RawServoCalibration(6, 4, 100.0, 101.0, 2.5, 1.5),
        CHEST: RawServoCalibration(6, 4, 110.0, 111.0, 2.5, 1.5, camera_pitch_deg=-3.0),
    }
    config = SimpleNamespace(task="align to basket")

    references, errors = ground_dual_raw_servo_references(
        config,
        {HEAD: rgbd_snapshot(HEAD, marker=1), CHEST: rgbd_snapshot(CHEST, marker=2)},
        calibrations,
        tmp_path,
        client_factory=Client,
    )

    assert set(references) == {HEAD, CHEST}
    assert errors == {}
    assert len(calls) == 4
    assert references[HEAD].stream_name == HEAD
    assert references[CHEST].stream_name == CHEST
    assert (tmp_path / HEAD / "initial_rgb.png").is_file()
    assert (tmp_path / CHEST / "initial_depth_raw.png").is_file()


def test_dual_grounding_isolates_one_camera_failure(tmp_path: Path) -> None:
    class Client:
        def run(self, **kwargs):
            image_path = Path(kwargs["image_paths"][0])
            if image_path.parent.name == CHEST:
                raise RuntimeError("chest grounding failed")
            if kwargs["schema_filename"] == "target_schema.json":
                return target_payload()
            return table_payload()

    calibration = RawServoCalibration(6, 4, 100.0, 101.0, 2.5, 1.5)
    references, errors = ground_dual_raw_servo_references(
        SimpleNamespace(task="align"),
        {HEAD: rgbd_snapshot(HEAD), CHEST: rgbd_snapshot(CHEST)},
        {HEAD: calibration, CHEST: calibration},
        tmp_path,
        client_factory=Client,
    )

    assert set(references) == {HEAD}
    assert "chest grounding failed" in errors[CHEST]


def test_dual_grounding_stops_waiting_after_one_view_grace_expires(
    tmp_path: Path,
) -> None:
    release_chest = threading.Event()

    class Client:
        def run(self, **kwargs):
            image_path = Path(kwargs["image_paths"][0])
            if image_path.parent.name == CHEST:
                release_chest.wait(timeout=2.0)
            if kwargs["schema_filename"] == "target_schema.json":
                return target_payload()
            return table_payload()

    calibration = RawServoCalibration(6, 4, 100.0, 101.0, 2.5, 1.5)
    started_at = time.monotonic()
    references, errors = ground_dual_raw_servo_references(
        SimpleNamespace(task="align"),
        {HEAD: rgbd_snapshot(HEAD), CHEST: rgbd_snapshot(CHEST)},
        {HEAD: calibration, CHEST: calibration},
        tmp_path,
        client_factory=Client,
        initialization_grace_s=0.05,
    )
    elapsed = time.monotonic() - started_at
    release_chest.set()

    assert set(references) == {HEAD}
    assert "exceeded 0.05s" in errors[CHEST]
    assert "process terminated" in errors[CHEST]
    assert elapsed < 0.5


def test_dual_grounding_terminates_slow_camera_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_codex = tmp_path / "fake_codex.py"
    slow_finished = tmp_path / "slow_finished.txt"
    fake_codex.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import json",
                "from pathlib import Path",
                "import sys",
                "import time",
                "if sys.argv[1:3] == ['login', 'status']:",
                "    print('Logged in with ChatGPT')",
                "    raise SystemExit(0)",
                f"if Path.cwd().name == {CHEST!r}:",
                "    time.sleep(10.0)",
                f"    Path({str(slow_finished)!r}).write_text('not terminated')",
                "schema_path = Path(sys.argv[sys.argv.index('--output-schema') + 1])",
                "if schema_path.name == 'target_schema.json':",
                f"    print({json.dumps(target_payload())!r})",
                "else:",
                f"    print({json.dumps(table_payload())!r})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    monkeypatch.setenv("CODEX_BIN", str(fake_codex))
    config = SimpleNamespace(
        task="align",
        vision_backend="codex",
        model="test-model",
        reasoning_effort="low",
        codex_fast=False,
        codex_timeout_seconds=20.0,
        qwenvl_model="unused",
        qwenvl_base_url="unused",
        qwenvl_thinking_budget=1,
    )
    calibration = RawServoCalibration(6, 4, 100.0, 101.0, 2.5, 1.5)

    started_at = time.monotonic()
    references, errors = ground_dual_raw_servo_references(
        config,
        {HEAD: rgbd_snapshot(HEAD), CHEST: rgbd_snapshot(CHEST)},
        {HEAD: calibration, CHEST: calibration},
        tmp_path / "run",
        initialization_grace_s=0.2,
    )
    elapsed = time.monotonic() - started_at
    time.sleep(0.2)

    assert set(references) == {HEAD}
    assert "process terminated" in errors[CHEST]
    assert elapsed < 8.0
    assert not slow_finished.exists()


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("inf"), float("nan")])
def test_dual_grounding_rejects_invalid_initialization_grace(
    tmp_path: Path, invalid: float
) -> None:
    with pytest.raises(ValueError, match="initialization grace"):
        ground_dual_raw_servo_references(
            SimpleNamespace(task="align"),
            {HEAD: rgbd_snapshot(HEAD)},
            {HEAD: RawServoCalibration(6, 4, 100.0, 101.0, 2.5, 1.5)},
            tmp_path,
            client_factory=lambda: object(),
            initialization_grace_s=invalid,
        )


def test_latest_reference_gate_updates_only_every_fifth_complete_frame() -> None:
    gate = LatestReferenceGate(interval_frames=5, min_confidence=0.35)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    common = dict(
        stream_name=HEAD,
        rgb=rgb,
        camera_timestamp=12.0,
        target_prompt="blue basket",
        target_bbox_xyxy=(160.0, 120.0, 320.0, 360.0),
        table_bboxes_xyxy=((20.0, 200.0, 620.0, 470.0),),
        target_confidence=0.9,
        target_geometry_valid=True,
        table_geometry_valid=True,
    )

    skipped, reason = gate.consider(frame_index=4, **common)
    accepted, accepted_reason = gate.consider(frame_index=5, **common)

    assert skipped is None
    assert reason == "interval"
    assert accepted is not None
    assert accepted.kind == "latest"
    assert accepted.target_bbox == pytest.approx((250.0, 250.0, 500.0, 750.0))
    assert accepted.table_bboxes[0] == pytest.approx(
        (31.25, 416.6666667, 968.75, 979.1666667)
    )
    assert accepted_reason == "accepted"


def test_latest_reference_gate_does_not_treat_frame_zero_as_fifth_frame() -> None:
    gate = LatestReferenceGate(interval_frames=5, min_confidence=0.35)

    candidate, reason = gate.consider(
        frame_index=0,
        stream_name=HEAD,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        camera_timestamp=12.0,
        target_prompt="blue basket",
        target_bbox_xyxy=(160.0, 120.0, 320.0, 360.0),
        table_bboxes_xyxy=((20.0, 200.0, 620.0, 470.0),),
        target_confidence=0.9,
        target_geometry_valid=True,
        table_geometry_valid=True,
    )

    assert candidate is None
    assert reason == "interval"


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"table_bboxes_xyxy": ()}, "missing_table"),
        ({"target_geometry_valid": False}, "invalid_target_geometry"),
        ({"table_geometry_valid": False}, "invalid_table_geometry"),
        ({"target_confidence": 0.2}, "low_confidence"),
    ],
)
def test_latest_reference_gate_rejects_incomplete_candidates(
    overrides: dict[str, object],
    expected_reason: str,
) -> None:
    gate = LatestReferenceGate(interval_frames=5, min_confidence=0.35)
    values: dict[str, object] = {
        "frame_index": 5,
        "stream_name": HEAD,
        "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "camera_timestamp": 12.0,
        "target_prompt": "blue basket",
        "target_bbox_xyxy": (160.0, 120.0, 320.0, 360.0),
        "table_bboxes_xyxy": ((20.0, 200.0, 620.0, 470.0),),
        "target_confidence": 0.9,
        "target_geometry_valid": True,
        "table_geometry_valid": True,
    }
    values.update(overrides)

    candidate, reason = gate.consider(**values)

    assert candidate is None
    assert reason == expected_reason



def test_latest_reference_gate_accepts_target_near_image_top() -> None:
    gate = LatestReferenceGate(interval_frames=5, min_confidence=0.35)

    candidate, reason = gate.consider(
        frame_index=5,
        stream_name=HEAD,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        camera_timestamp=12.0,
        target_prompt="blue basket",
        target_bbox_xyxy=(160.0, 0.0, 320.0, 50.0),
        table_bboxes_xyxy=((20.0, 200.0, 620.0, 470.0),),
        target_confidence=0.9,
        target_geometry_valid=True,
        table_geometry_valid=True,
    )

    assert candidate is not None
    assert reason == "accepted"
    assert candidate.target_bbox == pytest.approx(
        (250.0, 0.0, 500.0, 104.1666667)
    )

def test_async_dual_reference_updater_validates_target_and_desk() -> None:
    target_embedding = object()
    desk_embedding = object()

    class Encoder:
        def extract(self, _request):
            return dual_servo.EncodedDualReference(
                target_embedding=target_embedding,
                surface_embedding=desk_embedding,
                target_confidence=0.91,
                target_iou=0.92,
                surface_confidence=0.93,
                surface_iou=0.94,
            )

    updater = dual_servo.AsyncDualReferenceUpdater(
        Encoder,
        min_confidence=0.35,
        min_iou=0.5,
    )
    request = dual_servo.DualReferenceUpdateRequest(
        generation=3,
        frame_index=5,
        reference=reference(HEAD, kind="latest", marker=5),
    )
    try:
        updater.submit(request)
        deadline = time.monotonic() + 1.0
        result = None
        while result is None and time.monotonic() < deadline:
            result = updater.poll_latest()
            time.sleep(0.001)
    finally:
        updater.close()

    assert result is not None
    assert result.accepted
    assert result.reason == "accepted"
    assert result.stream_name == HEAD
    assert result.frame_index == 5
    assert result.target_embedding is target_embedding
    assert result.surface_embedding is desk_embedding


def test_dual_reference_updater_text_mode_validates_only_desk() -> None:
    updater = dual_servo.AsyncDualReferenceUpdater(
        lambda: None,
        min_confidence=0.35,
        min_iou=0.5,
    )
    request = dual_servo.DualReferenceUpdateRequest(
        generation=3,
        frame_index=5,
        reference=reference(HEAD, kind="latest", marker=5),
        require_target_embedding=False,
    )
    encoded = dual_servo.EncodedDualReference(
        target_embedding=object(),
        surface_embedding=object(),
        target_confidence=0.0,
        target_iou=0.0,
        surface_confidence=0.93,
        surface_iou=0.94,
    )
    try:
        result = updater._validate(request, encoded)
    finally:
        updater.close()

    assert result.accepted
    assert not result.target_validated
    assert result.target_confidence == 0.0
    assert result.surface_confidence == pytest.approx(0.93)


def test_qwen_fallback_forces_8b_model_and_uses_current_camera_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_ground(
        config,
        rgb_path,
        output_dir,
        *,
        client_factory,
        calibration,
    ):
        captured.update(
            config=config,
            rgb_path=Path(rgb_path),
            output_dir=Path(output_dir),
            client_factory=client_factory,
            calibration=calibration,
        )
        return (
            SimpleNamespace(
                target_prompt="blue basket",
                target_bbox=(300.0, 250.0, 600.0, 700.0),
            ),
            ((25.0, 100.0, 975.0, 900.0),),
        )

    monkeypatch.setattr(
        dual_servo,
        "ground_raw_servo_references",
        fake_ground,
    )
    config = SimpleNamespace(
        task="align",
        dual_qwenvl_fallback_model="qwen3-vl-8b-instruct",
        qwenvl_base_url="https://example.invalid/v1",
        qwenvl_thinking_budget=500,
        codex_timeout_seconds=600.0,
    )
    snapshot = rgbd_snapshot(HEAD, marker=7)
    calibration = RawServoCalibration(6, 4, 100.0, 101.0, 2.5, 1.5)

    result = dual_servo.ground_qwen_fallback_reference(
        config, HEAD, snapshot, calibration, tmp_path / "origin_qwen_03"
    )

    qwen_config = captured["config"]
    assert qwen_config.vision_backend == "qwenvl"
    assert qwen_config.qwenvl_model == "qwen3-vl-8b-instruct"
    assert captured["calibration"] is calibration
    assert captured["rgb_path"] == tmp_path / "origin_qwen_03/qwen_reference_rgb.png"
    assert result.stream_name == HEAD
def worker_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        task="align",
        output_root=str(tmp_path),
        dual_head_camera_stream=HEAD,
        dual_chest_camera_stream=CHEST,
        raw_reference_update_interval_frames=5,
        raw_reference_update_min_confidence=0.35,
        dual_match_tolerance_frames=30,
        dual_qwenvl_fallback_model="qwen3-vl-8b-instruct",
        qwenvl_base_url="https://example.invalid/v1",
        qwenvl_thinking_budget=500,
        codex_timeout_seconds=600.0,
        raw_servo_hz=10000.0,
        camera_host="localhost",
        camera_port=5555,
        camera_timeout_ms=100,
        camera_intrinsics_path="ignored.json",
        raw_yoloe_model_path="model.pt",
        raw_yoloe_confidence=0.25,
        raw_yoloe_imgsz=640,
        raw_yoloe_device="0",
    )


def large_snapshot(stream_name: str, *, marker: int, timestamp: float) -> AlignedRGBDSnapshot:
    return AlignedRGBDSnapshot(
        rgb=np.full((480, 640, 3), marker, dtype=np.uint8),
        depth_raw=np.full((480, 640), 1000, dtype=np.uint16),
        fx=607.0,
        fy=608.0,
        cx=320.0,
        cy=240.0,
        depth_scale_m=0.001,
        depth_aligned_to=stream_name,
        depth_source=None,
        timestamp=timestamp,
    )


class FakeDualCamera:
    def __init__(self) -> None:
        self.capture_count = 0
        self.closed = False

    def capture(self) -> DualRGBDCapture:
        self.capture_count += 1
        marker = min(self.capture_count, 250)
        return DualRGBDCapture(
            snapshots={
                HEAD: large_snapshot(HEAD, marker=marker, timestamp=float(self.capture_count)),
                CHEST: large_snapshot(CHEST, marker=marker, timestamp=float(self.capture_count)),
            },
            errors={},
        )

    def close(self) -> None:
        self.closed = True


def worker_references() -> dict[str, DualCameraReference]:
    return {
        HEAD: DualCameraReference(
            HEAD,
            np.full((480, 640, 3), 1, dtype=np.uint8),
            "blue basket",
            (300.0, 250.0, 600.0, 700.0),
            ((25.0, 100.0, 975.0, 900.0),),
            1.0,
            "initial",
        ),
        CHEST: DualCameraReference(
            CHEST,
            np.full((480, 640, 3), 2, dtype=np.uint8),
            "blue basket",
            (300.0, 250.0, 600.0, 700.0),
            ((25.0, 100.0, 975.0, 900.0),),
            1.0,
            "initial",
        ),
    }


def worker_qwen_reference(
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
        target_bbox=(300.0, 250.0, 600.0, 700.0),
        table_bboxes=((25.0, 100.0, 975.0, 900.0),),
        camera_timestamp=snapshot.timestamp,
        kind="qwen",
    )


def run_worker_with_tracker(
    tmp_path: Path,
    tracker: object,
    *,
    stop_event: threading.Event | None = None,
    diagnostics: AsyncFrameDiagnosticsWriter | None = None,
    table_required: Callable[[], bool] | None = None,
    latest_reference_updater_factory: Callable[[], object] | None = None,
    qwen_reference_factory: Callable[..., DualCameraReference] = (
        worker_qwen_reference
    ),
) -> tuple[list[RawServoEvent], FakeDualCamera]:
    requests: queue.Queue[int | None] = queue.Queue()
    requests.put(1)
    requests.put(None)
    events: queue.Queue[RawServoEvent] = queue.Queue()
    gate = GenerationGate()
    gate.activate(1)
    camera = FakeDualCamera()
    calibrations = {
        HEAD: RawServoCalibration(640, 480, 607.0, 608.0, 320.0, 240.0, camera_pitch_deg=0.0),
        CHEST: RawServoCalibration(640, 480, 607.0, 608.0, 320.0, 240.0, camera_pitch_deg=-3.0),
    }
    run_dual_raw_servo_worker(
        worker_config(tmp_path),
        requests,
        events,
        gate,
        stop_event or threading.Event(),
        diagnostics=diagnostics,
        camera_factory=lambda: camera,
        tracker_factory=lambda: tracker,
        latest_reference_updater_factory=latest_reference_updater_factory,
        qwen_reference_factory=qwen_reference_factory,
        calibration_factory=lambda _config: calibrations,
        reference_factory=lambda *_args, **_kwargs: (worker_references(), {}),
        table_required=table_required,
    )
    emitted: list[RawServoEvent] = []
    while not events.empty():
        emitted.append(events.get_nowait())
    return emitted, camera


def test_worker_gives_each_attempt_thirty_frames_before_failover_exit(
    tmp_path: Path,
) -> None:
    class MissingTracker:
        def __init__(self) -> None:
            self.starts: list[int | str] = []

        def start(self, reference_rgb, **_kwargs):
            self.starts.append(int(reference_rgb[0, 0, 0]))
            return {"reference_boxes_xyxy": [], "reference_class_ids": []}

        def start_text(self, _reference_rgb, *, target_prompt, **_kwargs):
            self.starts.append(f"hybrid:{target_prompt}")
            return {"prompt_mode": "target_text_surface_visual"}

        def start_all_text(self, *, target_prompt, surface_prompt):
            self.starts.append(f"all_text:{target_prompt}:{surface_prompt}")
            return {"prompt_mode": "target_text_surface_text"}

        def track(self, _rgb):
            return []

    tracker = MissingTracker()

    emitted, camera = run_worker_with_tracker(tmp_path, tracker)

    invalid = [event for event in emitted if event.kind == "invalid"]
    switching = [event for event in emitted if event.kind == "switching"]
    errors = [event for event in emitted if event.kind == "error"]
    assert len(invalid) == 145
    assert [
        event.details["match_invalid_frames"] for event in invalid[28::29]
    ] == [29, 29, 29, 29, 29]
    assert len(switching) == 4
    assert [event.details["live_stream"] for event in switching] == [
        HEAD, HEAD, CHEST, CHEST
    ]
    assert tracker.starts == [
        1,
        "hybrid:blue basket",
        62,
        "all_text:blue basket:desk",
        123,
    ]
    assert len(errors) == 1
    assert "both-camera Qwen failover exhausted" in (errors[0].error or "")
    diagnostic_indices = [
        event.frame.frame_index for event in emitted if event.frame is not None
    ]
    assert diagnostic_indices == list(range(150))
    summary_path = next(tmp_path.glob("dual_raw_yoloe_*")) / "initial_reference_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["selected_initial_stream"] == HEAD
    assert summary["streams"][HEAD]["eligible"] is True
    assert summary["streams"][CHEST]["eligible"] is True
    assert camera.capture_count == 153
    assert camera.closed


def test_worker_switch_boundaries_flush_contiguous_async_diagnostics(
    tmp_path: Path,
) -> None:
    class MissingTracker:
        def start(self, _reference_rgb, **_kwargs):
            return {"reference_boxes_xyxy": [], "reference_class_ids": []}

        def start_text(self, _reference_rgb, **_kwargs):
            return {"prompt_mode": "target_text_surface_visual"}

        def start_all_text(self, **_kwargs):
            return {"prompt_mode": "target_text_surface_text"}

        def track(self, _rgb):
            return []

    recorded_indices: list[int] = []

    class RecordingWriter:
        def __init__(self, _output_dir: str | Path):
            pass

        def write(self, frame: DetectionFrameData, **_kwargs) -> None:
            recorded_indices.append(frame.frame_index)

    diagnostics = AsyncFrameDiagnosticsWriter(
        logger=lambda _message: None,
        writer_factory=RecordingWriter,
    )
    emitted, _camera = run_worker_with_tracker(
        tmp_path,
        MissingTracker(),
        diagnostics=diagnostics,
    )
    runtime = RawServoRuntime(
        BasePosePlannerConfig(
            task="align",
            mode="dual_raw_yoloe_servo",
            output_root=str(tmp_path),
        ),
        publish=lambda _message: None,
        logger=lambda _message: None,
        diagnostics=diagnostics,
    )
    runtime.handle_key("n", now=1.0)
    for offset, event in enumerate(emitted, start=1):
        runtime.accept_event(event, now=1.0 + offset * 0.001)
    diagnostics.close(drain=True)

    assert recorded_indices == list(range(150))


def _valid_instances() -> list[TrackedInstance]:
    target_mask = np.zeros((480, 640), dtype=bool)
    target_mask[180:340, 220:440] = True
    table_mask = np.zeros((480, 640), dtype=bool)
    table_mask[120:360, 100:540] = True
    return [
        TrackedInstance(1, 0, "blue basket", 0.9, (220.0, 180.0, 440.0, 340.0), target_mask),
        TrackedInstance(2, 1, "desk", 0.8, (100.0, 120.0, 540.0, 360.0), table_mask),
    ]


def test_worker_allows_table_loss_after_required_visual_phases(
    tmp_path: Path,
) -> None:
    stop_event = threading.Event()

    class TableThenTargetOnlyTracker:
        def __init__(self) -> None:
            self.calls = 0

        def start(self, _reference_rgb, **_kwargs):
            return {"reference_boxes_xyxy": [], "reference_class_ids": []}

        def track(self, _rgb):
            self.calls += 1
            instances = _valid_instances()
            if self.calls == 1:
                return instances
            stop_event.set()
            return [instances[0]]

    emitted, _camera = run_worker_with_tracker(
        tmp_path,
        TableThenTargetOnlyTracker(),
        stop_event=stop_event,
        table_required=lambda: False,
    )

    initialized = [event for event in emitted if event.kind == "initialized"]
    observations = [event for event in emitted if event.kind == "observation"]
    assert len(initialized) == 1
    assert initialized[0].observation is not None
    assert initialized[0].observation.table is not None
    assert len(observations) == 1
    assert observations[0].observation is not None
    assert observations[0].observation.table is None
    assert observations[0].details["surface_track_id"] is None
    assert observations[0].details["latest_reference_update"] == "interval"
    assert not [event for event in emitted if event.kind == "invalid"]


def test_worker_still_requires_table_during_initialization(
    tmp_path: Path,
) -> None:
    stop_event = threading.Event()

    class InitialTargetOnlyTracker:
        def start(self, _reference_rgb, **_kwargs):
            return {"reference_boxes_xyxy": [], "reference_class_ids": []}

        def track(self, _rgb):
            stop_event.set()
            return [_valid_instances()[0]]

    emitted, _camera = run_worker_with_tracker(
        tmp_path,
        InitialTargetOnlyTracker(),
        stop_event=stop_event,
        table_required=lambda: False,
    )

    assert not [event for event in emitted if event.kind == "initialized"]
    invalid = [event for event in emitted if event.kind == "invalid"]
    assert len(invalid) == 1
    assert invalid[0].error == "missing tracked table"



def test_worker_origin_qwen_uses_fresh_same_camera_frame(tmp_path: Path) -> None:
    stop_event = threading.Event()

    class RecoverOnQwenTracker:
        def __init__(self) -> None:
            self.attempt = 0
            self.calls = 0
            self.starts: list[int | str] = []

        def _begin(self, marker: int | str):
            self.attempt += 1
            self.calls = 0
            self.starts.append(marker)
            return {"reference_boxes_xyxy": [], "reference_class_ids": []}

        def start(self, reference_rgb, **_kwargs):
            return self._begin(int(reference_rgb[0, 0, 0]))

        def start_text(self, _reference_rgb, **_kwargs):
            return self._begin("hybrid")

        def start_all_text(self, **_kwargs):
            return self._begin("all_text")

        def track(self, _rgb):
            self.calls += 1
            if self.attempt == 1 and self.calls <= 6:
                return _valid_instances()
            if self.attempt == 3:
                stop_event.set()
                return _valid_instances()
            return []

    tracker = RecoverOnQwenTracker()

    emitted, _camera = run_worker_with_tracker(
        tmp_path,
        tracker,
        stop_event=stop_event,
    )

    initialized = [event for event in emitted if event.kind == "initialized"]
    switching = [event for event in emitted if event.kind == "switching"]
    assert len(initialized) == 2
    assert len(switching) == 2
    assert tracker.starts == [1, "hybrid", 68]
    assert switching[0].details["reference_kind"] == "initial"
    assert switching[0].details["reference_source_stream"] == HEAD
    assert switching[0].details["prompt_mode"] == "target_text_surface_visual"
    assert switching[1].details["failover_stage"] == "origin_qwen"
    qwen_initialized = initialized[-1]
    assert qwen_initialized.details["live_stream"] == HEAD
    assert qwen_initialized.details["reference_kind"] == "qwen"
    assert qwen_initialized.details["reference_source_stream"] == HEAD
    assert qwen_initialized.details["prompt_mode"] == "visual"


def test_worker_applies_saved_reference_before_and_during_origin_text(
    tmp_path: Path,
) -> None:
    stop_event = threading.Event()

    class Updater:
        def __init__(self) -> None:
            self.pending: list[object] = []
            self.submitted: list[tuple[str, int]] = []
            self.closed = False

        def submit(self, request) -> None:
            stream = request.reference.stream_name
            frame = request.frame_index
            self.submitted.append((stream, frame))
            self.pending.append(
                SimpleNamespace(
                    generation=request.generation,
                    frame_index=frame,
                    stream_name=stream,
                    accepted=True,
                    target_validated=request.require_target_embedding,
                    reference=request.reference,
                    target_embedding=f"target:{stream}:{frame}",
                    surface_embedding=f"desk:{stream}:{frame}",
                    target_confidence=0.91,
                    target_iou=0.92,
                    surface_confidence=0.93,
                    surface_iou=0.94,
                    reason="accepted",
                )
            )

        def poll_latest(self):
            if not self.pending:
                return None
            return self.pending.pop(0)

        def close(self) -> None:
            self.closed = True

    class Tracker:
        def __init__(self) -> None:
            self.attempt = 0
            self.calls = 0
            self.starts: list[tuple[str, str]] = []
            self.installed: list[tuple[str, object, object]] = []

        def _begin(self, mode: str, surface_prompt: str):
            self.attempt += 1
            self.calls = 0
            self.starts.append((mode, surface_prompt))
            return {"reference_boxes_xyxy": [], "reference_class_ids": []}

        def start(self, _reference_rgb, *, surface_prompt, **_kwargs):
            return self._begin("visual", surface_prompt)

        def start_text(self, _reference_rgb, *, surface_prompt, **_kwargs):
            return self._begin("text", surface_prompt)

        def track(self, _rgb):
            self.calls += 1
            if self.attempt == 1 and self.calls > 7:
                return []
            return _valid_instances()

        def install_reference_embeddings(
            self,
            *,
            target_embedding,
            surface_embedding,
        ) -> None:
            self.installed.append(
                ("visual", target_embedding, surface_embedding)
            )

        def install_surface_embedding(self, surface_embedding) -> None:
            self.installed.append(("text", None, surface_embedding))
            stop_event.set()

    updater = Updater()
    tracker = Tracker()

    emitted, _camera = run_worker_with_tracker(
        tmp_path,
        tracker,
        stop_event=stop_event,
        latest_reference_updater_factory=lambda: updater,
    )

    assert updater.submitted == [(HEAD, 5), (HEAD, 40)]
    assert updater.closed
    assert tracker.starts[:2] == [("visual", "desk"), ("text", "desk")]
    assert tracker.installed == [
        ("visual", f"target:{HEAD}:5", f"desk:{HEAD}:5"),
        ("text", None, f"desk:{HEAD}:40"),
    ]
    applied = [
        event
        for event in emitted
        if event.details.get("latest_reference_refresh", {}).get("status")
        == "applied"
    ]
    assert [event.details["live_stream"] for event in applied] == [HEAD, HEAD]
    assert [event.details["reference_kind"] for event in applied] == [
        "latest",
        "latest",
    ]
    assert [
        event.details["latest_reference_refresh"]["source_frame"]
        for event in applied
    ] == [5, 40]


def servo_observation() -> RawServoObservation:
    return RawServoObservation(
        target=TargetGeometry(
            forward_m=1.0,
            right_m=0.0,
            body_xyz_m=(1.0, 0.0, 0.8),
            valid_depth_pixels=100,
            valid_ratio=1.0,
            median_depth_m=1.0,
        ),
        table=TableGeometry(
            yaw_error_rad=0.2,
            line_length_m=0.8,
            inlier_count=100,
            residual_m=0.01,
            line_center_xy_m=(1.0, 0.0),
        ),
        camera_timestamp=1.0,
        target_track_id=1,
        surface_track_id=2,
    )


def test_runtime_switching_holds_zero_in_same_generation_and_rejects_old_attempt(
    tmp_path: Path,
) -> None:
    messages: list[str] = []
    runtime = RawServoRuntime(
        BasePosePlannerConfig(
            task="align",
            mode="dual_raw_yoloe_servo",
            output_root=str(tmp_path),
            raw_head_target_distance_m=0.95,
            raw_chest_target_distance_m=0.75,
        ),
        publish=messages.append,
        logger=lambda _message: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    assert runtime.accept_event(
        RawServoEvent(
            1,
            "initialized",
            observation=servo_observation(),
            output_dir=str(tmp_path),
            details={"attempt_id": 1, "live_stream": HEAD},
        ),
        now=1.1,
    )
    assert runtime.controller.phase in {
        ServoPhase.YAW_ALIGN,
        ServoPhase.YAW_TRIM,
    }
    assert runtime.controller.current.vx == 0.0
    assert runtime.controller.current.wz > 0.0
    assert runtime.controller.target_distance_m == pytest.approx(0.95)

    assert runtime.accept_event(
        RawServoEvent(
            1,
            "switching",
            output_dir=str(tmp_path),
            error="missing target",
            details={
                "attempt_id": 2,
                "live_stream": CHEST,
                "reference_source_stream": CHEST,
                "reference_kind": "initial",
            },
        ),
        now=1.2,
    )

    assert runtime.phase == "switching"
    assert runtime.generation == 1
    assert runtime.gate.is_active(1)
    assert runtime.controller.phase is ServoPhase.FORWARD_APPROACH
    assert runtime.controller.target_distance_m == pytest.approx(0.75)
    immediate = json.loads(messages[-1])
    assert immediate["action"] == "hold"
    assert immediate["velocity"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}

    runtime.publish_due(1.3)
    heartbeat = json.loads(messages[-1])
    assert heartbeat["action"] == "hold"
    assert heartbeat["velocity"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}

    assert not runtime.accept_event(
        RawServoEvent(
            1,
            "observation",
            observation=servo_observation(),
            details={"attempt_id": 1},
        ),
        now=1.31,
    )
    assert runtime.accept_event(
        RawServoEvent(
            1,
            "initialized",
            observation=servo_observation(),
            details={"attempt_id": 2},
        ),
        now=1.4,
    )
    assert runtime.phase == "aligning"
    assert runtime.controller.phase is ServoPhase.FORWARD_APPROACH
    assert runtime.controller.target_distance_m == pytest.approx(0.75)
    assert runtime.controller.current.vx > 0.0
    assert runtime.controller.current.vy == 0.0
    assert runtime.controller.current.wz == 0.0
    assert not runtime.accept_event(
        RawServoEvent(
            1,
            "observation",
            observation=servo_observation(),
            details={"attempt_id": 3},
        ),
        now=1.41,
    )


def test_runtime_budget_starts_at_detection_and_switch_does_not_extend_it(
    tmp_path: Path,
) -> None:
    runtime = RawServoRuntime(
        BasePosePlannerConfig(
            task="align",
            mode="dual_raw_yoloe_servo",
            output_root=str(tmp_path),
            raw_max_run_s=60.0,
        ),
        publish=lambda _message: None,
        logger=lambda _message: None,
    )
    runtime.handle_key("n", now=1.0)
    runtime.publish_due(999.0)
    assert runtime.phase == "inference"
    assert runtime.navigation_started_at is None
    runtime.accept_event(
        RawServoEvent(1, "detecting", details={"attempt_id": 1}),
        now=1000.0,
    )
    runtime.accept_event(
        RawServoEvent(1, "initialized", observation=servo_observation(), details={"attempt_id": 1}),
        now=1000.1,
    )
    runtime.accept_event(
        RawServoEvent(1, "switching", details={"attempt_id": 2}),
        now=1050.0,
    )
    runtime.accept_event(
        RawServoEvent(1, "initialized", observation=servo_observation(), details={"attempt_id": 2}),
        now=1050.1,
    )

    runtime.publish_due(1060.01)

    assert runtime.phase == "idle"
    assert not runtime.gate.is_active(1)


def test_diagnostics_record_dual_camera_attempt_provenance(tmp_path: Path) -> None:
    writer = FrameDiagnosticsWriter(tmp_path, review_stride=5)
    frame = DetectionFrameData(
        frame_index=1,
        camera_timestamp=2.0,
        rgb=np.zeros((4, 6, 3), dtype=np.uint8),
        camera_stream=CHEST,
        attempt_id=2,
        failover_stage="origin_text",
        reference_source_stream=HEAD,
        reference_kind="initial",
        prompt_mode="target_text_surface_visual",
    )

    writer.write(
        frame,
        controller_state={"phase": "yaw_align"},
        command={"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 0.15},
    )

    row = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text().strip())
    assert row["camera_stream"] == CHEST
    assert row["attempt_id"] == 2
    assert row["failover_stage"] == "origin_text"
    assert row["reference_source_stream"] == HEAD
    assert row["reference_kind"] == "initial"
    assert row["prompt_mode"] == "target_text_surface_visual"


def test_dual_raw_yoloe_mode_is_registered_with_expected_defaults() -> None:
    assert "dual_raw_yoloe_servo" in BASE_POSE_MODES
    planner = BasePosePlannerConfig(task="align", mode="dual_raw_yoloe_servo")
    launch = InferenceLaunchConfig(
        planner_input="base_pose",
        base_pose_mode="dual_raw_yoloe_servo",
    )

    assert planner.dual_head_camera_stream == HEAD
    assert planner.dual_chest_camera_stream == CHEST
    assert planner.dual_chest_camera_pitch_deg == -3.0
    assert planner.dual_match_tolerance_frames == 30
    assert planner.dual_initialization_grace_s == pytest.approx(30.0)
    assert planner.dual_qwenvl_fallback_model == "qwen3-vl-8b-instruct"
    assert planner.raw_head_target_distance_m == pytest.approx(0.9)
    assert planner.raw_chest_target_distance_m == pytest.approx(0.8)
    assert planner.raw_max_run_s == pytest.approx(180.0)
    assert planner.raw_post_stop_sample_s == pytest.approx(3.0)
    assert planner.raw_allow_missing_table == 0
    assert launch.base_pose_dual_head_camera_stream == HEAD
    assert launch.base_pose_dual_chest_camera_stream == CHEST
    assert launch.base_pose_dual_chest_camera_pitch_deg == -3.0
    assert launch.base_pose_dual_initialization_grace_s == pytest.approx(30.0)
    assert launch.base_pose_dual_qwenvl_fallback_model == "qwen3-vl-8b-instruct"
    assert launch.base_pose_raw_head_target_distance_m == pytest.approx(0.9)
    assert launch.base_pose_raw_chest_target_distance_m == pytest.approx(0.8)
    assert launch.base_pose_raw_max_run_s == pytest.approx(180.0)
    assert launch.base_pose_raw_post_stop_sample_s == pytest.approx(3.0)
    assert launch.base_pose_raw_allow_missing_table == 0


def test_dual_raw_yoloe_launch_uses_direct_dual_rgbd_and_raw_relay() -> None:
    config = InferenceLaunchConfig(
        planner_input="base_pose",
        base_pose_mode="dual_raw_yoloe_servo",
        base_pose_task="align to basket",
        camera_host="robot-camera",
        camera_port=5555,
        base_pose_dual_head_camera_stream=HEAD,
        base_pose_dual_chest_camera_stream=CHEST,
        base_pose_dual_chest_camera_pitch_deg=-3.0,
        base_pose_dual_match_tolerance_frames=30,
    )

    planner_command = build_planner_input_command(config, Path("/workspace/sonic"))
    relay_command = build_reasan_planner_command(config, Path("/workspace/sonic"))

    assert "--mode dual_raw_yoloe_servo" in planner_command
    assert "--camera-host robot-camera --camera-port 5555" in planner_command
    assert "--dual-head-camera-stream ego_view" in planner_command
    assert "--dual-chest-camera-stream chest_view" in planner_command
    assert "--dual-chest-camera-pitch-deg -3.0" in planner_command
    assert "--dual-match-tolerance-frames 30" in planner_command
    assert "--dual-initialization-grace-s 30.0" in planner_command
    assert "--dual-qwenvl-fallback-model " in planner_command
    assert "qwen3-vl-8b-instruct" in planner_command
    assert "--raw-head-target-distance-m 0.9" in planner_command
    assert "--raw-chest-target-distance-m 0.8" in planner_command
    assert "--raw-max-run-s 180.0" in planner_command
    assert "--raw-post-stop-sample-s 3.0" in planner_command
    assert "--raw-allow-missing-table 0" in planner_command
    assert ".venv_lingbot_depth" not in planner_command
    assert uses_base_pose_manual_keyboard(config)
    assert "--manual-source" in relay_command
    assert "--orientation-telemetry-output" in relay_command


def test_existing_raw_yoloe_launch_does_not_gain_dual_flags() -> None:
    config = InferenceLaunchConfig(
        planner_input="base_pose",
        base_pose_mode="raw_yoloe_servo",
        base_pose_task="align to basket",
    )

    command = build_planner_input_command(config, Path("/workspace/sonic"))

    assert "--mode raw_yoloe_servo" in command
    assert "--dual-head-camera-stream" not in command
    assert "--dual-chest-camera-stream" not in command
