"""Software-only tests for raw-depth YOLOE base-pose visual servo."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
import types

import numpy as np
import pytest

sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.scripts.base_pose_planner import BasePosePlannerConfig
from gear_sonic.scripts.launch_inference import (
    InferenceLaunchConfig,
    build_planner_input_command,
)
from gear_sonic.utils.inference.base_pose import (
    AlignedRGBDSnapshot,
    BasePoseCameraError,
    BasePoseValidationError,
)
from gear_sonic.utils.inference import base_pose_visual_servo as raw_servo
from gear_sonic.utils.inference.base_pose_visual_servo import (
    GenerationGate,
    RawServoCalibration,
    RawServoEvent,
    RawServoObservation,
    RawServoRuntime,
    ServoCommand,
    ServoPhase,
    AsyncTargetReferenceUpdater,
    EncodedTargetReference,
    TargetReferenceRequest,
    TargetReferenceUpdateGate,
    TableGeometry,
    TargetGeometry,
    TrackedInstance,
    VisualServoController,
    YoloePersistentTracker,
    RAW_SERVO_TARGET_SCHEMA,
    build_raw_servo_target_prompt,
    estimate_table_geometry,
    estimate_target_geometry,
    _observation,
    run_raw_servo_worker,
    validate_raw_servo_target,
)
from gear_sonic.utils.inference.base_pose_visual_servo_diagnostics import (
    AsyncFrameDiagnosticsWriter,
    DetectionFrameData,
)
from tools.yoloe26m.reference_pipeline import (
    GROUNDING_SCHEMA,
    GroundingValidationError,
    build_grounding_prompt,
)


def test_target_reference_gate_freezes_at_top_and_refreshes_immediately_after_recovery() -> None:
    gate = TargetReferenceUpdateGate(
        interval_frames=5,
        min_confidence=0.35,
        freeze_y2_px=75.0,
        resume_y2_px=110.0,
    )

    assert gate.consider(5, confidence=0.9, bbox_xyxy=(0.0, 0.0, 20.0, 74.0)) == (
        False,
        "vertical_danger",
    )
    assert gate.consider(6, confidence=0.9, bbox_xyxy=(0.0, 0.0, 20.0, 100.0)) == (
        False,
        "vertical_recovery_pending",
    )
    assert gate.consider(7, confidence=0.9, bbox_xyxy=(0.0, 0.0, 20.0, 110.0)) == (
        True,
        "vertical_recovered",
    )
    assert not gate.consider(
        8, confidence=0.9, bbox_xyxy=(0.0, 0.0, 20.0, 200.0)
    )[0]
    assert gate.consider(10, confidence=0.9, bbox_xyxy=(0.0, 0.0, 20.0, 200.0)) == (
        True,
        "interval",
    )
    assert gate.consider(15, confidence=0.3, bbox_xyxy=(0.0, 0.0, 20.0, 200.0)) == (
        False,
        "low_confidence",
    )


def test_async_reference_updater_is_nonblocking_and_keeps_latest_waiting_frame() -> None:
    release = threading.Event()
    started = threading.Event()
    encoded_frames: list[int] = []

    class Encoder:
        def extract(self, request: TargetReferenceRequest) -> EncodedTargetReference:
            encoded_frames.append(request.frame_index)
            if request.frame_index == 5:
                started.set()
                assert release.wait(2.0)
            return EncodedTargetReference(
                embedding=f"embedding-{request.frame_index}",
                confidence=0.9,
                bbox_xyxy=request.target_bbox_xyxy,
            )

    updater = AsyncTargetReferenceUpdater(
        Encoder,
        min_confidence=0.35,
        min_iou=0.5,
    )
    try:
        first = TargetReferenceRequest(
            generation=1,
            frame_index=5,
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            target_prompt="basket",
            target_bbox_xyxy=(0.0, 0.0, 4.0, 4.0),
        )
        started_at = time.monotonic()
        updater.submit(first)
        assert time.monotonic() - started_at < 0.05
        assert started.wait(1.0)
        updater.submit(replace(first, frame_index=10))
        updater.submit(replace(first, frame_index=15))
        release.set()

        deadline = time.monotonic() + 2.0
        latest = None
        while time.monotonic() < deadline:
            result = updater.poll_latest()
            if result is not None:
                latest = result
                if result.frame_index == 15:
                    break
            time.sleep(0.01)

        assert encoded_frames == [5, 15]
        assert latest is not None and latest.accepted
        assert latest.frame_index == 15
        assert latest.embedding == "embedding-15"
    finally:
        updater.close()


def test_async_reference_updater_rejects_failed_self_validation() -> None:
    class Encoder:
        def extract(self, request: TargetReferenceRequest) -> EncodedTargetReference:
            return EncodedTargetReference(
                embedding="candidate",
                confidence=0.2,
                bbox_xyxy=request.target_bbox_xyxy,
            )

    updater = AsyncTargetReferenceUpdater(
        Encoder,
        min_confidence=0.35,
        min_iou=0.5,
    )
    try:
        updater.submit(
            TargetReferenceRequest(
                generation=2,
                frame_index=5,
                rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                target_prompt="basket",
                target_bbox_xyxy=(0.0, 0.0, 4.0, 4.0),
            )
        )
        deadline = time.monotonic() + 1.0
        result = None
        while result is None and time.monotonic() < deadline:
            result = updater.poll_latest()
            time.sleep(0.005)
        assert result is not None and not result.accepted
        assert result.reason == "confidence_below_threshold"
        assert result.embedding is None
    finally:
        updater.close()


def test_tracker_target_embedding_swap_preserves_initial_table_and_predictor() -> None:
    torch = pytest.importorskip("torch")

    class Model:
        def __init__(self) -> None:
            self.predictor = object()
            self.installed = None

        def set_classes(self, classes, *, embeddings):
            self.installed = (classes, embeddings.clone())

    tracker = object.__new__(YoloePersistentTracker)
    tracker.model = Model()
    tracker.class_names = ("basket", "table")
    tracker._initial_surface_embedding = torch.tensor([[[3.0, 4.0]]])
    predictor = tracker.model.predictor

    tracker.install_target_embedding(torch.tensor([[[1.0, 2.0]]]))

    classes, embeddings = tracker.model.installed
    assert classes == ["basket", "table"]
    assert torch.equal(embeddings, torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    assert tracker.model.predictor is predictor


def test_worker_rolls_target_reference_and_freezes_it_during_vertical_recenter(
    tmp_path,
) -> None:
    stop_event = threading.Event()
    submitted_frames: list[int] = []
    pending_results: list[object] = []

    class Camera:
        def capture(self):
            return snapshot()

        def close(self):
            pass

    class Client:
        def run(self, **kwargs):
            if kwargs["schema_filename"] == "target_schema.json":
                return target_payload()
            return table_payload()

    class Updater:
        closed = False

        def submit(self, request):
            submitted_frames.append(request.frame_index)
            pending_results.append(
                types.SimpleNamespace(
                    generation=request.generation,
                    frame_index=request.frame_index,
                    accepted=True,
                    embedding=f"embedding-{request.frame_index}",
                    confidence=0.9,
                    iou=0.9,
                    reason="accepted",
                )
            )

        def poll_latest(self):
            if not pending_results:
                return None
            return pending_results.pop(0)

        def close(self):
            self.closed = True

    class Tracker:
        def __init__(self):
            self.calls = 0
            self.installed: list[str] = []

        def start(self, *_args, **_kwargs):
            return {
                "class_names": ["blue basket", "table"],
                "reference_boxes_xyxy": [
                    [192.0, 120.0, 384.0, 336.0],
                    [16.0, 48.0, 304.0, 432.0],
                    [336.0, 57.6, 624.0, 422.4],
                ],
                "reference_class_ids": [0, 1, 1],
            }

        def track(self, _rgb):
            self.calls += 1
            frame_index = self.calls - 1
            y2 = {2: 74.0, 3: 100.0, 4: 110.0, 5: 120.0}.get(
                frame_index, 300.0
            )
            y1 = max(0.0, y2 - 120.0)
            target_mask = np.zeros((480, 640), dtype=bool)
            target_mask[int(y1) : max(int(y1) + 1, int(y2)), 220:440] = True
            table_mask = np.zeros((480, 640), dtype=bool)
            table_mask[120:360, 100:540] = True
            if frame_index == 5:
                stop_event.set()
            return [
                TrackedInstance(
                    1,
                    0,
                    "blue basket",
                    0.9,
                    (220.0, y1, 440.0, y2),
                    target_mask,
                ),
                TrackedInstance(
                    2,
                    1,
                    "table",
                    0.8,
                    (100.0, 120.0, 540.0, 360.0),
                    table_mask,
                ),
            ]

        def install_target_embedding(self, embedding):
            self.installed.append(embedding)

    config = BasePosePlannerConfig(
        task="align to basket",
        output_root=str(tmp_path),
        camera_pitch_deg=0.0,
        raw_servo_hz=1000.0,
        raw_reference_update_interval_frames=1,
    )
    requests: queue.Queue[int | None] = queue.Queue()
    events: queue.Queue[RawServoEvent] = queue.Queue()
    generation_gate = GenerationGate()
    generation_gate.activate(1)
    requests.put(1)
    updater = Updater()
    tracker = Tracker()

    run_raw_servo_worker(
        config,
        requests,
        events,
        generation_gate,
        stop_event,
        camera_factory=Camera,
        client_factory=Client,
        tracker_factory=lambda: tracker,
        reference_updater_factory=lambda: updater,
    )

    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())
    by_frame = {
        item.frame.frame_index: item
        for item in emitted
        if item.frame is not None
    }
    assert submitted_frames == [1, 4, 5]
    assert tracker.installed == ["embedding-1", "embedding-4"]
    assert by_frame[2].details["target_reference_update_paused"] == (
        "vertical_danger"
    )
    assert by_frame[3].details["target_reference_update_paused"] == (
        "vertical_recovery_pending"
    )
    assert by_frame[4].details["target_reference_update_applied"]["source_frame"] == 1
    assert by_frame[4].details["target_reference_update_scheduled"]["trigger"] == (
        "vertical_recovered"
    )
    assert updater.closed


def snapshot(
    depth_raw: np.ndarray | None = None,
    *,
    fx: float = 607.878662,
    fy: float = 608.063232,
    cx: float = 319.858765,
    cy: float = 259.731140,
) -> AlignedRGBDSnapshot:
    if depth_raw is None:
        depth_raw = np.full((480, 640), 1000, dtype=np.uint16)
    return AlignedRGBDSnapshot(
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_raw=depth_raw,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        depth_scale_m=0.001,
        depth_aligned_to="ego_view",
        depth_source=None,
        timestamp=10.0,
    )


def observation(
    *,
    forward: float = 0.80,
    right: float = 0.0,
    yaw: float = 0.0,
    bbox: tuple[float, float, float, float] = (260.0, 100.0, 380.0, 300.0),
    include_table: bool = True,
    image_height: int = 480,
) -> RawServoObservation:
    return RawServoObservation(
        target=TargetGeometry(
            forward_m=forward,
            right_m=right,
            body_xyz_m=(forward, -right, 0.8),
            valid_depth_pixels=1000,
            valid_ratio=1.0,
            median_depth_m=1.0,
        ),
        table=(
            TableGeometry(
                yaw_error_rad=yaw,
                line_length_m=0.7,
                inlier_count=200,
                residual_m=0.005,
                line_center_xy_m=(1.0, 0.0),
            )
            if include_table
            else None
        ),
        camera_timestamp=10.0,
        target_track_id=1,
        surface_track_id=2 if include_table else None,
        target_bbox_xyxy=bbox,
        image_width=640,
        image_height=image_height,
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
        "limitations": "partially occluded",
    }


def table_payload() -> dict[str, object]:
    return {
        "status": "READY",
        "target": "table",
        "boxes": [
            {"bbox_2d": [25.0, 100.0, 475.0, 900.0], "confidence": 0.91},
            {"bbox_2d": [525.0, 120.0, 975.0, 880.0], "confidence": 0.84},
        ],
        "limitations": "",
    }


def test_target_schema_is_strict_and_contains_no_motion_plan() -> None:
    spec = validate_raw_servo_target(target_payload())
    assert spec.target_prompt == "blue plastic basket"

    invalid = target_payload()
    invalid["command_sequence"] = []
    with pytest.raises(BasePoseValidationError, match="invalid object schema"):
        validate_raw_servo_target(invalid)

    unsure = target_payload()
    unsure["status"] = "UNSURE"
    with pytest.raises(BasePoseValidationError, match="returned UNSURE"):
        validate_raw_servo_target(unsure)


def test_fixed_calibration_fov_transform_and_live_intrinsic_guard() -> None:
    calibration = RawServoCalibration()
    assert calibration.horizontal_fov_deg == pytest.approx(55.526508, abs=1e-5)
    assert calibration.vertical_fov_deg == pytest.approx(43.077882, abs=1e-5)
    calibration.validate_snapshot(snapshot())

    optical_axis = calibration.camera_to_body(np.array([[0.0, 0.0, 1.0]]))[0]
    assert optical_axis[0] == pytest.approx(math.cos(math.radians(25.0)))
    assert optical_axis[1] == pytest.approx(0.0)
    assert optical_axis[2] == pytest.approx(1.2 - math.sin(math.radians(25.0)))

    with pytest.raises(BasePoseCameraError, match="intrinsics differ"):
        calibration.validate_snapshot(snapshot(fx=611.0))


def test_target_uses_eroded_mask_depth_and_bbox_center_for_lateral_error() -> None:
    depth = np.full((480, 640), 1000, dtype=np.uint16)
    depth[200:205, 250:390] = 9000
    mask = np.zeros((480, 640), dtype=bool)
    mask[180:340, 220:440] = True
    calibration = RawServoCalibration(camera_pitch_deg=0.0)

    geometry = estimate_target_geometry(
        snapshot(depth),
        mask,
        calibration,
        bbox_xyxy=(300.0, 180.0, 460.0, 340.0),
    )

    assert geometry.median_depth_m == pytest.approx(1.0)
    assert geometry.valid_depth_pixels >= 200
    assert geometry.right_m == pytest.approx(
        (380.0 - calibration.cx) / calibration.fx, abs=1e-5
    )


def test_table_mask_depth_recovers_front_facing_horizontal_edge() -> None:
    mask = np.zeros((480, 640), dtype=bool)
    mask[120:360, 100:540] = True
    calibration = RawServoCalibration(camera_height_m=0.0, camera_pitch_deg=0.0)

    geometry = estimate_table_geometry(snapshot(), mask, calibration)

    assert geometry.line_length_m > 0.20
    assert geometry.inlier_count >= 50
    assert geometry.residual_m <= 0.02
    assert geometry.yaw_error_rad == pytest.approx(0.0, abs=0.03)


def test_controller_coarse_yaw_keeps_translation_zero() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    command = controller.update(
        observation(forward=1.0, right=0.2, yaw=math.radians(20.0)), now=1.1
    )
    assert command.vx == 0.0
    assert command.vy == 0.0
    assert command.wz == pytest.approx(0.05)


def test_controller_coarse_yaw_caps_heading_slew_at_point_one_five() -> None:
    controller = VisualServoController()
    controller.reset(1.0)

    commands = [
        controller.update(
            observation(yaw=math.radians(20.0)), now=1.1 + index * 0.1
        )
        for index in range(4)
    ]

    assert [command.wz for command in commands] == pytest.approx(
        [0.05, 0.10, 0.15, 0.15]
    )


@pytest.mark.parametrize(
    ("guard_fraction", "recovery_fraction"),
    [
        (0.0, 0.30),
        (-0.01, 0.30),
        (0.30, 0.30),
        (0.31, 0.30),
        (0.25, 0.50),
        (0.25, 0.51),
        (math.nan, 0.30),
        (0.25, math.inf),
    ],
)
def test_controller_rejects_invalid_horizontal_guard_and_recovery_fractions(
    guard_fraction: float, recovery_fraction: float
) -> None:
    with pytest.raises(ValueError, match="horizontal"):
        VisualServoController(
            horizontal_guard_fraction=guard_fraction,
            horizontal_recovery_fraction=recovery_fraction,
        )


def test_controller_nonzero_translation_preserves_direction_at_minimum_speed() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET
    command = controller.update(
        observation(forward=1.2, right=0.26666666666666666), now=1.1
    )

    assert math.hypot(command.vx, command.vy) == pytest.approx(0.30)
    assert command.vx > 0.0
    assert command.vy < 0.0
    assert command.vx / -command.vy == pytest.approx(1.25)
    assert command.wz == 0.0


def test_controller_minimum_speed_scaling_can_exceed_prefloor_lateral_limit() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.RECENTER

    command = controller.update(observation(right=1.0), now=1.1)

    assert command.velocity == pytest.approx((0.0, -0.30, 0.0))
    assert math.hypot(command.vx, command.vy) >= 0.30


def test_controller_accepts_configurable_prefloor_lateral_limit() -> None:
    controller = VisualServoController(max_lateral_speed_m_s=0.12)
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET

    command = controller.update(observation(forward=1.2, right=1.0), now=1.1)

    assert math.hypot(command.vx, command.vy) == pytest.approx(0.30)
    assert command.vx / -command.vy == pytest.approx(0.20 / 0.12)


@pytest.mark.parametrize("limit", [0.0, -0.1, math.nan, math.inf])
def test_controller_rejects_invalid_lateral_speed_limit(limit: float) -> None:
    with pytest.raises(ValueError, match="lateral speed"):
        VisualServoController(max_lateral_speed_m_s=limit)


def test_controller_linear_speed_floor_never_rounds_below_minimum() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET

    command = controller.update(observation(forward=0.10, right=-0.266), now=1.1)

    assert math.hypot(command.vx, command.vy) >= 0.30


def test_controller_ignores_accumulated_translation_and_yaw_for_termination() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET
    controller.cumulative_translation_m = 10.0
    controller.cumulative_yaw_rad = 10.0

    command = controller.update(observation(forward=1.0), now=1.1)

    assert not controller.terminal
    assert math.hypot(command.vx, command.vy) == pytest.approx(0.30)


def test_raw_servo_default_standoff_is_point_eight_meters(tmp_path) -> None:
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _: None,
        logger=lambda _: None,
    )
    runtime.controller.reset(1.0)
    runtime.controller.phase = ServoPhase.TRANSLATE_TARGET

    for index in range(5):
        command = runtime.controller.update(
            observation(forward=0.80, include_table=False),
            now=1.1 + index * 0.1,
        )

    assert command.velocity == (0.0, 0.0, 0.0)
    assert runtime.controller.terminal_reason == "aligned"


def test_raw_servo_runtime_uses_configured_prefloor_lateral_limit(tmp_path) -> None:
    runtime = RawServoRuntime(
        BasePosePlannerConfig(
            task="align", output_root=str(tmp_path), raw_max_lateral_speed_m_s=0.12
        ),
        publish=lambda _: None,
        logger=lambda _: None,
    )
    runtime.controller.reset(1.0)
    runtime.controller.phase = ServoPhase.TRANSLATE_TARGET

    command = runtime.controller.update(
        observation(forward=1.2, right=1.0), now=1.1
    )

    assert command.vx / -command.vy == pytest.approx(0.20 / 0.12)


def test_controller_default_completion_tolerances_are_ten_centimeters() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET

    for index in range(5):
        command = controller.update(
            observation(forward=0.90, right=0.10, include_table=False),
            now=1.1 + index * 0.1,
        )

    assert command.velocity == (0.0, 0.0, 0.0)
    assert controller.terminal_reason == "aligned"


@pytest.mark.parametrize(
    ("forward_tolerance_m", "lateral_tolerance_m", "forward", "right"),
    [
        (0.02, 0.10, 0.821, 0.09),
        (0.10, 0.02, 0.89, 0.021),
    ],
)
def test_controller_completion_tolerances_are_independently_configurable(
    forward_tolerance_m: float,
    lateral_tolerance_m: float,
    forward: float,
    right: float,
) -> None:
    controller = VisualServoController(
        forward_tolerance_m=forward_tolerance_m,
        lateral_tolerance_m=lateral_tolerance_m,
    )
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET

    for index in range(5):
        command = controller.update(
            observation(forward=forward, right=right, include_table=False),
            now=1.1 + index * 0.1,
        )

    assert command.velocity != (0.0, 0.0, 0.0)
    assert controller.stable_frames == 0
    assert controller.terminal_reason is None


@pytest.mark.parametrize("invalid", [0.0, -0.01, math.nan, math.inf])
@pytest.mark.parametrize(
    "field", ["forward_tolerance_m", "lateral_tolerance_m"]
)
def test_controller_rejects_invalid_completion_tolerances(
    field: str, invalid: float
) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        VisualServoController(**{field: invalid})


def test_runtime_passes_configured_horizontal_intervals_to_controller(
    tmp_path,
) -> None:
    runtime = RawServoRuntime(
        BasePosePlannerConfig(
            task="align",
            output_root=str(tmp_path),
            raw_horizontal_guard_fraction=0.20,
            raw_horizontal_recovery_fraction=0.28,
        ),
        publish=lambda _message: None,
        logger=lambda _message: None,
    )

    assert runtime.controller.guard_fraction == pytest.approx(0.20)
    assert runtime.controller.recovery_low_fraction == pytest.approx(0.28)
    assert runtime.controller.recovery_high_fraction == pytest.approx(0.72)


def test_runtime_passes_configured_completion_tolerances_to_controller(
    tmp_path,
) -> None:
    runtime = RawServoRuntime(
        BasePosePlannerConfig(
            task="align",
            output_root=str(tmp_path),
            raw_forward_tolerance_m=0.06,
            raw_lateral_tolerance_m=0.08,
        ),
        publish=lambda _message: None,
        logger=lambda _message: None,
    )

    assert runtime.controller.forward_tolerance_m == pytest.approx(0.06)
    assert runtime.controller.lateral_tolerance_m == pytest.approx(0.08)


def test_raw_servo_default_runtime_limit_is_sixty_seconds(tmp_path) -> None:
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _: None,
        logger=lambda _: None,
    )
    runtime.controller.reset(1.0)
    runtime.controller.phase = ServoPhase.TRANSLATE_TARGET

    runtime.controller.update(observation(forward=1.0), now=60.9)
    assert not runtime.controller.terminal

    runtime.controller.update(observation(forward=1.0), now=61.0)
    assert runtime.controller.terminal_reason == "maximum run time reached"


def test_controller_holds_zero_until_thirtieth_missing_frame() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET
    moving = controller.update(observation(forward=0.91), now=1.1)
    assert moving.vx > 0.0

    for index in range(29):
        stopped = controller.note_invalid(
            "target absent", hard=False, now=1.2 + 0.1 * index
        )
        assert stopped.velocity == (0.0, 0.0, 0.0)
        assert not controller.terminal

    stopped = controller.note_invalid("target absent", hard=False, now=4.1)
    assert stopped.velocity == (0.0, 0.0, 0.0)
    assert controller.terminal_reason == "tracking lost: target absent"


def test_valid_target_resets_missing_frame_streak_and_resumes_motion() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET
    controller.update(observation(forward=0.91), now=1.1)

    for index in range(3):
        controller.note_invalid(
            "missing tracked target", hard=False, now=1.2 + 0.1 * index
        )

    resumed = controller.update(observation(forward=0.91), now=1.5)
    assert resumed.vx > 0.0
    assert controller.invalid_frames == 0

    for index in range(29):
        held = controller.note_invalid(
            "missing tracked target", hard=False, now=1.6 + 0.1 * index
        )
        assert held.velocity == (0.0, 0.0, 0.0)
        assert not controller.terminal
    assert controller.invalid_frames == 29


def test_missing_required_table_holds_until_thirtieth_frame() -> None:
    controller = VisualServoController()
    controller.reset(1.0)

    for index in range(29):
        held = controller.update(
            observation(include_table=False), now=1.1 + 0.1 * index
        )
        assert held.velocity == (0.0, 0.0, 0.0)
        assert not controller.terminal

    stopped = controller.update(observation(include_table=False), now=4.0)
    assert stopped.velocity == (0.0, 0.0, 0.0)
    assert controller.terminal_reason == "tracking lost: missing table geometry"


def test_controller_requires_five_stable_visual_frames_and_hard_loss_is_immediate() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET
    for index in range(4):
        command = controller.update(
            observation(include_table=False), now=1.1 + index * 0.1
        )
        assert not controller.terminal
        assert command.velocity == (0.0, 0.0, 0.0)
    controller.update(observation(include_table=False), now=1.5)
    assert controller.terminal_reason == "aligned"

    controller.reset(2.0)
    controller.update(observation(forward=0.9), now=2.1)
    controller.note_invalid("track ID changed", hard=True, now=2.2)
    assert controller.terminal_reason == "track ID changed"
    assert controller.current.velocity == (0.0, 0.0, 0.0)


def test_runtime_camera_soft_stale_holds_zero_without_finishing(tmp_path) -> None:
    messages: list[str] = []
    config = BasePosePlannerConfig(task="align to basket", output_root=str(tmp_path))
    runtime = RawServoRuntime(config, publish=messages.append, logger=lambda _: None)
    assert runtime.handle_key("n", now=1.0) == "started"
    runtime.phase = "aligning"
    runtime.controller.reset(1.0)
    runtime.controller.current = ServoCommand(0.02, 0.0, 0.0, 0.15)
    runtime.last_observation_at = 1.0
    runtime.next_publish_at = 1.0

    moving = runtime.publish_due(1.39)
    runtime.next_publish_at = 1.0
    held = runtime.publish_due(1.401)

    assert moving is not None and moving.vx == pytest.approx(0.02)
    assert held is not None and held.velocity == (0.0, 0.0, 0.0)
    assert runtime.phase == "aligning"
    assert runtime.soft_stale
    assert runtime.gate.is_active(1)
    assert not runtime.controller.terminal
    assert runtime.controller.current.velocity == (0.0, 0.0, 0.0)
    assert '"vx":0.0' in messages[-1]


def test_runtime_uses_point_four_soft_stale_and_one_slot_mailbox(tmp_path) -> None:
    config = BasePosePlannerConfig(task="align", output_root=str(tmp_path))
    runtime = RawServoRuntime(
        config,
        publish=lambda _message: None,
        logger=lambda _message: None,
    )

    assert config.raw_camera_stale_s == pytest.approx(0.4)
    assert runtime.observation_events.maxsize == 1


def test_runtime_soft_stale_recovers_on_first_fresh_observation(tmp_path) -> None:
    messages: list[str] = []
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=messages.append,
        logger=lambda _: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    runtime.phase = "aligning"
    runtime.controller.reset(1.0)
    runtime.controller.current = ServoCommand(0.02, 0.0, 0.0, 0.15)
    runtime.last_observation_at = 1.0
    runtime.next_publish_at = 1.0
    runtime.publish_due(1.401)

    accepted = runtime.accept_event(
        RawServoEvent(
            generation=1,
            kind="observation",
            observation=observation(yaw=math.radians(20.0)),
            produced_at_monotonic=1.45,
        ),
        now=1.46,
    )

    assert accepted
    assert not runtime.soft_stale
    assert runtime.phase == "aligning"
    assert runtime.controller.current.wz > 0.0
    assert not runtime.controller.terminal
    assert json.loads(messages[-1])["velocity"]["wz"] > 0.0


def test_runtime_soft_stale_rejects_old_observation(tmp_path) -> None:
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    runtime.phase = "aligning"
    runtime.controller.reset(1.0)
    runtime.last_observation_at = 1.0
    runtime.next_publish_at = 1.0
    runtime.publish_due(1.401)

    accepted = runtime.accept_event(
        RawServoEvent(
            generation=1,
            kind="observation",
            observation=observation(yaw=math.radians(20.0)),
            produced_at_monotonic=1.0,
        ),
        now=1.5,
    )

    assert not accepted
    assert runtime.soft_stale
    assert runtime.controller.current.velocity == (0.0, 0.0, 0.0)
    assert runtime.phase == "aligning"


def test_poll_events_uses_a_fresh_receipt_time_for_each_event(tmp_path) -> None:
    receipt_times = iter((1.1, 1.2))
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _: None,
        monotonic=lambda: next(receipt_times),
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    runtime.phase = "aligning"
    runtime.controller.reset(1.0)
    runtime.events.put(RawServoEvent(generation=1, kind="invalid", error="first"))
    runtime.events.put(RawServoEvent(generation=1, kind="invalid", error="second"))

    assert runtime.poll_events() == 2
    assert runtime.last_observation_at == pytest.approx(1.2)


def test_poll_events_preserves_recovery_between_missing_frame_streaks(
    tmp_path,
) -> None:
    receipt_times = iter(1.1 + 0.01 * index for index in range(40))
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _message: None,
        monotonic=lambda: next(receipt_times),
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    runtime.phase = "aligning"
    runtime.controller.reset(1.0)
    runtime.controller.phase = ServoPhase.TRANSLATE_TARGET

    def frame(index: int) -> DetectionFrameData:
        return DetectionFrameData(
            frame_index=index,
            camera_timestamp=float(index),
            rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        )

    for index in range(1, 21):
        runtime.events.put(
            RawServoEvent(
                generation=1,
                kind="invalid",
                error="missing tracked target",
                frame=frame(index),
            )
        )
    runtime.observation_events.put(
        RawServoEvent(
            generation=1,
            kind="observation",
            observation=observation(forward=0.9),
            frame=frame(21),
        )
    )
    for index in range(22, 32):
        runtime.events.put(
            RawServoEvent(
                generation=1,
                kind="invalid",
                error="missing tracked target",
                frame=frame(index),
            )
        )

    assert runtime.poll_events() == 31
    assert runtime.phase == "aligning"
    assert runtime.controller.invalid_frames == 10
    assert runtime.last_applied_frame_index == 31


def test_poll_events_applies_at_most_one_observation_per_iteration(
    tmp_path,
) -> None:
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _message: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    runtime.phase = "aligning"
    runtime.controller.reset(1.0)
    runtime.controller.phase = ServoPhase.TRANSLATE_TARGET

    def frame(index: int) -> DetectionFrameData:
        return DetectionFrameData(
            frame_index=index,
            camera_timestamp=float(index),
            rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        )

    first = RawServoEvent(
        generation=1,
        kind="observation",
        observation=observation(forward=0.9),
        frame=frame(1),
    )
    second = RawServoEvent(
        generation=1,
        kind="observation",
        observation=observation(forward=0.9),
        frame=frame(3),
    )
    clock_calls = 0

    def monotonic() -> float:
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 1:
            runtime.observation_events.put_nowait(second)
        return 1.0 + 0.01 * clock_calls

    runtime.monotonic = monotonic
    runtime.observation_events.put_nowait(first)
    runtime.events.put_nowait(
        RawServoEvent(
            generation=1,
            kind="invalid",
            error="missing tracked target",
            frame=frame(2),
        )
    )

    assert runtime.poll_events() == 2
    assert runtime.last_applied_frame_index == 2
    assert runtime.observation_events.get_nowait() is second


def test_worker_routing_keeps_only_latest_waiting_observation() -> None:
    reliable: queue.Queue[RawServoEvent] = queue.Queue()
    mailbox: queue.Queue[RawServoEvent] = queue.Queue(maxsize=1)
    first = RawServoEvent(generation=1, kind="observation")
    latest = RawServoEvent(generation=1, kind="observation")

    assert raw_servo._route_worker_event(reliable, mailbox, first) is None
    displaced = raw_servo._route_worker_event(reliable, mailbox, latest)

    assert displaced is first
    assert mailbox.get_nowait() is latest
    assert reliable.empty()


def test_worker_publish_records_displaced_observation_as_not_applied(tmp_path) -> None:
    reliable: queue.Queue[RawServoEvent] = queue.Queue()
    mailbox: queue.Queue[RawServoEvent] = queue.Queue(maxsize=1)
    diagnostics = AsyncFrameDiagnosticsWriter(logger=lambda _message: None)
    output_dir = tmp_path / "run"

    def event(index: int) -> RawServoEvent:
        return RawServoEvent(
            generation=1,
            kind="observation",
            output_dir=str(output_dir),
            frame=DetectionFrameData(
                frame_index=index,
                camera_timestamp=float(index),
                rgb=np.zeros((48, 64, 3), dtype=np.uint8),
            ),
        )

    first = event(0)
    latest = event(1)
    raw_servo._publish_worker_event(reliable, mailbox, diagnostics, first)
    raw_servo._publish_worker_event(reliable, mailbox, diagnostics, latest)
    routed_latest = mailbox.get_nowait()
    assert routed_latest.produced_at_monotonic is not None
    diagnostics.submit_decision(
        1,
        1,
        control_applied=True,
        controller_state={"phase": "yaw_align"},
        command={"vx": 0.0, "vy": 0.0, "wz": 0.05, "duration_s": 0.15},
    )
    diagnostics.close(drain=True)

    rows = [
        json.loads(line)
        for line in (output_dir / "raw_servo_frames.jsonl").read_text().splitlines()
    ]
    assert [row["frame_index"] for row in rows] == [0, 1]
    assert rows[0]["control_applied"] is False
    assert rows[0]["controller"] is None
    assert rows[0]["command"] is None
    assert rows[1]["control_applied"] is True


def test_worker_routing_never_conflates_lifecycle_or_safety_events() -> None:
    reliable: queue.Queue[RawServoEvent] = queue.Queue()
    mailbox: queue.Queue[RawServoEvent] = queue.Queue(maxsize=1)
    initialized = RawServoEvent(generation=1, kind="initialized")
    invalid = RawServoEvent(generation=1, kind="invalid", error="missing target")
    error = RawServoEvent(generation=1, kind="error", error="camera failed")

    for event in (initialized, invalid, error):
        assert raw_servo._route_worker_event(reliable, mailbox, event) is None

    assert [reliable.get_nowait().kind for _ in range(3)] == [
        "initialized",
        "invalid",
        "error",
    ]
    assert mailbox.empty()


def test_runtime_rejects_out_of_order_observation_frame(tmp_path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    runtime.phase = "aligning"
    runtime.controller.reset(1.0)

    def frame(index: int) -> DetectionFrameData:
        return DetectionFrameData(
            frame_index=index,
            camera_timestamp=float(index),
            rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        )

    assert runtime.accept_event(
        RawServoEvent(
            generation=1,
            kind="observation",
            observation=observation(yaw=math.radians(20.0)),
            output_dir=str(output_dir),
            frame=frame(2),
        ),
        now=1.1,
    )
    assert not runtime.accept_event(
        RawServoEvent(
            generation=1,
            kind="observation",
            observation=observation(yaw=math.radians(30.0)),
            output_dir=str(output_dir),
            frame=frame(1),
        ),
        now=1.2,
    )
    assert runtime.last_applied_frame_index == 2


def test_runtime_deadline_stops_before_republishing_nonzero_command(tmp_path) -> None:
    messages: list[str] = []
    config = BasePosePlannerConfig(task="align to basket", output_root=str(tmp_path))
    runtime = RawServoRuntime(config, publish=messages.append, logger=lambda _: None)
    assert runtime.handle_key("n", now=1.0) == "started"
    runtime.phase = "aligning"
    runtime.controller.reset(1.0)
    runtime.controller.current = ServoCommand(0.30, 0.0, 0.0, 0.15)
    runtime.last_observation_at = 60.9
    runtime.next_publish_at = 61.05

    command = runtime.publish_due(61.0)

    assert command is not None and command.velocity == (0.0, 0.0, 0.0)
    assert runtime.phase == "idle"
    assert runtime.controller.terminal_reason == "maximum run time reached"
    assert len(messages) == 4
    assert all('"vx":0.0' in message for message in messages[-3:])



def test_observation_allows_missing_table_and_carries_target_box() -> None:
    mask = np.zeros((480, 640), dtype=bool)
    mask[180:340, 220:440] = True
    target = types.SimpleNamespace(
        track_id=11,
        bbox_xyxy=(220.0, 180.0, 440.0, 340.0),
        mask=mask,
    )

    value = _observation(
        snapshot(), target, None, RawServoCalibration(camera_pitch_deg=0.0)
    )

    assert value.table is None
    assert value.surface_track_id is None
    assert value.target_bbox_xyxy == target.bbox_xyxy
    assert value.image_width == 640
    assert value.image_height == 480


def test_runtime_writes_each_initialized_frame_after_command_update(tmp_path) -> None:
    messages: list[str] = []
    config = BasePosePlannerConfig(task="align", output_root=str(tmp_path))
    runtime = RawServoRuntime(config, publish=messages.append, logger=lambda _: None)
    assert runtime.handle_key("n", now=1.0) == "started"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    frame = DetectionFrameData(
        frame_index=0,
        camera_timestamp=10.0,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        target_bbox_xyxy=(24.0, 10.0, 40.0, 34.0),
        target_track_id=1,
        target_confidence=0.9,
        surface_bbox_xyxy=(1.0, 1.0, 63.0, 47.0),
        surface_track_id=2,
        surface_confidence=0.8,
        perception_kind="initialized",
    )

    event = RawServoEvent(
        generation=1,
        kind="initialized",
        observation=observation(yaw=math.radians(20.0)),
        output_dir=str(output_dir),
        frame=frame,
    )
    runtime.diagnostics.submit_frame(1, output_dir, frame)
    accepted = runtime.accept_event(event, now=1.1)
    runtime.flush_diagnostics()

    assert accepted
    record = json.loads((output_dir / "raw_servo_frames.jsonl").read_text())
    assert record["frame_index"] == 0
    assert record["controller"]["phase"] == "yaw_align"
    assert record["annotated_image"] is None
    assert not (output_dir / "frames").exists()




def test_runtime_frame_records_vertical_recenter_state(tmp_path) -> None:
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _message: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    output_dir = tmp_path / "vertical"
    output_dir.mkdir()
    frame = DetectionFrameData(
        frame_index=0,
        camera_timestamp=10.0,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        target_bbox_xyxy=(26.0, 0.0, 38.0, 16.0),
        target_track_id=1,
        target_confidence=0.9,
        perception_kind="initialized",
    )

    event = RawServoEvent(
        generation=1,
        kind="initialized",
        observation=observation(bbox=(260.0, 0.0, 380.0, 74.0)),
        output_dir=str(output_dir),
        frame=frame,
    )
    runtime.diagnostics.submit_frame(1, output_dir, frame)
    runtime.accept_event(event, now=1.1)
    runtime.flush_diagnostics()

    record = json.loads((output_dir / "raw_servo_frames.jsonl").read_text())
    assert record["controller"]["phase"] == "vertical_recenter"
    assert record["controller"]["vertical_resume_phase"] == "yaw_align"
    assert record["controller"]["vertical_recenter_stable_frames"] == 0
    event = json.loads(
        (output_dir / "raw_servo_events.jsonl").read_text().splitlines()[-1]
    )
    assert event["vertical_resume_phase"] == "yaw_align"
    assert event["vertical_recenter_stable_frames"] == 0


def test_runtime_cancel_marks_waiting_observation_not_applied(tmp_path) -> None:
    output_dir = tmp_path / "cancelled"
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _message: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    frame = DetectionFrameData(
        frame_index=0,
        camera_timestamp=1.0,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
    )
    waiting = RawServoEvent(
        generation=1,
        kind="observation",
        output_dir=str(output_dir),
        frame=frame,
    )
    runtime.diagnostics.submit_frame(1, output_dir, frame)
    runtime.observation_events.put(waiting)

    runtime.cancel("operator_space", now=1.1)
    runtime.flush_diagnostics()

    row = json.loads((output_dir / "raw_servo_frames.jsonl").read_text())
    assert row["control_applied"] is False
    assert row["controller"] is None
    assert row["command"] is None


def test_diagnostic_writer_failure_never_terminates_runtime(tmp_path) -> None:
    warnings: list[str] = []

    class FailingWriter:
        def __init__(self, _output_dir) -> None:
            pass

        def write(self, *_args, **_kwargs) -> None:
            raise OSError("injected diagnostic failure")

    diagnostics = AsyncFrameDiagnosticsWriter(
        logger=warnings.append,
        writer_factory=FailingWriter,
    )
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _message: None,
        diagnostics=diagnostics,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    output_dir = tmp_path / "failed"
    frame = DetectionFrameData(
        frame_index=0,
        camera_timestamp=1.0,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
    )
    diagnostics.submit_frame(1, output_dir, frame)

    assert runtime.accept_event(
        RawServoEvent(
            generation=1,
            kind="initialized",
            observation=observation(yaw=math.radians(20.0)),
            output_dir=str(output_dir),
            frame=frame,
        ),
        now=1.1,
    )
    runtime.flush_diagnostics()

    assert runtime.phase == "aligning"
    assert not runtime.controller.terminal
    assert len(warnings) == 1
    assert "injected diagnostic failure" in warnings[0]


def test_runtime_publishes_zero_immediately_on_first_invalid_frame(tmp_path) -> None:
    messages: list[str] = []
    config = BasePosePlannerConfig(task="align", output_root=str(tmp_path))
    runtime = RawServoRuntime(config, publish=messages.append, logger=lambda _: None)
    assert runtime.handle_key("n", now=1.0) == "started"
    runtime.phase = "aligning"
    runtime.controller.reset(1.0)
    runtime.controller.current = ServoCommand(0.0, 0.0, 0.2, 0.15)

    runtime.accept_event(
        RawServoEvent(generation=1, kind="invalid", error="missing table"),
        now=1.01,
    )

    assert len(messages) == 2
    assert "\"vx\":0.0" in messages[-1]
    assert "\"vy\":0.0" in messages[-1]
    assert "\"wz\":0.0" in messages[-1]
    assert runtime.phase == "aligning"


def test_worker_preserves_target_and_surface_reacquisition_after_observation_failure(
    tmp_path, monkeypatch
) -> None:
    target_mask = np.zeros((480, 640), dtype=bool)
    target_mask[180:340, 220:440] = True
    table_mask = np.zeros((480, 640), dtype=bool)
    table_mask[120:360, 100:540] = True
    target = TrackedInstance(
        1, 0, "blue basket", 0.9, (220.0, 180.0, 440.0, 340.0), target_mask
    )
    reacquired_target = TrackedInstance(
        8, 0, "blue basket", 0.95, (222.0, 182.0, 442.0, 342.0), target_mask
    )
    table = TrackedInstance(
        2, 1, "table", 0.8, (100.0, 120.0, 540.0, 360.0), table_mask
    )
    reacquired_table = TrackedInstance(
        9, 1, "table", 0.85, (100.0, 120.0, 540.0, 360.0), table_mask
    )
    stop_event = threading.Event()
    tracker_start: list[dict[str, object]] = []

    class Camera:
        def capture(self):
            return snapshot()

        def close(self):
            pass

    class Client:
        def run(self, **kwargs):
            if kwargs["schema_filename"] == "target_schema.json":
                return target_payload()
            return table_payload()

    class Tracker:
        calls = 0

        def start(self, *_args, **kwargs):
            tracker_start.append(kwargs)
            return {
                "class_names": ["blue basket", "table"],
                "reference_boxes_xyxy": [
                    [192, 120, 384, 336],
                    [16, 48, 304, 432],
                    [336, 57.6, 624, 422.4],
                ],
                "reference_class_ids": [0, 1, 1],
            }

        def track(self, _rgb):
            self.calls += 1
            if self.calls == 1:
                return [target, table]
            if self.calls == 3:
                stop_event.set()
            return [reacquired_target, reacquired_table]

    original_observation = raw_servo._observation
    observation_calls = 0

    def fail_first_reacquired_observation(*args, **kwargs):
        nonlocal observation_calls
        observation_calls += 1
        if observation_calls == 2:
            raise ValueError("temporary target geometry failure")
        return original_observation(*args, **kwargs)

    monkeypatch.setattr(raw_servo, "_observation", fail_first_reacquired_observation)

    config = BasePosePlannerConfig(
        task="align to basket",
        output_root=str(tmp_path),
        camera_pitch_deg=0.0,
    )
    requests: queue.Queue[int | None] = queue.Queue()
    events: queue.Queue[RawServoEvent] = queue.Queue()
    gate = GenerationGate()
    gate.activate(1)
    requests.put(1)
    requests.put(None)

    run_raw_servo_worker(
        config,
        requests,
        events,
        gate,
        stop_event,
        camera_factory=Camera,
        client_factory=Client,
        tracker_factory=Tracker,
    )

    initialized = events.get_nowait()
    failed_reacquisition = events.get_nowait()
    followup = events.get_nowait()
    assert initialized.kind == "initialized"
    assert initialized.frame is not None and initialized.frame.frame_index == 0
    assert failed_reacquisition.kind == "invalid"
    assert failed_reacquisition.error == "temporary target geometry failure"
    assert (
        failed_reacquisition.frame is not None
        and failed_reacquisition.frame.frame_index == 1
    )
    assert followup.kind == "observation"
    assert followup.observation is not None and followup.observation.table is not None
    assert followup.observation.target_track_id == 8
    assert followup.observation.surface_track_id == 9
    assert followup.details == {
        "surface_id_mismatch": False,
        "target_reacquired": True,
        "previous_target_id": 1,
        "target_track_id": 8,
        "surface_reacquired": True,
        "previous_surface_id": 2,
        "surface_track_id": 9,
    }
    assert followup.frame is not None and followup.frame.frame_index == 2
    assert followup.frame.surface_bbox_xyxy == reacquired_table.bbox_xyxy
    assert tracker_start[0]["surface_bboxes"] == (
        (25.0, 100.0, 475.0, 900.0),
        (525.0, 120.0, 975.0, 880.0),
    )
    output_dir = Path(initialized.output_dir or "")
    assert (output_dir / "target_result.json").is_file()
    assert (output_dir / "table_result.json").is_file()


def test_runtime_records_target_reacquisition_event_details(tmp_path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _message: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    runtime.accept_event(
        RawServoEvent(
            generation=1,
            kind="initialized",
            observation=observation(),
            output_dir=str(output_dir),
        ),
        now=1.1,
    )
    details = {
        "surface_id_mismatch": False,
        "target_reacquired": True,
        "previous_target_id": 1,
        "target_track_id": 8,
    }

    runtime.accept_event(
        RawServoEvent(
            generation=1,
            kind="observation",
            observation=observation(),
            output_dir=str(output_dir),
            details=details,
        ),
        now=1.2,
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "raw_servo_events.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["event"] == "control_update"
    assert rows[-1]["event_details"] == details


def test_grounding_failure_prevents_tracker_initialization(tmp_path) -> None:
    stop_event = threading.Event()
    tracker_factory_calls: list[bool] = []

    class Camera:
        def capture(self):
            return snapshot()

        def close(self):
            pass

    class Client:
        def run(self, **kwargs):
            if kwargs["schema_filename"] == "target_schema.json":
                return target_payload()
            stop_event.set()
            value = table_payload()
            value["status"] = "NOT_FOUND"
            value["boxes"] = []
            value["limitations"] = "no table visible"
            return value

    def tracker_factory():
        tracker_factory_calls.append(True)
        raise AssertionError("tracker must not start after grounding failure")

    config = BasePosePlannerConfig(
        task="align to basket",
        output_root=str(tmp_path),
        camera_pitch_deg=0.0,
    )
    requests: queue.Queue[int | None] = queue.Queue()
    events: queue.Queue[RawServoEvent] = queue.Queue()
    gate = GenerationGate()
    gate.activate(1)
    requests.put(1)
    requests.put(None)

    run_raw_servo_worker(
        config,
        requests,
        events,
        gate,
        stop_event,
        camera_factory=Camera,
        client_factory=Client,
        tracker_factory=tracker_factory,
    )

    event = events.get_nowait()
    assert event.kind == "error"
    assert "grounding status is NOT_FOUND" in (event.error or "")
    assert tracker_factory_calls == []


def test_raw_launch_default_standoff_is_point_eight_meters() -> None:
    config = InferenceLaunchConfig(
        planner_input="base_pose",
        base_pose_mode="raw_yoloe_servo",
        base_pose_task="align to the blue basket",
    )

    command = build_planner_input_command(config, Path("/workspace/sonic"))

    assert "--raw-target-distance-m 0.8" in command
    assert "--raw-forward-tolerance-m 0.1" in command
    assert "--raw-lateral-tolerance-m 0.1" in command
    assert "--raw-max-lateral-speed-m-s 0.16" in command


def test_raw_launch_uses_direct_camera_and_never_starts_lingbot() -> None:
    config = InferenceLaunchConfig(
        planner_input="base_pose",
        base_pose_mode="raw_yoloe_servo",
        base_pose_task="align to the blue basket",
        camera_host="head-camera",
        camera_port=5555,
        base_pose_raw_target_distance_m=0.65,
        base_pose_raw_forward_tolerance_m=0.06,
        base_pose_raw_lateral_tolerance_m=0.08,
    )

    command = build_planner_input_command(config, Path("/workspace/sonic"))

    assert "--mode raw_yoloe_servo" in command
    assert "--camera-host head-camera --camera-port 5555" in command
    assert "--camera-pitch-deg -25.0" in command
    assert "--raw-target-distance-m 0.65" in command
    assert "--raw-forward-tolerance-m 0.06" in command
    assert "--raw-lateral-tolerance-m 0.08" in command
    assert "yoloe-26m-seg.pt" in command
    assert ".venv_lingbot_depth" not in command
    assert "run_lingbot_depth_viewer.py" not in command


def test_target_prompt_and_schema_no_longer_request_table() -> None:
    prompt = build_raw_servo_target_prompt("align to the blue basket", RawServoCalibration())

    assert "primary_target" in prompt
    assert "support_surface" not in prompt
    assert "operation platform" not in prompt.lower()
    assert "visible table" not in prompt.lower()
    assert '"table"' not in prompt.lower()
    assert "used directly as YOLOE reference-image visual prompts" in prompt
    assert "support_surface" not in RAW_SERVO_TARGET_SCHEMA["properties"]


def test_table_prompt_exactly_reuses_auto_reference_prompt() -> None:
    assert raw_servo.build_raw_servo_table_prompt() == build_grounding_prompt("table")


def test_table_validator_preserves_every_box() -> None:
    assert raw_servo.validate_raw_servo_table(table_payload()) == (
        (25.0, 100.0, 475.0, 900.0),
        (525.0, 120.0, 975.0, 880.0),
    )


def test_table_validator_rejects_response_for_different_target() -> None:
    value = table_payload()
    value["target"] = "chair"

    with pytest.raises(
        GroundingValidationError,
        match="grounding response target does not match the operator target",
    ):
        raw_servo.validate_raw_servo_table(value)


def test_grounding_requests_use_independent_clients_and_run_concurrently(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "initial_rgb.png"
    rgb_path.write_bytes(b"png")
    barrier = threading.Barrier(2, timeout=1.0)
    calls: list[dict[str, object]] = []
    clients: list[object] = []

    class Client:
        def __init__(self) -> None:
            clients.append(self)

        def run(self, **kwargs):
            calls.append(kwargs)
            barrier.wait()
            if kwargs["schema_filename"] == "target_schema.json":
                return target_payload()
            return table_payload()

    target, table_bboxes = raw_servo.ground_raw_servo_references(
        BasePosePlannerConfig(task="align to basket"),
        rgb_path,
        tmp_path,
        client_factory=Client,
    )

    assert len(clients) == 2
    assert clients[0] is not clients[1]
    assert {Path(call["image_paths"][0]) for call in calls} == {rgb_path}
    table_call = next(
        call for call in calls if call["schema_filename"] == "table_schema.json"
    )
    assert table_call["prompt"] == build_grounding_prompt("table")
    assert table_call["schema"] is GROUNDING_SCHEMA
    assert target.target_prompt == "blue plastic basket"
    assert table_bboxes == (
        (25.0, 100.0, 475.0, 900.0),
        (525.0, 120.0, 975.0, 880.0),
    )
    assert (tmp_path / "target_result.json").is_file()
    assert (tmp_path / "table_result.json").is_file()


class _FakeVisualPromptModel:
    def __init__(self) -> None:
        torch = pytest.importorskip("torch")
        self.predictor = object()
        self.embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        self.model = types.SimpleNamespace(pe=self.embeddings)
        self.predict_calls: list[dict[str, object]] = []
        self.class_calls: list[tuple[list[str], object]] = []

    def predict(self, **kwargs: object) -> list[object]:
        self.predict_calls.append(kwargs)
        return [object()]

    def set_classes(self, names: list[str], embeddings: object = None) -> None:
        self.class_calls.append((names, embeddings))


def test_tracker_initializes_two_visual_classes_from_codex_reference_boxes() -> None:
    tracker = object.__new__(YoloePersistentTracker)
    tracker.model = _FakeVisualPromptModel()
    tracker.confidence = 0.25
    tracker.imgsz = 640
    tracker.device = "0"
    tracker.class_names = None
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    artifact = tracker.start(
        rgb,
        target_prompt="blue plastic basket",
        target_bbox=(250.0, 200.0, 750.0, 700.0),
        surface_prompt="table",
        surface_bboxes=(
            (0.0, 100.0, 450.0, 900.0),
            (550.0, 100.0, 1000.0, 900.0),
        ),
    )

    assert artifact == {
        "class_names": ["blue plastic basket", "table"],
        "reference_boxes_xyxy": [
            [160.0, 96.0, 480.0, 336.0],
            [0.0, 48.0, 288.0, 432.0],
            [352.0, 48.0, 640.0, 432.0],
        ],
        "reference_class_ids": [0, 1, 1],
    }
    call = tracker.model.predict_calls[0]
    np.testing.assert_array_equal(
        call["visual_prompts"]["bboxes"],
        np.asarray(artifact["reference_boxes_xyxy"], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        call["visual_prompts"]["cls"], np.asarray([0, 1, 1], dtype=np.int32)
    )
    assert call["refer_image"].shape == (480, 640, 3)
    assert call["source"].shape == (480, 640, 3)
    names, installed = tracker.model.class_calls[0]
    assert names == ["blue plastic basket", "table"]
    assert installed is tracker.model.embeddings
    assert tracker._initial_surface_embedding.tolist() == [[[3.0, 4.0]]]
    assert tracker.model.predictor is None


def test_initial_table_selection_considers_every_grounded_box() -> None:
    mask = np.ones((480, 640), dtype=bool)
    left = TrackedInstance(
        10, 1, "table", 0.95, (0.0, 48.0, 100.0, 432.0), mask
    )
    right = TrackedInstance(
        11, 1, "table", 0.80, (352.0, 48.0, 640.0, 432.0), mask
    )

    chosen = raw_servo.select_initial_instance(
        [left, right],
        class_index=1,
        grounded_bboxes=(
            (0.0, 100.0, 50.0, 900.0),
            (550.0, 100.0, 1000.0, 900.0),
        ),
        width=640,
        height=480,
    )

    assert chosen.track_id == 11


def _tracked_instance(
    track_id: int,
    class_index: int,
    confidence: float,
) -> TrackedInstance:
    return TrackedInstance(
        track_id,
        class_index,
        "blue basket" if class_index == 0 else "table",
        confidence,
        (220.0, 100.0, 440.0, 300.0),
        np.ones((480, 640), dtype=bool),
    )


def test_target_resolver_prefers_expected_same_class_id() -> None:
    expected = _tracked_instance(1, 0, 0.40)
    higher_confidence = _tracked_instance(8, 0, 0.95)

    chosen, reacquired, mismatch = raw_servo._resolve_target(
        [higher_confidence, expected], 1
    )

    assert chosen is expected
    assert not reacquired
    assert not mismatch


def test_target_resolver_adopts_highest_confidence_same_class_new_id() -> None:
    low = _tracked_instance(7, 0, 0.60)
    high = _tracked_instance(8, 0, 0.95)

    chosen, reacquired, mismatch = raw_servo._resolve_target([low, high], 1)

    assert chosen is high
    assert reacquired
    assert not mismatch


def test_target_resolver_distinguishes_missing_from_wrong_class() -> None:
    table = _tracked_instance(2, 1, 0.90)
    assert raw_servo._resolve_target([table], 1) == (None, False, False)

    wrong_class = _tracked_instance(1, 1, 0.90)
    assert raw_servo._resolve_target([wrong_class], 1) == (None, False, True)


def test_latest_g1_initial_box_does_not_trigger_recenter() -> None:
    controller = VisualServoController()
    controller.reset(1.0)

    controller.update(
        observation(
            yaw=math.radians(20.0),
            bbox=(292.0, 0.0, 560.0, 115.875),
        ),
        now=1.1,
    )

    assert controller.phase is ServoPhase.YAW_ALIGN


def test_latest_g3_initial_box_does_not_trigger_recenter() -> None:
    controller = VisualServoController()
    controller.reset(1.0)

    controller.update(
        observation(
            yaw=math.radians(20.0),
            bbox=(267.75, 82.0, 621.0, 317.0),
        ),
        now=1.1,
    )

    assert controller.phase is ServoPhase.YAW_ALIGN


def test_horizontal_guard_uses_center_25_and_75_percent_boundaries() -> None:
    controller = VisualServoController()

    assert controller._visibility_guarded(
        observation(bbox=(0.0, 100.0, 319.0, 300.0))
    )
    assert not controller._visibility_guarded(
        observation(bbox=(0.0, 100.0, 320.0, 300.0))
    )
    assert not controller._visibility_guarded(
        observation(bbox=(320.0, 100.0, 640.0, 300.0))
    )
    assert controller._visibility_guarded(
        observation(bbox=(321.0, 100.0, 640.0, 300.0))
    )


def test_horizontal_recovery_uses_center_30_to_70_percent_boundaries() -> None:
    controller = VisualServoController()

    assert not controller._target_center_recovered(
        observation(bbox=(141.5, 100.0, 241.5, 300.0))
    )
    assert controller._target_center_recovered(
        observation(bbox=(142.0, 100.0, 242.0, 300.0))
    )
    assert controller._target_center_recovered(
        observation(bbox=(398.0, 100.0, 498.0, 300.0))
    )
    assert not controller._target_center_recovered(
        observation(bbox=(398.5, 100.0, 498.5, 300.0))
    )


@pytest.mark.parametrize(
    "bbox",
    [
        (0.0, 100.0, 319.0, 300.0),
        (321.0, 100.0, 640.0, 300.0),
    ],
)
def test_visibility_guard_stops_yaw_and_enters_recenter(
    bbox: tuple[float, float, float, float],
) -> None:
    controller = VisualServoController()
    controller.reset(1.0)

    command = controller.update(
        observation(yaw=math.radians(20.0), bbox=bbox), now=1.1
    )

    assert command.velocity == (0.0, 0.0, 0.0)
    assert controller.phase is ServoPhase.RECENTER
    assert controller.resume_phase is ServoPhase.YAW_ALIGN
    assert controller.last_transition_reason == "target visibility guard"


def test_recenter_uses_only_vy_and_resumes_after_three_centered_frames() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.update(
        observation(
            right=0.10,
            yaw=math.radians(20.0),
            bbox=(385.0, 100.0, 640.0, 300.0),
        ),
        now=1.1,
    )

    recovering = controller.update(
        observation(right=0.10, yaw=math.radians(20.0)), now=1.2
    )
    assert recovering.vx == 0.0
    assert recovering.vy == pytest.approx(-0.30)
    assert recovering.wz == 0.0
    for index in range(2):
        controller.update(
            observation(right=0.0, yaw=math.radians(20.0)),
            now=1.3 + index * 0.1,
        )

    assert controller.phase is ServoPhase.YAW_ALIGN
    assert controller.current.velocity == (0.0, 0.0, 0.0)


def test_top_guard_stops_and_enters_vertical_recenter() -> None:
    controller = VisualServoController()
    controller.reset(1.0)

    command = controller.update(
        observation(
            yaw=math.radians(20.0),
            bbox=(260.0, 0.0, 380.0, 74.0),
        ),
        now=1.1,
    )

    assert command.velocity == (0.0, 0.0, 0.0)
    assert controller.phase is ServoPhase.VERTICAL_RECENTER
    assert controller.vertical_resume_phase is ServoPhase.YAW_ALIGN


def test_vertical_recenter_moves_only_forward_then_holds_for_three_frames() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.update(
        observation(bbox=(260.0, 0.0, 380.0, 74.0)),
        now=1.1,
    )

    recovering = controller.update(
        observation(bbox=(260.0, 0.0, 380.0, 109.0)),
        now=1.2,
    )
    assert recovering.velocity == pytest.approx((0.30, 0.0, 0.0))
    assert controller.phase is ServoPhase.VERTICAL_RECENTER

    for index in range(3):
        command = controller.update(
            observation(bbox=(260.0, 0.0, 380.0, 110.0)),
            now=1.3 + index * 0.1,
        )
        assert command.velocity == (0.0, 0.0, 0.0)

    assert controller.phase is ServoPhase.YAW_ALIGN
    assert controller.vertical_resume_phase is None


def test_vertical_recenter_recovery_frames_must_be_consecutive() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.update(
        observation(bbox=(260.0, 0.0, 380.0, 74.0)),
        now=1.1,
    )
    for index in range(2):
        controller.update(
            observation(bbox=(260.0, 0.0, 380.0, 110.0)),
            now=1.2 + index * 0.1,
        )
    assert controller.vertical_recenter_stable_frames == 2

    controller.note_invalid("missing tracked target", hard=False, now=1.4)
    assert controller.vertical_recenter_stable_frames == 0

    controller.update(
        observation(bbox=(260.0, 0.0, 380.0, 110.0)),
        now=1.5,
    )
    assert controller.phase is ServoPhase.VERTICAL_RECENTER
    assert controller.vertical_recenter_stable_frames == 1


def test_vertical_guard_and_recovery_use_literal_bottom_edge_boundaries() -> None:
    controller = VisualServoController()

    guarded = observation(bbox=(260.0, 0.0, 380.0, 74.0))
    clear = observation(bbox=(260.0, 0.0, 380.0, 75.0))
    not_recovered = observation(bbox=(260.0, 0.0, 380.0, 109.0))
    recovered = observation(bbox=(260.0, 0.0, 380.0, 110.0))

    assert controller._vertical_visibility_guarded(guarded)
    assert not controller._vertical_visibility_guarded(clear)
    assert not controller._target_vertical_recovered(not_recovered)
    assert controller._target_vertical_recovered(recovered)


def test_vertical_recenter_preserves_interrupted_horizontal_resume_phase() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.RECENTER
    controller.resume_phase = ServoPhase.YAW_TRIM

    controller.update(
        observation(bbox=(385.0, 0.0, 640.0, 74.0)), now=1.1
    )

    assert controller.phase is ServoPhase.VERTICAL_RECENTER
    assert controller.vertical_resume_phase is ServoPhase.RECENTER
    assert controller.resume_phase is ServoPhase.YAW_TRIM

    for index in range(3):
        controller.update(
            observation(bbox=(385.0, 0.0, 640.0, 110.0)),
            now=1.2 + index * 0.1,
        )

    assert controller.phase is ServoPhase.RECENTER
    assert controller.resume_phase is ServoPhase.YAW_TRIM
    assert controller.vertical_resume_phase is None


def test_vertical_recenter_precedes_simultaneous_horizontal_recenter() -> None:
    controller = VisualServoController()
    controller.reset(1.0)

    controller.update(
        observation(
            right=0.10,
            yaw=math.radians(20.0),
            bbox=(385.0, 0.0, 640.0, 74.0),
        ),
        now=1.1,
    )

    assert controller.phase is ServoPhase.VERTICAL_RECENTER
    assert controller.vertical_resume_phase is ServoPhase.YAW_ALIGN
    assert controller.resume_phase is None

    for index in range(3):
        command = controller.update(
            observation(
                right=0.10,
                yaw=math.radians(20.0),
                bbox=(385.0, 0.0, 640.0, 110.0),
            ),
            now=1.2 + index * 0.1,
        )

    assert command.velocity == (0.0, 0.0, 0.0)
    assert controller.phase is ServoPhase.RECENTER
    assert controller.resume_phase is ServoPhase.YAW_ALIGN
    assert controller.vertical_resume_phase is None


def test_dual_recenter_resumes_translation_without_table_geometry() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET

    controller.update(
        observation(
            right=0.10,
            bbox=(385.0, 0.0, 640.0, 74.0),
            include_table=False,
        ),
        now=1.1,
    )
    for index in range(3):
        controller.update(
            observation(
                right=0.10,
                bbox=(385.0, 0.0, 640.0, 110.0),
                include_table=False,
            ),
            now=1.2 + index * 0.1,
        )

    assert controller.phase is ServoPhase.RECENTER
    assert controller.resume_phase is ServoPhase.TRANSLATE_TARGET

    for index in range(3):
        command = controller.update(
            observation(include_table=False),
            now=1.5 + index * 0.1,
        )

    assert command.velocity == (0.0, 0.0, 0.0)
    assert controller.phase is ServoPhase.TRANSLATE_TARGET
    assert controller.invalid_frames == 0


def test_trim_yaw_cap_and_three_frame_lock_before_translation() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.YAW_TRIM
    controller.current = ServoCommand(0.0, 0.0, 0.25, 0.15)

    command = controller.update(observation(yaw=math.radians(6.0)), now=1.1)
    assert command.wz == pytest.approx(0.10)
    assert controller.phase is ServoPhase.YAW_TRIM

    for index in range(4):
        command = controller.update(
            observation(yaw=math.radians(2.0)), now=1.2 + index * 0.1
        )

    assert controller.phase is ServoPhase.TRANSLATE_TARGET
    assert command.velocity == (0.0, 0.0, 0.0)


def test_translate_phase_accepts_missing_table_and_finishes_in_five_frames() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET

    for index in range(5):
        command = controller.update(
            observation(include_table=False), now=1.1 + index * 0.1
        )

    assert command.velocity == (0.0, 0.0, 0.0)
    assert controller.phase is ServoPhase.DONE
    assert controller.terminal_reason == "aligned"
