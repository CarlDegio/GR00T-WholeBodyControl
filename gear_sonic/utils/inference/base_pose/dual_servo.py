"""Dual-camera text-prompt detection and failover for raw YOLOE BasePose."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gear_sonic.camera.calibration import (
    CameraCalibrationError,
    load_camera_intrinsics,
)
from gear_sonic.utils.inference.base_pose.sensor import (
    AlignedRGBDSnapshot,
    BasePoseCameraError,
)
from gear_sonic.utils.inference.base_pose.servo import (
    HEAD_MONITOR_HOLD_EVENT,
    GenerationGate,
    RawServoCalibration,
    RawServoEvent,
    RawServoObservation,
    TableGeometry,
    TrackedInstance,
    YoloePersistentTracker,
    _diagnostic_frame,
    _observation,
    _publish_worker_event,
    _resolve_target,
    _surface_geometry_components,
)
from gear_sonic.utils.inference.base_pose.diagnostics import (
    AsyncFrameDiagnosticsWriter,
)

JOINT_CHEST_LOSS_GRACE_FRAMES = 5
JOINT_CHEST_LOSS_ZERO_HOLD_FRAMES = 5
JOINT_HEAD_YAW_LOSS_GRACE_FRAMES = 10
JOINT_HEAD_YAW_LOSS_ZERO_HOLD_FRAMES = 10
TEXT_PROMPT_MODE = "target_text_surface_text"


def _joint_head_yaw_loss_details(missing_frames: int) -> dict[str, Any]:
    """Describe continued control, zero hold, or recovery after head-yaw loss."""

    missing = int(missing_frames)
    if missing <= 0:
        raise ValueError("head yaw missing frame count must be positive")
    switch_frame = (
        JOINT_HEAD_YAW_LOSS_GRACE_FRAMES
        + JOINT_HEAD_YAW_LOSS_ZERO_HOLD_FRAMES
    )
    if missing >= switch_frame:
        stage = "switch_state"
    elif missing >= JOINT_HEAD_YAW_LOSS_GRACE_FRAMES:
        stage = "zero_hold"
    else:
        stage = "grace"
    return {
        "missing_frames": missing,
        "grace_frames": JOINT_HEAD_YAW_LOSS_GRACE_FRAMES,
        "zero_hold_frames": JOINT_HEAD_YAW_LOSS_ZERO_HOLD_FRAMES,
        "switch_frame": switch_frame,
        "stage": stage,
    }


@dataclass(frozen=True)
class DualCameraAttempt:
    attempt_id: int
    live_stream: str
    stage: str
    origin_stream: str


class DualCameraFailoverCoordinator:
    """Select the active view and bounded text-detection failover sequence."""

    def __init__(
        self,
        stream_names: Sequence[str],
        initially_eligible_streams: Sequence[str],
    ):
        names = tuple(str(name) for name in stream_names)
        if len(names) != 2 or len(set(names)) != 2 or any(not name for name in names):
            raise ValueError("dual-camera failover requires two distinct streams")
        eligible = frozenset(str(name) for name in initially_eligible_streams)
        if not eligible.issubset(names):
            raise ValueError("initial eligibility contains an unknown stream")
        self.stream_names = names
        self.initially_eligible_streams = eligible
        self._next_attempt_id = 1
        self._active_attempt_id: int | None = None

    def _attempt(
        self,
        live_stream: str,
        *,
        stage: str,
        origin_stream: str,
    ) -> DualCameraAttempt:
        attempt = DualCameraAttempt(
            attempt_id=self._next_attempt_id,
            live_stream=live_stream,
            stage=stage,
            origin_stream=origin_stream,
        )
        self._next_attempt_id += 1
        self._active_attempt_id = None
        return attempt

    def start(self) -> DualCameraAttempt:
        for stream_name in self.stream_names:
            if stream_name in self.initially_eligible_streams:
                return self._attempt(
                    stream_name,
                    stage="initial",
                    origin_stream=stream_name,
                )
        raise RuntimeError("neither camera is initially eligible")

    def mark_success(self, attempt: DualCameraAttempt) -> None:
        if attempt.live_stream not in self.stream_names:
            raise ValueError("successful attempt belongs to an unknown stream")
        self._active_attempt_id = attempt.attempt_id

    def begin_head_monitor_reacquisition(self) -> DualCameraAttempt:
        """Preempt chest tracking after the background head monitor fires."""

        head_stream = self.stream_names[0]
        return self._attempt(
            head_stream,
            stage="head_monitor",
            origin_stream=head_stream,
        )

    def advance_after_failure(
        self,
        attempt: DualCameraAttempt,
    ) -> DualCameraAttempt | None:
        starts_new_cycle = (
            self._active_attempt_id == attempt.attempt_id
            or attempt.stage in {"initial", "head_monitor"}
        )
        origin = attempt.live_stream if starts_new_cycle else attempt.origin_stream
        if starts_new_cycle:
            live_stream = self.stream_names[0]
        elif attempt.live_stream == self.stream_names[0]:
            live_stream = self.stream_names[1]
        else:
            return None
        return self._attempt(
            live_stream,
            stage=(
                "origin_text"
                if live_stream == origin
                else "alternate_text"
            ),
            origin_stream=origin,
        )


def dual_calibrations_from_config(
    config: Any,
) -> dict[str, RawServoCalibration]:
    """Load per-stream intrinsics and combine them with per-mount extrinsics."""

    try:
        intrinsics = load_camera_intrinsics(config.camera_intrinsics_path)
    except CameraCalibrationError as exc:
        raise BasePoseCameraError(str(exc)) from exc
    head_stream = str(config.dual_head_camera_stream)
    chest_stream = str(config.dual_chest_camera_stream)
    try:
        head = intrinsics[head_stream]
        chest = intrinsics[chest_stream]
    except KeyError as exc:
        raise BasePoseCameraError(
            f"saved calibration is missing {exc.args[0]}"
        ) from exc
    return {
        head_stream: RawServoCalibration(
            width=head.width,
            height=head.height,
            fx=head.fx,
            fy=head.fy,
            cx=head.cx,
            cy=head.cy,
            camera_pitch_deg=float(config.camera_pitch_deg),
            camera_roll_deg=float(config.camera_roll_deg),
            camera_yaw_deg=float(config.camera_yaw_deg),
            camera_forward_offset_m=float(config.camera_forward_offset_m),
            camera_lateral_offset_m=float(config.camera_lateral_offset_m),
        ),
        chest_stream: RawServoCalibration(
            width=chest.width,
            height=chest.height,
            fx=chest.fx,
            fy=chest.fy,
            cx=chest.cx,
            cy=chest.cy,
            camera_pitch_deg=float(config.dual_chest_camera_pitch_deg),
            camera_roll_deg=float(config.dual_chest_camera_roll_deg),
            camera_yaw_deg=float(config.dual_chest_camera_yaw_deg),
            camera_forward_offset_m=float(
                config.dual_chest_camera_forward_offset_m
            ),
            camera_lateral_offset_m=float(
                config.dual_chest_camera_lateral_offset_m
            ),
        ),
    }


def detect_dual_camera_eligibility(
    config: Any,
    snapshots: Mapping[str, AlignedRGBDSnapshot],
    calibrations: Mapping[str, RawServoCalibration],
    *,
    tracker: Any,
    logger: Callable[[str], None] = print,
) -> tuple[set[str], dict[str, str]]:
    """Select initial cameras using fixed-text YOLOE detections only."""

    target_prompt = config.target_prompt
    prompt_artifact = tracker.start_all_text(target_prompt=target_prompt)
    logger(
        "[RawServo] YOLOE prompt "
        f"target={target_prompt!r} surface={config.surface_prompt!r} "
        f"tracker={prompt_artifact}"
    )

    eligible_streams: set[str] = set()
    errors: dict[str, str] = {}
    for stream_name in calibrations:
        snapshot = snapshots.get(stream_name)
        if snapshot is None:
            continue
        try:
            calibrations[stream_name].validate_snapshot(snapshot)
            # Head and chest are independent views, not consecutive video
            # frames. Reset BoT-SORT so each camera's one-shot eligibility
            # check is frame 1 and can activate a new track immediately.
            tracker.reset_tracking()
            instances = list(tracker.track(snapshot.rgb))
            target = _best_instance(instances, 0)
            if target is None:
                raise RuntimeError(
                    f"YOLOE text prompt {target_prompt!r} found no target"
                )
            logger(
                "[RawServo] YOLOE initial detection "
                f"stream={stream_name} eligible=true "
                f"track_id={target.track_id} confidence={target.confidence:.4f} "
                f"bbox={target.bbox_xyxy} "
                f"target_candidates={sum(item.class_index == 0 for item in instances)} "
                f"surface_candidates={sum(item.class_index == 1 for item in instances)}"
            )
            eligible_streams.add(stream_name)
        except Exception as exc:
            errors[stream_name] = str(exc)
            logger(
                "[RawServo] YOLOE initial detection "
                f"stream={stream_name} eligible=false error={exc}"
            )
    return eligible_streams, errors


def _best_instance(
    instances: Sequence[TrackedInstance],
    class_index: int,
) -> TrackedInstance | None:
    candidates = [item for item in instances if item.class_index == class_index]
    return max(candidates, key=lambda item: item.confidence, default=None)


def _select_tracked_instances(
    instances: Sequence[TrackedInstance],
    *,
    target_id: int | None,
    surface_id: int | None,
) -> tuple[TrackedInstance | None, TrackedInstance | None, bool, bool]:
    if target_id is None:
        target = _best_instance(instances, 0)
        target_reacquired = False
    else:
        target, target_reacquired, _ = _resolve_target(instances, target_id)
    if surface_id is None:
        surface = _best_instance(instances, 1)
        surface_reacquired = False
    else:
        surface, surface_reacquired, _ = _resolve_target(
            instances,
            surface_id,
            class_index=1,
        )
    return target, surface, target_reacquired, surface_reacquired


def _observe_tracked_snapshot(
    snapshot: AlignedRGBDSnapshot,
    tracker: Any,
    calibration: RawServoCalibration,
    *,
    target_id: int | None,
    surface_id: int | None,
    require_table: bool,
    include_table_geometry: bool = True,
) -> tuple[
    TrackedInstance,
    TrackedInstance | None,
    RawServoObservation,
    bool,
    bool,
]:
    """Build one observation while keeping IDs local to one camera stream."""
    calibration.validate_snapshot(snapshot)
    instances = list(tracker.track(snapshot.rgb))
    target, surface, target_reacquired, surface_reacquired = (
        _select_tracked_instances(
            instances,
            target_id=target_id,
            surface_id=surface_id,
        )
    )
    if target is None:
        raise ValueError("missing tracked target")
    if surface is None and require_table:
        raise ValueError("missing tracked table")
    observation = _observation(
        snapshot,
        target,
        surface,
        calibration,
        include_table_geometry=include_table_geometry,
    )
    if observation.table is None and require_table:
        raise ValueError(
            observation.table_geometry_error or "missing table geometry"
        )
    return (
        target,
        surface,
        observation,
        target_reacquired,
        surface_reacquired,
    )


def _timed_observe_tracked_snapshot(
    *args: Any, **kwargs: Any
) -> tuple[tuple[Any, ...], float]:
    started_at = time.perf_counter()
    result = _observe_tracked_snapshot(*args, **kwargs)
    return result, 1000.0 * (time.perf_counter() - started_at)


def _observe_head_yaw_snapshot(
    snapshot: AlignedRGBDSnapshot,
    tracker: Any,
    calibration: RawServoCalibration,
    *,
    target_id: int | None,
    surface_id: int | None,
) -> tuple[
    TrackedInstance | None,
    TrackedInstance | None,
    TableGeometry | None,
    np.ndarray | None,
    np.ndarray | None,
    str | None,
]:
    """Measure live head-camera desk yaw without requiring a head basket."""
    calibration.validate_snapshot(snapshot)
    instances = list(tracker.track(snapshot.rgb))
    target, surface, _, _ = (
        _select_tracked_instances(
            instances,
            target_id=target_id,
            surface_id=surface_id,
        )
    )
    if surface is None:
        return (
            target,
            None,
            None,
            None,
            None,
            "missing tracked desk for live head yaw",
        )
    table, table_error, desk_mask, table_rgb_edges = _surface_geometry_components(
        snapshot,
        surface,
        calibration,
        target_mask=None if target is None else target.mask,
    )
    return (
        target,
        surface,
        table,
        desk_mask,
        table_rgb_edges,
        None if table is not None else table_error or "missing live head desk geometry",
    )


@dataclass(frozen=True)
class HeadCameraMonitorResult:
    """One result from the text-only head-camera monitor."""

    triggered: bool
    target: TrackedInstance | None
    surface: TrackedInstance | None


class HeadCameraTextMonitor:
    """Track the head stream with the initial target prompt while chest is active."""

    def __init__(
        self,
        tracker: Any,
        target_prompt: str,
        *,
        required_target_frames: int = 1,
        release_missing_frames: int = 3,
    ):
        required = int(required_target_frames)
        release_missing = int(release_missing_frames)
        target = str(target_prompt).strip()
        if not target:
            raise ValueError("head monitor target prompt must be non-empty")
        self.target_prompt = target
        if required <= 0:
            raise ValueError("head reacquisition frame count must be positive")
        if release_missing <= 0:
            raise ValueError("head release missing frame count must be positive")
        self.tracker = tracker
        self.required_target_frames = required
        self.release_missing_frames = release_missing
        self.target_streak = 0
        self.missing_streak = 0
        self.hold_active = False

    def start(self) -> None:
        self.target_streak = 0
        self.missing_streak = 0
        self.hold_active = False
        self.tracker.start_all_text(
            target_prompt=self.target_prompt,
        )

    def note_missing_frame(self) -> None:
        self.target_streak = 0
        self.missing_streak = min(
            self.missing_streak + 1,
            self.release_missing_frames,
        )
        if self.missing_streak >= self.release_missing_frames:
            self.hold_active = False

    def inspect(
        self,
        snapshot: AlignedRGBDSnapshot,
    ) -> HeadCameraMonitorResult:
        instances = list(self.tracker.track(snapshot.rgb))
        target = _best_instance(instances, 0)
        surface = _best_instance(instances, 1)
        if target is None:
            self.note_missing_frame()
        else:
            self.missing_streak = 0
            self.hold_active = True
            self.target_streak += 1
        return HeadCameraMonitorResult(
            triggered=(
                target is not None
                and self.target_streak >= self.required_target_frames
            ),
            target=target,
            surface=surface,
        )


def _capture_head_yaw(
    camera: Any,
    stream_name: str,
    *,
    timeout_ms: int,
    tracker: Any,
    calibration: RawServoCalibration,
    target_id: int | None,
    surface_id: int | None,
) -> tuple[
    AlignedRGBDSnapshot | None,
    TrackedInstance | None,
    TrackedInstance | None,
    TableGeometry | None,
    np.ndarray | None,
    np.ndarray | None,
    str | None,
]:
    snapshot: AlignedRGBDSnapshot | None = None
    try:
        snapshot = camera.capture_stream(stream_name, timeout_ms=timeout_ms)
        target, surface, table, desk_mask, edges, error = (
            _observe_head_yaw_snapshot(
                snapshot,
                tracker,
                calibration,
                target_id=target_id,
                surface_id=surface_id,
            )
        )
    except Exception as exc:
        target = surface = table = desk_mask = edges = None
        error = str(exc)
    return snapshot, target, surface, table, desk_mask, edges, error


def _claim_new_stream_snapshot(
    last_timestamp_by_stream: dict[str, float],
    stream_name: str,
    snapshot: AlignedRGBDSnapshot,
) -> bool:
    """Claim a strictly newer frame for inference on one camera stream."""

    timestamp = float(snapshot.timestamp)
    previous = last_timestamp_by_stream.get(stream_name)
    if previous is not None and timestamp <= previous:
        return False
    last_timestamp_by_stream[stream_name] = timestamp
    return True


def _attempt_details(
    attempt: DualCameraAttempt,
    *,
    target_prompt: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "attempt_id": attempt.attempt_id,
        "live_stream": attempt.live_stream,
        "failover_stage": attempt.stage,
        "origin_stream": attempt.origin_stream,
        "prompt_mode": TEXT_PROMPT_MODE,
        "target_prompt": target_prompt,
        "perception_schedule": "new_frame_latest_only",
        **extra,
    }


def _attempt_frame_details(
    attempt: DualCameraAttempt,
    *,
    camera_stream: str | None = None,
) -> dict[str, Any]:
    return {
        "camera_stream": camera_stream or attempt.live_stream,
        "attempt_id": attempt.attempt_id,
        "failover_stage": attempt.stage,
        "prompt_mode": TEXT_PROMPT_MODE,
    }


def _log_initial_eligibility_summary(
    logger: Callable[[str], None],
    stream_names: Sequence[str],
    eligible_streams: set[str],
    errors: Mapping[str, str],
    selected_stream: str | None,
) -> None:
    states = ", ".join(
        f"{stream_name}="
        + (
            "eligible"
            if stream_name in eligible_streams
            else f"error:{errors.get(stream_name, 'unavailable')}"
        )
        for stream_name in stream_names
    )
    logger(
        "[RawServo] YOLOE initial selection "
        f"selected={selected_stream or 'none'} streams=[{states}]"
    )


def run_dual_raw_servo_worker(
    config: Any,
    requests: queue.Queue[int | None],
    events: queue.Queue[RawServoEvent],
    gate: GenerationGate,
    stop_event: threading.Event,
    *,
    observation_events: queue.Queue[RawServoEvent] | None = None,
    diagnostics: AsyncFrameDiagnosticsWriter | None = None,
    camera_factory: Callable[[], Any],
    table_required: Callable[[], bool],
    position_fallback_allowed: Callable[[], bool],
    tracker_factory: Callable[[], Any] | None = None,
    head_monitor_tracker_factory: Callable[[], Any] | None = None,
    calibration_factory: (
        Callable[[Any], Mapping[str, RawServoCalibration]] | None
    ) = None,
    eligibility_factory: (
        Callable[
            ...,
            tuple[set[str], dict[str, str]],
        ]
        | None
    ) = None,
) -> None:
    """Run one bounded dual-camera failover session per operator generation."""

    logger = diagnostics.logger if diagnostics is not None else print
    calibrations: dict[str, RawServoCalibration] | None = None
    stream_names = (
        str(config.dual_head_camera_stream),
        str(config.dual_chest_camera_stream),
    )
    surface_prompt = config.surface_prompt
    target_prompt = config.target_prompt
    tolerance = config.dual_match_tolerance_frames
    head_reacquire_frames = config.dual_head_reacquire_frames
    head_release_missing_frames = config.dual_head_release_missing_frames
    camera: Any | None = None
    tracker: Any = (
        tracker_factory()
        if tracker_factory is not None
        else YoloePersistentTracker(
            config.raw_yoloe_model_path,
            confidence=config.raw_yoloe_confidence,
            imgsz=config.raw_yoloe_imgsz,
            device=config.raw_yoloe_device,
            surface_prompt=surface_prompt,
        )
    )
    perception_pool = ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="dual-raw-servo-perception",
    )
    head_monitor_tracker: Any | None = None
    try:
        while not stop_event.is_set():
            try:
                generation = requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if generation is None or stop_event.is_set():
                return
            output_dir: Path | None = None
            try:
                if calibrations is None:
                    calibrations = dict(
                        calibration_factory(config)
                        if calibration_factory is not None
                        else dual_calibrations_from_config(config)
                    )
                if camera is None:
                    camera = camera_factory()
                initial_capture = camera.capture()
                stamp = time.strftime("%Y%m%d_%H%M%S")
                output_dir = (
                    Path(config.output_root).resolve()
                    / f"dual_raw_yoloe_{stamp}_g{generation}"
                )
                output_dir.mkdir(parents=True, exist_ok=False)
                if eligibility_factory is not None:
                    eligible_streams, grounding_errors = eligibility_factory(
                        config,
                        initial_capture.snapshots,
                        calibrations,
                        output_dir,
                    )
                else:
                    eligible_streams, grounding_errors = (
                        detect_dual_camera_eligibility(
                            config,
                            initial_capture.snapshots,
                            calibrations,
                            tracker=tracker,
                            logger=logger,
                        )
                    )
                initial_errors = {
                    **dict(initial_capture.errors),
                    **grounding_errors,
                }
                selected_initial_stream = next(
                    (
                        stream_name
                        for stream_name in stream_names
                        if stream_name in eligible_streams
                    ),
                    None,
                )
                _log_initial_eligibility_summary(
                    logger,
                    stream_names,
                    eligible_streams,
                    initial_errors,
                    selected_initial_stream,
                )
                if selected_initial_stream is None:
                    message = "; ".join(
                        f"{name}: {error}"
                        for name, error in sorted(initial_errors.items())
                    ) or "neither camera passed initial text detection"
                    raise RuntimeError(message)
                coordinator = DualCameraFailoverCoordinator(
                    stream_names,
                    eligible_streams,
                )
                head_monitor: HeadCameraTextMonitor | None = None

                attempt = coordinator.start()
                frame_index = -1
                last_inference_timestamp_by_stream: dict[str, float] = {}
                while gate.is_active(generation) and not stop_event.is_set():
                    try:
                        tracker.start_all_text(
                            target_prompt=target_prompt,
                        )
                        chest_fallback_ready = False
                        chest_fallback_init_error: str | None = None
                        if head_monitor_tracker is None:
                            if head_monitor_tracker_factory is not None:
                                head_monitor_tracker = (
                                    head_monitor_tracker_factory()
                                )
                            elif tracker_factory is None:
                                head_monitor_tracker = YoloePersistentTracker(
                                    config.raw_yoloe_model_path,
                                    confidence=config.raw_yoloe_confidence,
                                    imgsz=config.raw_yoloe_imgsz,
                                    device=config.raw_yoloe_device,
                                    surface_prompt=surface_prompt,
                                )
                        if attempt.live_stream == stream_names[1]:
                            if head_monitor_tracker is not None:
                                head_monitor = HeadCameraTextMonitor(
                                    head_monitor_tracker,
                                    target_prompt,
                                    required_target_frames=head_reacquire_frames,
                                    release_missing_frames=(
                                        head_release_missing_frames
                                    ),
                                )
                                head_monitor.start()
                        else:
                            head_monitor = None
                            if head_monitor_tracker is not None:
                                try:
                                    head_monitor_tracker.start_all_text(
                                        target_prompt=target_prompt
                                    )
                                    chest_fallback_ready = True
                                except Exception as exc:
                                    chest_fallback_init_error = str(exc)
                    except Exception as exc:
                        failure_reason = (
                            f"{attempt.stage} initialization failed: {exc}"
                        )
                    else:
                        failure_reason = ""
                    if stop_event.is_set() or not gate.is_active(generation):
                        break
                    if not failure_reason:
                        _publish_worker_event(
                            events,
                            observation_events,
                            diagnostics,
                            RawServoEvent(
                                generation,
                                "detecting",
                                output_dir=str(output_dir),
                                details=_attempt_details(
                                    attempt,
                                    target_prompt=target_prompt,
                                    chest_position_fallback_available=(
                                        chest_fallback_ready
                                    ),
                                    chest_position_fallback_error=(
                                        chest_fallback_init_error
                                    ),
                                ),
                            ),
                        )
                    invalid_frames = 0
                    initialized = False
                    target_id: int | None = None
                    surface_id: int | None = None
                    fallback_target_id: int | None = None
                    fallback_surface_id: int | None = None
                    position_source_stream = attempt.live_stream
                    joint_chest_tracking_active = False
                    joint_chest_missing_frames = 0
                    joint_head_yaw_missing_frames = 0
                    failure_frame = None
                    preempt_attempt: DualCameraAttempt | None = None
                    switch_details: dict[str, Any] = {}
                    while (
                        not failure_reason
                        and gate.is_active(generation)
                        and not stop_event.is_set()
                    ):
                        snapshot: AlignedRGBDSnapshot | None = None
                        target: TrackedInstance | None = None
                        surface: TrackedInstance | None = None
                        observation = None
                        hard_failure = False
                        target_reacquired = False
                        surface_reacquired = False
                        monitor_details: dict[str, Any] = {}
                        basket_source_details: dict[str, Any] = {}
                        yaw_source_details: dict[str, Any] = {}
                        head_snapshot: AlignedRGBDSnapshot | None = None
                        live_head_snapshot: AlignedRGBDSnapshot | None = None
                        live_head_target: TrackedInstance | None = None
                        live_head_surface: TrackedInstance | None = None
                        live_head_table: TableGeometry | None = None
                        live_head_desk_mask: np.ndarray | None = None
                        live_head_table_rgb_edges: np.ndarray | None = None
                        live_head_yaw_error: str | None = None
                        monitor_result: HeadCameraMonitorResult | None = None
                        fallback_head_yaw_attempted = False
                        joint_chest_target_missing = False
                        chest_active = attempt.live_stream == stream_names[1]
                        require_table = (
                            False
                            if chest_active
                            else bool(table_required())
                        )
                        fallback_allowed = (
                            attempt.live_stream == stream_names[0]
                            and chest_fallback_ready
                            and bool(position_fallback_allowed())
                        )
                        control_source_stream = (
                            position_source_stream
                            if fallback_allowed
                            else attempt.live_stream
                        )
                        source_is_active = (
                            control_source_stream == attempt.live_stream
                        )
                        source_tracker = (
                            tracker
                            if source_is_active
                            else head_monitor_tracker
                        )
                        source_target_id = (
                            target_id
                            if source_is_active
                            else fallback_target_id
                        )
                        source_surface_id = (
                            surface_id
                            if source_is_active
                            else fallback_surface_id
                        )
                        previous_target_id = source_target_id
                        previous_surface_id = source_surface_id
                        position_future: Any | None = None
                        parallel_started_at: float | None = None
                        head_branch_started_at: float | None = None
                        head_branch_elapsed_ms: float | None = None
                        try:
                            snapshot = camera.capture_stream(
                                control_source_stream,
                                timeout_ms=max(
                                    1,
                                    int(float(config.raw_camera_stale_s) * 1000.0),
                                ),
                            )
                            if not _claim_new_stream_snapshot(
                                last_inference_timestamp_by_stream,
                                control_source_stream,
                                snapshot,
                            ):
                                continue
                            frame_index += 1
                            parallel_joint_observation = (
                                control_source_stream == stream_names[1]
                                and (
                                    (
                                        attempt.live_stream == stream_names[1]
                                        and head_monitor is not None
                                    )
                                    or attempt.live_stream == stream_names[0]
                                )
                            )
                            if parallel_joint_observation:
                                parallel_started_at = time.perf_counter()
                                position_future = perception_pool.submit(
                                    _timed_observe_tracked_snapshot,
                                    snapshot,
                                    source_tracker,
                                    calibrations[control_source_stream],
                                    target_id=source_target_id,
                                    surface_id=source_surface_id,
                                    require_table=False,
                                    include_table_geometry=False,
                                )
                            if (
                                attempt.live_stream == stream_names[1]
                                and head_monitor is not None
                            ):
                                head_branch_started_at = time.perf_counter()
                                was_holding = head_monitor.hold_active
                                try:
                                    head_snapshot = camera.poll_stream(
                                        stream_names[0]
                                    )
                                    if head_snapshot is None:
                                        raise BasePoseCameraError(
                                            "waiting for newer head-monitor RGB-D"
                                        )
                                    calibrations[stream_names[0]].validate_snapshot(
                                        head_snapshot
                                    )
                                    monitor_result = head_monitor.inspect(
                                        head_snapshot
                                    )
                                except Exception as exc:
                                    head_monitor.note_missing_frame()
                                    monitor_state = {
                                        "status": "error",
                                        "target_streak": (
                                            head_monitor.target_streak
                                        ),
                                        "required_target_frames": (
                                            head_reacquire_frames
                                        ),
                                        "missing_streak": (
                                            head_monitor.missing_streak
                                        ),
                                        "release_missing_frames": (
                                            head_release_missing_frames
                                        ),
                                        "hold_active": (
                                            head_monitor.hold_active
                                            and not joint_chest_tracking_active
                                        ),
                                        "error": str(exc),
                                    }
                                else:
                                    if monitor_result.surface is None:
                                        live_head_yaw_error = (
                                            "missing tracked desk for live head yaw"
                                        )
                                    else:
                                        try:
                                            (
                                                live_head_table,
                                                live_head_yaw_error,
                                                live_head_desk_mask,
                                                live_head_table_rgb_edges,
                                            ) = _surface_geometry_components(
                                                head_snapshot,
                                                monitor_result.surface,
                                                calibrations[stream_names[0]],
                                                target_mask=(
                                                    None
                                                    if monitor_result.target is None
                                                    else monitor_result.target.mask
                                                ),
                                            )
                                        except Exception as exc:
                                            live_head_table = None
                                            live_head_desk_mask = None
                                            live_head_table_rgb_edges = None
                                            live_head_yaw_error = str(exc)
                                    effective_hold = (
                                        head_monitor.hold_active
                                        and live_head_table is None
                                        and not joint_chest_tracking_active
                                    )
                                    monitor_state = {
                                        "status": (
                                            "joint_tracking"
                                            if live_head_table is not None
                                            else "triggered"
                                            if monitor_result.triggered
                                            else "tracking"
                                        ),
                                        "target_streak": (
                                            head_monitor.target_streak
                                        ),
                                        "required_target_frames": (
                                            head_reacquire_frames
                                        ),
                                        "missing_streak": (
                                            head_monitor.missing_streak
                                        ),
                                        "release_missing_frames": (
                                            head_release_missing_frames
                                        ),
                                        "hold_active": (
                                            effective_hold
                                        ),
                                        "desk_detected": (
                                            monitor_result.surface is not None
                                        ),
                                        "yaw_valid": live_head_table is not None,
                                        "yaw_error": live_head_yaw_error,
                                    }
                                monitor_details["head_monitor"] = monitor_state
                                if (
                                    bool(monitor_state.get("hold_active", False))
                                    and (
                                        not was_holding
                                        or (
                                            monitor_result is not None
                                            and monitor_result.triggered
                                        )
                                    )
                                ):
                                    _publish_worker_event(
                                        events,
                                        observation_events,
                                        diagnostics,
                                        RawServoEvent(
                                            generation=generation,
                                            kind=HEAD_MONITOR_HOLD_EVENT,
                                            output_dir=str(output_dir),
                                            details=_attempt_details(
                                                attempt,
                                                target_prompt=target_prompt,
                                                head_monitor=monitor_state,
                                            ),
                                        ),
                                    )
                                if monitor_result is not None:
                                    if (
                                        monitor_result.triggered
                                        and live_head_table is None
                                        and not joint_chest_tracking_active
                                    ):
                                        preempt_attempt = (
                                            coordinator
                                            .begin_head_monitor_reacquisition()
                                        )
                                        failure_reason = (
                                            "head target reacquired for "
                                            f"{head_reacquire_frames} consecutive "
                                            "frames using text detection"
                                        )
                                        switch_details = monitor_details
                                        failure_frame = _diagnostic_frame(
                                            frame_index,
                                            head_snapshot,
                                            monitor_result.target,
                                            monitor_result.surface,
                                            None,
                                            kind="switching",
                                            error=failure_reason,
                                            **_attempt_frame_details(
                                                preempt_attempt
                                            ),
                                        )
                                        if position_future is not None:
                                            try:
                                                position_future.result()
                                            except Exception:
                                                pass
                                        break
                                head_branch_elapsed_ms = 1000.0 * (
                                    time.perf_counter() - head_branch_started_at
                                )
                            if (
                                attempt.live_stream == stream_names[0]
                                and control_source_stream == stream_names[1]
                            ):
                                fallback_head_yaw_attempted = True
                                head_branch_started_at = time.perf_counter()
                                (
                                    live_head_snapshot,
                                    live_head_target,
                                    live_head_surface,
                                    live_head_table,
                                    live_head_desk_mask,
                                    live_head_table_rgb_edges,
                                    live_head_yaw_error,
                                ) = _capture_head_yaw(
                                    camera,
                                    stream_names[0],
                                    timeout_ms=max(
                                        1,
                                        int(config.raw_camera_stale_s * 1000.0),
                                    ),
                                    tracker=tracker,
                                    calibration=calibrations[stream_names[0]],
                                    target_id=target_id,
                                    surface_id=surface_id,
                                )
                                head_branch_elapsed_ms = 1000.0 * (
                                    time.perf_counter() - head_branch_started_at
                                )
                            if position_future is None:
                                position_result = _observe_tracked_snapshot(
                                    snapshot,
                                    source_tracker,
                                    calibrations[control_source_stream],
                                    target_id=source_target_id,
                                    surface_id=source_surface_id,
                                    require_table=(
                                        require_table if source_is_active else False
                                    ),
                                    include_table_geometry=(
                                        control_source_stream != stream_names[1]
                                    ),
                                )
                            else:
                                (
                                    position_result,
                                    position_branch_elapsed_ms,
                                ) = position_future.result()
                                assert parallel_started_at is not None
                                monitor_details["parallel_perception"] = {
                                    "enabled": True,
                                    "position_stream": control_source_stream,
                                    "yaw_stream": stream_names[0],
                                    "position_branch_ms": (
                                        position_branch_elapsed_ms
                                    ),
                                    "head_branch_ms": head_branch_elapsed_ms,
                                    "joint_wall_ms": 1000.0
                                    * (
                                        time.perf_counter()
                                        - parallel_started_at
                                    ),
                                    "chest_table_geometry_skipped": True,
                                }
                            (
                                target,
                                surface,
                                observation,
                                target_reacquired,
                                surface_reacquired,
                            ) = position_result
                            if attempt.live_stream == stream_names[0]:
                                basket_source_details = {
                                    "preferred_stream": control_source_stream,
                                    "selected_stream": control_source_stream,
                                    "source_switched": False,
                                    f"{control_source_stream}_valid": True,
                                    "chest_fallback_used": (
                                        control_source_stream == stream_names[1]
                                    ),
                                    "without_stage_switch": True,
                                }
                        except (BasePoseCameraError, ValueError) as exc:
                            primary_error = str(exc)
                            joint_chest_target_missing = (
                                joint_chest_tracking_active
                                and control_source_stream == stream_names[1]
                                and primary_error == "missing tracked target"
                            )
                            if joint_chest_target_missing:
                                # Once chest position and live head yaw have
                                # formed a joint observation, do not abandon the
                                # chest target on a single dropped detection.
                                perception_error = primary_error
                                hard_failure = False
                                basket_source_details = {
                                    "preferred_stream": stream_names[1],
                                    "selected_stream": None,
                                    "source_switched": False,
                                    f"{stream_names[1]}_valid": False,
                                    f"{stream_names[1]}_error": primary_error,
                                    "chest_fallback_used": False,
                                    "without_stage_switch": True,
                                }
                            elif fallback_allowed:
                                alternate_stream = (
                                    stream_names[1]
                                    if control_source_stream == stream_names[0]
                                    else stream_names[0]
                                )
                                alternate_is_active = (
                                    alternate_stream == attempt.live_stream
                                )
                                alternate_tracker = (
                                    tracker
                                    if alternate_is_active
                                    else head_monitor_tracker
                                )
                                alternate_target_id = (
                                    target_id
                                    if alternate_is_active
                                    else fallback_target_id
                                )
                                alternate_surface_id = (
                                    surface_id
                                    if alternate_is_active
                                    else fallback_surface_id
                                )
                                try:
                                    fallback_snapshot = camera.capture_stream(
                                        alternate_stream,
                                        timeout_ms=max(
                                            1,
                                            int(
                                                float(config.raw_camera_stale_s)
                                                * 1000.0
                                            ),
                                        ),
                                    )
                                    (
                                        fallback_target,
                                        fallback_surface,
                                        fallback_observation,
                                        fallback_target_reacquired,
                                        fallback_surface_reacquired,
                                    ) = _observe_tracked_snapshot(
                                        fallback_snapshot,
                                        alternate_tracker,
                                        calibrations[alternate_stream],
                                        target_id=alternate_target_id,
                                        surface_id=alternate_surface_id,
                                        require_table=False,
                                        include_table_geometry=(
                                            alternate_stream != stream_names[1]
                                        ),
                                    )
                                except Exception as fallback_exc:
                                    perception_error = (
                                        f"{control_source_stream} basket invalid: "
                                        f"{primary_error}; {alternate_stream} "
                                        "basket invalid: "
                                        f"{fallback_exc}"
                                    )
                                    hard_failure = (
                                        isinstance(exc, BasePoseCameraError)
                                        and isinstance(
                                            fallback_exc,
                                            BasePoseCameraError,
                                        )
                                    )
                                    basket_source_details = {
                                        "preferred_stream": (
                                            control_source_stream
                                        ),
                                        "selected_stream": None,
                                        "source_switched": False,
                                        f"{control_source_stream}_valid": False,
                                        f"{control_source_stream}_error": (
                                            primary_error
                                        ),
                                        f"{alternate_stream}_valid": False,
                                        f"{alternate_stream}_error": str(
                                            fallback_exc
                                        ),
                                        "chest_fallback_used": False,
                                        "without_stage_switch": True,
                                    }
                                else:
                                    if snapshot is None:
                                        frame_index += 1
                                    snapshot = fallback_snapshot
                                    target = fallback_target
                                    surface = fallback_surface
                                    observation = fallback_observation
                                    target_reacquired = (
                                        fallback_target_reacquired
                                    )
                                    surface_reacquired = (
                                        fallback_surface_reacquired
                                    )
                                    previous_target_id = alternate_target_id
                                    previous_surface_id = alternate_surface_id
                                    previous_source_stream = (
                                        control_source_stream
                                    )
                                    control_source_stream = alternate_stream
                                    position_source_stream = alternate_stream
                                    perception_error = ""
                                    hard_failure = False
                                    basket_source_details = {
                                        "preferred_stream": (
                                            previous_source_stream
                                        ),
                                        "selected_stream": alternate_stream,
                                        "source_switched": True,
                                        f"{previous_source_stream}_valid": False,
                                        f"{previous_source_stream}_error": (
                                            primary_error
                                        ),
                                        f"{alternate_stream}_valid": True,
                                        "chest_fallback_used": (
                                            alternate_stream == stream_names[1]
                                        ),
                                        "without_stage_switch": True,
                                    }
                            else:
                                perception_error = primary_error
                                hard_failure = isinstance(
                                    exc,
                                    BasePoseCameraError,
                                )
                        except Exception as exc:
                            perception_error = str(exc)
                            hard_failure = True
                        else:
                            perception_error = ""

                        if (
                            not perception_error
                            and observation is not None
                            and attempt.live_stream == stream_names[1]
                            and live_head_table is not None
                            and head_snapshot is not None
                        ):
                            observation = replace(
                                observation,
                                table=live_head_table,
                                table_geometry_error=None,
                                desk_mask=live_head_desk_mask,
                                table_rgb_edges=None,
                                table_camera_stream=stream_names[0],
                            )
                            yaw_source_details = {
                                "stream": stream_names[0],
                                "realtime": True,
                                "valid": True,
                                "camera_timestamp": head_snapshot.timestamp,
                                "surface_track_id": (
                                    None
                                    if monitor_result is None
                                    or monitor_result.surface is None
                                    else monitor_result.surface.track_id
                                ),
                            }
                        elif (
                            not perception_error
                            and observation is not None
                            and attempt.live_stream == stream_names[1]
                        ):
                            yaw_source_details = {
                                "stream": stream_names[0],
                                "realtime": True,
                                "valid": False,
                                "error": (
                                    live_head_yaw_error
                                    or "live head yaw unavailable"
                                ),
                            }
                        elif (
                            not perception_error
                            and observation is not None
                            and attempt.live_stream == stream_names[0]
                            and control_source_stream == stream_names[1]
                        ):
                            if not fallback_head_yaw_attempted:
                                (
                                    live_head_snapshot,
                                    live_head_target,
                                    live_head_surface,
                                    live_head_table,
                                    live_head_desk_mask,
                                    live_head_table_rgb_edges,
                                    live_head_yaw_error,
                                ) = _capture_head_yaw(
                                    camera,
                                    stream_names[0],
                                    timeout_ms=max(
                                        1,
                                        int(config.raw_camera_stale_s * 1000.0),
                                    ),
                                    tracker=tracker,
                                    calibration=calibrations[stream_names[0]],
                                    target_id=target_id,
                                    surface_id=surface_id,
                                )
                            if (
                                live_head_yaw_error is not None
                                or live_head_snapshot is None
                                or live_head_surface is None
                                or live_head_table is None
                            ):
                                error = (
                                    live_head_yaw_error
                                    or "live head yaw unavailable"
                                )
                                yaw_source_details = {
                                    "stream": stream_names[0],
                                    "realtime": True,
                                    "valid": False,
                                    "error": error,
                                }
                                if not joint_chest_tracking_active:
                                    perception_error = (
                                        f"live head yaw invalid: {error}"
                                    )
                                    hard_failure = False
                            else:
                                if live_head_target is not None:
                                    target_id = live_head_target.track_id
                                surface_id = live_head_surface.track_id
                                observation = replace(
                                    observation,
                                    table=live_head_table,
                                    table_geometry_error=None,
                                    desk_mask=live_head_desk_mask,
                                    table_rgb_edges=None,
                                    table_camera_stream=stream_names[0],
                                )
                                yaw_source_details = {
                                    "stream": stream_names[0],
                                    "realtime": True,
                                    "valid": True,
                                    "camera_timestamp": (
                                        live_head_snapshot.timestamp
                                    ),
                                    "surface_track_id": (
                                        live_head_surface.track_id
                                    ),
                                }
                        elif (
                            not perception_error
                            and observation is not None
                            and attempt.live_stream == stream_names[0]
                        ):
                            yaw_source_details = {
                                "stream": stream_names[0],
                                "realtime": True,
                                "valid": observation.table is not None,
                                "camera_timestamp": snapshot.timestamp,
                                "surface_track_id": (
                                    None
                                    if surface is None
                                    else surface.track_id
                                ),
                                "error": observation.table_geometry_error,
                            }

                        head_yaw_loss: dict[str, Any] | None = None
                        if (
                            not perception_error
                            and observation is not None
                            and control_source_stream == stream_names[1]
                        ):
                            joint_chest_missing_frames = 0
                            if bool(yaw_source_details.get("valid", False)):
                                joint_chest_tracking_active = True
                                joint_head_yaw_missing_frames = 0
                            elif joint_chest_tracking_active:
                                joint_head_yaw_missing_frames += 1
                                head_yaw_loss = _joint_head_yaw_loss_details(
                                    joint_head_yaw_missing_frames
                                )
                                if head_yaw_loss["stage"] == "switch_state":
                                    failure_reason = (
                                        "head yaw missing for 20 consecutive "
                                        "frames after joint observation"
                                    )
                                    switch_details = {
                                        "head_yaw_loss": head_yaw_loss,
                                    }
                                    failure_frame = _diagnostic_frame(
                                        frame_index,
                                        snapshot,
                                        target,
                                        surface,
                                        observation,
                                        kind="switching",
                                        error=failure_reason,
                                        **_attempt_frame_details(
                                            attempt,
                                            camera_stream=control_source_stream,
                                        ),
                                    )
                                    break
                        elif control_source_stream != stream_names[1]:
                            joint_head_yaw_missing_frames = 0

                        if perception_error:
                            diagnostic_frame = (
                                None
                                if snapshot is None
                                else _diagnostic_frame(
                                    frame_index,
                                    snapshot,
                                    target,
                                    surface,
                                    None,
                                    kind="invalid",
                                    error=perception_error,
                                    **_attempt_frame_details(
                                        attempt,
                                        camera_stream=control_source_stream,
                                    ),
                                )
                            )
                            if hard_failure:
                                failure_reason = perception_error
                                failure_frame = diagnostic_frame
                                break
                            invalid_frames += 1
                            chest_target_loss: dict[str, Any] | None = None
                            switch_position_to_head = False
                            if joint_chest_target_missing:
                                joint_chest_missing_frames += 1
                                missing_frames = joint_chest_missing_frames
                                total_loss_frames = (
                                    JOINT_CHEST_LOSS_GRACE_FRAMES
                                    + JOINT_CHEST_LOSS_ZERO_HOLD_FRAMES
                                )
                                if missing_frames >= total_loss_frames:
                                    loss_stage = "switch_to_head"
                                elif (
                                    missing_frames
                                    >= JOINT_CHEST_LOSS_GRACE_FRAMES
                                ):
                                    loss_stage = "zero_hold"
                                else:
                                    loss_stage = "grace"
                                chest_target_loss = {
                                    "missing_frames": missing_frames,
                                    "grace_frames": (
                                        JOINT_CHEST_LOSS_GRACE_FRAMES
                                    ),
                                    "zero_hold_frames": (
                                        JOINT_CHEST_LOSS_ZERO_HOLD_FRAMES
                                    ),
                                    "stage": loss_stage,
                                }
                                if missing_frames >= total_loss_frames:
                                    if attempt.live_stream == stream_names[0]:
                                        switch_position_to_head = True
                                    else:
                                        failure_reason = (
                                            "chest target missing for 10 consecutive "
                                            "frames after joint observation"
                                        )
                                        failure_frame = diagnostic_frame
                                        break
                            elif invalid_frames >= tolerance:
                                failure_reason = perception_error
                                failure_frame = diagnostic_frame
                                break
                            if diagnostic_frame is not None:
                                _publish_worker_event(
                                    events,
                                    observation_events,
                                    diagnostics,
                                    RawServoEvent(
                                        generation,
                                        "invalid",
                                        output_dir=str(output_dir),
                                        error=perception_error,
                                        hard=False,
                                        details=_attempt_details(
                                            attempt,
                                            target_prompt=target_prompt,
                                            match_invalid_frames=invalid_frames,
                                            control_source_stream=(
                                                control_source_stream
                                            ),
                                            basket_source=basket_source_details,
                                            yaw_source=yaw_source_details,
                                            chest_target_loss=(
                                                chest_target_loss
                                            ),
                                            **monitor_details,
                                        ),
                                        frame=diagnostic_frame,
                                    ),
                                )
                            if switch_position_to_head:
                                position_source_stream = stream_names[0]
                                joint_chest_tracking_active = False
                                joint_chest_missing_frames = 0
                            continue

                        assert snapshot is not None
                        assert target is not None
                        assert observation is not None
                        invalid_frames = 0
                        if not initialized:
                            coordinator.mark_success(attempt)
                            initialized = True
                            event_kind = "initialized"
                        else:
                            event_kind = "observation"
                        if control_source_stream == attempt.live_stream:
                            target_id = target.track_id
                            if surface is not None:
                                surface_id = surface.track_id
                        else:
                            fallback_target_id = target.track_id
                            fallback_surface_id = (
                                None if surface is None else surface.track_id
                            )
                        details = _attempt_details(
                            attempt,
                            target_prompt=target_prompt,
                            match_invalid_frames=0,
                            control_source_stream=control_source_stream,
                            basket_source=basket_source_details,
                            yaw_source=yaw_source_details,
                            head_yaw_loss=head_yaw_loss,
                            target_track_id=target.track_id,
                            surface_track_id=(
                                None if surface is None else surface.track_id
                            ),
                            **monitor_details,
                        )
                        if target_reacquired:
                            details.update(
                                {
                                    "target_reacquired": True,
                                    "previous_target_id": previous_target_id,
                                }
                            )
                        if surface_reacquired:
                            details.update(
                                {
                                    "surface_reacquired": True,
                                    "previous_surface_id": previous_surface_id,
                                }
                            )
                        _publish_worker_event(
                            events,
                            observation_events,
                            diagnostics,
                            RawServoEvent(
                                generation,
                                event_kind,
                                observation=observation,
                                output_dir=str(output_dir),
                                details=details,
                                frame=_diagnostic_frame(
                                    frame_index,
                                    snapshot,
                                    target,
                                    surface,
                                    observation,
                                    kind=event_kind,
                                    **_attempt_frame_details(
                                        attempt,
                                        camera_stream=control_source_stream,
                                    ),
                                ),
                            ),
                        )
                    if stop_event.is_set() or not gate.is_active(generation):
                        break
                    next_attempt = (
                        preempt_attempt
                        or coordinator.advance_after_failure(attempt)
                    )
                    if next_attempt is None:
                        terminal_error = (
                            "both-camera text failover exhausted: "
                            f"{failure_reason or 'perception failed'}"
                        )
                        _publish_worker_event(
                            events,
                            observation_events,
                            diagnostics,
                            RawServoEvent(
                                generation,
                                "error",
                                output_dir=str(output_dir),
                                error=terminal_error,
                                hard=True,
                                details=_attempt_details(
                                    attempt,
                                    target_prompt=target_prompt,
                                    terminal_cause=(
                                        failure_reason or "perception failed"
                                    ),
                                ),
                                frame=failure_frame,
                            ),
                        )
                        gate.cancel(generation)
                        break
                    attempt = next_attempt
                    _publish_worker_event(
                        events,
                        observation_events,
                        diagnostics,
                        RawServoEvent(
                            generation,
                            "switching",
                            output_dir=str(output_dir),
                            error=failure_reason or "perception failed",
                            hard=False,
                            details=_attempt_details(
                                attempt,
                                target_prompt=target_prompt,
                                **switch_details,
                            ),
                            frame=failure_frame,
                        ),
                    )
            except Exception as exc:
                _publish_worker_event(
                    events,
                    observation_events,
                    diagnostics,
                    RawServoEvent(
                        generation,
                        "error",
                        output_dir=None if output_dir is None else str(output_dir),
                        error=str(exc),
                        hard=True,
                    ),
                )
                gate.cancel(generation)
    finally:
        perception_pool.shutdown(wait=True, cancel_futures=True)
        if camera is not None:
            camera.close()
