"""Dual-camera reference selection and failover for raw YOLOE BasePose."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
import math
import multiprocessing
import os
import queue
import signal
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gear_sonic.camera.calibration import (
    CameraCalibrationError,
    load_camera_intrinsics,
)
from gear_sonic.utils.inference.base_pose import (
    AlignedRGBDSnapshot,
    BasePoseCameraError,
    DualAlignedRGBDCamera,
    _write_json,
)
from gear_sonic.utils.inference.base_pose_visual_servo import (
    GenerationGate,
    RawServoEvent,
    RawServoCalibration,
    TrackedInstance,
    YoloePersistentTracker,
    _diagnostic_frame,
    _observation,
    _publish_worker_event,
    _resolve_target,
    _save_png,
    ground_raw_servo_references,
)
from gear_sonic.utils.inference.base_pose_visual_servo_diagnostics import (
    AsyncFrameDiagnosticsWriter,
)


def _reference_bbox(
    value: Sequence[float],
    *,
    field: str,
) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError(f"{field} must contain four values")
    bbox = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in bbox):
        raise ValueError(f"{field} must be finite")
    x1, y1, x2, y2 = bbox
    if not (0.0 <= x1 < x2 <= 1000.0 and 0.0 <= y1 < y2 <= 1000.0):
        raise ValueError(f"{field} must be an ordered normalized bbox")
    return bbox


@dataclass(frozen=True)
class DualCameraReference:
    """One coherent visual prompt whose image and boxes share a camera frame."""

    stream_name: str
    rgb: np.ndarray
    target_prompt: str
    target_bbox: tuple[float, float, float, float]
    table_bboxes: tuple[tuple[float, float, float, float], ...]
    camera_timestamp: float
    kind: str = "initial"

    def __post_init__(self) -> None:
        stream_name = str(self.stream_name).strip()
        if not stream_name:
            raise ValueError("reference stream_name must be non-empty")
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("reference RGB must be an HxWx3 uint8 image")
        target_prompt = str(self.target_prompt).strip()
        if not target_prompt:
            raise ValueError("reference target_prompt must be non-empty")
        target_bbox = _reference_bbox(self.target_bbox, field="target_bbox")
        table_bboxes = tuple(
            _reference_bbox(value, field="table_bbox")
            for value in self.table_bboxes
        )
        if not table_bboxes:
            raise ValueError("reference requires at least one table bbox")
        timestamp = float(self.camera_timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("reference camera_timestamp must be finite")
        kind = str(self.kind)
        if kind not in {"initial", "latest"}:
            raise ValueError("reference kind must be initial or latest")
        object.__setattr__(self, "stream_name", stream_name)
        object.__setattr__(self, "rgb", rgb.copy())
        object.__setattr__(self, "target_prompt", target_prompt)
        object.__setattr__(self, "target_bbox", target_bbox)
        object.__setattr__(self, "table_bboxes", table_bboxes)
        object.__setattr__(self, "camera_timestamp", timestamp)
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True)
class DualCameraAttempt:
    attempt_id: int
    live_stream: str
    reference: DualCameraReference
    stage: str
    origin_stream: str


class DualCameraFailoverCoordinator:
    """Select the active view and bounded reference fallback sequence."""

    def __init__(
        self,
        stream_names: Sequence[str],
        initial_references: Mapping[str, DualCameraReference],
    ):
        names = tuple(str(name) for name in stream_names)
        if len(names) != 2 or len(set(names)) != 2 or any(not name for name in names):
            raise ValueError("dual-camera failover requires two distinct streams")
        initial = dict(initial_references)
        for stream_name, reference in initial.items():
            if stream_name not in names or reference.stream_name != stream_name:
                raise ValueError("initial reference stream does not match its key")
            if reference.kind != "initial":
                raise ValueError("initial reference bank only accepts initial references")
        self.stream_names = names
        self.initial_references = initial
        self.latest_references: dict[str, DualCameraReference] = {}
        self._next_attempt_id = 1
        self._active_attempt_id: int | None = None

    def _attempt(
        self,
        live_stream: str,
        reference: DualCameraReference,
        *,
        stage: str,
        origin_stream: str,
    ) -> DualCameraAttempt:
        attempt = DualCameraAttempt(
            attempt_id=self._next_attempt_id,
            live_stream=live_stream,
            reference=reference,
            stage=stage,
            origin_stream=origin_stream,
        )
        self._next_attempt_id += 1
        self._active_attempt_id = None
        return attempt

    def start(self) -> DualCameraAttempt:
        for stream_name in self.stream_names:
            reference = self.initial_references.get(stream_name)
            if reference is not None:
                return self._attempt(
                    stream_name,
                    reference,
                    stage="initial",
                    origin_stream=stream_name,
                )
        raise RuntimeError("neither camera has a valid initial reference")

    def save_latest(self, reference: DualCameraReference) -> None:
        if reference.stream_name not in self.stream_names:
            raise ValueError("latest reference belongs to an unknown stream")
        if reference.kind != "latest":
            raise ValueError("latest reference bank only accepts latest references")
        self.latest_references[reference.stream_name] = reference

    def mark_success(self, attempt: DualCameraAttempt) -> None:
        if attempt.live_stream not in self.stream_names:
            raise ValueError("successful attempt belongs to an unknown stream")
        self._active_attempt_id = attempt.attempt_id

    def _other(self, stream_name: str) -> str:
        if stream_name == self.stream_names[0]:
            return self.stream_names[1]
        if stream_name == self.stream_names[1]:
            return self.stream_names[0]
        raise ValueError("attempt belongs to an unknown stream")

    def advance_after_failure(
        self,
        attempt: DualCameraAttempt,
    ) -> DualCameraAttempt | None:
        if self._active_attempt_id == attempt.attempt_id or attempt.stage == "initial":
            origin = attempt.live_stream
            alternate = self._other(origin)
            reference = self.initial_references.get(alternate)
            if reference is None:
                reference = self.initial_references.get(origin)
            if reference is None:
                return None
            return self._attempt(
                alternate,
                reference,
                stage="alternate_initial",
                origin_stream=origin,
            )
        if attempt.stage == "alternate_initial":
            origin = attempt.origin_stream
            reference = self.latest_references.get(origin)
            if reference is None:
                reference = self.initial_references.get(origin)
            if reference is None:
                return None
            return self._attempt(
                origin,
                reference,
                stage="origin_latest",
                origin_stream=origin,
            )
        if attempt.stage == "origin_latest":
            return None
        raise ValueError(f"unsupported failover stage: {attempt.stage}")


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
            camera_height_m=float(config.camera_height_m),
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
            camera_height_m=float(config.dual_chest_camera_height_m),
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


def ground_dual_raw_servo_references(
    config: Any,
    snapshots: Mapping[str, AlignedRGBDSnapshot],
    calibrations: Mapping[str, RawServoCalibration],
    output_dir: str | Path,
    *,
    client_factory: Callable[[], Any] | None = None,
    initialization_grace_s: float = 30.0,
) -> tuple[dict[str, DualCameraReference], dict[str, str]]:
    """Ground both views, bounding the second result after one view succeeds."""

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    grace_s = float(initialization_grace_s)
    if not math.isfinite(grace_s) or grace_s <= 0.0:
        raise ValueError("dual initialization grace must be finite and positive")

    stream_names = tuple(name for name in calibrations if name in snapshots)
    if not stream_names:
        return {}, {}

    if client_factory is None:
        return _ground_dual_in_processes(
            config,
            snapshots,
            calibrations,
            root,
            stream_names=stream_names,
            initialization_grace_s=grace_s,
        )
    return _ground_dual_in_threads(
        config,
        snapshots,
        calibrations,
        root,
        stream_names=stream_names,
        client_factory=client_factory,
        initialization_grace_s=grace_s,
    )


def _ground_camera_reference(
    config: Any,
    stream_name: str,
    snapshot: AlignedRGBDSnapshot,
    calibration: RawServoCalibration,
    root: Path,
    *,
    client_factory: Callable[[], Any] | None = None,
) -> DualCameraReference:
    calibration.validate_snapshot(snapshot)
    stream_dir = root / stream_name
    stream_dir.mkdir(parents=True, exist_ok=False)
    rgb_path = stream_dir / "initial_rgb.png"
    _save_png(rgb_path, snapshot.rgb, rgb=True)
    assert snapshot.depth_raw is not None
    _save_png(stream_dir / "initial_depth_raw.png", snapshot.depth_raw)
    spec, table_bboxes = ground_raw_servo_references(
        config,
        rgb_path,
        stream_dir,
        client_factory=client_factory,
        calibration=calibration,
    )
    return DualCameraReference(
        stream_name=stream_name,
        rgb=snapshot.rgb,
        target_prompt=spec.target_prompt,
        target_bbox=spec.target_bbox,
        table_bboxes=table_bboxes,
        camera_timestamp=snapshot.timestamp,
        kind="initial",
    )


def _timeout_error(grace_s: float, successful_stream: str) -> str:
    return (
        "initial grounding exceeded "
        f"{grace_s:g}s after {successful_stream} became eligible; "
        "initialization process terminated"
    )


def _ground_dual_in_threads(
    config: Any,
    snapshots: Mapping[str, AlignedRGBDSnapshot],
    calibrations: Mapping[str, RawServoCalibration],
    root: Path,
    *,
    stream_names: Sequence[str],
    client_factory: Callable[[], Any],
    initialization_grace_s: float,
) -> tuple[dict[str, DualCameraReference], dict[str, str]]:
    """Injected-client path; production uses killable processes below."""

    references: dict[str, DualCameraReference] = {}
    errors: dict[str, str] = {}
    pool = ThreadPoolExecutor(
        max_workers=len(stream_names),
        thread_name_prefix="dual-raw-servo-grounding",
    )
    pending = {
        pool.submit(
            _ground_camera_reference,
            config,
            stream_name,
            snapshots[stream_name],
            calibrations[stream_name],
            root,
            client_factory=client_factory,
        ): stream_name
        for stream_name in stream_names
    }
    successful_stream: str | None = None
    deadline: float | None = None
    timed_out = False
    try:
        while pending:
            timeout = (
                None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            done, _ = wait(
                tuple(pending), timeout=timeout, return_when=FIRST_COMPLETED
            )
            if not done:
                assert successful_stream is not None
                for future, stream_name in tuple(pending.items()):
                    future.cancel()
                    errors[stream_name] = _timeout_error(
                        initialization_grace_s, successful_stream
                    )
                    pending.pop(future)
                timed_out = True
                break
            for future in done:
                stream_name = pending.pop(future)
                try:
                    references[stream_name] = future.result()
                except Exception as exc:
                    errors[stream_name] = str(exc)
                else:
                    if successful_stream is None:
                        successful_stream = stream_name
                        deadline = time.monotonic() + initialization_grace_s
    finally:
        pool.shutdown(wait=not timed_out, cancel_futures=True)
    return references, errors


_PROCESS_CONFIG_FIELDS = (
    "task",
    "vision_backend",
    "model",
    "reasoning_effort",
    "codex_fast",
    "codex_timeout_seconds",
    "qwenvl_model",
    "qwenvl_base_url",
    "qwenvl_thinking_budget",
)


def _ground_camera_process_entry(
    config_values: Mapping[str, Any],
    stream_name: str,
    snapshot: AlignedRGBDSnapshot,
    calibration: RawServoCalibration,
    root: str,
    results: Any,
    process_group_ready: Any,
) -> None:
    try:
        os.setsid()
        process_group_ready.set()
    except OSError:
        pass
    try:
        reference = _ground_camera_reference(
            SimpleNamespace(**dict(config_values)),
            stream_name,
            snapshot,
            calibration,
            Path(root),
        )
        payload = {
            "target_prompt": reference.target_prompt,
            "target_bbox": reference.target_bbox,
            "table_bboxes": reference.table_bboxes,
            "camera_timestamp": reference.camera_timestamp,
        }
        results.put((stream_name, payload, None))
    except BaseException as exc:
        results.put((stream_name, None, str(exc)))


def _terminate_grounding_process(process: Any, process_group_ready: Any) -> None:
    if not process.is_alive():
        process.join(timeout=0.2)
        return
    if process_group_ready.is_set() and process.pid is not None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        if process_group_ready.is_set() and process.pid is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.join(timeout=1.0)


def _ground_dual_in_processes(
    config: Any,
    snapshots: Mapping[str, AlignedRGBDSnapshot],
    calibrations: Mapping[str, RawServoCalibration],
    root: Path,
    *,
    stream_names: Sequence[str],
    initialization_grace_s: float,
) -> tuple[dict[str, DualCameraReference], dict[str, str]]:
    """Run one killable grounding process per camera."""

    config_values = {
        name: getattr(config, name) for name in _PROCESS_CONFIG_FIELDS
    }
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes: dict[str, tuple[Any, Any]] = {}
    references: dict[str, DualCameraReference] = {}
    errors: dict[str, str] = {}
    pending = set(stream_names)
    successful_stream: str | None = None
    deadline: float | None = None
    try:
        for stream_name in stream_names:
            process_group_ready = context.Event()
            process = context.Process(
                target=_ground_camera_process_entry,
                args=(
                    config_values,
                    stream_name,
                    snapshots[stream_name],
                    calibrations[stream_name],
                    str(root),
                    results,
                    process_group_ready,
                ),
                name=f"dual-grounding-{stream_name}",
            )
            process.start()
            processes[stream_name] = (process, process_group_ready)

        while pending:
            if deadline is not None and time.monotonic() >= deadline:
                assert successful_stream is not None
                for stream_name in tuple(pending):
                    process, process_group_ready = processes[stream_name]
                    _terminate_grounding_process(process, process_group_ready)
                    errors[stream_name] = _timeout_error(
                        initialization_grace_s, successful_stream
                    )
                    pending.remove(stream_name)
                break
            timeout = 0.1
            if deadline is not None:
                timeout = min(timeout, max(0.0, deadline - time.monotonic()))
            try:
                stream_name, payload, error = results.get(timeout=timeout)
            except queue.Empty:
                for stream_name in tuple(pending):
                    process, _ = processes[stream_name]
                    if process.exitcode is not None:
                        errors[stream_name] = (
                            "initial grounding process exited without a result "
                            f"(exit code {process.exitcode})"
                        )
                        pending.remove(stream_name)
                continue
            if stream_name not in pending:
                continue
            pending.remove(stream_name)
            process, _ = processes[stream_name]
            process.join(timeout=1.0)
            if error is not None or payload is None:
                errors[stream_name] = error or "initial grounding failed"
                continue
            references[stream_name] = DualCameraReference(
                stream_name=stream_name,
                rgb=snapshots[stream_name].rgb,
                target_prompt=payload["target_prompt"],
                target_bbox=payload["target_bbox"],
                table_bboxes=payload["table_bboxes"],
                camera_timestamp=payload["camera_timestamp"],
                kind="initial",
            )
            if successful_stream is None:
                successful_stream = stream_name
                deadline = time.monotonic() + initialization_grace_s
    finally:
        for stream_name in tuple(pending):
            process, process_group_ready = processes[stream_name]
            _terminate_grounding_process(process, process_group_ready)
        for process, _ in processes.values():
            process.join(timeout=0.2)
        results.close()
        results.join_thread()
    return references, errors


def _pixel_bbox_to_normalized(
    bbox_xyxy: Sequence[float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("reference image dimensions must be positive")
    if len(bbox_xyxy) != 4:
        raise ValueError("pixel bbox must contain four values")
    x1, y1, x2, y2 = (float(item) for item in bbox_xyxy)
    return _reference_bbox(
        (
            1000.0 * x1 / width,
            1000.0 * y1 / height,
            1000.0 * x2 / width,
            1000.0 * y2 / height,
        ),
        field="normalized_bbox",
    )


class LatestReferenceGate:
    """Capture one coherent dual-class reference at a bounded frame cadence."""

    def __init__(
        self,
        *,
        interval_frames: int = 5,
        min_confidence: float = 0.35,
        freeze_y2_px: float = 75.0,
        resume_y2_px: float = 110.0,
    ):
        if interval_frames <= 0:
            raise ValueError("latest reference interval must be positive")
        confidence = float(min_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("latest reference confidence must be in [0,1]")
        self.interval_frames = int(interval_frames)
        self.min_confidence = confidence
        self.freeze_y2_px = float(freeze_y2_px)
        self.resume_y2_px = float(resume_y2_px)
        if self.freeze_y2_px >= self.resume_y2_px:
            raise ValueError("latest reference resume boundary must exceed freeze")
        self.frozen = False

    def consider(
        self,
        *,
        frame_index: int,
        stream_name: str,
        rgb: np.ndarray,
        camera_timestamp: float,
        target_prompt: str,
        target_bbox_xyxy: Sequence[float],
        table_bboxes_xyxy: Sequence[Sequence[float]],
        target_confidence: float,
        target_geometry_valid: bool,
        table_geometry_valid: bool,
    ) -> tuple[DualCameraReference | None, str]:
        if int(frame_index) <= 0 or int(frame_index) % self.interval_frames:
            return None, "interval"
        if not table_bboxes_xyxy:
            return None, "missing_table"
        if not target_geometry_valid:
            return None, "invalid_target_geometry"
        if not table_geometry_valid:
            return None, "invalid_table_geometry"
        y2 = float(target_bbox_xyxy[3])
        if y2 < self.freeze_y2_px:
            self.frozen = True
            return None, "vertical_danger"
        if self.frozen:
            if y2 < self.resume_y2_px:
                return None, "vertical_recovery_pending"
            if float(target_confidence) < self.min_confidence:
                return None, "low_confidence"
            self.frozen = False
        elif y2 < self.resume_y2_px:
            return None, "reference_near_top"
        if float(target_confidence) < self.min_confidence:
            return None, "low_confidence"
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            return None, "invalid_rgb"
        height, width = image.shape[:2]
        reference = DualCameraReference(
            stream_name=stream_name,
            rgb=image,
            target_prompt=target_prompt,
            target_bbox=_pixel_bbox_to_normalized(
                target_bbox_xyxy,
                width=width,
                height=height,
            ),
            table_bboxes=tuple(
                _pixel_bbox_to_normalized(bbox, width=width, height=height)
                for bbox in table_bboxes_xyxy
            ),
            camera_timestamp=camera_timestamp,
            kind="latest",
        )
        return reference, "accepted"


def _best_instance(
    instances: Sequence[TrackedInstance],
    class_index: int,
) -> TrackedInstance | None:
    candidates = [item for item in instances if item.class_index == class_index]
    return max(candidates, key=lambda item: item.confidence, default=None)


def _attempt_details(
    attempt: DualCameraAttempt,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "attempt_id": attempt.attempt_id,
        "live_stream": attempt.live_stream,
        "failover_stage": attempt.stage,
        "origin_stream": attempt.origin_stream,
        "reference_source_stream": attempt.reference.stream_name,
        "reference_kind": attempt.reference.kind,
        **extra,
    }


def _attempt_frame_details(attempt: DualCameraAttempt) -> dict[str, Any]:
    return {
        "camera_stream": attempt.live_stream,
        "attempt_id": attempt.attempt_id,
        "failover_stage": attempt.stage,
        "reference_source_stream": attempt.reference.stream_name,
        "reference_kind": attempt.reference.kind,
    }


def _write_initial_reference_summary(
    output_dir: Path,
    stream_names: Sequence[str],
    references: Mapping[str, DualCameraReference],
    errors: Mapping[str, str],
    selected_stream: str | None,
) -> None:
    _write_json(
        output_dir / "initial_reference_summary.json",
        {
            "selected_initial_stream": selected_stream,
            "streams": {
                stream_name: {
                    "eligible": stream_name in references,
                    "error": errors.get(stream_name),
                    "reference": (
                        None
                        if stream_name not in references
                        else {
                            "camera_timestamp": references[
                                stream_name
                            ].camera_timestamp,
                            "target_prompt": references[
                                stream_name
                            ].target_prompt,
                            "target_bbox": references[stream_name].target_bbox,
                            "table_bboxes": references[
                                stream_name
                            ].table_bboxes,
                        }
                    ),
                }
                for stream_name in stream_names
            },
        },
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
    camera_factory: Callable[[], Any] | None = None,
    client_factory: Callable[[], Any] | None = None,
    tracker_factory: Callable[[], Any] | None = None,
    calibration_factory: (
        Callable[[Any], Mapping[str, RawServoCalibration]] | None
    ) = None,
    reference_factory: (
        Callable[
            ...,
            tuple[dict[str, DualCameraReference], dict[str, str]],
        ]
        | None
    ) = None,
    table_required: Callable[[], bool] | None = None,
) -> None:
    """Run one bounded dual-camera failover session per operator generation."""

    calibrations: dict[str, RawServoCalibration] | None = None
    stream_names = (
        str(config.dual_head_camera_stream),
        str(config.dual_chest_camera_stream),
    )
    tolerance = int(config.dual_match_tolerance_frames)
    if tolerance <= 0:
        raise ValueError("dual match tolerance must be positive")
    camera: Any | None = None
    tracker: Any | None = None
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
                    camera = (
                        camera_factory()
                        if camera_factory is not None
                        else DualAlignedRGBDCamera(
                            config.camera_host,
                            config.camera_port,
                            stream_names=stream_names,
                            timeout_ms=config.camera_timeout_ms,
                            calibration_path=config.camera_intrinsics_path,
                        )
                    )
                initial_capture = camera.capture()
                stamp = time.strftime("%Y%m%d_%H%M%S")
                output_dir = (
                    Path(config.output_root).resolve()
                    / f"dual_raw_yoloe_{stamp}_g{generation}"
                )
                output_dir.mkdir(parents=True, exist_ok=False)
                if reference_factory is not None:
                    initial_references, grounding_errors = reference_factory(
                        config,
                        initial_capture.snapshots,
                        calibrations,
                        output_dir,
                        client_factory=client_factory,
                    )
                else:
                    initial_references, grounding_errors = (
                        ground_dual_raw_servo_references(
                            config,
                            initial_capture.snapshots,
                            calibrations,
                            output_dir,
                            client_factory=client_factory,
                            initialization_grace_s=float(
                                getattr(
                                    config,
                                    "dual_initialization_grace_s",
                                    30.0,
                                )
                            ),
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
                        if stream_name in initial_references
                    ),
                    None,
                )
                _write_initial_reference_summary(
                    output_dir,
                    stream_names,
                    initial_references,
                    initial_errors,
                    selected_initial_stream,
                )
                if selected_initial_stream is None:
                    message = "; ".join(
                        f"{name}: {error}"
                        for name, error in sorted(initial_errors.items())
                    ) or "neither camera produced an initial reference"
                    raise RuntimeError(message)
                coordinator = DualCameraFailoverCoordinator(
                    stream_names,
                    initial_references,
                )
                latest_gates = {
                    stream_name: LatestReferenceGate(
                        interval_frames=(
                            config.raw_reference_update_interval_frames
                        ),
                        min_confidence=(
                            config.raw_reference_update_min_confidence
                        ),
                        freeze_y2_px=getattr(
                            config,
                            "raw_reference_update_freeze_y2_px",
                            75.0,
                        ),
                        resume_y2_px=getattr(
                            config,
                            "raw_reference_update_resume_y2_px",
                            110.0,
                        ),
                    )
                    for stream_name in stream_names
                }
                if tracker is None:
                    tracker = (
                        tracker_factory()
                        if tracker_factory is not None
                        else YoloePersistentTracker(
                            config.raw_yoloe_model_path,
                            confidence=config.raw_yoloe_confidence,
                            imgsz=config.raw_yoloe_imgsz,
                            device=config.raw_yoloe_device,
                        )
                    )
                attempt = coordinator.start()
                frame_index = -1
                next_frame_at = time.monotonic()
                generation_finished = False
                while gate.is_active(generation) and not stop_event.is_set():
                    try:
                        tracker.start(
                            attempt.reference.rgb,
                            target_prompt=attempt.reference.target_prompt,
                            target_bbox=attempt.reference.target_bbox,
                            surface_prompt="table",
                            surface_bboxes=attempt.reference.table_bboxes,
                        )
                    except Exception as exc:
                        failure_reason = f"YOLOE initialization failed: {exc}"
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
                                details=_attempt_details(attempt),
                            ),
                        )
                    invalid_frames = 0
                    initialized = False
                    target_id: int | None = None
                    surface_id: int | None = None
                    failure_frame = None
                    while (
                        not failure_reason
                        and gate.is_active(generation)
                        and not stop_event.is_set()
                    ):
                        delay = next_frame_at - time.monotonic()
                        if delay > 0.0:
                            stop_event.wait(min(delay, 0.05))
                            continue
                        next_frame_at = time.monotonic() + 1.0 / config.raw_servo_hz
                        snapshot: AlignedRGBDSnapshot | None = None
                        target: TrackedInstance | None = None
                        surface: TrackedInstance | None = None
                        observation = None
                        hard_failure = False
                        target_reacquired = False
                        surface_reacquired = False
                        previous_target_id = target_id
                        previous_surface_id = surface_id
                        try:
                            capture = camera.capture()
                            snapshot = capture.require(attempt.live_stream)
                            frame_index += 1
                            calibration = calibrations[attempt.live_stream]
                            calibration.validate_snapshot(snapshot)
                            instances = list(tracker.track(snapshot.rgb))
                            require_table = (
                                not initialized
                                or table_required is None
                                or bool(table_required())
                            )
                            if initialized:
                                assert target_id is not None
                                target, target_reacquired, _ = _resolve_target(
                                    instances,
                                    target_id,
                                )
                                if surface_id is None:
                                    surface = _best_instance(instances, 1)
                                else:
                                    (
                                        surface,
                                        surface_reacquired,
                                        _,
                                    ) = _resolve_target(
                                        instances,
                                        surface_id,
                                        class_index=1,
                                    )
                            else:
                                target = _best_instance(instances, 0)
                                surface = _best_instance(instances, 1)
                            if target is None:
                                raise ValueError("missing tracked target")
                            if surface is None and require_table:
                                raise ValueError("missing tracked table")
                            observation = _observation(
                                snapshot,
                                target,
                                surface,
                                calibration,
                            )
                            if observation.table is None and require_table:
                                raise ValueError(
                                    observation.table_geometry_error
                                    or "missing table geometry"
                                )
                        except BasePoseCameraError as exc:
                            perception_error = str(exc)
                            hard_failure = True
                        except ValueError as exc:
                            perception_error = str(exc)
                        except Exception as exc:
                            perception_error = str(exc)
                            hard_failure = True
                        else:
                            perception_error = ""

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
                                    **_attempt_frame_details(attempt),
                                )
                            )
                            if hard_failure:
                                failure_reason = perception_error
                                failure_frame = diagnostic_frame
                                break
                            invalid_frames += 1
                            if invalid_frames >= tolerance:
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
                                            match_invalid_frames=invalid_frames,
                                        ),
                                        frame=diagnostic_frame,
                                    ),
                                )
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
                        target_id = target.track_id
                        if surface is not None:
                            surface_id = surface.track_id
                        latest, latest_reason = latest_gates[
                            attempt.live_stream
                        ].consider(
                            frame_index=frame_index + 1,
                            stream_name=attempt.live_stream,
                            rgb=snapshot.rgb,
                            camera_timestamp=snapshot.timestamp,
                            target_prompt=attempt.reference.target_prompt,
                            target_bbox_xyxy=target.bbox_xyxy,
                            table_bboxes_xyxy=(
                                ()
                                if surface is None
                                else (surface.bbox_xyxy,)
                            ),
                            target_confidence=target.confidence,
                            target_geometry_valid=True,
                            table_geometry_valid=observation.table is not None,
                        )
                        details = _attempt_details(
                            attempt,
                            match_invalid_frames=0,
                            target_track_id=target_id,
                            surface_track_id=(
                                None if surface is None else surface.track_id
                            ),
                            latest_reference_update=latest_reason,
                        )
                        if latest is not None:
                            coordinator.save_latest(latest)
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
                                    **_attempt_frame_details(attempt),
                                ),
                            ),
                        )
                    if stop_event.is_set() or not gate.is_active(generation):
                        break
                    next_attempt = coordinator.advance_after_failure(attempt)
                    if next_attempt is None:
                        _publish_worker_event(
                            events,
                            observation_events,
                            diagnostics,
                            RawServoEvent(
                                generation,
                                "error",
                                output_dir=str(output_dir),
                                error=(
                                    "three-stage failover exhausted: "
                                    f"{failure_reason or 'perception failed'}"
                                ),
                                hard=True,
                                details=_attempt_details(attempt),
                                frame=failure_frame,
                            ),
                        )
                        gate.cancel(generation)
                        generation_finished = True
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
                            details=_attempt_details(attempt),
                            frame=failure_frame,
                        ),
                    )
                if generation_finished:
                    continue
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
        if camera is not None:
            camera.close()
