"""Dual-camera text-prompt detection and failover for raw YOLOE BasePose."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
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
    _write_json,
)
from gear_sonic.utils.inference.base_pose_sensor import (
    SensorGatewayDualBasePoseCamera,
)
from gear_sonic.utils.inference.base_pose_visual_servo import (
    DEFAULT_SURFACE_TEXT_PROMPT,
    GenerationGate,
    HEAD_MONITOR_HOLD_EVENT,
    RawServoEvent,
    RawServoCalibration,
    RawServoObservation,
    TableGeometry,
    TrackedInstance,
    YoloePersistentTracker,
    YoloeTargetReferenceEncoder,
    bbox_iou,
    normalized_bbox_to_pixels,
    _diagnostic_frame,
    _observation,
    _publish_worker_event,
    _resolve_target,
    _surface_geometry_components,
    _temporary_png,
    ground_raw_servo_references,
)
from gear_sonic.utils.inference.base_pose_visual_servo_diagnostics import (
    AsyncFrameDiagnosticsWriter,
)


DEFAULT_DUAL_QWEN_FALLBACK_MODEL = "qwen3-vl-8b-instruct"
QWEN_HOLD_STAGES = frozenset({"head_monitor_qwen"})
JOINT_CHEST_LOSS_GRACE_FRAMES = 5
JOINT_CHEST_LOSS_ZERO_HOLD_FRAMES = 5
JOINT_HEAD_YAW_LOSS_GRACE_FRAMES = 10
JOINT_HEAD_YAW_LOSS_ZERO_HOLD_FRAMES = 10


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
    """Camera-selection evidence; YOLOE detection itself uses fixed text."""

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
        timestamp = float(self.camera_timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("reference camera_timestamp must be finite")
        kind = str(self.kind)
        if kind not in {"initial", "latest", "qwen", "head_monitor"}:
            raise ValueError("unsupported dual-camera reference kind")
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

    def _initial_reference(self, stream_name: str) -> DualCameraReference:
        other = self._other(stream_name)
        reference = self.initial_references.get(stream_name)
        if reference is None:
            reference = self.initial_references.get(other)
        if reference is None:
            raise RuntimeError("neither camera has an initial reference")
        return reference

    def begin_head_monitor_reacquisition(
        self,
        reference: DualCameraReference | None = None,
        *,
        use_qwen: bool = False,
    ) -> DualCameraAttempt:
        """Preempt chest tracking after the background head monitor fires."""

        head_stream = self.stream_names[0]
        if reference is not None:
            if reference.stream_name != head_stream:
                raise ValueError(
                    "head monitor reference must come from the head stream"
                )
            if reference.kind != "head_monitor":
                raise ValueError(
                    "head monitor reference must use kind=head_monitor"
                )
        elif not use_qwen:
            raise ValueError(
                "direct head reacquisition requires a monitor reference"
            )
        selected = reference or self._initial_reference(head_stream)
        return self._attempt(
            head_stream,
            selected,
            stage="head_monitor_qwen" if use_qwen else "head_monitor",
            origin_stream=head_stream,
        )

    def _recovery_sequence(
        self,
        origin_stream: str,
    ) -> tuple[tuple[str, str], ...]:
        head_stream, chest_stream = self.stream_names
        if origin_stream == head_stream:
            return (
                (head_stream, "origin_text"),
                (head_stream, "origin_qwen"),
                (chest_stream, "alternate_text"),
                (chest_stream, "alternate_qwen"),
            )
        if origin_stream == chest_stream:
            return (
                (head_stream, "alternate_text"),
                (head_stream, "alternate_qwen"),
                (chest_stream, "origin_text"),
                (chest_stream, "origin_qwen"),
            )
        raise ValueError("recovery origin belongs to an unknown stream")

    def advance_after_failure(
        self,
        attempt: DualCameraAttempt,
    ) -> DualCameraAttempt | None:
        if (
            attempt.stage == "head_monitor_qwen"
            and self._active_attempt_id != attempt.attempt_id
        ):
            chest_stream = self.stream_names[1]
            return self._attempt(
                chest_stream,
                self._initial_reference(chest_stream),
                stage="origin_text",
                origin_stream=chest_stream,
            )

        starts_new_cycle = (
            self._active_attempt_id == attempt.attempt_id
            or attempt.stage in {"initial", "head_monitor", "head_monitor_qwen"}
        )
        origin = attempt.live_stream if starts_new_cycle else attempt.origin_stream
        sequence = self._recovery_sequence(origin)
        if starts_new_cycle:
            next_index = 0
        else:
            try:
                current_index = sequence.index(
                    (attempt.live_stream, attempt.stage)
                )
            except ValueError as exc:
                raise ValueError(
                    f"unsupported failover stage: {attempt.stage}"
                ) from exc
            next_index = current_index + 1
        if next_index >= len(sequence):
            return None
        live_stream, stage = sequence[next_index]
        return self._attempt(
            live_stream,
            self._initial_reference(live_stream),
            stage=stage,
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
    assert snapshot.depth_raw is not None
    with _temporary_png(snapshot.rgb, rgb=True) as rgb_path:
        spec = ground_raw_servo_references(
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
        table_bboxes=(),
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
    "qwenvl_model",
    "qwenvl_base_url",
    "qwenvl_thinking_budget",
    "qwenvl_timeout_seconds",
    "dual_chest_camera_stream",
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


def ground_qwen_fallback_reference(
    config: Any,
    stream_name: str,
    snapshot: AlignedRGBDSnapshot,
    calibration: RawServoCalibration,
    output_dir: str | Path,
    *,
    client_factory: Callable[[], Any] | None = None,
) -> DualCameraReference:
    """Re-ground one current camera frame with the fixed Qwen fallback model."""

    calibration.validate_snapshot(snapshot)
    workdir = Path(output_dir).resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    qwen_model = str(
        getattr(
            config,
            "dual_qwenvl_fallback_model",
            DEFAULT_DUAL_QWEN_FALLBACK_MODEL,
        )
    ).strip()
    if not qwen_model:
        raise ValueError("dual Qwen fallback model must be non-empty")
    qwen_config = SimpleNamespace(
        task=config.task,
        qwenvl_model=qwen_model,
        qwenvl_base_url=config.qwenvl_base_url,
        qwenvl_enable_thinking=False,
        qwenvl_thinking_budget=config.qwenvl_thinking_budget,
        qwenvl_timeout_seconds=config.qwenvl_timeout_seconds,
    )
    with _temporary_png(snapshot.rgb, rgb=True) as rgb_path:
        spec = ground_raw_servo_references(
            qwen_config,
            rgb_path,
            workdir,
            client_factory=client_factory,
            calibration=calibration,
        )
    reference = DualCameraReference(
        stream_name=stream_name,
        rgb=snapshot.rgb,
        target_prompt=spec.target_prompt,
        target_bbox=spec.target_bbox,
        table_bboxes=(),
        camera_timestamp=snapshot.timestamp,
        kind="qwen",
    )
    _write_json(
        workdir / "qwen_reference_summary.json",
        {
            "model": qwen_model,
            "stream_name": stream_name,
            "camera_timestamp": snapshot.timestamp,
            "target_prompt": reference.target_prompt,
            "target_bbox": reference.target_bbox,
            "table_bboxes": reference.table_bboxes,
        },
    )
    return reference


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
    ):
        if interval_frames <= 0:
            raise ValueError("latest reference interval must be positive")
        confidence = float(min_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("latest reference confidence must be in [0,1]")
        self.interval_frames = int(interval_frames)
        self.min_confidence = confidence

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


@dataclass(frozen=True)
class DualReferenceUpdateRequest:
    generation: int
    frame_index: int
    reference: DualCameraReference
    require_target_embedding: bool = True


@dataclass(frozen=True)
class EncodedDualReference:
    target_embedding: Any
    surface_embedding: Any
    target_confidence: float
    target_iou: float
    surface_confidence: float
    surface_iou: float


@dataclass(frozen=True)
class DualReferenceUpdateResult:
    generation: int
    frame_index: int
    stream_name: str
    accepted: bool
    target_validated: bool
    reference: DualCameraReference
    target_embedding: Any | None
    surface_embedding: Any | None
    target_confidence: float | None
    target_iou: float | None
    surface_confidence: float | None
    surface_iou: float | None
    reason: str


class YoloeDualReferenceEncoder(YoloeTargetReferenceEncoder):
    """Independent YOLOE model that encodes one coherent target/desk frame."""

    @staticmethod
    def _best_detection(
        result: Any,
        *,
        class_index: int,
        expected_boxes: Sequence[tuple[float, float, float, float]],
    ) -> tuple[float, float]:
        if result.boxes is None or len(result.boxes) == 0:
            return 0.0, 0.0
        boxes = [
            tuple(float(value) for value in row)
            for row in result.boxes.xyxy.detach().cpu().numpy()
        ]
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()
        candidates = [
            index for index, value in enumerate(classes) if value == class_index
        ]
        if not candidates:
            return 0.0, 0.0
        best_index = max(
            candidates,
            key=lambda index: (
                max(bbox_iou(boxes[index], expected) for expected in expected_boxes),
                float(confidences[index]),
            ),
        )
        overlap = max(
            bbox_iou(boxes[best_index], expected) for expected in expected_boxes
        )
        return float(confidences[best_index]), float(overlap)

    def extract(
        self,
        request: DualReferenceUpdateRequest,
    ) -> EncodedDualReference:
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        reference = request.reference
        rgb = reference.rgb
        height, width = rgb.shape[:2]
        target_box = normalized_bbox_to_pixels(
            reference.target_bbox,
            width,
            height,
        )
        surface_boxes = tuple(
            normalized_bbox_to_pixels(bbox, width, height)
            for bbox in reference.table_bboxes
        )
        prompt_boxes = np.asarray(
            (target_box, *surface_boxes),
            dtype=np.float32,
        )
        prompt_classes = np.asarray(
            (0, *([1] * len(surface_boxes))),
            dtype=np.int32,
        )
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        self.model.predictor = None
        try:
            results = self.model.predict(
                source=bgr,
                refer_image=bgr,
                visual_prompts={
                    "bboxes": prompt_boxes,
                    "cls": prompt_classes,
                },
                predictor=YOLOEVPSegPredictor,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.confidence,
                half=True,
                quantize=16,
                verbose=False,
                save=False,
            )
        finally:
            self.model.predictor = None
        if len(results) != 1:
            raise RuntimeError(
                f"YOLOE dual reference returned {len(results)} results"
            )
        embeddings = getattr(self.model.model, "pe", None)
        if embeddings is None:
            raise RuntimeError("YOLOE did not produce dual visual embeddings")
        if embeddings.ndim != 3 or embeddings.shape[1] != 2:
            raise RuntimeError(
                "YOLOE expected two visual embeddings, got "
                f"{tuple(embeddings.shape)}"
            )
        result = results[0]
        target_confidence, target_iou = self._best_detection(
            result,
            class_index=0,
            expected_boxes=(target_box,),
        )
        surface_confidence, surface_iou = self._best_detection(
            result,
            class_index=1,
            expected_boxes=surface_boxes,
        )
        return EncodedDualReference(
            target_embedding=embeddings[:, :1].detach().clone(),
            surface_embedding=embeddings[:, 1:2].detach().clone(),
            target_confidence=target_confidence,
            target_iou=target_iou,
            surface_confidence=surface_confidence,
            surface_iou=surface_iou,
        )


class AsyncDualReferenceUpdater:
    """Encode the newest coherent target/desk reference off the servo thread."""

    _STOP = object()

    def __init__(
        self,
        encoder_factory: Callable[[], Any],
        *,
        min_confidence: float,
        min_iou: float,
    ) -> None:
        self._encoder_factory = encoder_factory
        self._min_confidence = float(min_confidence)
        self._min_iou = float(min_iou)
        self._requests: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._results: queue.Queue[DualReferenceUpdateResult] = queue.Queue(
            maxsize=1
        )
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="yoloe-dual-reference-updater",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _put_latest(destination: queue.Queue[Any], item: Any) -> None:
        while True:
            try:
                destination.put_nowait(item)
                return
            except queue.Full:
                try:
                    destination.get_nowait()
                except queue.Empty:
                    pass

    def submit(self, request: DualReferenceUpdateRequest) -> None:
        if not self._closed.is_set():
            self._put_latest(self._requests, request)

    def poll_latest(self) -> DualReferenceUpdateResult | None:
        latest = None
        while True:
            try:
                latest = self._results.get_nowait()
            except queue.Empty:
                return latest

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._put_latest(self._requests, self._STOP)
        self._thread.join(timeout=10.0)

    def _result(
        self,
        request: DualReferenceUpdateRequest,
        *,
        accepted: bool,
        reason: str,
        encoded: EncodedDualReference | None = None,
    ) -> DualReferenceUpdateResult:
        return DualReferenceUpdateResult(
            generation=request.generation,
            frame_index=request.frame_index,
            stream_name=request.reference.stream_name,
            accepted=accepted,
            target_validated=request.require_target_embedding,
            reference=request.reference,
            target_embedding=(
                None if encoded is None else encoded.target_embedding
            ),
            surface_embedding=(
                None if encoded is None else encoded.surface_embedding
            ),
            target_confidence=(
                None if encoded is None else encoded.target_confidence
            ),
            target_iou=None if encoded is None else encoded.target_iou,
            surface_confidence=(
                None if encoded is None else encoded.surface_confidence
            ),
            surface_iou=None if encoded is None else encoded.surface_iou,
            reason=reason,
        )

    def _validate(
        self,
        request: DualReferenceUpdateRequest,
        encoded: EncodedDualReference,
    ) -> DualReferenceUpdateResult:
        checks = [
            ("surface", encoded.surface_confidence, encoded.surface_iou)
        ]
        if request.require_target_embedding:
            checks.insert(
                0, ("target", encoded.target_confidence, encoded.target_iou)
            )
        for role, confidence, overlap in checks:
            if confidence < self._min_confidence:
                return self._result(
                    request,
                    accepted=False,
                    reason=f"{role}_confidence_below_threshold",
                    encoded=encoded,
                )
            if overlap < self._min_iou:
                return self._result(
                    request,
                    accepted=False,
                    reason=f"{role}_iou_below_threshold",
                    encoded=encoded,
                )
        return self._result(
            request,
            accepted=True,
            reason="accepted",
            encoded=encoded,
        )

    def _run(self) -> None:
        encoder = None
        try:
            encoder = self._encoder_factory()
        except Exception:
            pass
        while True:
            item = self._requests.get()
            if item is self._STOP:
                return
            if not isinstance(item, DualReferenceUpdateRequest):
                continue
            request = item
            try:
                if encoder is None:
                    encoder = self._encoder_factory()
                result = self._validate(request, encoder.extract(request))
            except Exception as exc:
                result = self._result(
                    request,
                    accepted=False,
                    reason=f"error:{type(exc).__name__}:{exc}",
                )
            if self._closed.is_set():
                return
            self._put_latest(self._results, result)


def _best_instance(
    instances: Sequence[TrackedInstance],
    class_index: int,
) -> TrackedInstance | None:
    candidates = [item for item in instances if item.class_index == class_index]
    return max(candidates, key=lambda item: item.confidence, default=None)


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
    target_reacquired = False
    surface_reacquired = False
    if target_id is None:
        target = _best_instance(instances, 0)
    else:
        target, target_reacquired, _ = _resolve_target(instances, target_id)
    if surface_id is None:
        surface = _best_instance(instances, 1)
    else:
        surface, surface_reacquired, _ = _resolve_target(
            instances,
            surface_id,
            class_index=1,
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
    TrackedInstance,
    TableGeometry,
    np.ndarray,
    bool,
    bool,
]:
    """Measure live head-camera desk yaw without requiring a head basket."""
    calibration.validate_snapshot(snapshot)
    instances = list(tracker.track(snapshot.rgb))
    target_reacquired = False
    surface_reacquired = False
    if target_id is None:
        target = _best_instance(instances, 0)
    else:
        target, target_reacquired, _ = _resolve_target(instances, target_id)
    if surface_id is None:
        surface = _best_instance(instances, 1)
    else:
        surface, surface_reacquired, _ = _resolve_target(
            instances,
            surface_id,
            class_index=1,
        )
    if surface is None:
        raise ValueError("missing tracked desk for live head yaw")
    table, table_error, desk_mask, _ = _surface_geometry_components(
        snapshot,
        surface,
        calibration,
        target_mask=None if target is None else target.mask,
    )
    if table is None:
        raise ValueError(table_error or "missing live head desk geometry")
    return (
        target,
        surface,
        table,
        desk_mask,
        target_reacquired,
        surface_reacquired,
    )


@dataclass(frozen=True)
class HeadCameraMonitorResult:
    """One result from the text-only head-camera monitor."""

    target_streak: int
    missing_streak: int
    hold_active: bool
    triggered: bool
    reference: DualCameraReference | None
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
            return HeadCameraMonitorResult(
                target_streak=self.target_streak,
                missing_streak=self.missing_streak,
                hold_active=self.hold_active,
                triggered=False,
                reference=None,
                target=None,
                surface=surface,
            )
        self.missing_streak = 0
        self.hold_active = True
        self.target_streak += 1
        streak = self.target_streak
        if streak < self.required_target_frames:
            return HeadCameraMonitorResult(
                target_streak=streak,
                missing_streak=0,
                hold_active=True,
                triggered=False,
                reference=None,
                target=target,
                surface=surface,
            )

        self.target_streak = 0
        height, width = snapshot.rgb.shape[:2]
        reference = DualCameraReference(
            stream_name=snapshot.depth_aligned_to,
            rgb=snapshot.rgb,
            target_prompt=self.target_prompt,
            target_bbox=_pixel_bbox_to_normalized(
                target.bbox_xyxy, width=width, height=height
            ),
            table_bboxes=(),
            camera_timestamp=snapshot.timestamp,
            kind="head_monitor",
        )
        return HeadCameraMonitorResult(
            target_streak=streak,
            missing_streak=0,
            hold_active=True,
            triggered=True,
            reference=reference,
            target=target,
            surface=surface,
        )


def _attempt_prompt_mode(_attempt: DualCameraAttempt) -> str:
    return "target_text_surface_text"


def _attempt_uses_target_text(_attempt: DualCameraAttempt) -> bool:
    """All online BasePose target detections use the configured text prompt."""

    return True


def _attempt_uses_surface_text(_attempt: DualCameraAttempt) -> bool:
    """The BasePose surface class is always the fixed ``desk`` text prompt."""
    return True


def _capture_camera_stream(
    camera: Any,
    stream_name: str,
    *,
    timeout_ms: int,
) -> AlignedRGBDSnapshot:
    capture_stream = getattr(camera, "capture_stream", None)
    if callable(capture_stream):
        return capture_stream(stream_name, timeout_ms=timeout_ms)
    return camera.capture().require(stream_name)


def _poll_camera_stream(
    camera: Any,
    stream_name: str,
) -> AlignedRGBDSnapshot | None:
    poll_stream = getattr(camera, "poll_stream", None)
    if callable(poll_stream):
        return poll_stream(stream_name)
    try:
        return camera.capture().require(stream_name)
    except BasePoseCameraError:
        return None


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
        "prompt_mode": _attempt_prompt_mode(attempt),
        "target_prompt": attempt.reference.target_prompt,
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
        "reference_source_stream": attempt.reference.stream_name,
        "reference_kind": attempt.reference.kind,
        "prompt_mode": _attempt_prompt_mode(attempt),
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
    head_monitor_tracker_factory: Callable[[], Any] | None = None,
    latest_reference_updater_factory: Callable[[], Any] | None = None,
    qwen_reference_factory: Callable[..., DualCameraReference] | None = None,
    handoff_hold_event: threading.Event | None = None,
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
    position_fallback_allowed: Callable[[], bool] | None = None,
) -> None:
    """Run one bounded dual-camera failover session per operator generation."""

    calibrations: dict[str, RawServoCalibration] | None = None
    stream_names = (
        str(config.dual_head_camera_stream),
        str(config.dual_chest_camera_stream),
    )
    surface_prompt = str(
        getattr(config, "surface_prompt", DEFAULT_SURFACE_TEXT_PROMPT)
    ).strip()
    if not surface_prompt:
        raise ValueError("surface_prompt must be non-empty")
    target_prompt = str(getattr(config, "target_prompt", "")).strip()
    if not target_prompt:
        raise ValueError("target_prompt must be non-empty")
    tolerance = int(config.dual_match_tolerance_frames)
    if tolerance <= 0:
        raise ValueError("dual match tolerance must be positive")
    head_reacquire_frames = int(getattr(config, "dual_head_reacquire_frames", 1))
    if head_reacquire_frames <= 0:
        raise ValueError("dual head reacquisition frame count must be positive")
    head_release_missing_frames = int(
        getattr(config, "dual_head_release_missing_frames", 3)
    )
    if head_release_missing_frames <= 0:
        raise ValueError("dual head release missing frame count must be positive")
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
    reference_updater: Any | None = None
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
                    camera = (
                        camera_factory()
                        if camera_factory is not None
                        else SensorGatewayDualBasePoseCamera(
                            config.sensor_gateway_endpoint,
                            stream_depths={
                                stream_names[0]: config.dual_head_depth_stream,
                                stream_names[1]: config.dual_chest_depth_stream,
                            },
                            timeout_ms=config.camera_timeout_ms,
                            request_timeout_ms=(
                                config.sensor_gateway_request_timeout_ms
                            ),
                            max_age_ms=config.sensor_gateway_max_age_ms,
                            max_skew_ms=config.sensor_gateway_max_skew_ms,
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
                # Qwen decides whether a camera contains the requested object,
                # but its reference bbox and generated label are not used by
                # YOLOE. Both YOLOE classes are fixed text embeddings.
                initial_references = {
                    stream_name: replace(
                        reference,
                        target_prompt=target_prompt,
                    )
                    for stream_name, reference in initial_references.items()
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
                monitor_reference = initial_references.get(stream_names[0])
                if monitor_reference is None:
                    monitor_reference = initial_references[selected_initial_stream]
                head_monitor_target_prompt = monitor_reference.target_prompt
                head_monitor: HeadCameraTextMonitor | None = None
                pending_head_qwen: dict[int, AlignedRGBDSnapshot] = {}

                latest_gates = {
                    stream_name: LatestReferenceGate(
                        interval_frames=(
                            config.raw_reference_update_interval_frames
                        ),
                        min_confidence=(
                            config.raw_reference_update_min_confidence
                        ),
                    )
                    for stream_name in stream_names
                }
                if reference_updater is None:
                    if latest_reference_updater_factory is not None:
                        reference_updater = latest_reference_updater_factory()
                attempt = coordinator.start()
                frame_index = -1
                next_frame_at = time.monotonic()
                generation_finished = False
                while gate.is_active(generation) and not stop_event.is_set():
                    try:
                        if attempt.stage in {
                            "origin_qwen",
                            "alternate_qwen",
                            "head_monitor_qwen",
                        }:
                            qwen_snapshot = pending_head_qwen.pop(
                                attempt.attempt_id, None
                            )
                            if qwen_snapshot is None:
                                qwen_snapshot = _capture_camera_stream(
                                    camera,
                                    attempt.live_stream,
                                    timeout_ms=int(config.camera_timeout_ms),
                                )
                            qwen_calibration = calibrations[attempt.live_stream]
                            qwen_output_dir = (
                                output_dir
                                / f"{attempt.stage}_{attempt.attempt_id:02d}"
                            )
                            qwen_reference = (
                                qwen_reference_factory(
                                    config,
                                    attempt.live_stream,
                                    qwen_snapshot,
                                    qwen_calibration,
                                    qwen_output_dir,
                                )
                                if qwen_reference_factory is not None
                                else ground_qwen_fallback_reference(
                                    config,
                                    attempt.live_stream,
                                    qwen_snapshot,
                                    qwen_calibration,
                                    qwen_output_dir,
                                )
                            )
                            if (
                                qwen_reference.stream_name != attempt.live_stream
                                or qwen_reference.kind != "qwen"
                            ):
                                raise ValueError(
                                    "Qwen fallback reference must use the active "
                                    "camera and kind=qwen"
                                )
                            qwen_reference = replace(
                                qwen_reference,
                                target_prompt=target_prompt,
                            )
                            attempt = replace(
                                attempt,
                                reference=qwen_reference,
                            )
                        tracker.start_all_text(
                            target_prompt=attempt.reference.target_prompt,
                        )
                        chest_fallback_ready = False
                        chest_fallback_init_error: str | None = None
                        if attempt.live_stream == stream_names[1]:
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
                            if head_monitor_tracker is not None:
                                head_monitor = HeadCameraTextMonitor(
                                    head_monitor_tracker,
                                    head_monitor_target_prompt,
                                    required_target_frames=head_reacquire_frames,
                                    release_missing_frames=(
                                        head_release_missing_frames
                                    ),
                                )
                                head_monitor.start()
                        else:
                            head_monitor = None
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
                            if head_monitor_tracker is not None:
                                try:
                                    chest_reference = initial_references.get(
                                        stream_names[1]
                                    )
                                    head_monitor_tracker.start_all_text(
                                        target_prompt=(
                                            attempt.reference.target_prompt
                                            if chest_reference is None
                                            else chest_reference.target_prompt
                                        )
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
                        monitor_details: dict[str, Any] = {}
                        basket_source_details: dict[str, Any] = {}
                        yaw_source_details: dict[str, Any] = {}
                        head_snapshot: AlignedRGBDSnapshot | None = None
                        live_head_snapshot: AlignedRGBDSnapshot | None = None
                        live_head_target: TrackedInstance | None = None
                        live_head_surface: TrackedInstance | None = None
                        live_head_table: TableGeometry | None = None
                        live_head_desk_mask: np.ndarray | None = None
                        live_head_yaw_error: str | None = None
                        fallback_head_yaw_attempted = False
                        joint_chest_target_missing = False
                        chest_active = attempt.live_stream == stream_names[1]
                        require_table = (
                            False
                            if chest_active
                            else table_required is None
                            or bool(table_required())
                        )
                        fallback_allowed = (
                            attempt.live_stream == stream_names[0]
                            and chest_fallback_ready
                            and (
                                position_fallback_allowed is None
                                or bool(position_fallback_allowed())
                            )
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
                            snapshot = _capture_camera_stream(
                                camera,
                                control_source_stream,
                                timeout_ms=max(
                                    1,
                                    int(float(config.raw_camera_stale_s) * 1000.0),
                                ),
                            )
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
                                monitor_result: HeadCameraMonitorResult | None = None
                                try:
                                    head_snapshot = _poll_camera_stream(
                                        camera,
                                        stream_names[0],
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
                                                _,
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
                                            live_head_yaw_error = str(exc)
                                    effective_hold = (
                                        monitor_result.hold_active
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
                                            monitor_result.target_streak
                                        ),
                                        "required_target_frames": (
                                            head_reacquire_frames
                                        ),
                                        "missing_streak": (
                                            monitor_result.missing_streak
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
                                        if monitor_result.reference is not None:
                                            preempt_attempt = (
                                                coordinator
                                                .begin_head_monitor_reacquisition(
                                                    monitor_result.reference
                                                )
                                            )
                                            failure_reason = (
                                                "head target reacquired for "
                                                f"{head_reacquire_frames} consecutive "
                                                "frames with desk"
                                            )
                                        else:
                                            preempt_attempt = (
                                                coordinator
                                                .begin_head_monitor_reacquisition(
                                                    use_qwen=True
                                                )
                                            )
                                            pending_head_qwen[
                                                preempt_attempt.attempt_id
                                            ] = head_snapshot
                                            failure_reason = (
                                                "head target reacquired for "
                                                f"{head_reacquire_frames} consecutive "
                                                "frames without desk; requesting Qwen"
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
                                try:
                                    live_head_snapshot = _capture_camera_stream(
                                        camera,
                                        stream_names[0],
                                        timeout_ms=max(
                                            1,
                                            int(
                                                float(config.raw_camera_stale_s)
                                                * 1000.0
                                            ),
                                        ),
                                    )
                                    (
                                        live_head_target,
                                        live_head_surface,
                                        live_head_table,
                                        live_head_desk_mask,
                                        _,
                                        _,
                                    ) = _observe_head_yaw_snapshot(
                                        live_head_snapshot,
                                        tracker,
                                        calibrations[stream_names[0]],
                                        target_id=target_id,
                                        surface_id=surface_id,
                                    )
                                except Exception as exc:
                                    live_head_yaw_error = str(exc)
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
                                    fallback_snapshot = _capture_camera_stream(
                                        camera,
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
                                try:
                                    live_head_snapshot = _capture_camera_stream(
                                        camera,
                                        stream_names[0],
                                        timeout_ms=max(
                                            1,
                                            int(
                                                float(config.raw_camera_stale_s)
                                                * 1000.0
                                            ),
                                        ),
                                    )
                                    (
                                        live_head_target,
                                        live_head_surface,
                                        live_head_table,
                                        live_head_desk_mask,
                                        _,
                                        _,
                                    ) = _observe_head_yaw_snapshot(
                                        live_head_snapshot,
                                        tracker,
                                        calibrations[stream_names[0]],
                                        target_id=target_id,
                                        surface_id=surface_id,
                                    )
                                except Exception as exc:
                                    live_head_yaw_error = str(exc)
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
                        refresh_details: dict[str, Any] = {}
                        if (
                            reference_updater is not None
                            and control_source_stream == attempt.live_stream
                        ):
                            update_result = reference_updater.poll_latest()
                            if update_result is not None:
                                if update_result.generation != generation:
                                    refresh_details[
                                        "latest_reference_refresh"
                                    ] = {
                                        "status": "discarded",
                                        "reason": "generation",
                                        "source_generation": (
                                            update_result.generation
                                        ),
                                        "source_frame": update_result.frame_index,
                                    }
                                elif (
                                    update_result.stream_name
                                    != attempt.live_stream
                                ):
                                    refresh_details[
                                        "latest_reference_refresh"
                                    ] = {
                                        "status": "discarded",
                                        "reason": "stream",
                                        "source_stream": (
                                            update_result.stream_name
                                        ),
                                        "source_frame": update_result.frame_index,
                                    }
                                elif frame_index - update_result.frame_index > max(
                                    5,
                                    2
                                    * config.raw_reference_update_interval_frames,
                                ):
                                    refresh_details[
                                        "latest_reference_refresh"
                                    ] = {
                                        "status": "discarded",
                                        "reason": "stale",
                                        "source_frame": update_result.frame_index,
                                        "age_frames": (
                                            frame_index
                                            - update_result.frame_index
                                        ),
                                    }
                                elif _attempt_uses_surface_text(attempt):
                                    refresh_details[
                                        "latest_reference_refresh"
                                    ] = {
                                        "status": "discarded",
                                        "reason": "surface_text_prompt",
                                        "source_frame": update_result.frame_index,
                                    }
                                elif (
                                    not _attempt_uses_target_text(attempt)
                                    and not update_result.target_validated
                                ):
                                    refresh_details[
                                        "latest_reference_refresh"
                                    ] = {
                                        "status": "discarded",
                                        "reason": "prompt_mode",
                                        "source_frame": update_result.frame_index,
                                        "source_mode": "target_text",
                                    }
                                elif not update_result.accepted:
                                    refresh_details[
                                        "latest_reference_refresh"
                                    ] = {
                                        "status": "rejected",
                                        "reason": update_result.reason,
                                        "source_frame": update_result.frame_index,
                                        "target_confidence": (
                                            update_result.target_confidence
                                        ),
                                        "target_iou": update_result.target_iou,
                                        "surface_confidence": (
                                            update_result.surface_confidence
                                        ),
                                        "surface_iou": update_result.surface_iou,
                                    }
                                else:
                                    try:
                                        if update_result.surface_embedding is None:
                                            raise RuntimeError(
                                                "accepted refresh has no desk PE"
                                            )
                                        if _attempt_uses_target_text(attempt):
                                            tracker.install_surface_embedding(
                                                update_result.surface_embedding
                                            )
                                        else:
                                            if update_result.target_embedding is None:
                                                raise RuntimeError(
                                                    "accepted refresh has no target PE"
                                                )
                                            tracker.install_reference_embeddings(
                                                target_embedding=(
                                                    update_result.target_embedding
                                                ),
                                                surface_embedding=(
                                                    update_result.surface_embedding
                                                ),
                                            )
                                    except Exception as exc:
                                        refresh_details[
                                            "latest_reference_refresh"
                                        ] = {
                                            "status": "error",
                                            "source_frame": (
                                                update_result.frame_index
                                            ),
                                            "error": str(exc),
                                        }
                                    else:
                                        attempt = replace(
                                            attempt,
                                            reference=update_result.reference,
                                        )
                                        refresh_details[
                                            "latest_reference_refresh"
                                        ] = {
                                            "status": "applied",
                                            "source_frame": (
                                                update_result.frame_index
                                            ),
                                            "source_stream": (
                                                update_result.stream_name
                                            ),
                                            "target_confidence": (
                                                update_result.target_confidence
                                            ),
                                            "target_iou": update_result.target_iou,
                                            "surface_confidence": (
                                                update_result.surface_confidence
                                            ),
                                            "surface_iou": update_result.surface_iou,
                                        }
                        if control_source_stream == attempt.live_stream:
                            latest, latest_reason = latest_gates[
                                attempt.live_stream
                            ].consider(
                                frame_index=frame_index,
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
                                table_geometry_valid=(
                                    observation.table is not None
                                ),
                            )
                        else:
                            latest = None
                            latest_reason = "chest_position_fallback"
                        details = _attempt_details(
                            attempt,
                            match_invalid_frames=0,
                            control_source_stream=control_source_stream,
                            basket_source=basket_source_details,
                            yaw_source=yaw_source_details,
                            head_yaw_loss=head_yaw_loss,
                            target_track_id=target.track_id,
                            surface_track_id=(
                                None if surface is None else surface.track_id
                            ),
                            latest_reference_update=latest_reason,
                            **refresh_details,
                            **monitor_details,
                        )
                        if latest is not None:
                            if (
                                reference_updater is not None
                                and not _attempt_uses_surface_text(attempt)
                            ):
                                reference_updater.submit(
                                    DualReferenceUpdateRequest(
                                        generation=generation,
                                        frame_index=frame_index,
                                        reference=latest,
                                        require_target_embedding=(
                                            not _attempt_uses_target_text(attempt)
                                        ),
                                    )
                                )
                                details[
                                    "latest_reference_refresh_scheduled"
                                ] = {
                                    "source_frame": frame_index,
                                    "source_stream": attempt.live_stream,
                                }
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
                            "both-camera Qwen failover exhausted: "
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
                                    terminal_cause=(
                                        failure_reason or "perception failed"
                                    ),
                                ),
                                frame=failure_frame,
                            ),
                        )
                        gate.cancel(generation)
                        generation_finished = True
                        break
                    attempt = next_attempt
                    qwen_handoff_requires_hold = (
                        attempt.stage in QWEN_HOLD_STAGES
                    )
                    if (
                        qwen_handoff_requires_hold
                        and handoff_hold_event is not None
                    ):
                        handoff_hold_event.clear()
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
                            details=_attempt_details(attempt, **switch_details),
                            frame=failure_frame,
                        ),
                    )
                    if (
                        qwen_handoff_requires_hold
                        and handoff_hold_event is not None
                    ):
                        while (
                            gate.is_active(generation)
                            and not stop_event.is_set()
                            and not handoff_hold_event.wait(timeout=0.05)
                        ):
                            pass
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
        perception_pool.shutdown(wait=True, cancel_futures=True)
        if reference_updater is not None:
            reference_updater.close()
        if camera is not None:
            camera.close()
