"""Per-frame JSONL diagnostics for the raw YOLOE visual-servo loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import queue
import threading
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class DetectionFrameData:
    frame_index: int
    camera_timestamp: float
    image_size: tuple[int, int]
    camera_stream: str | None = None
    attempt_id: int | None = None
    failover_stage: str | None = None
    prompt_mode: str | None = None
    target_bbox_xyxy: tuple[float, float, float, float] | None = None
    target_mask: np.ndarray | None = field(default=None, repr=False)
    target_track_id: int | None = None
    target_confidence: float | None = None
    surface_bbox_xyxy: tuple[float, float, float, float] | None = None
    surface_mask: np.ndarray | None = field(default=None, repr=False)
    completed_surface_mask: np.ndarray | None = field(default=None, repr=False)
    surface_track_id: int | None = None
    surface_confidence: float | None = None
    target_geometry: Mapping[str, Any] | None = None
    table_geometry: Mapping[str, Any] | None = None
    table_geometry_error: str | None = None
    perception_kind: str = "observation"
    perception_error: str | None = None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _json_compatible(value: Any) -> Any:
    """Convert diagnostic values to strict JSON without losing later frames."""
    if isinstance(value, (float, np.floating)):
        normalized = float(value)
        return normalized if math.isfinite(normalized) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _bbox_json(
    value: tuple[float, float, float, float] | None,
) -> list[float] | None:
    return None if value is None else [float(item) for item in value]


def _phase_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


class FrameDiagnosticsWriter:
    """Append perception and control records without producing image artifacts."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).resolve()
        self.jsonl_path = self.output_dir / "raw_servo_frames.jsonl"

    @staticmethod
    def _detection(
        bbox: tuple[float, float, float, float] | None,
        track_id: int | None,
        confidence: float | None,
        mask: np.ndarray | None,
    ) -> dict[str, Any]:
        return {
            "bbox_xyxy": _bbox_json(bbox),
            "track_id": None if track_id is None else int(track_id),
            "confidence": _optional_float(confidence),
            "mask_pixels": None if mask is None else int(np.count_nonzero(mask)),
        }

    @staticmethod
    def _controller(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "phase": _phase_name(value.get("phase")),
            "resume_phase": _phase_name(value.get("resume_phase")),
            "transition_reason": value.get("transition_reason"),
            "filtered_errors": value.get("filtered_errors"),
            "visual_yaw_error_rad": value.get("visual_yaw_error_rad"),
            "desired_heading_rad": value.get("desired_heading_rad"),
            "heading_setpoint_error_rad": value.get("heading_setpoint_error_rad"),
            "yaw_error_source": value.get("yaw_error_source"),
            "yaw_error_trusted": value.get("yaw_error_trusted"),
            "post_stop_sample_frames": value.get("post_stop_sample_frames"),
            "post_stop_deviation_frames": value.get("post_stop_deviation_frames"),
            "post_stop_sample_count": value.get("post_stop_sample_count"),
            "post_stop_valid_sample_count": value.get(
                "post_stop_valid_sample_count"
            ),
            "post_stop_invalid_sample_count": value.get(
                "post_stop_invalid_sample_count"
            ),
            "post_stop_out_of_tolerance_streak": value.get(
                "post_stop_out_of_tolerance_streak"
            ),
            "post_stop_max_out_of_tolerance_streak": value.get(
                "post_stop_max_out_of_tolerance_streak"
            ),
            "post_stop_realign_count": value.get("post_stop_realign_count"),
            "invalid_frames": value.get("invalid_frames"),
            "vertical_recenter_armed": value.get("vertical_recenter_armed"),
            "vertical_recenter_elapsed_s": value.get(
                "vertical_recenter_elapsed_s"
            ),
            "vertical_recenter_stable_frames": value.get(
                "vertical_recenter_stable_frames"
            ),
            "stable_frames": value.get("stable_frames"),
            "yaw_stable_frames": value.get("yaw_stable_frames"),
            "recenter_stable_frames": value.get("recenter_stable_frames"),
        }

    @staticmethod
    def _command(value: Mapping[str, Any]) -> dict[str, float]:
        return {
            "vx": float(value.get("vx", 0.0)),
            "vy": float(value.get("vy", 0.0)),
            "wz": float(value.get("wz", 0.0)),
        }

    def write(
        self,
        frame: DetectionFrameData,
        *,
        controller_state: Mapping[str, Any] | None,
        command: Mapping[str, Any] | None,
        control_applied: bool = True,
        orientation: Mapping[str, Any] | None = None,
    ) -> None:
        applied = bool(control_applied)
        if applied and (controller_state is None or command is None):
            raise ValueError("applied diagnostic frame requires control metadata")
        record = {
            "frame_index": int(frame.frame_index),
            "camera_timestamp": float(frame.camera_timestamp),
            "camera_stream": frame.camera_stream,
            "attempt_id": (
                None if frame.attempt_id is None else int(frame.attempt_id)
            ),
            "failover_stage": frame.failover_stage,
            "prompt_mode": frame.prompt_mode,
            "perception_kind": frame.perception_kind,
            "perception_error": frame.perception_error,
            "control_applied": applied,
            "detections": {
                "target": self._detection(
                    frame.target_bbox_xyxy,
                    frame.target_track_id,
                    frame.target_confidence,
                    frame.target_mask,
                ),
                "table": self._detection(
                    frame.surface_bbox_xyxy,
                    frame.surface_track_id,
                    frame.surface_confidence,
                    frame.surface_mask,
                ),
            },
            "geometry": {
                "target": (
                    None
                    if frame.target_geometry is None
                    else dict(frame.target_geometry)
                ),
                "table": (
                    None
                    if frame.table_geometry is None
                    else dict(frame.table_geometry)
                ),
                "table_error": frame.table_geometry_error,
            },
            "controller": (
                None if not applied else self._controller(controller_state)
            ),
            "command": None if not applied else self._command(command),
            "orientation": None if orientation is None else dict(orientation),
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _json_compatible(record),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()


@dataclass(frozen=True)
class _ProducedFrame:
    generation: int
    output_dir: Path
    frame: DetectionFrameData


@dataclass(frozen=True)
class _ControlDecision:
    generation: int
    frame_index: int
    control_applied: bool
    controller_state: Mapping[str, Any] | None
    command: Mapping[str, Any] | None
    orientation: Mapping[str, Any] | None


@dataclass(frozen=True)
class _CloseWriter:
    drain: bool


class AsyncFrameDiagnosticsWriter:
    """Join produced frames with control decisions and write them off-thread."""

    def __init__(
        self,
        *,
        logger: Callable[[str], None] = print,
        writer_factory: Callable[[str | Path], FrameDiagnosticsWriter] = (
            FrameDiagnosticsWriter
        ),
    ):
        self.logger = logger
        self.writer_factory = writer_factory
        self._items: queue.Queue[object] = queue.Queue()
        self._submit_lock = threading.Lock()
        self._closed = False
        self._frames: dict[tuple[int, int], _ProducedFrame] = {}
        self._decisions: dict[tuple[int, int], _ControlDecision] = {}
        self._next_index: dict[int, int] = {}
        self._writers: dict[int, FrameDiagnosticsWriter] = {}
        self._reported_failures: set[tuple[int, str, str]] = set()
        self._thread = threading.Thread(
            target=self._run,
            name="raw-servo-diagnostics",
            daemon=True,
        )
        self._thread.start()

    def submit_frame(
        self,
        generation: int,
        output_dir: str | Path,
        frame: DetectionFrameData,
    ) -> None:
        item = _ProducedFrame(int(generation), Path(output_dir).resolve(), frame)
        with self._submit_lock:
            if self._closed:
                raise RuntimeError("diagnostic writer is closed")
            self._items.put_nowait(item)

    def submit_decision(
        self,
        generation: int,
        frame_index: int,
        *,
        control_applied: bool,
        controller_state: Mapping[str, Any] | None,
        command: Mapping[str, Any] | None,
        orientation: Mapping[str, Any] | None = None,
    ) -> None:
        item = _ControlDecision(
            int(generation),
            int(frame_index),
            bool(control_applied),
            None if controller_state is None else dict(controller_state),
            None if command is None else dict(command),
            None if orientation is None else dict(orientation),
        )
        with self._submit_lock:
            if self._closed:
                raise RuntimeError("diagnostic writer is closed")
            self._items.put_nowait(item)

    def close(self, *, drain: bool = True) -> None:
        with self._submit_lock:
            if self._closed:
                return
            self._closed = True
            self._items.put_nowait(_CloseWriter(bool(drain)))
        self._thread.join()

    def _report_failure(
        self, generation: int, output_dir: Path, exc: Exception
    ) -> None:
        key = (generation, type(exc).__name__, str(exc))
        if key not in self._reported_failures:
            self._reported_failures.add(key)
            self.logger(
                "[RawServo] WARNING diagnostic frame write failed for "
                f"{output_dir}; continuing with later frames: {exc}"
            )

    def _write(
        self, produced: _ProducedFrame, decision: _ControlDecision
    ) -> bool:
        generation = produced.generation
        try:
            writer = self._writers.get(generation)
            if writer is None:
                writer = self.writer_factory(produced.output_dir)
                self._writers[generation] = writer
            writer.write(
                produced.frame,
                control_applied=decision.control_applied,
                controller_state=decision.controller_state,
                command=decision.command,
                orientation=decision.orientation,
            )
            return True
        except Exception as exc:
            self._report_failure(generation, produced.output_dir, exc)
            return False

    def _flush_ready(self, generation: int) -> None:
        next_index = self._next_index.setdefault(generation, 0)
        while True:
            key = (generation, next_index)
            produced = self._frames.get(key)
            decision = self._decisions.get(key)
            if produced is None or decision is None:
                return
            self._frames.pop(key, None)
            self._decisions.pop(key, None)
            self._write(produced, decision)
            next_index += 1
            self._next_index[generation] = next_index

    def _drain_pending(self) -> None:
        for key in list(self._frames):
            if key not in self._decisions:
                self._decisions[key] = _ControlDecision(
                    key[0], key[1], False, None, None, None
                )
        generations = sorted({key[0] for key in self._frames})
        for generation in generations:
            for key in sorted(
                (key for key in self._frames if key[0] == generation),
                key=lambda value: value[1],
            ):
                produced = self._frames.pop(key)
                decision = self._decisions.pop(key)
                self._write(produced, decision)

    def _run(self) -> None:
        while True:
            item = self._items.get()
            if isinstance(item, _CloseWriter):
                if item.drain:
                    self._drain_pending()
                return
            if isinstance(item, _ProducedFrame):
                key = (item.generation, int(item.frame.frame_index))
                self._frames[key] = item
                self._flush_ready(item.generation)
                continue
            if isinstance(item, _ControlDecision):
                key = (item.generation, item.frame_index)
                self._decisions[key] = item
                self._flush_ready(item.generation)
