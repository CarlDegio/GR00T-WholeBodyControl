"""Per-camera-frame diagnostics for the raw YOLOE visual-servo loop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import queue
import threading
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from gear_sonic.utils.inference.base_pose import _atomic_write_bytes


@dataclass(frozen=True)
class CameraFrameData:
    """Pixel-space diagnostics belonging to one physical camera frame."""

    camera_stream: str
    camera_timestamp: float
    rgb: np.ndarray
    target_bbox_xyxy: tuple[float, float, float, float] | None = None
    target_mask: np.ndarray | None = field(default=None, repr=False)
    surface_mask: np.ndarray | None = field(default=None, repr=False)
    completed_surface_mask: np.ndarray | None = field(default=None, repr=False)
    table_rgb_edges: np.ndarray | None = field(default=None, repr=False)
    table_geometry: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DetectionFrameData:
    frame_index: int
    camera_timestamp: float
    rgb: np.ndarray
    depth_raw: np.ndarray | None = field(default=None, repr=False)
    depth_scale_m: float | None = None
    table_rgb_edges: np.ndarray | None = field(default=None, repr=False)
    camera_stream: str | None = None
    attempt_id: int | None = None
    failover_stage: str | None = None
    reference_source_stream: str | None = None
    reference_kind: str | None = None
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
    additional_camera_frames: tuple[CameraFrameData, ...] = ()
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
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _bbox_json(value: tuple[float, float, float, float] | None) -> list[float] | None:
    return None if value is None else [float(item) for item in value]


def _phase_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


class FrameDiagnosticsWriter:
    """Write frame telemetry, with image artifacts disabled by default."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        review_stride: int = 5,
        save_images: bool = False,
    ):
        if review_stride <= 0:
            raise ValueError("review_stride must be positive")
        self.output_dir = Path(output_dir).resolve()
        self.jsonl_path = self.output_dir / "raw_servo_frames.jsonl"
        self.review_stride = int(review_stride)
        self.save_images = bool(save_images)
        self.review_raw_dir = self.output_dir / "review_samples" / "raw"
        self.review_depth_dir = self.output_dir / "review_samples" / "depth"
        self.review_edges_dir = self.output_dir / "review_samples" / "edges"
        self.review_masks_dir = self.output_dir / "review_samples" / "masks"
        self.completion_dir = self.output_dir / "completion_capture"
        self.completion_manifest_path = self.completion_dir / "manifest.jsonl"
        self._recent_applied_frames: deque[DetectionFrameData] = deque(maxlen=3)
        self._completion_capture_active = False
        self._completion_captured: set[tuple[str, int, str]] = set()

    @staticmethod
    def _encode_png(image: np.ndarray, *, name: str) -> bytes:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OSError(f"failed to encode {name}")
        return encoded.tobytes()

    @staticmethod
    def _table_edge_overlay(
        mask: np.ndarray | None,
        table_geometry: Mapping[str, Any] | None,
    ) -> np.ndarray | None:
        if mask is None or table_geometry is None:
            return None
        edge_endpoints = table_geometry.get("line_endpoints_px")
        if edge_endpoints is None:
            return None
        try:
            raw_endpoints = np.asarray(edge_endpoints, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if raw_endpoints.shape != (2, 2) or not np.all(np.isfinite(raw_endpoints)):
            return None
        height, width = mask.shape
        endpoints = np.rint(raw_endpoints).astype(int)
        endpoints[:, 0] = np.clip(endpoints[:, 0], 0, width - 1)
        endpoints[:, 1] = np.clip(endpoints[:, 1], 0, height - 1)
        start = tuple(int(item) for item in endpoints[0])
        end = tuple(int(item) for item in endpoints[1])
        binary = mask.astype(bool).astype(np.uint8) * 255
        overlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.line(overlay, start, end, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(overlay, start, 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, end, 5, (0, 255, 255), -1, cv2.LINE_AA)
        return overlay

    @staticmethod
    def _safe_stream_name(value: str | None) -> str:
        raw = "unknown" if value is None else str(value)
        safe = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in raw
        ).strip("_")
        return safe or "unknown"

    @staticmethod
    def _primary_camera_frame(frame: DetectionFrameData) -> CameraFrameData:
        return CameraFrameData(
            camera_stream=frame.camera_stream or "unknown",
            camera_timestamp=frame.camera_timestamp,
            rgb=frame.rgb,
            target_bbox_xyxy=frame.target_bbox_xyxy,
            target_mask=frame.target_mask,
            surface_mask=frame.surface_mask,
            completed_surface_mask=frame.completed_surface_mask,
            table_rgb_edges=frame.table_rgb_edges,
            table_geometry=frame.table_geometry,
        )

    @classmethod
    def _camera_frames(cls, frame: DetectionFrameData) -> tuple[CameraFrameData, ...]:
        ordered = (cls._primary_camera_frame(frame), *frame.additional_camera_frames)
        result: list[CameraFrameData] = []
        seen: set[str] = set()
        for camera_frame in ordered:
            stream = str(camera_frame.camera_stream)
            if stream in seen:
                continue
            seen.add(stream)
            result.append(camera_frame)
        return tuple(result)

    @staticmethod
    def _validate_camera_frame(frame: CameraFrameData) -> np.ndarray:
        rgb = np.asarray(frame.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("completion RGB must be uint8 HxWx3")
        for name, value in (
            ("target", frame.target_mask),
            ("table", frame.surface_mask),
            ("completed table", frame.completed_surface_mask),
            ("table RGB edge", frame.table_rgb_edges),
        ):
            if value is not None and np.asarray(value).shape != rgb.shape[:2]:
                raise ValueError(f"completion {name} image shape does not match RGB")
        return rgb

    @staticmethod
    def _completion_overlay(frame: CameraFrameData) -> np.ndarray:
        rgb = FrameDiagnosticsWriter._validate_camera_frame(frame)
        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        table_mask = (
            frame.surface_mask
            if frame.surface_mask is not None
            else frame.completed_surface_mask
        )
        if table_mask is not None:
            selected = np.asarray(table_mask) > 0
            tint = overlay.copy()
            tint[selected] = (255, 170, 0)
            overlay = cv2.addWeighted(tint, 0.35, overlay, 0.65, 0.0)
        if frame.target_bbox_xyxy is not None:
            bbox = np.asarray(frame.target_bbox_xyxy, dtype=np.float64)
            if bbox.shape == (4,) and np.all(np.isfinite(bbox)):
                height, width = overlay.shape[:2]
                x1, y1, x2, y2 = np.rint(bbox).astype(int)
                x1 = int(np.clip(x1, 0, width - 1))
                x2 = int(np.clip(x2, 0, width - 1))
                y1 = int(np.clip(y1, 0, height - 1))
                y2 = int(np.clip(y2, 0, height - 1))
                cv2.rectangle(
                    overlay, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA
                )
                cv2.putText(
                    overlay,
                    "target",
                    (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        geometry = frame.table_geometry
        endpoints = None if geometry is None else geometry.get("line_endpoints_px")
        if endpoints is not None:
            values = np.asarray(endpoints, dtype=np.float64)
            if values.shape == (2, 2) and np.all(np.isfinite(values)):
                height, width = overlay.shape[:2]
                points = np.rint(values).astype(int)
                points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                start = tuple(int(item) for item in points[0])
                end = tuple(int(item) for item in points[1])
                cv2.line(overlay, start, end, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.circle(overlay, start, 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, end, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(frame.camera_stream),
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def _write_completion_frame(
        self,
        frame: DetectionFrameData,
        *,
        stage: str,
    ) -> None:
        for camera_frame in self._camera_frames(frame):
            stream = self._safe_stream_name(camera_frame.camera_stream)
            key = (stage, int(frame.frame_index), stream)
            if key in self._completion_captured:
                continue
            self._completion_captured.add(key)
            rgb = self._validate_camera_frame(camera_frame)
            stage_dir = self.completion_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{int(frame.frame_index):06d}_{stream}"
            raw_relative = Path("completion_capture") / stage / f"{stem}_raw.png"
            overlay_relative = (
                Path("completion_capture") / stage / f"{stem}_overlay.png"
            )
            _atomic_write_bytes(
                self.output_dir / raw_relative,
                self._encode_png(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    name=f"completion raw RGB {stem}",
                ),
            )
            _atomic_write_bytes(
                self.output_dir / overlay_relative,
                self._encode_png(
                    self._completion_overlay(camera_frame),
                    name=f"completion overlay {stem}",
                ),
            )
            table_mask_relative: Path | None = None
            table_mask = camera_frame.surface_mask
            if table_mask is not None:
                table_mask_relative = (
                    Path("completion_capture") / stage / f"{stem}_table_mask.png"
                )
                binary = (np.asarray(table_mask) > 0).astype(np.uint8) * 255
                _atomic_write_bytes(
                    self.output_dir / table_mask_relative,
                    self._encode_png(binary, name=f"completion table mask {stem}"),
                )
            completed_mask_relative: Path | None = None
            if camera_frame.completed_surface_mask is not None:
                completed_mask_relative = (
                    Path("completion_capture")
                    / stage
                    / f"{stem}_table_completed_mask.png"
                )
                completed = (
                    np.asarray(camera_frame.completed_surface_mask) > 0
                ).astype(np.uint8) * 255
                _atomic_write_bytes(
                    self.output_dir / completed_mask_relative,
                    self._encode_png(
                        completed,
                        name=f"completion completed table mask {stem}",
                    ),
                )
            table_edges_relative: Path | None = None
            if camera_frame.table_rgb_edges is not None:
                table_edges_relative = (
                    Path("completion_capture") / stage / f"{stem}_table_edges.png"
                )
                edges = np.asarray(camera_frame.table_rgb_edges).astype(
                    np.uint8, copy=False
                )
                _atomic_write_bytes(
                    self.output_dir / table_edges_relative,
                    self._encode_png(edges, name=f"completion table edges {stem}"),
                )
            geometry = camera_frame.table_geometry
            manifest_record = {
                "stage": stage,
                "frame_index": int(frame.frame_index),
                "camera_stream": camera_frame.camera_stream,
                "camera_timestamp": float(camera_frame.camera_timestamp),
                "raw_rgb": raw_relative.as_posix(),
                "overlay_rgb": overlay_relative.as_posix(),
                "table_mask": (
                    None
                    if table_mask_relative is None
                    else table_mask_relative.as_posix()
                ),
                "table_completed_mask": (
                    None
                    if completed_mask_relative is None
                    else completed_mask_relative.as_posix()
                ),
                "table_rgb_edges": (
                    None
                    if table_edges_relative is None
                    else table_edges_relative.as_posix()
                ),
                "target_bbox_xyxy": _bbox_json(camera_frame.target_bbox_xyxy),
                "table_edge_endpoints_px": (
                    None
                    if geometry is None
                    else geometry.get("line_endpoints_px")
                ),
            }
            self.completion_dir.mkdir(parents=True, exist_ok=True)
            with self.completion_manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        _json_compatible(manifest_record),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                handle.flush()

    def _capture_completion_images(
        self,
        frame: DetectionFrameData,
        *,
        control_applied: bool,
        completion_capture: str | None,
    ) -> None:
        if control_applied:
            self._recent_applied_frames.append(frame)
        if completion_capture == "start":
            for context_frame in self._recent_applied_frames:
                self._write_completion_frame(context_frame, stage="completion")
            self._completion_capture_active = True
            return
        if self._completion_capture_active:
            self._write_completion_frame(frame, stage="post_stop")
        if completion_capture == "finish":
            self._completion_capture_active = False

    def _write_review_artifacts(self, frame: DetectionFrameData) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sampled": False,
            "raw_rgb": None,
            "raw_depth": None,
            "depth_scale_m": None,
            "table_rgb_edges": None,
            "target_mask": None,
            "table_mask": None,
            "table_completed_mask": None,
            "table_edge_overlay": None,
        }
        if not self.save_images:
            return result
        if frame.frame_index % self.review_stride:
            return result
        self.review_raw_dir.mkdir(parents=True, exist_ok=True)
        self.review_depth_dir.mkdir(parents=True, exist_ok=True)
        self.review_edges_dir.mkdir(parents=True, exist_ok=True)
        self.review_masks_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{int(frame.frame_index):06d}"
        raw_relative = Path("review_samples") / "raw" / f"{stem}.png"
        raw_bgr = cv2.cvtColor(np.asarray(frame.rgb), cv2.COLOR_RGB2BGR)
        _atomic_write_bytes(
            self.output_dir / raw_relative,
            self._encode_png(raw_bgr, name=f"review RGB {stem}"),
        )
        result.update(sampled=True, raw_rgb=raw_relative.as_posix())
        if frame.depth_raw is not None:
            depth = np.asarray(frame.depth_raw)
            if depth.shape != frame.rgb.shape[:2] or depth.dtype != np.uint16:
                raise ValueError("review depth must be aligned uint16 RGB-D")
            depth_relative = Path("review_samples") / "depth" / f"{stem}.png"
            _atomic_write_bytes(
                self.output_dir / depth_relative,
                self._encode_png(depth, name=f"review raw depth {stem}"),
            )
            result["raw_depth"] = depth_relative.as_posix()
            result["depth_scale_m"] = _optional_float(frame.depth_scale_m)
        if frame.table_rgb_edges is not None:
            edge_image = np.asarray(frame.table_rgb_edges)
            if edge_image.shape != frame.rgb.shape[:2]:
                raise ValueError("table RGB edge image shape does not match RGB")
            edge_relative = (
                Path("review_samples") / "edges" / f"{stem}_table_rgb.png"
            )
            _atomic_write_bytes(
                self.output_dir / edge_relative,
                self._encode_png(
                    edge_image.astype(np.uint8, copy=False),
                    name=f"review table RGB edges {stem}",
                ),
            )
            result["table_rgb_edges"] = edge_relative.as_posix()
        for key, suffix, mask in (
            ("target_mask", "target", frame.target_mask),
            ("table_mask", "table", frame.surface_mask),
            (
                "table_completed_mask",
                "table_completed",
                frame.completed_surface_mask,
            ),
        ):
            if mask is None:
                continue
            if mask.shape != frame.rgb.shape[:2]:
                raise ValueError(f"{suffix} mask shape does not match RGB")
            relative = Path("review_samples") / "masks" / f"{stem}_{suffix}.png"
            binary = mask.astype(bool).astype(np.uint8) * 255
            _atomic_write_bytes(
                self.output_dir / relative,
                self._encode_png(binary, name=f"review {suffix} mask {stem}"),
            )
            result[key] = relative.as_posix()
        edge_overlay = self._table_edge_overlay(
            frame.surface_mask, frame.table_geometry
        )
        if edge_overlay is not None:
            relative = (
                Path("review_samples")
                / "masks"
                / f"{stem}_table_edge.png"
            )
            _atomic_write_bytes(
                self.output_dir / relative,
                self._encode_png(
                    edge_overlay, name=f"review table edge overlay {stem}"
                ),
            )
            result["table_edge_overlay"] = relative.as_posix()
        return result

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
            "post_stop_duration_s": value.get("post_stop_duration_s"),
            "post_stop_elapsed_s": value.get("post_stop_elapsed_s"),
            "post_stop_sample_count": value.get("post_stop_sample_count"),
            "post_stop_valid_sample_count": value.get("post_stop_valid_sample_count"),
            "post_stop_invalid_sample_count": value.get("post_stop_invalid_sample_count"),
            "invalid_frames": value.get("invalid_frames"),
            "vertical_recenter_armed": value.get("vertical_recenter_armed"),
            "vertical_recenter_elapsed_s": value.get("vertical_recenter_elapsed_s"),
            "vertical_recenter_stable_frames": value.get("vertical_recenter_stable_frames"),
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
            "duration_s": float(value.get("duration_s", 0.0)),
        }

    def write(
        self,
        frame: DetectionFrameData,
        *,
        controller_state: Mapping[str, Any] | None,
        command: Mapping[str, Any] | None,
        control_applied: bool = True,
        orientation: Mapping[str, Any] | None = None,
        completion_capture: str | None = None,
    ) -> None:
        applied = bool(control_applied)
        if applied and (controller_state is None or command is None):
            raise ValueError("applied diagnostic frame requires control metadata")
        controller = None if not applied else self._controller(controller_state)
        normalized_command = None if not applied else self._command(command)
        if completion_capture not in {None, "start", "finish"}:
            raise ValueError("completion_capture must be start, finish, or None")
        self._capture_completion_images(
            frame,
            control_applied=applied,
            completion_capture=completion_capture,
        )
        review_artifacts = self._write_review_artifacts(frame)
        record = {
            "frame_index": int(frame.frame_index),
            "camera_timestamp": float(frame.camera_timestamp),
            "camera_stream": frame.camera_stream,
            "attempt_id": (
                None if frame.attempt_id is None else int(frame.attempt_id)
            ),
            "failover_stage": frame.failover_stage,
            "reference_source_stream": frame.reference_source_stream,
            "reference_kind": frame.reference_kind,
            "prompt_mode": frame.prompt_mode,
            "perception_kind": frame.perception_kind,
            "perception_error": frame.perception_error,
            "control_applied": applied,
            "completion_capture": completion_capture,
            "annotated_image": None,
            "review_artifacts": review_artifacts,
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
                "target": None if frame.target_geometry is None else dict(frame.target_geometry),
                "table": None if frame.table_geometry is None else dict(frame.table_geometry),
                "table_error": frame.table_geometry_error,
            },
            "controller": controller,
            "command": normalized_command,
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
    completion_capture: str | None


@dataclass(frozen=True)
class _CloseWriter:
    drain: bool


def _owned_camera_frame(frame: CameraFrameData) -> CameraFrameData:
    return replace(
        frame,
        rgb=np.asarray(frame.rgb).copy(),
        target_mask=(
            None if frame.target_mask is None else np.asarray(frame.target_mask).copy()
        ),
        surface_mask=(
            None
            if frame.surface_mask is None
            else np.asarray(frame.surface_mask).copy()
        ),
        completed_surface_mask=(
            None
            if frame.completed_surface_mask is None
            else np.asarray(frame.completed_surface_mask).copy()
        ),
        table_rgb_edges=(
            None
            if frame.table_rgb_edges is None
            else np.asarray(frame.table_rgb_edges).copy()
        ),
        table_geometry=(
            None if frame.table_geometry is None else dict(frame.table_geometry)
        ),
    )


def _owned_frame(frame: DetectionFrameData) -> DetectionFrameData:
    return replace(
        frame,
        rgb=np.asarray(frame.rgb).copy(),
        depth_raw=(
            None if frame.depth_raw is None else np.asarray(frame.depth_raw).copy()
        ),
        table_rgb_edges=(
            None
            if frame.table_rgb_edges is None
            else np.asarray(frame.table_rgb_edges).copy()
        ),
        target_mask=(
            None if frame.target_mask is None else np.asarray(frame.target_mask).copy()
        ),
        surface_mask=(
            None if frame.surface_mask is None else np.asarray(frame.surface_mask).copy()
        ),
        completed_surface_mask=(
            None
            if frame.completed_surface_mask is None
            else np.asarray(frame.completed_surface_mask).copy()
        ),
        target_geometry=(
            None if frame.target_geometry is None else dict(frame.target_geometry)
        ),
        table_geometry=(
            None if frame.table_geometry is None else dict(frame.table_geometry)
        ),
        additional_camera_frames=tuple(
            _owned_camera_frame(camera_frame)
            for camera_frame in frame.additional_camera_frames
        ),
    )


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
        item = _ProducedFrame(
            int(generation),
            Path(output_dir).resolve(),
            _owned_frame(frame),
        )
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
        completion_capture: str | None = None,
    ) -> None:
        item = _ControlDecision(
            int(generation),
            int(frame_index),
            bool(control_applied),
            None if controller_state is None else dict(controller_state),
            None if command is None else dict(command),
            None if orientation is None else dict(orientation),
            completion_capture,
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
                completion_capture=decision.completion_capture,
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
                    key[0], key[1], False, None, None, None, None
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
