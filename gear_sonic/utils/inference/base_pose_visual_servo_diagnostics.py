"""Per-camera-frame diagnostics for the raw YOLOE visual-servo loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Mapping

import cv2
import numpy as np

from gear_sonic.utils.inference.base_pose import _atomic_write_bytes


@dataclass(frozen=True)
class DetectionFrameData:
    frame_index: int
    camera_timestamp: float
    rgb: np.ndarray
    target_bbox_xyxy: tuple[float, float, float, float] | None = None
    target_mask: np.ndarray | None = field(default=None, repr=False)
    target_track_id: int | None = None
    target_confidence: float | None = None
    surface_bbox_xyxy: tuple[float, float, float, float] | None = None
    surface_mask: np.ndarray | None = field(default=None, repr=False)
    surface_track_id: int | None = None
    surface_confidence: float | None = None
    target_geometry: Mapping[str, Any] | None = None
    table_geometry: Mapping[str, Any] | None = None
    table_geometry_error: str | None = None
    perception_kind: str = "observation"
    perception_error: str | None = None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _bbox_json(value: tuple[float, float, float, float] | None) -> list[float] | None:
    return None if value is None else [float(item) for item in value]


def _phase_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


class FrameDiagnosticsWriter:
    """Write one JSONL record and one annotated JPEG for every camera frame."""

    def __init__(self, output_dir: str | Path, *, review_stride: int = 5):
        if review_stride <= 0:
            raise ValueError("review_stride must be positive")
        self.output_dir = Path(output_dir).resolve()
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "raw_servo_frames.jsonl"
        self.review_stride = int(review_stride)
        self.review_raw_dir = self.output_dir / "review_samples" / "raw"
        self.review_masks_dir = self.output_dir / "review_samples" / "masks"

    @staticmethod
    def _encode_png(image: np.ndarray, *, name: str) -> bytes:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OSError(f"failed to encode {name}")
        return encoded.tobytes()

    def _write_review_artifacts(self, frame: DetectionFrameData) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sampled": False,
            "raw_rgb": None,
            "target_mask": None,
            "table_mask": None,
        }
        if frame.frame_index % self.review_stride:
            return result
        self.review_raw_dir.mkdir(parents=True, exist_ok=True)
        self.review_masks_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{int(frame.frame_index):06d}"
        raw_relative = Path("review_samples") / "raw" / f"{stem}.png"
        raw_bgr = cv2.cvtColor(np.asarray(frame.rgb), cv2.COLOR_RGB2BGR)
        _atomic_write_bytes(
            self.output_dir / raw_relative,
            self._encode_png(raw_bgr, name=f"review RGB {stem}"),
        )
        result.update(sampled=True, raw_rgb=raw_relative.as_posix())
        for key, suffix, mask in (
            ("target_mask", "target", frame.target_mask),
            ("table_mask", "table", frame.surface_mask),
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
            "invalid_frames": value.get("invalid_frames"),
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

    @staticmethod
    def _blend_mask(image: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int]) -> None:
        if mask is None or mask.shape != image.shape[:2]:
            return
        selected = mask.astype(bool)
        image[selected] = (
            image[selected].astype(np.float32) * 0.58
            + np.asarray(color, dtype=np.float32) * 0.42
        ).astype(np.uint8)

    @staticmethod
    def _draw_box(
        image: np.ndarray,
        bbox: tuple[float, float, float, float] | None,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        if bbox is None:
            return
        x1, y1, x2, y2 = (int(round(item)) for item in bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (max(0, x1), max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    def _annotate(
        self,
        frame: DetectionFrameData,
        controller: Mapping[str, Any],
        command: Mapping[str, float],
    ) -> np.ndarray:
        image = cv2.cvtColor(np.asarray(frame.rgb), cv2.COLOR_RGB2BGR)
        self._blend_mask(image, frame.surface_mask, (190, 70, 30))
        self._blend_mask(image, frame.target_mask, (40, 190, 40))
        self._draw_box(
            image,
            frame.surface_bbox_xyxy,
            (255, 130, 50),
            f"table id={frame.surface_track_id}",
        )
        self._draw_box(
            image,
            frame.target_bbox_xyxy,
            (30, 255, 30),
            f"target id={frame.target_track_id}",
        )
        height, width = image.shape[:2]
        for fraction, color in (
            (0.08, (30, 30, 255)),
            (0.43, (0, 180, 255)),
            (0.50, (255, 255, 255)),
            (0.57, (0, 180, 255)),
            (0.92, (30, 30, 255)),
        ):
            x = int(round(width * fraction))
            cv2.line(image, (x, 0), (x, height - 1), color, 1)
        errors = controller.get("filtered_errors")
        error_text = "errors=null"
        if errors is not None and len(errors) == 3:
            error_text = "e f/r/y=" + "/".join(f"{float(item):+.3f}" for item in errors)
        lines = (
            f"frame={frame.frame_index} phase={controller.get('phase')} kind={frame.perception_kind}",
            error_text,
            f"cmd vx/vy/wz={command['vx']:+.3f}/{command['vy']:+.3f}/{command['wz']:+.3f}",
        )
        if frame.perception_error:
            lines += (f"error={frame.perception_error}",)
        for index, line in enumerate(lines):
            origin = (6, 18 + index * 18)
            cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def write(
        self,
        frame: DetectionFrameData,
        *,
        controller_state: Mapping[str, Any],
        command: Mapping[str, Any],
    ) -> None:
        controller = self._controller(controller_state)
        normalized_command = self._command(command)
        review_artifacts = self._write_review_artifacts(frame)
        image = self._annotate(frame, controller, normalized_command)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise OSError(f"failed to encode diagnostic frame {frame.frame_index}")
        image_name = f"{int(frame.frame_index):06d}.jpg"
        _atomic_write_bytes(self.frames_dir / image_name, encoded.tobytes())
        record = {
            "frame_index": int(frame.frame_index),
            "camera_timestamp": float(frame.camera_timestamp),
            "perception_kind": frame.perception_kind,
            "perception_error": frame.perception_error,
            "annotated_image": f"frames/{image_name}",
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
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
