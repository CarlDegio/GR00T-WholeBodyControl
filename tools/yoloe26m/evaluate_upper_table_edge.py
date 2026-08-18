#!/usr/bin/env python3
"""Replay sampled dual-camera YOLOE frames and evaluate mask-only upper edges."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from gear_sonic.utils.inference.base_pose_dual_visual_servo import (
    DualCameraReference,
    DualReferenceUpdateRequest,
    YoloeDualReferenceEncoder,
)
from gear_sonic.utils.inference.base_pose_visual_servo import (
    TrackedInstance,
    YoloePersistentTracker,
    _resolve_target,
    select_initial_instance,
)


DEFAULT_RUN_DIR = Path(
    "outputs/base_pose_adjustment/dual_raw_yoloe_20260817_231516_g8"
)
DEFAULT_MODEL = Path("tools/yoloe26m/weights/yoloe-26m-seg.pt")


@dataclass(frozen=True)
class PixelEdge:
    endpoints_px: tuple[tuple[float, float], tuple[float, float]]
    length_px: float
    inlier_count: int
    residual_px: float
    upper_support_ratio: float
    median_upper_gap_px: float
    median_row_px: float
    selected_upper: bool


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"failed to decode RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _normalized_bbox(
    bbox: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return (
        1000.0 * x1 / width,
        1000.0 * y1 / height,
        1000.0 * x2 / width,
        1000.0 * y2 / height,
    )


def _best_instance(
    instances: Sequence[TrackedInstance], class_index: int
) -> TrackedInstance | None:
    candidates = [item for item in instances if item.class_index == class_index]
    return max(candidates, key=lambda item: item.confidence, default=None)


def _pixel_edge(mask: np.ndarray) -> PixelEdge:
    binary = np.asarray(mask, dtype=np.uint8)
    contour = binary - cv2.erode(
        binary, np.ones((7, 7), dtype=np.uint8), iterations=1
    )
    v, u = np.nonzero(contour > 0)
    if len(u) < 50:
        raise ValueError("table contour has fewer than 50 pixels")

    foreground = binary > 0
    populated_columns = np.any(foreground, axis=0)
    upper_rows = np.full(binary.shape[1], np.inf, dtype=np.float64)
    upper_rows[populated_columns] = np.argmax(
        foreground[:, populated_columns], axis=0
    )

    pixels = np.column_stack((u, v)).astype(np.float64)
    if len(pixels) > 2500:
        indices = np.linspace(0, len(pixels) - 1, 2500, dtype=int)
        pixels = pixels[indices]

    rng = np.random.default_rng(0)
    best: tuple[tuple[float, ...], np.ndarray] | None = None
    for _ in range(500):
        a, b = pixels[rng.choice(len(pixels), size=2, replace=False)]
        direction = b - a
        norm = float(np.linalg.norm(direction))
        if norm < 30.0:
            continue
        direction /= norm
        distances = np.abs(
            (pixels[:, 0] - a[0]) * direction[1]
            - (pixels[:, 1] - a[1]) * direction[0]
        )
        inliers = distances <= 3.0
        count = int(np.count_nonzero(inliers))
        if count < 50:
            continue
        inlier_pixels = pixels[inliers]
        projection = (inlier_pixels - a) @ direction
        length = float(np.max(projection) - np.min(projection))
        gaps = (
            inlier_pixels[:, 1]
            - upper_rows[np.rint(inlier_pixels[:, 0]).astype(int)]
        )
        upper_support = float(np.mean(gaps <= 6.0))
        median_gap = float(np.median(gaps))
        median_row = float(np.median(inlier_pixels[:, 1]))
        viable = float(length >= 50.0)
        upper_candidate = upper_support >= 0.50 and median_gap <= 6.0
        if upper_candidate:
            rank = (
                viable,
                1.0,
                upper_support,
                -median_gap,
                -median_row,
                float(count),
                length,
            )
        else:
            rank = (
                viable,
                0.0,
                float(count),
                length,
                -float(np.median(distances[inliers])),
                0.0,
                0.0,
            )
        if best is None or rank > best[0]:
            best = (rank, inliers)

    if best is None:
        raise ValueError("no stable pixel table edge found")

    inlier_pixels = pixels[best[1]]
    center = np.mean(inlier_pixels, axis=0)
    _, _, vh = np.linalg.svd(inlier_pixels - center, full_matrices=False)
    direction = vh[0]
    projection = (inlier_pixels - center) @ direction
    residuals = np.abs(
        (inlier_pixels[:, 0] - center[0]) * direction[1]
        - (inlier_pixels[:, 1] - center[1]) * direction[0]
    )
    endpoint_indices = (int(np.argmin(projection)), int(np.argmax(projection)))
    endpoints = tuple(
        tuple(float(value) for value in center + projection[index] * direction)
        for index in endpoint_indices
    )
    if endpoints[1] < endpoints[0]:
        endpoints = (endpoints[1], endpoints[0])
    gaps = (
        inlier_pixels[:, 1]
        - upper_rows[np.rint(inlier_pixels[:, 0]).astype(int)]
    )
    upper_support = float(np.mean(gaps <= 6.0))
    median_gap = float(np.median(gaps))
    return PixelEdge(
        endpoints_px=endpoints,
        length_px=float(np.max(projection) - np.min(projection)),
        inlier_count=int(len(inlier_pixels)),
        residual_px=float(np.median(residuals)),
        upper_support_ratio=upper_support,
        median_upper_gap_px=median_gap,
        median_row_px=float(np.median(inlier_pixels[:, 1])),
        selected_upper=upper_support >= 0.50 and median_gap <= 6.0,
    )


def _validate_reference_update(
    encoded: Any,
    *,
    min_confidence: float,
    min_iou: float,
) -> tuple[bool, str]:
    for role, confidence, overlap in (
        ("target", encoded.target_confidence, encoded.target_iou),
        ("surface", encoded.surface_confidence, encoded.surface_iou),
    ):
        if float(confidence) < min_confidence:
            return False, f"{role}_confidence_below_threshold"
        if float(overlap) < min_iou:
            return False, f"{role}_iou_below_threshold"
    return True, "accepted"


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise OSError(f"failed to save mask: {path}")


def _save_overlay(
    rgb: np.ndarray,
    surface: TrackedInstance,
    edge: PixelEdge | None,
    path: Path,
    *,
    frame_index: int,
    edge_error: str | None,
) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = surface.mask.astype(bool)
    tint = np.zeros_like(bgr)
    tint[:] = (180, 80, 30)
    bgr[mask] = cv2.addWeighted(bgr[mask], 0.55, tint[mask], 0.45, 0.0)
    x1, y1, x2, y2 = (int(round(value)) for value in surface.bbox_xyxy)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 170, 40), 2)
    if edge is not None:
        start = tuple(int(round(value)) for value in edge.endpoints_px[0])
        end = tuple(int(round(value)) for value in edge.endpoints_px[1])
        cv2.line(bgr, start, end, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(bgr, start, 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(bgr, end, 5, (0, 255, 255), -1, cv2.LINE_AA)
        status = (
            f"upper={edge.selected_upper} support={edge.upper_support_ratio:.2f} "
            f"gap={edge.median_upper_gap_px:.1f}px"
        )
    else:
        status = f"edge_error={edge_error}"
    for row, label in enumerate(
        (
            f"frame={frame_index} table_conf={surface.confidence:.3f}",
            status,
        )
    ):
        cv2.putText(
            bgr,
            label,
            (8, 22 + 20 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            label,
            (8, 22 + 20 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"failed to save overlay: {path}")


def evaluate(
    run_dir: Path,
    output_dir: Path,
    *,
    model_path: Path,
    device: str,
    confidence: float,
    imgsz: int,
    min_reference_confidence: float,
    min_reference_iou: float,
) -> dict[str, Any]:
    raw_dir = run_dir / "review_samples" / "raw"
    frame_paths = sorted(raw_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"no PNG frames found in {raw_dir}")

    summary = json.loads((run_dir / "initial_reference_summary.json").read_text())
    selected_stream = str(summary["selected_initial_stream"])
    initial = summary["streams"][selected_stream]["reference"]
    if initial is None:
        raise ValueError(f"selected stream {selected_stream} has no reference")
    initial_rgb = _load_rgb(run_dir / selected_stream / "initial_rgb.png")
    target_prompt = str(initial["target_prompt"])
    initial_target = tuple(float(value) for value in initial["target_bbox"])
    initial_tables = tuple(
        tuple(float(value) for value in bbox)
        for bbox in initial["table_bboxes"]
    )

    tracker = YoloePersistentTracker(
        model_path,
        confidence=confidence,
        imgsz=imgsz,
        device=device,
    )
    encoder = YoloeDualReferenceEncoder(
        model_path,
        confidence=confidence,
        imgsz=imgsz,
        device=device,
    )
    tracker.start(
        initial_rgb,
        target_prompt=target_prompt,
        target_bbox=initial_target,
        surface_prompt="desk",
        surface_bboxes=initial_tables,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    target_id: int | None = None
    surface_id: int | None = None
    for sequence_index, frame_path in enumerate(frame_paths):
        frame_index = int(frame_path.stem)
        rgb = _load_rgb(frame_path)
        height, width = rgb.shape[:2]
        instances = tracker.track(rgb)
        if sequence_index == 0:
            target = select_initial_instance(
                instances,
                class_index=0,
                grounded_bboxes=(initial_target,),
                width=width,
                height=height,
            )
            surface = select_initial_instance(
                instances,
                class_index=1,
                grounded_bboxes=initial_tables,
                width=width,
                height=height,
            )
            target_reacquired = False
            surface_reacquired = False
        else:
            if target_id is None:
                target = _best_instance(instances, 0)
                target_reacquired = target is not None
            else:
                target, target_reacquired, _ = _resolve_target(
                    list(instances), target_id, class_index=0
                )
            if surface_id is None:
                surface = _best_instance(instances, 1)
                surface_reacquired = surface is not None
            else:
                surface, surface_reacquired, _ = _resolve_target(
                    list(instances), surface_id, class_index=1
                )

        row: dict[str, Any] = {
            "sequence_index": sequence_index,
            "frame_index": frame_index,
            "frame_path": str(frame_path.relative_to(run_dir)),
            "target": None,
            "table": None,
            "edge": None,
            "edge_error": None,
            "reference_update": None,
            "target_reacquired": target_reacquired,
            "surface_reacquired": surface_reacquired,
        }
        if target is not None:
            target_id = target.track_id
            row["target"] = {
                "bbox_xyxy": list(target.bbox_xyxy),
                "confidence": target.confidence,
                "track_id": target.track_id,
                "mask_pixels": int(np.count_nonzero(target.mask)),
            }
        if surface is not None:
            surface_id = surface.track_id
            row["table"] = {
                "bbox_xyxy": list(surface.bbox_xyxy),
                "confidence": surface.confidence,
                "track_id": surface.track_id,
                "mask_pixels": int(np.count_nonzero(surface.mask)),
            }
            try:
                edge = _pixel_edge(surface.mask)
            except ValueError as exc:
                edge = None
                row["edge_error"] = str(exc)
            else:
                row["edge"] = asdict(edge)
            _save_mask(
                surface.mask,
                output_dir / "masks" / f"{frame_index:06d}_table.png",
            )
            _save_overlay(
                rgb,
                surface,
                edge,
                output_dir / "overlays" / f"{frame_index:06d}.jpg",
                frame_index=frame_index,
                edge_error=row["edge_error"],
            )

        if target is not None and surface is not None:
            reference = DualCameraReference(
                stream_name=selected_stream,
                rgb=rgb,
                target_prompt=target_prompt,
                target_bbox=_normalized_bbox(target.bbox_xyxy, width, height),
                table_bboxes=(
                    _normalized_bbox(surface.bbox_xyxy, width, height),
                ),
                camera_timestamp=float(sequence_index),
                kind="latest",
            )
            request = DualReferenceUpdateRequest(
                generation=1,
                frame_index=frame_index,
                reference=reference,
                require_target_embedding=True,
            )
            try:
                encoded = encoder.extract(request)
                accepted, reason = _validate_reference_update(
                    encoded,
                    min_confidence=min_reference_confidence,
                    min_iou=min_reference_iou,
                )
            except Exception as exc:
                accepted = False
                reason = f"error:{type(exc).__name__}:{exc}"
                encoded = None
            if accepted and encoded is not None:
                tracker.install_reference_embeddings(
                    target_embedding=encoded.target_embedding,
                    surface_embedding=encoded.surface_embedding,
                )
            row["reference_update"] = {
                "accepted": accepted,
                "reason": reason,
                "target_confidence": (
                    None if encoded is None else encoded.target_confidence
                ),
                "target_iou": None if encoded is None else encoded.target_iou,
                "surface_confidence": (
                    None if encoded is None else encoded.surface_confidence
                ),
                "surface_iou": None if encoded is None else encoded.surface_iou,
            }

        records.append(row)
        (output_dir / "frames.jsonl").write_text(
            "".join(
                json.dumps(item, ensure_ascii=False) + "\n"
                for item in records
            )
        )
        print(
            f"frame={frame_index:06d} "
            f"table={'yes' if surface is not None else 'no'} "
            f"upper={None if row['edge'] is None else row['edge']['selected_upper']} "
            f"refresh={None if row['reference_update'] is None else row['reference_update']['reason']}",
            flush=True,
        )

    detected = [row for row in records if row["table"] is not None]
    edged = [row for row in detected if row["edge"] is not None]
    upper = [
        row
        for row in edged
        if bool(row["edge"]["selected_upper"])
    ]
    updates = [
        row["reference_update"]
        for row in records
        if row["reference_update"] is not None
    ]
    report = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "frame_count": len(records),
        "table_detection_count": len(detected),
        "table_detection_rate": len(detected) / len(records),
        "edge_count": len(edged),
        "edge_rate_given_table": (
            len(edged) / len(detected) if detected else 0.0
        ),
        "automatic_upper_count": len(upper),
        "automatic_upper_rate_given_table": (
            len(upper) / len(detected) if detected else 0.0
        ),
        "reference_update_count": len(updates),
        "reference_update_accepted_count": sum(
            bool(item["accepted"]) for item in updates
        ),
        "config": {
            "model_path": str(model_path),
            "device": device,
            "confidence": confidence,
            "imgsz": imgsz,
            "reference_min_confidence": min_reference_confidence,
            "reference_min_iou": min_reference_iou,
            "mask_contour_kernel": [7, 7],
            "upper_band_px": 6.0,
            "upper_min_support_ratio": 0.50,
            "pixel_ransac_threshold_px": 3.0,
            "pixel_min_length_px": 50.0,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--reference-min-confidence", type=float, default=0.35)
    parser.add_argument("--reference-min-iou", type=float, default=0.50)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (run_dir / "upper_edge_replay_no_depth").resolve()
    )
    report = evaluate(
        run_dir,
        output_dir,
        model_path=args.model.resolve(),
        device=args.device,
        confidence=args.confidence,
        imgsz=args.imgsz,
        min_reference_confidence=args.reference_min_confidence,
        min_reference_iou=args.reference_min_iou,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
