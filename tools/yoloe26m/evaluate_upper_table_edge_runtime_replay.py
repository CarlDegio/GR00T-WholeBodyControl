#!/usr/bin/env python3
"""Replay sampled YOLOE frames with the prompt modes recorded by the live run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from gear_sonic.utils.inference.base_pose_dual_visual_servo import (
    DualCameraReference,
    DualReferenceUpdateRequest,
    YoloeDualReferenceEncoder,
)
from gear_sonic.utils.inference.base_pose_visual_servo import (
    YoloePersistentTracker,
    _resolve_target,
)
from tools.yoloe26m.evaluate_upper_table_edge import (
    DEFAULT_MODEL,
    DEFAULT_RUN_DIR,
    _best_instance,
    _load_rgb,
    _normalized_bbox,
    _pixel_edge,
    _save_mask,
    _save_overlay,
)


def _validate_update(
    encoded: Any,
    *,
    require_target_embedding: bool,
    min_confidence: float,
    min_iou: float,
) -> tuple[bool, str]:
    checks = [("surface", encoded.surface_confidence, encoded.surface_iou)]
    if require_target_embedding:
        checks.insert(0, ("target", encoded.target_confidence, encoded.target_iou))
    for role, confidence, overlap in checks:
        if float(confidence) < min_confidence:
            return False, f"{role}_confidence_below_threshold"
        if float(overlap) < min_iou:
            return False, f"{role}_iou_below_threshold"
    return True, "accepted"


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
    frame_paths = sorted((run_dir / "review_samples" / "raw").glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError("no sampled raw frames found")

    runtime_rows = {
        int(row["frame_index"]): row
        for row in (
            json.loads(line)
            for line in (run_dir / "raw_servo_frames.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
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

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    prompt_mode: str | None = None
    target_id: int | None = None
    surface_id: int | None = None

    for sequence_index, frame_path in enumerate(frame_paths):
        frame_index = int(frame_path.stem)
        runtime_row = runtime_rows[frame_index]
        desired_mode = str(runtime_row["prompt_mode"])
        mode_changed = desired_mode != prompt_mode
        if mode_changed:
            if desired_mode == "visual":
                tracker.start(
                    initial_rgb,
                    target_prompt=target_prompt,
                    target_bbox=initial_target,
                    surface_prompt="desk",
                    surface_bboxes=initial_tables,
                )
            elif desired_mode == "target_text_surface_visual":
                tracker.start_text(
                    initial_rgb,
                    target_prompt=target_prompt,
                    target_bbox=initial_target,
                    surface_prompt="desk",
                    surface_bboxes=initial_tables,
                )
            else:
                raise ValueError(f"unsupported sampled prompt mode: {desired_mode}")
            prompt_mode = desired_mode
            target_id = None
            surface_id = None

        rgb = _load_rgb(frame_path)
        height, width = rgb.shape[:2]
        instances = tracker.track(rgb)
        if mode_changed:
            target = _best_instance(instances, 0)
            surface = _best_instance(instances, 1)
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
            "prompt_mode": prompt_mode,
            "mode_changed": mode_changed,
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

        if (
            target is not None
            and surface is not None
            and target.confidence >= min_reference_confidence
        ):
            require_target_embedding = prompt_mode == "visual"
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
                require_target_embedding=require_target_embedding,
            )
            try:
                encoded = encoder.extract(request)
                accepted, reason = _validate_update(
                    encoded,
                    require_target_embedding=require_target_embedding,
                    min_confidence=min_reference_confidence,
                    min_iou=min_reference_iou,
                )
            except Exception as exc:
                encoded = None
                accepted = False
                reason = f"error:{type(exc).__name__}:{exc}"
            if accepted and encoded is not None:
                if require_target_embedding:
                    tracker.install_reference_embeddings(
                        target_embedding=encoded.target_embedding,
                        surface_embedding=encoded.surface_embedding,
                    )
                else:
                    tracker.install_surface_embedding(encoded.surface_embedding)
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
        refresh = row["reference_update"]
        print(
            f"frame={frame_index:06d} mode={prompt_mode} "
            f"table={'yes' if surface is not None else 'no'} "
            f"upper={None if row['edge'] is None else row['edge']['selected_upper']} "
            f"refresh={None if refresh is None else refresh['reason']}",
            flush=True,
        )

    detected = [row for row in records if row["table"] is not None]
    edged = [row for row in detected if row["edge"] is not None]
    upper = [row for row in edged if bool(row["edge"]["selected_upper"])]
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
        "edge_rate_given_table": len(edged) / len(detected) if detected else 0.0,
        "automatic_upper_count": len(upper),
        "automatic_upper_rate_given_table": (
            len(upper) / len(detected) if detected else 0.0
        ),
        "reference_update_count": len(updates),
        "reference_update_accepted_count": sum(
            bool(item["accepted"]) for item in updates
        ),
        "prompt_mode_switches": [
            {
                "frame_index": row["frame_index"],
                "prompt_mode": row["prompt_mode"],
            }
            for row in records
            if row["mode_changed"]
        ],
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
        else (run_dir / "upper_edge_replay_no_depth_runtime_modes").resolve()
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
