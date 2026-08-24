#!/usr/bin/env python3
"""Run YOLOE target/surface segmentation and RGB edge filtering from VA output.

The image-only test intentionally stops before the production depth-based line
selection. It saves every intermediate mask and all Hough line candidates that
survive the remaining production filters.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import cv2
import numpy as np

from gear_sonic.utils.inference.base_pose.servo import (
    TrackedInstance,
    YoloePersistentTracker,
    _dilate_table_edge_mask,
    _largest_filled_component,
    _rgb_edge_line_segments,
    _rgb_mask_edge_intersection,
    _table_edge_target_exclusion,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "tools/yoloe26m/weights/yoloe-26m-seg.pt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/va_yoloe_edge_tests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Feed a standalone VA result into YOLOE and the RGB/Hough edge "
            "pipeline, skipping only depth-based line selection."
        )
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--va-result",
        required=True,
        type=Path,
        help="validated_result.json from tools/test_va_align_grounding.py",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--target-text",
        help="Optional YOLOE target prompt override; VA bbox remains unchanged.",
    )
    parser.add_argument(
        "--surface-text",
        help="Optional YOLOE surface prompt override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New output directory; defaults under outputs/va_yoloe_edge_tests.",
    )
    return parser.parse_args()


def load_va_selection(path: Path) -> tuple[str, str, list[float]]:
    source = path.expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"target", "surface"}:
        raise ValueError("VA selection must contain only target and surface")
    target = value["target"]
    surface = value["surface"]
    if not isinstance(target, dict) or not isinstance(surface, dict):
        raise ValueError("VA target and surface must be objects")
    target_text = str(target.get("text", "")).strip()
    surface_text = str(surface.get("text", "")).strip()
    bbox = target.get("bbox_2d")
    if not target_text or not surface_text:
        raise ValueError("VA target and surface text must be non-empty")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            and 0.0 <= float(item) <= 1000.0
            for item in bbox
        )
    ):
        raise ValueError("VA target bbox_2d must contain four normalized values")
    x1, y1, x2, y2 = map(float, bbox)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("VA target bbox_2d has invalid corner order")
    return target_text, surface_text, [x1, y1, x2, y2]


def new_output_dir(requested: Path | None, target_text: str) -> Path:
    if requested is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", target_text).strip("_").lower()
        path = DEFAULT_OUTPUT_ROOT / f"{stamp}_{slug or 'target'}"
    else:
        path = requested.expanduser()
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def normalized_bbox_to_pixels(
    bbox: Sequence[float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    return (
        float(bbox[0]) * width / 1000.0,
        float(bbox[1]) * height / 1000.0,
        float(bbox[2]) * width / 1000.0,
        float(bbox[3]) * height / 1000.0,
    )


def bbox_iou(
    left: Sequence[float], right: Sequence[float]
) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def best_instance(
    instances: Sequence[TrackedInstance], class_index: int
) -> TrackedInstance | None:
    candidates = [item for item in instances if item.class_index == class_index]
    return max(candidates, key=lambda item: item.confidence, default=None)


def predict_without_tracking(
    tracker: YoloePersistentTracker,
    rgb: np.ndarray,
) -> list[TrackedInstance]:
    """Use the same configured YOLOE classes without requiring BoT-SORT IDs."""

    from ultralytics.utils import ops

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result = tracker.model.predict(
        bgr,
        device=tracker.device,
        imgsz=tracker.imgsz,
        conf=tracker.confidence,
        half=True,
        quantize=16,
        verbose=False,
        save=False,
    )[0]
    if result.boxes is None or result.masks is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    mask_data = result.masks.data
    scaled = ops.scale_masks(
        mask_data[:, None],
        rgb.shape[:2],
        padding=True,
        mode="nearest",
    )[:, 0]
    masks = scaled.detach().cpu().numpy() > 0.5
    instances = []
    for index, class_index in enumerate(classes):
        if class_index not in {0, 1}:
            continue
        instances.append(
            TrackedInstance(
                track_id=index + 1,
                class_index=int(class_index),
                confidence=float(confidences[index]),
                bbox_xyxy=tuple(float(item) for item in boxes[index]),
                mask=masks[index],
            )
        )
    return instances


def write_mask(path: Path, mask: np.ndarray) -> None:
    image = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write mask: {path}")


def overlay_mask(
    bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float = 0.35,
) -> np.ndarray:
    output = bgr.copy()
    binary = np.asarray(mask) > 0
    tint = np.zeros_like(output)
    tint[:, :] = color
    output[binary] = cv2.addWeighted(
        output, 1.0 - alpha, tint, alpha, 0.0
    )[binary]
    return output


def instance_record(
    item: TrackedInstance,
    *,
    class_names: Sequence[str],
    va_bbox_px: Sequence[float],
) -> dict[str, Any]:
    return {
        "track_id": item.track_id,
        "class_index": item.class_index,
        "class_name": class_names[item.class_index],
        "confidence": item.confidence,
        "bbox_xyxy": list(item.bbox_xyxy),
        "mask_pixels": int(np.count_nonzero(item.mask)),
        "iou_with_va_target_bbox": bbox_iou(item.bbox_xyxy, va_bbox_px),
    }


def draw_detections(
    bgr: np.ndarray,
    instances: Sequence[TrackedInstance],
    class_names: Sequence[str],
    va_bbox_px: Sequence[float],
) -> np.ndarray:
    output = bgr.copy()
    colors = ((0, 255, 0), (255, 120, 0))
    for item in instances:
        x1, y1, x2, y2 = (int(round(value)) for value in item.bbox_xyxy)
        color = colors[item.class_index]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        label = (
            f"{class_names[item.class_index]} "
            f"{item.confidence:.3f} id={item.track_id}"
        )
        cv2.putText(
            output,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    vx1, vy1, vx2, vy2 = (int(round(value)) for value in va_bbox_px)
    cv2.rectangle(output, (vx1, vy1), (vx2, vy2), (255, 0, 255), 3)
    cv2.putText(
        output,
        "VA target bbox",
        (vx1, max(24, vy1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def draw_segments(
    bgr: np.ndarray,
    segments: Sequence[Any],
    *,
    selected_only: bool,
) -> np.ndarray:
    output = bgr.copy()
    displayed = segments[:1] if selected_only else segments
    for index, segment in enumerate(displayed):
        start = tuple(int(round(value)) for value in segment.endpoint_a)
        end = tuple(int(round(value)) for value in segment.endpoint_b)
        color = (0, 0, 255) if selected_only else (0, 255, 255)
        thickness = 5 if selected_only else 2
        cv2.line(output, start, end, color, thickness, cv2.LINE_AA)
        if not selected_only:
            center = (
                int(round(0.5 * (start[0] + end[0]))),
                int(round(0.5 * (start[1] + end[1]))),
            )
            cv2.putText(
                output,
                f"{index}:{segment.length:.0f}px",
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return output


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def run(args: argparse.Namespace) -> int:
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image does not exist: {image_path}")
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be in (0, 1]")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")

    va_target_text, va_surface_text, va_bbox = load_va_selection(args.va_result)
    target_text = (
        args.target_text.strip() if args.target_text is not None else va_target_text
    )
    surface_text = (
        args.surface_text.strip() if args.surface_text is not None else va_surface_text
    )
    if not target_text or not surface_text:
        raise ValueError("YOLOE prompt overrides must be non-empty")
    output_dir = new_output_dir(args.output_dir, target_text)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to decode image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    va_bbox_px = normalized_bbox_to_pixels(
        va_bbox,
        width=width,
        height=height,
    )
    save_image(output_dir / "00_input.jpg", bgr)

    tracker = YoloePersistentTracker(
        args.model,
        confidence=args.confidence,
        imgsz=args.imgsz,
        device=args.device,
        surface_prompt=surface_text,
    )
    prompt_artifact = tracker.start_all_text(target_prompt=target_text)
    instances = list(tracker.track(rgb))
    detection_mode = "track"
    if not instances:
        instances = predict_without_tracking(tracker, rgb)
        detection_mode = "predict_fallback_after_empty_track"
    class_names = (target_text, surface_text)
    target = best_instance(instances, 0)
    surface = best_instance(instances, 1)

    detection_records = [
        instance_record(
            item,
            class_names=class_names,
            va_bbox_px=va_bbox_px,
        )
        for item in instances
    ]
    save_image(
        output_dir / "01_yoloe_detections.jpg",
        draw_detections(bgr, instances, class_names, va_bbox_px),
    )

    base_result: dict[str, Any] = {
        "image": str(image_path),
        "va_result": str(args.va_result.expanduser().resolve()),
        "image_size": {"width": width, "height": height},
        "target_prompt": target_text,
        "surface_prompt": surface_text,
        "va_target_text": va_target_text,
        "va_surface_text": va_surface_text,
        "prompt_overrides_applied": {
            "target": target_text != va_target_text,
            "surface": surface_text != va_surface_text,
        },
        "va_target_bbox_normalized": va_bbox,
        "va_target_bbox_pixels": list(va_bbox_px),
        "prompt_artifact": prompt_artifact,
        "detection_mode": detection_mode,
        "yoloe_confidence_threshold": args.confidence,
        "yoloe_imgsz": args.imgsz,
        "detections": detection_records,
        "selected_target_track_id": None if target is None else target.track_id,
        "selected_surface_track_id": None if surface is None else surface.track_id,
        "depth_filter_applied": False,
    }

    if target is None or surface is None:
        missing = []
        if target is None:
            missing.append("target")
        if surface is None:
            missing.append("surface")
        base_result.update(
            status="missing_yoloe_" + "_and_".join(missing),
            filtered_line_count=0,
            selected_line=None,
        )
        (output_dir / "result.json").write_text(
            json.dumps(base_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(base_result, ensure_ascii=False, indent=2))
        print(f"Saved partial artifacts to: {output_dir}", file=sys.stderr)
        return 1

    write_mask(output_dir / "02_target_mask.png", target.mask)
    write_mask(output_dir / "03_surface_mask_raw.png", surface.mask)
    surface_component = _largest_filled_component(surface.mask)
    write_mask(output_dir / "04_surface_mask_largest_filled.png", surface_component)
    surface_mask = _dilate_table_edge_mask(surface_component)
    write_mask(output_dir / "05_surface_mask_dilated.png", surface_mask)
    target_exclusion = _table_edge_target_exclusion(target.mask, surface_mask.shape)
    write_mask(output_dir / "06_target_exclusion_mask.png", target_exclusion)
    effective_surface_mask = (surface_mask > 0) & ~(target_exclusion > 0)
    write_mask(output_dir / "07_effective_surface_mask.png", effective_surface_mask)

    edge_intersection = _rgb_mask_edge_intersection(
        rgb,
        surface_mask,
        exclusion_mask=target_exclusion,
    )
    save_image(output_dir / "08_surface_rgb_edge_intersection.png", edge_intersection)
    edge_overlay = bgr.copy()
    edge_overlay[edge_intersection > 0] = (0, 0, 255)
    save_image(output_dir / "09_surface_rgb_edges_overlay.jpg", edge_overlay)

    combined_mask_overlay = overlay_mask(bgr, surface_mask, (255, 0, 0))
    combined_mask_overlay = overlay_mask(
        combined_mask_overlay,
        target.mask,
        (0, 255, 0),
    )
    save_image(output_dir / "10_target_surface_masks_overlay.jpg", combined_mask_overlay)

    segments = _rgb_edge_line_segments(edge_intersection)
    save_image(
        output_dir / "11_all_filtered_hough_lines.jpg",
        draw_segments(bgr, segments, selected_only=False),
    )
    selected_line = None
    if segments:
        longest = segments[0]
        endpoint_a = np.asarray(longest.endpoint_a, dtype=np.float64)
        endpoint_b = np.asarray(longest.endpoint_b, dtype=np.float64)
        if tuple(endpoint_b) < tuple(endpoint_a):
            endpoint_a, endpoint_b = endpoint_b, endpoint_a
        delta = endpoint_b - endpoint_a
        selected_line = {
            "selection": "longest_rgb_hough_candidate_without_depth",
            "endpoint_a": endpoint_a.tolist(),
            "endpoint_b": endpoint_b.tolist(),
            "length_px": float(np.linalg.norm(delta)),
            "image_yaw_error_rad": math.atan2(
                -float(delta[1]),
                float(delta[0]),
            ),
        }
        save_image(
            output_dir / "12_longest_line_no_depth.jpg",
            draw_segments(bgr, segments, selected_only=True),
        )

    line_records = [
        {
            "rank_by_length": index,
            "endpoint_a": segment.endpoint_a.tolist(),
            "endpoint_b": segment.endpoint_b.tolist(),
            "length_px": segment.length,
        }
        for index, segment in enumerate(segments)
    ]
    base_result.update(
        status="ok" if segments else "no_filtered_hough_line",
        target_selection="highest_confidence_class_0_matching_production",
        surface_selection="highest_confidence_class_1_matching_production",
        selected_target=instance_record(
            target,
            class_names=class_names,
            va_bbox_px=va_bbox_px,
        ),
        selected_surface=instance_record(
            surface,
            class_names=class_names,
            va_bbox_px=va_bbox_px,
        ),
        edge_filters={
            "largest_surface_component_and_hole_fill": True,
            "surface_mask_dilation_px": 3,
            "target_exclusion_radius_px": 20,
            "gaussian_kernel": 5,
            "canny_low": 50,
            "canny_high": 150,
            "hough_threshold": 25,
            "hough_min_line_length_px": 75.0,
            "hough_max_line_gap_px": 12,
            "depth_candidate_selection": "skipped",
        },
        filtered_line_count=len(segments),
        filtered_lines=line_records,
        selected_line=selected_line,
    )
    (output_dir / "result.json").write_text(
        json.dumps(base_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(base_result, ensure_ascii=False, indent=2))
    print(f"Saved artifacts to: {output_dir}", file=sys.stderr)
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
