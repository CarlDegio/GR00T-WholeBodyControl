#!/usr/bin/env python3
"""Feed a VA ALIGN result through YOLOE and RGB edge-line filtering.

The script mirrors BasePose's non-depth yaw-align edge stages. It deliberately
omits depth sampling/ranking and chooses the longest surviving Hough segment
for visualization. When both roles use identical text, one YOLOE instance and
mask is reused and self-exclusion is disabled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "tools/yoloe26m/weights/yoloe-26m-seg.pt"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vatest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VA-guided YOLOE segmentation and non-depth edge filtering."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--yaw-align-target", required=True)
    parser.add_argument(
        "--target-bbox",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        required=True,
        help="VA target bbox in normalized [0,1000] coordinates.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalized_bbox_to_pixels(
    bbox: Sequence[float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    if len(bbox) != 4 or not all(math.isfinite(float(item)) for item in bbox):
        raise ValueError("normalized bbox must contain four finite values")
    x1, y1, x2, y2 = map(float, bbox)
    if not (0.0 <= x1 < x2 <= 1000.0 and 0.0 <= y1 < y2 <= 1000.0):
        raise ValueError("normalized bbox must have ordered [0,1000] corners")
    return (
        x1 * width / 1000.0,
        y1 * height / 1000.0,
        x2 * width / 1000.0,
        y2 * height / 1000.0,
    )


def bbox_iou(
    left: Sequence[float], right: Sequence[float]
) -> float:
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def write_image(path: Path, image: Any) -> None:
    import cv2

    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def colorize_mask(mask: Any, *, color: tuple[int, int, int]) -> Any:
    import numpy as np

    binary = np.asarray(mask) > 0
    rendered = np.zeros((*binary.shape, 3), dtype=np.uint8)
    rendered[binary] = color
    return rendered


def draw_bbox(
    image: Any,
    bbox: Sequence[float],
    *,
    color: tuple[int, int, int],
    label: str,
    thickness: int = 4,
) -> None:
    import cv2

    x1, y1, x2, y2 = (int(round(float(item))) for item in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        image,
        label,
        (max(0, x1), max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def select_detection(
    detections: Sequence[dict[str, Any]],
    *,
    class_index: int,
    reference_bbox: Sequence[float] | None,
) -> dict[str, Any] | None:
    candidates = [
        detection
        for detection in detections
        if int(detection["class_index"]) == int(class_index)
    ]
    if not candidates:
        return None
    if reference_bbox is None:
        return max(candidates, key=lambda item: float(item["confidence"]))
    return max(
        candidates,
        key=lambda item: (
            bbox_iou(item["bbox_xyxy"], reference_bbox),
            float(item["confidence"]),
        ),
    )


def result_detections(result: Any, image_shape: tuple[int, int]) -> list[dict[str, Any]]:
    import numpy as np
    from ultralytics.utils import ops

    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    mask_data = result.masks.data
    scaled = ops.scale_masks(
        mask_data[:, None],
        image_shape,
        padding=True,
        mode="nearest",
    )[:, 0]
    masks = scaled.detach().cpu().numpy() > 0.5
    return [
        {
            "index": int(index),
            "class_index": int(classes[index]),
            "confidence": float(confidences[index]),
            "bbox_xyxy": [float(item) for item in boxes[index]],
            "mask": np.asarray(masks[index], dtype=np.uint8),
        }
        for index in range(len(boxes))
    ]


def serialize_detection(
    detection: dict[str, Any],
    *,
    class_names: Sequence[str],
    reference_bbox: Sequence[float] | None = None,
) -> dict[str, Any]:
    class_index = int(detection["class_index"])
    return {
        "index": int(detection["index"]),
        "class_index": class_index,
        "class_name": class_names[class_index],
        "confidence": float(detection["confidence"]),
        "bbox_xyxy": list(detection["bbox_xyxy"]),
        "reference_bbox_iou": (
            None
            if reference_bbox is None
            else bbox_iou(detection["bbox_xyxy"], reference_bbox)
        ),
        "mask_pixels": int(detection["mask"].sum()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLOE

    from gear_sonic.utils.inference.base_pose.servo import (
        _dilate_yaw_align_edge_mask,
        _largest_filled_component,
        _rgb_edge_line_segments,
        _rgb_mask_edge_intersection,
        _yaw_align_edge_target_exclusion,
    )

    image_path = args.image.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image does not exist: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLOE model does not exist: {model_path}")
    # Stage artifacts use fixed names so an interrupted diagnostic run can be
    # resumed into the requested directory without deleting unrelated files.
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to decode image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_bgr.shape[:2]
    target_reference = normalized_bbox_to_pixels(
        args.target_bbox,
        width=width,
        height=height,
    )
    yaw_align_target_reference = None

    reference_overlay = image_bgr.copy()
    draw_bbox(
        reference_overlay,
        target_reference,
        color=(255, 255, 0),
        label=f"VA TARGET: {args.target}",
    )
    write_image(output_dir / "01_input_va_bbox.jpg", reference_overlay)

    same_prompt = " ".join(args.target.casefold().split()) == " ".join(
        args.yaw_align_target.casefold().split()
    )
    class_names = [args.target] if same_prompt else [args.target, args.yaw_align_target]
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(model_path.parents[1] / "config"),
    )
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    device = args.device if args.device else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLOE(str(model_path))
    embeddings = model.get_text_pe(class_names)
    model.set_classes(class_names, embeddings=embeddings)
    results = model.predict(
        source=str(image_path),
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=str(device) != "cpu",
        quantize=16 if str(device) != "cpu" else None,
        verbose=False,
        save=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"expected one YOLOE result, got {len(results)}")
    detections = result_detections(results[0], (height, width))

    detections_overlay = image_bgr.copy()
    colors = ((0, 255, 0), (255, 0, 255))
    for detection in detections:
        class_index = int(detection["class_index"])
        draw_bbox(
            detections_overlay,
            detection["bbox_xyxy"],
            color=colors[class_index % len(colors)],
            label=(
                f"{class_names[class_index]} "
                f"{float(detection['confidence']):.3f}"
            ),
            thickness=3,
        )
    draw_bbox(
        detections_overlay,
        target_reference,
        color=(255, 255, 0),
        label="VA REFERENCE",
        thickness=2,
    )
    write_image(output_dir / "02_yoloe_detections.jpg", detections_overlay)

    target_detection = select_detection(
        detections,
        class_index=0,
        reference_bbox=target_reference,
    )
    if target_detection is None:
        result = {
            "status": "NO_TARGET_DETECTION",
            "image": str(image_path),
            "target": args.target,
            "yaw_align_target": args.yaw_align_target,
            "class_names": class_names,
            "detections": [],
            "depth_filter_applied": False,
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    if same_prompt:
        yaw_align_target_detection = target_detection
        yaw_align_target_mode = "reuse_target_instance"
    else:
        yaw_align_target_detection = select_detection(
            detections,
            class_index=1,
            reference_bbox=yaw_align_target_reference,
        )
        yaw_align_target_mode = "independent_yaw_align_target_instance"
    if yaw_align_target_detection is None:
        result = {
            "status": "NO_YAW_ALIGN_TARGET_DETECTION",
            "image": str(image_path),
            "target": args.target,
            "yaw_align_target": args.yaw_align_target,
            "class_names": class_names,
            "detections": [
                serialize_detection(
                    item,
                    class_names=class_names,
                    reference_bbox=target_reference,
                )
                for item in detections
            ],
            "depth_filter_applied": False,
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    target_mask = np.asarray(target_detection["mask"], dtype=np.uint8)
    raw_yaw_align_target_mask = np.asarray(yaw_align_target_detection["mask"], dtype=np.uint8)
    filled_yaw_align_target_mask = _largest_filled_component(raw_yaw_align_target_mask)
    dilated_yaw_align_target_mask = _dilate_yaw_align_edge_mask(filled_yaw_align_target_mask)
    exclusion_mask = (
        None
        if same_prompt
        else _yaw_align_edge_target_exclusion(target_mask, dilated_yaw_align_target_mask.shape)
    )
    edge_intersection = _rgb_mask_edge_intersection(
        image_rgb,
        dilated_yaw_align_target_mask,
        exclusion_mask=exclusion_mask,
    )
    segments = _rgb_edge_line_segments(edge_intersection)
    selected_segment = segments[0] if segments else None

    write_image(
        output_dir / "03_target_mask.png",
        colorize_mask(target_mask, color=(255, 255, 255)),
    )
    write_image(
        output_dir / "04_yaw_align_target_mask_raw.png",
        colorize_mask(raw_yaw_align_target_mask, color=(255, 255, 255)),
    )
    write_image(
        output_dir / "05_yaw_align_target_mask_filled.png",
        colorize_mask(filled_yaw_align_target_mask, color=(255, 255, 255)),
    )
    write_image(
        output_dir / "06_yaw_align_target_mask_dilated.png",
        colorize_mask(dilated_yaw_align_target_mask, color=(255, 255, 255)),
    )
    if exclusion_mask is not None:
        write_image(
            output_dir / "07_target_exclusion_mask.png",
            colorize_mask(exclusion_mask, color=(255, 255, 255)),
        )
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150, apertureSize=3, L2gradient=True)
    write_image(output_dir / "08_canny_edges.png", canny)
    write_image(output_dir / "09_masked_rgb_edges.png", edge_intersection)

    candidates_overlay = image_bgr.copy()
    serialized_segments = []
    for index, segment in enumerate(segments):
        start = tuple(int(round(item)) for item in segment.endpoint_a)
        end = tuple(int(round(item)) for item in segment.endpoint_b)
        cv2.line(candidates_overlay, start, end, (0, 165, 255), 2, cv2.LINE_AA)
        serialized_segments.append(
            {
                "rank_by_length": index + 1,
                "endpoint_a": [float(item) for item in segment.endpoint_a],
                "endpoint_b": [float(item) for item in segment.endpoint_b],
                "length_px": float(segment.length),
            }
        )
    write_image(output_dir / "10_hough_candidates.jpg", candidates_overlay)

    selected_overlay = image_bgr.copy()
    selected_line = None
    if selected_segment is not None:
        endpoint_a = np.asarray(selected_segment.endpoint_a, dtype=np.float64)
        endpoint_b = np.asarray(selected_segment.endpoint_b, dtype=np.float64)
        if tuple(endpoint_b) < tuple(endpoint_a):
            endpoint_a, endpoint_b = endpoint_b, endpoint_a
        start = tuple(int(round(item)) for item in endpoint_a)
        end = tuple(int(round(item)) for item in endpoint_b)
        cv2.line(selected_overlay, start, end, (0, 0, 255), 7, cv2.LINE_AA)
        midpoint = tuple(
            int(round(item)) for item in 0.5 * (endpoint_a + endpoint_b)
        )
        cv2.putText(
            selected_overlay,
            "LONGEST NON-DEPTH EDGE",
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        yaw_error_rad = math.atan2(
            -float(endpoint_b[1] - endpoint_a[1]),
            float(endpoint_b[0] - endpoint_a[0]),
        )
        selected_line = {
            "endpoint_a": [float(item) for item in endpoint_a],
            "endpoint_b": [float(item) for item in endpoint_b],
            "length_px": float(selected_segment.length),
            "pixel_yaw_error_rad": yaw_error_rad,
            "pixel_yaw_error_deg": math.degrees(yaw_error_rad),
            "selection_rule": "longest_after_non_depth_filters",
        }
    draw_bbox(
        selected_overlay,
        target_detection["bbox_xyxy"],
        color=(0, 255, 0),
        label=f"YOLOE {args.target}",
        thickness=3,
    )
    write_image(output_dir / "11_selected_line.jpg", selected_overlay)

    result = {
        "status": "OK" if selected_line is not None else "NO_HOUGH_SEGMENT",
        "image": str(image_path),
        "image_size": {"width": width, "height": height},
        "model": str(model_path),
        "device": str(device),
        "imgsz": int(args.imgsz),
        "confidence_threshold": float(args.conf),
        "iou_threshold": float(args.iou),
        "va_result": {
            "target": {
                "text": args.target,
                "bbox_2d_normalized": list(map(float, args.target_bbox)),
                "bbox_xyxy_pixels": list(target_reference),
            },
            "yaw_align_target": {
                "text": args.yaw_align_target,
                "bbox_2d_normalized": None,
                "bbox_xyxy_pixels": None,
            },
        },
        "class_names": class_names,
        "same_target_yaw_align_target_prompt": same_prompt,
        "yaw_align_target_mode": yaw_align_target_mode,
        "target_exclusion_applied": exclusion_mask is not None,
        "depth_filter_applied": False,
        "selection_without_depth": "longest_hough_segment",
        "detections": [
            serialize_detection(
                item,
                class_names=class_names,
                reference_bbox=(
                    target_reference
                    if int(item["class_index"]) == 0
                    else yaw_align_target_reference
                ),
            )
            for item in detections
        ],
        "selected_target_detection": serialize_detection(
            target_detection,
            class_names=class_names,
            reference_bbox=target_reference,
        ),
        "selected_yaw_align_target_detection": serialize_detection(
            yaw_align_target_detection,
            class_names=class_names,
            reference_bbox=(target_reference if same_prompt else yaw_align_target_reference),
        ),
        "non_depth_filtering": {
            "largest_filled_yaw_align_target_component": True,
            "yaw_align_target_mask_dilation_radius_px": 3,
            "target_exclusion_radius_px": None if same_prompt else 20,
            "gaussian_kernel": 5,
            "canny_low": 50,
            "canny_high": 150,
            "hough_threshold": 25,
            "hough_min_length_px_strictly_greater_than": 75.0,
            "hough_max_line_gap_px": 12,
            "candidate_count": len(segments),
        },
        "hough_segments": serialized_segments,
        "selected_line": selected_line,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    try:
        result = run(parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
