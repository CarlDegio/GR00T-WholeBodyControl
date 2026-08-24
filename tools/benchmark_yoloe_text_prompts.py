#!/usr/bin/env python3
"""Benchmark alternative YOLOE text prompts on one VA-grounded image.

Each phrase is installed as a single YOLOE class and evaluated independently.
Target detections are matched to the VA target bbox by IoU. Surface detections
must geometrically cover the target's horizontal extent and the support band
immediately below it. This avoids ranking a confident detection elsewhere in
the image as a useful prompt replacement.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "tools/yoloe26m/weights/yoloe-26m-seg.pt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/yoloe_prompt_benchmarks"

TARGET_PROMPTS = (
    "black package",
    "package",
    "black parcel",
    "parcel",
    "black plastic package",
    "black plastic parcel",
    "black bag",
    "plastic bag",
    "black plastic bag",
    "black mailing bag",
    "black mailer bag",
    "black shipping bag",
    "black parcel bag",
    "poly mailer",
    "black poly mailer",
    "black poly mailer bag",
    "black pouch",
    "black sack",
    "black trash bag",
    "black garbage bag",
    "black wrapped package",
    "black wrapped parcel",
    "wrapped black package",
    "black plastic wrapped package",
    "black plastic-wrapped package",
    "black plastic-wrapped parcel",
    "black sealed package",
    "sealed black package",
    "black shipping package",
    "black soft package",
    "black packet",
    "black plastic packet",
    "black plastic pouch",
    "black parcel pouch",
    "black mailer package",
    "black bag package",
    "plastic wrapped package",
    "black plastic wrapped parcel",
    "black plastic shipping package",
    "black plastic soft package",
    "soft black plastic package",
)

SURFACE_PROMPTS = (
    "shelf",
    "metal shelf",
    "white shelf",
    "white metal shelf",
    "rack shelf",
    "storage shelf",
    "warehouse shelf",
    "industrial shelf",
    "horizontal shelf",
    "shelf board",
    "shelf platform",
    "metal shelf platform",
    "shelving",
    "metal shelving",
    "shelving unit",
    "metal shelving unit",
    "rack",
    "metal rack",
    "storage rack",
    "warehouse rack",
    "metal storage rack",
    "shelf of a metal rack",
    "middle shelf of a rack",
    "horizontal metal shelf",
    "storage rack shelf",
    "rack tier",
    "metal rack tier",
    "steel shelf",
    "steel shelf board",
    "steel shelf platform",
    "metal shelf board",
    "metal shelf panel",
    "shelf panel",
    "metal shelving platform",
    "metal storage shelf platform",
    "industrial metal shelf platform",
    "horizontal metal shelf platform",
    "metal rack shelf platform",
    "shelf surface",
    "metal shelf surface",
    "storage shelf surface",
    "rack shelf surface",
    "horizontal shelf surface",
    "metal platform",
    "steel platform",
    "horizontal metal platform",
    "industrial metal shelf",
    "warehouse metal shelf",
    "metal shelf level",
    "metal rack level",
    "shelving board",
    "table",
    "table top",
    "tabletop",
    "table surface",
    "metal table",
    "metal table top",
    "metal tabletop",
    "metal table surface",
    "workbench",
    "work table",
    "metal work table",
    "counter",
    "countertop",
    "counter surface",
    "bench",
    "metal bench",
    "tray",
    "metal tray",
    "shelf tray",
    "rack platform",
    "metal rack platform",
    "ledge",
    "metal ledge",
    "metal board",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--va-result", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
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


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def load_target_bbox(path: Path, width: int, height: int) -> list[float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    bbox = value["target"]["bbox_2d"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("VA target bbox is missing or invalid")
    normalized = [float(item) for item in bbox]
    if not all(math.isfinite(item) for item in normalized):
        raise ValueError("VA target bbox must be finite")
    return [
        normalized[0] * width / 1000.0,
        normalized[1] * height / 1000.0,
        normalized[2] * width / 1000.0,
        normalized[3] * height / 1000.0,
    ]


def result_detections(result: Any, image_shape: tuple[int, int]) -> list[dict[str, Any]]:
    from ultralytics.utils import ops

    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    masks = ops.scale_masks(
        result.masks.data[:, None],
        image_shape,
        padding=True,
        mode="nearest",
    )[:, 0]
    masks = masks.detach().cpu().numpy() > 0.5
    return [
        {
            "confidence": float(confidences[index]),
            "bbox_xyxy": [float(item) for item in boxes[index]],
            "mask": masks[index],
        }
        for index in range(len(boxes))
    ]


def surface_geometry(
    detection: dict[str, Any], target_bbox: Sequence[float]
) -> dict[str, float | bool]:
    x1, y1, x2, y2 = detection["bbox_xyxy"]
    tx1, ty1, tx2, ty2 = map(float, target_bbox)
    target_width = max(1.0, tx2 - tx1)
    horizontal_overlap = max(0.0, min(x2, tx2) - max(x1, tx1)) / target_width
    target_center_x = 0.5 * (tx1 + tx2)
    target_bottom = ty2
    covers_support_zone = bool(
        horizontal_overlap >= 0.60
        and x1 <= target_center_x <= x2
        and y1 <= target_bottom + 80.0
        and y2 >= target_bottom + 20.0
    )
    mask = detection["mask"]
    height, width = mask.shape
    band_x1 = max(0, int(round(tx1)))
    band_x2 = min(width, int(round(tx2)))
    band_y1 = max(0, int(round(target_bottom)))
    band_y2 = min(height, int(round(target_bottom + 100.0)))
    band = mask[band_y1:band_y2, band_x1:band_x2]
    band_coverage = float(np.mean(band)) if band.size else 0.0
    return {
        "horizontal_target_overlap": horizontal_overlap,
        "support_band_mask_coverage": band_coverage,
        "covers_support_zone": covers_support_zone,
    }


def serializable_detection(
    detection: dict[str, Any] | None,
    *,
    target_bbox: Sequence[float],
    surface: bool,
) -> dict[str, Any] | None:
    if detection is None:
        return None
    record: dict[str, Any] = {
        "confidence": detection["confidence"],
        "bbox_xyxy": detection["bbox_xyxy"],
        "mask_pixels": int(np.count_nonzero(detection["mask"])),
        "iou_with_va_target_bbox": bbox_iou(
            detection["bbox_xyxy"], target_bbox
        ),
    }
    if surface:
        record.update(surface_geometry(detection, target_bbox))
    return record


def choose_detection(
    detections: Sequence[dict[str, Any]],
    *,
    target_bbox: Sequence[float],
    surface: bool,
) -> dict[str, Any] | None:
    if not detections:
        return None
    if not surface:
        matched = max(
            detections,
            key=lambda item: (
                bbox_iou(item["bbox_xyxy"], target_bbox),
                item["confidence"],
            ),
        )
        return matched if bbox_iou(matched["bbox_xyxy"], target_bbox) >= 0.20 else None
    useful = [
        item
        for item in detections
        if surface_geometry(item, target_bbox)["covers_support_zone"]
    ]
    return max(useful, key=lambda item: item["confidence"], default=None)


def annotate(
    image: np.ndarray,
    *,
    prompt: str,
    detection: dict[str, Any] | None,
    target_bbox: Sequence[float],
    surface: bool,
) -> np.ndarray:
    output = image.copy()
    tx1, ty1, tx2, ty2 = (int(round(item)) for item in target_bbox)
    cv2.rectangle(output, (tx1, ty1), (tx2, ty2), (255, 0, 255), 2)
    if detection is not None:
        mask = detection["mask"]
        tint = np.zeros_like(output)
        tint[:, :] = (255, 100, 0) if surface else (0, 200, 0)
        output[mask] = cv2.addWeighted(output, 0.65, tint, 0.35, 0)[mask]
        x1, y1, x2, y2 = (
            int(round(item)) for item in detection["bbox_xyxy"]
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 3)
        score = detection["confidence"]
        label = f"{prompt}: {score:.4f}"
    else:
        label = f"{prompt}: NO MATCH"
    cv2.rectangle(output, (0, 0), (output.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def contact_sheet(paths: Sequence[Path], output: Path, columns: int = 3) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    images = [item for item in images if item is not None]
    if not images:
        return
    tile_width = 640
    tile_height = int(round(images[0].shape[0] * tile_width / images[0].shape[1]))
    tiles = [cv2.resize(item, (tile_width, tile_height)) for item in images]
    rows = math.ceil(len(tiles) / columns)
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = tile
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"failed to write contact sheet: {output}")


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    va_path = args.va_result.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not image_path.is_file() or not va_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("image, VA result, and model must exist")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode input image")
    height, width = image.shape[:2]
    target_bbox = load_target_bbox(va_path, width, height)
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = DEFAULT_OUTPUT_ROOT / stamp
    else:
        output_dir = args.output_dir.expanduser()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir()

    import torch
    from ultralytics import YOLOE

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(model_path.parents[1] / "config"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    model = YOLOE(str(model_path))
    model.to(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    prompts = list(TARGET_PROMPTS + SURFACE_PROMPTS)
    embeddings = model.get_text_pe(prompts)
    records: list[dict[str, Any]] = []
    annotated: dict[str, list[Path]] = {"target": [], "surface": []}
    for index, prompt in enumerate(prompts):
        kind = "target" if index < len(TARGET_PROMPTS) else "surface"
        model.set_classes([prompt], embeddings=embeddings[:, index : index + 1])
        model.predictor = None
        result = model.predict(
            source=str(image_path),
            device=args.device,
            imgsz=args.imgsz,
            conf=args.confidence,
            half=True,
            quantize=16,
            verbose=False,
            save=False,
        )[0]
        detections = result_detections(result, (height, width))
        chosen = choose_detection(
            detections,
            target_bbox=target_bbox,
            surface=kind == "surface",
        )
        record = {
            "kind": kind,
            "prompt": prompt,
            "matched_detection": serializable_detection(
                chosen,
                target_bbox=target_bbox,
                surface=kind == "surface",
            ),
            "raw_detection_count": len(detections),
            "raw_max_confidence": max(
                (item["confidence"] for item in detections), default=None
            ),
        }
        records.append(record)
        annotated_path = annotated_dir / f"{kind}_{index:02d}_{slugify(prompt)}.jpg"
        rendered = annotate(
            image,
            prompt=prompt,
            detection=chosen,
            target_bbox=target_bbox,
            surface=kind == "surface",
        )
        if not cv2.imwrite(str(annotated_path), rendered):
            raise OSError(f"failed to write {annotated_path}")
        annotated[kind].append(annotated_path)
        confidence = None if chosen is None else chosen["confidence"]
        print(f"{kind:7s} {prompt:32s} {confidence}")

    rankings: dict[str, list[dict[str, Any]]] = {}
    for kind in ("target", "surface"):
        ranked = sorted(
            (item for item in records if item["kind"] == kind),
            key=lambda item: (
                -1.0
                if item["matched_detection"] is None
                else item["matched_detection"]["confidence"]
            ),
            reverse=True,
        )
        rankings[kind] = ranked
        path_by_prompt = {
            path.stem.split("_", 2)[2]: path for path in annotated[kind]
        }
        ordered_paths = [path_by_prompt[slugify(item["prompt"])] for item in ranked]
        contact_sheet(ordered_paths, output_dir / f"{kind}_ranking_contact_sheet.jpg")

    summary = {
        "image": str(image_path),
        "va_result": str(va_path),
        "model": str(model_path),
        "device": args.device,
        "imgsz": args.imgsz,
        "detection_threshold": args.confidence,
        "va_target_bbox_pixels": target_bbox,
        "matching": {
            "target": "bbox IoU >= 0.20; rank matched detections by confidence",
            "surface": (
                "bbox overlaps >=60% of target width, contains target center x, "
                "and crosses the 20-80 px support zone below target"
            ),
        },
        "rankings": rankings,
    }
    (output_dir / "benchmark.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved benchmark to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
