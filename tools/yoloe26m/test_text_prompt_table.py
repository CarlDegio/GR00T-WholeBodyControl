#!/usr/bin/env python3
"""Evaluate YOLOE table detection with text prompts on collected RGB frames.

When --source is omitted, the newest BasePose ``review_samples/raw`` directory
under ``outputs/base_pose_adjustment`` is used. This makes the default command
exercise recently collected robot-camera data rather than a downloaded demo.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL = SCRIPT_DIR / "weights" / "yoloe-26m-seg.pt"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PromptMetrics:
    prompt: str
    images_tested: int
    frames_with_detection: int
    frame_detection_rate: float
    total_detections: int
    mean_confidence: float | None
    median_confidence: float | None
    max_confidence: float | None
    mean_inference_ms: float | None
    throughput_fps: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLOE open-vocabulary table detection on current collected frames "
            "and compare text prompts."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        help=(
            "Image or directory of images. Defaults to the newest "
            "outputs/base_pose_adjustment/*/review_samples/raw directory."
        ),
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["table", "desk", "wooden table", "wooden desk"],
        help="Text prompts evaluated independently (default: %(default)s).",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        help="Run output directory. Defaults to a new timestamped directory.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=30,
        help="Uniformly sample at most this many images; 0 uses every image.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument(
        "--device",
        help="Ultralytics device such as 0 or cpu; auto-selects CUDA when available.",
    )
    parser.add_argument(
        "--no-half", action="store_true", help="Disable FP16 on CUDA."
    )
    parser.add_argument(
        "--contact-sheet-size",
        type=int,
        default=16,
        help="Maximum annotated frames in each prompt contact sheet; 0 disables it.",
    )
    return parser.parse_args()


def _prompt_slug(prompt: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip()).strip("_").lower()
    return slug or "prompt"


def _new_output_dir(requested: Path | None) -> Path:
    if requested is not None:
        path = requested.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=False)
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = DEFAULT_OUTPUT_ROOT / f"text_prompt_table_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def discover_latest_base_pose_frames(repo_root: Path = REPO_ROOT) -> Path:
    candidates = []
    pattern = "outputs/base_pose_adjustment/*/review_samples/raw"
    for directory in repo_root.glob(pattern):
        if directory.is_dir() and any(
            item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            for item in directory.iterdir()
        ):
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(
            "no current BasePose RGB samples found; pass --source IMAGE_OR_DIRECTORY"
        )
    return max(candidates, key=lambda item: item.stat().st_mtime).resolve()


def collect_images(source: Path) -> list[Path]:
    path = source.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"source is not a supported image: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"source does not exist: {path}")
    images = sorted(
        item.resolve()
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"source directory contains no supported images: {path}")
    return images


def uniformly_sample(items: Sequence[Path], limit: int) -> list[Path]:
    if limit < 0:
        raise ValueError("--max-images must be non-negative")
    if limit == 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indices]


def _mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_prompt(prompt: str, frames: Sequence[dict[str, Any]]) -> PromptMetrics:
    confidences = [
        float(detection["confidence"])
        for frame in frames
        for detection in frame["detections"]
    ]
    inference_times = [
        float(frame["speed_ms"]["inference"])
        for frame in frames
        if frame["speed_ms"].get("inference") is not None
    ]
    frames_with_detection = sum(bool(frame["detections"]) for frame in frames)
    mean_inference_ms = _mean_or_none(inference_times)
    return PromptMetrics(
        prompt=prompt,
        images_tested=len(frames),
        frames_with_detection=frames_with_detection,
        frame_detection_rate=frames_with_detection / len(frames) if frames else 0.0,
        total_detections=len(confidences),
        mean_confidence=_mean_or_none(confidences),
        median_confidence=statistics.median(confidences) if confidences else None,
        max_confidence=max(confidences) if confidences else None,
        mean_inference_ms=mean_inference_ms,
        throughput_fps=(1000.0 / mean_inference_ms) if mean_inference_ms else None,
    )


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _result_record(result: Any, image_path: Path, annotated_path: Path) -> dict[str, Any]:
    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().tolist() if boxes is not None else []
    confidences = boxes.conf.detach().cpu().tolist() if boxes is not None else []
    classes = boxes.cls.detach().cpu().tolist() if boxes is not None else []
    height, width = result.orig_shape
    image_area = float(height * width)
    detections = []
    for coordinates, confidence, class_index in zip(xyxy, confidences, classes):
        x1, y1, x2, y2 = map(float, coordinates)
        class_id = int(class_index)
        detections.append(
            {
                "class_id": class_id,
                "class_name": str(result.names[class_id]),
                "confidence": float(confidence),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_area_ratio": max(0.0, (x2 - x1) * (y2 - y1)) / image_area,
            }
        )
    speed = getattr(result, "speed", {}) or {}
    return {
        "image": str(image_path),
        "annotated_image": str(annotated_path),
        "image_size": {"width": width, "height": height},
        "detections": detections,
        "mask_count": 0 if result.masks is None else len(result.masks.data),
        "speed_ms": {
            "preprocess": _finite_float(speed.get("preprocess")),
            "inference": _finite_float(speed.get("inference")),
            "postprocess": _finite_float(speed.get("postprocess")),
        },
    }


def _selected_evenly(items: Sequence[Path], limit: int) -> list[Path]:
    if limit <= 0:
        return []
    return uniformly_sample(items, min(limit, len(items)))


def write_contact_sheet(paths: Sequence[Path], output_path: Path) -> None:
    import cv2
    import numpy as np

    selected = list(paths)
    if not selected:
        return
    columns = min(4, len(selected))
    rows = math.ceil(len(selected) / columns)
    tile_width, tile_height, label_height = 320, 240, 28
    sheet = np.full(
        (rows * (tile_height + label_height), columns * tile_width, 3),
        245,
        dtype=np.uint8,
    )
    for index, path in enumerate(selected):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x0 = column * tile_width
        y0 = row * (tile_height + label_height)
        sheet[y0 : y0 + tile_height, x0 : x0 + tile_width] = image
        cv2.putText(
            sheet,
            path.stem,
            (x0 + 6, y0 + tile_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output_path), sheet):
        raise OSError(f"failed to write contact sheet: {output_path}")


def _format_optional(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_markdown_report(
    output_path: Path,
    *,
    source: Path,
    model_path: Path,
    conf: float,
    metrics: Sequence[PromptMetrics],
) -> None:
    ranked = sorted(
        metrics,
        key=lambda item: (
            item.frame_detection_rate,
            item.mean_confidence if item.mean_confidence is not None else -1.0,
        ),
        reverse=True,
    )
    lines = [
        "# YOLOE text-prompt table evaluation",
        "",
        f"- Source: `{source}`",
        f"- Model: `{model_path}`",
        f"- Confidence threshold: `{conf}`",
        "",
        "| Prompt | Images | Frames detected | Detection rate | Detections | "
        "Mean confidence | Mean inference (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        lines.append(
            f"| `{item.prompt}` | {item.images_tested} | "
            f"{item.frames_with_detection} | {item.frame_detection_rate:.1%} | "
            f"{item.total_detections} | {_format_optional(item.mean_confidence)} | "
            f"{_format_optional(item.mean_inference_ms, 1)} |"
        )
    if ranked:
        lines.extend(
            [
                "",
                f"Best proxy result: `{ranked[0].prompt}` with "
                f"{ranked[0].frame_detection_rate:.1%} frame detection rate.",
            ]
        )
    lines.extend(
        [
            "",
            "> This dataset has no table ground-truth annotations. Detection rate and "
            "confidence are screening metrics, not precision, recall, or mAP. Review "
            "the contact sheets for false positives and mask quality.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _validate_args(args: argparse.Namespace) -> list[str]:
    prompts = [prompt.strip() for prompt in args.prompts if prompt.strip()]
    if not prompts:
        raise ValueError("at least one non-empty --prompts value is required")
    if not 0.0 < args.conf <= 1.0:
        raise ValueError("--conf must be in (0, 1]")
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou must be in (0, 1]")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if args.max_images < 0:
        raise ValueError("--max-images must be non-negative")
    if args.contact_sheet_size < 0:
        raise ValueError("--contact-sheet-size must be non-negative")
    return prompts


def run(args: argparse.Namespace) -> Path:
    prompts = _validate_args(args)
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLOE model not found: {model_path}; run tools/yoloe26m/setup.sh"
        )
    source = (
        args.source.expanduser().resolve()
        if args.source is not None
        else discover_latest_base_pose_frames()
    )
    all_images = collect_images(source)
    images = uniformly_sample(all_images, args.max_images)
    output_dir = _new_output_dir(args.output)

    import cv2
    import torch
    from ultralytics import YOLOE

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    use_half = not args.no_half and device != "cpu"
    print(f"Source: {source}")
    print(f"Images: {len(images)} selected from {len(all_images)}")
    print(f"Device: {device}; FP16: {use_half}")
    print(f"Output: {output_dir}")

    model = YOLOE(str(model_path))
    all_metrics: list[PromptMetrics] = []
    for prompt_index, prompt in enumerate(prompts, start=1):
        prompt_dir = output_dir / f"prompt_{prompt_index:02d}_{_prompt_slug(prompt)}"
        annotated_dir = prompt_dir / "annotated"
        annotated_dir.mkdir(parents=True)
        embeddings = model.get_text_pe([prompt])
        model.set_classes([prompt], embeddings=embeddings)
        results: Iterable[Any] = model.predict(
            source=[str(path) for path in images],
            stream=True,
            device=device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            half=use_half,
            verbose=False,
        )
        frame_records: list[dict[str, Any]] = []
        annotated_paths: list[Path] = []
        for image_index, (image_path, result) in enumerate(zip(images, results)):
            annotated_path = annotated_dir / (
                f"{image_index:04d}_{image_path.stem}_annotated.jpg"
            )
            if not cv2.imwrite(str(annotated_path), result.plot()):
                raise OSError(f"failed to write annotated image: {annotated_path}")
            annotated_paths.append(annotated_path)
            frame_records.append(_result_record(result, image_path, annotated_path))
        if len(frame_records) != len(images):
            raise RuntimeError(
                f"YOLOE returned {len(frame_records)} results for {len(images)} images"
            )
        metrics = summarize_prompt(prompt, frame_records)
        all_metrics.append(metrics)
        prompt_run = {"metrics": asdict(metrics), "frames": frame_records}
        (prompt_dir / "detections.json").write_text(
            json.dumps(prompt_run, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        sheet_paths = _selected_evenly(annotated_paths, args.contact_sheet_size)
        if sheet_paths:
            write_contact_sheet(sheet_paths, prompt_dir / "contact_sheet.jpg")
        print(
            f"[{prompt}] {metrics.frames_with_detection}/{metrics.images_tested} "
            f"frames ({metrics.frame_detection_rate:.1%}), "
            f"{metrics.total_detections} detections, "
            f"mean conf={_format_optional(metrics.mean_confidence)}"
        )

    summary = {
        "source": str(source),
        "model": str(model_path),
        "device": str(device),
        "half": use_half,
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "available_images": len(all_images),
        "sampled_images": [str(path) for path in images],
        "metrics": [asdict(item) for item in all_metrics],
        "ground_truth_available": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_report(
        output_dir / "report.md",
        source=source,
        model_path=model_path,
        conf=args.conf,
        metrics=all_metrics,
    )
    return output_dir


def main() -> int:
    try:
        output_dir = run(parse_args())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
