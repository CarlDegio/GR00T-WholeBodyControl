#!/usr/bin/env python3
"""Run YOLOE-26M text-prompted instance segmentation on images, videos, or a camera."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(SCRIPT_DIR / "config"))

import torch  # noqa: E402
from ultralytics import YOLOE  # noqa: E402

DEFAULT_MODEL = SCRIPT_DIR / "weights" / "yoloe-26m-seg.pt"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Image/video path, URL, directory, glob, stream URL, or camera index.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["person", "bus"],
        help="English text prompts. Example: --classes person cup 'cardboard box'",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--project", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", default="predict")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means no limit.")
    parser.add_argument("--show", action="store_true", help="Open an OpenCV preview window.")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use FP16. Defaults to enabled on CUDA and disabled on CPU.",
    )
    return parser.parse_args()


def normalize_source(source: str) -> Any:
    """Treat a bare non-negative integer as a camera index."""
    return int(source) if source.isdigit() else source


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Run setup.sh first.")

    use_half = args.device != "cpu" if args.half is None else args.half
    model = YOLOE(str(model_path))
    model.set_classes(args.classes)

    stream = model.predict(
        source=normalize_source(args.source),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        quantize=16 if use_half else None,
        save=True,
        project=str(args.project.expanduser().resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        show=args.show,
        stream=True,
        verbose=True,
    )

    frame_count = 0
    detection_count = 0
    class_counts: Counter[str] = Counter()
    save_dir: Path | None = None
    speed_totals: Counter[str] = Counter()

    try:
        for result in stream:
            frame_count += 1
            save_dir = Path(result.save_dir)
            if result.boxes is not None:
                detection_count += len(result.boxes)
                for class_id in result.boxes.cls.int().tolist():
                    class_counts[result.names[class_id]] += 1
            speed_totals.update(result.speed)
            if args.max_frames and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"Frames: {frame_count}")
    print(f"Detections: {detection_count} {dict(class_counts)}")
    if frame_count:
        mean_speed = {key: round(value / frame_count, 2) for key, value in speed_totals.items()}
        print(f"Mean speed (ms/image): {mean_speed}")
    if save_dir is not None:
        print(f"Saved results: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
