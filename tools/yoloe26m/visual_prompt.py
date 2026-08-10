#!/usr/bin/env python3
"""Run YOLOE-26M on one image using example boxes from a reference image."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(SCRIPT_DIR / "config"))

from tools.yoloe26m.reference_pipeline import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    PixelBox,
    YoloVisualPromptConfig,
    prepare_numbered_run_dir,
    run_yoloe_visual_prompt,
)


def _default_device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source_image", type=Path, help="Second image to detect.")
    parser.add_argument(
        "--refer-image", type=Path, required=True, help="Reference image containing the examples."
    )
    parser.add_argument(
        "--target-name", required=True, help="Display name for the visual category, e.g. 'blue basket'."
    )
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        action="append",
        required=True,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Example box in original reference-image pixels; repeat for every example.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--project", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use FP16; defaults to on for CUDA and off for CPU.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.target_name.strip():
        raise ValueError("--target-name must be non-empty")
    boxes = [PixelBox(*parts, confidence=1.0) for parts in args.bbox]
    run_dir = prepare_numbered_run_dir(args.project, args.target_name)
    use_half = args.device != "cpu" if args.half is None else args.half
    summary = run_yoloe_visual_prompt(
        YoloVisualPromptConfig(
            source_image=args.source_image,
            reference_image=args.refer_image,
            target_name=args.target_name,
            model_path=args.model,
            run_dir=run_dir,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            half=use_half,
        ),
        boxes,
    )
    print(f"Detections: {summary.detection_count}; masks: {summary.mask_count}")
    print(f"Saved results: {summary.run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
