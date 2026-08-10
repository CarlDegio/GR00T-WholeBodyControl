#!/usr/bin/env python3
"""Ground a target in a reference image with Codex/Qwen, then run YOLOE on a second image."""

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
    YoloVisualPromptConfig,
    create_grounding_client,
    prepare_numbered_run_dir,
    run_auto_reference_pipeline,
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
        "--refer-image", type=Path, required=True, help="First image sent to the grounding model."
    )
    parser.add_argument(
        "--target", required=True, help="Exact object the first-layer model must box, e.g. 'blue basket'."
    )
    parser.add_argument("--backend", choices=("codex", "qwenvl"), required=True)
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
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--codex-model", default="gpt-5.6-sol")
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        default="xhigh",
    )
    parser.add_argument(
        "--codex-fast", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--qwen-model", default=None, help="Defaults to the current BasePose Qwen-VL Plus model."
    )
    parser.add_argument(
        "--qwen-base-url", default=None, help="Defaults to the current BasePose DashScope URL."
    )
    parser.add_argument("--qwen-thinking-budget", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = args.target.strip()
    if not target:
        raise ValueError("--target must be non-empty")
    run_dir = prepare_numbered_run_dir(args.project, target)
    use_half = args.device != "cpu" if args.half is None else args.half
    config = YoloVisualPromptConfig(
        source_image=args.source_image,
        reference_image=args.refer_image,
        target_name=target,
        model_path=args.model,
        run_dir=run_dir,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=use_half,
    )
    client = create_grounding_client(
        args.backend,
        codex_model=args.codex_model,
        codex_reasoning_effort=args.codex_reasoning_effort,
        codex_fast=args.codex_fast,
        timeout_seconds=args.timeout_seconds,
        qwen_model=args.qwen_model,
        qwen_base_url=args.qwen_base_url,
        qwen_thinking_budget=args.qwen_thinking_budget,
    )
    summary = run_auto_reference_pipeline(
        config,
        backend=args.backend,
        client=client,
    )
    print(f"First layer: {args.backend}; target: {target}")
    print(f"Detections: {summary.detection_count}; masks: {summary.mask_count}")
    print(f"Saved results: {summary.run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
