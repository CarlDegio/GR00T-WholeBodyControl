#!/usr/bin/env python3
"""Benchmark warm YOLOE-26M end-to-end latency with fixed text prompts."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(SCRIPT_DIR / "config"))

import cv2  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLOE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SCRIPT_DIR / "assets" / "bus.jpg")
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR / "weights" / "yoloe-26m-seg.pt")
    parser.add_argument("--classes", nargs="+", default=["person", "bus"])
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.source.expanduser().resolve()))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.source}")

    model = YOLOE(str(args.model.expanduser().resolve()))
    model.set_classes(args.classes)
    predict_args = {
        "device": args.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "quantize": 16 if args.device != "cpu" else None,
        "save": False,
        "verbose": False,
    }

    for _ in range(args.warmup):
        model.predict(image, **predict_args)
    synchronize(args.device)

    if args.device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    wall_ms: list[float] = []
    inference_ms: list[float] = []
    internal_ms: list[float] = []
    last_result = None
    for _ in range(args.runs):
        synchronize(args.device)
        started = time.perf_counter()
        last_result = model.predict(image, **predict_args)[0]
        synchronize(args.device)
        wall_ms.append((time.perf_counter() - started) * 1000)
        inference_ms.append(last_result.speed["inference"])
        internal_ms.append(sum(last_result.speed.values()))

    p95_index = max(0, int(len(wall_ms) * 0.95) - 1)
    sorted_wall = sorted(wall_ms)
    mean_wall = statistics.mean(wall_ms)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {args.device} ({gpu_name})")
    print(f"PyTorch: {torch.__version__}; CUDA wheel: {torch.version.cuda}; FP16: {args.device != 'cpu'}")
    print(f"Input: {args.imgsz}px; prompts: {args.classes}; warmup/runs: {args.warmup}/{args.runs}")
    print(
        f"Wall latency: mean={mean_wall:.2f} ms, median={statistics.median(wall_ms):.2f} ms, "
        f"p95={sorted_wall[p95_index]:.2f} ms, throughput={1000 / mean_wall:.1f} FPS"
    )
    print(
        f"Ultralytics timing: inference={statistics.mean(inference_ms):.2f} ms, "
        f"preprocess+inference+postprocess={statistics.mean(internal_ms):.2f} ms"
    )
    if last_result is not None:
        masks = len(last_result.masks) if last_result.masks is not None else 0
        boxes = len(last_result.boxes) if last_result.boxes is not None else 0
        print(f"Last result: boxes={boxes}, masks={masks}")
    if args.device != "cpu" and torch.cuda.is_available():
        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak allocated CUDA memory after warmup: {peak_mib:.0f} MiB")


if __name__ == "__main__":
    main()
