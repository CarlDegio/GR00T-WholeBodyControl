#!/usr/bin/env python3
"""Reproduce chest-camera YOLOE reacquisition on saved raw-servo frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from gear_sonic.utils.inference.base_pose_visual_servo import (
    YoloePersistentTracker,
)


DEFAULT_RUN = Path(
    "outputs/base_pose_adjustment/dual_raw_yoloe_20260817_135206_g1"
)
DEFAULT_FRAMES = (0, 60, 90, 185, 245)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"failed to decode {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _normalized_bbox(
    bbox: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return (
        1000.0 * x1 / width,
        1000.0 * y1 / height,
        1000.0 * x2 / width,
        1000.0 * y2 / height,
    )


def _raw_detections(result: Any) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().int().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    ids = None if boxes.id is None else boxes.id.detach().cpu().int().tolist()
    rows: list[dict[str, Any]] = []
    for index, (bbox, class_id, confidence) in enumerate(
        zip(xyxy, classes, confidences)
    ):
        rows.append(
            {
                "bbox_xyxy": [float(value) for value in bbox],
                "class_id": int(class_id),
                "class_name": str(result.names[int(class_id)]),
                "confidence": float(confidence),
                "track_id": None if ids is None else int(ids[index]),
            }
        )
    return rows


def _save_render(result: Any, path: Path) -> None:
    rendered = result.plot()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), rendered):
        raise OSError(f"failed to save {path}")


def _predict(
    tracker: YoloePersistentTracker,
    rgb: np.ndarray,
    *,
    confidence: float,
) -> Any:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return tracker.model.predict(
        bgr,
        device=tracker.device,
        imgsz=tracker.imgsz,
        conf=confidence,
        half=True,
        quantize=16,
        verbose=False,
        save=False,
    )[0]


def _track(tracker: YoloePersistentTracker, rgb: np.ndarray) -> Any:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return tracker.model.track(
        bgr,
        persist=True,
        tracker="botsort.yaml",
        device=tracker.device,
        imgsz=tracker.imgsz,
        conf=tracker.confidence,
        half=True,
        quantize=16,
        verbose=False,
        save=False,
    )[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path("outputs/yoloe_chest_reacquisition_20260817_135206_g1").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = "blue plastic basket on the desk"

    records = {
        int(row["frame_index"]): row
        for row in (
            json.loads(line)
            for line in (run_dir / "raw_servo_frames.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    images: dict[int, np.ndarray] = {}
    for frame_index in args.frames:
        path = (
            run_dir
            / "review_samples"
            / "raw"
            / f"{int(frame_index):06d}.png"
        )
        images[int(frame_index)] = _load_rgb(path)

    summary = json.loads((run_dir / "initial_reference_summary.json").read_text())
    initial = summary["streams"]["chest_view"]["reference"]
    initial_rgb = _load_rgb(run_dir / "chest_view" / "initial_rgb.png")
    initial_target = tuple(float(value) for value in initial["target_bbox"])
    initial_tables = tuple(
        tuple(float(value) for value in bbox) for bbox in initial["table_bboxes"]
    )

    latest_row = records[50]
    latest_rgb = images[0] if 50 == 0 else _load_rgb(
        run_dir / "review_samples" / "raw" / "000050.png"
    )
    height, width = latest_rgb.shape[:2]
    latest_target = _normalized_bbox(
        latest_row["detections"]["target"]["bbox_xyxy"], width, height
    )
    latest_tables = (
        _normalized_bbox(
            latest_row["detections"]["table"]["bbox_xyxy"], width, height
        ),
    )

    tracker = YoloePersistentTracker(
        Path("tools/yoloe26m/weights/yoloe-26m-seg.pt"),
        confidence=0.25,
        imgsz=640,
        device="0",
    )

    def text_two() -> None:
        tracker.class_names = (prompt, "desk")
        embeddings = tracker.model.get_text_pe(list(tracker.class_names))
        tracker.model.set_classes(
            list(tracker.class_names), embeddings=embeddings
        )
        tracker.model.predictor = None

    def text_target_only() -> None:
        tracker.class_names = (prompt,)
        tracker._initial_surface_embedding = None
        tracker.model.set_classes([prompt])
        tracker.model.predictor = None

    def visual_two(
        rgb: np.ndarray,
        target: tuple[float, float, float, float],
        tables: tuple[tuple[float, float, float, float], ...],
    ) -> None:
        tracker.start(
            rgb,
            target_prompt=prompt,
            target_bbox=target,
            surface_prompt="desk",
            surface_bboxes=tables,
        )

    def initial_visual_two() -> None:
        visual_two(initial_rgb, initial_target, initial_tables)

    def latest_visual_two() -> None:
        visual_two(latest_rgb, latest_target, latest_tables)

    def latest_visual_target_only() -> None:
        latest_visual_two()
        embeddings = tracker.model.model.pe[:, :1].detach().clone()
        tracker.class_names = (prompt,)
        tracker.model.set_classes([prompt], embeddings=embeddings)
        tracker.model.predictor = None

    installers: dict[str, Callable[[], None]] = {
        "text_two_class": text_two,
        "text_target_only": text_target_only,
        "initial_visual_two_class": initial_visual_two,
        "latest_visual_two_class": latest_visual_two,
        "latest_visual_target_only": latest_visual_target_only,
    }

    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "prompt": prompt,
        "frames": list(images),
        "latest_reference_frame": 50,
        "modes": {},
    }
    for mode, install in installers.items():
        mode_report: dict[str, Any] = {"predict": {}, "track": {}}
        install()
        for frame_index, rgb in images.items():
            exact = _predict(tracker, rgb, confidence=0.25)
            low = _predict(tracker, rgb, confidence=0.01)
            mode_report["predict"][str(frame_index)] = {
                "conf_0.25": _raw_detections(exact),
                "conf_0.01": _raw_detections(low),
            }
            _save_render(
                exact,
                output_dir / mode / f"{frame_index:06d}_predict_conf025.jpg",
            )

        install()
        for frame_index, rgb in images.items():
            tracked = _track(tracker, rgb)
            mode_report["track"][str(frame_index)] = _raw_detections(tracked)
            _save_render(
                tracked,
                output_dir / mode / f"{frame_index:06d}_track.jpg",
            )
        report["modes"][mode] = mode_report
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"completed {mode}", flush=True)

    print(output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
