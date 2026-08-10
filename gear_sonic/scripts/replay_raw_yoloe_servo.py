"""Reconstruct sampled raw-YOLOE diagnostics without running robot services."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np


def _atomic_write_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _resolve_artifact(run_dir: Path, relative: str, description: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"{description} must be relative to run directory")
    candidate = (run_dir / relative).resolve()
    try:
        candidate.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"{description} escapes run directory") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"missing {description}: {candidate}")
    return candidate


def _load_mask(
    path: Path, image_shape: tuple[int, int], description: str
) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise OSError(f"failed to decode {description}: {path}")
    if mask.shape != image_shape:
        raise ValueError(
            f"{description} mask shape {mask.shape} does not match RGB {image_shape}"
        )
    return mask > 0


def _blend(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    image[mask] = (
        image[mask].astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha
    ).astype(np.uint8)


def _draw_detection(
    image: np.ndarray,
    value: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
) -> None:
    bbox = value.get("bbox_xyxy")
    if bbox is None:
        return
    x1, y1, x2, y2 = (int(round(float(item))) for item in bbox)
    confidence = value.get("confidence")
    suffix = "none" if confidence is None else f"{float(confidence):.3f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        f"{label} {suffix}",
        (max(0, x1), max(14, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def render_review_samples(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    alpha: float = 0.42,
) -> list[Path]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    root = Path(run_dir).resolve()
    jsonl = root / "raw_servo_frames.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"missing frame telemetry: {jsonl}")
    records = [
        json.loads(line)
        for line in jsonl.read_text().splitlines()
        if line.strip()
    ]
    sampled = [
        row
        for row in records
        if row.get("review_artifacts", {}).get("sampled") is True
    ]
    if not sampled:
        return []
    destination = (
        (root / "review_reconstructed").resolve()
        if output_dir is None
        else Path(output_dir).resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for row in sampled:
        review = row["review_artifacts"]
        raw_value = review.get("raw_rgb")
        if not isinstance(raw_value, str):
            raise ValueError("sampled frame has no review raw RGB path")
        raw_path = _resolve_artifact(root, raw_value, "review raw RGB")
        image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"failed to decode review raw RGB: {raw_path}")
        for key, color, description in (
            ("table_mask", (190, 70, 30), "table"),
            ("target_mask", (40, 190, 40), "target"),
        ):
            relative = review.get(key)
            if relative is None:
                continue
            if not isinstance(relative, str):
                raise ValueError(f"{description} mask path must be a string or null")
            mask_path = _resolve_artifact(root, relative, f"review {description} mask")
            mask = _load_mask(mask_path, image.shape[:2], description)
            _blend(image, mask, color, alpha)
        detections = row.get("detections", {})
        _draw_detection(
            image, detections.get("table", {}), (255, 130, 50), "table"
        )
        _draw_detection(
            image, detections.get("target", {}), (30, 255, 30), "target"
        )
        controller = row.get("controller", {})
        command = row.get("command", {})
        lines = (
            f"frame={int(row['frame_index'])} phase={controller.get('phase')}",
            f"errors={controller.get('filtered_errors')}",
            "cmd vx/vy/wz="
            f"{float(command.get('vx', 0.0)):+.3f}/"
            f"{float(command.get('vy', 0.0)):+.3f}/"
            f"{float(command.get('wz', 0.0)):+.3f}",
        )
        for index, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (6, 18 + index * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        path = destination / f"{int(row['frame_index']):06d}.jpg"
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not ok:
            raise OSError(
                f"failed to encode reconstructed frame {row['frame_index']}"
            )
        _atomic_write_bytes(path, encoded.tobytes())
        rendered.append(path)
    return rendered


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Reconstruct saved YOLOE bbox and mask overlays.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--alpha", type=float, default=0.42)
    args = parser.parse_args(argv)
    rendered = render_review_samples(
        args.run_dir,
        args.output_dir,
        alpha=args.alpha,
    )
    destination = (
        args.run_dir / "review_reconstructed"
        if args.output_dir is None
        else args.output_dir
    ).resolve()
    print(f"rendered {len(rendered)} review frames to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
