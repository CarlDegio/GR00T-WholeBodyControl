#!/usr/bin/env python3
"""Visualize RGB-edge and completed-mask intersection on local review frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from gear_sonic.utils.inference.base_pose_visual_servo import (
    _largest_filled_component,
)


DEFAULT_REVIEW_DIR = Path(
    "outputs/base_pose_adjustment/dual_raw_yoloe_20260819_175053_g1/"
    "review_samples"
)
DEFAULT_OUTPUT_DIR = DEFAULT_REVIEW_DIR / "table_edge_rgb_mask_intersection_mock_40"

CANNY_LOW = 50
CANNY_HIGH = 150
GAUSSIAN_KERNEL = 5
TARGET_EXCLUSION_RADIUS_PX = 20
HOUGH_THRESHOLD = 25
HOUGH_MIN_LINE_LENGTH_PX = 45
HOUGH_MAX_LINE_GAP_PX = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--overview-size", type=int, default=8)
    return parser.parse_args()


def evenly_spaced_paths(paths: list[Path], count: int) -> list[Path]:
    if count < 1:
        raise ValueError("--count must be positive")
    if count >= len(paths):
        return paths
    indices = np.rint(np.linspace(0, len(paths) - 1, count)).astype(int)
    return [paths[index] for index in indices]


def effective_table_mask(
    table_mask: np.ndarray,
    target_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    completed = _largest_filled_component(table_mask)
    if target_mask is None:
        return completed, None
    target = (np.asarray(target_mask) > 0).astype(np.uint8)
    if target.shape != completed.shape:
        raise ValueError("target mask is not aligned with table mask")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * TARGET_EXCLUSION_RADIUS_PX + 1,
            2 * TARGET_EXCLUSION_RADIUS_PX + 1,
        ),
    )
    exclusion = cv2.dilate(target, kernel, iterations=1)
    effective = completed.copy()
    effective[exclusion > 0] = 0
    return effective, exclusion


def rgb_mask_edge_lines(
    rgb: np.ndarray,
    effective_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int, float]]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(
        gray,
        (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL),
        0,
    )
    edges = cv2.Canny(
        blurred,
        CANNY_LOW,
        CANNY_HIGH,
        apertureSize=3,
        L2gradient=True,
    )
    intersection = np.zeros_like(edges)
    intersection[effective_mask > 0] = edges[effective_mask > 0]
    detected = cv2.HoughLinesP(
        intersection,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH_PX,
        maxLineGap=HOUGH_MAX_LINE_GAP_PX,
    )
    lines: list[tuple[int, int, int, int, float]] = []
    if detected is not None:
        for x1, y1, x2, y2 in detected[:, 0, :]:
            length = math.hypot(float(x2 - x1), float(y2 - y1))
            if length > HOUGH_MIN_LINE_LENGTH_PX:
                lines.append(
                    (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        length,
                    )
                )
    lines.sort(key=lambda line: line[4], reverse=True)
    return edges, intersection, lines


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 28), (15, 15, 15), -1)
    cv2.putText(
        result,
        text,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return result


def visualize(
    rgb: np.ndarray,
    completed_mask: np.ndarray,
    effective_mask: np.ndarray,
    exclusion: np.ndarray | None,
    edges: np.ndarray,
    intersection: np.ndarray,
    lines: list[tuple[int, int, int, int, float]],
    *,
    title: str,
) -> np.ndarray:
    input_panel = rgb.copy()
    blue = np.zeros_like(input_panel)
    blue[:, :] = (220, 130, 30)
    table_pixels = completed_mask > 0
    input_panel[table_pixels] = cv2.addWeighted(
        input_panel[table_pixels],
        0.58,
        blue[table_pixels],
        0.42,
        0.0,
    )
    if exclusion is not None:
        red = np.zeros_like(input_panel)
        red[:, :] = (40, 40, 230)
        excluded_pixels = (exclusion > 0) & table_pixels
        input_panel[excluded_pixels] = cv2.addWeighted(
            input_panel[excluded_pixels],
            0.45,
            red[excluded_pixels],
            0.55,
            0.0,
        )
    input_panel = add_header(input_panel, f"RGB + effective mask | {title}")

    canny_panel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    canny_panel = add_header(
        canny_panel,
        f"Canny RGB edges | thresholds {CANNY_LOW}/{CANNY_HIGH}",
    )

    intersection_panel = (rgb.astype(np.float32) * 0.22).astype(np.uint8)
    intersection_panel[intersection > 0] = (30, 240, 80)
    boundary = completed_mask - cv2.erode(
        completed_mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    intersection_panel[boundary > 0] = (160, 160, 160)
    intersection_panel[intersection > 0] = (30, 240, 80)
    intersection_panel = add_header(
        intersection_panel,
        f"edge AND mask | {np.count_nonzero(intersection)} pixels",
    )

    line_panel = (rgb.astype(np.float32) * 0.48).astype(np.uint8)
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (155, 89, 182),
        (230, 126, 34),
        (26, 188, 156),
        (231, 76, 60),
    ]
    for index, (x1, y1, x2, y2, _length) in enumerate(lines):
        color = palette[index % len(palette)]
        cv2.line(line_panel, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        cv2.circle(line_panel, (x1, y1), 3, color, -1, cv2.LINE_AA)
        cv2.circle(line_panel, (x2, y2), 3, color, -1, cv2.LINE_AA)
    line_panel = add_header(
        line_panel,
        f"Hough candidates: {len(lines)} | length > {HOUGH_MIN_LINE_LENGTH_PX}px",
    )
    return np.concatenate(
        (input_panel, canny_panel, intersection_panel, line_panel),
        axis=1,
    )


def write_overviews(
    frame_paths: list[Path],
    output_dir: Path,
    chunk_size: int,
) -> None:
    for start in range(0, len(frame_paths), chunk_size):
        chunk = frame_paths[start : start + chunk_size]
        thumbnails: list[np.ndarray] = []
        for path in chunk:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            thumbnails.append(
                cv2.resize(image, (768, 144), interpolation=cv2.INTER_AREA)
            )
        columns = 2
        rows = math.ceil(len(thumbnails) / columns)
        blank = np.zeros_like(thumbnails[0])
        thumbnails.extend([blank] * (rows * columns - len(thumbnails)))
        overview = np.concatenate(
            [
                np.concatenate(
                    thumbnails[row * columns : (row + 1) * columns],
                    axis=1,
                )
                for row in range(rows)
            ],
            axis=0,
        )
        end = start + len(chunk) - 1
        cv2.imwrite(
            str(output_dir / f"overview_{start:02d}_{end:02d}.jpg"),
            overview,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )


def main() -> None:
    args = parse_args()
    mask_dir = args.review_dir / "masks"
    raw_dir = args.review_dir / "raw"
    mask_paths = sorted(mask_dir.glob("*_table.png"))
    selected_paths = evenly_spaced_paths(mask_paths, args.count)
    if not selected_paths:
        raise FileNotFoundError(f"No *_table.png masks found in {mask_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cv2.setRNGSeed(0)

    rows: list[dict[str, object]] = []
    frame_paths: list[Path] = []
    for test_index, mask_path in enumerate(selected_paths):
        frame_id = mask_path.stem.removesuffix("_table")
        rgb = cv2.imread(str(raw_dir / f"{frame_id}.png"), cv2.IMREAD_COLOR)
        table_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or table_mask is None:
            raise RuntimeError(f"Failed to read RGB/mask pair for {frame_id}")
        completed_mask = _largest_filled_component(table_mask)
        target_path = mask_dir / f"{frame_id}_target.png"
        target_mask = (
            cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
            if target_path.exists()
            else None
        )
        effective_mask, exclusion = effective_table_mask(
            completed_mask,
            target_mask,
        )
        edges, intersection, lines = rgb_mask_edge_lines(rgb, effective_mask)
        canvas = visualize(
            rgb,
            completed_mask,
            effective_mask,
            exclusion,
            edges,
            intersection,
            lines,
            title=f"mock {test_index:02d} | source {frame_id}",
        )
        frame_path = frame_dir / f"{test_index:02d}_{frame_id}.jpg"
        cv2.imwrite(
            str(frame_path),
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        frame_paths.append(frame_path)
        rows.append(
            {
                "mock_index": test_index,
                "source_frame": frame_id,
                "target_exclusion": target_mask is not None,
                "mask_area_px": int(np.count_nonzero(completed_mask)),
                "effective_mask_area_px": int(np.count_nonzero(effective_mask)),
                "canny_edge_pixels": int(np.count_nonzero(edges)),
                "intersection_edge_pixels": int(np.count_nonzero(intersection)),
                "intersection_fraction": round(
                    float(np.count_nonzero(intersection))
                    / max(float(np.count_nonzero(edges)), 1.0),
                    4,
                ),
                "candidate_count": len(lines),
                "longest_candidate_px": round(
                    max((line[4] for line in lines), default=0.0),
                    3,
                ),
            }
        )

    with (args.output_dir / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    candidate_counts = [int(row["candidate_count"]) for row in rows]
    summary = {
        "test_count": len(rows),
        "unique_source_frames": len({row["source_frame"] for row in rows}),
        "target_exclusion_count": sum(
            bool(row["target_exclusion"]) for row in rows
        ),
        "zero_candidate_tests": sum(count == 0 for count in candidate_counts),
        "candidate_count_mean": round(float(np.mean(candidate_counts)), 3),
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "intersection_edge_pixels_mean": round(
            float(np.mean([row["intersection_edge_pixels"] for row in rows])),
            3,
        ),
        "parameters": {
            "gaussian_kernel": GAUSSIAN_KERNEL,
            "canny_low": CANNY_LOW,
            "canny_high": CANNY_HIGH,
            "target_exclusion_radius_px": TARGET_EXCLUSION_RADIUS_PX,
            "hough_threshold": HOUGH_THRESHOLD,
            "hough_min_line_length_px": HOUGH_MIN_LINE_LENGTH_PX,
            "hough_max_line_gap_px": HOUGH_MAX_LINE_GAP_PX,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_overviews(frame_paths, args.output_dir, args.overview_size)
    print(json.dumps(summary, indent=2))
    print(f"visualizations: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

