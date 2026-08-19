#!/usr/bin/env python3
"""Visual mock evaluation for ordered-contour table-edge segmentation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MASK_DIR = Path(
    "outputs/base_pose_adjustment/dual_raw_yoloe_20260819_175053_g1/"
    "review_samples/masks"
)
DEFAULT_RAW_DIR = DEFAULT_MASK_DIR.parent / "raw"
DEFAULT_OUTPUT_DIR = DEFAULT_MASK_DIR.parent / "table_edge_ordered_contour_mock_200"


@dataclass
class Segment:
    points: np.ndarray
    center: np.ndarray
    direction: np.ndarray
    residual_q90: float
    residual_max: float
    length_px: float
    endpoint_a: np.ndarray
    endpoint_b: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--approx-epsilon-px", type=float, default=6.0)
    parser.add_argument("--residual-q90-px", type=float, default=5.0)
    parser.add_argument("--merge-angle-deg", type=float, default=12.0)
    parser.add_argument("--min-candidate-length-px", type=float, default=45.0)
    parser.add_argument("--min-split-points", type=int, default=12)
    parser.add_argument("--max-split-depth", type=int, default=6)
    parser.add_argument("--overview-size", type=int, default=25)
    return parser.parse_args()


def largest_external_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)


def circular_smooth(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(points) < window:
        return points.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    sigma = max(1.0, window / 4.0)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    smoothed = np.zeros_like(points, dtype=np.float64)
    for offset, weight in zip(offsets.astype(int), weights, strict=True):
        smoothed += np.roll(points, -offset, axis=0) * weight
    return smoothed


def fit_segment(points: np.ndarray) -> Segment:
    center = points.mean(axis=0)
    centered = points - center
    if len(points) < 2 or np.allclose(centered, 0.0):
        direction = np.array([1.0, 0.0], dtype=np.float64)
    else:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    normal = np.array([-direction[1], direction[0]])
    residuals = np.abs(centered @ normal)
    projections = centered @ direction
    endpoint_a = center + direction * float(projections.min(initial=0.0))
    endpoint_b = center + direction * float(projections.max(initial=0.0))
    return Segment(
        points=points,
        center=center,
        direction=direction,
        residual_q90=float(np.quantile(residuals, 0.90)),
        residual_max=float(residuals.max(initial=0.0)),
        length_px=float(projections.max(initial=0.0) - projections.min(initial=0.0)),
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
    )


def split_segment(
    points: np.ndarray,
    *,
    residual_q90_px: float,
    min_split_points: int,
    max_depth: int,
    depth: int = 0,
) -> list[Segment]:
    fitted = fit_segment(points)
    if fitted.residual_q90 <= residual_q90_px:
        return [fitted]
    if depth >= max_depth or len(points) < 2 * min_split_points + 1:
        return []

    normal = np.array([-fitted.direction[1], fitted.direction[0]])
    residuals = np.abs((points - fitted.center) @ normal)
    valid = np.arange(min_split_points, len(points) - min_split_points)
    if len(valid) == 0:
        return []
    split_at = int(valid[np.argmax(residuals[valid])])
    left = split_segment(
        points[: split_at + 1],
        residual_q90_px=residual_q90_px,
        min_split_points=min_split_points,
        max_depth=max_depth,
        depth=depth + 1,
    )
    right = split_segment(
        points[split_at:],
        residual_q90_px=residual_q90_px,
        min_split_points=min_split_points,
        max_depth=max_depth,
        depth=depth + 1,
    )
    return left + right


def contour_anchor_indices(points: np.ndarray, epsilon_px: float) -> list[int]:
    contour = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour, epsilon_px, True)[:, 0, :].astype(np.float64)
    indices: list[int] = []
    for vertex in approx:
        index = int(np.argmin(np.sum((points - vertex) ** 2, axis=1)))
        indices.append(index)
    return sorted(set(indices))


def ordered_arcs(points: np.ndarray, anchors: list[int]) -> list[np.ndarray]:
    if len(anchors) < 2:
        return [np.concatenate([points, points[:1]], axis=0)]
    arcs: list[np.ndarray] = []
    for index, start in enumerate(anchors):
        end = anchors[(index + 1) % len(anchors)]
        if end > start:
            arc = points[start : end + 1]
        else:
            arc = np.concatenate([points[start:], points[: end + 1]], axis=0)
        if len(arc) >= 2:
            arcs.append(arc)
    return arcs


def direction_delta_deg(first: Segment, second: Segment) -> float:
    dot = float(np.clip(abs(first.direction @ second.direction), 0.0, 1.0))
    return math.degrees(math.acos(dot))


def try_merge(
    first: Segment,
    second: Segment,
    *,
    residual_q90_px: float,
    merge_angle_deg: float,
) -> Segment | None:
    if direction_delta_deg(first, second) > merge_angle_deg:
        return None
    joined = np.concatenate([first.points, second.points[1:]], axis=0)
    fitted = fit_segment(joined)
    if fitted.residual_q90 > residual_q90_px:
        return None
    return fitted


def merge_adjacent_segments(
    segments: list[Segment],
    *,
    residual_q90_px: float,
    merge_angle_deg: float,
) -> list[Segment]:
    if len(segments) < 2:
        return segments
    changed = True
    while changed and len(segments) > 1:
        changed = False
        merged: list[Segment] = []
        index = 0
        while index < len(segments):
            if index + 1 < len(segments):
                candidate = try_merge(
                    segments[index],
                    segments[index + 1],
                    residual_q90_px=residual_q90_px,
                    merge_angle_deg=merge_angle_deg,
                )
                if candidate is not None:
                    merged.append(candidate)
                    index += 2
                    changed = True
                    continue
            merged.append(segments[index])
            index += 1
        segments = merged

    if len(segments) > 1:
        wrap = try_merge(
            segments[-1],
            segments[0],
            residual_q90_px=residual_q90_px,
            merge_angle_deg=merge_angle_deg,
        )
        if wrap is not None:
            segments = [wrap, *segments[1:-1]]
    return segments


def segment_ordered_contour(
    mask: np.ndarray,
    *,
    smooth_window: int,
    approx_epsilon_px: float,
    residual_q90_px: float,
    merge_angle_deg: float,
    min_split_points: int,
    max_split_depth: int,
) -> tuple[np.ndarray | None, np.ndarray | None, list[Segment]]:
    raw_contour = largest_external_contour(mask)
    if raw_contour is None or len(raw_contour) < 3:
        return raw_contour, None, []
    smooth_contour = circular_smooth(raw_contour, smooth_window)
    anchors = contour_anchor_indices(smooth_contour, approx_epsilon_px)
    segments: list[Segment] = []
    for arc in ordered_arcs(smooth_contour, anchors):
        segments.extend(
            split_segment(
                arc,
                residual_q90_px=residual_q90_px,
                min_split_points=min_split_points,
                max_depth=max_split_depth,
            )
        )
    segments = merge_adjacent_segments(
        segments,
        residual_q90_px=residual_q90_px,
        merge_angle_deg=merge_angle_deg,
    )
    return raw_contour, smooth_contour, segments


def elastic_variant(
    mask: np.ndarray,
    raw: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = mask.shape
    y, x = np.mgrid[:height, :width].astype(np.float32)
    amplitude = float(rng.uniform(1.5, 3.0))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
    dx = amplitude * (
        np.sin(2.0 * np.pi * y / 137.0 + phase[0])
        + 0.45 * np.sin(2.0 * np.pi * x / 211.0 + phase[1])
    )
    dy = amplitude * (
        np.sin(2.0 * np.pi * x / 149.0 + phase[2])
        + 0.45 * np.sin(2.0 * np.pi * y / 197.0 + phase[3])
    )
    map_x = x + dx.astype(np.float32)
    map_y = y + dy.astype(np.float32)
    warped_mask = cv2.remap(
        mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )
    warped_raw = cv2.remap(
        raw, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return warped_mask, warped_raw, amplitude


PALETTE = [
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (155, 89, 182),
    (230, 126, 34),
    (26, 188, 156),
    (231, 76, 60),
    (149, 165, 166),
]


def point_tuple(point: np.ndarray) -> tuple[int, int]:
    return tuple(np.rint(point).astype(int).tolist())


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (15, 15, 15), -1)
    cv2.putText(
        output,
        text,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return output


def visualize(
    raw: np.ndarray,
    mask: np.ndarray,
    raw_contour: np.ndarray | None,
    smooth_contour: np.ndarray | None,
    segments: list[Segment],
    *,
    min_candidate_length_px: float,
    title: str,
) -> np.ndarray:
    if raw.shape[:2] != mask.shape:
        raw = cv2.resize(raw, (mask.shape[1], mask.shape[0]))
    overlay = raw.copy()
    tint = np.zeros_like(overlay)
    tint[:, :] = (220, 130, 30)
    inside = mask > 0
    overlay[inside] = cv2.addWeighted(overlay[inside], 0.56, tint[inside], 0.44, 0.0)
    if raw_contour is not None:
        cv2.polylines(
            overlay,
            [np.rint(raw_contour).astype(np.int32)],
            True,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    overlay = add_header(overlay, f"input: {title}")

    colored = (raw.astype(np.float32) * 0.42).astype(np.uint8)
    if smooth_contour is not None:
        cv2.polylines(
            colored,
            [np.rint(smooth_contour).astype(np.int32)],
            True,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )
    for index, segment in enumerate(segments):
        color = PALETTE[index % len(PALETTE)]
        pts = np.rint(segment.points).astype(np.int32)
        cv2.polylines(colored, [pts], False, color, 3, cv2.LINE_AA)
    colored = add_header(colored, f"ordered arcs: {len(segments)} segments")

    fitted = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    fitted = (fitted.astype(np.float32) * 0.24).astype(np.uint8)
    candidates = [s for s in segments if s.length_px > min_candidate_length_px]
    for index, segment in enumerate(segments):
        is_candidate = segment.length_px > min_candidate_length_px
        color = PALETTE[index % len(PALETTE)] if is_candidate else (85, 85, 85)
        thickness = 3 if is_candidate else 1
        a = point_tuple(segment.endpoint_a)
        b = point_tuple(segment.endpoint_b)
        cv2.line(fitted, a, b, color, thickness, cv2.LINE_AA)
        if is_candidate:
            midpoint = point_tuple((segment.endpoint_a + segment.endpoint_b) * 0.5)
            cv2.putText(
                fitted,
                str(index),
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.circle(fitted, a, 3, color, -1, cv2.LINE_AA)
            cv2.circle(fitted, b, 3, color, -1, cv2.LINE_AA)
    fitted = add_header(
        fitted,
        f"TLS lines: {len(candidates)} candidates (length > {min_candidate_length_px:.0f}px)",
    )
    return np.concatenate([overlay, colored, fitted], axis=1)


def make_overviews(frame_paths: list[Path], output_dir: Path, chunk_size: int) -> None:
    for start in range(0, len(frame_paths), chunk_size):
        chunk = frame_paths[start : start + chunk_size]
        thumbs: list[np.ndarray] = []
        for path in chunk:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            thumb = cv2.resize(image, (576, 144), interpolation=cv2.INTER_AREA)
            thumbs.append(thumb)
        columns = 5
        rows = math.ceil(len(thumbs) / columns)
        blank = np.zeros_like(thumbs[0])
        thumbs.extend([blank] * (rows * columns - len(thumbs)))
        overview = np.concatenate(
            [np.concatenate(thumbs[row * columns : (row + 1) * columns], axis=1) for row in range(rows)],
            axis=0,
        )
        end = start + len(chunk) - 1
        cv2.imwrite(str(output_dir / f"overview_{start:03d}_{end:03d}.jpg"), overview)


def main() -> None:
    args = parse_args()
    mask_paths = sorted(args.mask_dir.glob("*_table.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No *_table.png masks found in {args.mask_dir}")
    if args.count < 1:
        raise ValueError("--count must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    extra_count = max(0, args.count - len(mask_paths))
    extra_indices = rng.choice(len(mask_paths), size=extra_count, replace=False).tolist()
    jobs = [(path, "original") for path in mask_paths[: args.count]]
    jobs.extend((mask_paths[index], "elastic") for index in extra_indices)

    rows: list[dict[str, object]] = []
    frame_paths: list[Path] = []
    for test_index, (mask_path, variant) in enumerate(jobs):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read {mask_path}")
        frame_id = mask_path.stem.removesuffix("_table")
        raw_path = args.raw_dir / f"{frame_id}.png"
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None:
            raw = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        amplitude = 0.0
        if variant == "elastic":
            mask, raw, amplitude = elastic_variant(mask, raw, rng)

        raw_contour, smooth_contour, segments = segment_ordered_contour(
            mask,
            smooth_window=args.smooth_window,
            approx_epsilon_px=args.approx_epsilon_px,
            residual_q90_px=args.residual_q90_px,
            merge_angle_deg=args.merge_angle_deg,
            min_split_points=args.min_split_points,
            max_split_depth=args.max_split_depth,
        )
        candidates = [s for s in segments if s.length_px > args.min_candidate_length_px]
        contour_length = (
            float(cv2.arcLength(np.rint(raw_contour).astype(np.int32).reshape(-1, 1, 2), True))
            if raw_contour is not None
            else 0.0
        )
        candidate_support = sum(
            float(np.linalg.norm(np.diff(segment.points, axis=0), axis=1).sum())
            for segment in candidates
        )
        q90_values = [segment.residual_q90 for segment in candidates]
        title = f"mock {test_index:03d} | source {frame_id} | {variant}"
        canvas = visualize(
            raw,
            mask,
            raw_contour,
            smooth_contour,
            segments,
            min_candidate_length_px=args.min_candidate_length_px,
            title=title,
        )
        frame_path = frame_dir / f"{test_index:03d}_{frame_id}_{variant}.jpg"
        cv2.imwrite(str(frame_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        frame_paths.append(frame_path)
        rows.append(
            {
                "mock_index": test_index,
                "source_frame": frame_id,
                "variant": variant,
                "elastic_amplitude_px": round(amplitude, 3),
                "mask_area_px": int(np.count_nonzero(mask)),
                "contour_points": 0 if raw_contour is None else len(raw_contour),
                "contour_length_px": round(contour_length, 3),
                "segment_count": len(segments),
                "candidate_count": len(candidates),
                "candidate_coverage": round(candidate_support / max(contour_length, 1e-9), 4),
                "longest_candidate_px": round(max((s.length_px for s in candidates), default=0.0), 3),
                "mean_candidate_q90_px": round(float(np.mean(q90_values)) if q90_values else 0.0, 3),
                "max_candidate_q90_px": round(max(q90_values, default=0.0), 3),
            }
        )

    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "test_count": len(rows),
        "unique_source_masks": len({row["source_frame"] for row in rows}),
        "original_count": sum(row["variant"] == "original" for row in rows),
        "elastic_count": sum(row["variant"] == "elastic" for row in rows),
        "failed_contours": sum(int(row["contour_points"]) == 0 for row in rows),
        "zero_candidate_tests": sum(int(row["candidate_count"]) == 0 for row in rows),
        "candidate_count_mean": round(float(np.mean([row["candidate_count"] for row in rows])), 3),
        "candidate_count_min": int(min(row["candidate_count"] for row in rows)),
        "candidate_count_max": int(max(row["candidate_count"] for row in rows)),
        "candidate_coverage_mean": round(float(np.mean([row["candidate_coverage"] for row in rows])), 4),
        "longest_candidate_px_mean": round(
            float(np.mean([row["longest_candidate_px"] for row in rows])), 3
        ),
        "parameters": {
            "smooth_window": args.smooth_window,
            "approx_epsilon_px": args.approx_epsilon_px,
            "residual_q90_px": args.residual_q90_px,
            "merge_angle_deg": args.merge_angle_deg,
            "min_candidate_length_px": args.min_candidate_length_px,
            "min_split_points": args.min_split_points,
            "max_split_depth": args.max_split_depth,
            "seed": args.seed,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_overviews(frame_paths, args.output_dir, args.overview_size)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"visualizations: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
