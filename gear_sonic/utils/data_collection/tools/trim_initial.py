"""
Trim the inactive lead-in from each LeRobot v2.1 dataset episode.

For every episode, the script finds the first row where action.motion_token
differs from row 0, then keeps data from the previous row onward. For example,
if rows 0..20 share the same motion token and row 21 changes, the rewritten
episode starts from original row 20.

Example:

    python -m gear_sonic.utils.data_collection.tools.trim_initial \\
        --dataset-path "outputs/0512move pillow2" \\
        --dry-run

    python -m gear_sonic.utils.data_collection.tools.trim_initial \\
        --dataset-path "outputs/0512move pillow2" \\
        --yes
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
MOTION_TOKEN_COLUMN = "action.motion_token"
KEYFRAME_INTERVAL = 50


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value


def get_parquet_path(dataset_path: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return dataset_path / data_path.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )


def get_video_keys(info: dict[str, Any]) -> list[str]:
    video_keys = info.get("video_keys")
    if video_keys:
        return list(video_keys)

    return [
        key
        for key, value in info.get("features", {}).items()
        if value.get("dtype") in {"video", "image"}
    ]


def get_video_paths(
    dataset_path: Path,
    info: dict[str, Any],
    episode_index: int,
) -> dict[str, Path]:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    video_path = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    return {
        key: dataset_path
        / video_path.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=key,
        )
        for key in get_video_keys(info)
    }


def find_trim_start(df: pd.DataFrame) -> tuple[int, Optional[int]]:
    if MOTION_TOKEN_COLUMN not in df.columns:
        raise SystemExit(f"ERROR: missing parquet column: {MOTION_TOKEN_COLUMN}")
    if len(df) == 0:
        return 0, None

    first_token = np.asarray(df[MOTION_TOKEN_COLUMN].iloc[0])
    for idx, value in enumerate(df[MOTION_TOKEN_COLUMN].iloc[1:], start=1):
        if not np.array_equal(np.asarray(value), first_token):
            return max(idx - 1, 0), idx

    return 0, None


def video_indices_from_rows(df: pd.DataFrame, start_row: int, fps: int) -> np.ndarray:
    kept = df.iloc[start_row:]
    if "frame_index" in kept.columns:
        return kept["frame_index"].to_numpy(dtype=np.int64)
    if "timestamp" in kept.columns:
        return np.rint(kept["timestamp"].to_numpy(dtype=np.float64) * fps).astype(np.int64)
    return np.arange(start_row, len(df), dtype=np.int64)


def trim_video_ffmpeg(
    src_video_path: Path,
    dst_video_path: Path,
    video_key: str,
    episode_index: int,
    start_frame: int,
    end_frame: Optional[int],
    fps: int,
) -> int:
    dst_video_path.parent.mkdir(parents=True, exist_ok=True)

    trim_expr = f"trim=start_frame={start_frame}"
    if end_frame is not None:
        trim_expr += f":end_frame={end_frame}"
    vf = f"{trim_expr},setpts=PTS-STARTPTS"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_video_path),
        "-vf",
        vf,
        "-an",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-g",
        str(KEYFRAME_INTERVAL),
        "-keyint_min",
        str(KEYFRAME_INTERVAL),
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        str(dst_video_path),
    ]
    subprocess.run(cmd, check=True)

    expected_frames = max((end_frame or start_frame) - start_frame, 0)
    print(
        f"  video done: episode {episode_index:06d} {video_key} "
        f"frames={expected_frames}",
        flush=True,
    )
    return expected_frames


def dataframe_to_episode_buffer(df: pd.DataFrame, features: dict[str, Any]) -> dict[str, Any]:
    buffer = {}
    for key, feature in features.items():
        if feature.get("dtype") == "video" or key not in df.columns:
            continue

        series = df[key]
        first = series.iloc[0] if len(series) else None
        if isinstance(first, np.ndarray):
            buffer[key] = np.stack(series.to_numpy())
        else:
            buffer[key] = series.to_numpy()
    return buffer


def compute_episode_stats(episode_index: int, df: pd.DataFrame, info: dict[str, Any]) -> dict[str, Any]:
    from lerobot.common.datasets.lerobot_dataset import (
        compute_episode_stats as lerobot_compute_episode_stats,
    )

    non_video_features = {
        key: value
        for key, value in info.get("features", {}).items()
        if value.get("dtype") != "video"
    }
    episode_buffer = dataframe_to_episode_buffer(df, non_video_features)
    stats = lerobot_compute_episode_stats(episode_buffer, non_video_features)
    return {"episode_index": episode_index, "stats": to_jsonable(stats)}


@dataclass
class RemoveInitialConfig:
    """Remove inactive initial frames from every episode in a dataset."""

    dataset_path: str
    """Dataset root containing meta/info.json."""

    output_path: Optional[str] = None
    """Optional output dataset path. If omitted, the dataset is rewritten in-place."""

    yes: bool = False
    """Confirm an in-place rewrite."""

    dry_run: bool = False
    """Print per-episode trim points without writing anything."""

    video_workers: int = 3
    """Number of ffmpeg video trim jobs to run concurrently."""


def copy_static_meta(source_path: Path, dest_path: Path) -> None:
    meta_dir = dest_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for src_meta in (source_path / "meta").iterdir():
        if src_meta.name in {"info.json", "episodes.jsonl", "episodes_stats.jsonl"}:
            continue
        if src_meta.is_file():
            shutil.copy2(src_meta, meta_dir / src_meta.name)


def scan_trim_plan(source_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info = load_json(source_path / "meta" / "info.json")
    episodes = load_jsonl(source_path / "meta" / "episodes.jsonl")
    fps = int(info.get("fps", 50))
    plan = []

    for ep_meta in episodes:
        episode_index = int(ep_meta["episode_index"])
        parquet_path = get_parquet_path(source_path, info, episode_index)
        if not parquet_path.exists():
            raise SystemExit(f"ERROR: missing parquet for episode {episode_index}: {parquet_path}")

        df = pd.read_parquet(parquet_path, columns=[MOTION_TOKEN_COLUMN])
        start_row, change_row = find_trim_start(df)
        video_indices = video_indices_from_rows(df, start_row, fps)
        plan.append({
            "episode_index": episode_index,
            "old_length": int(len(df)),
            "new_length": int(len(df) - start_row),
            "start_row": int(start_row),
            "change_row": None if change_row is None else int(change_row),
            "video_start_index": int(video_indices[0]) if len(video_indices) else None,
        })

    return info, plan


def build_trimmed_dataset(
    source_path: Path,
    dest_path: Path,
    video_workers: int,
) -> tuple[int, int, int]:
    info, plan = scan_trim_plan(source_path)
    episodes = load_jsonl(source_path / "meta" / "episodes.jsonl")
    fps = int(info.get("fps", 50))
    chunks_size = int(info.get("chunks_size", 1000))
    plan_by_episode = {item["episode_index"]: item for item in plan}

    copy_static_meta(source_path, dest_path)

    total_frames = 0
    video_jobs = []
    rewritten_episodes = []
    rewritten_stats = []
    total_episodes = len(episodes)

    with ThreadPoolExecutor(max_workers=max(video_workers, 1)) as executor:
        for new_episode_index, ep_meta in enumerate(episodes):
            old_episode_index = int(ep_meta["episode_index"])
            item = plan_by_episode[old_episode_index]
            start_row = int(item["start_row"])

            src_parquet = get_parquet_path(source_path, info, old_episode_index)
            df = pd.read_parquet(src_parquet)
            video_indices = video_indices_from_rows(df, start_row, fps)
            video_start = int(video_indices[0]) if len(video_indices) else 0
            video_end = int(video_indices[-1]) + 1 if len(video_indices) else video_start
            df = df.iloc[start_row:].copy().reset_index(drop=True)

            ep_len = len(df)
            df["episode_index"] = new_episode_index
            df["index"] = range(total_frames, total_frames + ep_len)
            df["frame_index"] = range(ep_len)
            if "timestamp" in df.columns:
                df["timestamp"] = [idx / fps for idx in range(ep_len)]

            dst_parquet = get_parquet_path(dest_path, info, new_episode_index)
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst_parquet)
            print(
                f"episode {new_episode_index + 1}/{total_episodes} "
                f"parquet done: old={old_episode_index:06d} "
                f"new={new_episode_index:06d} keep_from={start_row} "
                f"length={ep_len}",
                flush=True,
            )

            for video_key, src_video in get_video_paths(source_path, info, old_episode_index).items():
                dst_video = get_video_paths(dest_path, info, new_episode_index)[video_key]
                if not src_video.exists():
                    print(f"WARNING: missing video for episode {old_episode_index}: {src_video}")
                    continue
                video_jobs.append(
                    executor.submit(
                        trim_video_ffmpeg,
                        src_video,
                        dst_video,
                        video_key,
                        new_episode_index,
                        video_start,
                        video_end,
                        fps,
                    )
                )

            new_meta = ep_meta.copy()
            new_meta["episode_index"] = new_episode_index
            new_meta["length"] = ep_len
            rewritten_episodes.append(new_meta)
            rewritten_stats.append(compute_episode_stats(new_episode_index, df, info))
            total_frames += ep_len
            print(
                f"episode {new_episode_index + 1}/{total_episodes} "
                f"stats done: new={new_episode_index:06d}",
                flush=True,
            )

        total_videos = 0
        for future in as_completed(video_jobs):
            future.result()
            total_videos += 1
            print(f"videos completed: {total_videos}/{len(video_jobs)}", flush=True)

    info["total_episodes"] = len(rewritten_episodes)
    info["total_frames"] = total_frames
    info["total_videos"] = total_videos
    info["total_chunks"] = (len(rewritten_episodes) + chunks_size - 1) // chunks_size
    info["splits"] = {"train": f"0:{len(rewritten_episodes)}"}

    write_json(dest_path / "meta" / "info.json", info)
    write_jsonl(dest_path / "meta" / "episodes.jsonl", rewritten_episodes)
    write_jsonl(dest_path / "meta" / "episodes_stats.jsonl", rewritten_stats)

    return len(rewritten_episodes), total_frames, total_videos


def print_plan(source_path: Path, plan: list[dict[str, Any]]) -> None:
    print(f"Dataset: {source_path}")
    print("Trim plan:")
    total_removed = 0
    total_kept = 0
    for item in plan:
        removed = int(item["old_length"] - item["new_length"])
        total_removed += removed
        total_kept += int(item["new_length"])
        change = item["change_row"]
        change_text = "none" if change is None else str(change)
        print(
            f"  episode {item['episode_index']:06d}: "
            f"change_row={change_text}, keep_from={item['start_row']}, "
            f"length {item['old_length']} -> {item['new_length']} "
            f"({removed} removed)"
        )
    print(f"Total frames after trim: {total_kept} ({total_removed} removed)")


def main(cfg: RemoveInitialConfig) -> None:
    source_path = Path(cfg.dataset_path).expanduser().resolve()
    if not (source_path / "meta" / "info.json").exists():
        raise SystemExit(f"ERROR: not a dataset root: {source_path}")

    _, plan = scan_trim_plan(source_path)
    print_plan(source_path, plan)

    if cfg.dry_run:
        print("Dry run only; no files were changed.")
        return

    output_path = Path(cfg.output_path).expanduser().resolve() if cfg.output_path else None
    in_place = output_path is None
    if in_place and not cfg.yes:
        raise SystemExit("ERROR: in-place rewrite requires --yes. Use --dry-run to inspect first.")

    if not in_place:
        if output_path.exists():
            raise SystemExit(f"ERROR: output path already exists: {output_path}")
        episodes_count, total_frames, total_videos = build_trimmed_dataset(
            source_path=source_path,
            dest_path=output_path,
            video_workers=cfg.video_workers,
        )
        print("Trim complete.")
        print(f"Episodes: {episodes_count}")
        print(f"Frames:   {total_frames}")
        print(f"Videos:   {total_videos}")
        print(f"Output:   {output_path}")
        return

    tmp_path = Path(
        tempfile.mkdtemp(prefix=f".{source_path.name}.remove-initial-", dir=source_path.parent)
    )
    try:
        episodes_count, total_frames, total_videos = build_trimmed_dataset(
            source_path=source_path,
            dest_path=tmp_path,
            video_workers=cfg.video_workers,
        )

        backup_path = source_path.with_name(f"{source_path.name}.backup-before-remove-initial")
        suffix = 1
        while backup_path.exists():
            backup_path = source_path.with_name(
                f"{source_path.name}.backup-before-remove-initial-{suffix}"
            )
            suffix += 1

        source_path.rename(backup_path)
        tmp_path.rename(source_path)
        print(f"Backup: {backup_path}")
        print("Trim complete.")
        print(f"Episodes: {episodes_count}")
        print(f"Frames:   {total_frames}")
        print(f"Videos:   {total_videos}")
        print(f"Output:   {source_path}")
    except Exception:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        raise


if __name__ == "__main__":
    import tyro

    main(tyro.cli(RemoveInitialConfig))
