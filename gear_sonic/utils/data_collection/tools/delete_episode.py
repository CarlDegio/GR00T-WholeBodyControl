"""
Delete one or more episodes from a LeRobot v2.1 dataset.

The script rebuilds the dataset with contiguous episode indices so deleting a
middle episode does not leave gaps in data paths, videos, or metadata.

Example:

    python -m gear_sonic.utils.data_collection.tools.delete_episode \\
        --dataset-path "outputs/0512move pillow" \\
        --episode-index 0 2 \\
        --yes
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import tempfile
from typing import Optional

import pandas as pd
def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def get_parquet_path(dataset_path: Path, info: dict, episode_index: int) -> Path:
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


def get_video_keys(info: dict) -> list[str]:
    video_keys = info.get("video_keys")
    if video_keys:
        return list(video_keys)

    return [
        key
        for key, value in info.get("features", {}).items()
        if value.get("dtype") in {"video", "image"}
    ]


def get_video_paths(dataset_path: Path, info: dict, episode_index: int) -> dict[str, Path]:
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


def episode_exists(dataset_path: Path, info: dict, episode_index: int) -> bool:
    if get_parquet_path(dataset_path, info, episode_index).exists():
        return True
    return any(
        path.exists()
        for path in get_video_paths(dataset_path, info, episode_index).values()
    )


def scalar_stats(series: pd.Series) -> dict:
    if len(series) == 0:
        return {"min": [], "max": [], "mean": [], "std": [], "count": [0]}

    std = float(series.std()) if len(series) > 1 else 0.0
    return {
        "min": [series.min().item() if hasattr(series.min(), "item") else series.min()],
        "max": [series.max().item() if hasattr(series.max(), "item") else series.max()],
        "mean": [float(series.mean())],
        "std": [std],
        "count": [int(len(series))],
    }


def update_episode_stats(stats_row: dict, df: pd.DataFrame, new_episode_index: int) -> dict:
    row = stats_row.copy()
    row["episode_index"] = new_episode_index

    stats = row.get("stats")
    if not isinstance(stats, dict):
        return row

    for key in ("timestamp", "frame_index", "episode_index", "index"):
        if key in df.columns:
            stats[key] = scalar_stats(df[key])

    return row


@dataclass
class DeleteDatasetEpisodeConfig:
    """Delete episodes from a LeRobot dataset and repair indices."""

    dataset_path: str
    """Dataset root containing meta/info.json."""

    episode_index: list[int] = field(default_factory=list)
    """Episode index or indices to delete. Example: --episode-index 0 2."""

    output_path: Optional[str] = None
    """Optional output dataset path. If omitted, the dataset is rewritten in-place."""

    yes: bool = False
    """Confirm an in-place rewrite."""

    dry_run: bool = False
    """Print the planned changes without writing anything."""


def build_rewritten_dataset(
    source_path: Path,
    dest_path: Path,
    delete_indices: set[int],
) -> tuple[int, int, int]:
    info = load_json(source_path / "meta" / "info.json")
    episodes = load_jsonl(source_path / "meta" / "episodes.jsonl")
    episode_stats = load_jsonl(source_path / "meta" / "episodes_stats.jsonl")
    stats_by_episode = {
        int(row["episode_index"]): row
        for row in episode_stats
        if "episode_index" in row
    }

    kept_episodes = [
        row for row in episodes if int(row["episode_index"]) not in delete_indices
    ]
    if not kept_episodes:
        raise SystemExit("ERROR: deleting these episodes would leave the dataset empty.")

    meta_dir = dest_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for src_meta in (source_path / "meta").iterdir():
        if src_meta.name in {"info.json", "episodes.jsonl", "episodes_stats.jsonl"}:
            continue
        if src_meta.is_file():
            shutil.copy2(src_meta, meta_dir / src_meta.name)

    fps = int(info.get("fps", 50))
    chunks_size = int(info.get("chunks_size", 1000))
    total_frames = 0
    rewritten_episodes = []
    rewritten_stats = []
    copied_videos = 0
    old_to_new_episode_index = {}

    for new_idx, ep_meta in enumerate(kept_episodes):
        old_idx = int(ep_meta["episode_index"])
        old_to_new_episode_index[old_idx] = new_idx
        src_parquet = get_parquet_path(source_path, info, old_idx)
        if not src_parquet.exists():
            raise SystemExit(f"ERROR: missing parquet for kept episode {old_idx}: {src_parquet}")

        df = pd.read_parquet(src_parquet)
        ep_len = len(df)
        df["episode_index"] = new_idx
        df["index"] = range(total_frames, total_frames + ep_len)
        df["frame_index"] = range(ep_len)
        if "timestamp" in df.columns:
            df["timestamp"] = [i / fps for i in range(ep_len)]

        dst_parquet = get_parquet_path(dest_path, info, new_idx)
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_parquet)

        for video_key, src_video in get_video_paths(source_path, info, old_idx).items():
            dst_video = get_video_paths(dest_path, info, new_idx)[video_key]
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            if src_video.exists():
                shutil.copy2(src_video, dst_video)
                copied_videos += 1
            else:
                print(f"WARNING: missing video for kept episode {old_idx}: {src_video}")

        new_meta = ep_meta.copy()
        new_meta["episode_index"] = new_idx
        new_meta["length"] = ep_len
        rewritten_episodes.append(new_meta)

        if old_idx in stats_by_episode:
            rewritten_stats.append(
                update_episode_stats(stats_by_episode[old_idx], df, new_idx)
            )

        total_frames += ep_len

    info["total_episodes"] = len(rewritten_episodes)
    info["total_frames"] = total_frames
    info["total_videos"] = copied_videos
    info["total_chunks"] = (len(rewritten_episodes) + chunks_size - 1) // chunks_size
    info["splits"] = {"train": f"0:{len(rewritten_episodes)}"}
    if "discarded_episode_indices" in info:
        remapped_discarded = []
        for old_idx in info.get("discarded_episode_indices", []):
            new_idx = old_to_new_episode_index.get(int(old_idx))
            if new_idx is not None:
                remapped_discarded.append(new_idx)
        info["discarded_episode_indices"] = remapped_discarded

    write_json(meta_dir / "info.json", info)
    write_jsonl(meta_dir / "episodes.jsonl", rewritten_episodes)
    if episode_stats:
        write_jsonl(meta_dir / "episodes_stats.jsonl", rewritten_stats)

    return len(rewritten_episodes), total_frames, copied_videos


def main(cfg: DeleteDatasetEpisodeConfig) -> None:
    source_path = Path(cfg.dataset_path).expanduser().resolve()
    if not (source_path / "meta" / "info.json").exists():
        raise SystemExit(f"ERROR: not a dataset root: {source_path}")

    if not cfg.episode_index:
        raise SystemExit("ERROR: provide at least one --episode-index.")

    delete_indices = set(cfg.episode_index)
    if min(delete_indices) < 0:
        raise SystemExit("ERROR: episode indices must be non-negative.")

    info = load_json(source_path / "meta" / "info.json")
    episodes = load_jsonl(source_path / "meta" / "episodes.jsonl")
    meta_indices = {int(row["episode_index"]) for row in episodes}
    stats_indices = {
        int(row["episode_index"])
        for row in load_jsonl(source_path / "meta" / "episodes_stats.jsonl")
        if "episode_index" in row
    }

    missing = [
        idx
        for idx in sorted(delete_indices)
        if idx not in meta_indices
        and idx not in stats_indices
        and not episode_exists(source_path, info, idx)
    ]
    if missing:
        raise SystemExit(f"ERROR: episode index not found: {missing}")

    kept_count = len([idx for idx in meta_indices if idx not in delete_indices])
    print(f"Dataset: {source_path}")
    print(f"Delete episodes: {sorted(delete_indices)}")
    print(f"Episodes in meta before: {len(meta_indices)}")
    print(f"Episodes in meta after:  {kept_count}")

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
        episodes_count, total_frames, total_videos = build_rewritten_dataset(
            source_path=source_path,
            dest_path=output_path,
            delete_indices=delete_indices,
        )
        print("Delete complete.")
        print(f"Episodes: {episodes_count}")
        print(f"Frames:   {total_frames}")
        print(f"Videos:   {total_videos}")
        print(f"Output:   {output_path}")
        return

    tmp_path = Path(
        tempfile.mkdtemp(prefix=f".{source_path.name}.delete-", dir=source_path.parent)
    )
    try:
        episodes_count, total_frames, total_videos = build_rewritten_dataset(
            source_path=source_path,
            dest_path=tmp_path,
            delete_indices=delete_indices,
        )

        backup_path = source_path.with_name(f"{source_path.name}.backup-before-delete")
        suffix = 1
        while backup_path.exists():
            backup_path = source_path.with_name(
                f"{source_path.name}.backup-before-delete-{suffix}"
            )
            suffix += 1

        source_path.rename(backup_path)
        tmp_path.rename(source_path)
        print(f"Backup: {backup_path}")

        print("Delete complete.")
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

    main(tyro.cli(DeleteDatasetEpisodeConfig))
