from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import pandas as pd
from PIL import Image


def resolve_dataset_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()

    if (candidate / "meta" / "info.json").exists():
        return candidate

    for parent in [candidate, *candidate.parents]:
        if (parent / "meta" / "info.json").exists():
            return parent

    raise FileNotFoundError(
        f"Could not find a dataset root from {candidate}. Expected meta/info.json to exist."
    )


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_info(dataset_path: str | Path) -> dict[str, Any]:
    dataset_root = resolve_dataset_path(dataset_path)
    return load_json(dataset_root / "meta" / "info.json")


def load_modality_config(dataset_path: str | Path) -> dict[str, Any]:
    dataset_root = resolve_dataset_path(dataset_path)
    return load_json(dataset_root / "meta" / "modality.json")


def load_tasks(dataset_path: str | Path) -> list[dict[str, Any]]:
    dataset_root = resolve_dataset_path(dataset_path)
    return load_jsonl(dataset_root / "meta" / "tasks.jsonl")


def load_episodes(dataset_path: str | Path) -> list[dict[str, Any]]:
    dataset_root = resolve_dataset_path(dataset_path)
    return load_jsonl(dataset_root / "meta" / "episodes.jsonl")


def build_task_index(tasks: list[dict[str, Any]]) -> dict[int, str]:
    return {int(item["task_index"]): item["task"] for item in tasks}


def get_video_keys(info: dict[str, Any]) -> list[str]:
    video_keys = info.get("video_keys")
    if video_keys:
        return list(video_keys)

    features = info.get("features", {})
    return [
        key
        for key, value in features.items()
        if value.get("dtype") in {"video", "image"}
    ]


def get_episode_parquet_path(
    dataset_path: str | Path,
    info: dict[str, Any],
    episode_index: int,
) -> Path:
    dataset_root = resolve_dataset_path(dataset_path)
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return dataset_root / data_path.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )


def get_episode_video_paths(
    dataset_path: str | Path,
    info: dict[str, Any],
    episode_index: int,
) -> dict[str, Path]:
    dataset_root = resolve_dataset_path(dataset_path)
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    video_path = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )

    result: dict[str, Path] = {}
    for video_key in get_video_keys(info):
        result[video_key] = dataset_root / video_path.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=video_key,
        )
    return result


def load_episode_dataframe(
    dataset_path: str | Path,
    episode_index: int,
    info: dict[str, Any] | None = None,
) -> pd.DataFrame:
    dataset_root = resolve_dataset_path(dataset_path)
    info = info or load_info(dataset_root)
    parquet_path = get_episode_parquet_path(dataset_root, info, episode_index)
    return pd.read_parquet(parquet_path)


def get_video_stream_info(video_path: str | Path) -> dict[str, Any]:
    path = Path(video_path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = None
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        return {
            "path": str(path),
            "width": int(stream.width),
            "height": int(stream.height),
            "frames": int(stream.frames),
            "duration_sec": duration,
            "fps": str(stream.average_rate) if stream.average_rate else None,
            "codec": stream.codec_context.name if stream.codec_context else None,
        }


def compute_episode_metrics(df: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "num_rows": int(len(df)),
        "timestamp_start": float(df["timestamp"].iloc[0]) if len(df) else None,
        "timestamp_end": float(df["timestamp"].iloc[-1]) if len(df) else None,
        "duration_sec": float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])
        if len(df) > 1
        else 0.0,
    }

    if "teleop.stream_mode" in df.columns:
        counts = df["teleop.stream_mode"].value_counts().sort_index()
        metrics["stream_mode_counts"] = {int(k): int(v) for k, v in counts.items()}

    if "teleop.smpl_pose" in df.columns and len(df):
        zero_rows = 0
        for value in df["teleop.smpl_pose"]:
            arr = np.asarray(value)
            if arr.size > 0 and np.all(arr == 0):
                zero_rows += 1
        metrics["zero_smpl_pose_rows"] = zero_rows

    return metrics


def choose_row_indices(num_rows: int, max_rows: int) -> list[int]:
    if num_rows <= 0 or max_rows <= 0:
        return []
    if max_rows >= num_rows:
        return list(range(num_rows))

    sampled = np.linspace(0, num_rows - 1, num=max_rows, dtype=int)
    return sorted({int(index) for index in sampled})


def preview_value(value: Any, max_items: int = 6) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        preview = value[:max_items].tolist()
        if value.shape[0] > max_items:
            return {
                "shape": list(value.shape),
                "preview": preview,
                "truncated": True,
            }
        return preview

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (list, tuple)):
        preview = list(value[:max_items])
        if len(value) > max_items:
            return {
                "length": len(value),
                "preview": preview,
                "truncated": True,
            }
        return preview

    return value


def row_to_jsonable(
    row: pd.Series,
    columns: Iterable[str] | None = None,
    preview_items: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    selected_columns = list(columns) if columns is not None else list(row.index)

    for column in selected_columns:
        value = row[column]
        if preview_items is None:
            result[column] = value_to_python(value)
        else:
            result[column] = preview_value(value, max_items=preview_items)

    return result


def value_to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def value_to_csv_cell(value: Any) -> str | int | float | bool:
    python_value = value_to_python(value)

    if python_value is None:
        return ""

    if isinstance(python_value, (str, int, float, bool)):
        return python_value

    return json.dumps(python_value, ensure_ascii=False)


def dataframe_to_csv_ready(
    df: pd.DataFrame,
    task_index_to_name: dict[int, str] | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    selected_columns = list(columns) if columns is not None else list(df.columns)
    csv_df = pd.DataFrame(index=df.index)
    csv_df["row_index"] = df.index.astype(int)

    if "task_index" in df.columns:
        csv_df["task"] = [
            task_index_to_name.get(int(task_idx), "<unknown>")
            if task_index_to_name is not None
            else int(task_idx)
            for task_idx in df["task_index"].tolist()
        ]

    for column in selected_columns:
        csv_df[column] = [value_to_csv_cell(value) for value in df[column].tolist()]

    return csv_df


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def save_video_frames(
    video_path: str | Path,
    frame_indices: Iterable[int],
    output_dir: str | Path,
    image_prefix: str,
) -> list[Path]:
    wanted = sorted({int(index) for index in frame_indices if int(index) >= 0})
    if not wanted:
        return []

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    wanted_set = set(wanted)
    saved_paths: list[Path] = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx not in wanted_set:
                continue

            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            output_path = output_root / f"{image_prefix}_frame_{frame_idx:06d}.jpg"
            image.save(output_path, quality=95)
            saved_paths.append(output_path)

            if len(saved_paths) == len(wanted):
                break

    return saved_paths
