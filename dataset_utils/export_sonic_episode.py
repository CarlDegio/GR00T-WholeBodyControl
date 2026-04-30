from __future__ import annotations

import argparse
from pathlib import Path

try:
    from dataset_utils.sonic_dataset_common import (
        build_task_index,
        choose_row_indices,
        compute_episode_metrics,
        dataframe_to_csv_ready,
        dump_json,
        get_episode_parquet_path,
        get_episode_video_paths,
        get_video_stream_info,
        load_episode_dataframe,
        load_episodes,
        load_info,
        load_tasks,
        resolve_dataset_path,
        row_to_jsonable,
        save_csv,
        save_jsonl,
        save_video_frames,
    )
except ImportError:
    from sonic_dataset_common import (
        build_task_index,
        choose_row_indices,
        compute_episode_metrics,
        dataframe_to_csv_ready,
        dump_json,
        get_episode_parquet_path,
        get_episode_video_paths,
        get_video_stream_info,
        load_episode_dataframe,
        load_episodes,
        load_info,
        load_tasks,
        resolve_dataset_path,
        row_to_jsonable,
        save_csv,
        save_jsonl,
        save_video_frames,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one SONIC episode to JSONL and optionally extract video frames.",
    )
    parser.add_argument("dataset_path", help="Path to the dataset root under outputs/.")
    parser.add_argument("episode_index", type=int, help="Episode index to export.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for exported artifacts. Defaults to dataset_utils/exports/<dataset>/episode_<id>/.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional subset of parquet columns to export.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="When not using --all-rows, export up to this many evenly sampled rows.",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Export every row from the parquet episode.",
    )
    parser.add_argument(
        "--extract-video-frames",
        action="store_true",
        help="Decode and save selected RGB frames from each video stream.",
    )
    parser.add_argument(
        "--frame-indices",
        type=int,
        nargs="*",
        default=None,
        help="Frame indices to extract. Defaults to [0, middle, last].",
    )
    parser.add_argument(
        "--image-keys",
        nargs="*",
        default=None,
        help="Optional subset of video keys to decode.",
    )
    return parser


def get_default_output_dir(dataset_root: Path, episode_index: int) -> Path:
    return (
        Path("dataset_utils")
        / "exports"
        / dataset_root.name
        / f"episode_{episode_index:06d}"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = resolve_dataset_path(args.dataset_path)
    info = load_info(dataset_root)
    tasks = load_tasks(dataset_root)
    episodes = load_episodes(dataset_root)
    task_index = build_task_index(tasks)

    df = load_episode_dataframe(dataset_root, args.episode_index, info=info)
    parquet_path = get_episode_parquet_path(dataset_root, info, args.episode_index)
    video_paths = get_episode_video_paths(dataset_root, info, args.episode_index)
    metrics = compute_episode_metrics(df)

    output_dir = Path(args.output_dir) if args.output_dir else get_default_output_dir(
        dataset_root, args.episode_index
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_rows:
        selected_rows = list(range(len(df)))
    else:
        selected_rows = choose_row_indices(len(df), args.max_rows)

    selected_columns = args.columns if args.columns else list(df.columns)

    rows_payload = []
    for row_index in selected_rows:
        row = df.iloc[row_index]
        row_payload = {
            "row_index": int(row_index),
            "task": task_index.get(int(row["task_index"]), "<unknown>"),
        }
        row_payload.update(row_to_jsonable(row, columns=selected_columns, preview_items=None))
        rows_payload.append(row_payload)

    episode_meta = next(
        (item for item in episodes if int(item["episode_index"]) == args.episode_index),
        None,
    )
    video_summary = {
        video_key: get_video_stream_info(video_path)
        for video_key, video_path in video_paths.items()
    }

    summary = {
        "dataset_root": str(dataset_root),
        "episode_index": args.episode_index,
        "episode_meta": episode_meta,
        "parquet_path": str(parquet_path),
        "metrics": metrics,
        "selected_columns": selected_columns,
        "exported_row_count": len(rows_payload),
        "video_summary": video_summary,
    }

    summary_path = output_dir / f"episode_{args.episode_index:06d}_summary.json"
    rows_path = output_dir / f"episode_{args.episode_index:06d}_rows.jsonl"
    csv_path = output_dir / f"episode_{args.episode_index:06d}_full.csv"
    dump_json(summary, summary_path)
    save_jsonl(rows_payload, rows_path)
    csv_df = dataframe_to_csv_ready(
        df=df,
        task_index_to_name=task_index,
        columns=selected_columns,
    )
    save_csv(csv_df, csv_path)

    print(f"summary_json: {summary_path}")
    print(f"rows_jsonl: {rows_path}")
    print(f"full_csv: {csv_path}")

    if args.extract_video_frames:
        if len(df) == 0:
            frame_indices: list[int] = []
        elif args.frame_indices:
            frame_indices = [int(index) for index in args.frame_indices]
        else:
            frame_indices = sorted({0, len(df) // 2, len(df) - 1})

        selected_image_keys = args.image_keys if args.image_keys else list(video_paths.keys())
        for video_key in selected_image_keys:
            if video_key not in video_paths:
                raise KeyError(
                    f"Unknown video key '{video_key}'. Available keys: {sorted(video_paths.keys())}"
                )

            safe_key = video_key.replace(".", "_")
            saved_paths = save_video_frames(
                video_path=video_paths[video_key],
                frame_indices=frame_indices,
                output_dir=output_dir / safe_key,
                image_prefix=safe_key,
            )
            print(f"{video_key}: saved {len(saved_paths)} frames")
            for saved_path in saved_paths:
                print(f"  {saved_path}")


if __name__ == "__main__":
    main()
