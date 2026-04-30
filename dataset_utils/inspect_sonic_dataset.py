from __future__ import annotations

import argparse
import json

try:
    from dataset_utils.sonic_dataset_common import (
        build_task_index,
        compute_episode_metrics,
        get_episode_parquet_path,
        get_episode_video_paths,
        get_video_stream_info,
        load_episode_dataframe,
        load_episodes,
        load_info,
        load_modality_config,
        load_tasks,
        resolve_dataset_path,
        row_to_jsonable,
    )
except ImportError:
    from sonic_dataset_common import (
        build_task_index,
        compute_episode_metrics,
        get_episode_parquet_path,
        get_episode_video_paths,
        get_video_stream_info,
        load_episode_dataframe,
        load_episodes,
        load_info,
        load_modality_config,
        load_tasks,
        resolve_dataset_path,
        row_to_jsonable,
    )


DEFAULT_PREVIEW_COLUMNS = [
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
    "observation.state",
    "observation.eef_state",
    "action.wbc",
    "teleop.delta_heading",
    "teleop.smpl_pose",
    "teleop.body_quat_w",
    "teleop.left_hand_joints",
    "teleop.right_hand_joints",
    "teleop.stream_mode",
    "teleop.planner_mode",
    "teleop.vr_3pt_position",
]


def print_section(title: str) -> None:
    print()
    print(f"[{title}]")


def print_dataset_summary(dataset_root, info, tasks, episodes, modality) -> None:
    print_section("Dataset")
    print(f"root: {dataset_root}")
    print(f"fps: {info.get('fps')}")
    print(f"total_episodes: {info.get('total_episodes')}")
    print(f"total_frames: {info.get('total_frames')}")
    print(f"total_tasks: {info.get('total_tasks')}")
    print(f"total_videos: {info.get('total_videos')}")
    print(f"chunks_size: {info.get('chunks_size')}")

    print_section("Tasks")
    for item in tasks:
        print(f"{item['task_index']}: {item['task']}")

    print_section("Episodes")
    for item in episodes:
        task_list = ", ".join(item.get("tasks", []))
        print(
            f"episode={item['episode_index']:06d} length={item['length']} tasks=[{task_list}]"
        )

    print_section("Script Config")
    print(json.dumps(info.get("script_config", {}), ensure_ascii=False, indent=2))

    print_section("Features")
    for key, value in info.get("features", {}).items():
        print(f"{key}: dtype={value.get('dtype')} shape={value.get('shape')}")

    print_section("Modality")
    for group_name, group_values in modality.items():
        print(f"{group_name}: {len(group_values)} entries")
        for key, value in group_values.items():
            original_key = value.get("original_key")
            if original_key is not None:
                print(f"  {key} -> {original_key}")
            else:
                print(f"  {key}: start={value.get('start')} end={value.get('end')}")


def preview_episode(
    dataset_root,
    info,
    tasks,
    episode_index: int,
    preview_rows: int,
    preview_array_items: int,
    all_columns: bool,
) -> None:
    task_index = build_task_index(tasks)
    df = load_episode_dataframe(dataset_root, episode_index, info=info)
    parquet_path = get_episode_parquet_path(dataset_root, info, episode_index)
    video_paths = get_episode_video_paths(dataset_root, info, episode_index)
    metrics = compute_episode_metrics(df)

    print_section(f"Episode {episode_index:06d}")
    print(f"parquet: {parquet_path}")
    print(f"num_rows: {metrics['num_rows']}")
    print(f"duration_sec: {metrics['duration_sec']:.3f}")
    print(
        "timestamp_range: "
        f"{metrics['timestamp_start']} -> {metrics['timestamp_end']}"
    )
    if "stream_mode_counts" in metrics:
        print(f"stream_mode_counts: {metrics['stream_mode_counts']}")
    if "zero_smpl_pose_rows" in metrics:
        print(f"zero_smpl_pose_rows: {metrics['zero_smpl_pose_rows']}")

    print_section("Episode Videos")
    for video_key, video_path in video_paths.items():
        video_info = get_video_stream_info(video_path)
        print(
            f"{video_key}: path={video_info['path']} "
            f"frames={video_info['frames']} "
            f"resolution={video_info['width']}x{video_info['height']} "
            f"fps={video_info['fps']}"
        )

    selected_columns = list(df.columns) if all_columns else [
        column for column in DEFAULT_PREVIEW_COLUMNS if column in df.columns
    ]

    print_section("Preview Columns")
    for column in selected_columns:
        print(column)

    print_section("Preview Rows")
    if len(df) == 0:
        print("episode has no rows")
        return

    row_indices = list(range(min(preview_rows, len(df))))
    tail_start = max(len(df) - preview_rows, 0)
    row_indices.extend(range(tail_start, len(df)))
    row_indices = sorted(set(row_indices))

    for row_index in row_indices:
        row = df.iloc[row_index]
        row_payload = row_to_jsonable(
            row,
            columns=selected_columns,
            preview_items=preview_array_items,
        )
        task_name = task_index.get(int(row["task_index"]), "<unknown>")
        print(
            f"row={row_index} frame={int(row['frame_index'])} "
            f"timestamp={float(row['timestamp']):.3f} task={task_name}"
        )
        print(json.dumps(row_payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a SONIC-exported LeRobot dataset and preview episode contents.",
    )
    parser.add_argument("dataset_path", help="Path to the dataset root under outputs/.")
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Preview a specific episode in detail.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=2,
        help="Number of rows to preview from the head and tail of the episode.",
    )
    parser.add_argument(
        "--preview-array-items",
        type=int,
        default=6,
        help="How many array elements to print per field in previews.",
    )
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Preview every parquet column instead of a curated subset.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = resolve_dataset_path(args.dataset_path)
    info = load_info(dataset_root)
    tasks = load_tasks(dataset_root)
    episodes = load_episodes(dataset_root)
    modality = load_modality_config(dataset_root)

    print_dataset_summary(dataset_root, info, tasks, episodes, modality)

    if args.episode_index is not None:
        preview_episode(
            dataset_root=dataset_root,
            info=info,
            tasks=tasks,
            episode_index=args.episode_index,
            preview_rows=args.preview_rows,
            preview_array_items=args.preview_array_items,
            all_columns=args.all_columns,
        )


if __name__ == "__main__":
    main()
