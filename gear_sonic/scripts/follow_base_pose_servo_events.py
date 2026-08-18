"""Follow the newest per-run raw-YOLOE servo event log."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import TextIO


LOG_NAME = "raw_servo_events.jsonl"
RUN_PATTERNS = ("raw_yoloe_*_g*", "dual_raw_yoloe_*_g*")


def find_latest_event_log(output_root: Path) -> Path | None:
    """Return the most recently modified per-run event log, if one exists."""
    candidates: list[tuple[int, str, Path]] = []
    for pattern in RUN_PATTERNS:
        for run_dir in output_root.glob(pattern):
            event_log = run_dir / LOG_NAME
            try:
                modified_ns = event_log.stat().st_mtime_ns
            except OSError:
                continue
            candidates.append((modified_ns, str(event_log), event_log))
    if not candidates:
        return None
    return max(candidates)[2]


def _open_log(path: Path, *, start_at_end: bool) -> TextIO:
    handle = path.open("r", encoding="utf-8")
    if start_at_end:
        handle.seek(0, 2)
    return handle


def follow_event_logs(output_root: Path, poll_interval_s: float = 0.1) -> None:
    """Print appended JSONL records and switch to each newer navigation run."""
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    initial_logs = {
        path.resolve()
        for pattern in RUN_PATTERNS
        for path in root.glob(f"{pattern}/{LOG_NAME}")
    }
    active_path: Path | None = None
    handle: TextIO | None = None

    print(f"[YOLOE log] Waiting for navigation events below {root}", flush=True)
    try:
        while True:
            latest = find_latest_event_log(root)
            if latest is not None:
                latest = latest.resolve()
            if latest is not None and latest != active_path:
                if handle is not None:
                    handle.close()
                start_at_end = active_path is None and latest in initial_logs
                try:
                    handle = _open_log(latest, start_at_end=start_at_end)
                except OSError:
                    handle = None
                else:
                    active_path = latest
                    position = "new events" if start_at_end else "from first event"
                    print(f"\n[YOLOE log] Following {latest} ({position})", flush=True)

            printed = False
            if handle is not None:
                while line := handle.readline():
                    print(line, end="", flush=True)
                    printed = True
                try:
                    if active_path is not None and handle.tell() > active_path.stat().st_size:
                        handle.seek(0)
                except OSError:
                    handle.close()
                    handle = None
                    active_path = None
            if not printed:
                time.sleep(poll_interval_s)
    finally:
        if handle is not None:
            handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously print the newest raw-YOLOE servo event log."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/base_pose_adjustment"),
    )
    parser.add_argument("--poll-interval-s", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.poll_interval_s <= 0:
        raise SystemExit("--poll-interval-s must be positive")
    follow_event_logs(args.output_root, args.poll_interval_s)


if __name__ == "__main__":
    main()
