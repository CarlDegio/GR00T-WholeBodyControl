#!/usr/bin/env python3
"""Simultaneously test four RealSense cameras and report frame-read stats.

This script opens four RealSense color streams at 640x480@30 FPS, continuously
reads frames without displaying them, and prints per-camera statistics so it is
easy to spot the case where a device enumerates successfully but never returns
images.

By default it tries to load ``head``, ``chest``, ``left``, and ``right`` serial
numbers from ``camera_serial_num.txt`` next to this script. All serials can
also be overridden from the command line.

Example:
    python test_3camera.py

    python test_3camera.py --cameras head left right

    python test_3camera.py \
        --head-serial 347522071257 \
        --chest-serial 408122070390 \
        --left-serial 218622279421 \
        --right-serial 352122270966
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import signal
import threading
import time
from typing import Any

import numpy as np
import pyrealsense2 as rs


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SERIAL_FILE = REPO_ROOT / "camera_serial_num.txt"
CAMERA_NAMES = ("head", "chest", "left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open four RealSense color streams at the same time and report "
            "per-camera frame-read statistics."
        )
    )
    parser.add_argument("--head-serial", default=None, help="Head camera RealSense serial")
    parser.add_argument("--chest-serial", default=None, help="Chest camera RealSense serial")
    parser.add_argument("--left-serial", default=None, help="Left camera RealSense serial")
    parser.add_argument("--right-serial", default=None, help="Right camera RealSense serial")
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=CAMERA_NAMES,
        default=list(CAMERA_NAMES),
        help="Cameras to open, defaults to all four",
    )
    parser.add_argument(
        "--serial-file",
        default=str(DEFAULT_SERIAL_FILE),
        help="Path to serial mapping file, defaults to ./camera_serial_num.txt",
    )
    parser.add_argument("--width", type=int, default=640, help="Color stream width")
    parser.add_argument("--height", type=int, default=480, help="Color stream height")
    parser.add_argument("--fps", type=int, default=30, help="Target color stream FPS")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=1000,
        help="Per-frame wait timeout in milliseconds",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=1.0,
        help="How often to print statistics in seconds",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional test duration in seconds, 0 means run until Ctrl-C",
    )
    return parser.parse_args()


def load_serial_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            mapping[key] = value
    return mapping


def list_connected_devices() -> list[dict[str, str]]:
    devices = []
    for device in rs.context().query_devices():
        devices.append(
            {
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
            }
        )
    devices.sort(key=lambda item: item["serial"])
    return devices


def resolve_selected_serials(args: argparse.Namespace) -> dict[str, str]:
    file_mapping = load_serial_mapping(Path(args.serial_file))
    all_serials = {
        "head": args.head_serial or file_mapping.get("head"),
        "chest": args.chest_serial or file_mapping.get("chest"),
        "left": args.left_serial or file_mapping.get("left"),
        "right": args.right_serial or file_mapping.get("right"),
    }
    serials = {name: all_serials[name] for name in args.cameras}

    missing = [name for name, serial in serials.items() if not serial]
    if missing:
        raise ValueError(
            "Missing serial numbers for: "
            + ", ".join(missing)
            + ". Pass them via command line or put them into camera_serial_num.txt."
        )
    return serials


@dataclass
class CameraStats:
    name: str
    serial: str
    open_ok: bool = False
    open_error: str | None = None
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    exception_count: int = 0
    first_frame_time: float | None = None
    last_frame_time: float | None = None
    last_frame_shape: tuple[int, ...] | None = None
    stream_profile: str | None = None
    last_error: str | None = None
    start_time: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_open_ok(self, stream_profile: str) -> None:
        with self._lock:
            self.open_ok = True
            self.stream_profile = stream_profile
            self.open_error = None

    def set_open_error(self, message: str) -> None:
        with self._lock:
            self.open_ok = False
            self.open_error = message
            self.last_error = message

    def record_success(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        with self._lock:
            self.success_count += 1
            if self.first_frame_time is None:
                self.first_frame_time = now
            self.last_frame_time = now
            self.last_frame_shape = tuple(frame.shape)
            self.last_error = None

    def record_timeout(self, message: str) -> None:
        with self._lock:
            self.failure_count += 1
            self.timeout_count += 1
            self.last_error = message

    def record_exception(self, message: str) -> None:
        with self._lock:
            self.failure_count += 1
            self.exception_count += 1
            self.last_error = message

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "serial": self.serial,
                "open_ok": self.open_ok,
                "open_error": self.open_error,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "timeout_count": self.timeout_count,
                "exception_count": self.exception_count,
                "first_frame_time": self.first_frame_time,
                "last_frame_time": self.last_frame_time,
                "last_frame_shape": self.last_frame_shape,
                "stream_profile": self.stream_profile,
                "last_error": self.last_error,
                "uptime": time.monotonic() - self.start_time,
            }


def format_profile(profile: rs.pipeline_profile) -> str:
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    stream_format = color_profile.format()
    return (
        f"{color_profile.width()}x{color_profile.height()} "
        f"@ {color_profile.fps()}fps "
        f"format={stream_format}"
    )


def camera_worker(
    name: str,
    serial: str,
    width: int,
    height: int,
    fps: int,
    timeout_ms: int,
    stop_event: threading.Event,
    stats: CameraStats,
) -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    try:
        profile = pipeline.start(config)
        stats.set_open_ok(format_profile(profile))
        print(f"[{name}] pipeline started, serial={serial}, {stats.snapshot()['stream_profile']}")
    except Exception as exc:
        stats.set_open_error(str(exc))
        print(f"[{name}] failed to start pipeline: {exc}")
        return

    try:
        while not stop_event.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    stats.record_timeout("No color frame returned")
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                if color_image.size == 0:
                    stats.record_timeout("Empty color image")
                    continue

                stats.record_success(color_image)
            except RuntimeError as exc:
                stats.record_timeout(str(exc))
            except Exception as exc:
                stats.record_exception(str(exc))
    finally:
        try:
            pipeline.stop()
        except Exception as exc:
            print(f"[{name}] failed to stop pipeline cleanly: {exc}")


def print_device_list(devices: list[dict[str, str]]) -> None:
    print("Detected RealSense devices:")
    if not devices:
        print("  <none>")
        return

    for device in devices:
        print(
            f"  serial={device['serial']} | "
            f"name={device['name']} | "
            f"firmware={device['firmware']}"
        )


def print_selected_mapping(serials: dict[str, str], camera_names: list[str]) -> None:
    print("Selected camera mapping:")
    for name in camera_names:
        print(f"  {name:<5} -> {serials[name]}")


def render_snapshot_line(
    snapshot: dict[str, Any],
    prev_snapshot: dict[str, Any] | None,
    interval_seconds: float,
) -> str:
    interval_success = snapshot["success_count"]
    if prev_snapshot is not None:
        interval_success -= prev_snapshot["success_count"]

    interval_fps = interval_success / interval_seconds if interval_seconds > 0 else 0.0
    last_frame_time = snapshot["last_frame_time"]
    if last_frame_time is None:
        stall = "never"
        first_frame = "no"
    else:
        stall = f"{time.monotonic() - last_frame_time:.2f}s"
        first_frame = "yes"

    open_state = "ok" if snapshot["open_ok"] else "fail"
    shape = snapshot["last_frame_shape"] or "-"
    stream_profile = snapshot["stream_profile"] or "-"
    last_error = snapshot["last_error"] or "-"

    return (
        f"[{snapshot['name']}] open={open_state} first_frame={first_frame} "
        f"total_ok={snapshot['success_count']} total_fail={snapshot['failure_count']} "
        f"timeout={snapshot['timeout_count']} exc={snapshot['exception_count']} "
        f"fps_1s={interval_fps:.2f} stall={stall} shape={shape} "
        f"profile={stream_profile} last_error={last_error}"
    )


def main() -> int:
    args = parse_args()
    camera_names = list(dict.fromkeys(args.cameras))
    serials = resolve_selected_serials(args)
    devices = list_connected_devices()

    print_device_list(devices)
    if not devices:
        print("No RealSense devices found.")
        return 1

    connected_serials = {device["serial"] for device in devices}
    missing_serials = {
        name: serial for name, serial in serials.items() if serial not in connected_serials
    }
    if missing_serials:
        print_selected_mapping(serials, camera_names)
        for name, serial in missing_serials.items():
            print(f"Missing device for {name}: serial={serial}")
        return 1

    print_selected_mapping(serials, camera_names)
    print(
        f"Starting {len(camera_names)}-camera test with color stream "
        f"{args.width}x{args.height}@{args.fps}fps, "
        f"timeout={args.timeout_ms}ms"
    )
    print("Press Ctrl-C to stop.")

    stop_event = threading.Event()
    stats = {
        name: CameraStats(name=name, serial=serial)
        for name, serial in serials.items()
    }

    def _handle_signal(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    threads = []
    for name in camera_names:
        thread = threading.Thread(
            target=camera_worker,
            args=(
                name,
                serials[name],
                args.width,
                args.height,
                args.fps,
                args.timeout_ms,
                stop_event,
                stats[name],
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    prev_snapshots: dict[str, dict[str, Any]] = {}
    start_time = time.monotonic()
    last_report_time = start_time

    try:
        while not stop_event.is_set():
            now = time.monotonic()
            elapsed = now - start_time
            if args.duration > 0 and elapsed >= args.duration:
                stop_event.set()
                break

            time.sleep(args.report_interval)
            report_now = time.monotonic()
            interval_seconds = report_now - last_report_time
            last_report_time = report_now

            print("")
            print(f"=== {elapsed + args.report_interval:.1f}s ===")
            for name in camera_names:
                snapshot = stats[name].snapshot()
                print(
                    render_snapshot_line(
                        snapshot=snapshot,
                        prev_snapshot=prev_snapshots.get(name),
                        interval_seconds=interval_seconds,
                    )
                )
                prev_snapshots[name] = snapshot
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)

        print("")
        print("=== final summary ===")
        for name in camera_names:
            snapshot = stats[name].snapshot()
            print(
                render_snapshot_line(
                    snapshot=snapshot,
                    prev_snapshot=None,
                    interval_seconds=max(snapshot["uptime"], 1e-6),
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
