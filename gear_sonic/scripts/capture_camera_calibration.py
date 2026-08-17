#!/usr/bin/env python3
"""Capture and preserve one complete head/chest RGB-D calibration packet."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Callable

from gear_sonic.camera.calibration import (
    CameraCalibrationError,
    persist_calibration_capture,
)
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


DEFAULT_ACTIVE_PATH = Path("gear_sonic/config/camera_intrinsics.json")
DEFAULT_BACKUP_ROOT = Path("outputs/camera_calibration")


def capture_camera_calibration(
    *,
    camera_host: str,
    camera_port: int,
    active_path: str | Path,
    backup_root: str | Path,
    timeout_sec: float,
    ready_file: str | Path | None = None,
    capture_id: str | None = None,
    client_factory: Callable[..., Any] = ComposedCameraClientSensor,
) -> Path:
    if timeout_sec <= 0.0:
        raise ValueError("timeout_sec must be greater than zero")
    identifier = capture_id or time.strftime("%Y%m%d_%H%M%S")
    client = client_factory(
        server_ip=camera_host,
        port=int(camera_port),
        decode_images=True,
    )
    if ready_file is not None:
        ready_path = Path(ready_file)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.touch()

    deadline = time.monotonic() + timeout_sec
    latest_error: CameraCalibrationError | None = None
    try:
        while time.monotonic() < deadline:
            message = client.read(blocking=False)
            if message is None:
                time.sleep(0.01)
                continue
            try:
                backup_dir = persist_calibration_capture(
                    message,
                    active_path=active_path,
                    backup_root=backup_root,
                    capture_id=identifier,
                    robot_host=camera_host,
                )
            except CameraCalibrationError as exc:
                latest_error = exc
                time.sleep(0.01)
                continue
            print(
                f"Saved complete dual-camera calibration to {Path(active_path).resolve()} "
                f"and {backup_dir.resolve()}",
                flush=True,
            )
            return backup_dir
    finally:
        client.close()

    suffix = f"; latest invalid packet: {latest_error}" if latest_error else ""
    raise TimeoutError(
        "timed out waiting for complete dual-camera calibration from "
        f"{camera_host}:{camera_port}{suffix}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one complete head/chest RGB-D calibration packet."
    )
    parser.add_argument("--camera-host", default="192.168.123.164")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--active-path", type=Path, default=DEFAULT_ACTIVE_PATH)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--capture-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_camera_calibration(
        camera_host=args.camera_host,
        camera_port=args.camera_port,
        active_path=args.active_path,
        backup_root=args.backup_root,
        timeout_sec=args.timeout_sec,
        ready_file=args.ready_file,
        capture_id=args.capture_id,
    )


if __name__ == "__main__":
    main()

