#!/usr/bin/env python3
"""List connected RealSense cameras, let the user pick one, and show its RGB stream."""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
import pyrealsense2 as rs


def list_devices() -> list[dict[str, str]]:
    """Return connected RealSense devices with human-readable metadata."""
    context = rs.context()
    devices = context.query_devices()

    device_infos: list[dict[str, str]] = []
    for device in devices:
        device_infos.append(
            {
                "name": device.get_info(rs.camera_info.name),
                "serial_number": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
            }
        )

    device_infos.sort(key=lambda item: item["serial_number"])
    return device_infos


def prompt_device_index(devices: list[dict[str, str]]) -> int:
    """Ask the user to choose a device index."""
    while True:
        selected = input(f"Select a camera [0-{len(devices) - 1}]: ").strip()

        if not selected.isdigit():
            print("Please enter a valid number.")
            continue

        index = int(selected)
        if 0 <= index < len(devices):
            return index

        print("Selection out of range.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List RealSense cameras, choose one, and display its RGB stream."
    )
    parser.add_argument("--width", type=int, default=640, help="RGB stream width")
    parser.add_argument("--height", type=int, default=480, help="RGB stream height")
    parser.add_argument("--fps", type=int, default=30, help="RGB stream FPS")
    args = parser.parse_args()

    devices = list_devices()
    if not devices:
        print("No RealSense cameras found.")
        return 1

    print("Detected RealSense cameras:")
    for index, device in enumerate(devices):
        print(
            f"[{index}] {device['name']} | "
            f"serial_number={device['serial_number']} | "
            f"firmware={device['firmware_version']}"
        )

    selected_index = prompt_device_index(devices)
    selected_device = devices[selected_index]

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_device["serial_number"])
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    window_name = f"RealSense RGB - {selected_device['serial_number']}"
    print(f"Opening RGB stream for serial_number={selected_device['serial_number']}")
    print("Press 'q' or ESC to quit.")

    started = False

    try:
        pipeline.start(config)
        started = True

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow(window_name, color_image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
