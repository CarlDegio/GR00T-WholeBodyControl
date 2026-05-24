"""Print PICO controller stick values once per second.

This is a lightweight diagnostic script that mirrors the controller-axis read
path used by ``pico_manager_thread_server.py`` without starting any robot
control or ZMQ publishing.

Usage:
    python gear_sonic/scripts/read_pico_sticks.py
"""

from __future__ import annotations

import argparse
import subprocess
import time


try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None


def get_controller_axes() -> tuple[float, float, float, float]:
    """Fetch joystick axes in the same format as pico_manager_thread_server.py."""
    if xrt is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        left_axis = xrt.get_left_axis()  # expected [x, y]
        right_axis = xrt.get_right_axis()  # expected [x, y]
        lx = float(left_axis[0]) if len(left_axis) >= 1 else 0.0
        ly = float(left_axis[1]) if len(left_axis) >= 2 else 0.0
        rx = float(right_axis[0]) if len(right_axis) >= 1 else 0.0
        ry = float(right_axis[1]) if len(right_axis) >= 2 else 0.0
        return lx, ly, rx, ry
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-start-service",
        action="store_true",
        help="Do not run /opt/apps/roboticsservice/runService.sh before xrt.init().",
    )
    parser.add_argument(
        "--wait-body-data",
        action="store_true",
        help="Wait until XRoboToolkit reports body data availability before printing.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Print interval in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk before reading PICO input."
        )

    if not args.no_start_service:
        subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])

    xrt.init()

    if args.wait_body_data:
        print("Waiting for XRoboToolkit body data...")
        while not xrt.is_body_data_available():
            time.sleep(0.1)
        print("XRoboToolkit body data available.")

    print("Reading PICO stick axes. Press Ctrl+C to stop.")
    try:
        while True:
            lx, ly, rx, ry = get_controller_axes()
            timestamp_ns = None
            try:
                timestamp_ns = xrt.get_time_stamp_ns()
            except Exception:
                pass

            timestamp_text = (
                f" timestamp_ns={timestamp_ns}" if timestamp_ns is not None else ""
            )
            print(
                f"left_stick=({lx:+.4f}, {ly:+.4f}) "
                f"right_stick=({rx:+.4f}, {ry:+.4f}){timestamp_text}",
                flush=True,
            )
            time.sleep(max(args.interval, 0.01))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
