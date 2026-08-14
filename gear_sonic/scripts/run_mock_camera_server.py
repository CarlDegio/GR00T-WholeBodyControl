#!/usr/bin/env python3
"""Publish an animated SONIC head-camera test source without a robot."""

from __future__ import annotations

import argparse
import signal

from gear_sonic.pico_video.mock_camera import MockCameraPublisher, MockCameraSettings


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    publisher = MockCameraPublisher(
        MockCameraSettings(
            port=args.port,
            width=args.width,
            height=args.height,
            fps=args.fps,
            jpeg_quality=args.jpeg_quality,
        )
    )

    def stop_publisher(_signum: int, _frame: object) -> None:
        publisher.stop()

    signal.signal(signal.SIGINT, stop_publisher)
    signal.signal(signal.SIGTERM, stop_publisher)
    publisher.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
