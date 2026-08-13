#!/usr/bin/env python3
"""Stream SensorGateway head-camera frames to XRoboToolkit Remote Vision."""

from __future__ import annotations

import argparse
import logging
import signal

from gear_sonic.pico_video.bridge import BridgeSettings, PicoVideoBridge
from gear_sonic.pico_video.gateway_source import SensorGatewayVideoSource
from gear_sonic.runtime.client import SensorGatewayClient
from gear_sonic.runtime.config import load_runtime_profile


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="", help="Base runtime YAML profile")
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Partial runtime YAML overlay; may be repeated",
    )
    parser.add_argument(
        "--gateway-endpoint",
        default="",
        help="Override SensorGateway metadata endpoint, e.g. tcp://127.0.0.1:5560",
    )
    parser.add_argument("--control-host", default="0.0.0.0")
    parser.add_argument("--control-port", type=int, default=13579)
    parser.add_argument("--stream", default="camera_encoded/ego_view")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", type=int, default=4_000_000)
    parser.add_argument("--max-age-ms", type=float, default=250.0)
    parser.add_argument("--stale-fps", type=float, default=2.0)
    parser.add_argument(
        "--encoder",
        choices=("h264_nvenc", "libx264"),
        default="h264_nvenc",
    )
    parser.add_argument("--stats-interval-s", type=float, default=5.0)
    parser.add_argument("--request-timeout-ms", type=int, default=250)
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_bridge_settings(args: argparse.Namespace) -> BridgeSettings:
    gateway_endpoint = args.gateway_endpoint
    if not gateway_endpoint:
        profile = load_runtime_profile(
            args.profile or None,
            overlays=tuple(args.overlay),
        )
        address = profile.endpoint("sensor_gateway_metadata")
        gateway_endpoint = f"tcp://{address.host}:{address.port}"
    return BridgeSettings(
        gateway_endpoint=gateway_endpoint,
        stream=args.stream,
        control_host=args.control_host,
        control_port=args.control_port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        bitrate=args.bitrate,
        max_age_ms=args.max_age_ms,
        stale_fps=args.stale_fps,
        encoder=args.encoder,
        stats_interval_s=args.stats_interval_s,
    )


def main() -> int:
    args = build_argument_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = resolve_bridge_settings(args)
    client = SensorGatewayClient(
        settings.gateway_endpoint,
        request_timeout_ms=args.request_timeout_ms,
    )
    source = SensorGatewayVideoSource(
        client,
        max_age_ms=settings.max_age_ms,
        stream=settings.stream,
    )
    bridge = PicoVideoBridge(
        settings,
        source=source,
        source_close=client.close,
    )

    def stop_bridge(_signum: int, _frame: object) -> None:
        bridge.stop()

    signal.signal(signal.SIGINT, stop_bridge)
    signal.signal(signal.SIGTERM, stop_bridge)
    bridge.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
