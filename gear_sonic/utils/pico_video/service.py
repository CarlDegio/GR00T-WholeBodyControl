#!/usr/bin/env python3
"""Stream SensorGateway ego and wrist cameras to XRoboToolkit Remote Vision."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import errno
import ipaddress
import logging
import signal
import threading

from gear_sonic.utils.pico_video.bridge import (
    BridgeSettings,
    PicoUsbLinkChanged,
    PicoVideoBridge,
)
from gear_sonic.utils.pico_video.gateway_source import SensorGatewayVideoSource
from gear_sonic.utils.pico_video.usb_network import (
    PicoUsbNetwork,
    PicoUsbNetworkError,
    discover_pico_usb_network,
)
from gear_sonic.runtime.gateway.sensor_client import SensorGatewayClient
from gear_sonic.runtime.profile import load_runtime_profile


LOGGER = logging.getLogger(__name__)


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
        help="Test-only SensorGateway override; requires --network-mode local-test",
    )
    parser.add_argument(
        "--network-mode",
        choices=("usb-only", "local-test"),
        default="usb-only",
        help="Use native PICO USBOnly (default) or explicit loopback-only testing",
    )
    parser.add_argument(
        "--pico-usb-interface",
        default="",
        help="Select one PICO RNDIS interface when multiple headsets are attached",
    )
    parser.add_argument(
        "--control-host",
        default="",
        help="Loopback bind override; valid only with --network-mode local-test",
    )
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
    parser.add_argument(
        "--stay-alive",
        action="store_true",
        help="Wait for USBOnly and restart after PICO disconnects or address changes",
    )
    parser.add_argument(
        "--usb-retry-interval-s",
        type=float,
        default=2.0,
        help="Delay between USBOnly discovery and bridge restart attempts",
    )
    parser.add_argument(
        "--usb-check-interval-s",
        type=float,
        default=2.0,
        help="Interval for checking that the active PICO USBOnly link is unchanged",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


UsbNetworkFactory = Callable[[str | None], PicoUsbNetwork]


def resolve_bridge_settings(
    args: argparse.Namespace,
    *,
    usb_network_factory: UsbNetworkFactory = discover_pico_usb_network,
) -> BridgeSettings:
    gateway_endpoint = args.gateway_endpoint
    if not gateway_endpoint:
        profile = load_runtime_profile(
            args.profile or None,
            overlays=tuple(args.overlay),
        )
        address = profile.endpoint("sensor_gateway_metadata")
        gateway_endpoint = f"tcp://{address.host}:{address.port}"
    pico_usb: PicoUsbNetwork | None = None
    if args.network_mode == "usb-only":
        if args.gateway_endpoint:
            raise ValueError(
                "--gateway-endpoint is available only with --network-mode local-test; "
                "select production endpoints with --profile/--overlay"
            )
        if args.control_host:
            raise ValueError(
                "--control-host is available only with --network-mode local-test"
            )
        pico_usb = usb_network_factory(args.pico_usb_interface or None)
        control_host = pico_usb.workstation_ip
    else:
        if args.pico_usb_interface:
            raise ValueError(
                "--pico-usb-interface cannot be used with --network-mode local-test"
            )
        control_host = args.control_host or "127.0.0.1"
        try:
            local_test_address = ipaddress.ip_address(control_host)
        except ValueError as exc:
            raise ValueError("local-test control host must be a loopback IPv4 address") from exc
        if not isinstance(local_test_address, ipaddress.IPv4Address) or not local_test_address.is_loopback:
            raise ValueError("local-test control host must be a loopback IPv4 address")
    return BridgeSettings(
        gateway_endpoint=gateway_endpoint,
        stream=args.stream,
        control_host=control_host,
        control_port=args.control_port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        bitrate=args.bitrate,
        max_age_ms=args.max_age_ms,
        stale_fps=args.stale_fps,
        encoder=args.encoder,
        stats_interval_s=args.stats_interval_s,
        pico_usb=pico_usb,
    )


UsbNetworkProbe = Callable[[], PicoUsbNetwork]
BridgeFactory = Callable[
    [
        argparse.Namespace,
        BridgeSettings,
        UsbNetworkProbe | None,
        threading.Event,
    ],
    PicoVideoBridge,
]


def _build_bridge(
    args: argparse.Namespace,
    settings: BridgeSettings,
    usb_network_probe: UsbNetworkProbe | None,
    shutdown_event: threading.Event,
) -> PicoVideoBridge:
    client = SensorGatewayClient(
        settings.gateway_endpoint,
        request_timeout_ms=args.request_timeout_ms,
    )
    try:
        source = SensorGatewayVideoSource(
            client,
            max_age_ms=settings.max_age_ms,
            stream=settings.stream,
        )
        return PicoVideoBridge(
            settings,
            source=source,
            source_close=client.close,
            usb_network_probe=usb_network_probe,
            usb_check_interval_s=args.usb_check_interval_s,
            shutdown_event=shutdown_event,
        )
    except Exception:
        client.close()
        raise


_TRANSIENT_USB_ERRNOS = frozenset(
    {
        errno.EADDRNOTAVAIL,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.ENODEV,
        errno.ENXIO,
    }
)


def _usb_transition_confirmed(
    error: OSError,
    settings: BridgeSettings,
    *,
    interface: str | None,
    usb_network_factory: UsbNetworkFactory,
) -> bool:
    expected = settings.pico_usb
    if expected is None or error.errno not in _TRANSIENT_USB_ERRNOS:
        return False
    try:
        current = usb_network_factory(interface)
    except PicoUsbNetworkError:
        return True
    return current != expected


def run_bridge_supervisor(
    args: argparse.Namespace,
    *,
    usb_network_factory: UsbNetworkFactory = discover_pico_usb_network,
    bridge_factory: BridgeFactory = _build_bridge,
    stop_event: threading.Event | None = None,
) -> None:
    """Run once, or keep rediscovering native USBOnly when supervision is enabled."""

    if args.stay_alive and args.network_mode != "usb-only":
        raise ValueError("--stay-alive is available only with --network-mode usb-only")
    if args.usb_retry_interval_s <= 0.0:
        raise ValueError("USB retry interval must be positive")
    if args.usb_check_interval_s <= 0.0:
        raise ValueError("USB check interval must be positive")

    stopped = stop_event or threading.Event()
    while not stopped.is_set():
        try:
            settings = resolve_bridge_settings(
                args,
                usb_network_factory=usb_network_factory,
            )
        except PicoUsbNetworkError as exc:
            if not args.stay_alive:
                raise
            LOGGER.info(
                "Waiting for PICO native USBOnly network: %s",
                exc,
            )
            stopped.wait(args.usb_retry_interval_s)
            continue

        usb_network_probe: UsbNetworkProbe | None = None
        if settings.pico_usb is not None:
            interface = args.pico_usb_interface or None
            usb_network_probe = lambda: usb_network_factory(interface)
        if stopped.is_set():
            return

        bridge: PicoVideoBridge | None = None
        try:
            bridge = bridge_factory(
                args,
                settings,
                usb_network_probe,
                stopped,
            )
            if stopped.is_set():
                bridge.stop()
                return
            bridge.serve_forever()
            if args.stay_alive and not stopped.is_set():
                LOGGER.warning(
                    "PICO video bridge exited; restarting after USB rediscovery"
                )
        except PicoUsbLinkChanged as exc:
            if not args.stay_alive:
                raise
            LOGGER.warning(
                "PICO video bridge stopped (%s); rediscovering USBOnly",
                exc,
            )
        except OSError as exc:
            if not args.stay_alive or not _usb_transition_confirmed(
                exc,
                settings,
                interface=args.pico_usb_interface or None,
                usb_network_factory=usb_network_factory,
            ):
                raise
            LOGGER.warning(
                "PICO USBOnly socket stopped (%s); rediscovering the link",
                exc,
            )
        finally:
            if bridge is not None:
                bridge.stop()

        if not args.stay_alive or stopped.is_set():
            return
        stopped.wait(args.usb_retry_interval_s)


def main() -> int:
    args = build_argument_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stop_event = threading.Event()

    def stop_bridge(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop_bridge)
    signal.signal(signal.SIGTERM, stop_bridge)
    run_bridge_supervisor(
        args,
        stop_event=stop_event,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
