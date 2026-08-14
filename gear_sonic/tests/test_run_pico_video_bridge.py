from __future__ import annotations

import errno
import threading

import pytest

from gear_sonic.pico_video.bridge import PicoUsbLinkChanged
from gear_sonic.pico_video.usb_network import PicoUsbNetwork, PicoUsbNetworkError
from gear_sonic.scripts.run_pico_video_bridge import (
    build_argument_parser,
    run_bridge_supervisor,
    resolve_bridge_settings,
)


PICO_USB = PicoUsbNetwork(
    interface="enx4662be5cb0cb",
    workstation_ip="192.168.123.61",
    pico_ip="192.168.123.242",
    prefix_length=24,
    serial="PA9410MGL1090624G",
)


def test_bridge_cli_uses_sonic_head_and_sensor_gateway_defaults() -> None:
    parser = build_argument_parser()
    args = parser.parse_args(
        ["--gateway-endpoint", "tcp://127.0.0.1:6000", "--control-port", "14579"]
    )

    settings = resolve_bridge_settings(args, usb_network_factory=lambda _interface: PICO_USB)

    assert settings.gateway_endpoint == "tcp://127.0.0.1:6000"
    assert settings.control_host == "192.168.123.61"
    assert settings.control_port == 14579
    assert settings.pico_usb == PICO_USB
    assert settings.stream == "camera_encoded/ego_view"
    assert settings.width == 1280
    assert settings.height == 480
    assert settings.fps == 30
    assert settings.bitrate == 4_000_000
    assert settings.max_age_ms == 250.0
    assert settings.stale_fps == 2.0
    assert settings.encoder == "h264_nvenc"
    assert args.stay_alive is False
    assert args.usb_retry_interval_s == 2.0
    assert args.usb_check_interval_s == 2.0


def test_bridge_cli_allows_portable_encoder_and_runtime_overrides() -> None:
    args = build_argument_parser().parse_args(
        [
            "--gateway-endpoint",
            "tcp://127.0.0.1:6001",
            "--network-mode",
            "local-test",
            "--control-host",
            "127.0.0.1",
            "--encoder",
            "libx264",
            "--max-age-ms",
            "400",
            "--stale-fps",
            "1",
        ]
    )

    settings = resolve_bridge_settings(args)

    assert settings.gateway_endpoint == "tcp://127.0.0.1:6001"
    assert settings.control_host == "127.0.0.1"
    assert settings.pico_usb is None
    assert settings.encoder == "libx264"
    assert settings.max_age_ms == 400.0
    assert settings.stale_fps == 1.0


def test_bridge_cli_fails_when_usb_only_is_not_available() -> None:
    args = build_argument_parser().parse_args(
        ["--gateway-endpoint", "tcp://127.0.0.1:6001"]
    )

    with pytest.raises(RuntimeError, match="PICO USBOnly"):
        resolve_bridge_settings(
            args,
            usb_network_factory=lambda _interface: (_ for _ in ()).throw(
                RuntimeError("PICO USBOnly is unavailable")
            ),
        )


def test_bridge_cli_rejects_manual_bind_address_in_usb_only_mode() -> None:
    args = build_argument_parser().parse_args(
        [
            "--gateway-endpoint",
            "tcp://127.0.0.1:6001",
            "--control-host",
            "0.0.0.0",
        ]
    )

    with pytest.raises(ValueError, match="local-test"):
        resolve_bridge_settings(args, usb_network_factory=lambda _interface: PICO_USB)


@pytest.mark.parametrize("control_host", ["0.0.0.0", "192.168.1.20"])
def test_local_test_mode_rejects_non_loopback_bind_address(control_host: str) -> None:
    args = build_argument_parser().parse_args(
        [
            "--gateway-endpoint",
            "tcp://127.0.0.1:6001",
            "--network-mode",
            "local-test",
            "--control-host",
            control_host,
        ]
    )

    with pytest.raises(ValueError, match="loopback"):
        resolve_bridge_settings(args)


def test_stay_alive_waits_for_usb_and_restarts_after_link_change() -> None:
    args = build_argument_parser().parse_args(
        [
            "--gateway-endpoint",
            "tcp://127.0.0.1:6001",
            "--stay-alive",
            "--usb-retry-interval-s",
            "0.001",
            "--usb-check-interval-s",
            "0.001",
        ]
    )
    stop_event = threading.Event()
    discoveries = 0
    bridges: list[FakeBridge] = []

    def discover(_interface: str | None) -> PicoUsbNetwork:
        nonlocal discoveries
        discoveries += 1
        if discoveries == 1:
            raise PicoUsbNetworkError("not connected yet")
        return PICO_USB

    class FakeBridge:
        def __init__(self) -> None:
            self.stopped = False
            bridges.append(self)

        def serve_forever(self) -> None:
            if len(bridges) == 1:
                raise PicoUsbLinkChanged("link changed")
            stop_event.set()

        def stop(self) -> None:
            self.stopped = True

    run_bridge_supervisor(
        args,
        usb_network_factory=discover,
        bridge_factory=lambda _args, _settings, _probe, _shutdown: FakeBridge(),
        stop_event=stop_event,
    )

    assert discoveries == 3
    assert len(bridges) == 2
    assert all(bridge.stopped for bridge in bridges)


def test_stay_alive_is_restricted_to_usb_only_mode() -> None:
    args = build_argument_parser().parse_args(
        [
            "--gateway-endpoint",
            "tcp://127.0.0.1:6001",
            "--network-mode",
            "local-test",
            "--stay-alive",
        ]
    )

    with pytest.raises(ValueError, match="usb-only"):
        run_bridge_supervisor(args)


def test_stop_requested_during_bridge_construction_prevents_serve() -> None:
    args = build_argument_parser().parse_args(
        [
            "--gateway-endpoint",
            "tcp://127.0.0.1:6001",
            "--stay-alive",
        ]
    )
    stop_event = threading.Event()

    class FakeBridge:
        def __init__(self) -> None:
            self.served = False
            self.stopped = False

        def serve_forever(self) -> None:
            self.served = True

        def stop(self) -> None:
            self.stopped = True

    bridge = FakeBridge()

    def build(_args, _settings, _probe, _shutdown):
        stop_event.set()
        return bridge

    run_bridge_supervisor(
        args,
        usb_network_factory=lambda _interface: PICO_USB,
        bridge_factory=build,
        stop_event=stop_event,
    )

    assert bridge.stopped is True
    assert bridge.served is False


@pytest.mark.parametrize(
    "failure",
    [
        PermissionError("SO_BINDTODEVICE denied"),
        OSError("control port already in use"),
        RuntimeError("permanent bridge configuration error"),
    ],
)
def test_stay_alive_does_not_hide_permanent_bridge_failures(
    failure: Exception,
) -> None:
    args = build_argument_parser().parse_args(
        [
            "--gateway-endpoint",
            "tcp://127.0.0.1:6001",
            "--stay-alive",
            "--usb-retry-interval-s",
            "0.001",
        ]
    )
    stop_event = threading.Event()
    discoveries = 0

    def discover(_interface: str | None) -> PicoUsbNetwork:
        nonlocal discoveries
        discoveries += 1
        if discoveries > 1:
            stop_event.set()
        return PICO_USB

    class FailingBridge:
        def serve_forever(self) -> None:
            raise failure

        def stop(self) -> None:
            return None

    with pytest.raises(type(failure), match=str(failure)):
        run_bridge_supervisor(
            args,
            usb_network_factory=discover,
            bridge_factory=lambda _a, _s, _p, _shutdown: FailingBridge(),
            stop_event=stop_event,
        )

    assert discoveries == 1


def test_stay_alive_retries_transient_socket_error_only_after_usb_loss() -> None:
    args = build_argument_parser().parse_args(
        [
            "--gateway-endpoint",
            "tcp://127.0.0.1:6001",
            "--stay-alive",
            "--usb-retry-interval-s",
            "0.001",
        ]
    )
    stop_event = threading.Event()
    discoveries = 0
    bridges = 0

    def discover(_interface: str | None) -> PicoUsbNetwork:
        nonlocal discoveries
        discoveries += 1
        if discoveries == 2:
            raise PicoUsbNetworkError("link disappeared")
        return PICO_USB

    class FakeBridge:
        def serve_forever(self) -> None:
            nonlocal bridges
            bridges += 1
            if bridges == 1:
                raise OSError(errno.ENETDOWN, "USB interface went down")
            stop_event.set()

        def stop(self) -> None:
            return None

    run_bridge_supervisor(
        args,
        usb_network_factory=discover,
        bridge_factory=lambda _a, _s, _p, _shutdown: FakeBridge(),
        stop_event=stop_event,
    )

    assert discoveries == 3
    assert bridges == 2
