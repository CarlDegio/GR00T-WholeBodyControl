from __future__ import annotations

import pytest

from gear_sonic.pico_video.usb_network import PicoUsbNetwork
from gear_sonic.scripts.run_pico_video_bridge import (
    build_argument_parser,
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
