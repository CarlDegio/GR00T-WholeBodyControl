from __future__ import annotations

from gear_sonic.scripts.run_pico_video_bridge import (
    build_argument_parser,
    resolve_bridge_settings,
)


def test_bridge_cli_uses_sonic_head_and_sensor_gateway_defaults() -> None:
    parser = build_argument_parser()
    args = parser.parse_args(
        ["--gateway-endpoint", "tcp://127.0.0.1:6000", "--control-port", "14579"]
    )

    settings = resolve_bridge_settings(args)

    assert settings.gateway_endpoint == "tcp://127.0.0.1:6000"
    assert settings.control_host == "0.0.0.0"
    assert settings.control_port == 14579
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
    assert settings.encoder == "libx264"
    assert settings.max_age_ms == 400.0
    assert settings.stale_fps == 1.0
