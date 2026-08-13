from __future__ import annotations

import struct

import pytest

from gear_sonic.pico_video.protocol import (
    CameraRequest,
    ControlFrameDecoder,
    ProtocolError,
    frame_video_access_unit,
    parse_camera_request,
)


def _control_packet(command: bytes, data: bytes) -> bytes:
    body = struct.pack("<I", len(command)) + command
    body += struct.pack("<I", len(data)) + data
    return struct.pack(">I", len(body)) + body


def _camera_payload(
    *,
    width: int = 1280,
    height: int = 480,
    fps: int = 30,
    bitrate: int = 4_000_000,
    enable_mv_hevc: int = 0,
    render_mode: int = 0,
    port: int = 12345,
    camera: bytes = b"ZED",
    ip: bytes = b"192.0.2.10",
) -> bytes:
    fixed = struct.pack(
        "<7i",
        width,
        height,
        fps,
        bitrate,
        enable_mv_hevc,
        render_mode,
        port,
    )
    return b"\xca\xfe\x01" + fixed + bytes((len(camera),)) + camera + bytes((len(ip),)) + ip


def test_control_decoder_handles_fragmented_and_coalesced_packets() -> None:
    open_packet = _control_packet(b"OPEN_CAMERA", b"abc")
    close_packet = _control_packet(b"CLOSE_CAMERA", b"")
    decoder = ControlFrameDecoder()

    assert decoder.feed(open_packet[:3]) == ()
    assert decoder.feed(open_packet[3:] + close_packet) == (
        ("OPEN_CAMERA", b"abc"),
        ("CLOSE_CAMERA", b""),
    )


def test_control_decoder_rejects_malformed_inner_lengths() -> None:
    malformed_body = struct.pack("<I", 20) + b"OPEN"
    decoder = ControlFrameDecoder()

    with pytest.raises(ProtocolError, match="command length"):
        decoder.feed(struct.pack(">I", len(malformed_body)) + malformed_body)


def test_control_decoder_rejects_oversized_body_before_buffering_it() -> None:
    decoder = ControlFrameDecoder(max_body_bytes=64)

    with pytest.raises(ProtocolError, match="body size"):
        decoder.feed(struct.pack(">I", 65))


def test_camera_request_matches_vendor_little_endian_payload() -> None:
    assert parse_camera_request(_camera_payload()) == CameraRequest(
        width=1280,
        height=480,
        fps=30,
        bitrate=4_000_000,
        enable_mv_hevc=False,
        render_mode=0,
        port=12345,
        camera="ZED",
        ip="192.0.2.10",
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\xca\xfe\x02",
        b"\xca\xfe\x01" + b"\0" * 28,
        _camera_payload() + b"trailing",
    ],
)
def test_camera_request_rejects_truncated_unsupported_or_trailing_payload(
    payload: bytes,
) -> None:
    with pytest.raises(ProtocolError):
        parse_camera_request(payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 0},
        {"height": -1},
        {"fps": 0},
        {"bitrate": 0},
        {"enable_mv_hevc": 2},
        {"port": 65_536},
        {"camera": b"USB"},
        {"ip": b"not-an-ip"},
    ],
)
def test_camera_request_rejects_unsafe_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ProtocolError):
        parse_camera_request(_camera_payload(**overrides))


def test_video_access_unit_uses_big_endian_length_prefix() -> None:
    assert frame_video_access_unit(b"\x00\x00\x00\x01\x09\xf0") == (
        b"\x00\x00\x00\x06\x00\x00\x00\x01\x09\xf0"
    )


def test_video_access_unit_rejects_empty_payload() -> None:
    with pytest.raises(ProtocolError, match="empty"):
        frame_video_access_unit(b"")
