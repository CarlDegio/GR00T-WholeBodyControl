"""Wire contracts used by XRoboToolkit Remote Vision."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import struct


class ProtocolError(ValueError):
    """A Remote Vision packet violates the supported wire contract."""


@dataclass(frozen=True)
class CameraRequest:
    """Validated payload from an XRoboToolkit ``OPEN_CAMERA`` command."""

    width: int
    height: int
    fps: int
    bitrate: int
    enable_mv_hevc: bool
    render_mode: int
    port: int
    camera: str
    ip: str


def _parse_control_body(body: bytes) -> tuple[str, bytes]:
    if len(body) < 8:
        raise ProtocolError("control body is too short")

    command_length = struct.unpack_from("<i", body, 0)[0]
    if command_length <= 0 or command_length > len(body) - 8:
        raise ProtocolError(f"invalid command length: {command_length}")

    command_end = 4 + command_length
    try:
        command = body[4:command_end].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProtocolError("control command is not valid UTF-8") from exc

    data_length = struct.unpack_from("<i", body, command_end)[0]
    if data_length < 0:
        raise ProtocolError(f"invalid data length: {data_length}")
    data_start = command_end + 4
    if data_start + data_length != len(body):
        raise ProtocolError(
            f"invalid data length: {data_length} for {len(body) - data_start} bytes"
        )
    return command, body[data_start:]


class ControlFrameDecoder:
    """Incrementally decode length-framed control messages from a TCP stream."""

    def __init__(self, *, max_body_bytes: int = 2 * 1024 * 1024) -> None:
        if max_body_bytes < 8:
            raise ValueError("max_body_bytes must permit a control envelope")
        self._buffer = bytearray()
        self._max_body_bytes = int(max_body_bytes)

    def feed(self, chunk: bytes) -> tuple[tuple[str, bytes], ...]:
        if not isinstance(chunk, bytes | bytearray | memoryview):
            raise TypeError("control chunk must be bytes-like")
        self._buffer.extend(chunk)
        messages: list[tuple[str, bytes]] = []
        while len(self._buffer) >= 4:
            body_size = struct.unpack_from(">I", self._buffer, 0)[0]
            if body_size < 8 or body_size > self._max_body_bytes:
                raise ProtocolError(f"invalid control body size: {body_size}")
            packet_size = 4 + body_size
            if len(self._buffer) < packet_size:
                break
            body = bytes(self._buffer[4:packet_size])
            del self._buffer[:packet_size]
            messages.append(_parse_control_body(body))
        return tuple(messages)


def _read_compact_string(payload: bytes, offset: int, name: str) -> tuple[str, int]:
    if offset >= len(payload):
        raise ProtocolError(f"camera request omitted {name} length")
    length = payload[offset]
    offset += 1
    end = offset + length
    if length == 0 or end > len(payload):
        raise ProtocolError(f"camera request has invalid {name} length")
    try:
        value = payload[offset:end].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"camera request {name} is not valid UTF-8") from exc
    return value, end


def parse_camera_request(payload: bytes) -> CameraRequest:
    """Parse the version-one XRoboToolkit CameraRequestSerializer payload."""

    fixed_size = 3 + 7 * 4
    if len(payload) < fixed_size:
        raise ProtocolError("camera request is truncated")
    if payload[:2] != b"\xca\xfe":
        raise ProtocolError("camera request has invalid magic")
    if payload[2] != 1:
        raise ProtocolError(f"unsupported camera request version: {payload[2]}")

    (
        width,
        height,
        fps,
        bitrate,
        enable_mv_hevc,
        render_mode,
        port,
    ) = struct.unpack_from("<7i", payload, 3)
    camera, offset = _read_compact_string(payload, fixed_size, "camera")
    ip, offset = _read_compact_string(payload, offset, "IP")
    if offset != len(payload):
        raise ProtocolError("camera request has trailing bytes")

    if width <= 0 or height <= 0 or width > 8192 or height > 8192:
        raise ProtocolError("camera dimensions are outside the supported range")
    if width % 2 or height % 2 or width * height > 33_554_432:
        raise ProtocolError("camera dimensions must be even and safely bounded")
    if fps <= 0 or fps > 240:
        raise ProtocolError("camera FPS is outside the supported range")
    if bitrate <= 0 or bitrate > 100_000_000:
        raise ProtocolError("camera bitrate is outside the supported range")
    if enable_mv_hevc not in (0, 1):
        raise ProtocolError("camera HEVC flag must be zero or one")
    if render_mode < 0 or render_mode > 16:
        raise ProtocolError("camera render mode is outside the supported range")
    if port <= 0 or port > 65_535:
        raise ProtocolError("camera video port is outside the supported range")
    if camera != "ZED":
        raise ProtocolError(f"unsupported camera type: {camera}")
    try:
        ipaddress.ip_address(ip)
    except ValueError as exc:
        raise ProtocolError(f"camera request has invalid IP address: {ip}") from exc

    return CameraRequest(
        width=width,
        height=height,
        fps=fps,
        bitrate=bitrate,
        enable_mv_hevc=bool(enable_mv_hevc),
        render_mode=render_mode,
        port=port,
        camera=camera,
        ip=ip,
    )


def frame_video_access_unit(payload: bytes) -> bytes:
    """Prefix one H.264 access unit with the PICO receiver's network length."""

    if not payload:
        raise ProtocolError("cannot frame an empty video access unit")
    if len(payload) > 0xFFFF_FFFF:
        raise ProtocolError("video access unit exceeds the framing limit")
    return struct.pack(">I", len(payload)) + payload
