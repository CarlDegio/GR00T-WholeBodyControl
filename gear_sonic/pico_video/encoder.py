"""Low-latency FFmpeg H.264 encoding and Annex-B access-unit parsing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import select
import subprocess
import threading
from typing import Any, Callable, Iterator

import numpy as np


class EncoderError(RuntimeError):
    """The H.264 stream or its FFmpeg process violated the encoder contract."""


@dataclass(frozen=True)
class EncoderSettings:
    width: int
    height: int
    fps: int
    bitrate: int
    encoder: str = "h264_nvenc"
    ffmpeg_binary: str = "ffmpeg"


def _validate_settings(settings: EncoderSettings) -> None:
    if settings.width <= 0 or settings.height <= 0:
        raise ValueError("encoder dimensions must be positive")
    if settings.width % 2 or settings.height % 2:
        raise ValueError("encoder dimensions must be even")
    if settings.width > 8192 or settings.height > 8192:
        raise ValueError("encoder dimensions exceed the supported limit")
    if settings.fps <= 0 or settings.fps > 240:
        raise ValueError("encoder FPS is outside the supported range")
    if settings.bitrate <= 0 or settings.bitrate > 100_000_000:
        raise ValueError("encoder bitrate is outside the supported range")
    if settings.encoder not in {"h264_nvenc", "libx264"}:
        raise ValueError(f"unsupported encoder: {settings.encoder}")
    if not settings.ffmpeg_binary:
        raise ValueError("FFmpeg binary cannot be empty")


def ffmpeg_command(settings: EncoderSettings) -> list[str]:
    """Build a fixed-cadence RGB-to-Annex-B low-latency encoder command."""

    _validate_settings(settings)
    if settings.encoder == "h264_nvenc":
        codec = [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-tune",
            "ull",
            "-rc",
            "cbr",
            "-rc-lookahead",
            "0",
            "-forced-idr",
            "1",
        ]
    else:
        codec = [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-x264-params",
            "repeat-headers=1:aud=1:scenecut=0",
        ]
    return [
        settings.ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{settings.width}x{settings.height}",
        "-r",
        str(settings.fps),
        "-i",
        "pipe:0",
        "-an",
        *codec,
        "-b:v",
        str(settings.bitrate),
        "-maxrate",
        str(settings.bitrate),
        "-bufsize",
        str(settings.bitrate),
        "-g",
        str(settings.fps),
        "-bf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-bsf:v",
        "h264_metadata=aud=insert",
        "-f",
        "h264",
        "pipe:1",
    ]


def _annex_b_nal_units(data: bytearray) -> list[tuple[int, int]]:
    """Return ``(start_offset, nal_type)`` for complete Annex-B NAL headers."""

    units: list[tuple[int, int]] = []
    offset = 0
    size = len(data)
    while offset + 3 <= size:
        start_size = 0
        if offset + 4 <= size and data[offset : offset + 4] == b"\x00\x00\x00\x01":
            start_size = 4
        elif data[offset : offset + 3] == b"\x00\x00\x01":
            start_size = 3
        if start_size:
            header_offset = offset + start_size
            if header_offset < size:
                units.append((offset, data[header_offset] & 0x1F))
            offset += start_size
        else:
            offset += 1
    return units


class AnnexBAccessUnitParser:
    """Group an Annex-B byte stream into access units using AUD NAL boundaries."""

    def __init__(self, *, max_buffer_bytes: int = 8 * 1024 * 1024) -> None:
        if max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be positive")
        self._buffer = bytearray()
        self._max_buffer_bytes = int(max_buffer_bytes)

    def feed(self, chunk: bytes) -> tuple[bytes, ...]:
        if not isinstance(chunk, bytes | bytearray | memoryview):
            raise TypeError("Annex-B chunk must be bytes-like")
        self._buffer.extend(chunk)
        if len(self._buffer) > self._max_buffer_bytes:
            raise EncoderError("Annex-B stream exceeded the access-unit buffer limit")

        units = _annex_b_nal_units(self._buffer)
        if units and units[0][0] > 0 and any(self._buffer[: units[0][0]]):
            raise EncoderError("garbage appeared before Annex-B start code")
        aud_offsets = [offset for offset, nal_type in units if nal_type == 9]
        if len(aud_offsets) < 2:
            return ()

        output: list[bytes] = []
        start = 0
        for next_aud in aud_offsets[1:]:
            access_unit = bytes(self._buffer[start:next_aud])
            if not access_unit:
                raise EncoderError("empty Annex-B access unit")
            output.append(access_unit)
            start = next_aud
        del self._buffer[:start]
        return tuple(output)

    def flush(self) -> tuple[bytes, ...]:
        if not self._buffer:
            return ()
        units = _annex_b_nal_units(self._buffer)
        if not any(nal_type == 9 for _, nal_type in units):
            raise EncoderError("Annex-B stream ended without an AUD")
        payload = bytes(self._buffer)
        self._buffer.clear()
        return (payload,)


class FfmpegH264Encoder:
    """Own one FFmpeg subprocess that accepts RGB frames and yields H.264 AUs."""

    def __init__(
        self,
        settings: EncoderSettings,
        *,
        process_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.settings = settings
        command = ffmpeg_command(settings)
        self._process = process_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if (
            self._process.stdin is None
            or self._process.stdout is None
            or self._process.stderr is None
        ):
            raise EncoderError("FFmpeg process did not expose all required pipes")
        self._write_lock = threading.Lock()
        self._write_cancel = threading.Event()
        self._input_closed = False
        self._closing = False
        try:
            self._stdin_fd: int | None = self._process.stdin.fileno()
            os.set_blocking(self._stdin_fd, False)
        except (AttributeError, OSError, ValueError):
            self._stdin_fd = None
        self._stderr_chunks: deque[bytes] = deque(maxlen=32)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="pico-ffmpeg-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        while True:
            chunk = self._process.stderr.read(4096)
            if not chunk:
                return
            self._stderr_chunks.append(bytes(chunk))

    def wait_for_stderr_drain(self, *, timeout_s: float) -> None:
        """Wait briefly for diagnostics after an observed child-process exit."""

        self._stderr_thread.join(timeout=max(0.0, timeout_s))

    def _diagnostic(self) -> str:
        self.wait_for_stderr_drain(timeout_s=0.1)
        raw = b"".join(self._stderr_chunks)[-8192:]
        return raw.decode("utf-8", errors="replace").strip()

    def _raise_if_exited(self) -> None:
        returncode = self._process.poll()
        if returncode is not None:
            diagnostic = self._diagnostic() or "no FFmpeg diagnostic"
            raise EncoderError(f"FFmpeg exited with code {returncode}: {diagnostic}")

    def write_frame(self, frame: np.ndarray) -> None:
        expected_shape = (self.settings.height, self.settings.width, 3)
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.shape != expected_shape
        ):
            raise EncoderError(
                f"encoder frame must be {self.settings.width}x{self.settings.height} RGB uint8"
            )
        self._raise_if_exited()
        contiguous = np.ascontiguousarray(frame)
        remaining = memoryview(contiguous).cast("B")
        with self._write_lock:
            if self._input_closed:
                raise EncoderError("FFmpeg input is closed")
            try:
                while remaining:
                    if self._write_cancel.is_set():
                        raise EncoderError("FFmpeg frame write was cancelled")
                    if self._stdin_fd is None:
                        written = self._process.stdin.write(remaining)
                    else:
                        _, writable, _ = select.select(
                            [],
                            [self._stdin_fd],
                            [],
                            0.05,
                        )
                        if not writable:
                            self._raise_if_exited()
                            continue
                        try:
                            written = os.write(self._stdin_fd, remaining)
                        except BlockingIOError:
                            continue
                    if written is None or written <= 0:
                        raise BrokenPipeError("FFmpeg stdin accepted no data")
                    remaining = remaining[written:]
            except (BrokenPipeError, OSError) as exc:
                diagnostic = self._diagnostic() or str(exc)
                raise EncoderError(f"FFmpeg frame write failed: {diagnostic}") from exc

    def finish_input(self) -> None:
        """Close FFmpeg stdin so a finite stream can drain and exit normally."""

        self._write_cancel.set()
        with self._write_lock:
            if self._input_closed:
                return
            self._input_closed = True
            try:
                self._process.stdin.close()
            except OSError:
                pass

    def iter_access_units(
        self,
        stop_event: threading.Event | None = None,
    ) -> Iterator[bytes]:
        """Yield complete Annex-B access units until EOF or requested shutdown."""

        parser = AnnexBAccessUnitParser()
        while stop_event is None or not stop_event.is_set():
            chunk = self._process.stdout.read(64 * 1024)
            if not chunk:
                break
            yield from parser.feed(chunk)
        if stop_event is None or not stop_event.is_set():
            yield from parser.flush()
            try:
                returncode = self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired as exc:
                raise EncoderError("FFmpeg stdout closed but the process did not exit") from exc
            if returncode != 0 and not self._closing:
                diagnostic = self._diagnostic() or "no FFmpeg diagnostic"
                raise EncoderError(
                    f"FFmpeg exited with code {returncode}: {diagnostic}"
                )

    def close(self) -> None:
        """Stop only this encoder's subprocess with a bounded grace period."""

        if self._closing:
            return
        self._closing = True
        self._write_cancel.set()
        self.finish_input()
        try:
            self._process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.0)
        try:
            self._process.stdout.close()
        except OSError:
            pass
        try:
            self._process.stderr.close()
        except OSError:
            pass
        self._stderr_thread.join(timeout=1.0)

    def __enter__(self) -> "FfmpegH264Encoder":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()
