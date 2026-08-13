from __future__ import annotations

from io import BytesIO
import threading

import numpy as np
import pytest

from gear_sonic.pico_video.encoder import (
    AnnexBAccessUnitParser,
    EncoderError,
    EncoderSettings,
    FfmpegH264Encoder,
    ffmpeg_command,
)


def test_annex_b_parser_groups_arbitrary_chunks_by_aud() -> None:
    first = b"\x00\x00\x00\x01\x09\xf0\x00\x00\x01\x65\xaa"
    second = b"\x00\x00\x01\x09\xf0\x00\x00\x01\x41\xbb"
    parser = AnnexBAccessUnitParser()

    assert parser.feed((first + second)[:9]) == ()
    assert parser.feed((first + second)[9:14]) == ()
    assert parser.feed((first + second)[14:]) == (first,)
    assert parser.flush() == (second,)


def test_annex_b_parser_keeps_parameter_sets_with_first_access_unit() -> None:
    headers = b"\x00\x00\x00\x01\x67\x64\x00\x1f\x00\x00\x01\x68\xee"
    first = b"\x00\x00\x01\x09\xf0\x00\x00\x01\x65\xaa"
    second = b"\x00\x00\x00\x01\x09\xf0\x00\x00\x01\x41\xbb"
    parser = AnnexBAccessUnitParser()

    assert parser.feed(headers[:2]) == ()
    assert parser.feed(headers[2:] + first + second) == (headers + first,)
    assert parser.flush() == (second,)


def test_annex_b_parser_rejects_garbage_and_unbounded_input() -> None:
    with pytest.raises(EncoderError, match="before Annex-B"):
        AnnexBAccessUnitParser().feed(b"garbage\x00\x00\x01\x09\xf0")

    parser = AnnexBAccessUnitParser(max_buffer_bytes=12)
    with pytest.raises(EncoderError, match="buffer limit"):
        parser.feed(b"\x00\x00\x01\x09\xf0" + b"x" * 8)


def test_annex_b_parser_rejects_stream_without_access_unit_delimiters() -> None:
    parser = AnnexBAccessUnitParser()
    parser.feed(b"\x00\x00\x00\x01\x67\x64\x00\x1f")

    with pytest.raises(EncoderError, match="AUD"):
        parser.flush()


def test_nvenc_command_requests_annex_b_aud_and_no_b_frames() -> None:
    command = ffmpeg_command(
        EncoderSettings(
            width=1280,
            height=480,
            fps=30,
            bitrate=4_000_000,
            encoder="h264_nvenc",
        )
    )

    assert command[:4] == ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    assert command[-3:] == ["-f", "h264", "pipe:1"]
    assert command[command.index("-bf") : command.index("-bf") + 2] == ["-bf", "0"]
    assert "aud=insert" in command[command.index("-bsf:v") + 1]
    assert command[command.index("-g") + 1] == "30"
    assert "h264_nvenc" in command


def test_libx264_command_enables_repeated_headers_and_zero_latency() -> None:
    command = ffmpeg_command(
        EncoderSettings(
            width=1280,
            height=480,
            fps=30,
            bitrate=4_000_000,
            encoder="libx264",
        )
    )

    assert "libx264" in command
    assert "zerolatency" in command
    assert "repeat-headers=1" in command[command.index("-x264-params") + 1]


@pytest.mark.parametrize(
    "settings",
    [
        EncoderSettings(width=1279, height=480, fps=30, bitrate=4_000_000),
        EncoderSettings(width=1280, height=0, fps=30, bitrate=4_000_000),
        EncoderSettings(width=1280, height=480, fps=0, bitrate=4_000_000),
        EncoderSettings(width=1280, height=480, fps=30, bitrate=0),
        EncoderSettings(
            width=1280,
            height=480,
            fps=30,
            bitrate=4_000_000,
            encoder="unknown",
        ),
    ],
)
def test_encoder_settings_reject_unsupported_values(settings: EncoderSettings) -> None:
    with pytest.raises(ValueError):
        ffmpeg_command(settings)


class _FakeProcess:
    def __init__(self, *, returncode: int | None, stderr: bytes = b"") -> None:
        self.stdin = BytesIO()
        self.stdout = BytesIO()
        self.stderr = BytesIO(stderr)
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def test_encoder_rejects_wrong_frame_contract_before_writing() -> None:
    process = _FakeProcess(returncode=None)
    encoder = FfmpegH264Encoder(
        EncoderSettings(width=1280, height=480, fps=30, bitrate=4_000_000),
        process_factory=lambda *args, **kwargs: process,
    )
    try:
        with pytest.raises(EncoderError, match="1280x480 RGB uint8"):
            encoder.write_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        assert process.stdin.getvalue() == b""
    finally:
        encoder.close()


def test_encoder_reports_early_ffmpeg_exit_with_stderr() -> None:
    process = _FakeProcess(returncode=2, stderr=b"encoder unavailable")
    encoder = FfmpegH264Encoder(
        EncoderSettings(width=1280, height=480, fps=30, bitrate=4_000_000),
        process_factory=lambda *args, **kwargs: process,
    )
    stderr_read = threading.Event()
    try:
        encoder.wait_for_stderr_drain(timeout_s=1.0)
        stderr_read.set()
        with pytest.raises(EncoderError, match="encoder unavailable"):
            encoder.write_frame(np.zeros((480, 1280, 3), dtype=np.uint8))
    finally:
        encoder.close()
    assert stderr_read.is_set()
