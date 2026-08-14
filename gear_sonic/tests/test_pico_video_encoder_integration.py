from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import threading

import numpy as np
import pytest

from gear_sonic.pico_video.encoder import EncoderSettings, FfmpegH264Encoder
from gear_sonic.pico_video.frames import compose_mono_sbs, render_test_card


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are not installed",
)
def test_libx264_stream_decodes_as_identical_1280x480_mono_sbs(
    tmp_path: Path,
) -> None:
    encoder = FfmpegH264Encoder(
        EncoderSettings(
            width=1280,
            height=480,
            fps=30,
            bitrate=4_000_000,
            encoder="libx264",
        )
    )
    access_units: list[bytes] = []
    errors: list[BaseException] = []

    def read_output() -> None:
        try:
            access_units.extend(encoder.iter_access_units())
        except BaseException as exc:
            errors.append(exc)

    reader = threading.Thread(target=read_output, name="test-h264-reader")
    reader.start()
    try:
        for frame_number in range(3):
            source = render_test_card(640, 480, frame_number=frame_number)
            encoder.write_frame(
                compose_mono_sbs(source, eye_width=640, height=480)
            )
        encoder.finish_input()
        reader.join(timeout=10.0)
        assert not reader.is_alive()
        assert errors == []
    finally:
        encoder.close()

    assert len(access_units) == 3
    stream_path = tmp_path / "pico-test.h264"
    stream_path.write_bytes(b"".join(access_units))
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
            "-of",
            "json",
            str(stream_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream == {"codec_name": "h264", "width": 1280, "height": 480}

    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "h264",
            "-i",
            str(stream_path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    frame = np.frombuffer(decoded, dtype=np.uint8).reshape(480, 1280, 3)
    mean_absolute_difference = np.abs(
        frame[:, :640].astype(np.int16) - frame[:, 640:].astype(np.int16)
    ).mean()
    assert mean_absolute_difference < 3.0
