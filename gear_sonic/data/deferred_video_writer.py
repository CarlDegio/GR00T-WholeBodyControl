"""Deferred video writer for encoded camera payloads.

This writer keeps compressed camera frames in memory during data collection and
only decodes/encodes the final MP4 when an episode is saved.  It is intended for
multi-camera collection where real-time H.264 encoding can steal CPU from the
control/data loop.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import av
import cv2
import msgpack_numpy as m
import numpy as np

from gear_sonic.camera.sensor_server import ImageUtils


class DeferredEncodedVideoWriter:
    """Store encoded image payloads in memory and write video on ``stop()``."""

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "h264",
        encoder_threads: int = 16,
    ):
        self.output_path = str(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.encoder_threads = encoder_threads
        self.frames: list[Any] = []

        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    def add_frame(self, frame: Any) -> None:
        self.frames.append(frame)

    def _decode_frame(self, frame: Any) -> np.ndarray:
        if isinstance(frame, bytes | bytearray):
            mat = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
            if mat is None:
                raise ValueError("Failed to decode JPEG bytes")
            return mat[..., ::-1]

        if isinstance(frame, str):
            mat = ImageUtils.decode_image(frame)
            if mat is None:
                raise ValueError("Failed to decode base64 JPEG image")
            return mat

        if isinstance(frame, np.ndarray):
            return frame

        if isinstance(frame, dict) and b"nd" in frame:
            decoded = m.decode(frame)
            if not isinstance(decoded, np.ndarray):
                raise ValueError(f"Decoded msgpack-numpy payload is not an ndarray: {type(decoded)}")
            return decoded

        raise TypeError(f"Unsupported deferred video frame type: {type(frame)}")

    def _assert_dimensions(self, frame: np.ndarray) -> None:
        assert (
            frame.shape[1] == self.width and frame.shape[0] == self.height
        ), (
            f"Incorrect frame dimensions. Input dimensions: {frame.shape[1]}x{frame.shape[0]}. "
            f"Expected dimensions: {self.width}x{self.height}"
        )

    def _configure_encoder_threads(self, stream) -> None:
        if self.encoder_threads <= 0:
            return
        try:
            stream.codec_context.thread_count = self.encoder_threads
        except Exception as exc:
            print(f"[DeferredVideoWriter] Could not set encoder thread_count: {exc}")
        try:
            stream.codec_context.options = {
                **dict(getattr(stream.codec_context, "options", {}) or {}),
                "threads": str(self.encoder_threads),
            }
        except Exception as exc:
            print(f"[DeferredVideoWriter] Could not set encoder thread option: {exc}")

    def stop(self) -> str:
        """Decode cached frames, encode the MP4, and return the output path."""
        container = av.open(self.output_path, mode="w")
        stream = container.add_stream(self.codec, rate=self.fps)
        stream.width = self.width
        stream.height = self.height
        self._configure_encoder_threads(stream)

        print(
            f"[DeferredVideoWriter] Encoding {len(self.frames)} frames to "
            f"{self.output_path} with codec={self.codec}, threads={self.encoder_threads}"
        )

        try:
            first_frame = True
            for encoded_frame in self.frames:
                frame = self._decode_frame(encoded_frame)
                self._assert_dimensions(frame)
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

                if first_frame:
                    stderr_fd = sys.stderr.fileno()
                    old_stderr = os.dup(stderr_fd)
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, stderr_fd)
                    try:
                        packets = stream.encode(video_frame)
                        for packet in packets:
                            container.mux(packet)
                    finally:
                        os.dup2(old_stderr, stderr_fd)
                        os.close(old_stderr)
                        os.close(devnull)
                    first_frame = False
                else:
                    packets = stream.encode(video_frame)
                    for packet in packets:
                        container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
            self.frames.clear()

        return self.output_path

    def cancel(self) -> None:
        self.frames.clear()
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
