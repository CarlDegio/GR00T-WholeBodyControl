"""Robot-free camera publisher using the production SONIC camera wire schema."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
import time

from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer
from gear_sonic.pico_video.frames import render_test_card


@dataclass(frozen=True)
class MockCameraSettings:
    port: int = 5555
    width: int = 640
    height: int = 480
    fps: int = 30
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if self.port <= 0 or self.port > 65_535:
            raise ValueError("mock camera port is outside the TCP range")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("mock camera dimensions must be positive")
        if self.fps <= 0 or self.fps > 240:
            raise ValueError("mock camera FPS is outside the supported range")
        if self.jpeg_quality < 1 or self.jpeg_quality > 100:
            raise ValueError("mock camera JPEG quality must be between 1 and 100")


class MockCameraPublisher:
    """Continuously publish animated ego and wrist camera test cards."""

    def __init__(self, settings: MockCameraSettings) -> None:
        self.settings = settings
        self._server = SensorServer()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._frame_number = 0

    def wait_until_ready(self, *, timeout_s: float) -> bool:
        return self._ready.wait(max(0.0, timeout_s))

    def publish_once(self) -> None:
        wall_time = datetime.now(timezone.utc)
        names = ("ego_view", "left_wrist", "right_wrist")
        images = {
            name: render_test_card(
                self.settings.width,
                self.settings.height,
                frame_number=self._frame_number + offset,
                wall_time=wall_time,
            )
            for name, offset in zip(names, (0, 1_000_000, 2_000_000), strict=True)
        }
        timestamp = time.time()
        schema = ImageMessageSchema(
            timestamps={name: timestamp for name in names},
            images=images,
            image_shapes={
                name: [self.settings.height, self.settings.width, 3]
                for name in names
            },
        )
        self._server.send_message(
            schema.serialize(jpeg_quality=self.settings.jpeg_quality)
        )
        self._frame_number += 1

    def run(self) -> None:
        self._server.start_server(self.settings.port)
        self._ready.set()
        period_s = 1.0 / self.settings.fps
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                self.publish_once()
                deadline += period_s
                delay = deadline - time.monotonic()
                if delay <= 0.0:
                    deadline = time.monotonic()
                    continue
                self._stop.wait(delay)
        finally:
            self._server.stop_server()

    def stop(self) -> None:
        self._stop.set()
