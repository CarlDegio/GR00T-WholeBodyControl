"""XRoboToolkit control server and one-session PICO video bridge."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import socket
import threading
import time
from typing import Any, Callable, Protocol

import numpy as np

from gear_sonic.pico_video.encoder import EncoderSettings, FfmpegH264Encoder
from gear_sonic.pico_video.frames import (
    FrameError,
    compose_mono_sbs,
    decode_jpeg_rgb,
    render_status_card,
)
from gear_sonic.pico_video.gateway_source import GatewayFrame
from gear_sonic.pico_video.protocol import (
    CameraRequest,
    ControlFrameDecoder,
    ProtocolError,
    frame_video_access_unit,
    parse_camera_request,
)


LOGGER = logging.getLogger(__name__)


class VideoSource(Protocol):
    status: str

    def poll(self) -> GatewayFrame | None: ...


class H264Encoder(Protocol):
    def write_frame(self, frame: np.ndarray) -> None: ...

    def iter_access_units(
        self, stop_event: threading.Event | None = None
    ) -> Any: ...

    def finish_input(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class BridgeSettings:
    gateway_endpoint: str
    stream: str = "camera_encoded/ego_view"
    control_host: str = "0.0.0.0"
    control_port: int = 13579
    width: int = 1280
    height: int = 480
    fps: int = 30
    bitrate: int = 4_000_000
    max_age_ms: float = 250.0
    stale_fps: float = 2.0
    encoder: str = "h264_nvenc"
    stats_interval_s: float = 5.0

    def __post_init__(self) -> None:
        if not self.gateway_endpoint:
            raise ValueError("SensorGateway endpoint cannot be empty")
        if self.stream != "camera_encoded/ego_view":
            raise ValueError(f"unsupported PICO video stream: {self.stream}")
        if not self.control_host:
            raise ValueError("control host cannot be empty")
        if self.control_port < 0 or self.control_port > 65_535:
            raise ValueError("control port is outside the TCP range")
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("output dimensions must be positive and even")
        if self.width != 1280 or self.height != 480:
            raise ValueError("the SONIC_HEAD profile requires 1280x480 output")
        if self.fps != 30:
            raise ValueError("the SONIC_HEAD profile requires 30 FPS")
        if self.bitrate <= 0 or self.bitrate > 100_000_000:
            raise ValueError("bridge bitrate is outside the supported range")
        if self.max_age_ms <= 0.0:
            raise ValueError("maximum source age must be positive")
        if self.stale_fps <= 0.0 or self.stale_fps > self.fps:
            raise ValueError("stale-card FPS is outside the supported range")
        if self.encoder not in {"h264_nvenc", "libx264"}:
            raise ValueError(f"unsupported encoder: {self.encoder}")
        if self.stats_interval_s < 0.0:
            raise ValueError("stats interval cannot be negative")


class LatestFrameSlot:
    """One persistent latest frame with a monotonic replacement version."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._version = 0
        self._frame: np.ndarray | None = None

    def put(self, frame: np.ndarray) -> int:
        with self._condition:
            self._version += 1
            self._frame = frame
            self._condition.notify_all()
            return self._version

    def get_after(
        self,
        version: int,
        *,
        timeout_s: float,
    ) -> tuple[int, np.ndarray | None]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            while self._version <= version:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return version, None
                self._condition.wait(remaining)
            return self._version, self._frame

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()


class VideoSession:
    """Own one outbound PICO socket, one encoder, and their worker threads."""

    def __init__(
        self,
        request: CameraRequest,
        settings: BridgeSettings,
        slot: LatestFrameSlot,
        *,
        encoder_factory: Callable[[EncoderSettings], H264Encoder],
    ) -> None:
        self.request = request
        self._settings = settings
        self._slot = slot
        self._encoder_factory = encoder_factory
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="pico-video-session",
            daemon=True,
        )
        self._socket_lock = threading.Lock()
        self._video_socket: socket.socket | None = None
        self._encoder: H264Encoder | None = None
        self.error = ""

    def start(self) -> None:
        self._thread.start()

    def _send_access_units(self, encoder: H264Encoder, connection: socket.socket) -> None:
        try:
            for access_unit in encoder.iter_access_units(self._stop):
                connection.sendall(frame_video_access_unit(access_unit))
        except (OSError, RuntimeError, ProtocolError) as exc:
            if not self._stop.is_set():
                self.error = str(exc)
                LOGGER.warning("PICO video sender stopped: %s", exc)
        finally:
            self._stop.set()
            self._slot.wake()

    def _run(self) -> None:
        encoder: H264Encoder | None = None
        sender: threading.Thread | None = None
        connection: socket.socket | None = None
        try:
            connection = socket.create_connection(
                (self.request.ip, self.request.port), timeout=2.0
            )
            connection.settimeout(2.0)
            with self._socket_lock:
                self._video_socket = connection
            encoder = self._encoder_factory(
                EncoderSettings(
                    width=self.request.width,
                    height=self.request.height,
                    fps=self.request.fps,
                    bitrate=self.request.bitrate,
                    encoder=self._settings.encoder,
                )
            )
            self._encoder = encoder
            sender = threading.Thread(
                target=self._send_access_units,
                args=(encoder, connection),
                name="pico-video-sender",
                daemon=True,
            )
            sender.start()
            version = 0
            while not self._stop.is_set():
                version, frame = self._slot.get_after(version, timeout_s=0.2)
                if frame is not None:
                    encoder.write_frame(frame)
        except (OSError, RuntimeError) as exc:
            if not self._stop.is_set():
                self.error = str(exc)
                LOGGER.warning("PICO video session stopped: %s", exc)
        finally:
            self._stop.set()
            if encoder is not None:
                encoder.finish_input()
            if sender is not None and sender is not threading.current_thread():
                sender.join(timeout=2.0)
            if encoder is not None:
                encoder.close()
            if connection is not None:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
            with self._socket_lock:
                self._video_socket = None

    def stop(self) -> None:
        self._stop.set()
        self._slot.wake()
        with self._socket_lock:
            connection = self._video_socket
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            encoder = self._encoder
            if encoder is not None:
                encoder.close()
            self._thread.join(timeout=1.0)


class PicoVideoBridge:
    """Bridge fresh SensorGateway JPEG frames into XRoboToolkit Remote Vision."""

    def __init__(
        self,
        settings: BridgeSettings,
        *,
        source: VideoSource,
        encoder_factory: Callable[[EncoderSettings], H264Encoder] = FfmpegH264Encoder,
        source_close: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self._source = source
        self._encoder_factory = encoder_factory
        self._source_close = source_close
        self._slot = LatestFrameSlot()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._listener_lock = threading.Lock()
        self._listener: socket.socket | None = None
        self._control_socket: socket.socket | None = None
        self._control_address: tuple[str, int] | None = None
        self._producer: threading.Thread | None = None
        self._session_lock = threading.Lock()
        self._session: VideoSession | None = None
        self._cleanup_lock = threading.Lock()
        self._cleaned = False
        self._frames_published = 0
        self._status_cards_published = 0

    @property
    def control_address(self) -> tuple[str, int]:
        if self._control_address is None:
            raise RuntimeError("PICO control server is not listening")
        return self._control_address

    def wait_until_ready(self, *, timeout_s: float) -> bool:
        return self._ready.wait(timeout=max(0.0, timeout_s))

    def _producer_loop(self) -> None:
        period_s = 1.0 / self.settings.fps
        status_period_s = 1.0 / self.settings.stale_fps
        next_status_at = 0.0
        next_stats_at = time.monotonic() + self.settings.stats_interval_s
        while not self._stop.is_set():
            started = time.monotonic()
            status = self._source.status
            try:
                source_frame = self._source.poll()
                status = self._source.status
                if source_frame is not None:
                    rgb = decode_jpeg_rgb(source_frame.jpeg)
                    composed = compose_mono_sbs(
                        rgb,
                        eye_width=self.settings.width // 2,
                        height=self.settings.height,
                    )
                    self._slot.put(composed)
                    self._frames_published += 1
                    next_status_at = started
                elif status != "READY" and started >= next_status_at:
                    self._slot.put(
                        render_status_card(
                            self.settings.width,
                            self.settings.height,
                            status,
                        )
                    )
                    self._status_cards_published += 1
                    next_status_at = started + status_period_s
            except FrameError as exc:
                status = "INVALID CAMERA FRAME"
                if started >= next_status_at:
                    self._slot.put(
                        render_status_card(
                            self.settings.width,
                            self.settings.height,
                            status,
                        )
                    )
                    self._status_cards_published += 1
                    next_status_at = started + status_period_s
                LOGGER.warning("PICO frame decode failed: %s", exc)
            except Exception as exc:
                status = "SENSOR SOURCE ERROR"
                if started >= next_status_at:
                    self._slot.put(
                        render_status_card(
                            self.settings.width,
                            self.settings.height,
                            status,
                        )
                    )
                    self._status_cards_published += 1
                    next_status_at = started + status_period_s
                LOGGER.exception("PICO source polling failed: %s", exc)

            if self.settings.stats_interval_s and started >= next_stats_at:
                LOGGER.info(
                    "PICO bridge frames=%d status_cards=%d source_status=%s",
                    self._frames_published,
                    self._status_cards_published,
                    status,
                )
                next_stats_at = started + self.settings.stats_interval_s
            self._stop.wait(max(0.0, period_s - (time.monotonic() - started)))

    def _validate_profile(self, request: CameraRequest) -> None:
        if (
            request.width != self.settings.width
            or request.height != self.settings.height
            or request.fps != self.settings.fps
        ):
            raise ProtocolError(
                "OPEN_CAMERA does not match SONIC_HEAD 1280x480@30 profile"
            )
        if request.bitrate > 100_000_000:
            raise ProtocolError("OPEN_CAMERA bitrate exceeds the safety limit")

    def _stop_session(self) -> None:
        with self._session_lock:
            session, self._session = self._session, None
        if session is not None:
            session.stop()

    def _replace_session(self, request: CameraRequest) -> None:
        self._stop_session()
        session = VideoSession(
            request,
            self.settings,
            self._slot,
            encoder_factory=self._encoder_factory,
        )
        with self._session_lock:
            self._session = session
        session.start()

    def _handle_control(self, connection: socket.socket) -> None:
        decoder = ControlFrameDecoder()
        connection.settimeout(0.2)
        try:
            while not self._stop.is_set():
                try:
                    chunk = connection.recv(64 * 1024)
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        return
                    raise
                if not chunk:
                    return
                try:
                    messages = decoder.feed(chunk)
                except ProtocolError as exc:
                    LOGGER.warning("Discarding malformed PICO control stream: %s", exc)
                    return
                for command, payload in messages:
                    if command == "OPEN_CAMERA":
                        try:
                            request = parse_camera_request(payload)
                            self._validate_profile(request)
                        except ProtocolError as exc:
                            LOGGER.warning("Ignoring invalid OPEN_CAMERA: %s", exc)
                            continue
                        self._replace_session(request)
                    elif command == "CLOSE_CAMERA":
                        self._stop_session()
                    else:
                        LOGGER.info("Ignoring unknown PICO control command: %s", command)
        finally:
            self._stop_session()

    def serve_forever(self) -> None:
        """Listen until ``stop`` is called; source/session errors remain isolated."""

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.settings.control_host, self.settings.control_port))
        listener.listen(2)
        listener.settimeout(0.2)
        with self._listener_lock:
            self._listener = listener
            bound_host, bound_port = listener.getsockname()[:2]
            advertised_host = (
                "127.0.0.1" if self.settings.control_host == "0.0.0.0" else bound_host
            )
            self._control_address = (advertised_host, bound_port)
        self._producer = threading.Thread(
            target=self._producer_loop,
            name="pico-frame-producer",
            daemon=True,
        )
        self._producer.start()
        self._ready.set()
        LOGGER.info("PICO control server listening on %s:%d", *self.control_address)
        try:
            while not self._stop.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                with connection:
                    with self._listener_lock:
                        self._control_socket = connection
                    try:
                        self._handle_control(connection)
                    finally:
                        with self._listener_lock:
                            if self._control_socket is connection:
                                self._control_socket = None
        finally:
            self.stop()

    def stop(self) -> None:
        """Request bounded shutdown of every resource owned by this bridge."""

        with self._cleanup_lock:
            if self._cleaned:
                return
            self._cleaned = True
        self._stop.set()
        self._slot.wake()
        with self._listener_lock:
            listener, self._listener = self._listener, None
            control, self._control_socket = self._control_socket, None
        for connection in (control, listener):
            if connection is not None:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
        self._stop_session()
        if self._source_close is not None:
            self._source_close()
        producer = self._producer
        if (
            producer is not None
            and producer.is_alive()
            and producer is not threading.current_thread()
        ):
            producer.join(timeout=2.0)
