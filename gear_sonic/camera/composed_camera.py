"""Composed camera server — orchestrates multiple camera drivers.

Runs each camera in its own thread with staggered initialization and
automatic reconnection.  Publishes all frames as a single merged
``ImageMessageSchema`` payload over ZMQ.

Usage (on robot)::

    python -m gear_sonic.camera.composed_camera \\
        --ego-view-camera oak \\
        --ego-view-device-id 18443010E1ABC12300 \\
        --port 5555

Supported camera types: ``oak``, ``oak_mono``, ``realsense``, ``orbbec``,
``usb``, or a path to an ``.mp4`` file for replay testing.

Run ``python -m gear_sonic.camera.composed_camera --help`` for all options.
"""

from collections import deque
from dataclasses import dataclass
import queue
import threading
import time
from typing import Any

import cv2  # noqa: F401 — imported early to avoid TSL segfault with camera SDKs
import numpy as np

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorClient,
    SensorServer,
)


def read_qr_code(data):
    """Measure end-to-end latency by decoding QR-code timestamps."""
    current_time = time.monotonic()
    detector = cv2.QRCodeDetector()
    for key, img in data["images"].items():
        decoded_time, bbox, _ = detector.detectAndDecode(img)
        if bbox is not None and decoded_time:
            print(f"{key} latency: {(current_time - float(decoded_time)) * 1e3:.1f} ms")
        else:
            print(f"{key} QR code not detected.")


@dataclass
class ComposedCameraConfig:
    """Camera configuration for the composed camera server."""

    ego_view_camera: str | None = "oak"
    """Camera type for ego view: oak, oak_mono, realsense, zed, usb, or None."""

    ego_view_device_id: str | None = None
    """Device ID for ego view camera (OAK MxID, RealSense serial, USB /dev/video index)."""

    head_camera: str | None = None
    """Camera type for head view."""

    head_device_id: str | None = None
    """Device ID for head camera."""

    chest_camera: str | None = None
    """Camera type for chest view."""

    chest_device_id: str | None = None
    """Device ID for chest camera."""

    left_wrist_camera: str | None = None
    """Camera type for left wrist view."""

    left_wrist_device_id: str | None = None
    """Device ID for left wrist camera."""

    right_wrist_camera: str | None = None
    """Camera type for right wrist view."""

    right_wrist_device_id: str | None = None
    """Device ID for right wrist camera."""

    fps: int = 30
    """Publish rate.  OAK cameras run at 30 FPS; lower values add latency."""

    realsense_enable_depth: bool = False
    """Whether RealSense cameras should publish depth alongside color."""

    orbbec_enable_depth: bool = False
    """Whether Orbbec cameras should publish aligned depth alongside color."""

    realsense_depth_mount: str = CameraMountPosition.CHEST_VIEW.value
    """RealSense mount that publishes aligned depth when depth is enabled."""
    realsense_depth_mounts: tuple[str, ...] = ()
    """RealSense mounts that publish aligned depth; overrides the singular setting."""

    publish_camera_info: bool = False
    """Include camera calibration metadata in composed messages."""


    run_as_server: bool = True
    """Run as ZMQ PUB server (set False for in-process usage)."""

    server: bool = True
    """Alias for run_as_server kept for backward compatibility."""

    port: int = 5555
    """ZMQ port for server / client communication."""

    test_latency: bool = False
    """Decode QR-code timestamps in each frame to measure latency."""

    queue_size: int = 1
    """Per-camera latest-frame buffer depth. Keep this at 1 to retain only the newest frame."""

    stale_frame_warning_count: int = 10
    """Warn after this many server reads reuse the same per-camera frame timestamp."""

    use_mjpeg: bool = False
    """Use on-device MJPEG encoding on OAK cameras to reduce USB bandwidth."""

    mjpeg_quality: int = 80
    """MJPEG quality 1-100 (only when use_mjpeg=True)."""

    def __post_init__(self):
        self.run_as_server = self.server
        valid_depth_mounts = {
            CameraMountPosition.EGO_VIEW.value,
            CameraMountPosition.HEAD.value,
            CameraMountPosition.CHEST_VIEW.value,
        }
        requested_depth_mounts = (
            self.realsense_depth_mounts
            if self.realsense_depth_mounts
            else (self.realsense_depth_mount,)
        )
        invalid = sorted(set(requested_depth_mounts) - valid_depth_mounts)
        if invalid:
            choices = ", ".join(sorted(valid_depth_mounts))
            raise ValueError(
                f"realsense depth mounts must be selected from {choices}, "
                f"got {invalid!r}"
            )

    @property
    def enabled_realsense_depth_mounts(self) -> frozenset[str]:
        mounts = self.realsense_depth_mounts or (self.realsense_depth_mount,)
        return frozenset(mounts)


class ComposedCameraSensor(Sensor, SensorServer):
    """Multi-camera orchestrator with per-camera threads and auto-reconnect."""

    def __init__(self, config: ComposedCameraConfig):
        self.config = config
        self.camera_queues: dict[str, queue.Queue] = {}
        self.camera_threads: dict[str, threading.Thread] = {}
        self.shutdown_events: dict[str, threading.Event] = {}
        self.error_events: dict[str, threading.Event] = {}
        self.error_messages: dict[str, str] = {}
        self._observation_spaces: dict[str, Any] = {}
        self._last_frame_timestamps: dict[str, float] = {}
        self._stale_frame_counts: dict[str, int] = {}

        camera_configs = self._get_camera_configs()

        for _idx, (mount_position, camera_config) in enumerate(camera_configs.items()):
            camera_queue = queue.Queue(maxsize=config.queue_size)
            shutdown_event = threading.Event()
            error_event = threading.Event()

            self.camera_queues[mount_position] = camera_queue
            self.shutdown_events[mount_position] = shutdown_event
            self.error_events[mount_position] = error_event

            thread = threading.Thread(
                target=self._camera_worker_wrapper,
                args=(
                    mount_position,
                    camera_config["camera_type"],
                    camera_config["device_id"],
                    camera_queue,
                    shutdown_event,
                    error_event,
                ),
            )
            thread.start()
            self.camera_threads[mount_position] = thread

            # Stagger init to avoid USB bandwidth contention
            init_timeout = 15.0
            init_start = time.time()
            while time.time() - init_start < init_timeout:
                if mount_position in self._observation_spaces:
                    print(
                        f"[{mount_position}] Camera ready, waiting 3s before next camera..."
                    )
                    time.sleep(3.0)
                    break
                time.sleep(0.5)
            else:
                print(f"[{mount_position}] Camera init timeout, proceeding anyway...")

        if config.run_as_server:
            print("Waiting for all cameras to be ready before starting server...")
            self._wait_for_all_cameras_ready(timeout=60.0)
            self.start_server(config.port)

    def _get_camera_configs(self) -> dict[str, dict]:
        camera_configs = {}

        if self.config.ego_view_camera is not None:
            camera_configs[CameraMountPosition.EGO_VIEW.value] = {
                "camera_type": self.config.ego_view_camera,
                "device_id": self.config.ego_view_device_id,
            }

        if self.config.head_camera is not None:
            camera_configs[CameraMountPosition.HEAD.value] = {
                "camera_type": self.config.head_camera,
                "device_id": self.config.head_device_id,
            }

        if self.config.chest_camera is not None:
            camera_configs[CameraMountPosition.CHEST_VIEW.value] = {
                "camera_type": self.config.chest_camera,
                "device_id": self.config.chest_device_id,
            }

        if self.config.left_wrist_camera is not None:
            camera_configs[CameraMountPosition.LEFT_WRIST.value] = {
                "camera_type": self.config.left_wrist_camera,
                "device_id": self.config.left_wrist_device_id,
            }

        if self.config.right_wrist_camera is not None:
            camera_configs[CameraMountPosition.RIGHT_WRIST.value] = {
                "camera_type": self.config.right_wrist_camera,
                "device_id": self.config.right_wrist_device_id,
            }

        return camera_configs

    def _wait_for_all_cameras_ready(self, timeout: float = 60.0):
        expected_cameras = set(self.camera_queues.keys())
        start_time = time.time()

        while time.time() - start_time < timeout:
            ready_cameras = set()
            for mount_position, camera_queue in self.camera_queues.items():
                if not camera_queue.empty():
                    ready_cameras.add(mount_position)

            if ready_cameras == expected_cameras:
                print(f"All {len(expected_cameras)} cameras ready: {ready_cameras}")
                time.sleep(1.0)
                return

            waiting_for = expected_cameras - ready_cameras
            print(
                f"Waiting for cameras: {waiting_for} "
                f"({len(ready_cameras)}/{len(expected_cameras)} ready)"
            )
            time.sleep(2.0)

        ready_cameras = set()
        for mount_position, camera_queue in self.camera_queues.items():
            if not camera_queue.empty() or mount_position in self._observation_spaces:
                ready_cameras.add(mount_position)
        missing = expected_cameras - ready_cameras
        print(
            f"[WARNING] Timeout waiting for all cameras. "
            f"Missing: {missing}. Starting anyway with: {ready_cameras}"
        )

    def _camera_worker_wrapper(
        self,
        mount_position: str,
        camera_type: str,
        device_id: str | None,
        image_queue: queue.Queue,
        shutdown_event: threading.Event,
        error_event: threading.Event,
    ):
        """Worker thread with auto-reconnection."""
        max_init_retries = 10
        max_reconnect_attempts = 5
        reconnect_count = 0

        while not shutdown_event.is_set() and reconnect_count < max_reconnect_attempts:
            camera = None
            try:
                init_retry_delay = 1.0

                for attempt in range(max_init_retries):
                    if shutdown_event.is_set():
                        return

                    try:
                        if reconnect_count > 0:
                            print(
                                f"[{mount_position}] Reconnecting camera "
                                f"(reconnect {reconnect_count}/{max_reconnect_attempts}, "
                                f"attempt {attempt + 1}/{max_init_retries})..."
                            )
                        else:
                            print(
                                f"[{mount_position}] Initializing camera "
                                f"(attempt {attempt + 1}/{max_init_retries})..."
                            )
                        camera = self._instantiate_camera(
                            mount_position, camera_type, device_id
                        )
                        print(f"[{mount_position}] Camera initialized successfully")
                        break
                    except Exception as e:
                        print(f"[{mount_position}] Camera init failed: {e}")
                        if attempt < max_init_retries - 1:
                            print(
                                f"[{mount_position}] Retrying in {init_retry_delay:.1f}s..."
                            )
                            time.sleep(init_retry_delay)
                            init_retry_delay = min(init_retry_delay * 1.5, 10.0)
                        else:
                            raise RuntimeError(
                                f"Camera {mount_position} ({camera_type}) failed to initialize "
                                f"after {max_init_retries} attempts: {e}"
                            )

                obs_space = camera.observation_space()
                if obs_space is not None:
                    self._observation_spaces[mount_position] = obs_space
                else:
                    self._observation_spaces[mount_position] = True

                consecutive_failures = 0
                max_consecutive_failures = 10
                warmup_period = True
                warmup_start_time = time.time()
                warmup_timeout = 5.0

                while not shutdown_event.is_set():
                    try:
                        frame = camera.read()
                    except Exception as e:
                        print(f"[{mount_position}] Frame read exception: {e}")
                        frame = None
                        consecutive_failures = max_consecutive_failures

                    if frame:
                        consecutive_failures = 0
                        warmup_period = False
                        try:
                            image_queue.put_nowait(frame)
                        except queue.Full:
                            try:
                                image_queue.get_nowait()
                                image_queue.put_nowait(frame)
                            except queue.Empty:
                                pass
                    else:
                        if warmup_period:
                            if time.time() - warmup_start_time > warmup_timeout:
                                print(
                                    f"[{mount_position}] Warmup timeout — will attempt reconnect"
                                )
                                break
                            time.sleep(0.1)
                        else:
                            consecutive_failures += 1
                            if consecutive_failures >= max_consecutive_failures:
                                print(
                                    f"[{mount_position}] Too many consecutive failures "
                                    f"({consecutive_failures}) — will attempt reconnect"
                                )
                                break
                            time.sleep(0.01)

                if camera is not None:
                    try:
                        camera.close()
                    except Exception as e:
                        print(f"[{mount_position}] Error closing camera: {e}")
                    camera = None

                if not shutdown_event.is_set():
                    reconnect_count += 1
                    print(
                        f"[{mount_position}] Waiting 5 seconds before reconnect attempt..."
                    )
                    time.sleep(5.0)

            except Exception as e:
                print(f"[{mount_position}] Camera error: {e}")
                if camera is not None:
                    try:
                        camera.close()
                    except Exception:
                        pass
                    camera = None

                if not shutdown_event.is_set():
                    reconnect_count += 1
                    if reconnect_count < max_reconnect_attempts:
                        print(
                            f"[{mount_position}] Waiting 5 seconds before reconnect "
                            f"attempt {reconnect_count}/{max_reconnect_attempts}..."
                        )
                        time.sleep(5.0)

        if reconnect_count >= max_reconnect_attempts and not shutdown_event.is_set():
            error_msg = (
                f"Camera {mount_position} ({camera_type}) failed "
                f"after {max_reconnect_attempts} reconnect attempts"
            )
            print(f"[ERROR] {error_msg}")
            self.error_messages[mount_position] = error_msg
            error_event.set()

    def _instantiate_camera(
        self, mount_position: str, camera_type: str, device_id: str | None = None
    ) -> Sensor:
        """Instantiate a camera sensor based on camera_type (lazy imports)."""
        if camera_type in ("oak", "oak_mono"):
            from gear_sonic.camera.drivers.oak import OAKConfig, OAKSensor

            oak_config = OAKConfig()
            oak_config.use_mjpeg = self.config.use_mjpeg
            oak_config.mjpeg_quality = self.config.mjpeg_quality
            if camera_type == "oak_mono":
                oak_config.enable_mono_cameras = True
            print(f"Initializing OAK sensor for camera type: {camera_type}")
            return OAKSensor(
                config=oak_config, mount_position=mount_position, device_id=device_id
            )

        elif camera_type == "realsense":
            from gear_sonic.camera.drivers.realsense import (
                RealSenseConfig,
                RealSenseSensor,
            )

            print(
                f"Initializing RealSense sensor for camera type: {camera_type}, "
                f"device: {device_id}"
            )
            realsense_config = RealSenseConfig()
            realsense_config.fps = self.config.fps
            realsense_config.enable_depth = (
                self.config.realsense_enable_depth
                and mount_position in self.config.enabled_realsense_depth_mounts
            )
            return RealSenseSensor(
                config=realsense_config,
                mount_position=mount_position,
                device_id=device_id,
            )

        elif camera_type == "orbbec":
            from gear_sonic.camera.drivers.orbbec import OrbbecConfig, OrbbecSensor

            print(
                f"Initializing Orbbec sensor for camera type: {camera_type}, "
                f"device: {device_id}"
            )
            orbbec_config = OrbbecConfig()
            orbbec_config.fps = self.config.fps
            orbbec_config.enable_depth = self.config.orbbec_enable_depth
            return OrbbecSensor(
                config=orbbec_config,
                mount_position=mount_position,
                device_id=device_id,
            )

        elif camera_type.endswith(".mp4"):
            from gear_sonic.camera.drivers.dummy import ReplayDummySensor

            print(f"Initializing Replay Dummy Sensor for camera type: {camera_type}")
            return ReplayDummySensor(video_path=camera_type)

        elif camera_type == "usb":
            from gear_sonic.camera.drivers.usb_camera import (
                USBCameraConfig,
                USBCameraSensor,
            )

            usb_config = USBCameraConfig()
            device_idx = int(device_id) if device_id else 0
            print(
                f"Initializing USB camera for type: {camera_type}, device: {device_idx}"
            )
            return USBCameraSensor(
                config=usb_config,
                mount_position=mount_position,
                device_index=device_idx,
            )

        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")

    def _check_for_errors(self):
        for mount_position, error_event in self.error_events.items():
            if error_event.is_set():
                error_msg = self.error_messages.get(
                    mount_position,
                    f"Camera {mount_position} encountered an unknown error",
                )
                raise RuntimeError(error_msg)

    def read(self):
        """Read the latest cached frame from each camera.

        The per-camera worker keeps overwriting old frames when a newer one arrives.
        The composer only peeks the most recent cached frame instead of consuming it,
        so a temporary phase mismatch between cameras does not cause the whole
        composed message to disappear for a cycle.
        """
        self._check_for_errors()

        expected_cameras = set(self.camera_queues.keys())
        message = {}

        for mount_position, camera_queue in self.camera_queues.items():
            frame = self._get_latest_from_queue(camera_queue)
            if frame is not None:
                self._track_stale_frame(mount_position, frame)
                message[mount_position] = frame

        if set(message.keys()) == expected_cameras:
            return message
        return None

    def _track_stale_frame(self, mount_position: str, frame: dict[str, Any]) -> None:
        timestamps = frame.get("timestamps", {})
        timestamp = timestamps.get(mount_position)
        if timestamp is None and timestamps:
            timestamp = max(timestamps.values())
        if timestamp is None:
            return

        last_timestamp = self._last_frame_timestamps.get(mount_position)
        if last_timestamp is None or timestamp > last_timestamp:
            self._last_frame_timestamps[mount_position] = timestamp
            self._stale_frame_counts[mount_position] = 0
            return

        stale_count = self._stale_frame_counts.get(mount_position, 0) + 1
        if stale_count >= self.config.stale_frame_warning_count:
            print(
                f"\033[93m[WARNING] Camera '{mount_position}' reused the same frame timestamp "
                f"for {stale_count} consecutive server reads. "
                "This camera may be dropping frames or stalled.\033[0m"
            )
            stale_count = 0
        self._stale_frame_counts[mount_position] = stale_count

    def _get_latest_from_queue(
        self, camera_queue: queue.Queue
    ) -> dict[str, Any] | None:
        # Peek instead of draining so each camera always exposes its latest frame
        # until the producer overwrites it with a newer one.
        with camera_queue.mutex:
            if not camera_queue.queue:
                return None
            return camera_queue.queue[-1]

    def close(self):
        for shutdown_event in self.shutdown_events.values():
            shutdown_event.set()
        for thread in self.camera_threads.values():
            thread.join(timeout=5.0)
        for camera_queue in self.camera_queues.values():
            try:
                while True:
                    camera_queue.get_nowait()
            except queue.Empty:
                pass
        if self.config.run_as_server:
            self.stop_server()

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Use serialize_message() for ComposedCameraSensor")

    def serialize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Merge per-camera data into a single ImageMessageSchema."""
        all_timestamps = {}
        all_images = {}
        all_camera_info = {}
        for _mount, camera_data in message.items():
            all_timestamps.update(camera_data.get("timestamps", {}))
            all_images.update(camera_data.get("images", {}))
            if self.config.publish_camera_info:
                all_camera_info.update(camera_data.get("camera_info", {}))
        img_schema = ImageMessageSchema(
            timestamps=all_timestamps,
            images=all_images,
            camera_info=all_camera_info,
        )
        return img_schema.serialize()

    def run_server(self):
        """Main server loop — reads, serializes and publishes frames."""
        idx = 0
        server_start_time = time.monotonic()
        fps_print_time = time.monotonic()
        frame_interval = 1.0 / self.config.fps

        while True:
            target_time = server_start_time + (idx + 1) * frame_interval

            message = self.read()
            if message:
                if self.config.test_latency:
                    read_qr_code(message)

                serialized_message = self.serialize_message(message)
                self.send_message(serialized_message)
                idx += 1

                if idx % 10 == 0:
                    print(
                        f"Image sending FPS: {10 / (time.monotonic() - fps_print_time):.2f}"
                    )
                    fps_print_time = time.monotonic()

            current_time = time.monotonic()
            sleep_time = target_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                if not message:
                    idx += 1

    def observation_space(self):
        try:
            import gymnasium as gym

            return gym.spaces.Dict(self._observation_spaces)
        except ImportError:
            return None


class ComposedCameraClientSensor(Sensor, SensorClient):
    """ZMQ client that deserializes merged camera frames from the server."""

    def __init__(
        self,
        server_ip: str = "localhost",
        port: int = 5555,
        decode_images: bool = True,
    ):
        self.start_client(server_ip, port)
        self.decode_images = decode_images

        self._latest_message = None
        self._avg_time_per_frame: deque = deque(maxlen=20)
        self._msg_received_time = 0
        self._start_time = 0.0
        self.idx = 0

        self._last_new_message_time = None
        self._last_staleness_warning_time = 0.0
        self._staleness_warning_interval = 2.0

        print("Initialized composed camera client sensor")

    def read(self, blocking: bool = False, **kwargs) -> dict[str, Any] | None:
        self._start_time = time.time()
        current_time = time.time()

        if blocking:
            message = self.receive_message()
            if not message:
                return None
        else:
            message = self.receive_message_nonblocking(timeout_ms=0)

        if message is not None:
            self.idx += 1
            self._latest_message = ImageMessageSchema.deserialize(
                message, decode_images=self.decode_images
            ).asdict()
            self._last_new_message_time = current_time

            if self.idx % 300 == 0:
                latency_parts = []
                log_time = time.time()
                for image_key, image_time in self._latest_message["timestamps"].items():
                    image_latency = (log_time - image_time) * 1000
                    latency_parts.append(f"{image_key}={image_latency:.2f}ms")
                print(f"[Image Latency] {', '.join(latency_parts)}")

            self._msg_received_time = time.time()
            self._avg_time_per_frame.append(self._msg_received_time - self._start_time)
        elif not blocking and self._latest_message is not None:
            if self._last_new_message_time is not None:
                time_since_last_message = current_time - self._last_new_message_time
                if time_since_last_message > 0.1:
                    if (
                        current_time - self._last_staleness_warning_time
                        >= self._staleness_warning_interval
                    ):
                        print(
                            f"[WARNING] No new image message received for "
                            f"{time_since_last_message*1000:.1f}ms. "
                            f"Reusing stale image. Check camera server connection."
                        )
                        self._last_staleness_warning_time = current_time

        return self._latest_message

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Client does not serialize")

    def close(self):
        self.stop_client()

    def fps(self) -> float:
        if len(self._avg_time_per_frame) == 0:
            return 0.0
        return float(1 / np.mean(self._avg_time_per_frame))


class _MjpegGrabber:
    """Background thread that reads an MJPEG stream via raw HTTP."""

    def __init__(self, url: str):
        self.url = url
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import requests

        resp = requests.get(self.url, stream=True, timeout=10)
        buf = b""
        for chunk in resp.iter_content(chunk_size=4096):
            if not self._running:
                break
            buf += chunk
            while True:
                soi = buf.find(b"\xff\xd8")
                if soi == -1:
                    break
                eoi = buf.find(b"\xff\xd9", soi + 2)
                if eoi == -1:
                    break
                jpeg_bytes = buf[soi : eoi + 2]
                buf = buf[eoi + 2 :]
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    with self.lock:
                        self.frame = frame

    def get(self) -> np.ndarray | None:
        with self.lock:
            return self.frame

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)


class ComposedCameraHttpClient:
    """Camera client that reads MJPEG streams over HTTP.

    Drop-in replacement for :class:`ComposedCameraClientSensor` when cameras
    are served via an HTTP MJPEG server.

    Usage::

        client = ComposedCameraHttpClient("http://<ROBOT_IP>:8000")
        data = client.read()  # {"images": {"left": ndarray, ...}, "timestamps": {...}}
    """

    DEFAULT_NAME_MAP = {
        "center": "ego_view",
        "chest": "chest_view",
        "left": "left_wrist",
        "right": "right_wrist",
    }

    def __init__(self, base_url: str, camera_name_map: dict[str, str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.camera_name_map = (
            camera_name_map if camera_name_map is not None else self.DEFAULT_NAME_MAP
        )
        self.camera_names: list[str] = []
        self._grabbers: dict[str, _MjpegGrabber] = {}
        self._connect()

    def _connect(self):
        import requests

        resp = requests.get(f"{self.base_url}/cameras", timeout=5)
        resp.raise_for_status()
        self.camera_names = resp.json()
        print(f"HTTP MJPEG: discovered cameras: {self.camera_names}")
        for name in self.camera_names:
            url = f"{self.base_url}/stream/{name}"
            self._grabbers[name] = _MjpegGrabber(url)

    def read(self, blocking: bool = False, **kwargs) -> dict[str, Any] | None:
        images = {}
        any_ok = False
        for name, grabber in self._grabbers.items():
            frame = grabber.get()
            mapped_name = self.camera_name_map.get(name, name)
            if frame is not None:
                images[mapped_name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                any_ok = True
            else:
                images[mapped_name] = None
        if not any_ok:
            return None
        return {"images": images, "timestamps": {n: time.time() for n in images}}

    def close(self):
        for grabber in self._grabbers.values():
            grabber.stop()


def run_server_from_config(config: ComposedCameraConfig) -> None:
    """Run the composed server and always release its camera workers."""
    composed_camera = ComposedCameraSensor(config)
    print("Running composed camera server...")
    try:
        composed_camera.run_server()
    except KeyboardInterrupt:
        print("Stopping camera server...")
    finally:
        composed_camera.close()



if __name__ == "__main__":
    import tyro

    config = tyro.cli(ComposedCameraConfig)

    if config.run_as_server:
        run_server_from_config(config)
    else:
        composed_client = ComposedCameraClientSensor(
            server_ip="localhost", port=config.port
        )
        try:
            while True:
                data = composed_client.read()
                if data is not None:
                    print(f"FPS: {composed_client.fps():.2f}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping client...")
            composed_client.close()
