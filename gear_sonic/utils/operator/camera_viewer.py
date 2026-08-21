"""SensorGateway camera viewer with optional recording.

Reads decoded RGB camera frames from the local SensorGateway shared-memory
Snapshot API and displays them using OpenCV. Supports recording to MP4.

Virtual environment setup (run from repo root):
    bash install_scripts/install_data_collection.sh
    source .venv_data_collection/bin/activate

Usage:
    python -m gear_sonic.utils.operator.camera_viewer \
        --profile gear_sonic/config/launch_inference.yaml

Controls (OpenCV window must be focused):
    R - Start/stop recording
    Q - Quit

Output structure:
    camera_recordings/
    └── rec_20260403_143052/
        ├── ego_view.mp4
        └── head_left_color_image.mp4
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional

import cv2
import numpy as np
from gear_sonic.runtime.gateway.sensor_client import (
    SensorGatewayClient,
    SensorGatewayClientError,
)
from gear_sonic.runtime.profile import default_runtime_profile_path, load_runtime_profile
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest


@dataclass
class CameraViewerConfig:
    """CLI config for the ROS-free camera viewer."""

    profile: str = str(default_runtime_profile_path())
    """Runtime profile containing the local SensorGateway endpoint."""

    fps: int = 30
    """Target display refresh rate (Hz)."""

    output_path: Optional[str] = None
    """Output directory for recordings. Auto-creates 'camera_recordings/' if not set."""

    codec: str = "mp4v"
    """Video codec for recording (e.g., 'mp4v', 'XVID')."""

    max_display_width: int = 640
    """Max width per camera tile in the display window."""


def _is_rgb_image(image: Any) -> bool:
    return (
        isinstance(image, np.ndarray)
        and image.ndim == 3
        and image.shape[2] == 3
    )


def _rgb_camera_names(images: Mapping[str, Any]) -> list[str]:
    """Return only three-channel streams suitable for this RGB viewer."""
    return sorted(name for name, image in images.items() if _is_rgb_image(image))


def _gateway_rgb_streams(health: Mapping[str, Any]) -> tuple[str, ...]:
    """Return sorted decoded RGB camera stream names advertised by Gateway."""
    streams = health.get("streams", {})
    if not isinstance(streams, Mapping):
        return ()
    return tuple(
        sorted(
            name
            for name in streams
            if isinstance(name, str)
            and name.startswith("camera/")
            and not name.endswith("_depth")
        )
    )


class GatewayCameraClient:
    """Expose SensorGateway RGB snapshots in the viewer's camera-message shape."""

    def __init__(
        self,
        endpoint: str,
        *,
        max_age_ms: float = 1500.0,
        max_skew_ms: float = 5.0,
        client: SensorGatewayClient | None = None,
    ) -> None:
        self.client = client or SensorGatewayClient(endpoint, request_timeout_ms=100)
        self.max_age_ms = float(max_age_ms)
        self.max_skew_ms = float(max_skew_ms)
        self._streams: tuple[str, ...] = ()

    def read(self, blocking: bool = False) -> dict[str, dict[str, np.ndarray]] | None:
        del blocking
        try:
            if not self._streams:
                self._streams = _gateway_rgb_streams(self.client.health())
                if not self._streams:
                    return None
            snapshot = self.client.read_snapshot(
                SnapshotRequest(
                    streams=self._streams,
                    max_age_ms=self.max_age_ms,
                    max_skew_ms=self.max_skew_ms,
                ),
                retries=0,
            )
        except SensorGatewayClientError:
            return None

        images = {
            stream.removeprefix("camera/"): image
            for stream, image in snapshot.arrays.items()
            if _is_rgb_image(image) and image.dtype == np.uint8
        }
        return {"images": images} if images else None

    def close(self) -> None:
        self.client.close()


def _wait_for_first_camera_frame(
    client: GatewayCameraClient,
    *,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.1,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, dict[str, np.ndarray]] | None:
    """Wait up to a wall-clock deadline, including time spent inside RPCs."""
    deadline = monotonic() + timeout_s
    while True:
        if deadline - monotonic() <= 1.0e-9:
            break
        sample = client.read(blocking=False)
        if sample and sample.get("images"):
            return sample
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            break
        sleep(min(poll_interval_s, remaining))
    return None


def _run_viewer(
    config: CameraViewerConfig,
    client: GatewayCameraClient,
    endpoint: str,
) -> None:

    print(f"Waiting for first camera frame from SensorGateway {endpoint}...")
    sample = _wait_for_first_camera_frame(client)

    if sample is None or not sample.get("images"):
        print("ERROR: No camera frames received from SensorGateway after 10s.")
        return

    camera_names = _rgb_camera_names(sample["images"])
    print(f"Detected {len(camera_names)} camera stream(s): {camera_names}")

    output_dir = Path(config.output_path) if config.output_path else Path("camera_recordings")

    is_recording = False
    video_writers: dict[str, cv2.VideoWriter] = {}
    frame_count = 0
    recording_start_time = 0.0
    recording_dir = Path(".")
    loop_period = 1.0 / config.fps

    window_name = "SONIC Camera Viewer"

    print(f"Target FPS: {config.fps}")
    print(f"Recordings will be saved to: {output_dir}")
    print("Controls: R = start/stop recording, Q = quit")

    try:
        while True:
            t_start = time.monotonic()

            image_data = client.read(blocking=False)
            if image_data is None or not image_data.get("images"):
                elapsed = time.monotonic() - t_start
                remaining = loop_period - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                continue

            tiles = []
            for name in camera_names:
                img = image_data["images"].get(name)
                if not _is_rgb_image(img):
                    continue

                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if is_recording and name in video_writers:
                    video_writers[name].write(img_bgr)

                h, w = img_bgr.shape[:2]
                if w > config.max_display_width:
                    scale = config.max_display_width / w
                    img_bgr = cv2.resize(
                        img_bgr, (config.max_display_width, int(h * scale))
                    )

                label = f"{name}"
                if is_recording:
                    label = f"[REC] {name}"
                cv2.putText(
                    img_bgr, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                tiles.append(img_bgr)

            if tiles:
                max_h = max(t.shape[0] for t in tiles)
                padded = []
                for t in tiles:
                    if t.shape[0] < max_h:
                        pad = np.zeros(
                            (max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8
                        )
                        t = np.vstack([t, pad])
                    padded.append(t)
                canvas = np.hstack(padded)

                if is_recording:
                    frame_count += 1
                    elapsed_rec = time.time() - recording_start_time
                    status = f"REC {frame_count}f / {elapsed_rec:.1f}s"
                    cv2.putText(
                        canvas, status, (canvas.shape[1] - 300, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )

                cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quit requested.")
                break
            elif key == ord("r"):
                if not is_recording:
                    recording_dir = output_dir / f"rec_{time.strftime('%Y%m%d_%H%M%S')}"
                    recording_dir.mkdir(parents=True, exist_ok=True)

                    fourcc = cv2.VideoWriter_fourcc(*config.codec)
                    video_writers = {}
                    for name in camera_names:
                        img = image_data["images"].get(name)
                        if _is_rgb_image(img):
                            h, w = img.shape[:2]
                            path = recording_dir / f"{name}.mp4"
                            video_writers[name] = cv2.VideoWriter(
                                str(path), fourcc, config.fps, (w, h)
                            )

                    is_recording = True
                    recording_start_time = time.time()
                    frame_count = 0
                    print(f"Recording started: {recording_dir}")
                else:
                    is_recording = False
                    for writer in video_writers.values():
                        writer.release()
                    video_writers = {}
                    duration = time.time() - recording_start_time
                    print(
                        f"Recording stopped - {duration:.1f}s, {frame_count} frames "
                        f"-> {recording_dir}"
                    )

            elapsed = time.monotonic() - t_start
            remaining = loop_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if video_writers:
            for writer in video_writers.values():
                writer.release()
            if is_recording:
                duration = time.time() - recording_start_time
                print(f"Final recording: {duration:.1f}s, {frame_count} frames")

        cv2.destroyAllWindows()


def main(config: CameraViewerConfig) -> None:
    profile = load_runtime_profile(config.profile or None)
    endpoint = profile.endpoint_uri("sensor_gateway_metadata")
    client = GatewayCameraClient(endpoint)
    try:
        _run_viewer(config, client, endpoint)
    finally:
        client.close()


if __name__ == "__main__":
    import tyro

    config = tyro.cli(CameraViewerConfig)
    main(config)
