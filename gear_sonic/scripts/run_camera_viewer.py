"""
ROS-free camera viewer with optional recording.

Connects to a ZMQ camera server (MuJoCo sim SensorServer or real robot camera)
and displays live camera feeds using OpenCV. Supports recording to MP4.

Virtual environment setup (run from repo root):
    bash install_scripts/install_data_collection.sh
    source .venv_data_collection/bin/activate

Usage:
    python gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5555

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
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional

import cv2
import numpy as np
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


@dataclass
class CameraViewerConfig:
    """CLI config for the ROS-free camera viewer."""

    camera_host: str = "localhost"
    """Camera server hostname."""

    camera_port: int = 5555
    """Camera server port."""

    fps: int = 30
    """Target display refresh rate (Hz)."""

    output_path: Optional[str] = None
    """Output directory for recordings. Auto-creates 'camera_recordings/' if not set."""

    codec: str = "mp4v"
    """Video codec for recording (e.g., 'mp4v', 'XVID')."""

    max_display_width: int = 640
    """Max width per camera tile in the display window."""

    camera_streams: str = ""
    """Optional comma-separated RGB stream names to display, in order."""

    window_name: str = "SONIC Camera Viewer"
    """OpenCV display-window title."""

    status_endpoint: str = ""
    """Optional Base Pose velocity/status PUB endpoint."""


@dataclass(frozen=True)
class BasePoseViewerStatus:
    """Latest Base Pose command state rendered by the camera viewer."""

    active_camera_stream: str | None
    velocity: tuple[float, float, float]


def _is_rgb_image(image: Any) -> bool:
    return (
        isinstance(image, np.ndarray)
        and image.ndim == 3
        and image.shape[2] == 3
    )


def _rgb_camera_names(images: Mapping[str, Any]) -> list[str]:
    """Return only three-channel streams suitable for this RGB viewer."""
    return sorted(name for name, image in images.items() if _is_rgb_image(image))


def select_rgb_camera_names(
    images: Mapping[str, Any], camera_streams: str
) -> list[str]:
    """Select requested RGB streams while preserving the requested order."""
    available = set(_rgb_camera_names(images))
    requested = [
        name.strip() for name in str(camera_streams).split(",") if name.strip()
    ]
    if not requested:
        return sorted(available)
    return [name for name in requested if name in available]


def parse_base_pose_viewer_status(raw: bytes | str) -> BasePoseViewerStatus:
    """Parse one Base Pose command without affecting the control subscriber."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError(f"invalid Base Pose status JSON: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("type") != "navila_reasan_velocity_command"
        or payload.get("source") != "base_pose"
    ):
        raise ValueError("message is not a Base Pose velocity command")
    velocity = payload.get("velocity")
    if not isinstance(velocity, dict):
        raise ValueError("Base Pose command has no velocity object")
    try:
        values = tuple(float(velocity[key]) for key in ("vx", "vy", "wz"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Base Pose command has invalid velocity values") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Base Pose command velocity must be finite")
    raw_camera_stream = payload.get("camera_stream")
    active_camera_stream = (
        None
        if raw_camera_stream is None or not str(raw_camera_stream).strip()
        else str(raw_camera_stream).strip()
    )
    return BasePoseViewerStatus(
        active_camera_stream=active_camera_stream,
        velocity=(values[0], values[1], values[2]),
    )


def read_latest_base_pose_viewer_status(
    socket: zmq.Socket,
    current: BasePoseViewerStatus,
) -> BasePoseViewerStatus:
    """Drain pending status messages and retain the latest valid command."""
    latest = current
    while True:
        try:
            raw = socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return latest
        try:
            latest = parse_base_pose_viewer_status(raw)
        except ValueError:
            continue


def camera_label_color(
    camera_stream: str,
    active_camera_stream: str | None,
) -> tuple[int, int, int]:
    """Use red for the camera currently consumed by Base Pose YOLOE."""
    if camera_stream == active_camera_stream:
        return (0, 0, 255)
    return (0, 255, 0)


def format_velocity_label(velocity: tuple[float, float, float]) -> str:
    """Format the exact Base Pose velocity command for the display overlay."""
    vx, vy, wz = velocity
    return f"CMD vx={vx:+.3f}  vy={vy:+.3f}  wz={wz:+.3f}"


def main(config: CameraViewerConfig):
    client = ComposedCameraClientSensor(server_ip=config.camera_host, port=config.camera_port)
    status_socket: zmq.Socket | None = None
    viewer_status = BasePoseViewerStatus(None, (0.0, 0.0, 0.0))

    print("Waiting for first camera frame...")
    sample = None
    for _ in range(100):
        sample = client.read(blocking=False)
        if sample and sample.get("images"):
            break
        time.sleep(0.1)

    if sample is None or not sample.get("images"):
        print("ERROR: No camera frames received after 10s. Check the camera server.")
        return

    camera_names = select_rgb_camera_names(
        sample["images"], config.camera_streams
    )
    if not camera_names:
        print(
            "ERROR: None of the requested RGB streams are available: "
            f"{config.camera_streams!r}"
        )
        client.close()
        return
    print(f"Detected {len(camera_names)} camera stream(s): {camera_names}")

    if config.status_endpoint:
        status_socket = zmq.Context.instance().socket(zmq.SUB)
        status_socket.setsockopt(zmq.SUBSCRIBE, b"")
        status_socket.setsockopt(zmq.CONFLATE, 1)
        status_socket.setsockopt(zmq.LINGER, 0)
        status_socket.connect(config.status_endpoint)

    output_dir = Path(config.output_path) if config.output_path else Path("camera_recordings")

    is_recording = False
    video_writers: dict[str, cv2.VideoWriter] = {}
    frame_count = 0
    recording_start_time = 0.0
    recording_dir = Path(".")
    loop_period = 1.0 / config.fps

    window_name = config.window_name

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

            if status_socket is not None:
                viewer_status = read_latest_base_pose_viewer_status(
                    status_socket, viewer_status
                )

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
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    camera_label_color(name, viewer_status.active_camera_stream),
                    2,
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

                velocity_label = format_velocity_label(viewer_status.velocity)
                cv2.putText(
                    canvas,
                    velocity_label,
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    4,
                )
                cv2.putText(
                    canvas,
                    velocity_label,
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

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

        client.close()
        if status_socket is not None:
            status_socket.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config = tyro.cli(CameraViewerConfig)
    main(config)
