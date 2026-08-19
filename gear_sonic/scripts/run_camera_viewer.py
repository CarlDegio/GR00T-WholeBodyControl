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
    target_bbox_xyxy: tuple[float, float, float, float] | None = None
    target_lateral_anchor_px: tuple[float, float] | None = None
    table_edge_endpoints_px: (
        tuple[tuple[float, float], tuple[float, float]] | None
    ) = None
    desk_mask_row_spans: tuple[tuple[int, int, int], ...] | None = None
    overlay_image_size: tuple[int, int] | None = None


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


def _finite_tuple(
    value: Any,
    *,
    length: int,
    description: str,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"Base Pose {description} must contain {length} values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Base Pose {description} contains invalid values"
        ) from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"Base Pose {description} must be finite")
    return result


def _parse_viewer_overlay(
    payload: Mapping[str, Any],
) -> tuple[
    tuple[float, float, float, float] | None,
    tuple[float, float] | None,
    tuple[tuple[float, float], tuple[float, float]] | None,
    tuple[tuple[int, int, int], ...] | None,
    tuple[int, int] | None,
]:
    raw_overlay = payload.get("viewer_overlay")
    if raw_overlay is None:
        return None, None, None, None, None
    if not isinstance(raw_overlay, dict):
        raise ValueError("Base Pose viewer overlay must be an object")
    raw_bbox = _finite_tuple(
        raw_overlay.get("target_bbox_xyxy"),
        length=4,
        description="viewer target bbox",
    )
    target_bbox = (
        raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3]
    )
    if target_bbox[2] < target_bbox[0] or target_bbox[3] < target_bbox[1]:
        raise ValueError("Base Pose viewer target bbox has invalid corner order")

    target_lateral_anchor = None
    raw_anchor = raw_overlay.get("target_lateral_anchor_px")
    if raw_anchor is not None:
        anchor = _finite_tuple(
            raw_anchor,
            length=2,
            description="viewer target lateral anchor",
        )
        target_lateral_anchor = (anchor[0], anchor[1])

    table_edge = None
    raw_edge = raw_overlay.get("table_edge_endpoints_px")
    if raw_edge is not None:
        if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
            raise ValueError(
                "Base Pose viewer table edge must contain two endpoints"
            )
        start = _finite_tuple(
            raw_edge[0],
            length=2,
            description="viewer table edge start",
        )
        end = _finite_tuple(
            raw_edge[1],
            length=2,
            description="viewer table edge end",
        )
        table_edge = ((start[0], start[1]), (end[0], end[1]))

    image_size = None
    raw_size = raw_overlay.get("image_size")
    if raw_size is not None:
        size = _finite_tuple(
            raw_size,
            length=2,
            description="viewer image size",
        )
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0 or (width, height) != size:
            raise ValueError(
                "Base Pose viewer image size must contain positive integers"
            )
        image_size = (width, height)

    desk_mask_row_spans = None
    raw_spans = raw_overlay.get("desk_mask_row_spans")
    if raw_spans is not None:
        if image_size is None:
            raise ValueError(
                "Base Pose viewer desk mask requires an image size"
            )
        if not isinstance(raw_spans, (list, tuple)):
            raise ValueError("Base Pose viewer desk mask spans must be a list")
        parsed_spans: list[tuple[int, int, int]] = []
        for raw_span in raw_spans:
            values = _finite_tuple(
                raw_span,
                length=3,
                description="viewer desk mask row span",
            )
            row, start, end = (int(value) for value in values)
            if (row, start, end) != values:
                raise ValueError(
                    "Base Pose viewer desk mask spans must contain integers"
                )
            width, height = image_size
            if not (0 <= row < height and 0 <= start < end <= width):
                raise ValueError(
                    "Base Pose viewer desk mask span is outside the image"
                )
            parsed_spans.append((row, start, end))
        desk_mask_row_spans = tuple(parsed_spans)
    return (
        target_bbox,
        target_lateral_anchor,
        table_edge,
        desk_mask_row_spans,
        image_size,
    )


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
    (
        target_bbox,
        target_lateral_anchor,
        table_edge,
        desk_mask_row_spans,
        image_size,
    ) = _parse_viewer_overlay(payload)
    return BasePoseViewerStatus(
        active_camera_stream=active_camera_stream,
        velocity=(values[0], values[1], values[2]),
        target_bbox_xyxy=target_bbox,
        target_lateral_anchor_px=target_lateral_anchor,
        table_edge_endpoints_px=table_edge,
        desk_mask_row_spans=desk_mask_row_spans,
        overlay_image_size=image_size,
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


def _outlined_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_base_pose_overlays(
    image_bgr: np.ndarray,
    camera_stream: str,
    status: BasePoseViewerStatus,
) -> np.ndarray:
    """Draw the exact target box and table edge consumed by Base Pose."""
    if (
        not _is_rgb_image(image_bgr)
        or camera_stream != status.active_camera_stream
        or status.target_bbox_xyxy is None
    ):
        return image_bgr
    height, width = image_bgr.shape[:2]
    source_width, source_height = status.overlay_image_size or (width, height)
    scale_x = width / source_width
    scale_y = height / source_height

    def point(x: float, y: float) -> tuple[int, int]:
        return (
            min(width - 1, max(0, int(round(x * scale_x)))),
            min(height - 1, max(0, int(round(y * scale_y)))),
        )

    if status.desk_mask_row_spans:
        source_mask = np.zeros((source_height, source_width), dtype=np.uint8)
        for row, start, end in status.desk_mask_row_spans:
            source_mask[row, start:end] = 255
        display_mask = (
            source_mask
            if (source_width, source_height) == (width, height)
            else cv2.resize(
                source_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        desk_color = (255, 128, 0)
        color_layer = np.empty_like(image_bgr)
        color_layer[:] = desk_color
        blended = cv2.addWeighted(image_bgr, 0.72, color_layer, 0.28, 0.0)
        np.copyto(image_bgr, blended, where=(display_mask > 0)[..., None])
        contours, _ = cv2.findContours(
            display_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            image_bgr,
            contours,
            -1,
            desk_color,
            2,
            cv2.LINE_AA,
        )

    x1, y1, x2, y2 = status.target_bbox_xyxy
    target_start = point(x1, y1)
    target_end = point(x2, y2)
    target_color = (0, 255, 0)
    cv2.rectangle(
        image_bgr,
        target_start,
        target_end,
        target_color,
        3,
        cv2.LINE_AA,
    )
    if status.target_lateral_anchor_px is not None:
        lateral_anchor = point(*status.target_lateral_anchor_px)
        cv2.drawMarker(
            image_bgr,
            lateral_anchor,
            target_color,
            cv2.MARKER_CROSS,
            16,
            2,
            cv2.LINE_AA,
        )
    _outlined_text(
        image_bgr,
        "TARGET (F/R)",
        (target_start[0], max(18, target_start[1] - 7)),
        target_color,
    )

    if status.table_edge_endpoints_px is not None:
        edge_start = point(*status.table_edge_endpoints_px[0])
        edge_end = point(*status.table_edge_endpoints_px[1])
        edge_color = (0, 0, 255)
        cv2.line(
            image_bgr,
            edge_start,
            edge_end,
            edge_color,
            3,
            cv2.LINE_AA,
        )
        for endpoint in (edge_start, edge_end):
            cv2.circle(
                image_bgr,
                endpoint,
                5,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
            )
        edge_center = (
            (edge_start[0] + edge_end[0]) // 2,
            (edge_start[1] + edge_end[1]) // 2,
        )
        _outlined_text(
            image_bgr,
            "TABLE EDGE (YAW)",
            (max(0, edge_center[0] - 75), max(18, edge_center[1] - 8)),
            edge_color,
        )
    return image_bgr


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

                draw_base_pose_overlays(img_bgr, name, viewer_status)

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
