#!/usr/bin/env python3
"""Display existing Gateway visualization frames in one OpenCV window."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Mapping

import cv2
import numpy as np

from gear_sonic.runtime.client import SensorGatewayClient
from gear_sonic.runtime.contracts import OperatorCommand
from gear_sonic.runtime.control_client import ControlGatewaySubscriber
from gear_sonic.runtime.control_gateway import BASE_POSE_RUNTIME_STATUS_COMMAND
from gear_sonic.runtime.snapshot import SnapshotRequest
from gear_sonic.runtime.visualization import VISUALIZATION_STREAMS


NAVIGATION_STREAM = "visualization/navdp_navigation"
HEAD_RGBD_STREAM = "visualization/navdp_head_rgbd"
LINGBOT_STREAM = "visualization/lingbot_depth"
HEAD_RGB_STREAM = "camera/ego_view"
CHEST_RGB_STREAM = "camera/chest_view"
LEFT_WRIST_RGB_STREAM = "camera/left_wrist"
RIGHT_WRIST_RGB_STREAM = "camera/right_wrist"
CAMERA_RGB_STREAMS = (
    HEAD_RGB_STREAM,
    CHEST_RGB_STREAM,
    LEFT_WRIST_RGB_STREAM,
    RIGHT_WRIST_RGB_STREAM,
)
DISPLAY_STREAMS = VISUALIZATION_STREAMS + CAMERA_RGB_STREAMS
BASE_POSE_ACTIVE_COLOR = (32, 178, 255)
BASE_POSE_TERMINAL_STATES = frozenset({"reached", "failed", "stopped"})


def _camera_stream_name(value: object) -> str | None:
    name = str(value or "").strip()
    if not name:
        return None
    return name if name.startswith("camera/") else f"camera/{name}"


@dataclass
class BasePoseViewerState:
    """Latest read-only BasePose state mirrored by ControlGateway."""

    active: bool = False
    generation: int = 0
    state: str = "idle"
    camera_stream: str | None = None
    action: str = ""
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reason: str = ""
    updated_at: float = 0.0

    def accept(self, command: OperatorCommand, *, now: float | None = None) -> bool:
        """Apply one typed observer event, ignoring unrelated or stale commands."""

        if command.name not in {
            "start_base_pose",
            "cancel_navigation",
            BASE_POSE_RUNTIME_STATUS_COMMAND,
        }:
            return False
        parameters = command.parameters
        try:
            generation = int(parameters.get("generation", self.generation))
        except (TypeError, ValueError):
            return False
        if generation < self.generation:
            return False
        timestamp = time.monotonic() if now is None else float(now)
        if command.name == "start_base_pose":
            self.active = True
            self.generation = generation
            self.state = "inference"
            self.camera_stream = None
            self.action = "detecting"
            self.velocity = (0.0, 0.0, 0.0)
            self.reason = ""
            self.updated_at = timestamp
            return True
        if command.name == "cancel_navigation":
            if not self.active:
                return False
            self.active = False
            self.generation = generation
            self.state = "stopped"
            self.action = "stop"
            self.velocity = (0.0, 0.0, 0.0)
            self.reason = str(parameters.get("reason", "cancelled"))
            self.updated_at = timestamp
            return True

        raw_velocity = parameters.get("velocity", (0.0, 0.0, 0.0))
        try:
            if isinstance(raw_velocity, Mapping):
                velocity = tuple(
                    float(raw_velocity[name]) for name in ("vx", "vy", "wz")
                )
            elif isinstance(raw_velocity, (list, tuple)):
                velocity = tuple(float(value) for value in raw_velocity)
            else:
                return False
            if len(velocity) != 3:
                return False
        except (KeyError, TypeError, ValueError):
            return False
        state = str(parameters.get("state", "motion"))
        camera_stream = _camera_stream_name(parameters.get("camera_stream"))
        self.active = state not in BASE_POSE_TERMINAL_STATES
        self.generation = generation
        self.state = state
        if camera_stream is not None:
            self.camera_stream = camera_stream
        self.action = str(
            parameters.get(
                "action",
                "stop" if state in BASE_POSE_TERMINAL_STATES else self.action,
            )
        )
        self.velocity = (velocity[0], velocity[1], velocity[2])
        self.reason = str(parameters.get("reason", ""))
        self.updated_at = timestamp
        return True

    def is_active_camera(self, stream: str) -> bool:
        return self.active and self.camera_stream == stream

    def status_text(self) -> str:
        camera = (
            "HEAD"
            if self.camera_stream == HEAD_RGB_STREAM
            else "CHEST"
            if self.camera_stream == CHEST_RGB_STREAM
            else "WAITING CAMERA"
        )
        vx, vy, wz = self.velocity
        parts = [
            f"BASEPOSE G{self.generation}",
            self.state.upper(),
            camera,
        ]
        if self.action:
            parts.append(self.action)
        parts.append(f"vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}")
        if self.reason:
            parts.append(self.reason)
        return " | ".join(parts)


def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize without cropping or adding content inside the source frame."""

    output = np.zeros((height, width, 3), dtype=np.uint8)
    if frame.size == 0:
        return output
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, int(round(frame.shape[1] * scale)))
    resized_height = max(1, int(round(frame.shape[0] * scale)))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    output[y : y + resized_height, x : x + resized_width] = resized
    return output


def _labeled_letterbox(
    frame: np.ndarray,
    width: int,
    height: int,
    label: str,
    *,
    active: bool = False,
) -> np.ndarray:
    """Add a title outside the source image so camera identity stays clear."""

    if frame.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if height < 28:
        return _letterbox(frame, width, height)
    title_height = 26
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[title_height:] = _letterbox(frame, width, height - title_height)
    cv2.putText(
        output,
        f"{label} | BASEPOSE ACTIVE" if active else label,
        (10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BASE_POSE_ACTIVE_COLOR if active else (235, 240, 248),
        1,
        cv2.LINE_AA,
    )
    if active:
        cv2.rectangle(
            output,
            (1, 1),
            (width - 2, height - 2),
            BASE_POSE_ACTIVE_COLOR,
            2,
        )
    return output


def _draw_base_pose_status(
    canvas: np.ndarray,
    state: BasePoseViewerState | None,
) -> None:
    if state is None or state.state == "idle":
        return
    bar_height = min(34, canvas.shape[0])
    overlay = canvas[:bar_height].copy()
    overlay[:] = (22, 28, 36)
    cv2.addWeighted(
        overlay,
        0.82,
        canvas[:bar_height],
        0.18,
        0.0,
        canvas[:bar_height],
    )
    color = BASE_POSE_ACTIVE_COLOR if state.active else (110, 190, 110)
    cv2.putText(
        canvas,
        state.status_text(),
        (12, min(24, bar_height - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        1,
        cv2.LINE_AA,
    )


def compose_visualization_canvas(
    frames: Mapping[str, np.ndarray],
    *,
    width: int = 1600,
    height: int = 900,
    base_pose: BasePoseViewerState | None = None,
) -> np.ndarray:
    """Place navigation, depth, chest, and wrist views into one canvas."""

    if width < 2 or height < 2:
        raise ValueError("viewer dimensions must be at least 2x2")
    top_height = height // 2
    remaining_height = height - top_height
    middle_height = remaining_height // 2
    bottom_height = remaining_height - middle_height
    left_width = width // 2
    right_width = width - left_width
    camera_widths = [width // 4] * 3
    camera_widths.append(width - sum(camera_widths))
    empty = np.empty((0, 0, 3), dtype=np.uint8)
    navigation = _letterbox(
        frames.get(NAVIGATION_STREAM, empty), width, top_height
    )
    head = _letterbox(
        frames.get(HEAD_RGBD_STREAM, empty), left_width, middle_height
    )
    lingbot = _letterbox(
        frames.get(LINGBOT_STREAM, empty), right_width, middle_height
    )
    head_rgb = _labeled_letterbox(
        frames.get(HEAD_RGB_STREAM, empty),
        camera_widths[0],
        bottom_height,
        "HEAD RGB",
        active=(
            base_pose is not None
            and base_pose.is_active_camera(HEAD_RGB_STREAM)
        ),
    )
    chest = _labeled_letterbox(
        frames.get(CHEST_RGB_STREAM, empty),
        camera_widths[1],
        bottom_height,
        "CHEST RGB",
        active=(
            base_pose is not None
            and base_pose.is_active_camera(CHEST_RGB_STREAM)
        ),
    )
    left_wrist = _labeled_letterbox(
        frames.get(LEFT_WRIST_RGB_STREAM, empty),
        camera_widths[2],
        bottom_height,
        "LEFT WRIST RGB",
    )
    right_wrist = _labeled_letterbox(
        frames.get(RIGHT_WRIST_RGB_STREAM, empty),
        camera_widths[3],
        bottom_height,
        "RIGHT WRIST RGB",
    )
    canvas = np.vstack(
        (
            navigation,
            np.hstack((head, lingbot)),
            np.hstack((head_rgb, chest, left_wrist, right_wrist)),
        )
    )
    _draw_base_pose_status(canvas, base_pose)
    return canvas


def gateway_frame_to_bgr(stream: str, payload: np.ndarray) -> np.ndarray | None:
    """Decode one Gateway payload according to its stream wire format."""

    array = np.asarray(payload)
    if stream in VISUALIZATION_STREAMS:
        return cv2.imdecode(array.astype(np.uint8, copy=False), cv2.IMREAD_COLOR)
    if stream in CAMERA_RGB_STREAMS:
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            return None
        image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        if stream == RIGHT_WRIST_RGB_STREAM:
            return cv2.rotate(image, cv2.ROTATE_180)
        return image
    raise ValueError(f"unsupported display stream: {stream}")


def _mock_frames(frame_index: int) -> dict[str, np.ndarray]:
    sizes = {
        NAVIGATION_STREAM: (300, 900),
        HEAD_RGBD_STREAM: (260, 640),
        LINGBOT_STREAM: (260, 640),
        HEAD_RGB_STREAM: (480, 640),
        CHEST_RGB_STREAM: (480, 640),
        LEFT_WRIST_RGB_STREAM: (480, 640),
        RIGHT_WRIST_RGB_STREAM: (480, 640),
    }
    titles = {
        NAVIGATION_STREAM: "NAVDP / MID360 / FAST-LIO",
        HEAD_RGBD_STREAM: "HEAD RGB-D",
        LINGBOT_STREAM: "LINGBOT DEPTH",
        HEAD_RGB_STREAM: "HEAD RGB",
        CHEST_RGB_STREAM: "CHEST RGB",
        LEFT_WRIST_RGB_STREAM: "LEFT WRIST RGB",
        RIGHT_WRIST_RGB_STREAM: "RIGHT WRIST RGB",
    }
    frames: dict[str, np.ndarray] = {}
    for stream in DISPLAY_STREAMS:
        height, width = sizes[stream]
        frame = np.full((height, width, 3), (18, 25, 38), dtype=np.uint8)
        cv2.putText(
            frame,
            titles[stream],
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (235, 240, 248),
            2,
            cv2.LINE_AA,
        )
        x = 30 + frame_index * 7 % max(1, width - 60)
        cv2.circle(frame, (x, height // 2), 14, (40, 210, 245), -1)
        frames[stream] = frame
    return frames


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensor-gateway-endpoint", default="tcp://127.0.0.1:5560"
    )
    parser.add_argument(
        "--control-gateway-endpoint", default="tcp://127.0.0.1:5565"
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--display-hz", type=float, default=10.0)
    parser.add_argument("--max-age-ms", type=float, default=1500.0)
    parser.add_argument("--window-name", default="SONIC Unified Visualization")
    parser.add_argument("--mock", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.display_hz <= 0.0:
        raise ValueError("display-hz must be positive")
    if args.max_age_ms <= 0.0:
        raise ValueError("max-age-ms must be positive")

    client = None
    control_subscriber = None
    base_pose = BasePoseViewerState()
    if not args.mock:
        client = SensorGatewayClient(
            args.sensor_gateway_endpoint,
            request_timeout_ms=100,
        )
        control_subscriber = ControlGatewaySubscriber(args.control_gateway_endpoint)
    latest: dict[str, np.ndarray] = {}
    frame_index = 0
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window_name, args.width, args.height)
    period = 1.0 / args.display_hz
    try:
        while True:
            started = time.monotonic()
            if args.mock:
                latest = _mock_frames(frame_index)
            else:
                assert client is not None
                assert control_subscriber is not None
                while True:
                    command = control_subscriber.read_command()
                    if command is None:
                        break
                    base_pose.accept(command)
                for stream in DISPLAY_STREAMS:
                    try:
                        snapshot = client.read_snapshot(
                            SnapshotRequest(
                                streams=(stream,),
                                max_age_ms=args.max_age_ms,
                                max_skew_ms=0.0,
                            ),
                            retries=0,
                        )
                        image = gateway_frame_to_bgr(
                            stream, snapshot.arrays[stream]
                        )
                        if image is not None:
                            latest[stream] = image
                    except Exception:
                        continue

            canvas = compose_visualization_canvas(
                latest,
                width=args.width,
                height=args.height,
                base_pose=base_pose,
            )
            cv2.imshow(args.window_name, canvas)
            key = cv2.waitKey(max(1, int(1000.0 * max(0.0, period - (time.monotonic() - started)))))
            if key & 0xFF in {27, ord("q")}:
                break
            frame_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        if client is not None:
            client.close()
        if control_subscriber is not None:
            control_subscriber.close()
        cv2.destroyWindow(args.window_name)


if __name__ == "__main__":
    main()
