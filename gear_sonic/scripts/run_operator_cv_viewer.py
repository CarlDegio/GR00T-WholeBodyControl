#!/usr/bin/env python3
"""Display existing Gateway visualization frames in one OpenCV window."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Mapping

import cv2
import numpy as np
import zmq

from gear_sonic.planner_control import (
    NavigationRuntimeStatus,
    decode_navigation_runtime_status_message,
)
from gear_sonic.runtime.client import SensorGatewayClient
from gear_sonic.runtime.contracts import OperatorCommand
from gear_sonic.runtime.control_client import ControlGatewaySubscriber
from gear_sonic.runtime.control_gateway import BASE_POSE_RUNTIME_STATUS_COMMAND
from gear_sonic.runtime.snapshot import SnapshotRequest
from gear_sonic.runtime.visualization import (
    NAVDP_ACTOR_RAY_STREAM,
    NAVDP_SLAM_2D_STREAM,
    VISUALIZATION_STREAMS,
)


ACTOR_RAY_STREAM = NAVDP_ACTOR_RAY_STREAM
SLAM_2D_STREAM = NAVDP_SLAM_2D_STREAM
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
DISPLAY_STREAMS = (
    ACTOR_RAY_STREAM,
    SLAM_2D_STREAM,
    LEFT_WRIST_RGB_STREAM,
    RIGHT_WRIST_RGB_STREAM,
    HEAD_RGB_STREAM,
    CHEST_RGB_STREAM,
)
BASE_POSE_ACTIVE_COLOR = (0, 0, 255)
NAVIGATION_TERMINAL_STATES = frozenset({"reached", "failed", "stopped"})
NAVIGATION_OWNER_COLORS = {
    "wasd": (255, 210, 40),
    "navdp": (40, 230, 255),
    "basepose": BASE_POSE_ACTIVE_COLOR,
}
WASD_VELOCITY_KEYS = {
    (0.3, 0.0, 0.0): "W",
    (-0.3, 0.0, 0.0): "S",
    (0.0, 0.15, 0.0): "A",
    (0.0, -0.15, 0.0): "D",
    (0.0, 0.0, 0.5): "Q",
    (0.0, 0.0, -0.5): "E",
}


def _camera_stream_name(value: object) -> str | None:
    name = str(value or "").strip()
    if not name:
        return None
    return name if name.startswith("camera/") else f"camera/{name}"


@dataclass
class NavigationViewerState:
    """Merge control lifecycle events with final executor velocity telemetry."""

    owner: str = "idle"
    active: bool = False
    generation: int = 0
    state: str = "idle"
    camera_stream: str | None = None
    action: str = ""
    requested_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reason: str = ""
    safety_reason: str = "stopped"
    updated_at: float = 0.0

    def accept_control(
        self,
        command: OperatorCommand,
        *,
        now: float | None = None,
    ) -> bool:
        """Apply lifecycle/camera metadata mirrored by ControlGateway."""

        if command.name not in {
            "start_navigation",
            "start_base_pose",
            "cancel_navigation",
            "navigation_status",
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
        if command.name == "start_navigation":
            self.owner = "navdp"
            self.active = True
            self.generation = generation
            self.state = "pending"
            self.camera_stream = None
            self.action = "target_detection"
            self.requested_velocity = (0.0, 0.0, 0.0)
            self.velocity = (0.0, 0.0, 0.0)
            self.reason = ""
            self.safety_reason = "stopped"
            self.updated_at = timestamp
            return True
        if command.name == "start_base_pose":
            self.owner = "basepose"
            self.active = True
            self.generation = generation
            self.state = "inference"
            self.camera_stream = None
            self.action = "detecting"
            self.requested_velocity = (0.0, 0.0, 0.0)
            self.velocity = (0.0, 0.0, 0.0)
            self.reason = ""
            self.safety_reason = "stopped"
            self.updated_at = timestamp
            return True
        if command.name == "cancel_navigation":
            if self.owner == "idle":
                return False
            self.active = False
            self.generation = generation
            self.state = "stopped"
            self.action = "stop"
            self.requested_velocity = (0.0, 0.0, 0.0)
            self.velocity = (0.0, 0.0, 0.0)
            self.reason = str(parameters.get("reason", "cancelled"))
            self.safety_reason = "stopped"
            self.updated_at = timestamp
            return True

        if command.name == "navigation_status":
            self.owner = "navdp"
            self.generation = generation
            self.state = str(parameters.get("state", "active"))
            self.active = self.state not in NAVIGATION_TERMINAL_STATES
            self.action = "navigate" if self.active else "stop"
            self.reason = str(parameters.get("reason", ""))
            if not self.active:
                self.requested_velocity = (0.0, 0.0, 0.0)
                self.velocity = (0.0, 0.0, 0.0)
            self.updated_at = timestamp
            return True

        state = str(parameters.get("state", "motion"))
        camera_stream = _camera_stream_name(parameters.get("camera_stream"))
        self.owner = "basepose"
        self.active = state not in NAVIGATION_TERMINAL_STATES
        self.generation = generation
        self.state = state
        if camera_stream is not None:
            self.camera_stream = camera_stream
        self.action = str(
            parameters.get(
                "action",
                "stop" if state in NAVIGATION_TERMINAL_STATES else self.action,
            )
        )
        if not self.active:
            self.requested_velocity = (0.0, 0.0, 0.0)
            self.velocity = (0.0, 0.0, 0.0)
        self.reason = str(parameters.get("reason", ""))
        self.updated_at = timestamp
        return True

    def accept_runtime(
        self,
        status: NavigationRuntimeStatus,
        *,
        now: float | None = None,
    ) -> bool:
        """Apply the final command emitted by PlannerExecutor."""

        if status.generation < self.generation:
            return False
        timestamp = time.monotonic() if now is None else float(now)
        owner = "idle"
        if status.mode == "nav_goal":
            owner = "navdp"
        elif status.mode == "manual_velocity":
            owner = "basepose" if status.source == "base_pose_agent" else "wasd"
        terminal_same_generation = (
            status.generation == self.generation
            and not self.active
            and self.state in NAVIGATION_TERMINAL_STATES
        )
        protected_terminal = terminal_same_generation and owner == self.owner
        if protected_terminal and status.mode != "stop":
            return False

        if owner != "idle" and not protected_terminal:
            if owner != self.owner or status.generation > self.generation:
                self.camera_stream = None
                self.reason = ""
            self.owner = owner
            self.active = True
            self.generation = status.generation
            if owner == "wasd":
                self.action = self._wasd_key(status.requested_velocity)
                self.active = self.action != "RELEASED"
                self.state = "active" if self.action != "RELEASED" else "released"
            elif owner == "navdp":
                self.action = "navigate"
                self.state = "active"
            elif self.state in {"idle", "stopped"}:
                self.state = "motion"
        elif status.generation > self.generation:
            self.generation = status.generation
            if self.owner != "idle":
                self.active = False
                self.state = "stopped"
                self.action = "stop"
                self.reason = ""

        self.requested_velocity = status.requested_velocity
        self.velocity = status.velocity
        self.safety_reason = status.reason
        self.updated_at = timestamp
        return True

    @staticmethod
    def _wasd_key(velocity: tuple[float, float, float]) -> str:
        for expected, key in WASD_VELOCITY_KEYS.items():
            if all(
                abs(actual - target) <= 1.0e-6
                for actual, target in zip(velocity, expected)
            ):
                return key
        return (
            "RELEASED"
            if all(abs(value) <= 1.0e-6 for value in velocity)
            else "MANUAL"
        )

    def is_active_camera(self, stream: str) -> bool:
        return self.owner == "basepose" and self.active and self.camera_stream == stream

    def status_text(self) -> str:
        if self.owner == "idle":
            return ""
        vx, vy, wz = self.velocity
        parts = [
            f"{self.owner.upper()} G{self.generation}",
            self.state.upper(),
        ]
        if self.owner == "basepose":
            camera = (
                "HEAD"
                if self.camera_stream == HEAD_RGB_STREAM
                else "CHEST"
                if self.camera_stream == CHEST_RGB_STREAM
                else "WAITING CAMERA"
            )
            parts.append(camera)
        if self.action:
            parts.append(self.action)
        parts.append(f"OUT vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}")
        if any(
            abs(requested - final) > 1.0e-6
            for requested, final in zip(self.requested_velocity, self.velocity)
        ):
            rvx, rvy, rwz = self.requested_velocity
            parts.append(f"REQ {rvx:+.2f}/{rvy:+.2f}/{rwz:+.2f}")
        if self.active and self.safety_reason not in {"clear", "stopped"}:
            parts.append(self.safety_reason)
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


def _draw_navigation_status(
    canvas: np.ndarray,
    state: NavigationViewerState | None,
) -> None:
    if state is None or state.owner == "idle":
        return
    bar_height = min(34, canvas.shape[0])
    status_area = canvas[-bar_height:]
    overlay = status_area.copy()
    overlay[:] = (22, 28, 36)
    cv2.addWeighted(
        overlay,
        0.82,
        status_area,
        0.18,
        0.0,
        status_area,
    )
    color = (
        NAVIGATION_OWNER_COLORS.get(state.owner, (235, 240, 248))
        if state.active
        else (110, 190, 110)
    )
    cv2.putText(
        status_area,
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
    navigation: NavigationViewerState | None = None,
) -> np.ndarray:
    """Compose four navigation/manipulation panels above two body cameras."""

    if width < 2 or height < 2:
        raise ValueError("viewer dimensions must be at least 2x2")
    top_height = height // 2
    bottom_height = height - top_height
    top_widths = [width // 4] * 3
    top_widths.append(width - sum(top_widths))
    bottom_widths = [width // 2, width - width // 2]
    empty = np.empty((0, 0, 3), dtype=np.uint8)
    actor_ray = _letterbox(
        frames.get(ACTOR_RAY_STREAM, empty),
        top_widths[0],
        top_height,
    )
    slam_2d = _letterbox(
        frames.get(SLAM_2D_STREAM, empty),
        top_widths[1],
        top_height,
    )
    left_wrist = _labeled_letterbox(
        frames.get(LEFT_WRIST_RGB_STREAM, empty),
        top_widths[2],
        top_height,
        "LEFT WRIST RGB",
    )
    right_wrist = _labeled_letterbox(
        frames.get(RIGHT_WRIST_RGB_STREAM, empty),
        top_widths[3],
        top_height,
        "RIGHT WRIST RGB",
    )
    head_rgb = _labeled_letterbox(
        frames.get(HEAD_RGB_STREAM, empty),
        bottom_widths[0],
        bottom_height,
        "HEAD RGB",
        active=(
            navigation is not None
            and navigation.is_active_camera(HEAD_RGB_STREAM)
        ),
    )
    chest = _labeled_letterbox(
        frames.get(CHEST_RGB_STREAM, empty),
        bottom_widths[1],
        bottom_height,
        "CHEST RGB",
        active=(
            navigation is not None
            and navigation.is_active_camera(CHEST_RGB_STREAM)
        ),
    )
    canvas = np.vstack(
        (
            np.hstack((actor_ray, slam_2d, left_wrist, right_wrist)),
            np.hstack((head_rgb, chest)),
        )
    )
    _draw_navigation_status(canvas, navigation)
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
        ACTOR_RAY_STREAM: (500, 500),
        SLAM_2D_STREAM: (500, 500),
        HEAD_RGB_STREAM: (480, 640),
        CHEST_RGB_STREAM: (480, 640),
        LEFT_WRIST_RGB_STREAM: (480, 640),
        RIGHT_WRIST_RGB_STREAM: (480, 640),
    }
    titles = {
        ACTOR_RAY_STREAM: "ACTORRAY / VELOCITY",
        SLAM_2D_STREAM: "FAST-LIO SLAM 2D",
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
    parser.add_argument(
        "--navigation-runtime-status-endpoint", default="tcp://127.0.0.1:5570"
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
    navigation_status_subscriber = None
    navigation_context = None
    navigation_state = NavigationViewerState()
    if not args.mock:
        client = SensorGatewayClient(
            args.sensor_gateway_endpoint,
            request_timeout_ms=100,
        )
        navigation_context = zmq.Context()
        control_subscriber = ControlGatewaySubscriber(
            args.control_gateway_endpoint,
            context=navigation_context,
        )
        navigation_status_subscriber = navigation_context.socket(zmq.SUB)
        navigation_status_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        navigation_status_subscriber.setsockopt(zmq.RCVHWM, 1)
        navigation_status_subscriber.setsockopt(zmq.CONFLATE, 1)
        navigation_status_subscriber.setsockopt(zmq.LINGER, 0)
        navigation_status_subscriber.connect(
            args.navigation_runtime_status_endpoint
        )
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
                assert navigation_status_subscriber is not None
                while True:
                    command = control_subscriber.read_command()
                    if command is None:
                        break
                    navigation_state.accept_control(command)
                while True:
                    try:
                        payload = navigation_status_subscriber.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        navigation_state.accept_runtime(
                            decode_navigation_runtime_status_message(payload)
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
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
                navigation=navigation_state,
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
        if navigation_status_subscriber is not None:
            navigation_status_subscriber.close(0)
        if navigation_context is not None:
            navigation_context.term()
        cv2.destroyWindow(args.window_name)


if __name__ == "__main__":
    main()
