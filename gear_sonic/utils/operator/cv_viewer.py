#!/usr/bin/env python3
"""Display existing Gateway visualization frames in one OpenCV window."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import cv2
import numpy as np
import zmq

from gear_sonic.utils.planner_control import (
    NavigationRuntimeStatus,
    decode_navigation_runtime_status_message,
)
from gear_sonic.runtime.gateway.sensor_client import SensorGatewayClient
from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.runtime.protocol import OperatorCommand
from gear_sonic.runtime.gateway.control_client import ControlGatewaySubscriber
from gear_sonic.runtime.gateway.control import BASE_POSE_RUNTIME_STATUS_COMMAND
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest
from gear_sonic.runtime.gateway.visualization import (
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
NAVIGATION_STATUS_BAR_HEIGHT = 38
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


def parse_base_pose_viewer_overlay(
    payload: Mapping[str, Any],
) -> tuple[
    tuple[float, float, float, float] | None,
    tuple[float, float] | None,
    tuple[tuple[float, float], tuple[float, float]] | None,
    tuple[tuple[int, int, int], ...] | None,
    tuple[int, int] | None,
    str | None,
    str | None,
]:
    """Validate optional BasePose geometry carried through ControlGateway."""

    raw_overlay = payload.get("viewer_overlay")
    if raw_overlay is None:
        return None, None, None, None, None, None, None
    if not isinstance(raw_overlay, Mapping):
        raise ValueError("Base Pose viewer overlay must be an object")
    default_stream = _camera_stream_name(payload.get("camera_stream"))
    target_camera_stream = (
        _camera_stream_name(raw_overlay.get("target_camera_stream"))
        or default_stream
    )
    yaw_align_geometry_camera_stream = (
        _camera_stream_name(raw_overlay.get("yaw_align_geometry_camera_stream"))
        or default_stream
    )

    target_bbox = None
    raw_bbox = raw_overlay.get("target_bbox_xyxy")
    if raw_bbox is not None:
        bbox = _finite_tuple(
            raw_bbox,
            length=4,
            description="viewer target bbox",
        )
        target_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
        if (
            target_bbox[2] < target_bbox[0]
            or target_bbox[3] < target_bbox[1]
        ):
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

    yaw_align_edge = None
    raw_edge = raw_overlay.get("yaw_align_edge_endpoints_px")
    if raw_edge is not None:
        if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
            raise ValueError(
                "Base Pose viewer yaw-align edge must contain two endpoints"
            )
        start = _finite_tuple(
            raw_edge[0], length=2, description="viewer yaw-align edge start"
        )
        end = _finite_tuple(
            raw_edge[1], length=2, description="viewer yaw-align edge end"
        )
        yaw_align_edge = ((start[0], start[1]), (end[0], end[1]))

    image_size = None
    raw_size = raw_overlay.get("image_size")
    if raw_size is not None:
        size = _finite_tuple(
            raw_size, length=2, description="viewer image size"
        )
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0 or (width, height) != size:
            raise ValueError(
                "Base Pose viewer image size must contain positive integers"
            )
        image_size = (width, height)

    completed_yaw_align_target_mask_row_spans = None
    raw_spans = raw_overlay.get("completed_yaw_align_target_mask_row_spans")
    if raw_spans is not None:
        if image_size is None:
            raise ValueError(
                "Base Pose viewer yaw-align target mask requires an image size"
            )
        if not isinstance(raw_spans, (list, tuple)):
            raise ValueError("Base Pose viewer yaw-align target mask spans must be a list")
        parsed_spans: list[tuple[int, int, int]] = []
        for raw_span in raw_spans:
            values = _finite_tuple(
                raw_span,
                length=3,
                description="viewer yaw-align target mask row span",
            )
            row, start, end = (int(value) for value in values)
            if (row, start, end) != values:
                raise ValueError(
                    "Base Pose viewer yaw-align target mask spans must contain integers"
                )
            width, height = image_size
            if not (0 <= row < height and 0 <= start < end <= width):
                raise ValueError(
                    "Base Pose viewer yaw-align target mask span is outside the image"
                )
            parsed_spans.append((row, start, end))
        completed_yaw_align_target_mask_row_spans = tuple(parsed_spans)

    return (
        target_bbox,
        target_lateral_anchor,
        yaw_align_edge,
        completed_yaw_align_target_mask_row_spans,
        image_size,
        target_camera_stream,
        yaw_align_geometry_camera_stream,
    )


@dataclass
class NavigationViewerState:
    """Keep only the BasePose image overlay mirrored by ControlGateway."""

    active: bool = False
    generation: int = 0
    camera_stream: str | None = None
    target_bbox_xyxy: tuple[float, float, float, float] | None = None
    target_lateral_anchor_px: tuple[float, float] | None = None
    yaw_align_edge_endpoints_px: (
        tuple[tuple[float, float], tuple[float, float]] | None
    ) = None
    completed_yaw_align_target_mask_row_spans: tuple[tuple[int, int, int], ...] | None = None
    overlay_image_size: tuple[int, int] | None = None
    target_camera_stream: str | None = None
    yaw_align_geometry_camera_stream: str | None = None

    def clear_base_pose_overlay(self) -> None:
        self.target_bbox_xyxy = None
        self.target_lateral_anchor_px = None
        self.yaw_align_edge_endpoints_px = None
        self.completed_yaw_align_target_mask_row_spans = None
        self.overlay_image_size = None
        self.target_camera_stream = None
        self.yaw_align_geometry_camera_stream = None

    def set_base_pose_overlay(self, parameters: Mapping[str, Any]) -> None:
        (
            self.target_bbox_xyxy,
            self.target_lateral_anchor_px,
            self.yaw_align_edge_endpoints_px,
            self.completed_yaw_align_target_mask_row_spans,
            self.overlay_image_size,
            self.target_camera_stream,
            self.yaw_align_geometry_camera_stream,
        ) = parse_base_pose_viewer_overlay(parameters)

    def accept_control(
        self,
        command: OperatorCommand,
    ) -> bool:
        """Apply only BasePose overlay metadata mirrored by ControlGateway."""

        if command.name not in {
            "start_navigation",
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
        if command.name == "start_navigation":
            self.clear_base_pose_overlay()
            self.active = False
            self.generation = generation
            self.camera_stream = None
            return True
        if command.name == "start_base_pose":
            self.clear_base_pose_overlay()
            self.active = True
            self.generation = generation
            self.camera_stream = None
            return True
        if command.name == "cancel_navigation":
            self.clear_base_pose_overlay()
            self.active = False
            self.generation = generation
            self.camera_stream = None
            return True
        state = str(parameters.get("state", "motion"))
        self.active = state not in NAVIGATION_TERMINAL_STATES
        self.generation = generation
        if not self.active:
            self.clear_base_pose_overlay()
            self.camera_stream = None
            return True
        self.set_base_pose_overlay(parameters)
        camera_stream = _camera_stream_name(parameters.get("camera_stream"))
        if camera_stream is not None:
            self.camera_stream = camera_stream
        return True

    def is_active_camera(self, stream: str) -> bool:
        return self.active and self.camera_stream == stream


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


def navigation_status_text(status: NavigationRuntimeStatus) -> tuple[str, str]:
    """Format the executor's source-neutral final velocity status."""

    if status.source == "navdp":
        owner = "navdp"
    elif status.source == "base_pose_agent":
        owner = "basepose"
    elif status.source in {"keyboard_planner", "operator_console"}:
        owner = "wasd"
    else:
        owner = status.source
    parts = [f"{owner.upper()} G{status.generation}"]
    if owner == "wasd":
        parts.append(_wasd_key(status.requested_velocity))
    vx, vy, wz = status.velocity
    parts.append(f"OUT vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}")
    if status.requested_velocity != status.velocity:
        rvx, rvy, rwz = status.requested_velocity
        parts.append(f"REQ {rvx:+.2f}/{rvy:+.2f}/{rwz:+.2f}")
    if status.reason not in {"clear", "stopped"}:
        parts.append(status.reason)
    return owner, " | ".join(parts)


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
    state: NavigationViewerState,
) -> np.ndarray:
    """Draw the exact target and yaw-align geometry consumed by BasePose."""

    draw_target = camera_stream == (
        state.target_camera_stream or state.camera_stream
    )
    draw_yaw_align_target = camera_stream == (
        state.yaw_align_geometry_camera_stream or state.camera_stream
    )
    if (
        image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or not state.active
        or not (draw_target or draw_yaw_align_target)
        or not any(
            (
                state.target_bbox_xyxy is not None,
                state.target_lateral_anchor_px is not None,
                state.yaw_align_edge_endpoints_px is not None,
                bool(state.completed_yaw_align_target_mask_row_spans),
            )
        )
    ):
        return image_bgr
    height, width = image_bgr.shape[:2]
    source_width, source_height = state.overlay_image_size or (width, height)
    scale_x = width / source_width
    scale_y = height / source_height

    def point(x: float, y: float) -> tuple[int, int]:
        return (
            min(width - 1, max(0, int(round(x * scale_x)))),
            min(height - 1, max(0, int(round(y * scale_y)))),
        )

    if draw_yaw_align_target and state.completed_yaw_align_target_mask_row_spans:
        source_mask = np.zeros((source_height, source_width), dtype=np.uint8)
        for row, start, end in state.completed_yaw_align_target_mask_row_spans:
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
        yaw_align_target_color = (255, 128, 0)
        color_layer = np.empty_like(image_bgr)
        color_layer[:] = yaw_align_target_color
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
            yaw_align_target_color,
            2,
            cv2.LINE_AA,
        )

    target_color = (0, 255, 0)
    if draw_target and state.target_bbox_xyxy is not None:
        x1, y1, x2, y2 = state.target_bbox_xyxy
        target_start = point(x1, y1)
        target_end = point(x2, y2)
        cv2.rectangle(
            image_bgr,
            target_start,
            target_end,
            target_color,
            3,
            cv2.LINE_AA,
        )
        _outlined_text(
            image_bgr,
            "TARGET (F/R)",
            (target_start[0], max(18, target_start[1] - 7)),
            target_color,
        )
    if draw_target and state.target_lateral_anchor_px is not None:
        cv2.drawMarker(
            image_bgr,
            point(*state.target_lateral_anchor_px),
            target_color,
            cv2.MARKER_CROSS,
            16,
            2,
            cv2.LINE_AA,
        )
    if draw_yaw_align_target and state.yaw_align_edge_endpoints_px is not None:
        edge_start = point(*state.yaw_align_edge_endpoints_px[0])
        edge_end = point(*state.yaw_align_edge_endpoints_px[1])
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
            "YAW-ALIGN EDGE",
            (max(0, edge_center[0] - 75), max(18, edge_center[1] - 8)),
            edge_color,
        )
    return image_bgr


def _draw_navigation_status(
    canvas: np.ndarray,
    status: NavigationRuntimeStatus | None,
) -> None:
    if status is None or status.mode == "stop":
        return
    owner, text = navigation_status_text(status)
    bar_height = min(NAVIGATION_STATUS_BAR_HEIGHT, canvas.shape[0])
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
    color = NAVIGATION_OWNER_COLORS.get(owner, (235, 240, 248))
    cv2.putText(
        status_area,
        text,
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
    base_pose: NavigationViewerState | None = None,
    navigation: NavigationRuntimeStatus | None = None,
) -> np.ndarray:
    """Compose four navigation/manipulation panels above two body cameras."""

    if width < 2 or height < 2:
        raise ValueError("viewer dimensions must be at least 2x2")
    status_height = min(NAVIGATION_STATUS_BAR_HEIGHT, max(0, height - 2))
    content_height = height - status_height
    top_height = content_height // 2
    bottom_height = content_height - top_height
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
    head_source = frames.get(HEAD_RGB_STREAM, empty)
    chest_source = frames.get(CHEST_RGB_STREAM, empty)
    if base_pose is not None:
        if head_source.size:
            head_source = draw_base_pose_overlays(
                head_source.copy(), HEAD_RGB_STREAM, base_pose
            )
        if chest_source.size:
            chest_source = draw_base_pose_overlays(
                chest_source.copy(), CHEST_RGB_STREAM, base_pose
            )
    head_rgb = _labeled_letterbox(
        head_source,
        bottom_widths[0],
        bottom_height,
        "HEAD RGB",
        active=(
            base_pose is not None
            and base_pose.is_active_camera(HEAD_RGB_STREAM)
        ),
    )
    chest = _labeled_letterbox(
        chest_source,
        bottom_widths[1],
        bottom_height,
        "CHEST RGB",
        active=(
            base_pose is not None
            and base_pose.is_active_camera(CHEST_RGB_STREAM)
        ),
    )
    content = np.vstack(
        (
            np.hstack((actor_ray, slam_2d, left_wrist, right_wrist)),
            np.hstack((head_rgb, chest)),
        )
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:content_height] = content
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
    parser.add_argument("--profile", default="", help="Base runtime YAML profile")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--display-hz", type=float, default=10.0)
    parser.add_argument("--max-age-ms", type=float, default=1500.0)
    parser.add_argument("--window-name", default="SONIC Unified Visualization")
    parser.add_argument("--mock", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    profile = load_runtime_profile(args.profile or None, overlays=tuple(args.overlay))
    if args.display_hz <= 0.0:
        raise ValueError("display-hz must be positive")
    if args.max_age_ms <= 0.0:
        raise ValueError("max-age-ms must be positive")

    client = None
    control_subscriber = None
    navigation_status_subscriber = None
    navigation_context = None
    navigation_state = NavigationViewerState()
    navigation_status = None
    if not args.mock:
        client = SensorGatewayClient(
            profile.endpoint_uri("sensor_gateway_metadata"),
            request_timeout_ms=100,
        )
        navigation_context = zmq.Context()
        control_subscriber = ControlGatewaySubscriber(
            profile.endpoint_uri("control_gateway_dispatch"),
            context=navigation_context,
        )
        navigation_status_subscriber = navigation_context.socket(zmq.SUB)
        navigation_status_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        navigation_status_subscriber.setsockopt(zmq.RCVHWM, 1)
        navigation_status_subscriber.setsockopt(zmq.CONFLATE, 1)
        navigation_status_subscriber.setsockopt(zmq.LINGER, 0)
        navigation_status_subscriber.connect(
            profile.endpoint_uri("navigation_runtime_status")
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
                    try:
                        navigation_state.accept_control(command)
                    except ValueError:
                        continue
                while True:
                    try:
                        payload = navigation_status_subscriber.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        navigation_status = decode_navigation_runtime_status_message(payload)
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
                base_pose=navigation_state,
                navigation=navigation_status,
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
