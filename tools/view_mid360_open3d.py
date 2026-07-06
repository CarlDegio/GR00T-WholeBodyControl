#!/usr/bin/env python3
"""View Unitree G1 MID-360 PointCloud2 DDS topic with Open3D.

This subscribes to the robot-published Unitree SDK2/CycloneDDS topic, not the
raw Livox UDP packets.

Example:
    python3 tools/view_mid360_open3d.py --interface enp3s0
    python3 tools/view_mid360_open3d.py --interface enp3s0 --once
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
LOCAL_SDK2_PY = REPO_ROOT / "external_dependencies" / "unitree_sdk2_python"
if LOCAL_SDK2_PY.exists():
    sys.path.insert(0, str(LOCAL_SDK2_PY))

np: Any | None = None
ChannelFactoryInitialize: Any | None = None
ChannelSubscriber: Any | None = None
PointCloud2_: Any | None = None
Imu_: Any | None = None
HGIMUState_: Any | None = None


DEFAULT_TOPIC = "rt/utlidar/cloud_livox_mid360"
DEFAULT_IMU_TOPIC = "rt/secondary_imu"
MID360_POINT_STEP = 22


@dataclass(frozen=True)
class CloudFrame:
    seq: int
    stamp_monotonic: float
    frame_id: str
    height: int
    width: int
    point_step: int
    row_step: int
    field_summary: str
    accumulated_frames: int
    pool_seconds: float
    points: np.ndarray
    colors: np.ndarray | None


def import_numpy() -> None:
    global np

    if np is not None:
        return

    try:
        import numpy as _np
    except Exception as exc:  # pragma: no cover - depends on host install
        raise SystemExit(
            "Failed to import numpy. Install runtime visualization dependencies first:\n"
            "  python3 -m pip install numpy open3d\n"
            f"Original error: {exc}"
        ) from exc

    np = _np


def import_unitree_sdk2() -> None:
    global ChannelFactoryInitialize, ChannelSubscriber, PointCloud2_, Imu_, HGIMUState_

    if ChannelFactoryInitialize is not None:
        return

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize as _ChannelFactoryInitialize
        from unitree_sdk2py.core.channel import ChannelSubscriber as _ChannelSubscriber
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_ as _PointCloud2
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_ as _HGIMUState
        from unitree_sensor_imu import Imu_ as _Imu
    except Exception as exc:  # pragma: no cover - depends on host DDS install
        raise SystemExit(
            "Failed to import unitree_sdk2py PointCloud2 support. Install the local SDK2 Python package first:\n"
            "  python3 -m pip install -e external_dependencies/unitree_sdk2_python\n"
            f"Original error: {exc}"
        ) from exc

    ChannelFactoryInitialize = _ChannelFactoryInitialize
    ChannelSubscriber = _ChannelSubscriber
    PointCloud2_ = _PointCloud2
    Imu_ = _Imu
    HGIMUState_ = _HGIMUState


def _field_summary(msg: Any) -> str:
    return ", ".join(f"{field.name}@{field.offset}/dt{field.datatype}/n{field.count}" for field in msg.fields)


def _pointcloud_data_as_memoryview(data: Any) -> memoryview:
    import_numpy()
    if isinstance(data, memoryview):
        return data.cast("B")
    if isinstance(data, (bytes, bytearray)):
        return memoryview(data)
    return memoryview(np.asarray(data, dtype=np.uint8))


def _mid360_dtype(point_step: int, is_bigendian: bool) -> np.dtype:
    import_numpy()
    if point_step < MID360_POINT_STEP:
        raise ValueError(f"Unsupported point_step={point_step}; expected at least {MID360_POINT_STEP}")

    endian = ">" if is_bigendian else "<"
    return np.dtype(
        {
            "names": ["x", "y", "z", "intensity", "ring", "time"],
            "formats": [f"{endian}f4", f"{endian}f4", f"{endian}f4", f"{endian}f4", f"{endian}u2", f"{endian}f4"],
            "offsets": [0, 4, 8, 12, 16, 18],
            "itemsize": point_step,
        }
    )


def _structured_points_from_msg(msg: Any) -> np.ndarray:
    import_numpy()
    point_step = int(msg.point_step)
    width = int(msg.width)
    height = int(msg.height) or 1
    row_step = int(msg.row_step) or width * point_step
    raw = _pointcloud_data_as_memoryview(msg.data)
    dtype = _mid360_dtype(point_step, bool(msg.is_bigendian))

    if height == 1 or row_step == width * point_step:
        count = min(width * height, len(raw) // point_step)
        return np.frombuffer(raw, dtype=dtype, count=count)

    rows = []
    for row in range(height):
        start = row * row_step
        stop = start + width * point_step
        if stop > len(raw):
            break
        rows.append(np.frombuffer(raw[start:stop], dtype=dtype, count=width))
    if not rows:
        return np.empty((0,), dtype=dtype)
    return np.concatenate(rows)


def pointcloud2_to_open3d_arrays(
    msg: Any,
    *,
    stride: int,
    max_range: float,
    intensity_max: float,
    color_mode: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    import_numpy()
    structured = _structured_points_from_msg(msg)
    if structured.size == 0:
        return np.empty((0, 3), dtype=np.float32), None

    points = np.column_stack((structured["x"], structured["y"], structured["z"])).astype(np.float32, copy=False)
    intensity = np.asarray(structured["intensity"], dtype=np.float32)

    valid = np.isfinite(points).all(axis=1)
    if max_range > 0:
        valid &= np.linalg.norm(points, axis=1) <= max_range

    points = points[valid]
    intensity = intensity[valid]

    if stride > 1:
        points = points[::stride]
        intensity = intensity[::stride]

    colors = None
    if color_mode == "white" and len(points):
        colors = np.ones((len(points), 3), dtype=np.float32)
    elif color_mode == "intensity" and len(points):
        if intensity_max > 0:
            denom = intensity_max
        else:
            finite = intensity[np.isfinite(intensity)]
            denom = float(np.percentile(finite, 95)) if finite.size else 1.0
            denom = max(denom, 1.0)
        gray = np.clip(intensity / denom, 0.0, 1.0).astype(np.float32, copy=False)
        colors = np.repeat(gray[:, None], 3, axis=1)

    return points, colors


class Mid360CloudReceiver:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._lock = threading.Lock()
        self._latest: CloudFrame | None = None
        self._frame_records = deque()
        self._seq = 0
        self._last_print = 0.0

        import_unitree_sdk2()
        self._subscriber = ChannelSubscriber(args.topic, PointCloud2_)
        self._subscriber.Init(self._on_cloud, 1)

    def latest_after(self, seq: int) -> CloudFrame | None:
        with self._lock:
            if self._latest is None or self._latest.seq == seq:
                return None
            return self._latest

    def wait_for_frame(self, timeout: float) -> CloudFrame | None:
        deadline = time.monotonic() + timeout
        last_seq = 0
        while time.monotonic() < deadline:
            frame = self.latest_after(last_seq)
            if frame is not None:
                return frame
            time.sleep(0.01)
        return None

    def close(self) -> None:
        self._subscriber.Close()

    def _on_cloud(self, msg: Any) -> None:
        try:
            points, colors = pointcloud2_to_open3d_arrays(
                msg,
                stride=self._args.stride,
                max_range=self._args.max_range,
                intensity_max=self._args.intensity_max,
                color_mode=self._args.color_mode,
            )
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_print >= 1.0:
                print(f"[parse] failed: {exc}", file=sys.stderr)
                self._last_print = now
            return

        self._seq += 1
        now = time.monotonic()
        with self._lock:
            self._frame_records.append((now, points, colors))

            if self._args.pool_seconds > 0:
                cutoff = now - self._args.pool_seconds
                while self._frame_records and self._frame_records[0][0] < cutoff:
                    self._frame_records.popleft()

            if self._args.accumulate_frames > 0:
                while len(self._frame_records) > self._args.accumulate_frames:
                    self._frame_records.popleft()

            point_frames = tuple(record[1] for record in self._frame_records)
            color_frames = tuple(record[2] for record in self._frame_records if record[2] is not None)

            if point_frames:
                display_points = np.concatenate(point_frames, axis=0)
            else:
                display_points = np.empty((0, 3), dtype=np.float32)

            display_colors = None
            if self._args.color_mode != "none" and color_frames:
                display_colors = np.concatenate(color_frames, axis=0)

            if self._args.max_points > 0 and len(display_points) > self._args.max_points:
                keep = np.linspace(0, len(display_points) - 1, self._args.max_points, dtype=np.intp)
                display_points = display_points[keep]
                if display_colors is not None:
                    display_colors = display_colors[keep]

            accumulated_frames = len(self._frame_records)
            seq = self._seq

        frame = CloudFrame(
            seq=seq,
            stamp_monotonic=now,
            frame_id=getattr(msg.header, "frame_id", ""),
            height=int(msg.height),
            width=int(msg.width),
            point_step=int(msg.point_step),
            row_step=int(msg.row_step),
            field_summary=_field_summary(msg),
            accumulated_frames=accumulated_frames,
            pool_seconds=self._args.pool_seconds,
            points=display_points,
            colors=display_colors,
        )
        with self._lock:
            self._latest = frame

        now = time.monotonic()
        if not self._args.once and self._args.print_every > 0 and now - self._last_print >= self._args.print_every:
            print_frame_summary(frame)
            self._last_print = now


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> Any | None:
    import_numpy()

    quat = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12 or not np.isfinite(norm):
        return None

    x, y, z, w = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> Any:
    import_numpy()

    cr = float(np.cos(roll))
    sr = float(np.sin(roll))
    cp = float(np.cos(pitch))
    sp = float(np.sin(pitch))
    cy = float(np.cos(yaw))
    sy = float(np.sin(yaw))

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def acceleration_to_tilt_rotation(ax: float, ay: float, az: float, previous_rotation: Any) -> Any | None:
    import_numpy()

    accel = np.array([ax, ay, az], dtype=np.float64)
    norm = float(np.linalg.norm(accel))
    if norm < 1e-9 or not np.isfinite(norm):
        return None

    z_axis = -accel / norm
    x_axis = previous_rotation[:, 0]
    x_axis = x_axis - float(np.dot(x_axis, z_axis)) * z_axis
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-9:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = x_axis - float(np.dot(x_axis, z_axis)) * z_axis
        x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-9:
        return None
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


class ImuOrientationReceiver:
    def __init__(self, topic: str, imu_type: str, inverse: bool) -> None:
        import_numpy()
        import_unitree_sdk2()

        self._inverse = inverse
        self._imu_type = imu_type
        self._lock = threading.Lock()
        self._rotation = np.eye(3, dtype=np.float64)
        self._seq = 0
        message_type = HGIMUState_ if imu_type == "unitree_hg" else Imu_
        self._subscriber = ChannelSubscriber(topic, message_type)
        self._subscriber.Init(self._on_imu, 1)

    def latest(self) -> tuple[int, Any]:
        with self._lock:
            return self._seq, self._rotation.copy()

    def close(self) -> None:
        self._subscriber.Close()

    def _on_imu(self, msg: Any) -> None:
        rotation = self._rotation_from_msg(msg)
        if rotation is None:
            return
        if self._inverse:
            rotation = rotation.T
        with self._lock:
            self._rotation = rotation
            self._seq += 1

    def _rotation_from_msg(self, msg: Any) -> Any | None:
        if self._imu_type == "unitree_hg":
            rpy = getattr(msg, "rpy", None)
            if rpy is not None and len(rpy) >= 3 and all(np.isfinite(float(v)) for v in rpy[:3]):
                return rpy_to_rotation_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))

            quat = getattr(msg, "quaternion", None)
            if quat is not None and len(quat) >= 4:
                return quaternion_to_rotation_matrix(float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0]))
            return None

        orientation = msg.orientation
        rotation = quaternion_to_rotation_matrix(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        if rotation is not None:
            return rotation

        linear_acceleration = msg.linear_acceleration
        return acceleration_to_tilt_rotation(
            float(linear_acceleration.x),
            float(linear_acceleration.y),
            float(linear_acceleration.z),
            self._rotation,
        )


def print_frame_summary(frame: CloudFrame) -> None:
    print(
        f"[cloud #{frame.seq}] frame_id={frame.frame_id!r} "
        f"height={frame.height} width={frame.width} point_step={frame.point_step} "
        f"row_step={frame.row_step} accumulated_frames={frame.accumulated_frames} "
        f"pool_seconds={frame.pool_seconds:g} "
        f"visible_points={len(frame.points)} fields=[{frame.field_summary}]",
        flush=True,
    )


def build_display_rotation(args: argparse.Namespace) -> Any:
    import_numpy()

    z_angle = np.deg2rad(args.z_axis_angle)
    xy_angle = np.deg2rad(args.xy_axis_angle)
    cos_z = float(np.cos(z_angle))
    sin_z = float(np.sin(z_angle))
    cos_xy = float(np.cos(xy_angle))
    sin_xy = float(np.sin(xy_angle))

    tilt_z_toward_x = np.array(
        [
            [cos_z, 0.0, sin_z],
            [0.0, 1.0, 0.0],
            [-sin_z, 0.0, cos_z],
        ],
        dtype=np.float64,
    )
    rotate_xy = np.array(
        [
            [cos_xy, -sin_xy, 0.0],
            [sin_xy, cos_xy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rotate_xy @ tilt_z_toward_x


def transform_points(points: Any, rotation: Any, flip_y: bool, flip_z: bool) -> Any:
    display_points = points.astype(np.float64, copy=True)
    if flip_y and len(display_points):
        display_points[:, 1] *= -1.0
    if flip_z and len(display_points):
        display_points[:, 2] *= -1.0
    if len(display_points):
        display_points = display_points @ rotation.T
    return display_points


def transform_geometry(geometry: Any, rotation: Any) -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    geometry.transform(transform)


def create_axis_lines(o3d: Any, size: float, rotation: Any) -> Any:
    axis = o3d.geometry.LineSet()
    axis.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32))
    axis.colors = o3d.utility.Vector3dVector(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.2, 1.0],
            ],
            dtype=np.float64,
        )
    )
    update_axis_lines(o3d, axis, size, rotation)
    return axis


def update_axis_lines(o3d: Any, axis: Any, size: float, rotation: Any) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [size, 0.0, 0.0],
            [0.0, size, 0.0],
            [0.0, 0.0, size],
        ],
        dtype=np.float64,
    )
    axis.points = o3d.utility.Vector3dVector(points @ rotation.T)


def set_default_view(vis: Any, zoom: float) -> None:
    control = vis.get_view_control()
    control.set_front([0.0, 1.0, 0.0])
    control.set_up([0.0, 0.0, 1.0])
    control.set_lookat([0.0, 0.0, 0.0])
    control.set_zoom(zoom)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Subscribe to Unitree SDK2 DDS MID-360 PointCloud2 and view it with Open3D."
    )
    parser.add_argument("--interface", "-i", help="Local NIC connected to the robot, e.g. enp3s0 or eth0")
    parser.add_argument("--domain-id", type=int, default=0, help="CycloneDDS domain id. G1 lidar uses 0.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="DDS PointCloud2 topic name")
    parser.add_argument("--imu-topic", default=DEFAULT_IMU_TOPIC, help="DDS Imu topic used by --track-axis-imu.")
    parser.add_argument(
        "--axis-imu-type",
        choices=("unitree_hg", "sensor_msgs"),
        default="unitree_hg",
        help="Message type for --imu-topic. Default tracks robot rt/secondary_imu.",
    )
    parser.add_argument("--list-interfaces", action="store_true", help="Print local network interface names and exit")
    parser.add_argument("--once", action="store_true", help="Receive one cloud frame, print metadata, then exit")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for the first frame in --once mode")
    parser.add_argument(
        "--accumulate-frames",
        type=int,
        default=None,
        help=(
            "Display the most recent N received cloud frames together. "
            "Default is 1, or unlimited when --pool-seconds is set."
        ),
    )
    parser.add_argument(
        "--pool-seconds",
        type=float,
        default=0.5,
        help="Display a sliding time pool of points received in the last N seconds; <=0 disables.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Maximum accumulated points sent to Open3D; <=0 disables this limit.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth point before visualization")
    parser.add_argument(
        "--max-range",
        "--range",
        dest="max_range",
        type=float,
        default=0.0,
        help="Only keep points within this 3D radius in meters from the lidar origin; <=0 disables.",
    )
    parser.add_argument("--intensity-max", type=float, default=255.0, help="Intensity value mapped to white; <=0 auto-scales")
    parser.add_argument(
        "--color-mode",
        choices=("intensity", "white", "none"),
        default="white",
        help="Point color mode: intensity grayscale, fixed white occupancy, or Open3D default.",
    )
    parser.add_argument("--no-color", action="store_true", help="Deprecated alias for --color-mode none")
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D point size")
    parser.add_argument(
        "--axis-size",
        type=float,
        default=1.5,
        help="Robot/lidar frame axis length in meters. X=red, Y=green, Z=blue.",
    )
    parser.add_argument(
        "--axis-line-width",
        type=float,
        default=1.0,
        help="Displayed XYZ axis line width. Open3D/backend support for thick lines may vary.",
    )
    parser.add_argument("--hide-axis", action="store_true", help="Hide the robot/lidar XYZ coordinate frame.")
    parser.add_argument(
        "--track-axis-imu",
        dest="track_axis_imu",
        action="store_true",
        default=True,
        help="Subscribe to --imu-topic and rotate the displayed XYZ axis with IMU orientation/rpy data.",
    )
    parser.add_argument(
        "--no-track-axis-imu",
        dest="track_axis_imu",
        action="store_false",
        help="Keep the displayed XYZ axis fixed instead of tracking IMU orientation.",
    )
    parser.add_argument(
        "--axis-imu-inverse",
        action="store_true",
        help="Use the inverse IMU orientation for axis tracking if the displayed axis rotates opposite to the robot.",
    )
    parser.add_argument(
        "--flip-y",
        dest="flip_y",
        action="store_true",
        default=True,
        help="Flip displayed point-cloud Y coordinates only; the green Y axis stays in the original direction.",
    )
    parser.add_argument(
        "--no-flip-y",
        dest="flip_y",
        action="store_false",
        help="Do not flip displayed point-cloud Y coordinates.",
    )
    parser.add_argument(
        "--flip-z",
        dest="flip_z",
        action="store_true",
        default=True,
        help="Flip displayed point-cloud Z coordinates only; the blue Z axis stays in the original direction.",
    )
    parser.add_argument(
        "--no-flip-z",
        dest="flip_z",
        action="store_false",
        help="Do not flip displayed point-cloud Z coordinates.",
    )
    parser.add_argument(
        "--z-axis-angle",
        type=float,
        default=0.0,
        help="Display-frame Z-axis tilt angle in degrees from the initial +Z axis toward the initial +X axis.",
    )
    parser.add_argument(
        "--xy-axis-angle",
        type=float,
        default=0.0,
        help="Display-frame X/Y-plane rotation in degrees; current +X angle from the initial +X axis.",
    )
    parser.add_argument(
        "--view-zoom",
        type=float,
        default=0.7,
        help="Initial Open3D view zoom after setting +X right, +Z up, +Y into the screen.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Open3D window width")
    parser.add_argument("--height", type=int, default=800, help="Open3D window height")
    parser.add_argument("--update-hz", type=float, default=30.0, help="Maximum Open3D update rate")
    parser.add_argument("--print-every", type=float, default=1.0, help="Seconds between frame metadata prints; <=0 disables")
    return parser


def list_interfaces() -> None:
    for _index, name in socket.if_nameindex():
        print(name)


def init_dds(domain_id: int, interface: str | None) -> None:
    import_unitree_sdk2()
    if interface:
        print(f"[dds] domain={domain_id} interface={interface}")
        ChannelFactoryInitialize(domain_id, interface)
    else:
        print(f"[dds] domain={domain_id} interface=auto")
        ChannelFactoryInitialize(domain_id)


def run_open3d(
    receiver: Mid360CloudReceiver,
    imu_receiver: ImuOrientationReceiver | None,
    args: argparse.Namespace,
) -> None:
    import_numpy()
    try:
        import open3d as o3d
    except Exception as exc:  # pragma: no cover - depends on host install
        raise SystemExit(
            "Failed to import Open3D. Install it in this Python environment first:\n"
            "  python3 -m pip install open3d\n"
            f"Original error: {exc}"
        ) from exc

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Unitree G1 MID-360 DDS PointCloud2",
        width=args.width,
        height=args.height,
        visible=True,
    )
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.0, 0.0, 0.0])
    render_opt.point_size = args.point_size
    render_opt.line_width = args.axis_line_width

    display_rotation = build_display_rotation(args)
    cloud = o3d.geometry.PointCloud()
    axis = None
    last_axis_seq = -1
    if not args.hide_axis and args.axis_size > 0:
        axis_rotation = display_rotation
        if imu_receiver is not None:
            last_axis_seq, imu_rotation = imu_receiver.latest()
            axis_rotation = display_rotation @ imu_rotation
        axis = create_axis_lines(o3d, args.axis_size, axis_rotation)
        vis.add_geometry(axis)

    added = False
    last_seq = 0
    update_period = 1.0 / max(args.update_hz, 1.0)
    next_update = 0.0

    print(f"[subscribe] topic={args.topic}")
    if imu_receiver is not None:
        print(f"[subscribe] imu_topic={args.imu_topic} imu_type={args.axis_imu_type} axis_tracking=on")
    elif args.hide_axis:
        print("[axis] hidden")
    else:
        print("[axis] tracking=off")
    print("[open3d] press Ctrl+C in the terminal or close the window to stop")

    try:
        while True:
            now = time.monotonic()
            if axis is not None and imu_receiver is not None:
                axis_seq, imu_rotation = imu_receiver.latest()
                if axis_seq != last_axis_seq:
                    update_axis_lines(o3d, axis, args.axis_size, display_rotation @ imu_rotation)
                    vis.update_geometry(axis)
                    last_axis_seq = axis_seq

            frame = receiver.latest_after(last_seq)
            if frame is not None and now >= next_update:
                display_points = transform_points(frame.points, display_rotation, args.flip_y, args.flip_z)
                cloud.points = o3d.utility.Vector3dVector(display_points)
                if frame.colors is not None:
                    cloud.colors = o3d.utility.Vector3dVector(frame.colors.astype(np.float64, copy=False))
                else:
                    cloud.colors = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))

                if not added:
                    vis.add_geometry(cloud)
                    vis.reset_view_point(True)
                    set_default_view(vis, args.view_zoom)
                    added = True
                else:
                    vis.update_geometry(cloud)

                last_seq = frame.seq
                next_update = now + update_period

            if not vis.poll_events():
                break
            vis.update_renderer()
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()


def main() -> int:
    args = create_argument_parser().parse_args()
    args.stride = max(1, args.stride)
    args.pool_seconds = max(0.0, args.pool_seconds)
    if args.accumulate_frames is None:
        args.accumulate_frames = 0 if args.pool_seconds > 0 else 1
    else:
        args.accumulate_frames = max(0, args.accumulate_frames)
    if args.no_color:
        args.color_mode = "none"

    if args.list_interfaces:
        list_interfaces()
        return 0

    init_dds(args.domain_id, args.interface)
    receiver = Mid360CloudReceiver(args)
    imu_receiver = None
    if not args.once and not args.hide_axis and args.track_axis_imu:
        imu_receiver = ImuOrientationReceiver(args.imu_topic, args.axis_imu_type, args.axis_imu_inverse)
    try:
        if args.once:
            print(f"[subscribe] topic={args.topic}; waiting up to {args.timeout:.1f}s")
            frame = receiver.wait_for_frame(args.timeout)
            if frame is None:
                print(
                    "No cloud frame received. Check that this PC is on 192.168.123.0/24, "
                    "the selected --interface is the local NIC connected to the robot, "
                    "and the robot publishes rt/utlidar/cloud_livox_mid360.",
                    file=sys.stderr,
                )
                return 2
            print_frame_summary(frame)
            return 0

        run_open3d(receiver, imu_receiver, args)
    finally:
        if imu_receiver is not None:
            imu_receiver.close()
        receiver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
