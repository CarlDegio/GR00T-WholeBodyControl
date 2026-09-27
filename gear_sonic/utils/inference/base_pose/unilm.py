"""UniLM-Nav adapted to G1: three model calls and a fixed-goal pose tracker.

Prompt source: https://arxiv.org/html/2607.06537v2, Appendix A, Figures 5–7.
Adaptations: chest camera, task-level manipulation handoff, G1 geometry, Qwen
coordinates, geometric heading (Section 4 / Appendix B.2), no arm commands.
This module never publishes robot commands directly.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Callable, Sequence
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from gear_sonic.utils.inference.base_pose.sensor import AlignedRGBDSnapshot


class UniLMError(RuntimeError):
    pass


@dataclass
class UniLMConfig:
    enabled: bool = False
    model: str = "qwen3.8-flash"
    base_url: str = "https://ws-6yzgj1m087a053ip.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    enable_thinking: bool = False
    timeout_s: float = 180.0
    camera: str = "chest"
    camera_height_m: float = 1.0
    history_size: int = 5
    sample_hz: float = 2.0
    pose_skew_s: float = 0.1
    odometry_timeout_s: float = 0.3
    position_tolerance_m: float = 0.05
    yaw_tolerance_deg: float = 7.0
    forward_speed_m_s: float = 0.4
    min_angular_speed_rad_s: float = 0.1
    max_angular_speed_rad_s: float = 0.3
    approach_hold_s: float = 1.0
    approach_stable_s: float = 0.3
    walk_duration_s: float = 1.0
    walk_pause_s: float = 0.5
    position_settle_s: float = 1.0
    stable_s: float = 1.0
    max_goal_distance_m: float = 3.0
    urdf_path: str = "decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
    output_root: str = "outputs/unilm_nav"

    def __post_init__(self):
        if self.camera not in {"head", "chest"}:
            raise ValueError("unilm_nav.camera must be head or chest")
        for name in (
            "timeout_s",
            "camera_height_m",
            "sample_hz",
            "pose_skew_s",
            "odometry_timeout_s",
            "position_tolerance_m",
            "yaw_tolerance_deg",
            "forward_speed_m_s",
            "min_angular_speed_rad_s",
            "max_angular_speed_rad_s",
            "approach_hold_s",
            "approach_stable_s",
            "walk_duration_s",
            "walk_pause_s",
            "position_settle_s",
            "stable_s",
            "max_goal_distance_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"unilm_nav.{name} must be finite and positive")
        if self.history_size != 5:
            raise ValueError("UniLM-Nav uses a five-observation memory")
        if self.forward_speed_m_s > 0.4 or self.max_angular_speed_rad_s > 0.3:
            raise ValueError("UniLM speeds exceed the existing Gateway servo envelope")
        if self.min_angular_speed_rad_s > self.max_angular_speed_rad_s:
            raise ValueError("UniLM minimum angular speed must not exceed the maximum")
        if self.approach_stable_s > self.approach_hold_s:
            raise ValueError("UniLM approach stability window must not exceed the hold")
        if not self.model.strip() or not self.base_url.strip():
            raise ValueError("UniLM model and base_url are required")


def enabled(profile) -> bool:
    return bool(profile.components.get("unilm_nav", {}).get("enabled", False))


@dataclass(frozen=True)
class CameraCalibration:
    camera_stream: str
    depth_stream: str
    camera_pitch_deg: float
    camera_roll_deg: float
    camera_yaw_deg: float
    camera_forward_offset_m: float
    camera_lateral_offset_m: float


def camera_calibration(config: UniLMConfig, base) -> CameraCalibration:
    """Select RGB-D and its matching extrinsics from the existing dual-camera config."""
    prefix = "dual_chest_" if config.camera == "chest" else ""
    return CameraCalibration(
        camera_stream=getattr(base, f"dual_{config.camera}_camera_stream"),
        depth_stream=getattr(base, f"dual_{config.camera}_depth_stream"),
        **{
            name: getattr(base, prefix + name)
            for name in (
                "camera_pitch_deg",
                "camera_roll_deg",
                "camera_yaw_deg",
                "camera_forward_offset_m",
                "camera_lateral_offset_m",
            )
        },
    )


def wrap(angle: float) -> float:
    return math.remainder(angle, 2 * math.pi)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float

    def world(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * x - s * y, self.y + s * x + c * y


@dataclass(frozen=True)
class Observation:
    frame: AlignedRGBDSnapshot
    pose: Pose


def rotation_rpy(rpy) -> np.ndarray:
    r, p, y = map(float, rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _collision_points(link, urdf_directory: Path):
    """Read URDF collision vertices for geometry summaries (no mesh dependency)."""
    for collision in link.findall("collision"):
        geometry = collision.find("geometry")
        mesh = geometry.find("mesh")
        box = geometry.find("box")
        if mesh is not None:
            data = (urdf_directory / mesh.get("filename")).read_bytes()
            count = int.from_bytes(data[80:84], "little")
            if len(data) != 84 + 50 * count:
                raise UniLMError("G1 collision mesh must be binary STL")
            dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
            vertices = (
                np.frombuffer(data, dtype=dtype, count=count, offset=84)["vertices"].reshape(-1, 3).astype(float)
            )
            vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        elif box is not None:
            size = np.fromstring(box.get("size"), sep=" ") / 2
            vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
        else:
            continue
        origin = collision.find("origin")
        if origin is not None:
            vertices = vertices @ rotation_rpy(np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).T
            vertices += np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        yield vertices


def robot_geometry(
    path: str, camera_height: float, *, reference_head_height_m: float = 1.1
) -> tuple[dict, np.ndarray]:
    """Nominal URDF geometry; return FAST-LIO IMU-to-pelvis transform too.

    FAST-LIO's `body` is the Livox IMU, not the pelvis. The fixed MID360
    LiDAR-to-IMU translation is from the platform's mid360.yaml. Waist neutral
    and standard sensor mounting are the confirmed platform assumptions.
    Standing geometry keeps the original 1.1 m head-mount height reference;
    the selected camera height describes the observation, not a stance change.
    """
    root = ET.parse(path).getroot()
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    links = {link.get("name"): link for link in root.findall("link")}
    cache: dict[str, np.ndarray] = {}

    def transform(link):
        if link in cache:
            return cache[link]
        if link not in joints:
            return np.eye(4)
        joint = joints[link]
        origin = joint.find("origin")
        local = np.eye(4)
        if origin is not None:
            local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            local[:3, :3] = rotation_rpy(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
        cache[link] = transform(joint.find("parent").get("link")) @ local
        return cache[link]

    camera_z = transform("d435_link")[2, 3]
    if camera_z <= 0:
        raise UniLMError("URDF is missing the nominal head mount")
    pelvis_height = reference_head_height_m - camera_z
    # Collision meshes are already shipped with the platform URDF.
    collision_vertices = {name: tuple(_collision_points(link, Path(path).parent)) for name, link in links.items()}
    bounds = []
    for name, sets in collision_vertices.items():
        for vertices in sets:
            matrix = transform(name)
            points = vertices @ matrix[:3, :3].T + matrix[:3, 3]
            bounds.extend((points[:, :2].min(axis=0), points[:, :2].max(axis=0)))
    arms = {}
    for side in ("left", "right"):
        shoulder = f"{side}_shoulder_pitch_link"
        wrist = f"{side}_wrist_yaw_link"
        if shoulder not in joints or wrist not in joints:
            raise UniLMError("URDF is missing G1 arm geometry")
        length = 0.0
        node = wrist
        while node != shoulder:
            joint = joints[node]
            xyz = np.fromstring(joint.find("origin").get("xyz", "0 0 0"), sep=" ")
            length += float(np.linalg.norm(xyz))
            node = joint.find("parent").get("link")
        pos = transform(shoulder)[:3, 3].copy()
        pos[2] += pelvis_height
        hand_extent = 0.0
        wrist_inverse = np.linalg.inv(transform(wrist))
        for name, sets in collision_vertices.items():
            node = name
            while node in joints and node != wrist:
                node = joints[node].find("parent").get("link")
            if node != wrist:
                continue
            matrix = wrist_inverse @ transform(name)
            for vertices in sets:
                points = vertices @ matrix[:3, :3].T + matrix[:3, 3]
                hand_extent = max(hand_extent, float(np.linalg.norm(points, axis=1).max()))
        arms[side] = {
            "shoulder_xyz_m": pos.tolist(),
            "shoulder_to_wrist_max_chain_m": length,
            "urdf_hand_extent_m": hand_extent,
            "max_arm_reach_m": 0.5,
        }
    lidar = transform("mid360_link")
    imu_to_lidar = np.eye(4)
    imu_to_lidar[:3, 3] = (-0.011, -0.02329, 0.04412)
    pelvis_to_imu = lidar @ np.linalg.inv(imu_to_lidar)
    description = {
        "robot": "Unitree G1 humanoid, current dex1 hands; fixed standing height",
        "coordinate_frame": "ground projection of pelvis; X forward, Y left, Z up",
        "arms": arms,
        "hand_geometry_note": (
            "Maximum arm reach for base-pose planning is fixed at 0.5 m per arm; "
            "URDF chain length and hand extent are reference geometry only; no telescopic arm or lift axis."
        ),
        "nominal_body_footprint_xy_bounds_m": [np.min(bounds, axis=0).tolist(), np.max(bounds, axis=0).tolist()],
        "camera_height_m": camera_height,
        "pelvis_height_m_nominal": pelvis_height,
        "mounting_assumption": "standard URDF MID360 mount and neutral waist",
        "urdf": str(path),
    }
    return description, np.linalg.inv(pelvis_to_imu)


def pose_from_odometry(values, imu_to_base: np.ndarray) -> Pose:
    v = np.asarray(values, dtype=float).reshape(-1)
    if len(v) < 7 or not np.isfinite(v[:7]).all():
        raise UniLMError("invalid odometry")
    q = v[3:7]
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        raise UniLMError("invalid odometry quaternion")
    x, y, z, w = q / norm
    mat = np.eye(4)
    mat[:3, 3] = v[:3]
    mat[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    mat = mat @ imu_to_base
    return Pose(float(mat[0, 3]), float(mat[1, 3]), math.atan2(mat[1, 0], mat[0, 0]))


def lift_point(observation: Observation, point, config: UniLMConfig, calibration) -> np.ndarray:
    u, v = point
    if not all(
        isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a) and 0 < a < 1000
        for a in point
    ):
        raise UniLMError("grounding u,v must be inside (0,1000)")
    frame = observation.frame
    if frame.depth_raw is None or frame.depth_scale_m is None:
        raise UniLMError("selected observation has no metric depth")
    camera_config = camera_calibration(config, calibration)
    if frame.depth_aligned_to != camera_config.camera_stream:
        raise UniLMError(f"selected observation depth is not aligned to {camera_config.camera_stream}")
    if (
        not all(math.isfinite(value) for value in (frame.fx, frame.fy, frame.cx, frame.cy, frame.depth_scale_m))
        or frame.fx <= 0
        or frame.fy <= 0
        or frame.depth_scale_m <= 0
    ):
        raise UniLMError("selected observation has invalid camera calibration")
    h, w = frame.rgb.shape[:2]
    px, py = min(w - 1, int(u * w / 1000)), min(h - 1, int(v * h / 1000))
    patch = frame.depth_raw[max(0, py - 2) : py + 3, max(0, px - 2) : px + 3].astype(float) * frame.depth_scale_m
    valid = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 5.0)]
    if not len(valid):
        raise UniLMError("no valid depth at selected interaction point")
    depth = float(np.median(valid))
    camera = np.array(
        [(u * w / 1000 - frame.cx) * depth / frame.fx, (v * h / 1000 - frame.cy) * depth / frame.fy, depth]
    )
    optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    rpy = np.radians(
        [camera_config.camera_roll_deg, -camera_config.camera_pitch_deg, camera_config.camera_yaw_deg]
    )
    return rotation_rpy(rpy) @ optical @ camera + np.array(
        [camera_config.camera_forward_offset_m, camera_config.camera_lateral_offset_m, config.camera_height_m]
    )


def view_prompt(task: str, count: int) -> str:
    return f"""You are given {count} first-person RGB images from a robot's camera,
with IDs {list(range(count))} in temporal order at the bottom right.
Manipulation task: {task}
Choose exactly ONE best image ID for the first interaction needed to begin this
task. The object (pick) or receptacle (place) should be clearly identifiable,
not blurry or heavily occluded, allowing a precise interaction point. The view
should also show a feasible approach: avoid blocked paths and unreachable areas.
For a multi-step task, retain the full task context but ground its first interaction.
If the target is not visible in any image, make your best guess from the context.
Return only JSON: {{"selected_id": <integer image ID>}}."""


def grounding_prompt(task: str) -> str:
    return f"""Manipulation task: {task}
Locate the task-relevant interaction point in this selected first-person image.
For picking, select a point on the visible graspable region of the object.
For placing an already held object, select a stable unoccupied interior region
of the receptacle's support surface, avoiding edges, barriers and other objects.
The point should be visible and accessible for both base approach and arm
interaction; avoid regions hidden by clutter or separated by large barriers.
For a multi-step task, localize the first interaction needed to begin the task.
If the intended object or surface is absent, predict its most likely image location.
Use Qwen's 0–1000 coordinate space: u increases right, v increases down;
0 < u < 1000 and 0 < v < 1000. Return only JSON: {{"u": <number>, "v": <number>}}."""


def pose_prompt(task: str, target, geometry: dict) -> str:
    return f"""You are a robot base-pose planner. Use ROS coordinates: X forward,
Y left, Z up; yaw is counter-clockwise radians, zero points forward.
Manipulation task: {task}
The robot in this reference image is at (0,0,0) in its local ground frame.
The red dot marks the interaction point, whose 3D position in that frame is
{json.dumps(list(target))} meters. The image and point refer to the same historical pose.
Robot configuration: {json.dumps(geometry)}
Determine a manipulation-ready relative base pose (x,y,theta). Consider the full
task and a feasible approach, visible obstacles, body clearance and arm reach.
Typically place the interaction point within 50–60% of the relevant arm's maximum
reach. The robot should face the interaction point. The pose must avoid collisions.
G1 has articulated arms; its existing VLA handles all manipulation. Keep its
standing height fixed. Return only JSON with x,y in meters and theta in radians:
{{"x": <number>, "y": <number>, "theta": <number>}}."""


class UniLMClient:
    def __init__(self, config: UniLMConfig, client=None):
        self.config = config
        if client is None:
            from openai import OpenAI

            key = os.environ.get("UNILM_NAV_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
            if not key:
                raise UniLMError("UniLM requires DASHSCOPE_API_KEY (or UNILM_NAV_API_KEY)")
            client = OpenAI(api_key=key, base_url=config.base_url, max_retries=0)
        self.client = client

    def call(
        self,
        prompt: str,
        images,
        validate: Callable,
        *,
        deadline: float,
        cancel: threading.Event,
        directory: Path,
        stage: str,
    ):
        content = [{"type": "text", "text": prompt}]
        for index, rgb in enumerate(images):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(directory / f"{stage}_{index}.png"), bgr)
            ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise UniLMError("image encoding failed")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()},
                }
            )
        request = dict(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": "Reason carefully and return strict JSON."
                    if self.config.enable_thinking
                    else "/no_think\nReturn strict JSON.",
                },
                {"role": "user", "content": content},
            ],
            extra_body={"enable_thinking": self.config.enable_thinking},
            response_format={"type": "json_object"},
        )
        for attempt in range(3):
            if cancel.is_set():
                raise UniLMError("cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise UniLMError("unilm_timeout")
            started = time.monotonic()
            record = {"stage": stage, "attempt": attempt + 1, "request": request}
            try:
                result = self.client.chat.completions.create(**request, timeout=min(60, remaining))
                record["response"] = result.model_dump() if hasattr(result, "model_dump") else str(result)
                parsed = validate(json.loads(result.choices[0].message.content))
                if cancel.is_set() or time.monotonic() >= deadline:
                    raise UniLMError("cancelled_or_timed_out")
                return parsed
            except Exception as exc:
                record["error"] = str(exc)
                if attempt == 2 or cancel.is_set():
                    raise UniLMError(f"{stage}_failed:{exc}") from exc
                if cancel.wait(min(1.0, max(0.0, deadline - time.monotonic()))):
                    raise UniLMError("cancelled")
            finally:
                record["elapsed_s"] = time.monotonic() - started
                (directory / f"{stage}_attempt_{attempt + 1}.json").write_text(
                    json.dumps(record, indent=2, ensure_ascii=False)
                )

    def infer(
        self,
        observations: Sequence[Observation],
        task: str,
        geometry: dict,
        calibration,
        *,
        deadline: float,
        cancel: threading.Event,
        directory: Path,
    ) -> tuple[Pose, dict]:
        if not observations:
            raise UniLMError("navigation_history_missing")
        views = []
        for i, obs in enumerate(observations):
            rgb = obs.frame.rgb.copy()
            cv2.putText(
                rgb, str(i), (rgb.shape[1] - 45, rgb.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3
            )
            views.append(rgb)

        def select(value):
            index = value["selected_id"]
            if type(index) is not int or not 0 <= index < len(views):
                raise ValueError("selected_id is out of range")
            return index

        index = self.call(
            view_prompt(task, len(views)),
            views,
            select,
            deadline=deadline,
            cancel=cancel,
            directory=directory,
            stage="view",
        )
        obs = observations[index]

        def point(value):
            values = (value["u"], value["v"])
            if not all(type(a) in (int, float) and math.isfinite(a) and 0 < a < 1000 for a in values):
                raise ValueError("u,v must be numbers inside (0,1000)")
            return values

        uv = self.call(
            grounding_prompt(task),
            [obs.frame.rgb],
            point,
            deadline=deadline,
            cancel=cancel,
            directory=directory,
            stage="grounding",
        )
        target = lift_point(obs, uv, self.config, calibration)
        marked = obs.frame.rgb.copy()
        h, w = marked.shape[:2]
        cv2.circle(marked, (int(uv[0] * w / 1000), int(uv[1] * h / 1000)), 6, (255, 0, 0), -1)

        def base(value):
            result = tuple(value[k] for k in ("x", "y", "theta"))
            if not all(type(a) in (int, float) and math.isfinite(a) for a in result):
                raise ValueError("base pose must contain finite numbers")
            return result

        x, y, model_theta = self.call(
            pose_prompt(task, target, geometry),
            [marked],
            base,
            deadline=deadline,
            cancel=cancel,
            directory=directory,
            stage="pose",
        )
        if math.hypot(target[0] - x, target[1] - y) < 1e-6:
            raise UniLMError("undefined_target_heading")
        theta = math.atan2(target[1] - y, target[0] - x)
        wx, wy = obs.pose.world(x, y)
        goal = Pose(wx, wy, wrap(obs.pose.yaw + theta))
        details = {
            "selected_id": index,
            "uv_1000": uv,
            "target_local_m": target.tolist(),
            "model_pose": [x, y, model_theta],
            "executed_local_yaw": theta,
            "goal_world": [goal.x, goal.y, goal.yaw],
            "geometry": geometry,
            "reference_pose": [obs.pose.x, obs.pose.y, obs.pose.yaw],
        }
        (directory / "prediction.json").write_text(json.dumps(details, indent=2))
        return goal, details


class PoseTracker:
    """Turn, verify a zero-velocity hold, walk straight, and pause to replan.

    Turn commands and readiness use the actual pose's yaw error. `heading_lag`
    is recorded for diagnostics only; a setpoint crossing the target must not
    reverse the turn while the robot is still short of the requested heading.
    Entering the position tolerance immediately stops all motion and starts
    the position-settle timer. Arrival stays latched through final yaw and
    completion, so later position drift cannot restart walking.
    """

    def __init__(self, goal: Pose, config: UniLMConfig):
        self.goal, self.config = goal, config
        self.approach_phase = "approach"
        self.approach_heading: float | None = None
        self.phase_until = 0.0
        self.approach_stable_since: float | None = None
        self.settle_until = 0.0
        self.position_acquired = False
        self.stable_since: float | None = None

    def _turn_velocity(self, error: float) -> float:
        c = self.config
        tolerance = math.radians(c.yaw_tolerance_deg)
        if abs(error) < tolerance:
            return 0.0
        speed = max(c.min_angular_speed_rad_s, min(c.max_angular_speed_rad_s, 1.5 * abs(error)))
        return math.copysign(speed, error)

    def update(self, pose: Pose, now: float, heading_lag: float) -> tuple[tuple[float, float, float], bool, dict]:
        c = self.config
        dx, dy = self.goal.x - pose.x, self.goal.y - pose.y
        distance = math.hypot(dx, dy)
        tolerance = math.radians(c.yaw_tolerance_deg)
        yaw_error = wrap(self.goal.yaw - pose.yaw)
        yaw_ready = abs(yaw_error) <= tolerance
        # Acquire position before choosing any motion for this control cycle.
        # Later XY drift is diagnostic only and cannot restart the approach.
        if not self.position_acquired and distance < c.position_tolerance_m:
            self.position_acquired = True
            self.settle_until = now + c.position_settle_s
        if self.approach_heading is None:
            self.approach_heading = math.atan2(dy, dx)
        # Freeze the approach heading from the pose at the end of each pause.
        # In particular, do not use a pose observed while still walking.
        if not self.position_acquired and self.approach_phase == "walk_pause" and now >= self.phase_until:
            self.approach_heading = math.atan2(dy, dx)
            self.approach_phase = "approach"
        desired = self.goal.yaw if self.position_acquired else self.approach_heading
        error = wrap(desired - pose.yaw)
        wz = self._turn_velocity(error)
        diagnostics = {
            "distance_m": distance,
            "yaw_error_rad": yaw_error,
            "approach_error_rad": error,
            "heading_lag_rad": heading_lag,
            "approach_heading_rad": self.approach_heading,
            "position_acquired": self.position_acquired,
            "phase": "final_yaw" if self.position_acquired else "approach",
        }
        if self.position_acquired and now < self.settle_until:
            self.stable_since = None
            diagnostics["phase"] = "position_settling"
            return (0.0, 0.0, 0.0), False, diagnostics
        if self.position_acquired and yaw_ready:
            if self.stable_since is None:
                self.stable_since = now
            done = now - self.stable_since >= c.stable_s
            diagnostics["phase"] = "complete" if done else "settling"
            return (0.0, 0.0, 0.0), done, diagnostics
        self.stable_since = None
        if self.position_acquired:
            return (0.0, 0.0, wz), False, diagnostics
        if self.approach_phase == "walk":
            if now < self.phase_until:
                diagnostics["phase"] = "walk"
                return (c.forward_speed_m_s, 0.0, 0.0), False, diagnostics
            self.approach_phase = "walk_pause"
            self.phase_until = now + c.walk_pause_s
        if self.approach_phase == "walk_pause":
            diagnostics["phase"] = "walk_pause"
            return (0.0, 0.0, 0.0), False, diagnostics

        if self.approach_phase == "approach_settling":
            if abs(error) >= tolerance:
                self.approach_stable_since = None
            elif self.approach_stable_since is None:
                self.approach_stable_since = now
            if now < self.phase_until:
                diagnostics["phase"] = "approach_settling"
                return (0.0, 0.0, 0.0), False, diagnostics
            if (
                self.approach_stable_since is not None
                and now - self.approach_stable_since >= c.approach_stable_s - 1e-9
            ):
                self.approach_phase = "walk"
                self.phase_until = now + c.walk_duration_s
                diagnostics["phase"] = "walk"
                return (c.forward_speed_m_s, 0.0, 0.0), False, diagnostics
            # Failed the final stability window: correct again, then require
            # another complete zero-velocity hold before permitting a step.
            self.approach_phase = "approach"

        if abs(error) < tolerance:
            self.approach_phase = "approach_settling"
            self.phase_until = now + c.approach_hold_s
            self.approach_stable_since = now
            diagnostics["phase"] = "approach_settling"
            return (0.0, 0.0, 0.0), False, diagnostics
        return (0.0, 0.0, wz), False, diagnostics
