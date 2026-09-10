"""Metric RGB-D and calibrated camera/body transforms using public sensors."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from gear_sonic.utils.inference.base_pose.types import (
    AlignedRGBDSnapshot,
    BasePoseCameraError,
)


# HomeRobot's legacy camera-pose conversion permutes BOTH matrix axes to ZXY.
_P = np.eye(4)[[2, 0, 1, 3]]
# StretchRobot.base_transformation = sim_obj.transformation @ Rx(-pi/2).
# This maps forward/left/up body coordinates into Habitat's X/Y/Z frame.
_BODY_TO_HABITAT = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def body_pose(gps: np.ndarray, compass: np.ndarray) -> np.ndarray:
    xy = np.asarray(gps, dtype=np.float64).reshape(-1)
    heading = np.asarray(compass, dtype=np.float64).reshape(-1)
    if xy.size != 2 or heading.size != 1 or not np.isfinite(np.r_[xy, heading]).all():
        raise BasePoseCameraError("Expected finite public GPS (2,) and compass (1,)")
    c, s = math.cos(float(heading[0])), math.sin(float(heading[0]))
    pose = np.array(
        [
            [c, -s, 0.0, xy[0]],
            [s, c, 0.0, xy[1]],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return pose


def camera_pose_in_start_frame(
    camera_pose: np.ndarray, *, use_opencv_camera_pose: bool
) -> np.ndarray:
    pose = np.asarray(camera_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise BasePoseCameraError("Expected a finite 4x4 camera pose")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
        raise BasePoseCameraError("Camera pose is not a homogeneous transform")
    # Both branches yield CV optical coordinates -> episode-start FLU frame.
    # CameraPoseSensor is relative to the episode start, NOT global world pose.
    converted = (
        pose.copy()
        if use_opencv_camera_pose
        else (_BODY_TO_HABITAT.T @ _P.T @ pose @ _P @ _CV_TO_GL)
    )
    if not np.allclose(converted[:3, :3].T @ converted[:3, :3], np.eye(3), atol=1e-4):
        raise BasePoseCameraError("Camera rotation is not orthonormal")
    return converted


@dataclass(frozen=True)
class MetricCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    body_from_camera: np.ndarray

    def validate_snapshot(self, snapshot: AlignedRGBDSnapshot) -> None:
        depth = snapshot.depth_raw
        if depth is None or depth.dtype.kind != "f" or snapshot.depth_scale_m != 1.0:
            raise BasePoseCameraError(
                "OVMM requires floating-point metric depth with scale 1"
            )
        if depth.shape != (self.height, self.width) or snapshot.rgb.shape != (
            self.height,
            self.width,
            3,
        ):
            raise BasePoseCameraError(
                "RGB, depth, masks, and calibration must share their resolution"
            )

    def camera_to_body(self, camera_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(camera_xyz, dtype=np.float64)
        return points @ self.body_from_camera[:3, :3].T + self.body_from_camera[:3, 3]


def adapt_observation(obs, *, timestamp: float, use_opencv_camera_pose: bool = False):
    rgb = np.asarray(obs.rgb)
    depth = np.asarray(obs.depth, dtype=np.float64).copy()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise BasePoseCameraError("Expected aligned HxWx3 RGB and HxW metric depth")
    # HomeRobot replaces normalized-depth endpoints with these sentinel values.
    depth[
        (~np.isfinite(depth)) | (depth <= 0) | (depth == 10000) | (depth == 10001)
    ] = 0
    intrinsics = np.asarray(obs.camera_K, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise BasePoseCameraError("Expected a finite 3x3 camera intrinsic matrix")
    fx, fy, cx, cy = (
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )
    if fx <= 0 or fy <= 0 or not np.allclose(intrinsics[2], [0, 0, 1]):
        raise BasePoseCameraError("Invalid pinhole camera intrinsics")
    start_from_body = body_pose(obs.gps, obs.compass)
    start_from_camera = camera_pose_in_start_frame(
        obs.camera_pose, use_opencv_camera_pose=use_opencv_camera_pose
    )
    calibration = MetricCalibration(
        rgb.shape[1],
        rgb.shape[0],
        fx,
        fy,
        cx,
        cy,
        np.linalg.inv(start_from_body) @ start_from_camera,
    )
    snapshot = AlignedRGBDSnapshot(
        rgb,
        depth,
        fx,
        fy,
        cx,
        cy,
        1.0,
        "head_rgb",
        "habitat_metric_depth",
        float(timestamp),
    )
    calibration.validate_snapshot(snapshot)
    return snapshot, calibration, start_from_body
