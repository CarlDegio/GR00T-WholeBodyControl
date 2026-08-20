#!/usr/bin/env python3
"""Run YOLOE BasePose with raw head and RGB-estimated chest depth."""

from __future__ import annotations

from dataclasses import dataclass
import math
from gear_sonic.camera.calibration import DEFAULT_CAMERA_INTRINSICS_PATH


@dataclass
class BasePoseAgentConfig:
    task: str
    target_prompt: str = "bluebasket"
    surface_prompt: str = "desk"
    planner_hz: float = 20.0
    final_stop_count: int = 3

    sensor_gateway_endpoint: str = "tcp://127.0.0.1:5560"
    sensor_gateway_request_timeout_ms: int = 100
    sensor_gateway_max_age_ms: float = 1000.0
    sensor_gateway_max_skew_ms: float = 5.0
    control_gateway_endpoint: str = "tcp://127.0.0.1:5565"
    control_gateway_intent_endpoint: str = "tcp://127.0.0.1:5561"

    camera_timeout_ms: int = 15000
    camera_intrinsics_path: str = str(DEFAULT_CAMERA_INTRINSICS_PATH)
    camera_pitch_deg: float = -38.0
    camera_roll_deg: float = 0.0
    camera_yaw_deg: float = 0.0
    camera_forward_offset_m: float = 0.0
    camera_lateral_offset_m: float = 0.0

    dual_head_camera_stream: str = "ego_view"
    dual_head_depth_stream: str = "camera/ego_view_depth"
    dual_chest_camera_stream: str = "chest_view"
    dual_chest_depth_stream: str = "camera/chest_view_depth"
    dual_chest_camera_pitch_deg: float = -3.0
    dual_chest_camera_roll_deg: float = 0.0
    dual_chest_camera_yaw_deg: float = 0.0
    dual_chest_camera_forward_offset_m: float = 0.0
    dual_chest_camera_lateral_offset_m: float = 0.0
    dual_match_tolerance_frames: int = 30
    dual_head_reacquire_frames: int = 1
    dual_head_release_missing_frames: int = 3
    dual_rgbd_buffer_size: int = 8
    dual_rgbd_poll_hz: float = 60.0

    output_root: str = "outputs/base_pose_adjustment"
    raw_yoloe_model_path: str = "tools/yoloe26m/weights/yoloe-26m-seg.pt"
    raw_yoloe_device: str = "0"
    raw_yoloe_confidence: float = 0.25
    raw_yoloe_imgsz: int = 640
    raw_head_target_distance_m: float = 1.00
    raw_chest_target_distance_m: float = 0.80
    raw_forward_tolerance_m: float = 0.10
    raw_lateral_tolerance_m: float = 0.10
    raw_min_linear_speed_m_s: float = 0.40
    raw_max_lateral_speed_m_s: float = 0.40
    raw_min_yaw_speed_rad_s: float = 0.10
    raw_yaw_tolerance_deg: float = 8.0
    raw_yaw_coarse_speed_rad_s: float = 0.30
    raw_yaw_trim_speed_rad_s: float = 0.20
    raw_forward_recenter_yaw_speed_rad_s: float = 0.30
    raw_horizontal_guard_fraction: float = 0.25
    raw_horizontal_recovery_fraction: float = 0.30
    raw_orientation_telemetry_source: str = "tcp://127.0.0.1:5569"
    raw_camera_stale_s: float = 0.4
    raw_max_run_s: float = 180.0
    raw_post_stop_sample_frames: int = 30
    raw_post_stop_deviation_frames: int = 10

    def __post_init__(self) -> None:
        self.target_prompt = str(self.target_prompt).strip()
        if not self.target_prompt:
            raise ValueError("target_prompt must be non-empty")
        self.surface_prompt = str(self.surface_prompt).strip()
        if not self.surface_prompt:
            raise ValueError("surface_prompt must be non-empty")
        for value, name in (
            (self.dual_match_tolerance_frames, "dual_match_tolerance_frames"),
            (self.dual_head_reacquire_frames, "dual_head_reacquire_frames"),
            (
                self.dual_head_release_missing_frames,
                "dual_head_release_missing_frames",
            ),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for value, name in (
            (self.raw_head_target_distance_m, "raw_head_target_distance_m"),
            (
                self.raw_chest_target_distance_m,
                "raw_chest_target_distance_m",
            ),
            (self.raw_forward_tolerance_m, "raw_forward_tolerance_m"),
            (self.raw_lateral_tolerance_m, "raw_lateral_tolerance_m"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.raw_post_stop_sample_frames < 0:
            raise ValueError("raw_post_stop_sample_frames must be non-negative")
        if not (
            self.raw_post_stop_sample_frames == 0
            or 0 < self.raw_post_stop_deviation_frames
            <= self.raw_post_stop_sample_frames
        ):
            raise ValueError(
                "raw_post_stop_deviation_frames must be in "
                "[1, raw_post_stop_sample_frames]"
            )


def main(config: BasePoseAgentConfig) -> None:
    from gear_sonic.scripts.base_pose_yolo_agent import run_base_pose_yolo_agent

    run_base_pose_yolo_agent(config)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(BasePoseAgentConfig))
