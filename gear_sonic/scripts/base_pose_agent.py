#!/usr/bin/env python3
"""Run YOLOE BasePose with raw head and RGB-estimated chest depth."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from gear_sonic.camera.calibration import DEFAULT_CAMERA_INTRINSICS_PATH


@dataclass
class BasePoseAgentConfig:
    task: str
    surface_prompt: str = "desk"
    mode: Literal[
        "raw_yoloe_servo",
        "dual_raw_yoloe_servo",
    ] = "dual_raw_yoloe_servo"
    qwenvl_model: str = "qwen3-vl-plus"
    qwenvl_base_url: str = (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    qwenvl_thinking_budget: int = 500
    qwenvl_timeout_seconds: float = 600.0
    planner_hz: float = 20.0
    final_stop_count: int = 3

    sensor_gateway_endpoint: str = "tcp://127.0.0.1:5560"
    sensor_gateway_request_timeout_ms: int = 100
    sensor_gateway_max_age_ms: float = 1000.0
    sensor_gateway_max_skew_ms: float = 5.0
    control_gateway_endpoint: str = "tcp://127.0.0.1:5565"
    control_gateway_intent_endpoint: str = "tcp://127.0.0.1:5561"

    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_timeout_ms: int = 15000
    camera_stream: str = "ego_view"
    depth_stream: str = "camera/ego_view_depth"
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
    dual_initialization_grace_s: float = 30.0
    dual_qwenvl_fallback_model: str = "qwen3-vl-8b-instruct"
    dual_rgbd_buffer_size: int = 8
    dual_rgbd_poll_hz: float = 60.0

    output_root: str = "outputs/base_pose_adjustment"
    raw_yoloe_model_path: str = "tools/yoloe26m/weights/yoloe-26m-seg.pt"
    raw_yoloe_device: str = "0"
    raw_yoloe_confidence: float = 0.25
    raw_yoloe_imgsz: int = 640
    raw_reference_update_interval_frames: int = 5
    raw_reference_update_min_confidence: float = 0.35
    raw_reference_update_min_iou: float = 0.50
    raw_servo_hz: float = 10.0
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
    raw_command_ttl_s: float = 0.15
    raw_camera_stale_s: float = 0.4
    raw_max_run_s: float = 180.0
    raw_post_stop_sample_s: float = 3.0
    raw_allow_missing_table: Literal[0, 1] = 0

    def __post_init__(self) -> None:
        self.surface_prompt = str(self.surface_prompt).strip()
        if not self.surface_prompt:
            raise ValueError("surface_prompt must be non-empty")
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


def main(config: BasePoseAgentConfig) -> None:
    from gear_sonic.scripts.base_pose_yolo_agent import run_base_pose_yolo_agent

    run_base_pose_yolo_agent(config)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(BasePoseAgentConfig))
