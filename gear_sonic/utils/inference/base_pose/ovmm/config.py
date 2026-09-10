"""Explicit Stretch adaptation of the deployed BasePose controller settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class BasePoseConfig:
    target_distance_m: float = 0.65
    far_approach_cutoff_m: float = 1.0
    forward_tolerance_m: float = 0.10
    lateral_tolerance_m: float = 0.06
    yaw_tolerance_deg: float = 3.0
    stable_frames: int = 5
    missing_tolerance_frames: int = 30
    post_stop_sample_frames: int = 30
    post_stop_deviation_frames: int = 10
    min_linear_speed_m_s: float = 0.35
    max_lateral_speed_m_s: float = 0.40
    lateral_pulse_enter_m: float = 0.10
    lateral_pulse_exit_m: float = 0.10
    lateral_pulse_max_s: float = 0.40
    lateral_pulse_settle_s: float = 0.50
    lateral_pulse_sample_frames: int = 3
    min_yaw_speed_rad_s: float = 0.10
    yaw_coarse_speed_rad_s: float = 0.30
    yaw_trim_speed_rad_s: float = 0.10
    forward_recenter_yaw_speed_rad_s: float = 0.30
    horizontal_guard_fraction: float = 0.25
    horizontal_recovery_fraction: float = 0.30
    control_dt_s: float = 0.05
    max_steps: int = 200
    episode_pick_reserve_steps: int = 5
    reference_recovery_max_attempts: int = 2
    target_small_mask_fallback: bool = False
    target_small_mask_min_pixels: int = 30
    reference_track_edges: bool = False
    reference_parallel_edge_max_offset_m: float = 0.40
    reference_bearing_fallback: bool = False
    reference_bearing_after_frames: int = 10
    reference_bearing_start_distance_m: float = 0.90
    reference_bearing_support_distance_m: float = 0.35
    max_displacement_m: float = 0.25
    max_turn_degrees: float = 30.0
    instance_match_distance_m: float = 0.35
    support_match_distance_m: float = 0.75
    target_memory_max_age_frames: int = 100
    acquisition_scan_step_deg: float = 15.0
    acquisition_tilt_min_deg: float = -85.0
    acquisition_tilt_max_deg: float = 45.0
    # Explicit oracle-map experiment; old frozen benchmarks stay opt-out.
    navmesh_recovery_enabled: bool = False
    navmesh_search_resolution_m: float = 0.001
    navmesh_clearance_m: float = 0.005
    navmesh_standoff_reserve_m: float = 0.003
    navmesh_min_target_distance_m: float = 0.30
    navmesh_max_target_distance_m: float = 0.90
    navmesh_max_lateral_tolerance_m: float = 0.0
    navmesh_local_radius_m: float = 0.60
    navmesh_grid_spacing_m: float = 0.10
    navmesh_execution_tolerance_m: float = 0.003
    navmesh_max_replans: int = 3
    navmesh_reframe_after_translation: bool = True

    def __post_init__(self) -> None:
        for key in (
            "control_dt_s",
            "max_displacement_m",
            "max_turn_degrees",
            "instance_match_distance_m",
            "support_match_distance_m",
            "acquisition_scan_step_deg",
            "navmesh_search_resolution_m",
            "navmesh_clearance_m",
            "navmesh_standoff_reserve_m",
            "navmesh_min_target_distance_m",
            "navmesh_max_target_distance_m",
            "navmesh_local_radius_m",
            "navmesh_grid_spacing_m",
            "navmesh_execution_tolerance_m",
            "reference_bearing_start_distance_m",
            "reference_bearing_support_distance_m",
            "reference_parallel_edge_max_offset_m",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if (
            isinstance(self.max_steps, bool)
            or self.max_steps < 1
            or int(self.max_steps) != self.max_steps
        ):
            raise ValueError("max_steps must be a positive integer")
        if (
            self.navmesh_recovery_enabled
            and not 0 < self.navmesh_min_target_distance_m < self.target_distance_m
        ):
            raise ValueError("navmesh minimum distance must be below nominal distance")
        if self.navmesh_max_target_distance_m < self.target_distance_m:
            raise ValueError("navmesh maximum distance must be at least nominal distance")
        if (not math.isfinite(self.navmesh_max_lateral_tolerance_m)
                or self.navmesh_max_lateral_tolerance_m < 0
                or 0 < self.navmesh_max_lateral_tolerance_m < self.lateral_tolerance_m):
            raise ValueError("navmesh lateral expansion must be disabled or at least the nominal tolerance")
        for key in ("episode_pick_reserve_steps", "reference_recovery_max_attempts"):
            value = getattr(self, key)
            if isinstance(value, bool) or value < 0 or int(value) != value:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in ("target_small_mask_min_pixels", "reference_bearing_after_frames"):
            value = getattr(self, key)
            if isinstance(value, bool) or value < 1 or int(value) != value:
                raise ValueError(f"{key} must be a positive integer")
        if (
            self.navmesh_recovery_enabled
            and self.navmesh_execution_tolerance_m >= self.lateral_tolerance_m
        ):
            raise ValueError(
                "Recovery pose tolerance must leave a lateral safety reserve"
            )
        if (
            self.navmesh_max_replans < 1
            or int(self.navmesh_max_replans) != self.navmesh_max_replans
        ):
            raise ValueError("navmesh_max_replans must be a positive integer")
        if (
            isinstance(self.target_memory_max_age_frames, bool)
            or int(self.target_memory_max_age_frames)
            != self.target_memory_max_age_frames
            or self.target_memory_max_age_frames < 1
        ):
            raise ValueError("target_memory_max_age_frames must be a positive integer")
        if not (
            -87.6
            <= self.acquisition_tilt_min_deg
            < self.acquisition_tilt_max_deg
            <= 45.2
        ):
            raise ValueError(
                "Acquisition tilt range must respect the Stretch joint limits"
            )

    def controller_kwargs(self) -> dict:
        adapter_only = {
            "control_dt_s",
            "max_steps",
            "episode_pick_reserve_steps",
            "reference_recovery_max_attempts",
            "target_small_mask_fallback",
            "target_small_mask_min_pixels",
            "reference_track_edges",
            "reference_parallel_edge_max_offset_m",
            "reference_bearing_fallback",
            "reference_bearing_after_frames",
            "reference_bearing_start_distance_m",
            "reference_bearing_support_distance_m",
            "max_displacement_m",
            "max_turn_degrees",
            "instance_match_distance_m",
            "support_match_distance_m",
            "target_memory_max_age_frames",
            "acquisition_scan_step_deg",
            "acquisition_tilt_min_deg",
            "acquisition_tilt_max_deg",
        }
        return {
            k: v
            for k, v in asdict(self).items()
            if k not in adapter_only and not k.startswith("navmesh_")
        }
