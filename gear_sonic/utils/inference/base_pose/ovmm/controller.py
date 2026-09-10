"""One fresh OVMM observation -> one existing servo update -> one waypoint."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import time

import numpy as np

from gear_sonic.utils.inference.base_pose.servo import (
    ServoPhase,
    VisualServoController,
    _observation,
)
from gear_sonic.utils.inference.base_pose.types import BasePoseCameraError
from .actions import ActionLimits, ResidualActionAdapter
from .config import BasePoseConfig
from .observation import adapt_observation
from .navmesh import RecoveryExecution
from .targets import MissingTarget, SemanticTargetProvider
from .geometry import ReferenceEdgeTracker, target_bearing_reference


class AlignmentStatus(str, Enum):
    RUNNING = "RUNNING"
    READY = "READY"
    LOST_TARGET = "LOST_TARGET"
    TIMEOUT = "TIMEOUT"
    FAILED = "FAILED"


@dataclass(frozen=True)
class AlignmentStep:
    xyt: np.ndarray
    status: AlignmentStatus
    diagnostics: dict

    @property
    def terminate(self) -> bool:
        return self.status is not AlignmentStatus.RUNNING


class BasePoseSession:
    def __init__(
        self,
        config: BasePoseConfig,
        limits: ActionLimits,
        *,
        use_opencv_camera_pose: bool = False,
    ):
        self.config = config
        self.controller = VisualServoController(**config.controller_kwargs())
        self.actions = ResidualActionAdapter(
            limits,
            dt_s=config.control_dt_s,
            max_displacement_m=config.max_displacement_m,
            max_turn_degrees=config.max_turn_degrees,
        )
        self.targets = SemanticTargetProvider(
            match_distance_m=config.instance_match_distance_m,
            support_distance_m=config.support_match_distance_m,
            small_mask_fallback=config.target_small_mask_fallback,
            small_mask_min_pixels=config.target_small_mask_min_pixels,
        )
        self.reference_tracker = ReferenceEdgeTracker(config.reference_parallel_edge_max_offset_m)
        self.use_opencv_camera_pose = use_opencv_camera_pose
        self.recovery = (
            RecoveryExecution(config, limits)
            if config.navmesh_recovery_enabled
            else None
        )
        self.reset()

    def reset(self, start_frame_id: int | None = None) -> None:
        self.start_frame_id = start_frame_id
        self.last_frame_id = None
        self.controller.reset(0.0, initial_phase=ServoPhase.FORWARD_APPROACH)
        self.actions.reset()
        self.targets.reset()
        self.status = AlignmentStatus.RUNNING
        self.last_result = None
        self.last_observation = None
        self.last_targets = None
        self.last_gps = None
        self.travel_m = 0.0
        self.reason = None
        self.valid_frames = 0
        self.yaw_frames = 0
        self.updates = 0
        self.reference_missing_frames = 0
        self.reference_recovery_requested = False
        self.reference_tracker.reset()
        self.reference_edge_missing_frames = 0
        self.bearing_fallback_active = False
        self.geometry_diagnostics = {}
        self._current_obs = None
        if self.recovery is not None:
            self.recovery.reset()
        self.controller.target_distance_m = self.config.target_distance_m
        self.controller.lateral_tolerance_m = self.config.lateral_tolerance_m

    def bind_navmesh(self, provider):
        if self.recovery is None:
            raise ValueError(
                "Explicit navmesh_recovery_enabled configuration is required"
            )
        self.recovery.provider = provider

    def observe_geometry(
        self, snapshot, targets, calibration, start_from_body, *,
        include_yaw_align_geometry=True, update_reference=False,
    ):
        """Use freshly selected target depth and explicitly identify yaw geometry."""
        observation = _observation(
            snapshot, targets.target, targets.yaw_reference, calibration,
            include_yaw_align_geometry=include_yaw_align_geometry,
            target_geometry=targets.target_geometry,
        )
        geometry = observation.yaw_align_geometry
        diagnostic = dict(source='rgb_edge' if geometry is not None else 'unavailable')
        if not include_yaw_align_geometry:
            if update_reference:
                self.reference_edge_missing_frames = 0
            diagnostic['source'] = 'distance_gated_approach'
        elif self.config.reference_track_edges:
            geometry, diagnostic = self.reference_tracker.resolve(
                observation, snapshot, calibration, start_from_body, update=update_reference,
            )
        if update_reference and include_yaw_align_geometry:
            self.reference_edge_missing_frames = (
                self.reference_edge_missing_frames+1 if geometry is None else 0
            )
        support_distance = (
            None if targets.yaw_reference_position_start is None
            else float(np.linalg.norm(targets.target_position_start-targets.yaw_reference_position_start))
        )
        associated_support = (
            targets.yaw_reference is not None and support_distance is not None
            and support_distance <= self.config.reference_bearing_support_distance_m
        )
        if (
            update_reference and include_yaw_align_geometry
            and self.config.reference_bearing_fallback
            and self.reference_edge_missing_frames >= self.config.reference_bearing_after_frames
            and targets.target_geometry.forward_m > self.config.reference_bearing_start_distance_m
            and associated_support
        ):
            self.bearing_fallback_active = True
        if (
            include_yaw_align_geometry and self.bearing_fallback_active
            and associated_support
        ):
            geometry = target_bearing_reference(targets)
            diagnostic = dict(source='rgbd_target_bearing' if geometry is not None else 'unavailable')
        self.geometry_diagnostics = dict(
            yaw_geometry_source=diagnostic['source'],
            reference_edge_tracking=diagnostic,
            reference_edge_missing_frames=self.reference_edge_missing_frames,
            bearing_fallback_active=self.bearing_fallback_active,
            bearing_reference_support_distance_m=support_distance,
            target_geometry_source=targets.target_geometry_source,
        )
        return replace(
            observation, yaw_align_geometry=geometry,
            yaw_align_geometry_error=(None if geometry is not None else
                diagnostic.get('reason', observation.yaw_align_geometry_error)),
            yaw_align_candidate_depth_stats=(geometry.candidate_depth_stats
                if geometry is not None else observation.yaw_align_candidate_depth_stats),
        )

    def step(self, obs, *, frame_id: int) -> AlignmentStep:
        started = time.perf_counter()
        frame_id = int(frame_id)
        if self.start_frame_id is None:
            self.start_frame_id = frame_id
        if self.last_frame_id is not None and frame_id <= self.last_frame_id:
            if frame_id < self.last_frame_id:
                raise ValueError(
                    "Environment frame IDs must increase; reset between episodes"
                )
            # Repeated calls must not replay a movement or count a stable frame.
            return AlignmentStep(
                np.zeros(3),
                self.status,
                {**self.last_result.diagnostics, "duplicate_frame": True},
            )
        self.last_frame_id = frame_id
        self._current_obs = obs
        steps = frame_id - self.start_frame_id
        now = steps * self.config.control_dt_s
        gps = np.asarray(obs.gps, dtype=float).reshape(-1)
        if gps.size == 2 and np.isfinite(gps).all():
            if self.last_gps is not None:
                self.travel_m += float(np.linalg.norm(gps - self.last_gps))
            self.last_gps = gps.copy()
        if self.status is not AlignmentStatus.RUNNING:
            return self._result(np.zeros(3), steps, started)
        if steps >= self.config.max_steps:
            self.status, self.reason = (
                AlignmentStatus.TIMEOUT,
                "maximum alignment environment steps reached",
            )
            self.actions.reset()
            return self._result(np.zeros(3), steps, started)
        if self.reference_recovery_requested:
            return self._result(np.zeros(3), steps, started)

        if self.recovery is not None:
            if self.recovery.provider is None:
                raise ValueError(
                    "Navmesh recovery is enabled but no explicit provider is bound"
                )
            self.recovery.observe(obs, frame_id)
            if self.recovery.completed:
                self.controller.reset(now, initial_phase=ServoPhase.FORWARD_APPROACH)
                self.actions.reset()
        self.last_observation = None
        self.last_targets = None
        self.geometry_diagnostics = dict(yaw_geometry_source='unavailable')
        heading = np.asarray(obs.compass, dtype=float).reshape(-1)
        orientation = None
        if heading.size == 1 and np.isfinite(heading).all():
            # A Habitat waypoint has completed before the next observation.
            # There is no asynchronous SONIC heading setpoint still in flight.
            orientation = {
                "actual_heading_rad": float(heading[0]),
                "heading_setpoint_rad": float(heading[0]),
                "state_age_s": 0.0,
                "telemetry_age_s": 0.0,
            }
        try:
            snapshot, calibration, start_from_body = adapt_observation(
                obs, timestamp=now, use_opencv_camera_pose=self.use_opencv_camera_pose
            )
            targets = self.targets.get_alignment_targets(
                obs, "pick", snapshot, calibration, start_from_body
            )
            observation = self.observe_geometry(
                snapshot, targets, calibration, start_from_body,
                include_yaw_align_geometry=targets.target_geometry.forward_m
                <= self.config.far_approach_cutoff_m,
                update_reference=True,
            )
            self.last_targets, self.last_observation = targets, observation
            self.valid_frames += 1
            self.yaw_frames += int(observation.yaw_align_geometry is not None)
            self.reason = observation.yaw_align_geometry_error or targets.reason
            if (
                self.recovery is not None
                and self.recovery.completed
                and self.config.navmesh_reframe_after_translation
                and (
                    observation.yaw_align_geometry is None
                    or abs(observation.yaw_align_geometry.yaw_error_rad)
                    > math.radians(self.config.yaw_tolerance_deg)
                )
            ):
                # Translation can move the previously aligned edge outside
                # the image, causing nearest-edge selection to change. Request
                # a real head observation; never substitute the old yaw.
                return self._request_reference_recovery(
                    "recovery endpoint needs fresh reference-edge framing", steps, started
                )
            if self.recovery is not None:
                recovery_xyt = self.recovery.command(
                    obs,
                    frame_id,
                    targets.target_geometry.forward_m,
                    targets.target_geometry.right_m,
                    observation.yaw_align_geometry.yaw_error_rad
                    if observation.yaw_align_geometry is not None else None,
                )
                self.controller.target_distance_m = self.recovery.effective_distance_m
                self.controller.lateral_tolerance_m = self.recovery.effective_lateral_tolerance_m
                if recovery_xyt is not None:
                    # Freeze the original servo until actual waypoints finish.
                    # Discard pre-recovery stability/filter/residual history.
                    self.controller.reset(
                        now, initial_phase=ServoPhase.FORWARD_APPROACH
                    )
                    self.actions.reset()
                    self.reference_missing_frames = 0
                    if self.recovery.failure:
                        self.status, self.reason = (
                            AlignmentStatus.FAILED,
                            self.recovery.failure,
                        )
                    return self._result(recovery_xyt, steps, started)
            reference_needed = (
                targets.target_geometry.forward_m <= self.config.far_approach_cutoff_m
                and self.controller.phase in {
                    # The first near-range approach observation transitions to
                    # yaw alignment and already counts as invalid in the core.
                    ServoPhase.FORWARD_APPROACH,
                    ServoPhase.YAW_ALIGN, ServoPhase.YAW_TRIM, ServoPhase.RECENTER,
                    ServoPhase.GLOBAL_YAW_ALIGN, ServoPhase.TRANSLATE_TARGET,
                }
            )
            self.reference_missing_frames = (
                self.reference_missing_frames + 1
                if reference_needed and observation.yaw_align_geometry is None else 0
            )
            if self.reference_missing_frames >= self.config.missing_tolerance_frames:
                return self._request_reference_recovery(
                    "target visible but required reference edge missing", steps, started
                )
            self.updates += 1
            command = self.controller.update(
                observation, now=now, orientation=orientation
            )
        except MissingTarget as exc:
            self.reference_missing_frames = 0
            self.updates += 1
            self.reason = str(exc)
            command = self.controller.note_invalid(
                self.reason, hard=False, now=now, orientation=orientation
            )
        except (BasePoseCameraError, ValueError) as exc:
            self.updates += 1
            self.reason = str(exc)
            command = self.controller.note_invalid(
                self.reason, hard=True, now=now, orientation=orientation
            )
            self.status = AlignmentStatus.FAILED

        if self.controller.terminal:
            reason = self.controller.terminal_reason
            if reason == "aligned":
                # The original post-stop sampler can finish with invalid
                # frames. Preserve its behavior but never label that READY.
                errors = self.controller.last_errors
                confirmed = (
                    self.last_observation is not None
                    and self.last_observation.yaw_align_geometry is not None
                    and abs(errors[0]) <= self.config.forward_tolerance_m + 1e-12
                    and abs(errors[1]) <= self.controller.lateral_tolerance_m + 1e-12
                    and abs(errors[2])
                    <= math.radians(self.config.yaw_tolerance_deg) + 1e-12
                )
                self.status = (
                    AlignmentStatus.READY if confirmed else AlignmentStatus.LOST_TARGET
                )
                self.reason = (
                    "aligned"
                    if confirmed
                    else "core aligned, stopped pose not confirmed by a valid frame"
                )
            elif self.status is AlignmentStatus.RUNNING:
                self.status = (
                    AlignmentStatus.TIMEOUT
                    if "run time" in reason
                    else (
                        AlignmentStatus.LOST_TARGET
                        if reason.startswith("tracking lost")
                        else AlignmentStatus.FAILED
                    )
                )
                self.reason = reason
        if self.status is AlignmentStatus.RUNNING:
            xyt = self.actions.step(command, phase=self.controller.phase.value)
        else:
            self.actions.reset()
            xyt = np.zeros(3)
        return self._result(xyt, steps, started)

    def _request_reference_recovery(self, reason, steps, started):
        self.reason = reason
        self.reference_recovery_requested = True
        self.actions.reset()
        self.controller.current = self.controller._zero()
        if self.recovery is not None:
            self.recovery.cancel_for_acquisition("reference edge reacquisition")
        return self._result(np.zeros(3), steps, started)

    def restart_after_acquisition(self, frame_id):
        """A changed head pose invalidates image-yaw history; retain the budget."""
        now = (frame_id - self.start_frame_id) * self.config.control_dt_s
        self.controller.reset(now, initial_phase=ServoPhase.FORWARD_APPROACH)
        self.actions.reset()
        self.status, self.reason = AlignmentStatus.RUNNING, None
        self.last_observation = None
        self.last_targets = None
        self.reference_missing_frames = 0
        self.reference_recovery_requested = False
        self.reference_tracker.reset()
        self.reference_edge_missing_frames = 0
        if self.recovery is not None:
            self.recovery.cancel_for_acquisition()
            self.controller.target_distance_m = self.recovery.effective_distance_m
            self.controller.lateral_tolerance_m = self.recovery.effective_lateral_tolerance_m

    def finish(self, status, reason, frame_id):
        self.status, self.reason = AlignmentStatus(status), reason
        self.actions.reset()
        self.controller.current = self.controller._zero()
        self.reference_recovery_requested = False
        if self.recovery is not None:
            self.recovery.cancel_for_acquisition(reason)
            self.recovery.blocked_axis = None
        return self._result(
            np.zeros(3), frame_id - self.start_frame_id, time.perf_counter()
        )

    def _result(self, xyt, steps, started) -> AlignmentStep:
        errors = [
            float(x) if math.isfinite(x) else None for x in self.controller.last_errors
        ]
        diagnostics = {
            **self.geometry_diagnostics,
            "status": self.status.value,
            "reason": self.reason,
            "core_terminal_reason": self.controller.terminal_reason,
            "phase": self.controller.phase.value,
            "transition_reason": self.controller.last_transition_reason,
            "steps": int(steps),
            "controller_updates": self.updates,
            "logical_time_s": steps * self.config.control_dt_s,
            "forward_error_m": errors[0],
            "right_error_m": errors[1],
            "image_yaw_error_rad": errors[2],
            "reference_yaw_error_rad": errors[2],
            "yaw_source": self.controller.yaw_error_source,
            "valid_frames": self.valid_frames,
            "yaw_frames": self.yaw_frames,
            "reference_missing_frames": self.reference_missing_frames,
            "reference_recovery_requested": self.reference_recovery_requested,
            "actual_travel_m": self.travel_m,
            "post_stop_valid_frames": self.controller.post_stop_valid_sample_count,
            "post_stop_invalid_frames": self.controller.post_stop_invalid_sample_count,
            "stable_frames": self.controller.stable_frames,
            "post_stop_realign_count": self.controller.post_stop_realign_count,
            "nominal_target_distance_m": self.config.target_distance_m,
            "effective_target_distance_m": self.controller.target_distance_m,
            "nominal_lateral_tolerance_m": self.config.lateral_tolerance_m,
            "effective_lateral_tolerance_m": self.controller.lateral_tolerance_m,
            "servo_velocity": list(self.controller.current.velocity),
            "action_xyt": np.asarray(xyt).tolist(),
            "action_residual": self.actions.residual.tolist(),
            "adapter_latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        if self.recovery is not None:
            diagnostics["navmesh_recovery"] = self.recovery.diagnostics()
            if self.recovery.path:
                diagnostics["phase"] = "NAVMESH_RECOVERY"
            if self._current_obs is not None:
                self.recovery.record(
                    np.asarray(xyt), self._current_obs, self.last_frame_id
                )
        if self.last_targets is not None:
            selection = self.last_targets
            diagnostics.update(
                target_bbox=list(selection.target.bbox_xyxy),
                reference_bbox=(
                    list(selection.yaw_reference.bbox_xyxy)
                    if selection.yaw_reference is not None
                    else None
                ),
                target_position_start=selection.target_position_start.tolist(),
                reference_association_reason=selection.reason,
                reference_mask_source=selection.reference_mask_source,
                raw_forward_m=selection.target_geometry.forward_m,
                raw_right_m=selection.target_geometry.right_m,
            )
        if self.last_observation is not None:
            geometry = self.last_observation.yaw_align_geometry
            diagnostics.update(
                raw_image_yaw_rad=(geometry.yaw_error_rad if geometry is not None
                    and self.geometry_diagnostics.get('yaw_geometry_source') in {'rgb_edge', 'tracked_rgb_edge', 'parallel_rgb_edge'} else None),
                raw_reference_yaw_rad=geometry.yaw_error_rad if geometry else None,
                selected_edge_px=geometry.line_endpoints_px if geometry else None,
                yaw_candidate_count=len(
                    self.last_observation.yaw_align_candidate_depth_stats
                ),
            )
        self.last_result = AlignmentStep(np.asarray(xyt), self.status, diagnostics)
        return self.last_result
