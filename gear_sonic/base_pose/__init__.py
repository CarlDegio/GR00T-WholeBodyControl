"""Near-field base alignment planning and planner-mode execution primitives."""

from gear_sonic.base_pose.execution import (
    BasePoseSequenceController,
    MotionSegment,
    VelocityCommand,
    plan_to_segments,
)
from gear_sonic.base_pose.policy import (
    BASE_POSE_MODES,
    BASE_POSE_OUTPUT_SCHEMA,
    BASE_POSE_VISION_BACKENDS,
    BasePoseCameraError,
    BasePoseConfig,
    BasePoseObservation,
    BasePosePlanner,
    BasePoseResult,
    BasePoseValidationError,
    CodexStructuredVisionClient,
    QwenVLStructuredVisionClient,
    build_base_pose_prompt,
    query_depth_regions,
    validate_base_pose_plan,
    validate_base_pose_observation,
)
from gear_sonic.base_pose.sensor import SensorGatewayBasePoseCamera

__all__ = [
    "BASE_POSE_MODES",
    "BASE_POSE_OUTPUT_SCHEMA",
    "BASE_POSE_VISION_BACKENDS",
    "BasePoseCameraError",
    "BasePoseConfig",
    "BasePoseObservation",
    "BasePosePlanner",
    "BasePoseResult",
    "BasePoseSequenceController",
    "BasePoseValidationError",
    "CodexStructuredVisionClient",
    "MotionSegment",
    "QwenVLStructuredVisionClient",
    "SensorGatewayBasePoseCamera",
    "VelocityCommand",
    "build_base_pose_prompt",
    "plan_to_segments",
    "query_depth_regions",
    "validate_base_pose_plan",
    "validate_base_pose_observation",
]
