"""Shared planner-velocity protocol, safety, and SONIC execution primitives."""

from gear_sonic.planner_control.executor import (
    PlannerExecutorDecision,
    PlannerVelocityExecutorCore,
    SafetySnapshot,
)
from gear_sonic.planner_control.protocol import (
    COMMAND_TYPE,
    RUNTIME_STATUS_TYPE,
    STATUS_TYPE,
    NavigationCommand,
    NavigationRuntimeStatus,
    PlannerVelocityCommand,
    build_navigation_message,
    build_navigation_runtime_status_message,
    build_planner_velocity_message,
    decode_navigation_message,
    decode_navigation_runtime_status_message,
    decode_planner_velocity_message,
)
from gear_sonic.planner_control.safety import depth_requires_stop
from gear_sonic.planner_control.sonic import SonicPlannerState

__all__ = [
    "COMMAND_TYPE",
    "RUNTIME_STATUS_TYPE",
    "STATUS_TYPE",
    "NavigationCommand",
    "NavigationRuntimeStatus",
    "PlannerExecutorDecision",
    "PlannerVelocityCommand",
    "PlannerVelocityExecutorCore",
    "SafetySnapshot",
    "SonicPlannerState",
    "build_navigation_message",
    "build_navigation_runtime_status_message",
    "build_planner_velocity_message",
    "decode_navigation_message",
    "decode_navigation_runtime_status_message",
    "decode_planner_velocity_message",
    "depth_requires_stop",
]
