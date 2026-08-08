"""Canonical endpoint inventory for the current SONIC inference profile.

This module is deliberately not imported by the existing launch scripts yet.
It records the topology without changing socket ownership or runtime behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class Transport(str, Enum):
    """Transport used by one logical runtime channel."""

    ZMQ = "zmq"
    HTTP = "http"


@dataclass(frozen=True)
class EndpointSpec:
    """One logical TCP endpoint and its present socket owner."""

    name: str
    port: int
    transport: Transport
    binder: str
    connectors: tuple[str, ...]
    active: bool = True
    purpose: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("endpoint name cannot be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"invalid port for {self.name}: {self.port}")
        if not self.binder:
            raise ValueError(f"endpoint {self.name} has no socket owner")

    def uri(self, host: str = "127.0.0.1") -> str:
        scheme = "tcp" if self.transport is Transport.ZMQ else "http"
        return f"{scheme}://{host}:{self.port}"


@dataclass(frozen=True)
class RosTopicSpec:
    """ROS 2 topic required by the current navigation profile."""

    name: str
    publisher: str
    consumers: tuple[str, ...]
    purpose: str = ""

    def __post_init__(self) -> None:
        if not self.name.startswith("/"):
            raise ValueError(f"ROS topic must be absolute: {self.name}")


def _build_endpoints() -> Mapping[str, EndpointSpec]:
    endpoints = (
        EndpointSpec(
            "policy_server",
            5550,
            Transport.ZMQ,
            "Isaac-GR00T PolicyServer",
            ("run_vla_inference",),
            purpose="VLA observation request and action response",
        ),
        EndpointSpec(
            "camera_server",
            5555,
            Transport.ZMQ,
            "ComposedCamera server",
            ("run_vla_inference", "LingBot Depth", "NavDP health monitor"),
            purpose="Raw robot RGB-D and wrist camera streams",
        ),
        EndpointSpec(
            "cpp_command",
            5556,
            Transport.ZMQ,
            "run_vla_inference",
            ("SONIC C++ deploy",),
            purpose="Current sole command/action stream to C++ deploy",
        ),
        EndpointSpec(
            "cpp_state",
            5557,
            Transport.ZMQ,
            "SONIC C++ deploy",
            ("run_vla_inference", "data exporter"),
            purpose="g1_debug and robot_config state topics",
        ),
        EndpointSpec(
            "navigation_command",
            5558,
            Transport.ZMQ,
            "SonicControlGateway",
            ("navdp_planner",),
            purpose="Manual velocity, semantic navigation goal, and stop commands",
        ),
        EndpointSpec(
            "navigation_status",
            5559,
            Transport.ZMQ,
            "navdp_planner",
            ("SonicControlGateway",),
            purpose="Navigation generation and state feedback",
        ),
        EndpointSpec(
            "planner_relay",
            5563,
            Transport.ZMQ,
            "navdp_planner",
            ("run_vla_inference",),
            purpose="SONIC planner binary messages selected by the launcher",
        ),
        EndpointSpec(
            "lingbot_depth",
            5564,
            Transport.ZMQ,
            "LingBot Depth",
            ("SensorGateway",),
            purpose="Completed chest depth ingress copied into shared memory",
        ),
        EndpointSpec(
            "operator_keyboard_legacy",
            5580,
            Transport.ZMQ,
            "SonicControlGateway legacy mirror",
            ("run_vla_inference", "data exporter"),
            purpose="Legacy unstructured keyboard events",
        ),
        EndpointSpec(
            "xnavdp_http",
            19999,
            Transport.HTTP,
            "X-NavDP policy server",
            ("navdp_planner", "NavDP health monitor"),
            purpose="RGB-D point-goal trajectory inference",
        ),
        EndpointSpec(
            "sensor_gateway_metadata",
            5560,
            Transport.ZMQ,
            "SonicSensorGateway",
            ("future gateway clients",),
            purpose="Shared-memory snapshots, health, and readiness queries",
        ),
        EndpointSpec(
            "control_gateway_intent",
            5561,
            Transport.ZMQ,
            "SonicControlGateway",
            ("operator CLI", "operator OpenCV viewer"),
            purpose="Structured operator intents entering ControlGateway",
        ),
        EndpointSpec(
            "control_gateway_status",
            5562,
            Transport.ZMQ,
            "SonicControlGateway",
            ("future operator console and diagnostics",),
            purpose="Control-ingress acknowledgements and future control state",
        ),
        EndpointSpec(
            "control_gateway_dispatch",
            5565,
            Transport.ZMQ,
            "SonicControlGateway",
            ("future VLA, navigation, and data-exporter clients",),
            purpose="Validated structured control commands for consumers",
        ),
        EndpointSpec(
            "sensor_gateway_visualization_ingress",
            5566,
            Transport.ZMQ,
            "SonicSensorGateway",
            ("navdp_planner", "LingBot Depth"),
            purpose="Best-effort rendered panels for the unified OpenCV viewer",
        ),
        EndpointSpec(
            "vla_timing_ingress",
            5567,
            Transport.ZMQ,
            "SonicSensorGateway",
            ("run_vla_inference",),
            purpose="Best-effort segmented VLA latency telemetry",
        ),
    )
    by_name = {endpoint.name: endpoint for endpoint in endpoints}
    if len(by_name) != len(endpoints):
        raise ValueError("duplicate runtime endpoint name")
    by_port: dict[int, str] = {}
    for endpoint in endpoints:
        previous = by_port.setdefault(endpoint.port, endpoint.name)
        if previous != endpoint.name:
            raise ValueError(
                f"runtime port {endpoint.port} is assigned to both {previous} and {endpoint.name}"
            )
    return MappingProxyType(by_name)


def _build_ros_topics() -> Mapping[str, RosTopicSpec]:
    topics = (
        RosTopicSpec(
            "/livox/lidar",
            "livox_ros_driver2",
            ("FAST-LIO2", "navdp_planner"),
            purpose="MID-360 point packets",
        ),
        RosTopicSpec(
            "/livox/imu",
            "livox_ros_driver2",
            ("FAST-LIO2",),
            purpose="MID-360 IMU samples",
        ),
        RosTopicSpec(
            "/Odometry_loc",
            "FAST-LIO2",
            ("navdp_planner", "NavDP readiness and health checks"),
            purpose="Robot pose in the local SLAM frame",
        ),
        RosTopicSpec(
            "/cloud_registered_1",
            "FAST-LIO2",
            ("navdp_planner", "NavDP readiness and health checks"),
            purpose="Registered point cloud used for the local SLAM display",
        ),
    )
    return MappingProxyType({topic.name: topic for topic in topics})


ENDPOINTS = _build_endpoints()
ROS_TOPICS = _build_ros_topics()


def get_endpoint(name: str) -> EndpointSpec:
    """Return a named endpoint with a useful error for configuration code."""

    try:
        return ENDPOINTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(ENDPOINTS))
        raise KeyError(f"unknown runtime endpoint {name!r}; known endpoints: {known}") from exc
