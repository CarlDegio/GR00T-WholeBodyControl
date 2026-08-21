"""Stable SONIC runtime contracts, configuration, and Gateway APIs.

``profile`` owns YAML configuration and endpoint topology, ``protocol`` owns
transport-neutral messages and wire codecs, and ``gateway`` owns the reusable
data plane.  Only ``gateway.services`` may compose those primitives with
application domains under :mod:`gear_sonic.utils`.
"""

from gear_sonic.runtime.gateway.sensor_client import (
    MaterializedSnapshot,
    SensorGatewayClient,
    SensorGatewayClientError,
    SensorGatewayTimeoutError,
    SnapshotUnavailableError,
)
from gear_sonic.runtime.profile import (
    ENDPOINT_SCHEMES,
    EndpointAddress,
    RuntimeProfile,
    default_runtime_profile_path,
    load_runtime_profile,
)
from gear_sonic.runtime.protocol import (
    MessageMetadata,
    OperatorCommand,
    SharedMemoryFrame,
)
from gear_sonic.runtime.gateway.control_client import (
    ControlGatewayIntentClient,
    ControlGatewaySubscriber,
)
from gear_sonic.runtime.gateway.control import (
    ControlGatewayCore,
    ControlGatewayRouter,
    ControlIngressEvent,
    NavigationControlAction,
    NavigationControlState,
    RoutedControlCommand,
)
from gear_sonic.runtime.gateway.diagnostics import (
    EndpointHealth,
    EndpointHealthMonitor,
    EndpointState,
)
from gear_sonic.runtime.gateway.shared_memory import (
    FrameOverwrittenError,
    SharedMemoryRing,
    read_shared_memory_frame,
)
from gear_sonic.runtime.gateway.snapshot import (
    SensorSnapshot,
    SensorSnapshotStore,
    SnapshotRequest,
    TimestampBasis,
)

__all__ = [
    "ControlGatewayCore",
    "ControlGatewayIntentClient",
    "ControlGatewayRouter",
    "ControlGatewaySubscriber",
    "ControlIngressEvent",
    "ENDPOINT_SCHEMES",
    "EndpointAddress",
    "EndpointHealth",
    "EndpointHealthMonitor",
    "EndpointState",
    "FrameOverwrittenError",
    "MessageMetadata",
    "MaterializedSnapshot",
    "OperatorCommand",
    "NavigationControlAction",
    "NavigationControlState",
    "RoutedControlCommand",
    "RuntimeProfile",
    "SensorSnapshot",
    "SensorSnapshotStore",
    "SensorGatewayClient",
    "SensorGatewayClientError",
    "SensorGatewayTimeoutError",
    "SharedMemoryFrame",
    "SharedMemoryRing",
    "SnapshotRequest",
    "SnapshotUnavailableError",
    "TimestampBasis",
    "default_runtime_profile_path",
    "load_runtime_profile",
    "read_shared_memory_frame",
]
