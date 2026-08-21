"""Allowed runtime endpoint names and their URI schemes."""

ENDPOINT_SCHEMES = {
    "policy_server": "tcp",
    "camera_server": "tcp",
    "cpp_command": "tcp",
    "cpp_state": "tcp",
    "navigation_command": "tcp",
    "navigation_status": "tcp",
    "planner_relay": "tcp",
    "depth_anything": "tcp",
    "xnavdp_http": "http",
    "sensor_gateway_metadata": "tcp",
    "control_gateway_intent": "tcp",
    "control_gateway_dispatch": "tcp",
    "sensor_gateway_visualization_ingress": "tcp",
    "runtime_metrics_ingress": "tcp",
    "navdp_velocity": "tcp",
    "orientation_telemetry": "tcp",
    "navigation_runtime_status": "tcp",
    "runtime_event_ingress": "tcp",
}
