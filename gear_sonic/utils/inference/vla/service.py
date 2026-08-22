"""Gateway-backed OpenPI VLA inference runner without a ROS 2 dependency.

Communication paths:
  1. Robot state and cameras -> SensorGateway snapshot/cache
  2. Operator commands       -> ControlGateway typed command stream
  3. Actions out             -> ZMQ PUB (motion token, hands, or planner commands)
  4. Planner relay           -> ZMQ SUB ``planner`` topic from the runtime profile,
                                forwarded
                                to the C++ command endpoint

``command`` topic (start/stop/mode) is sent only by this script (``k``/``i``/``o``).
The keyboard planner sidecar sends ``planner`` topic only — no ``command`` overlap.

Uses msgpack-numpy over ZMQ REQ/REP to communicate with an OpenPI policy server.
"""

from dataclasses import dataclass
import json
import logging
import math
import queue
import threading
import time
from typing import Any

import msgpack_numpy as mnp
import numpy as np
import zmq

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.runtime.gateway.control_client import (
    ControlGatewayIntentClient,
    ControlGatewaySubscriber,
)
from gear_sonic.runtime.profile import (
    RuntimeProfileSelection,
    load_component_config,
    load_runtime_profile,
)
from gear_sonic.runtime.telemetry import (
    VLA_TIMING_SEGMENTS,
    build_event,
    configure_file_logging,
    emit_event,
    open_telemetry_publisher,
    publish_metrics,
)
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.inference.vla.inference import (
    calculate_latency_compensated_index,
    prepare_observation_for_eval,
)
from gear_sonic.utils.inference.vla.ingress import VlaSensorGatewayIngress
from gear_sonic.utils.inference.vla.poses import (
    SONIC_STAND_UPPER_BODY_RAD,
    UPPER_BODY_MUJOCO_INDICES,
    VLA_INITIAL_UPPER_BODY_RAD,
)
from gear_sonic.utils.inference.vla.safety import VlaSafetyGate
from gear_sonic.utils.planner_control.executor_service import PlannerSafetySensorMonitor
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)


PLANNER_HEADER_SIZE = 1280
LOGGER = logging.getLogger("sonic.vla")


def _report_event(callback, level: int, code: str, message: str, **fields: object) -> None:
    if callback is None or callback(level, code, message, write_log=False, **fields):
        LOGGER.log(level, "%s | %s", code, message, exc_info=level >= logging.ERROR)


def _wrap_angle(angle: float) -> float:
    return math.remainder(float(angle), 2.0 * math.pi)


def _base_yaw_from_state(state_msg: dict | None) -> float | None:
    """Return measured robot yaw from a g1_debug state message (wxyz quaternion)."""
    if state_msg is None or "base_quat" not in state_msg:
        return None
    quat = np.asarray(state_msg["base_quat"], dtype=np.float64).reshape(-1)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        return None
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return None
    w, x, y, z = quat / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _facing_from_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw), math.sin(yaw), 0.0]


def _unpack_planner_message(message: bytes) -> tuple[int, dict[str, np.ndarray]]:
    """Decode one planner wire message without depending on another script module."""
    topic = b"planner"
    if not message.startswith(topic):
        raise ValueError("planner relay received a message with the wrong topic")
    header_start = len(topic)
    payload_start = header_start + PLANNER_HEADER_SIZE
    if len(message) < payload_start:
        raise ValueError("planner message is shorter than its header")

    header_bytes = message[header_start:payload_start].split(b"\x00", 1)[0]
    header = json.loads(header_bytes.decode("utf-8"))
    endian = "<" if header.get("endian", "le") == "le" else ">"
    dtype_map = {
        "f32": np.dtype(endian + "f4"),
        "f64": np.dtype(endian + "f8"),
        "i32": np.dtype(endian + "i4"),
        "i64": np.dtype(endian + "i8"),
        "u8": np.dtype("u1"),
        "bool": np.dtype("?"),
    }
    fields: dict[str, np.ndarray] = {}
    offset = payload_start
    for field in header.get("fields", []):
        dtype = dtype_map.get(field["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported planner dtype: {field['dtype']}")
        shape = tuple(int(dim) for dim in field["shape"])
        byte_count = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        end = offset + byte_count
        if end > len(message):
            raise ValueError(f"planner field {field['name']} exceeds payload")
        fields[field["name"]] = np.frombuffer(
            message[offset:end], dtype=dtype
        ).reshape(shape).astype(dtype.newbyteorder("="), copy=True)
        offset = end
    return int(header.get("v", 1)), fields


def _rotate_planner_message(message: bytes, yaw_offset: float) -> bytes:
    """Rotate planner movement and facing into the active VLA heading frame."""
    version, fields = _unpack_planner_message(message)
    cosine, sine = math.cos(yaw_offset), math.sin(yaw_offset)
    for name in ("movement", "facing"):
        vector = fields.get(name)
        if vector is None or vector.size != 3:
            raise ValueError(f"planner message has invalid {name} field")
        flat = vector.reshape(3).astype(np.float32, copy=True)
        x_value, y_value = float(flat[0]), float(flat[1])
        flat[0] = cosine * x_value - sine * y_value
        flat[1] = sine * x_value + cosine * y_value
        fields[name] = flat.reshape(vector.shape)
    return pack_pose_message(fields, topic="planner", version=version)


class PlannerHeadingAlignment:
    """Hold a fixed planner-to-robot yaw transform for one planner session."""

    def __init__(self):
        self.target_yaw: float | None = None
        self.yaw_offset: float | None = None
        self.last_facing_yaw = 0.0

    def begin(self, target_yaw: float | None) -> None:
        self.target_yaw = target_yaw
        self.yaw_offset = None
        self.last_facing_yaw = 0.0

    def clear(self) -> None:
        self.target_yaw = None
        self.yaw_offset = None

    def current_facing(self) -> list[float]:
        return _facing_from_yaw(self.last_facing_yaw)

    def align(self, message: bytes) -> bytes | None:
        if self.target_yaw is None:
            return None

        if self.yaw_offset is None:
            _version, fields = _unpack_planner_message(message)
            facing = fields.get("facing")
            if facing is None or facing.size != 3:
                raise ValueError("planner message has no valid facing direction")
            facing = facing.reshape(3)
            incoming_yaw = math.atan2(float(facing[1]), float(facing[0]))
            self.yaw_offset = _wrap_angle(self.target_yaw - incoming_yaw)
            LOGGER.info(
                "[HeadingSync] planner frame aligned: "
                f"planner_zero={self.target_yaw:+.3f}, incoming_yaw={incoming_yaw:+.3f}, "
                f"offset={self.yaw_offset:+.3f} rad"
            )
        aligned = _rotate_planner_message(message, self.yaw_offset)
        _version, aligned_fields = _unpack_planner_message(aligned)
        facing = aligned_fields["facing"].reshape(3)
        self.last_facing_yaw = math.atan2(float(facing[1]), float(facing[0]))
        return aligned


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    profile: str = ""
    overlay: tuple[str, ...] = ()
    action_publish_rate: int = 50
    action_horizon: int = 50
    inference_hz: float = 2.0
    sensor_gateway_poll_hz: float = 50.0
    sensor_gateway_request_timeout_ms: int = 100
    sensor_gateway_max_age_ms: float = 1000.0
    sensor_gateway_max_skew_ms: float = 5.0
    embodiment_tag: str = "unitree_g1_sonic"
    prompt: str = "demo"

def load_inference_config(
    profile: str = "",
    overlays: tuple[str, ...] = (),
) -> InferenceConfig:
    return load_component_config(
        InferenceConfig,
        "vla",
        profile or None,
        overlays=overlays,
    )


def parse_inference_config(args: list[str] | None = None) -> InferenceConfig:
    import tyro

    selection = tyro.cli(RuntimeProfileSelection, args=args)
    return load_inference_config(selection.profile, selection.overlay)


class _MsgpackNumpyPolicyClient:
    """Policy client using SONIC's fixed msgpack-numpy ZMQ wire format."""

    def __init__(self, host: str, port: int, timeout_ms: int = 15000) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.context = zmq.Context()
        self.socket = None
        self.last_timing_ms: dict[str, float] = {}
        self._init_socket()

    def _init_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(
        self,
        endpoint: str,
        data: dict | None = None,
        *,
        requires_input: bool = True,
    ):
        request = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        try:
            started = time.perf_counter()
            packed = mnp.packb(request)
            request_pack_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            self.socket.send(packed)
            response_bytes = self.socket.recv()
            policy_roundtrip_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            response = mnp.unpackb(response_bytes, raw=False)
            response_unpack_ms = (time.perf_counter() - started) * 1000.0
            self.last_timing_ms = {
                "request_pack": request_pack_ms,
                "policy_roundtrip": policy_roundtrip_ms,
                "response_unpack": response_unpack_ms,
            }
        except zmq.Again:
            self._init_socket()
            raise
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self, *, timeout_ms: int | None = None) -> bool:
        previous_timeout_ms = self.timeout_ms
        if timeout_ms is not None:
            self.timeout_ms = int(timeout_ms)
            self._init_socket()
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except (RuntimeError, zmq.ZMQError):
            return False
        finally:
            if timeout_ms is not None:
                self.timeout_ms = previous_timeout_ms
                self._init_socket()

    def get_action(self, observation: dict, options: dict | None = None):
        response = self.call_endpoint(
            "get_action",
            {"observation": observation, "options": options},
        )
        return tuple(response)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()


JPEG_VIDEO_MARKER = "__opencv_jpeg_rgb__"


def wrap_camera_jpeg_for_video(
    encoded: bytes | bytearray | memoryview,
    image_shape: tuple[int, int, int],
) -> dict[str, Any]:
    return {
        JPEG_VIDEO_MARKER: True,
        "shape": (1, 1, *image_shape),
        "dtype": "uint8",
        "data": bytes(encoded),
    }


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            if left_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"left_hand_joints must have shape [7], got {left_hand_joints.shape}"
                )
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            if right_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"right_hand_joints must have shape [7], got {right_hand_joints.shape}"
                )
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


def prepare_observation_from_sensors(
    sensor_gateway: VlaSensorGatewayIngress,
    robot_model,
    language_prompt: str,
):
    """Read sensors and prepare observation for the VLA policy.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    observation_started = time.perf_counter()
    timing_ms: dict[str, float] = {}

    started = time.perf_counter()
    camera_msg = sensor_gateway.read_camera()
    timing_ms["camera_read"] = (time.perf_counter() - started) * 1000.0
    if camera_msg is None:
        return None

    started = time.perf_counter()
    state_msg = sensor_gateway.read_state()
    timing_ms["state_read"] = (time.perf_counter() - started) * 1000.0
    if state_msg is None:
        return None

    required_image_keys = ("ego_view", "chest_view", "left_wrist", "right_wrist")
    missing_image_keys = [key for key in required_image_keys if key not in camera_msg["images"]]
    if missing_image_keys:
        raise ValueError(f"Missing required camera images: {missing_image_keys}")

    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )

    started = time.perf_counter()
    video = {
        name: wrap_camera_jpeg_for_video(
            camera_msg["images"][name], camera_msg["image_shapes"][name]
        )
        for name in required_image_keys
    }
    timing_ms["jpeg_prepare"] = (time.perf_counter() - started) * 1000.0

    observation = {
        "video": video,
        "state": {},
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }

    observation = prepare_observation_for_eval(robot_model, observation)

    # Projected gravity for Sonic latent embodiment
    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]

    timestamp = np.asarray(camera_msg["timestamps"]["ego_view"]).reshape(-1)
    if timestamp.size:
        raw_timestamp = float(timestamp[-1])
        if math.isfinite(raw_timestamp):
            # Camera source timestamps use Unix seconds. Avoid
            # reporting nonsense when a test or alternate source uses another clock.
            age_ms = (time.time() - raw_timestamp) * 1000.0
            if 0.0 <= age_ms <= 60_000.0:
                timing_ms["frame_age"] = age_ms
    timing_ms["observation_build"] = (
        time.perf_counter() - observation_started
    ) * 1000.0
    return observation, timing_ms


def run_policy_inference_and_process(
    policy,
    observation,
    event_callback=None,
    failure_callback=None,
):
    """Run OpenPI policy inference and process the returned action chunk.

    Returns:
        processed_action dict or None on error.
    """
    try:
        policy_started = time.perf_counter()
        action, _info = policy.get_action(observation)
        policy_total_ms = (time.perf_counter() - policy_started) * 1000.0
        timing_ms = dict(getattr(policy, "last_timing_ms", {}))
        timing_ms.setdefault("policy_roundtrip", policy_total_ms)

        postprocess_started = time.perf_counter()
        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = "motion_token" if "motion_token" in action else "action.motion_token"
        if np.abs(action[motion_key]).max() > 1.25:
            maximum = float(np.abs(action[motion_key]).max())
            _report_event(
                event_callback, logging.WARNING, "ACTION_REJECTED",
                f"{motion_key} max {maximum:.4f} exceeds 1.25",
                repeat_s=30.0,
            )
            return None

        processed_action = {
            key.replace("action.", ""): value for key, value in action.items()
        }
        timing_ms["action_postprocess"] = (
            time.perf_counter() - postprocess_started
        ) * 1000.0
        return processed_action, timing_ms
    except Exception as exc:
        _report_event(
            event_callback, logging.ERROR, "INFERENCE_FAILED", str(exc), repeat_s=30.0
        )
        if failure_callback is not None:
            failure_callback(str(exc))
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
    event_callback=None,
    failure_callback=None,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                request_generation = inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                worker_started = time.perf_counter()
                prepared = prepare_obs_fn()
                if prepared is None:
                    continue
                observation, timing_ms = prepared
                inference_start_time = time.monotonic()
                result = inference_fn(observation)
                if result is None:
                    continue
                processed_action, action_timing = result
                timing_ms.update(action_timing)
                timing_ms["worker_total"] = (
                    time.perf_counter() - worker_started
                ) * 1000.0

                if processed_action is not None:
                    item = (request_generation, processed_action, inference_start_time, timing_ms)
                    try:
                        result_queue.put_nowait(item)
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                        except queue.Empty:
                            pass
                        result_queue.put_nowait(item)
            finally:
                busy_event.clear()
        except Exception as exc:
            _report_event(
                event_callback, logging.ERROR, "WORKER_ERROR", str(exc), repeat_s=30.0
            )
            if failure_callback is not None:
                failure_callback(str(exc))


def _pose_policy_is_active(cpp_loop_running: bool, cpp_mode: str, pause_loop: bool) -> bool:
    """Return whether a fresh VLA inference/action is allowed to run."""
    return cpp_loop_running and cpp_mode == "POSE" and not pause_loop


def _should_schedule_vla_inference(
    *,
    cpp_mode: str,
    worker_is_busy: bool,
    request_queue_is_empty: bool,
    time_since_request: float,
    inference_interval: float,
) -> bool:
    """Capture POSE observations even when action publication is paused."""

    return (
        cpp_mode == "POSE"
        and not worker_is_busy
        and request_queue_is_empty
        and time_since_request >= inference_interval
    )


def _drain_queue(target: queue.Queue) -> int:
    """Remove queued work/results without waiting and return the item count."""
    drained = 0
    while True:
        try:
            target.get_nowait()
            drained += 1
        except queue.Empty:
            return drained


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    """Compute closed hand joint positions using G1GripperInverseKinematicsSolver."""
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def main(config: InferenceConfig):
    configure_file_logging("vla")
    pause_loop = True
    profile = load_runtime_profile(config.profile or None, overlays=config.overlay)
    policy_endpoint = profile.endpoint("policy_server")
    event_socket = open_telemetry_publisher(
        profile.endpoint_uri("runtime_event_ingress")
    )
    pending_events = queue.SimpleQueue()
    last_event_at: dict[str, float] = {}

    def record_event(
        level: int,
        code: str,
        message: str,
        *,
        repeat_s: float = 0.0,
        write_log: bool = True,
        **fields: object,
    ) -> bool:
        now = time.monotonic()
        if repeat_s and now - last_event_at.get(code, -math.inf) < repeat_s:
            return False
        last_event_at[code] = now
        payload = build_event("vla", level, code, message, **fields)
        if write_log:
            emit_event(payload, logger=LOGGER)
        pending_events.put(payload)
        return True

    def flush_events() -> None:
        while True:
            try:
                emit_event(pending_events.get_nowait(), socket=event_socket)
            except queue.Empty:
                return

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    n1_policy = _MsgpackNumpyPolicyClient(
        host=policy_endpoint.host,
        port=policy_endpoint.port,
    )
    timing_socket = open_telemetry_publisher(
        profile.endpoint_uri("runtime_metrics_ingress")
    )

    def activate_vla_metrics() -> None:
        publish_metrics(
            timing_socket,
            "vla",
            {},
            allowed_names=VLA_TIMING_SEGMENTS,
            activate=True,
        )

    if not n1_policy.ping(timeout_ms=1000):
        record_event(
            logging.WARNING,
            "POLICY_UNREACHABLE",
            "PolicyServer is not reachable; inference will fail until it recovers",
            host=policy_endpoint.host,
            port=policy_endpoint.port,
        )

    gateway_ingress = VlaSensorGatewayIngress(
        profile.endpoint_uri("sensor_gateway_metadata"),
        poll_hz=config.sensor_gateway_poll_hz,
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        max_skew_ms=config.sensor_gateway_max_skew_ms,
    )
    gateway_ingress.start()

    planner_safety_config = profile.component("planner_executor")
    vla_safety_gate = VlaSafetyGate(
        radar_timeout_s=float(planner_safety_config["radar_timeout_s"]),
        robot_state_timeout_s=float(
            planner_safety_config["sensor_gateway_max_age_ms"]
        ) / 1000.0,
    )
    vla_safety_monitor = PlannerSafetySensorMonitor(
        profile.endpoint_uri("sensor_gateway_metadata"),
        poll_hz=float(planner_safety_config["sensor_gateway_poll_hz"]),
        request_timeout_ms=int(
            planner_safety_config["sensor_gateway_request_timeout_ms"]
        ),
        max_age_ms=float(planner_safety_config["sensor_gateway_max_age_ms"]),
        include_robot_state=True,
    )
    vla_safety_monitor.start()

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(profile.endpoint_uri("cpp_command"))
    task_status_intent = ControlGatewayIntentClient(
        profile.endpoint_uri("control_gateway_intent"),
        source="vla_service",
        context=zmq_context,
    )
    time.sleep(0.1)

    control_listener = ControlGatewaySubscriber(
        profile.endpoint_uri("control_gateway_dispatch"),
        accepted_names={
            "select_pose_mode",
            "toggle_control_loop",
            "select_planner_mode",
            "toggle_policy_pause",
            "toggle_left_hand_initial_pose",
            "toggle_right_hand_initial_pose",
            "set_prompt",
            "start_vla_task",
            "hold_vla_task",
            "resume_vla_task",
            "stop_vla_task",
            "cancel_navigation",
        },
    )

    planner_relay_sub = zmq_context.socket(zmq.SUB)
    planner_relay_sub.setsockopt_string(zmq.SUBSCRIBE, "planner")
    planner_relay_sub.setsockopt(zmq.RCVTIMEO, 0)
    planner_relay_sub.connect(profile.endpoint_uri("planner_relay"))
    planner_heading_alignment = PlannerHeadingAlignment()

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    def _current_upper_body_planner_order(body_q: np.ndarray) -> np.ndarray:
        return np.array([body_q[i] for i in UPPER_BODY_MUJOCO_INDICES], dtype=np.float32)

    def publish_initial_pose():
        # Initial pose publishing in PLANNER mode
        if cpp_mode != "PLANNER" or not cpp_loop_running:
            record_event(
                logging.WARNING,
                "INITIAL_POSE_BLOCKED",
                "Initial pose requires an active PLANNER control loop",
            )
            return False

        duration = 3.0
        hz = 50
        period = 1.0 / hz

        zero_vel = np.zeros(17, dtype=np.float32)
        target_ub = np.array(VLA_INITIAL_UPPER_BODY_RAD, dtype=np.float32)

        state_msg = gateway_ingress.read_state(clear=False)
        start_ub = None
        if state_msg is not None and "body_q" in state_msg:
            body_q = np.asarray(state_msg["body_q"], dtype=np.float32)
            assert body_q.shape[0] == 29, "body_q must have shape (29,)"
            start_ub = _current_upper_body_planner_order(body_q)

        if start_ub is None:
            record_event(
                logging.ERROR,
                "INITIAL_POSE_FAILED",
                "Cannot read body state for the initial-pose ramp",
            )
            return False

        hold_facing = planner_heading_alignment.current_facing()
        hold_yaw = math.atan2(hold_facing[1], hold_facing[0])
        LOGGER.info("Holding planner facing %.3f rad during POSE entry", hold_yaw)

        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )

        t0 = time.monotonic()

        while True:
            loop_t0 = time.monotonic()
            elapsed = loop_t0 - t0
            if elapsed >= duration:
                break

            t = min(max(elapsed / duration, 0.0), 1.0)
            alpha = t * t * (3.0 - 2.0 * t)
            q_cmd = (1.0 - alpha) * start_ub + alpha * target_ub

            zmq_socket.send(
                build_planner_message(
                    0,
                    [0.0, 0.0, 0.0],
                    hold_facing,
                    speed=-1.0,
                    height=-1.0,
                    upper_body_position=q_cmd.tolist(),
                    upper_body_velocity=zero_vel.tolist(),
                    left_hand_position=left_hand.tolist(),
                    right_hand_position=right_hand.tolist(),
                )
            )
            remaining = period - (time.monotonic() - loop_t0)
            if remaining > 0:
                time.sleep(remaining)

        for _ in range(5):
            zmq_socket.send(
                build_planner_message(
                    0,
                    [0.0, 0.0, 0.0],
                    hold_facing,
                    speed=-1.0,
                    height=-1.0,
                    upper_body_position=target_ub.tolist(),
                    upper_body_velocity=zero_vel.tolist(),
                    left_hand_position=left_hand.tolist(),
                    right_hand_position=right_hand.tolist(),
                )
            )
            time.sleep(0.02)
        return True

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            planner_entry_yaw = None
            if start and planner and cpp_mode != "PLANNER":
                discarded = _discard_pending_planner_messages(planner_relay_sub)
                measured_yaw = _base_yaw_from_state(
                    gateway_ingress.read_state(clear=False)
                )
                # C++ reinitializes the planner heading from the measured base
                # orientation on every mode switch. In that new planner frame,
                # the current physical heading is therefore exactly yaw zero.
                planner_heading_alignment.begin(0.0)
                planner_entry_yaw = 0.0
                LOGGER.info(
                    "Preparing PLANNER entry: measured_yaw=%s discarded_stale=%s",
                    measured_yaw,
                    discarded,
                )
            elif start and not planner and cpp_mode != "POSE":
                planner_heading_alignment.clear()

            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            # This is a low-frequency state transition sent over PUB/SUB.
            # A single packet can be lost during subscriber reconnects or a
            # busy mode transition, while the Python side would otherwise
            # optimistically update ``cpp_mode``. Repeat the idempotent
            # command over a short window before publishing planner data.
            for _ in range(5):
                zmq_socket.send(cmd_msg)
                time.sleep(0.02)
            if start and planner and planner_entry_yaw is not None:
                # Do not let the C++ planner's default facing direction act in
                # the gap before the first fresh sidecar command arrives.
                zmq_socket.send(
                    build_planner_message(
                        0,
                        [0.0, 0.0, 0.0],
                        _facing_from_yaw(planner_entry_yaw),
                        speed=-1.0,
                        height=-1.0,
                    )
                )
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            LOGGER.info(
                "Sent C++ control command: start=%s mode=%s",
                start,
                "PLANNER" if planner else "POSE",
            )
            return True
        except Exception as exc:
            reported = record_event(
                logging.ERROR,
                "CPP_COMMAND_FAILED",
                str(exc),
                repeat_s=30.0,
                write_log=False,
                start=start,
                planner=planner,
            )
            if reported:
                LOGGER.exception("CPP_COMMAND_FAILED | %s", exc)
            return False

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_request_time = 0.0
    inference_interval = 1.0 / config.inference_hz

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_failures: queue.SimpleQueue[str] = queue.SimpleQueue()
    inference_failed_event = threading.Event()

    def report_inference_failure(reason: str) -> None:
        if inference_failed_event.is_set():
            return
        inference_failed_event.set()
        inference_failures.put(str(reason))
    inference_generation = 0
    active_generation = -1

    def invalidate_inference(reason: str):
        """Invalidate cached, queued, and currently-running inference work."""
        nonlocal cached_action_chunk, action_chunk_index, last_inference_request_time
        nonlocal inference_generation
        inference_generation += 1
        cached_action_chunk = None
        action_chunk_index = 0
        last_inference_request_time = 0.0
        queued = _drain_queue(inference_queue)
        results = _drain_queue(result_queue)
        LOGGER.info(
            "Invalidated inference generation %s: %s (queued=%s results=%s)",
            inference_generation,
            reason,
            queued,
            results,
        )

    zmq_frame_counter = 0
    task_generation = -1
    task_skill_id = 0
    task_window_id = 0
    task_active = False

    def publish_task_status(state: str, reason: str, *, window_id: int = 0) -> None:
        task_status_intent.send("vla_task_status", {
            "generation": task_generation,
            "skill_id": task_skill_id,
            "segment_id": window_id,
            "state": state,
            "reason": reason,
        })

    def fail_active_task(reason: str, message: str) -> None:
        nonlocal task_active, pause_loop
        task_active = False
        pause_loop = True
        invalidate_inference(reason)
        if cpp_mode == "POSE":
            send_cpp_control_command(start=True, planner=True)
        publish_task_status("failed", reason, window_id=task_window_id)
        record_event(
            logging.ERROR,
            "VLA_UNEXPECTED_TERMINATION",
            message,
            reason=reason,
            generation=task_generation,
            skill_id=task_skill_id,
            window_id=task_window_id,
        )

    def check_control_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal zmq_frame_counter
        nonlocal task_generation, task_skill_id, task_window_id, task_active

        command = control_listener.read_command()
        if command is None:
            return

        command_name = command.name
        if command_name in {
            "start_vla_task",
            "hold_vla_task",
            "resume_vla_task",
            "stop_vla_task",
            "cancel_navigation",
        }:
            generation = int(command.parameters.get("generation", -1))
            skill_id = int(command.parameters.get("skill_id", 0))
            window_id = int(command.parameters.get("window_id", 0))
            identity = (generation, skill_id)
            active_identity = (task_generation, task_skill_id)
            if command_name == "cancel_navigation":
                if generation < task_generation:
                    return
                task_generation = generation
                task_skill_id = 0
                task_window_id = 0
                task_active = False
                pause_loop = True
                invalidate_inference("navigation cancelled")
                if cpp_mode == "POSE":
                    send_cpp_control_command(start=True, planner=True)
                return
            if command_name == "start_vla_task":
                if identity < active_identity:
                    return
                if identity == active_identity:
                    return  # idempotent duplicate
                prompt = command.parameters.get("handoff_context")
                original_task = command.parameters.get("task")
                if not isinstance(prompt, str) or not prompt.strip():
                    record_event(logging.ERROR, "VLA_TASK_REJECTED", "Missing VLA handoff context")
                    return
                if not isinstance(original_task, str) or not original_task.strip():
                    record_event(logging.ERROR, "VLA_TASK_REJECTED", "Missing original task")
                    return
                safety_reason = vla_safety_gate.reason(
                    now=time.monotonic(),
                    safety=vla_safety_monitor.snapshot(),
                    robot_state_timestamp_s=(
                        vla_safety_monitor.orientation_snapshot().received_at_s
                    ),
                )
                if safety_reason != "clear":
                    record_event(
                        logging.ERROR,
                        "VLA_SAFETY_BLOCKED",
                        "VLA task start rejected by the shared safety gate",
                        reason=safety_reason,
                        generation=generation,
                        skill_id=skill_id,
                    )
                    task_status_intent.send("vla_task_status", {
                        "generation": generation,
                        "skill_id": skill_id,
                        "segment_id": window_id,
                        "state": "failed",
                        "reason": safety_reason,
                    })
                    return
                task_generation, task_skill_id = identity
                task_window_id = 0
                task_active = True
                while not inference_failures.empty():
                    try:
                        inference_failures.get_nowait()
                    except queue.Empty:
                        break
                inference_failed_event.clear()
                if not n1_policy.ping(timeout_ms=1000):
                    fail_active_task(
                        "vla_policy_unreachable",
                        "PolicyServer was unreachable when MANIPULATE started",
                    )
                    return
                language_prompt_ref[0] = prompt
                zmq_frame_counter = 0
                invalidate_inference("VLA task started")
                if cpp_mode == "PLANNER":
                    if not publish_initial_pose():
                        fail_active_task(
                            "vla_initial_pose_failed",
                            "VLA could not publish its initial pose",
                        )
                        return
                    if not send_cpp_control_command(start=True, planner=False):
                        fail_active_task(
                            "vla_pose_mode_failed",
                            "VLA could not enter POSE mode",
                        )
                        return
                elif cpp_mode == "OFF":
                    if not send_cpp_control_command(start=True, planner=False):
                        fail_active_task(
                            "vla_pose_mode_failed",
                            "VLA could not start POSE mode",
                        )
                        return
                pause_loop = False
                activate_vla_metrics()
                publish_task_status("active", "started", window_id=0)
                return
            if identity != active_identity or not task_active or window_id < task_window_id:
                return
            task_window_id = window_id
            if command_name == "hold_vla_task":
                if pause_loop:
                    return  # idempotent duplicate
                pause_loop = True
                invalidate_inference(f"VLA window {window_id} held")
                return
            if command_name == "resume_vla_task":
                prompt = command.parameters.get("handoff_context")
                if isinstance(prompt, str) and prompt.strip():
                    language_prompt_ref[0] = prompt
                if not pause_loop:
                    return  # idempotent duplicate
                pause_loop = False
                invalidate_inference(f"VLA window {window_id} resumed")
                activate_vla_metrics()
                return
            # stop_vla_task: keep the whole-body controller in safe planner idle.
            task_active = False
            pause_loop = True
            invalidate_inference(f"VLA task stopped at window {window_id}")
            if cpp_mode == "POSE":
                send_cpp_control_command(start=True, planner=True)
            return

        if command_name == "set_prompt":
            new_prompt = command.parameters.get("prompt")
            if isinstance(new_prompt, str) and new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                invalidate_inference("prompt changed")
                LOGGER.info("Inference prompt changed: %r -> %r", old_prompt, new_prompt)
            else:
                record_event(
                    logging.WARNING,
                    "INVALID_PROMPT",
                    "Empty inference prompt ignored",
                )
            return

        if command_name == "select_pose_mode":
            zmq_frame_counter = 0
            invalidate_inference("entering POSE mode")
            if cpp_mode == "PLANNER":
                if not publish_initial_pose():
                    return
                send_cpp_control_command(start=True, planner=False)
            if cpp_mode == "POSE":
                activate_vla_metrics()
        elif command_name == "select_planner_mode":
            zmq_frame_counter = 0
            invalidate_inference("entering PLANNER mode")
            if cpp_mode == "POSE":
                send_cpp_control_command(start=True, planner=True)
            if cpp_mode == "PLANNER":
                record_event(
                    logging.WARNING,
                    "PLANNER_BLOCKED",
                    "Planner mode active; remote VLA inference is suspended",
                )
        elif command_name == "toggle_policy_pause":
            if cpp_mode == "PLANNER":
                record_event(
                    logging.WARNING,
                    "PLANNER_BLOCKED",
                    "Switch to POSE mode before resuming remote VLA inference",
                )
            else:
                pause_loop = not pause_loop
                invalidate_inference(
                    "POSE policy resumed" if not pause_loop else "POSE policy paused"
                )
                if pause_loop:
                    record_event(
                        logging.INFO,
                        "INFERENCE_PAUSED",
                        "VLA action publication paused; remote inference remains warm",
                        remote_requests_continue=True,
                    )
                elif cpp_mode == "POSE":
                    activate_vla_metrics()

        elif command_name == "toggle_control_loop":
            invalidate_inference("C++ control toggled")
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                if send_cpp_control_command(start=False, planner=current_planner):
                    record_event(
                        logging.INFO,
                        "INFERENCE_PAUSED",
                        "Remote VLA inference paused because the control loop stopped",
                        remote_requests_continue=False,
                    )
            else:
                if send_cpp_control_command(start=True, planner=True):
                    record_event(
                        logging.WARNING,
                        "PLANNER_BLOCKED",
                        "Planner mode active; remote VLA inference is suspended",
                    )
        elif command_name == "toggle_left_hand_initial_pose":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
        elif command_name == "toggle_right_hand_initial_pose":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]

    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                sensor_gateway=gateway_ingress,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
            ),
            lambda obs: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                event_callback=record_event,
                failure_callback=report_inference_failure,
            ),
        ),
        kwargs={
            "event_callback": record_event,
            "failure_callback": report_inference_failure,
        },
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_control_input()
            flush_events()

            if task_active:
                try:
                    inference_failure = inference_failures.get_nowait()
                except queue.Empty:
                    inference_failure = None
                if inference_failure is not None:
                    fail_active_task(
                        f"vla_inference_failed:{inference_failure}",
                        "VLA inference failed; the task will not be retried",
                    )
                    _sleep_remaining(t_start, loop_period)
                    continue

            if task_active and cpp_mode == "POSE" and not pause_loop:
                safety_reason = vla_safety_gate.reason(
                    now=t_start,
                    safety=vla_safety_monitor.snapshot(),
                    robot_state_timestamp_s=(
                        vla_safety_monitor.orientation_snapshot().received_at_s
                    ),
                )
                if safety_reason != "clear":
                    task_active = False
                    pause_loop = True
                    invalidate_inference(f"VLA safety blocked: {safety_reason}")
                    send_cpp_control_command(start=True, planner=True)
                    task_status_intent.send("vla_task_status", {
                        "generation": task_generation,
                        "skill_id": task_skill_id,
                        "segment_id": task_window_id,
                        "state": "failed",
                        "reason": safety_reason,
                    })
                    record_event(
                        logging.ERROR,
                        "VLA_SAFETY_BLOCKED",
                        "VLA POSE action stopped by the shared safety gate",
                        reason=safety_reason,
                        generation=task_generation,
                        skill_id=task_skill_id,
                    )

            # Consume a result before deciding whether another request is due.
            try:
                (
                    result_generation,
                    processed_action,
                    inference_start_time,
                    timing_ms,
                ) = result_queue.get_nowait()
                inference_delay = time.monotonic() - inference_start_time
                timing_ms["action_ready"] = inference_delay * 1000.0
                if (
                    result_generation == inference_generation
                    and _pose_policy_is_active(cpp_loop_running, cpp_mode, pause_loop)
                ):
                    publish_metrics(
                        timing_socket,
                        "vla",
                        timing_ms,
                        allowed_names=VLA_TIMING_SEGMENTS,
                    )
                    action_chunk_index = calculate_latency_compensated_index(
                        inference_delay, config.action_publish_rate, config.action_horizon
                    )
                    cached_action_chunk = processed_action
                    if active_generation != inference_generation:
                        record_event(
                            logging.INFO,
                            "INFERENCE_ACTIVE",
                            "Remote VLA inference returned its first active result",
                            generation=inference_generation,
                            prompt=language_prompt_ref[0],
                            action_ready_ms=timing_ms["action_ready"],
                        )
                        active_generation = inference_generation
            except queue.Empty:
                pass

            worker_is_busy = inference_busy_event.is_set()
            now = time.monotonic()
            should_start = (
                not inference_failed_event.is_set()
                and _should_schedule_vla_inference(
                    cpp_mode=cpp_mode,
                    worker_is_busy=worker_is_busy,
                    request_queue_is_empty=inference_queue.empty(),
                    time_since_request=(now - last_inference_request_time),
                    inference_interval=inference_interval,
                )
            )

            if should_start:
                try:
                    inference_queue.put_nowait(inference_generation)
                    last_inference_request_time = now
                except queue.Full:
                    pass

            if cpp_loop_running and cpp_mode == "PLANNER":
                _relay_planner_messages(
                    planner_relay_sub,
                    zmq_socket,
                    planner_heading_alignment,
                    event_callback=record_event,
                )
                _sleep_remaining(t_start, loop_period)
                continue

            if pause_loop:
                time.sleep(0.2)
                continue

            if cached_action_chunk is None:
                _sleep_remaining(t_start, loop_period)
                continue

            processed_action = cached_action_chunk
            if processed_action:
                motion_token = np.asarray(
                    get_action_field(processed_action, "motion_token"), dtype=np.float32
                )
                left_hand_joints = np.asarray(
                    get_action_field(processed_action, "left_hand_joints"), dtype=np.float32
                )
                right_hand_joints = np.asarray(
                    get_action_field(processed_action, "right_hand_joints"), dtype=np.float32
                )

                # Action arrays arrive as (B, T, D); remove the batch dimension.
                if motion_token.ndim == 3:
                    motion_token = motion_token[0]
                if left_hand_joints.ndim == 3:
                    left_hand_joints = left_hand_joints[0]
                if right_hand_joints.ndim == 3:
                    right_hand_joints = right_hand_joints[0]

                horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                current_idx = min(action_chunk_index, horizon - 1)
                if motion_token.ndim == 2:
                    motion_token = motion_token[current_idx]
                if left_hand_joints.ndim == 2:
                    left_hand_joints = left_hand_joints[current_idx]
                if right_hand_joints.ndim == 2:
                    right_hand_joints = right_hand_joints[current_idx]

                frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                zmq_frame_counter += 1
                zmq_socket.send(
                    pack_latent_action_message(
                        motion_token,
                        frame_index,
                        left_hand_joints=left_hand_joints,
                        right_hand_joints=right_hand_joints,
                    )
                )

            action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        LOGGER.info("VLA inference loop terminated by user")
    except Exception as exc:
        reported = record_event(
            logging.ERROR, "VLA_FATAL", str(exc), write_log=False
        )
        if reported:
            LOGGER.exception("VLA_FATAL | %s", exc)
        flush_events()
        raise

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        planner_relay_sub.close()
        zmq_socket.close()
        gateway_ingress.close()
        vla_safety_monitor.close()
        control_listener.close()
        task_status_intent.close()
        zmq_context.term()
        n1_policy.close()
        flush_events()
        event_socket.close(linger=0)
        timing_socket.close(linger=0)
        LOGGER.info("Shutdown complete")


def _receive_latest_planner_message(planner_relay_sub: zmq.Socket) -> tuple[bytes | None, int]:
    """Drain the SUB queue and return only its newest planner message."""
    latest = None
    received = 0
    while planner_relay_sub.poll(0):
        latest = planner_relay_sub.recv(zmq.NOBLOCK)
        received += 1
    return latest, received


def _discard_pending_planner_messages(planner_relay_sub: zmq.Socket) -> int:
    """Discard planner commands accumulated while VLA POSE mode was active."""
    _latest, received = _receive_latest_planner_message(planner_relay_sub)
    return received


def _relay_planner_messages(
    planner_relay_sub: zmq.Socket,
    action_pub: zmq.Socket,
    heading_alignment: PlannerHeadingAlignment,
    event_callback=None,
) -> int:
    """Forward only the newest planner command in the current VLA heading frame."""
    message, received = _receive_latest_planner_message(planner_relay_sub)
    if message is None:
        return 0

    aligned = heading_alignment.align(message)
    if aligned is None:
        if event_callback is None:
            LOGGER.warning("PLANNER_ALIGNMENT_BLOCKED | waiting for mode-entry alignment")
        else:
            event_callback(
                logging.WARNING,
                "PLANNER_ALIGNMENT_BLOCKED",
                "Planner relay is waiting for mode-entry alignment",
                repeat_s=30.0,
            )
        return 0
    action_pub.send(aligned)
    return 1

def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    main(parse_inference_config())
