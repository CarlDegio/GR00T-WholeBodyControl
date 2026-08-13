"""Gateway-backed OpenPI VLA inference runner without a ROS 2 dependency.

Communication paths:
  1. Robot state and cameras -> SensorGateway snapshot/cache
  2. Operator commands       -> ControlGateway typed command stream
  3. Actions out             -> ZMQ PUB (motion token, hands, or planner commands)
  4. Planner relay           -> ZMQ SUB ``planner`` topic (port 5558), forwarded
                                to the C++ command endpoint

``command`` topic (start/stop/mode) is sent only by this script (``k``/``i``/``o``).
The keyboard planner sidecar sends ``planner`` topic only — no ``command`` overlap.

Uses msgpack-numpy over ZMQ REQ/REP to communicate with an OpenPI policy server.

Operator commands (received from ControlGateway):
  k  -> start / stop the C++ control loop (start defaults to PLANNER mode)
  o  -> switch to PLANNER mode (enables relay of WASD sidecar on :5558)
  i  -> switch to POSE mode (VLA latent actions; relay disabled)
  p  -> pause / resume the policy loop (POSE mode only)
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  c  -> start recording (handled by data exporter if running)
  e  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from dataclasses import dataclass
import json
import math
import queue
import threading
import time
from typing import Any

import msgpack_numpy as mnp
import numpy as np
import tyro
import zmq

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.runtime.vla_sensor_gateway import VlaSensorGatewayIngress
from gear_sonic.runtime.vla_timing import VlaTimingPublisher
from gear_sonic.runtime.control_client import ControlGatewaySubscriber
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.inference.initial_poses import SONIC_STAND_UPPER_BODY_RAD, VLA_INITIAL_UPPER_BODY_RAD, UPPER_BODY_MUJOCO_INDICES
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    concat_action,
    prepare_observation_for_eval,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
    build_planner_message
)


PLANNER_HEADER_SIZE = 1280


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
            print_green(
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

    # OpenPI policy server
    host: str = "localhost"
    """The host address of the OpenPI policy server."""

    port: int = 29999
    """The port of the OpenPI policy server."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 50
    """Action horizon of the VLA policy (number of future actions per inference)."""

    rate: float = 1 / 0.5
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    sensor_gateway_endpoint: str = "tcp://127.0.0.1:5560"
    """Local read-only SensorGateway Snapshot endpoint."""

    sensor_gateway_poll_hz: float = 50.0
    """Background rate for materializing VLA camera and C++ state caches."""

    sensor_gateway_request_timeout_ms: int = 100
    """Deadline for one local SensorGateway metadata request."""

    sensor_gateway_max_age_ms: float = 1000.0
    """Maximum accepted Gateway frame and cache age."""

    sensor_gateway_max_skew_ms: float = 5.0
    """Maximum receive-time skew among the four VLA camera frames."""

    timing_endpoint: str = "tcp://127.0.0.1:5567"
    """Best-effort VLA timing telemetry endpoint; publishing never blocks inference."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    control_gateway_endpoint: str = "tcp://127.0.0.1:5565"
    """Structured ControlGateway command endpoint."""

    # ZMQ: Planner relay (run_planner_keyboard sidecar -> forward to action port)
    planner_relay_zmq_host: str = "localhost"
    """Host for the planner relay SUB socket (sidecar PUB connects here)."""

    planner_relay_zmq_port: int = 5558
    """Port for the planner relay SUB socket."""

    # Embodiment
    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


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

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.ZMQError:
            return False

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
# Hold completed chunks so latency compensation selects points farther into the trajectory.
SIMULATED_INFERENCE_DELAY_SECONDS = 0.0


class _TimedObservation(dict):
    """Observation dict carrying local measurements outside the wire payload."""

    def __init__(self, value: dict, timing_ms: dict[str, float]):
        super().__init__(value)
        self.timing_ms = timing_ms


class _TimedAction(dict):
    """Processed action dict carrying measurements outside the action payload."""

    def __init__(self, value: dict, timing_ms: dict[str, float]):
        super().__init__(value)
        self.timing_ms = timing_ms


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
    log_errors: bool = False,
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
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    started = time.perf_counter()
    state_msg = sensor_gateway.read_state()
    timing_ms["state_read"] = (time.perf_counter() - started) * 1000.0
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
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
    return _TimedObservation(observation, timing_ms)


def run_policy_inference_and_process(policy, observation, robot_model):
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
            print(
                f"[Warning] action['{motion_key}'] max "
                f"({np.abs(action[motion_key]).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None

        processed_action = concat_action(robot_model, action)
        timing_ms["action_postprocess"] = (
            time.perf_counter() - postprocess_started
        ) * 1000.0
        return _TimedAction(processed_action, timing_ms)
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
    simulated_inference_delay_seconds: float = SIMULATED_INFERENCE_DELAY_SECONDS,
    timing_callback=None,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                request_generation = inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # ``None`` keeps the worker helper backward-compatible with old
            # callers while production requests always carry a generation.
            if request_generation is None:
                request_generation = 0

            busy_event.set()
            try:
                worker_started = time.perf_counter()
                observation = prepare_obs_fn()
                if observation is None:
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                timing_ms = dict(getattr(observation, "timing_ms", {}))
                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)
                timing_ms.update(getattr(processed_action, "timing_ms", {}))
                timing_ms["worker_total"] = (
                    time.perf_counter() - worker_started
                ) * 1000.0
                if timing_callback is not None:
                    try:
                        timing_callback(timing_ms)
                    except Exception:
                        # Diagnostics are deliberately best-effort and must not
                        # alter policy scheduling or produce repetitive logs.
                        pass

                if processed_action is not None:
                    if stop_event.wait(simulated_inference_delay_seconds):
                        continue
                    try:
                        result_queue.put_nowait(
                            (request_generation, processed_action, inference_start_time)
                        )
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait(
                                (request_generation, processed_action, inference_start_time)
                            )
                        except queue.Empty:
                            result_queue.put_nowait(
                                (request_generation, processed_action, inference_start_time)
                            )
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


def _pose_policy_is_active(cpp_loop_running: bool, cpp_mode: str, pause_loop: bool) -> bool:
    """Return whether a fresh VLA inference/action is allowed to run."""
    return cpp_loop_running and cpp_mode == "POSE" and not pause_loop


def _vla_inference_is_due(
    worker_is_busy: bool,
    request_queue_is_empty: bool,
    time_since_request: float,
    inference_interval: float,
) -> bool:
    """Schedule observations independently from robot action/mode gating."""
    return (
        not worker_is_busy
        and request_queue_is_empty
        and time_since_request >= inference_interval
    )


def _should_schedule_vla_inference(
    *,
    cpp_mode: str,
    worker_is_busy: bool,
    request_queue_is_empty: bool,
    time_since_request: float,
    inference_interval: float,
) -> bool:
    """Capture POSE observations even when action publication is paused."""

    return cpp_mode == "POSE" and _vla_inference_is_due(
        worker_is_busy=worker_is_busy,
        request_queue_is_empty=request_queue_is_empty,
        time_since_request=time_since_request,
        inference_interval=inference_interval,
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
    pause_loop = True

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    n1_policy = _MsgpackNumpyPolicyClient(host=config.host, port=config.port)
    timing_publisher = VlaTimingPublisher(config.timing_endpoint)
    print_green("Policy wire protocol: msgpack_numpy")

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        print("WARNING: PolicyServer not reachable. Inference will fail until server is up.")

    gateway_ingress = VlaSensorGatewayIngress(
        config.sensor_gateway_endpoint,
        poll_hz=config.sensor_gateway_poll_hz,
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        max_skew_ms=config.sensor_gateway_max_skew_ms,
    )
    gateway_ingress.start()
    print_green(f"VLA sensors: Gateway cache at {config.sensor_gateway_endpoint}")

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(
        f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    )
    print_green(f"Using embodiment tag: {config.embodiment_tag}")

    control_listener = ControlGatewaySubscriber(
        config.control_gateway_endpoint,
        accepted_names={
            "start_recording",
            "stop_recording_success",
            "stop_recording_failure",
            "select_pose_mode",
            "toggle_control_loop",
            "select_planner_mode",
            "toggle_policy_pause",
            "toggle_left_hand_initial_pose",
            "toggle_right_hand_initial_pose",
            "set_prompt",
        },
    )

    planner_relay_sub = zmq_context.socket(zmq.SUB)
    planner_relay_sub.setsockopt_string(zmq.SUBSCRIBE, "planner")
    planner_relay_sub.setsockopt(zmq.RCVTIMEO, 0)
    planner_relay_sub.connect(
        f"tcp://{config.planner_relay_zmq_host}:{config.planner_relay_zmq_port}"
    )
    planner_heading_alignment = PlannerHeadingAlignment()
    print_green(
        "Planner relay SUB connected to "
        f"tcp://{config.planner_relay_zmq_host}:{config.planner_relay_zmq_port} "
        f"with topic filter: planner"
        f"(forwarding to tcp://{config.action_zmq_host}:{config.action_zmq_port})"
    )

    telemetry = Telemetry(window_size=100)

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
            print("Warning: Cannot publish initial pose in non-PLANNER mode or if C++ loop is not running")
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
            print("Error: Cannot read current body_q for initial pose ramp. Aborting.")
            return False

        hold_facing = planner_heading_alignment.current_facing()
        hold_yaw = math.atan2(hold_facing[1], hold_facing[0])
        print_green(f"[HeadingSync] holding planner facing {hold_yaw:+.3f} rad during POSE entry")

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

        print(f"Moving to initial pose (PLANNER upper body ramp)...")
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


        print_green("Initial pose published")
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
                print(
                    "[HeadingSync] preparing PLANNER entry: "
                    f"measured_yaw={measured_yaw}, discarded_stale={discarded}"
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
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    last_inference_request_time = 0.0
    inference_interval = 1.0 / config.rate

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_generation = 0

    def invalidate_inference(reason: str):
        """Invalidate cached, queued, and currently-running inference work."""
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal last_inference_request_time
        nonlocal inference_generation
        inference_generation += 1
        cached_action_chunk = None
        action_chunk_index = 0
        last_inference_time = 0.0
        last_inference_request_time = 0.0
        queued = _drain_queue(inference_queue)
        results = _drain_queue(result_queue)
        print(
            f"[VLA] invalidated inference generation {inference_generation}: {reason} "
            f"(queued={queued}, results={results})"
        )

    zmq_frame_counter = 0

    def check_control_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter

        command = control_listener.read_command()
        if command is None:
            return

        command_name = command.name
        if command_name == "set_prompt":
            new_prompt = command.parameters.get("prompt")
            if isinstance(new_prompt, str) and new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                invalidate_inference("prompt changed")
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if command_name == "start_recording":
            print("Start recording command received; handled by data exporter")
        elif command_name == "stop_recording_success":
            print("Successful recording stop received; handled by data exporter")
        elif command_name == "stop_recording_failure":
            print("Failed recording stop received; handled by data exporter")
        elif command_name == "select_pose_mode":
            print("Switch to pose mode")
            zmq_frame_counter = 0
            print("Reset ZMQ frame counter")
            invalidate_inference("entering POSE mode")
            if cpp_mode == "PLANNER":
                if not publish_initial_pose():
                    print("Warning: POSE transition preparation failed; remaining in PLANNER mode")
                    return
                print("Switching to POSE mode")
                if send_cpp_control_command(start=True, planner=False):
                    print("Switched to POSE mode (from PLANNER mode)")
                else:
                    print("Warning: Failed to switch to POSE mode")
            elif cpp_mode == "POSE":
                print("Warning: C++ loop is already in POSE mode")
            else:
                print("Warning: C++ loop is not running")
        elif command_name == "select_planner_mode":
            print("Switch to planner mode")
            zmq_frame_counter = 0
            print("Reset ZMQ frame counter")
            invalidate_inference("entering PLANNER mode")
            if cpp_mode == "POSE":
                print("Switching to PLANNER mode")
                if send_cpp_control_command(start=True, planner=True):
                    print("Switched to PLANNER mode (from POSE mode)")
                else:
                    print("Warning: Failed to switch to PLANNER mode")
            elif cpp_mode == "PLANNER":
                print("Warning: C++ loop is already in PLANNER mode")
            else:
                print("Warning: C++ loop is not running")
        elif command_name == "toggle_policy_pause":
            if cpp_mode == "PLANNER":
                print("Warning: C++ loop is in PLANNER mode - press 'i' to switch to POSE mode")
            else:
                pause_loop = not pause_loop
                invalidate_inference(
                    "POSE policy resumed" if not pause_loop else "POSE policy paused"
                )
                print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
                if pause_loop:
                    print("Policy loop paused (C++ loop still running - press 'k' to stop)")
                else:
                    print("Policy loop resumed (C++ loop still running - press 'k' to stop)")

        elif command_name == "toggle_control_loop":
            invalidate_inference("C++ control toggled")
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif command_name == "toggle_left_hand_initial_pose":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(
                f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}"
            )
        elif command_name == "toggle_right_hand_initial_pose":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(
                f"Initial pose right hand: "
                f"{'closed' if initial_pose_right_hand_closed else 'open'}"
            )

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

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
                log_errors=True,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                robot_model=robot_model,
            ),
        ),
        kwargs={"timing_callback": timing_publisher.publish},
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_control_input()

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                result_generation, processed_action, inference_start_time = (
                    result_queue.get_nowait()
                )
                inference_delay = time.monotonic() - inference_start_time
                if (
                    result_generation == inference_generation
                    and _pose_policy_is_active(cpp_loop_running, cpp_mode, pause_loop)
                ):
                    action_chunk_index = calculate_latency_compensated_index(
                        inference_delay, config.action_publish_rate, config.action_horizon
                    )
                    cached_action_chunk = processed_action
                    last_inference_time = time.monotonic()
                    print_green(
                        f'New action chunk (prompt: "{language_prompt_ref[0]}", '
                        f"latency: {inference_delay:.3f}s)"
                    )
                elif result_generation != inference_generation:
                    print(
                        f"[VLA] dropped stale inference generation {result_generation}; "
                        f"active={inference_generation}, mode={cpp_mode}, "
                        f"paused={pause_loop}, latency={inference_delay:.3f}s"
                    )
                else:
                    # This can only be a request that was already in flight
                    # when POSE was paused or switched to PLANNER. Its action
                    # is discarded and no new observation is scheduled.
                    print(
                        f"[VLA] inference received; action held "
                        f"(mode={cpp_mode}, paused={pause_loop}, "
                        f"latency={inference_delay:.3f}s)"
                    )
            except queue.Empty:
                pass

            worker_is_busy = inference_busy_event.is_set()
            now = time.monotonic()
            should_start = _should_schedule_vla_inference(
                cpp_mode=cpp_mode,
                worker_is_busy=worker_is_busy,
                request_queue_is_empty=inference_queue.empty(),
                time_since_request=(now - last_inference_request_time),
                inference_interval=inference_interval,
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
                )
                print("In Planner mode...", end="", flush=True)
                _sleep_remaining(t_start, loop_period)
                print(".", end="", flush=True)
                continue

            if pause_loop:
                print("Pausing...", end="", flush=True)
                time.sleep(0.2)
                print(".", end="", flush=True)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                else:
                    motion_token = np.asarray(
                        get_action_field(processed_action, "motion_token"),
                        dtype=np.float32,
                    )
                    left_hand_joints = np.asarray(
                        get_action_field(processed_action, "left_hand_joints"),
                        dtype=np.float32,
                    )
                    right_hand_joints = np.asarray(
                        get_action_field(processed_action, "right_hand_joints"),
                        dtype=np.float32,
                    )

                    # Action arrays arrive as (B, T, D) from the model.
                    # Squeeze batch dim to get (T, D), then index by time step.
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

                    zmq_message = pack_latent_action_message(
                        motion_token,
                        frame_index,
                        left_hand_joints=left_hand_joints,
                        right_hand_joints=right_hand_joints,
                    )
                    zmq_socket.send(zmq_message)
                    if zmq_frame_counter % 50 == 0:
                        print_green(
                            f"ZMQ: Sent latent action - "
                            f"frame: {frame_index[0]}, "
                            f"token shape: {motion_token.shape}"
                        )

                action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(
                    context="VLA Inference Loop Missed", threshold=0.001
                )

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        planner_relay_sub.close()
        zmq_socket.close()
        zmq_context.term()
        gateway_ingress.close()
        control_listener.close()
        n1_policy.close()
        timing_publisher.close()
        print("Shutdown complete.")


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
) -> int:
    """Forward only the newest planner command in the current VLA heading frame."""
    message, received = _receive_latest_planner_message(planner_relay_sub)
    if message is None:
        return 0

    aligned = heading_alignment.align(message)
    if aligned is None:
        print("[HeadingSync] planner relay is waiting for mode-entry alignment")
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
    config = tyro.cli(InferenceConfig)
    main(config)
