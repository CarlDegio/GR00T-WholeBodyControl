"""NavDP trajectory conversion and asynchronous MPC logic."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import queue
import threading
import time
from typing import Any, Sequence

import numpy as np

from gear_sonic.utils.inference.navdp.navigation import (
    Pose2D,
    local_trajectory_to_world,
)
from gear_sonic.runtime.profile import load_runtime_profile

LOGGER = logging.getLogger("sonic.navdp")

XNAVDP_G1_MPC_DEFAULTS = dict(
    load_runtime_profile().component("xnavdp_mpc")
)


def xnavdp_control_to_body_velocity(
    linear_velocity: float, angular_velocity: float
) -> tuple[float, float, float]:
    """Map X-NavDP's unicycle control to a body-frame velocity command."""
    return float(linear_velocity), 0.0, float(angular_velocity)


def fastlio_heading_target_from_mpc(
    *,
    fastlio_yaw: float,
    mpc_angular_velocity: float,
    heading_preview_s: float,
) -> float:
    """Preview an absolute heading target in the FAST-LIO frame."""
    return math.remainder(
        float(fastlio_yaw)
        + float(mpc_angular_velocity) * float(heading_preview_s),
        2.0 * math.pi,
    )


def xnavdp_adaptive_speed(
    trajectory_length_m: float,
    maximum_curvature: float,
    *,
    desired_velocity: float = float(XNAVDP_G1_MPC_DEFAULTS["desired_velocity"]),
    max_linear_velocity: float = float(
        XNAVDP_G1_MPC_DEFAULTS["max_linear_velocity"]
    ),
    max_angular_velocity: float = float(
        XNAVDP_G1_MPC_DEFAULTS["max_angular_velocity"]
    ),
    reference_trajectory_length_m: float = float(
        XNAVDP_G1_MPC_DEFAULTS["reference_trajectory_length_m"]
    ),
    minimum_desired_velocity: float = float(
        XNAVDP_G1_MPC_DEFAULTS["minimum_desired_velocity"]
    ),
    curvature_speed_gain: float = float(
        XNAVDP_G1_MPC_DEFAULTS["curvature_speed_gain"]
    ),
) -> float:
    """Match X-NavDP's length- and curvature-limited reference speed."""
    length_scale = min(
        max(float(trajectory_length_m), 0.0) / float(reference_trajectory_length_m),
        1.0,
    )
    length_limit = float(desired_velocity) * length_scale
    curvature = max(float(maximum_curvature), 1.0e-6)
    curvature_limit = min(
        float(max_linear_velocity),
        float(curvature_speed_gain) * float(max_angular_velocity) / curvature,
    )
    return max(
        float(minimum_desired_velocity),
        min(float(max_linear_velocity), length_limit, curvature_limit),
    )


def prepare_internnav_world_reference(
    trajectory: np.ndarray,
    inference_pose: Pose2D,
    *,
    skip_points: int = 3,
    interpolation_ratio: int = int(XNAVDP_G1_MPC_DEFAULTS["interpolation_ratio"]),
) -> np.ndarray:
    """Match InternNav real-world preprocessing for a NavDP local trajectory."""
    points = np.asarray(trajectory, dtype=np.float64).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[int(skip_points) :]
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float64)
    world = local_trajectory_to_world(points, inference_pose).astype(np.float64)
    ratio = max(1, int(interpolation_ratio))
    if ratio == 1:
        return world
    source = np.arange(len(world), dtype=np.float64)
    target = np.linspace(0.0, len(world) - 1, len(world) * ratio)
    return np.column_stack(
        (np.interp(target, source, world[:, 0]), np.interp(target, source, world[:, 1]))
    )


class InternNavMpcController:
    """X-NavDP G1 nonlinear MPC adapted to SONIC's 10 Hz planner."""

    def __init__(
        self,
        world_reference: np.ndarray,
        *,
        horizon_steps: int = int(XNAVDP_G1_MPC_DEFAULTS["horizon_steps"]),
        desired_velocity: float = float(XNAVDP_G1_MPC_DEFAULTS["desired_velocity"]),
        max_linear_velocity: float = float(
            XNAVDP_G1_MPC_DEFAULTS["max_linear_velocity"]
        ),
        max_angular_velocity: float = float(
            XNAVDP_G1_MPC_DEFAULTS["max_angular_velocity"]
        ),
        reference_gap: int = int(XNAVDP_G1_MPC_DEFAULTS["reference_gap"]),
        dt: float = float(XNAVDP_G1_MPC_DEFAULTS["dt"]),
        reference_trajectory_length_m: float = float(
            XNAVDP_G1_MPC_DEFAULTS["reference_trajectory_length_m"]
        ),
        minimum_desired_velocity: float = float(
            XNAVDP_G1_MPC_DEFAULTS["minimum_desired_velocity"]
        ),
        lookahead_points: int = int(XNAVDP_G1_MPC_DEFAULTS["lookahead_points"]),
        linear_control_weight: float = float(
            XNAVDP_G1_MPC_DEFAULTS["linear_control_weight"]
        ),
        angular_control_weight: float = float(
            XNAVDP_G1_MPC_DEFAULTS["angular_control_weight"]
        ),
        curvature_speed_gain: float = float(
            XNAVDP_G1_MPC_DEFAULTS["curvature_speed_gain"]
        ),
    ) -> None:
        import casadi as ca

        self.horizon_steps = int(horizon_steps)
        self.desired_velocity = float(desired_velocity)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.reference_gap = int(reference_gap)
        self.dt = float(dt)
        self.reference_trajectory_length_m = float(reference_trajectory_length_m)
        self.minimum_desired_velocity = float(minimum_desired_velocity)
        self.lookahead_points = int(lookahead_points)
        self.linear_control_weight = float(linear_control_weight)
        self.angular_control_weight = float(angular_control_weight)
        self.curvature_speed_gain = float(curvature_speed_gain)
        self.reference_count = self.horizon_steps // self.reference_gap + 1
        self.world_reference = np.asarray(world_reference, dtype=np.float64).reshape(-1, 2)
        if len(self.world_reference) < 2:
            raise ValueError("MPC world reference requires at least two points")

        optimizer = ca.Opti()
        controls = optimizer.variable(self.horizon_steps, 2)
        states = optimizer.variable(self.horizon_steps + 1, 3)
        initial_state = optimizer.parameter(3)
        reference_states = optimizer.parameter(3 * self.reference_count)
        optimizer.subject_to(states[0, :] == initial_state.T)
        for index in range(self.horizon_steps):
            x, y, yaw = states[index, 0], states[index, 1], states[index, 2]
            linear, angular = controls[index, 0], controls[index, 1]
            state = ca.vertcat(x, y, yaw)
            control = ca.vertcat(linear, angular)

            def dynamics(value):
                return ca.vertcat(
                    control[0] * ca.cos(value[2]),
                    control[0] * ca.sin(value[2]),
                    control[1],
                )

            k1 = dynamics(state)
            k2 = dynamics(state + 0.5 * self.dt * k1)
            k3 = dynamics(state + 0.5 * self.dt * k2)
            k4 = dynamics(state + self.dt * k3)
            next_state = (state + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6).T
            optimizer.subject_to(states[index + 1, :] == next_state)

        state_weight = np.diag([10.0, 10.0, 0.0])
        control_weight = np.diag(
            [self.linear_control_weight, self.angular_control_weight]
        )
        objective = 0
        for index in range(self.horizon_steps):
            objective += ca.mtimes(
                [controls[index, :], control_weight, controls[index, :].T]
            )
            if index % self.reference_gap == 0:
                ref_index = index // self.reference_gap
                error = states[index, :] - reference_states[
                    ref_index * 3 : ref_index * 3 + 3
                ].T
                objective += ca.mtimes([error, state_weight, error.T])
        terminal_error = states[-1, :] - reference_states[-3:].T
        objective += ca.mtimes([terminal_error, state_weight, terminal_error.T])
        optimizer.minimize(objective)
        optimizer.subject_to(
            optimizer.bounded(
                -self.max_linear_velocity,
                controls[:, 0],
                self.max_linear_velocity,
            )
        )
        optimizer.subject_to(
            optimizer.bounded(
                -self.max_angular_velocity,
                controls[:, 1],
                self.max_angular_velocity,
            )
        )
        optimizer.solver(
            "ipopt",
            {
                "ipopt.max_iter": 100,
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.acceptable_tol": 1.0e-8,
                "ipopt.acceptable_obj_change_tol": 1.0e-6,
            },
        )
        self._optimizer = optimizer
        self._controls = controls
        self._states = states
        self._initial_state = initial_state
        self._reference_states = reference_states
        self._last_controls: np.ndarray | None = None
        self._last_states: np.ndarray | None = None
        self._update_desired_velocity()

    def update_reference(self, world_reference: np.ndarray) -> None:
        reference = np.asarray(world_reference, dtype=np.float64).reshape(-1, 2)
        if len(reference) < 2:
            raise ValueError("MPC world reference requires at least two points")
        self.world_reference = reference
        self._update_desired_velocity()

    def _update_desired_velocity(self) -> None:
        delta = np.diff(self.world_reference, axis=0)
        trajectory_length = float(np.linalg.norm(delta, axis=1).sum())
        dx = np.gradient(self.world_reference[:, 0])
        dy = np.gradient(self.world_reference[:, 1])
        if len(dy):
            dy[0] = 0.0
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1.0e-6)
        curvature = np.abs(dx * ddy - dy * ddx) / denominator
        curvature = np.convolve(curvature, np.ones(3) / 3.0, mode="same")
        maximum_curvature = float(np.max(curvature[:12])) if len(curvature) else 0.0
        self.desired_velocity = xnavdp_adaptive_speed(
            trajectory_length,
            maximum_curvature,
            desired_velocity=self.max_linear_velocity,
            max_linear_velocity=self.max_linear_velocity,
            max_angular_velocity=self.max_angular_velocity,
            reference_trajectory_length_m=self.reference_trajectory_length_m,
            minimum_desired_velocity=self.minimum_desired_velocity,
            curvature_speed_gain=self.curvature_speed_gain,
        )

    def _select_reference(self, pose: Pose2D) -> np.ndarray:
        distances = np.linalg.norm(
            self.world_reference - np.array([pose.x, pose.y]), axis=1
        )
        nearest = int(np.argmin(distances))
        start = min(nearest + self.lookahead_points, len(self.world_reference) - 1)
        remaining = self.world_reference[start:]
        arc = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(remaining, axis=0), axis=1)))
        )
        spacing = self.desired_velocity * self.reference_gap * self.dt
        indices = [
            int(np.searchsorted(arc, spacing * index, side="left"))
            for index in range(self.reference_count)
        ]
        indices = np.clip(indices, 0, len(remaining) - 1)
        xy = remaining[indices]
        return np.column_stack((xy, np.zeros(self.reference_count)))

    def solve(self, pose: Pose2D) -> tuple[float, float]:
        reference = self._select_reference(pose)
        self._optimizer.set_value(
            self._initial_state, np.array([pose.x, pose.y, pose.yaw])
        )
        self._optimizer.set_value(self._reference_states, reference.reshape(-1))
        controls_guess = (
            np.zeros((self.horizon_steps, 2))
            if self._last_controls is None
            else self._last_controls
        )
        states_guess = (
            np.zeros((self.horizon_steps + 1, 3))
            if self._last_states is None
            else self._last_states
        )
        self._optimizer.set_initial(self._controls, controls_guess)
        self._optimizer.set_initial(self._states, states_guess)
        solution = self._optimizer.solve()
        self._last_controls = np.asarray(solution.value(self._controls))
        self._last_states = np.asarray(solution.value(self._states))
        return float(self._last_controls[0, 0]), float(self._last_controls[0, 1])


@dataclass(frozen=True)
class MpcSolveRequest:
    generation: int
    reference_version: int
    world_reference: np.ndarray
    pose: Pose2D


@dataclass(frozen=True)
class MpcSolveResult:
    generation: int
    reference_version: int
    control: tuple[float, float] | None
    completed_time: float
    elapsed_s: float
    error: str | None = None
    return_status: str = "unknown"


class AsyncMpcSolver:
    """Own the CasADi controller on a worker thread and keep only latest work."""

    def __init__(self, *, controller_factory=InternNavMpcController) -> None:
        self._controller_factory = controller_factory
        self._requests: queue.Queue[MpcSolveRequest | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[MpcSolveResult] = queue.Queue(maxsize=1)
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _replace(queue_: queue.Queue, item) -> None:
        try:
            queue_.get_nowait()
        except queue.Empty:
            pass
        queue_.put_nowait(item)

    def submit(self, request: MpcSolveRequest) -> None:
        if not self._closed:
            self._replace(self._requests, request)

    def poll_latest(self) -> MpcSolveResult | None:
        latest = None
        while True:
            try:
                latest = self._results.get_nowait()
            except queue.Empty:
                return latest

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._replace(self._requests, None)
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        controller = None
        reference_version = -1
        while True:
            request = self._requests.get()
            if request is None:
                return
            started = time.perf_counter()
            control = None
            error = None
            return_status = "unknown"
            try:
                if controller is None:
                    controller = self._controller_factory(request.world_reference)
                elif request.reference_version != reference_version:
                    controller.update_reference(request.world_reference)
                reference_version = request.reference_version
                control = tuple(map(float, controller.solve(request.pose)))
                try:
                    return_status = controller._optimizer.stats().get(
                        "return_status", "unknown"
                    )
                except Exception:
                    return_status = "success"
            except Exception as exc:
                error = str(exc)
                try:
                    return_status = controller._optimizer.stats().get(
                        "return_status", "unknown"
                    )
                    controller._last_controls = None
                    controller._last_states = None
                except Exception:
                    pass
            self._replace(
                self._results,
                MpcSolveResult(
                    generation=request.generation,
                    reference_version=request.reference_version,
                    control=control,
                    completed_time=time.monotonic(),
                    elapsed_s=time.perf_counter() - started,
                    error=error,
                    return_status=return_status,
                ),
            )


class LatestMessageWorker:
    """Process expensive sensor messages off the ROS executor, newest first."""

    def __init__(self, processor) -> None:
        self._processor = processor
        self._messages: queue.Queue[Any | None] = queue.Queue(maxsize=1)
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, message: Any) -> None:
        if self._closed:
            return
        try:
            self._messages.get_nowait()
        except queue.Empty:
            pass
        self._messages.put_nowait(message)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._messages.get_nowait()
        except queue.Empty:
            pass
        self._messages.put_nowait(None)
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            message = self._messages.get()
            if message is None:
                return
            try:
                self._processor(message)
            except Exception as exc:
                LOGGER.exception("sensor processing failed: %s", exc)


def fresh_mpc_control(
    control: tuple[float, float],
    *,
    result_time: float,
    now: float,
    timeout_s: float,
) -> tuple[float, float]:
    """Use a successful MPC result only for a bounded amount of time."""
    return control if now - result_time <= timeout_s else (0.0, 0.0)


def should_abort_nav_for_zero_action(
    *,
    mode: str,
    selected_command: Sequence[float],
    command_available: bool,
) -> bool:
    """Treat an explicit zero MPC macro action as completion of this NAV task."""
    command = tuple(map(float, selected_command))
    return bool(
        mode == "nav_goal"
        and command_available
        and np.allclose(command, (0.0, 0.0, 0.0), atol=1.0e-9)
    )
