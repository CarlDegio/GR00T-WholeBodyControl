"""Bounded, deterministic recovery using a read-only local navmesh oracle.

All points are FLU XY relative to the body at planning time. The oracle owns
the Habitat-world conversion. No object transforms or ground-truth yaw enter
this module. Every path edge must cross the real action deadzone.
"""

from dataclasses import dataclass, replace
import heapq
import math

import numpy as np


@dataclass
class RecoveryPlan:
    waypoints: list
    effective_distance_m: float
    diagnostics: dict
    effective_lateral_tolerance_m: float = 0.06


class RecoveryPlanner:
    def __init__(self, config, limits):
        self.config = config
        self.limits = limits
        self.minimum = limits.min_displacement_m * (1 + 1e-5)
        self.maximum = min(config.max_displacement_m, limits.max_displacement_m) - 1e-6
        if self.minimum >= self.maximum:
            raise ValueError("Recovery needs room above the minimum action length")

    def _line(self, start, end):
        """Split a long line; short legs require a separately checked bridge."""
        length = float(np.linalg.norm(end - start))
        if length < 1e-8:
            return []
        count = int(math.ceil(length / self.maximum))
        if length / count < self.minimum:
            return None
        return [start + (end - start) * i / count for i in range(1, count + 1)]

    def _polyline(self, corners):
        points = []
        for start, end in zip(corners, corners[1:]):
            leg = self._line(start, end)
            if leg is None:
                return None
            points.extend(leg)
        return points

    def _route(self, oracle, goal, blocked_axis):
        zero = np.zeros(2)
        other = 1 - blocked_axis
        corner = zero.copy()
        corner[other] = goal[other]
        candidates = [("switch_axis_order", [zero, corner, goal])]
        # A short correction cannot be sent alone. Move on the other axis
        # first, then return while completing the originally blocked axis.
        sign = 1.0 if goal[other] >= 0 else -1.0
        leads = [
            goal[other] + sign * self.minimum,
            sign * self.minimum,
            -sign * self.minimum,
            goal[other] - sign * self.minimum,
        ]
        for lead in leads:
            bridge = zero.copy()
            bridge[other] = lead
            candidates.append(("switch_axis_bridge", [zero, bridge, goal]))
        candidates.append(("direct_combined_correction", [zero, goal]))
        corner = zero.copy()
        corner[blocked_axis] = goal[blocked_axis]
        candidates.append(("original_axis_order", [zero, corner, goal]))
        for strategy, corners in candidates:
            path = self._polyline(corners)
            if path and self._safe_path(oracle, path):
                return path, strategy
        return self._grid_route(oracle, goal)

    def _safe_path(self, oracle, points):
        start = np.zeros(2)
        for end in points:
            if not oracle.segment(start, end, self.config.navmesh_clearance_m):
                return False
            start = end
        return True

    def _grid_route(self, oracle, goal):
        """Finite local visibility graph; never claim completeness outside it."""
        spacing = self.config.navmesh_grid_spacing_m
        n = int(self.config.navmesh_local_radius_m / spacing)
        points = [np.zeros(2), goal]
        for x in range(-n, n + 1):
            for y in range(-n, n + 1):
                p = spacing * np.array([x, y], dtype=float)
                if np.linalg.norm(p) < 1e-8 or np.linalg.norm(p - goal) < 1e-8:
                    continue
                if oracle.endpoint(p, self.config.navmesh_clearance_m)["reachable"]:
                    points.append(p)
        points = np.asarray(points)
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        costs, parents = {0: 0.0}, {}
        queue = [(0.0, 0)]
        while queue:
            cost, i = heapq.heappop(queue)
            if cost > costs[i] + 1e-12:
                continue
            if i == 1:
                indices = [1]
                while parents[indices[-1]] != 0:
                    indices.append(parents[indices[-1]])
                return [points[j] for j in reversed(indices)], "local_visibility_graph"
            neighbors = np.flatnonzero(
                (distances[i] >= self.minimum) & (distances[i] <= self.maximum)
            )
            for j in neighbors:
                new_cost = cost + distances[i, j]
                if new_cost >= costs.get(int(j), math.inf) - 1e-12:
                    continue
                if not oracle.segment(
                    points[i], points[j], self.config.navmesh_clearance_m
                ):
                    continue
                costs[int(j)], parents[int(j)] = new_cost, i
                heapq.heappush(queue, (new_cost, int(j)))
        return None, "no_executable_path_in_local_graph"

    def plan(self, oracle, forward_m, right_m, blocked_axis):
        plan = self._plan(oracle, forward_m, right_m, blocked_axis)
        plan.effective_lateral_tolerance_m = self.config.lateral_tolerance_m
        plan.diagnostics.update(lateral_expanded_search=False,
            nominal_lateral_tolerance_m=self.config.lateral_tolerance_m,
            effective_lateral_tolerance_m=plan.effective_lateral_tolerance_m)
        maximum = self.config.navmesh_max_lateral_tolerance_m
        if plan.waypoints or maximum <= self.config.lateral_tolerance_m:
            return plan
        # Exhaust both original and outer distances at the nominal lateral
        # tolerance before relaxing this one endpoint constraint.
        expanded = RecoveryPlanner(replace(self.config,
            lateral_tolerance_m=maximum, navmesh_max_lateral_tolerance_m=0.0), self.limits
        )._plan(oracle, forward_m, right_m, blocked_axis)
        expanded.effective_lateral_tolerance_m = max(
            self.config.lateral_tolerance_m,
            abs(expanded.diagnostics.get('planned_right_residual_m', 0.0))
                + self.config.navmesh_execution_tolerance_m,
        )
        expanded.diagnostics.update(lateral_expanded_search=True,
            nominal_lateral_tolerance_m=self.config.lateral_tolerance_m,
            effective_lateral_tolerance_m=expanded.effective_lateral_tolerance_m,
            strict_lateral_search=dict(
                failure=plan.diagnostics.get('failure'),
                original_and_outer_distances_exhausted=True,
                distance_trials=len(plan.diagnostics['tested_distances']),
                endpoint_trials=sum(item['lateral_candidates_tested']
                                    for item in plan.diagnostics['tested_distances']),
            ))
        return expanded

    def _plan(self, oracle, forward_m, right_m, blocked_axis):
        nominal = self.config.target_distance_m
        goal = np.array([forward_m - nominal, -right_m])
        nominal_check = oracle.endpoint(goal, 0.0)
        report = {
            "experiment": "navmesh/oracle-map recovery",
            "nominal_distance_m": nominal,
            "nominal_goal_local_xy": goal.tolist(),
            "nominal_endpoint": nominal_check,
            "blocked_axis": "x" if blocked_axis == 0 else "y",
            "search_resolution_m": self.config.navmesh_search_resolution_m,
            "clearance_m": self.config.navmesh_clearance_m,
            "standoff_reserve_m": self.config.navmesh_standoff_reserve_m,
            "local_radius_m": self.config.navmesh_local_radius_m,
            "max_target_distance_m": self.config.navmesh_max_target_distance_m,
            "expanded_search": False,
            "tested_distances": [],
        }
        resolution = self.config.navmesh_search_resolution_m
        count = int(math.floor(
            (nominal - self.config.navmesh_min_target_distance_m) / resolution + 1e-9
        ))
        # Preserve the successful original search order. A reachable nominal
        # endpoint with no executable route must also exhaust the closer range.
        original_distances = ([nominal] if nominal_check["reachable"] else []) + [
            nominal - i * resolution for i in range(1, count + 1)
        ]
        outer_count = int(math.floor(
            (self.config.navmesh_max_target_distance_m - nominal) / resolution + 1e-9
        ))
        expanded_distances = [nominal + i * resolution for i in range(1, outer_count + 1)]
        bound = self.config.lateral_tolerance_m - self.config.navmesh_execution_tolerance_m
        n = int(math.floor(bound / resolution + 1e-9))
        lateral_residuals = [0.0] + [
            sign * i * resolution for i in range(1, n + 1) for sign in (1, -1)
        ]
        report.update(
            branch="nominal_reachable" if nominal_check["reachable"] else "nominal_unreachable",
            lateral_residual_bound_m=bound,
            lateral_search_resolution_m=resolution,
            objective="original range first; then nearest farther standoff; tie-break by smallest lateral residual",
        )
        for search_phase, distances in (
            ("original", original_distances), ("expanded", expanded_distances)
        ):
            if not distances:
                continue
            report["search_phase"] = search_phase
            if search_phase == "expanded":
                report["expanded_search"] = True
                report["expansion_reason"] = "no safe executable path in original standoff range"
            for distance in distances:
                trial = {
                    "distance_m": distance,
                    "search_phase": search_phase,
                    "reachable": False,
                    "lateral_candidates_tested": 0,
                    "rejection_reasons": {},
                }
                report["tested_distances"].append(trial)
                if (
                    distance < nominal
                    and distance + self.config.navmesh_standoff_reserve_m >= nominal - 1e-12
                ):
                    trial["rejection_reasons"]["strict standoff observation reserve"] = 1
                    continue
                residuals = [0.0] if distance == nominal else lateral_residuals
                for residual in residuals:
                    goal = np.array([forward_m - distance, residual - right_m])
                    check = oracle.endpoint(goal, self.config.navmesh_clearance_m)
                    trial["lateral_candidates_tested"] += 1
                    if not check["reachable"]:
                        reason = check["reason"]
                        trial["rejection_reasons"][reason] = (
                            trial["rejection_reasons"].get(reason, 0) + 1
                        )
                        continue
                    trial["reachable"] = True
                    path, strategy = self._route(oracle, goal, blocked_axis)
                    trial["route"] = strategy
                    if path:
                        report.update(
                            effective_distance_m=distance,
                            planned_right_residual_m=residual,
                            effective_endpoint=check,
                            strategy=strategy,
                            waypoints_local_xy=[p.tolist() for p in path],
                            endpoint_local_xy=goal.tolist(),
                        )
                        return RecoveryPlan(path, distance, report)
                    trial["rejection_reasons"][strategy] = (
                        trial["rejection_reasons"].get(strategy, 0) + 1
                    )
        report["failure"] = (
            "no safe executable path within configured distance/grid bounds"
        )
        return RecoveryPlan([], nominal, report)


class RecoveryExecution:
    """Confirm each real action on a fresh GPS frame before sending another."""

    def __init__(self, config, limits):
        self.config = config
        self.planner = RecoveryPlanner(config, limits)
        self.provider = None
        self.reset()

    def reset(self):
        self.pending = None
        self.path = []
        self.blocked_axis = None
        self.events = []
        self.plans = []
        self.completed = False
        self.failure = None
        self.effective_distance_m = self.config.target_distance_m
        self.effective_lateral_tolerance_m = self.config.lateral_tolerance_m

    @staticmethod
    def rotation(obs):
        heading = float(np.asarray(obs.compass).reshape(-1)[0])
        c, s = math.cos(heading), math.sin(heading)
        return np.array([[c, -s], [s, c]])

    def observe(self, obs, frame):
        self.completed = False
        if self.pending is None:
            return
        pending, self.pending = self.pending, None
        actual = pending["rotation"].T @ (np.asarray(obs.gps) - pending["gps"])
        command = pending["command"]
        violation = bool(self.provider.navmesh_violated())
        no_progress = np.linalg.norm(actual) < self.config.navmesh_execution_tolerance_m
        event = {
            "frame": frame,
            "command_frame": pending["frame"],
            "command_xy": command.tolist(),
            "actual_local_xy": actual.tolist(),
            "navmesh_violation": violation,
            "no_progress": bool(no_progress),
            "recovery_action": pending["recovery"],
        }
        self.events.append(event)
        if pending["recovery"]:
            error = float(np.linalg.norm(actual - command))
            event["execution_error_m"] = error
            if violation or error > self.config.navmesh_execution_tolerance_m:
                self.path = []
                self.blocked_axis = int(np.argmax(np.abs(command)))
                if not violation:
                    self.failure = (
                        "recovery action did not reach its commanded waypoint"
                    )
            else:
                self.path.pop(0)
                if not self.path:
                    self.completed = True
                    event["recovery_completed"] = True
        elif violation and no_progress:
            self.blocked_axis = int(np.argmax(np.abs(command)))

    def command(self, obs, frame, forward, right, image_yaw=None):
        if self.failure:
            return np.zeros(3)
        if not self.path and self.blocked_axis is not None:
            if len(self.plans) >= self.config.navmesh_max_replans:
                self.failure = "maximum navmesh recovery replans reached"
                return np.zeros(3)
            plan = self.planner.plan(
                self.provider.snapshot(), forward, right, self.blocked_axis
            )
            plan.diagnostics["frame"] = frame
            plan.diagnostics["image_yaw_at_plan_rad"] = image_yaw
            self.plans.append(plan.diagnostics)
            self.blocked_axis = None
            if not plan.waypoints:
                self.failure = plan.diagnostics["failure"]
                return np.zeros(3)
            self.effective_distance_m = plan.effective_distance_m
            self.effective_lateral_tolerance_m = plan.effective_lateral_tolerance_m
            rotation = self.rotation(obs)
            self.path = [np.asarray(obs.gps) + rotation @ p for p in plan.waypoints]
        if not self.path:
            return None
        local = self.rotation(obs).T @ (self.path[0] - np.asarray(obs.gps))
        length = float(np.linalg.norm(local))
        if not self.planner.minimum - 5e-7 <= length <= self.planner.maximum + 5e-7:
            self.failure = (
                "actual pose made the next recovery leg violate action limits"
            )
            return np.zeros(3)
        if not self.provider.snapshot().segment(
            np.zeros(2), local, self.config.navmesh_clearance_m
        ):
            self.failure = "fresh navmesh preflight rejected recovery leg"
            return np.zeros(3)
        return np.r_[local, 0.0]

    def record(self, xyt, obs, frame):
        if np.linalg.norm(xyt[:2]) == 0:
            return
        self.pending = {
            "command": np.asarray(xyt[:2]).copy(),
            "gps": np.asarray(obs.gps).copy(),
            "rotation": self.rotation(obs),
            "frame": frame,
            "recovery": bool(self.path),
        }

    def cancel_for_acquisition(self, reason="target reacquisition"):
        if self.path:
            self.events.append(
                {
                    "interrupted": reason,
                    "remaining_waypoints": len(self.path),
                }
            )
        self.path, self.pending = [], None

    def diagnostics(self):
        return {
            "enabled": True,
            "active": bool(self.path),
            "failure": self.failure,
            "nominal_distance_m": self.config.target_distance_m,
            "effective_distance_m": self.effective_distance_m,
            "nominal_lateral_tolerance_m": self.config.lateral_tolerance_m,
            "effective_lateral_tolerance_m": self.effective_lateral_tolerance_m,
            "remaining_waypoints": len(self.path),
            "plan_count": len(self.plans),
            "last_motion": self.events[-1] if self.events else None,
            "last_plan": self.plans[-1] if self.plans else None,
        }
