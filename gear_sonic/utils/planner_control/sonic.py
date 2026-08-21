"""SONIC planner binary-message conversion owned by the common executor."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message


@dataclass
class SonicPlannerState:
    heading: float = 0.0

    def directional_message(
        self,
        *,
        speed: float,
        movement_heading: float,
        facing_heading: float,
    ) -> bytes:
        self.heading = math.remainder(float(facing_heading), 2.0 * math.pi)
        movement = (
            math.cos(float(movement_heading)),
            math.sin(float(movement_heading)),
            0.0,
        )
        facing = (math.cos(self.heading), math.sin(self.heading), 0.0)
        return build_planner_message(
            1,
            movement if speed > 1.0e-6 else (0.0, 0.0, 0.0),
            facing,
            speed=max(0.0, float(speed)),
            height=-1.0,
        )

    def message(self, velocity: Sequence[float], dt: float = 0.0) -> bytes:
        vx, vy, wz = map(float, velocity)
        if dt:
            self.heading = math.remainder(
                self.heading + wz * float(dt), 2.0 * math.pi
            )
        cosine, sine = math.cos(self.heading), math.sin(self.heading)
        world_x = cosine * vx - sine * vy
        world_y = sine * vx + cosine * vy
        speed = math.hypot(world_x, world_y)
        movement = (
            (0.0, 0.0, 0.0)
            if speed < 1.0e-6
            else (world_x / speed, world_y / speed, 0.0)
        )
        facing = (cosine, sine, 0.0)
        return build_planner_message(1, movement, facing, speed=speed, height=-1.0)
