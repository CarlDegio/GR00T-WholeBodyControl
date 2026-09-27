"""Confirm a specific mode command using fresh C++ telemetry before navigation."""

from dataclasses import dataclass
import math


@dataclass
class PlannerHandoff:
    request_id: int
    deadline: float
    ready_since: float | None = None
    result: str | None = None
    last_publish: float = -math.inf

    def update(self, state_msg, now):
        if self.result is not None:
            return self.result
        status = (state_msg or {}).get("control_status", [])
        matches = (len(status) == 3 and status[0] == 1 and status[1] == 1
                   and status[2] == self.request_id and self.request_id > 0)
        if matches:
            if self.ready_since is None:
                self.ready_since = now
            # C++ ramps the upper body over two seconds. Hold navigation through it.
            if now - self.ready_since >= 2.0:
                self.result = "ready"
        else:
            self.ready_since = None
        if self.result is None and now >= self.deadline:
            self.result = "failed"
        return self.result
