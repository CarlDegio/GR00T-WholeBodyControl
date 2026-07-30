import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from reasan_planner import TtcPotentialField  # noqa: E402


def _rays(front_distance_m: float = 3.0) -> np.ndarray:
    rays = np.full(180, 3.0, dtype=np.float32)
    rays[90] = front_distance_m
    return rays


def test_planner_blends_ttc_and_distance_risks():
    field = TtcPotentialField(control_dt=0.02)

    result = field.step(np.array([0.5, 0.0, 0.0], dtype=np.float32), _rays(1.05))

    ttc_risk = 0.5**4
    distance_risk = (1.0 - 0.75 / 1.2) ** 4
    expected_danger = 0.7 * ttc_risk + 0.3 * distance_risk
    assert result.danger == pytest.approx(expected_danger, abs=1e-6)
    assert result.velocity[0] == pytest.approx(0.5 - 1.125 * expected_danger, abs=1e-6)


def test_planner_distance_risk_acts_at_low_nonzero_speed():
    field = TtcPotentialField(control_dt=0.02)

    result = field.step(np.array([0.01, 0.0, 0.0], dtype=np.float32), _rays(0.6))

    expected = 0.3 * (1.0 - 0.3 / 1.2) ** 4
    assert result.danger == pytest.approx(expected, abs=1e-6)
    assert result.velocity[0] == 0.0


def test_planner_zero_command_stays_zero_near_obstacle():
    field = TtcPotentialField(control_dt=0.02)

    result = field.step(np.zeros(3, dtype=np.float32), _rays(0.31))

    np.testing.assert_array_equal(result.velocity, np.zeros(3, dtype=np.float32))
    assert result.danger == 0.0


def test_planner_accepts_normalized_actor_rays_in_message_order():
    field = TtcPotentialField(control_dt=0.02)
    normalized = _rays(1.05) / 3.0

    result = field.step_normalized(np.array([0.5, 0.0, 0.0], dtype=np.float32), normalized)

    direct = TtcPotentialField(control_dt=0.02).step(
        np.array([0.5, 0.0, 0.0], dtype=np.float32), _rays(1.05)
    )
    np.testing.assert_allclose(result.velocity, direct.velocity, atol=1e-6)
    assert result.danger == pytest.approx(direct.danger, abs=1e-6)
