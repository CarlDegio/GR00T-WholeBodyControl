import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "mid360_reasan_open3d.py"
SPEC = importlib.util.spec_from_file_location("mid360_reasan_open3d", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_body_envelope_radius_uses_metric_panel_scale():
    assert MODULE.body_envelope_radius_px(0.3, 3.0, 205) == 20
    assert MODULE.body_envelope_radius_px(0.3, 1.5, 205) == 41


def test_mid360_viewer_has_no_open3d_display_options():
    options = MODULE.parser()._option_string_actions

    assert "--show-rays-3d" not in options
    assert "--point-size" not in options
    assert "--ray-line-width" not in options
    assert "--width" not in options
    assert "--height" not in options
    assert "--no-cv" not in options


def test_direct_actor_ray_uses_points_from_all_vertical_angles():
    points = MODULE.np.array(
        [
            [1.0, 0.0, 1.0],
            [0.6, 0.0, -0.6],
            [-2.0, 0.0, 0.0],
        ],
        dtype=MODULE.np.float32,
    )

    rays = MODULE.direct_actor_profile_from_points(points, max_range=3.0)

    assert rays.shape == (180,)
    assert rays[90] == pytest.approx((0.6**2 + 0.6**2) ** 0.5 / 3.0)
    assert min(rays[0], rays[179]) == pytest.approx(2.0 / 3.0)


def test_mid360_viewer_exposes_only_unified_direct_pipeline_options():
    options = MODULE.parser()._option_string_actions

    assert "--ray-source" not in options
    assert "--estimator" not in options
    assert "--imu-topic" not in options
    assert "--theta-range" not in options
    assert "--theta-res-deg" not in options
