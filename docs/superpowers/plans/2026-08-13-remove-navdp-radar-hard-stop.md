# Remove NavDP Radar Hard Stop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the NavDP MID-360 proximity hard stop while preserving radar freshness checks, radar-derived visualization/localization data, and depth-camera stopping.

**Architecture:** Remove the two radar proximity safety helpers from the NavDP control module and remove their use from the planner loop. The planner will still zero commands for stale inputs and depth-camera stops, while ActorRay generation remains available for visualization and recording.

**Tech Stack:** Python 3.10, NumPy, pytest, existing Gear Sonic NavDP modules.

## Global Constraints

- Remove only the MID-360 point-cloud proximity hard stop.
- Preserve `radar_timeout_s` and the `radar_timeout` freshness stop reason.
- Preserve radar ingestion, point processing, ActorRay generation, SLAM/localization, visualization, and recording.
- Preserve depth-camera stopping through `depth_requires_stop`.
- Leave the completed branch unmerged for end-to-end robot testing.

---

### Task 1: Remove the radar proximity hard-stop contract and runtime path

**Files:**
- Modify: `gear_sonic/tests/test_navdp_planner.py`
- Modify: `gear_sonic/navdp/control.py`
- Modify: `gear_sonic/scripts/navdp_planner.py`

**Interfaces:**
- Consumes: `depth_requires_stop(depth: np.ndarray) -> bool`, `actor_ray_from_points(points: np.ndarray) -> np.ndarray`, and `NavDPPlannerConfig.radar_timeout_s`.
- Produces: `_prepare_control_output(velocity, points, latest_depth) -> (velocity, current_rays, camera_stop)`, which preserves radar-derived ActorRay data while allowing only depth data to zero an otherwise valid velocity.

- [x] **Step 1: Write failing control-output behavior tests**

Add these tests near the existing ActorRay and depth-stop tests:

```python
def test_near_radar_point_is_visualized_without_stopping_control_output() -> None:
    velocity, rays, camera_stop = navdp_planner._prepare_control_output(
        (0.3, 0.0, 0.1),
        np.array([[0.09, 0.0, 0.0]], dtype=np.float32),
        np.ones((60, 60), dtype=np.float32),
    )

    assert velocity == pytest.approx((0.3, 0.0, 0.1))
    assert rays[90] == pytest.approx(0.09)
    assert not camera_stop


def test_depth_stop_still_zeros_control_output() -> None:
    depth = np.ones((60, 60), dtype=np.float32)
    depth.flat[:2001] = 0.09

    velocity, _, camera_stop = navdp_planner._prepare_control_output(
        (0.3, 0.0, 0.1),
        np.empty((0, 3), dtype=np.float32),
        depth,
    )

    assert velocity == (0.0, 0.0, 0.0)
    assert camera_stop
```

- [x] **Step 2: Run the new tests and verify they fail because the output boundary does not exist**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q \
  gear_sonic/tests/test_navdp_planner.py::test_near_radar_point_is_visualized_without_stopping_control_output \
  gear_sonic/tests/test_navdp_planner.py::test_depth_stop_still_zeros_control_output
```

Expected: both FAIL with `AttributeError` because `_prepare_control_output` does not exist.

- [x] **Step 3: Add the minimal control-output boundary**

Add this helper to `gear_sonic/scripts/navdp_planner.py`:

```python
def _prepare_control_output(
    velocity: Sequence[float],
    points: np.ndarray,
    latest_depth: np.ndarray | None,
) -> tuple[tuple[float, float, float], np.ndarray, bool]:
    command = tuple(map(float, velocity))
    camera_stop = latest_depth is not None and depth_requires_stop(latest_depth)
    if camera_stop:
        command = (0.0, 0.0, 0.0)
    return command, actor_ray_from_points(points), camera_stop
```

Import `Sequence` from `typing`, then replace the planner loop's depth/ray/hard-safety block with a call to this helper.

- [x] **Step 4: Delete the obsolete helper unit tests and imports**

Remove `apply_hard_safety` and `should_abort_nav_for_lidar` from the test module's direct imports. Delete `test_forward_135_degree_sector_stops_translation_but_not_pure_turning`, `test_hard_safety_rotates_sector_with_xy_translation`, `test_lateral_motion_ignores_obstacle_outside_its_motion_sector`, and `test_only_lidar_hard_stop_aborts_active_navigation` because they specify the behavior being removed.

- [x] **Step 5: Delete the radar hard-stop helpers**

Delete these complete definitions from `gear_sonic/navdp/control.py`:

```python
def apply_hard_safety(...):
    ...


def should_abort_nav_for_lidar(...):
    ...
```

Also remove the now-unused `actor_ray_from_points` import. Keep `math` and `Sequence` because the remaining MPC and zero-action control logic still uses them.

- [x] **Step 6: Remove the remainder of the planner's radar proximity hard-stop path**

In `gear_sonic/scripts/navdp_planner.py`:

- Remove both helpers from the control imports and `__all__`.
- Keep the `radar_timeout_s` freshness block unchanged.
- Use the new control-output boundary:

```python
velocity, current_rays, camera_stop = _prepare_control_output(
    velocity, points, latest_depth
)
```

- Delete construction of `safety_points`, `before_safety`, and `lidar_aborted`.
- Change the navigation-abort branch to trigger only on `zero_action_aborted` and report `navdp_zero_action`.
- Keep `safety_blocked` transition tracking for stale-input and depth-stop diagnostics, but remove its `radar_hard_stop` fallback:

```python
reason = stale_reason or ("depth_hard_stop" if camera_stop else "clear")
```

- [x] **Step 7: Run the new behavior tests and verify they pass**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q \
  gear_sonic/tests/test_navdp_planner.py::test_near_radar_point_is_visualized_without_stopping_control_output \
  gear_sonic/tests/test_navdp_planner.py::test_depth_stop_still_zeros_control_output
```

Expected: both tests pass.

- [x] **Step 8: Run the focused NavDP tests and verify they pass**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q \
  gear_sonic/tests/test_navdp_planner.py
```

Expected: all tests pass.

- [x] **Step 9: Verify preserved radar-timeout and depth-stop contracts**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q \
  gear_sonic/tests/test_runtime_config.py \
  gear_sonic/tests/test_launch_tmux_panes.py \
  gear_sonic/tests/test_navdp_planner.py -k 'radar_timeout or depth_stop or near_radar_point'
```

Expected: all selected tests pass.

- [x] **Step 10: Run the complete Gear Sonic regression suite**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q gear_sonic/tests
```

Expected: all tests pass.

- [x] **Step 11: Commit the implementation**

```bash
git add \
  docs/superpowers/plans/2026-08-13-remove-navdp-radar-hard-stop.md \
  gear_sonic/navdp/control.py \
  gear_sonic/scripts/navdp_planner.py \
  gear_sonic/tests/test_navdp_planner.py
git commit -m "refactor(navdp): remove radar proximity hard stop"
```
