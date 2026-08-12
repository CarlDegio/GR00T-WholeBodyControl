# Livox Front-Sector Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every MID-360 point in the robot-forward closed 90-degree sector before the scan reaches either FAST-LIO or SensorGateway/NavDP.

**Architecture:** Remap the Livox driver's `livox_ros_driver2/CustomMsg` output from `/livox/lidar` to `/livox/lidar_raw`, then run one repository-owned ROS2 Python node that filters each message and republishes it to the existing `/livox/lidar` contract. FAST-LIO, SensorGateway, readiness checks, and debug recording continue to use `/livox/lidar`, so all consumers see the same filtered scan and fail closed if the filter exits.

**Tech Stack:** Python 3.10, ROS2 Humble `rclpy`, `livox_ros_driver2/msg/CustomMsg`, `pytest`, existing SONIC tmux launcher.

## Global Constraints

- Robot-forward is `+X`; robot-left is `+Y` in the Livox message coordinates used by this stack.
- Production removes a point exactly when `x > 0 and abs(y) <= x`, including both 45-degree boundaries at every range and height.
- `/livox/lidar_raw` contains the driver output; `/livox/lidar` contains only filtered messages.
- Preserve `header`, `timebase`, `lidar_id`, `rsvd`, retained-point order, and every retained point field; update only `points` and `point_num`.
- Publish one output for every input, including a valid zero-point message.
- Do not remap or filter `/livox/imu`.
- FAST-LIO YAML, SensorGateway/NavDP subscriptions, runtime topic inventory, and external Livox/FAST-LIO source trees remain unchanged.
- Normal debug bags continue to record filtered `/livox/lidar`; `/livox/lidar_raw` stays available for manual diagnostics but is not recorded by default.
- Preserve all unrelated user changes already present in the dirty working tree; stage and commit only files named by each task.

---

## File Structure

- Create `gear_sonic/scripts/run_livox_front_sector_filter.py`: geometry predicate, point/message filtering, validated CLI settings, and lazy-imported ROS2 node runtime.
- Create `gear_sonic/tests/test_livox_front_sector_filter.py`: pure geometry, metadata preservation, empty-message, and CLI validation tests that run without ROS installed in the active Python environment.
- Modify `gear_sonic/scripts/launch_inference.py`: construct the parameter-equivalent raw Livox driver command, construct the filter command, launch driver → filter → FAST-LIO, log the filter, and clean up its PID.
- Modify `gear_sonic/tests/test_launch_tmux_panes.py`: lock down driver remapping, filter arguments, startup/cleanup participation, filtered debug recording, and unchanged downstream contracts.

### Task 1: Pure Forward-Sector Geometry

**Files:**
- Create: `gear_sonic/scripts/run_livox_front_sector_filter.py`
- Create: `gear_sonic/tests/test_livox_front_sector_filter.py`

**Interfaces:**
- Consumes: point-like objects exposing numeric `.x` and `.y` attributes.
- Produces: `is_inside_forward_sector(point: Any, sector_degrees: float = 90.0) -> bool` and `retain_points_outside_forward_sector(points: Sequence[Any], sector_degrees: float = 90.0) -> list[Any]`.

- [ ] **Step 1: Write the failing literal-fixture tests**

Create `gear_sonic/tests/test_livox_front_sector_filter.py` with tests that cover the exact closed production sector, rear/side points, arbitrary height, ordering, and a non-production sector argument:

```python
from __future__ import annotations

from types import SimpleNamespace

from gear_sonic.scripts.run_livox_front_sector_filter import (
    is_inside_forward_sector,
    retain_points_outside_forward_sector,
)


def point(x: float, y: float, z: float = 0.0, *, label: str = "") -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z, label=label)


def test_production_sector_removes_forward_points_and_closed_boundaries() -> None:
    assert is_inside_forward_sector(point(1.0, 0.0))
    assert is_inside_forward_sector(point(1.0, 1.0))
    assert is_inside_forward_sector(point(1.0, -1.0))
    assert is_inside_forward_sector(point(0.5, 0.25, 100.0))


def test_production_sector_keeps_origin_rear_and_points_outside_boundaries() -> None:
    assert not is_inside_forward_sector(point(0.0, 0.0))
    assert not is_inside_forward_sector(point(-1.0, 0.0))
    assert not is_inside_forward_sector(point(1.0, 1.0001))
    assert not is_inside_forward_sector(point(1.0, -1.0001))


def test_retained_points_keep_input_order_and_identity() -> None:
    left = point(1.0, 2.0, label="left")
    blocked = point(2.0, 0.0, label="blocked")
    rear = point(-1.0, 0.0, label="rear")

    retained = retain_points_outside_forward_sector([left, blocked, rear])

    assert retained == [left, rear]
    assert retained[0] is left
    assert retained[1] is rear


def test_sector_argument_supports_isolated_non_production_checks() -> None:
    candidate = point(1.0, 0.5)

    assert is_inside_forward_sector(candidate, sector_degrees=90.0)
    assert not is_inside_forward_sector(candidate, sector_degrees=30.0)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
./.venv_teleop/bin/python -m pytest \
  gear_sonic/tests/test_livox_front_sector_filter.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'gear_sonic.scripts.run_livox_front_sector_filter'`.

- [ ] **Step 3: Implement the minimal pure geometry helpers**

Create `gear_sonic/scripts/run_livox_front_sector_filter.py` with ROS-independent imports and the following functions. Keep the exact algebraic branch for 90 degrees so the production boundary is precisely `abs(y) <= x`:

```python
#!/usr/bin/env python3
"""Filter the robot-forward sector from Livox CustomMsg point clouds."""

from __future__ import annotations

import math
from typing import Any, Sequence


def _validate_sector_degrees(sector_degrees: float) -> float:
    value = float(sector_degrees)
    if not math.isfinite(value) or not 0.0 < value < 180.0:
        raise ValueError("sector_degrees must be finite and between 0 and 180")
    return value


def is_inside_forward_sector(
    point: Any,
    sector_degrees: float = 90.0,
) -> bool:
    sector = _validate_sector_degrees(sector_degrees)
    x = float(point.x)
    y = float(point.y)
    if x <= 0.0:
        return False
    if sector == 90.0:
        return abs(y) <= x
    return abs(math.degrees(math.atan2(y, x))) <= sector / 2.0


def retain_points_outside_forward_sector(
    points: Sequence[Any],
    sector_degrees: float = 90.0,
) -> list[Any]:
    sector = _validate_sector_degrees(sector_degrees)
    return [
        point
        for point in points
        if not is_inside_forward_sector(point, sector)
    ]
```

- [ ] **Step 4: Run the geometry tests and verify they pass**

Run:

```bash
./.venv_teleop/bin/python -m pytest \
  gear_sonic/tests/test_livox_front_sector_filter.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the pure geometry slice**

```bash
git add \
  gear_sonic/scripts/run_livox_front_sector_filter.py \
  gear_sonic/tests/test_livox_front_sector_filter.py
git commit -m "feat: add Livox front-sector geometry filter"
```

### Task 2: CustomMsg Preservation and ROS2 Filter Node

**Files:**
- Modify: `gear_sonic/scripts/run_livox_front_sector_filter.py`
- Modify: `gear_sonic/tests/test_livox_front_sector_filter.py`

**Interfaces:**
- Consumes: Task 1's `retain_points_outside_forward_sector(...)`; Livox messages exposing `header`, `timebase`, `point_num`, `lidar_id`, `rsvd`, and `points`.
- Produces: `FrontSectorFilterSettings`, `build_argument_parser()`, `resolve_settings(args)`, `filter_custom_message(message, message_factory, sector_degrees=90.0)`, `run_filter(settings)`, and `main()`.

- [ ] **Step 1: Add failing message and CLI tests**

Append these definitions and tests to `gear_sonic/tests/test_livox_front_sector_filter.py`:

```python
import pytest

from gear_sonic.scripts import run_livox_front_sector_filter as front_filter


class FakeCustomMessage:
    def __init__(self) -> None:
        self.header = None
        self.timebase = 0
        self.point_num = 0
        self.lidar_id = 0
        self.rsvd = []
        self.points = []


def custom_point(
    x: float,
    y: float,
    z: float,
    *,
    offset_time: int,
    reflectivity: int,
    tag: int,
    line: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        offset_time=offset_time,
        x=x,
        y=y,
        z=z,
        reflectivity=reflectivity,
        tag=tag,
        line=line,
    )


def test_custom_message_preserves_metadata_and_every_retained_point_field() -> None:
    header = SimpleNamespace(frame_id="livox_frame", stamp=object())
    blocked = custom_point(
        2.0, 0.0, 3.0, offset_time=10, reflectivity=11, tag=12, line=1
    )
    left = custom_point(
        1.0, 2.0, 4.0, offset_time=20, reflectivity=21, tag=22, line=2
    )
    rear = custom_point(
        -1.0, 0.0, 5.0, offset_time=30, reflectivity=31, tag=32, line=3
    )
    message = FakeCustomMessage()
    message.header = header
    message.timebase = 123456
    message.point_num = 3
    message.lidar_id = 7
    message.rsvd = [8, 9, 10]
    message.points = [blocked, left, rear]

    filtered = front_filter.filter_custom_message(message, FakeCustomMessage)

    assert filtered is not message
    assert filtered.header is header
    assert filtered.timebase == 123456
    assert filtered.lidar_id == 7
    assert filtered.rsvd == [8, 9, 10]
    assert filtered.point_num == 2
    assert filtered.points == [left, rear]
    assert filtered.points[0] is left
    assert vars(filtered.points[0]) == vars(left)
    assert vars(filtered.points[1]) == vars(rear)


def test_custom_message_publishes_a_valid_empty_result() -> None:
    message = FakeCustomMessage()
    message.points = [
        custom_point(
            1.0, 0.0, -50.0, offset_time=1, reflectivity=2, tag=3, line=4
        )
    ]
    message.point_num = 1

    filtered = front_filter.filter_custom_message(message, FakeCustomMessage)

    assert filtered.points == []
    assert filtered.point_num == 0


def test_cli_defaults_to_the_shared_raw_and_filtered_topics() -> None:
    settings = front_filter.resolve_settings(
        front_filter.build_argument_parser().parse_args([])
    )

    assert settings.input_topic == "/livox/lidar_raw"
    assert settings.output_topic == "/livox/lidar"
    assert settings.sector_degrees == 90.0


@pytest.mark.parametrize("value", ["0", "180", "nan", "inf"])
def test_cli_rejects_an_invalid_sector(value: str) -> None:
    args = front_filter.build_argument_parser().parse_args(
        ["--sector-degrees", value]
    )

    with pytest.raises(ValueError, match="sector_degrees"):
        front_filter.resolve_settings(args)


def test_cli_rejects_equal_or_relative_topics() -> None:
    parser = front_filter.build_argument_parser()
    with pytest.raises(ValueError, match="different"):
        front_filter.resolve_settings(
            parser.parse_args(
                ["--input-topic", "/livox/lidar", "--output-topic", "/livox/lidar"]
            )
        )
    with pytest.raises(ValueError, match="absolute ROS topic"):
        front_filter.resolve_settings(
            parser.parse_args(["--input-topic", "livox/lidar_raw"])
        )
```

- [ ] **Step 2: Run the new tests and verify the missing interfaces fail**

Run:

```bash
./.venv_teleop/bin/python -m pytest \
  gear_sonic/tests/test_livox_front_sector_filter.py -v
```

Expected: the Task 1 tests pass and the new tests fail because `filter_custom_message`, `resolve_settings`, and `build_argument_parser` are not defined.

- [ ] **Step 3: Implement settings and message filtering without top-level ROS imports**

Add `argparse`, `dataclass`, and `Callable` imports, then add the following code below the Task 1 helpers:

```python
import argparse
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FrontSectorFilterSettings:
    input_topic: str
    output_topic: str
    sector_degrees: float


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-topic", default="/livox/lidar_raw")
    parser.add_argument("--output-topic", default="/livox/lidar")
    parser.add_argument("--sector-degrees", type=float, default=90.0)
    return parser


def resolve_settings(args: argparse.Namespace) -> FrontSectorFilterSettings:
    input_topic = str(args.input_topic)
    output_topic = str(args.output_topic)
    if not input_topic.startswith("/") or not output_topic.startswith("/"):
        raise ValueError("input_topic and output_topic must be absolute ROS topics")
    if input_topic == output_topic:
        raise ValueError("input_topic and output_topic must be different")
    return FrontSectorFilterSettings(
        input_topic=input_topic,
        output_topic=output_topic,
        sector_degrees=_validate_sector_degrees(args.sector_degrees),
    )


def filter_custom_message(
    message: Any,
    message_factory: Callable[[], Any],
    sector_degrees: float = 90.0,
) -> Any:
    retained = retain_points_outside_forward_sector(
        message.points,
        sector_degrees,
    )
    filtered = message_factory()
    filtered.header = message.header
    filtered.timebase = message.timebase
    filtered.lidar_id = message.lidar_id
    filtered.rsvd = message.rsvd
    filtered.points = retained
    filtered.point_num = len(retained)
    return filtered
```

- [ ] **Step 4: Add the lazy-imported ROS2 runtime and entry point**

Append the runtime below `filter_custom_message`. Importing the module in unit tests must not import `rclpy` or `livox_ros_driver2`:

```python
def run_filter(settings: FrontSectorFilterSettings) -> None:
    import rclpy
    from livox_ros_driver2.msg import CustomMsg

    rclpy.init(args=None)
    node = rclpy.create_node("sonic_livox_front_sector_filter")
    publisher = node.create_publisher(CustomMsg, settings.output_topic, 10)

    def on_message(message: Any) -> None:
        publisher.publish(
            filter_custom_message(
                message,
                CustomMsg,
                settings.sector_degrees,
            )
        )

    subscription = node.create_subscription(
        CustomMsg,
        settings.input_topic,
        on_message,
        10,
    )
    node.get_logger().info(
        "Filtering %.1f degrees from %s to %s"
        % (
            settings.sector_degrees,
            settings.input_topic,
            settings.output_topic,
        )
    )
    try:
        rclpy.spin(node)
    finally:
        del subscription
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    settings = resolve_settings(build_argument_parser().parse_args())
    run_filter(settings)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run message/CLI tests and the ROS-independent help command**

Run:

```bash
./.venv_teleop/bin/python -m pytest \
  gear_sonic/tests/test_livox_front_sector_filter.py -v
./.venv_teleop/bin/python \
  gear_sonic/scripts/run_livox_front_sector_filter.py --help
```

Expected: all tests pass; help exits with status 0 and lists `--input-topic`, `--output-topic`, and `--sector-degrees`.

- [ ] **Step 6: Commit the message/node slice**

```bash
git add \
  gear_sonic/scripts/run_livox_front_sector_filter.py \
  gear_sonic/tests/test_livox_front_sector_filter.py
git commit -m "feat: publish filtered Livox CustomMsg scans"
```

### Task 3: Wire the Shared Filter into the Inference Launcher

**Files:**
- Modify: `gear_sonic/scripts/launch_inference.py:561-693`
- Modify: `gear_sonic/tests/test_launch_tmux_panes.py:10-305`

**Interfaces:**
- Consumes: Task 2's script CLI: `--input-topic /livox/lidar_raw --output-topic /livox/lidar --sector-degrees 90`.
- Produces: `build_livox_driver_command(config) -> str`, retained public `build_livox_command(config) -> str`, and `build_livox_filter_command() -> str`; the SensorGateway pane owns `livox_pid`, `livox_filter_pid`, and `fastlio_pid`.

- [ ] **Step 1: Replace launcher expectations with failing shared-chain tests**

Add `build_livox_driver_command` and `build_livox_filter_command` to the import list in `gear_sonic/tests/test_launch_tmux_panes.py`. In `test_navdp_stack_commands_use_ros_topics_and_official_xnavdp_server`, replace the Livox launch-file assertion with these exact contract assertions:

```python
    driver = build_livox_driver_command(config)
    lidar_filter = build_livox_filter_command()

    assert "ros2 run livox_ros_driver2 livox_ros_driver2_node" in livox
    assert "ros2 run livox_ros_driver2 livox_ros_driver2_node" in driver
    assert "-r /livox/lidar:=/livox/lidar_raw" in driver
    assert "-p xfer_format:=1" in driver
    assert "-p multi_topic:=0" in driver
    assert "-p data_src:=0" in driver
    assert "-p publish_freq:=10.0" in driver
    assert "-p output_data_type:=0" in driver
    assert "-p frame_id:=livox_frame" in driver
    assert "MID360_config.json" in driver
    assert "run_livox_front_sector_filter.py" in lidar_filter
    assert "--input-topic /livox/lidar_raw" in lidar_filter
    assert "--output-topic /livox/lidar" in lidar_filter
    assert "--sector-degrees 90" in lidar_filter
```

Update `test_runtime_sidecars_are_read_only_and_navdp_uses_gateway_by_default` and add a focused cleanup test:

```python
def test_livox_filter_is_between_driver_and_fastlio_and_participates_in_cleanup() -> None:
    config = InferenceLaunchConfig(
        opencv_viewer=False,
        planner_input="keyboard",
        slam_debug=False,
    )

    gateway = build_sensor_gateway_command(config, Path("/workspace/sonic"))

    driver_index = gateway.index("ros2 run livox_ros_driver2")
    filter_index = gateway.index("run_livox_front_sector_filter.py")
    fastlio_index = gateway.index("run_fastlio_supervisor.py")
    assert driver_index < filter_index < fastlio_index
    assert "/tmp/sonic_livox_driver.log" in gateway
    assert "/tmp/sonic_livox_filter.log" in gateway
    assert "/tmp/sonic_fastlio.log" in gateway
    assert "livox_pid=$!" in gateway
    assert "livox_filter_pid=$!" in gateway
    assert "fastlio_pid=$!" in gateway
    assert "kill $livox_pid $livox_filter_pid $fastlio_pid" in gateway
    assert "wait $livox_pid $livox_filter_pid $fastlio_pid" in gateway
```

Extend the SLAM debug test to prove the bag remains filtered and the filter has its own log:

```python
    assert "/livox/lidar" in recorder
    assert "/livox/lidar_raw" not in recorder
    assert '>"$slam_debug_dir/livox_filter.log"' in gateway
```

- [ ] **Step 2: Run launcher tests and verify the new command builders are missing**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_launch_tmux_panes.py -v
```

Expected: collection or tests fail because `build_livox_driver_command` and `build_livox_filter_command` do not exist and the current driver still uses `msg_MID360_launch.py`.

- [ ] **Step 3: Build a raw-output driver command equivalent to the existing launch file**

Add these builders near the existing `build_livox_command`. The installed config path deliberately resolves through `fastlio_workspace`, so no external repository file is changed:

```python
def build_livox_driver_command(config: InferenceLaunchConfig) -> str:
    user_config_path = (
        Path(config.fastlio_workspace)
        / "install"
        / "livox_ros_driver2"
        / "share"
        / "livox_ros_driver2"
        / "config"
        / "MID360_config.json"
    )
    return (
        "ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args "
        "-r __node:=livox_lidar_publisher "
        "-r /livox/lidar:=/livox/lidar_raw "
        "-p xfer_format:=1 "
        "-p multi_topic:=0 "
        "-p data_src:=0 "
        "-p publish_freq:=10.0 "
        "-p output_data_type:=0 "
        "-p frame_id:=livox_frame "
        "-p lvx_file_path:=/home/livox/livox_test.lvx "
        f"-p user_config_path:={shlex.quote(str(user_config_path))} "
        "-p cmdline_input_bd_code:=livox0000000001"
    )


def build_livox_command(config: InferenceLaunchConfig) -> str:
    return (
        "unset COLCON_CURRENT_PREFIX AMENT_PREFIX_PATH CMAKE_PREFIX_PATH; "
        "source /opt/ros/humble/setup.bash && "
        f"export LD_LIBRARY_PATH={shlex.quote(config.livox_sdk_lib)}:$LD_LIBRARY_PATH && "
        f"source {shlex.quote(config.fastlio_workspace)}/install/setup.bash && "
        f"{build_livox_driver_command(config)}"
    )


def build_livox_filter_command() -> str:
    return (
        "python gear_sonic/scripts/run_livox_front_sector_filter.py "
        "--input-topic /livox/lidar_raw "
        "--output-topic /livox/lidar "
        "--sector-degrees 90"
    )
```

- [ ] **Step 4: Insert driver → filter → FAST-LIO into the background process list**

In `build_sensor_gateway_command`, add the filter log alongside the existing Livox and FAST-LIO logs:

```python
    livox_log = "/tmp/sonic_livox_driver.log"
    livox_filter_log = "/tmp/sonic_livox_filter.log"
    fastlio_log = "/tmp/sonic_fastlio.log"
```

When SLAM debugging is enabled, redirect it into the same timestamped directory:

```python
        livox_log = '"$slam_debug_dir/livox_driver.log"'
        livox_filter_log = '"$slam_debug_dir/livox_filter.log"'
        fastlio_log = '"$slam_debug_dir/fastlio.log"'
```

Replace the ROS-stack population of `background_commands` with an ordered sequence. The bag recorder is started only after the filter process exists, and FAST-LIO starts after both:

```python
    if ros_stack_enabled:
        background_commands.extend(
            (
                (
                    "livox_pid",
                    build_livox_driver_command(config),
                    livox_log,
                ),
                (
                    "livox_filter_pid",
                    build_livox_filter_command(),
                    livox_filter_log,
                ),
            )
        )
        if slam_debug_enabled:
            background_commands.append(
                (
                    "slam_debug_pid",
                    build_slam_debug_command(config),
                    '"$slam_debug_dir/rosbag.log"',
                )
            )
        background_commands.append(
            (
                "fastlio_pid",
                build_fastlio_supervisor_command(config),
                fastlio_log,
            )
        )
```

The existing `pid_names` construction then automatically adds `livox_filter_pid` to both `kill` and `wait`. Leave `build_slam_debug_command`'s topic roles unchanged, but change its docstring from “raw inputs” to “filtered LiDAR/IMU inputs” so the documentation matches the new chain.

- [ ] **Step 5: Run launcher and filter test suites**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_launch_tmux_panes.py \
  gear_sonic/tests/test_livox_front_sector_filter.py -v
```

Expected: all tests pass. The generated command contains one raw driver publisher, one filter process, and unchanged `/livox/lidar` downstream consumers.

- [ ] **Step 6: Commit the launcher integration**

```bash
git add \
  gear_sonic/scripts/launch_inference.py \
  gear_sonic/tests/test_launch_tmux_panes.py
git commit -m "feat: filter Livox scans before FAST-LIO"
```

### Task 4: Regression and On-Robot Data-Path Verification

**Files:**
- Verify only: `gear_sonic/scripts/run_livox_front_sector_filter.py`
- Verify only: `gear_sonic/scripts/launch_inference.py`
- Verify only: `gear_sonic/tests/test_livox_front_sector_filter.py`
- Verify only: `gear_sonic/tests/test_launch_tmux_panes.py`

**Interfaces:**
- Consumes: production topics `/livox/lidar_raw`, `/livox/lidar`, `/livox/imu`, `/Odometry_loc`, `/cloud_registered_1`; SensorGateway RPC `tcp://127.0.0.1:5560`.
- Produces: evidence that filtered messages contain no point satisfying `x > 0 and abs(y) <= x`, FAST-LIO remains live, and SensorGateway observes the filtered stream.

- [ ] **Step 1: Run static and focused regression checks**

Run:

```bash
./.venv_inference/bin/python -m compileall -q \
  gear_sonic/scripts/run_livox_front_sector_filter.py \
  gear_sonic/scripts/launch_inference.py
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_livox_front_sector_filter.py \
  gear_sonic/tests/test_launch_tmux_panes.py \
  gear_sonic/tests/test_navdp_readiness_gate.py \
  gear_sonic/tests/test_sensor_gateway.py \
  gear_sonic/tests/test_runtime_config.py \
  gear_sonic/tests/test_runtime_contracts.py -v
git diff --check
```

Expected: compilation succeeds, all selected tests pass, and `git diff --check` prints nothing.

- [ ] **Step 2: Confirm the patch did not modify external stacks or runtime topic contracts**

Run:

```bash
git diff --name-only HEAD~3..HEAD
rg -n "lidar: /livox/lidar$|lid_topic:.*livox/lidar" \
  gear_sonic/config/launch_inference.yaml \
  /home/user/Project/FAST_LIO_LOCALIZATION_HUMANOID/FAST_LIO/config/mid360.yaml
```

Expected: the commit range names only the two new filter files, `launch_inference.py`, and `test_launch_tmux_panes.py`; both existing consumer configurations still point at `/livox/lidar`.

- [ ] **Step 3: Start the normal real-robot launcher with the robot stationary**

Before launch, place the robot in a supported stationary posture and keep the physical stop available. Start the same inference command normally used for deployment; do not start a second Livox driver alongside it. In another shell, source the ROS stack:

```bash
source /opt/ros/humble/setup.bash
source /home/user/Project/fastlio_humanoid_ws/install/setup.bash
```

Expected launcher logs:

```text
/tmp/sonic_livox_driver.log
/tmp/sonic_livox_filter.log
/tmp/sonic_fastlio.log
```

The filter log must contain a line equivalent to:

```text
Filtering 90.0 degrees from /livox/lidar_raw to /livox/lidar
```

- [ ] **Step 4: Verify ROS2 publishers and stream rates**

Run:

```bash
ros2 topic info /livox/lidar_raw -v
ros2 topic info /livox/lidar -v
timeout 8 ros2 topic hz /livox/lidar_raw
timeout 8 ros2 topic hz /livox/lidar
timeout 8 ros2 topic hz /livox/imu
```

Expected: `/livox/lidar_raw` has the Livox driver publisher; `/livox/lidar` has `sonic_livox_front_sector_filter` as its publisher; both point topics are approximately 10 Hz; IMU remains live and is still published directly by the driver.

- [ ] **Step 5: Prove a live filtered message has zero blocked points**

From the repository root, run this one-shot subscriber after sourcing ROS2 and the FAST-LIO workspace:

```bash
./.venv_teleop/bin/python -c 'exec("""\
import rclpy\n\
from livox_ros_driver2.msg import CustomMsg\n\
rclpy.init(args=None)\n\
node = rclpy.create_node(\"verify_livox_front_filter\")\n\
received = []\n\
subscription = node.create_subscription(CustomMsg, \"/livox/lidar\", received.append, 10)\n\
deadline_ns = node.get_clock().now().nanoseconds + 5_000_000_000\n\
while not received and node.get_clock().now().nanoseconds < deadline_ns:\n\
    rclpy.spin_once(node, timeout_sec=0.1)\n\
assert received, \"timed out waiting for /livox/lidar\"\n\
message = received[0]\n\
blocked = [point for point in message.points if point.x > 0.0 and abs(point.y) <= point.x]\n\
print({\"point_num\": message.point_num, \"len_points\": len(message.points), \"blocked\": len(blocked)})\n\
assert message.point_num == len(message.points)\n\
assert blocked == []\n\
node.destroy_subscription(subscription)\n\
node.destroy_node()\n\
rclpy.shutdown()\n\
""")'
```

Expected: the printed dictionary has equal `point_num` and `len_points`, and `blocked` is `0`.

- [ ] **Step 6: Verify FAST-LIO and SensorGateway consume the filtered chain**

Run:

```bash
timeout 8 ros2 topic hz /Odometry_loc
timeout 8 ros2 topic hz /cloud_registered_1
./.venv_teleop/bin/python -c 'from gear_sonic.runtime.client import SensorGatewayClient; c=SensorGatewayClient("tcp://127.0.0.1:5560", request_timeout_ms=1000); h=c.health(); print(h["streams"]["source/ros_lidar"]); c.close()'
```

Expected: FAST-LIO odometry and registered cloud both remain live; SensorGateway reports `source/ros_lidar` with increasing `message_count` and state `healthy` after its normal warmup.

- [ ] **Step 7: Verify fail-closed behavior and restore normal operation**

Resolve and terminate only the exact filter process owned by the current launcher, then check the filtered topic:

```bash
filter_pid="$(pgrep -f '^python gear_sonic/scripts/run_livox_front_sector_filter.py --input-topic /livox/lidar_raw --output-topic /livox/lidar --sector-degrees 90$')"
test -n "$filter_pid"
kill -TERM "$filter_pid"
timeout 5 ros2 topic hz /livox/lidar
```

Expected: after the filter exits, `/livox/lidar` produces no new samples even though `/livox/lidar_raw` remains live; raw points never reappear on the consumer topic. Stop the current launcher normally so its cleanup terminates the driver and FAST-LIO, then restart the normal launcher and repeat Steps 4–6 to restore and confirm the healthy chain.

- [ ] **Step 8: Record final verification evidence without another code commit**

Run:

```bash
git status --short
git log -3 --oneline
```

Expected: only pre-existing unrelated user changes remain in `git status`; the three implementation commits are visible and no runtime verification artifact was added to the repository.
