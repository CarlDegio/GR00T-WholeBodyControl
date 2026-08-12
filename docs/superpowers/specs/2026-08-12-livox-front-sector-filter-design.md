# Livox Front-Sector Filter Design

## Goal

Remove every MID-360 point in the robot's forward 90-degree horizontal sector
before the point cloud reaches either FAST-LIO or NavDP. This masks a physical
obstruction in front of the LiDAR while keeping localization, planning, hard
safety, visualization, and debug recordings consistent with one another.

## Coordinate and Mask Definition

The Livox point coordinates used by the current stack have robot-forward on
`+X` and robot-left on `+Y`. The MID-360 configuration applies a 180-degree
roll, which does not change the horizontal `+X` forward direction.

For every point `(x, y, z)`, remove the point when:

```text
x > 0 and abs(y) <= x
```

This is the closed horizontal sector from -45 degrees through +45 degrees.
The mask applies at every range and every height. Points exactly on either
45-degree boundary are removed. The origin and points with `x <= 0` remain
unless another existing consumer filter removes them later.

## Shared ROS2 Data Path

The Livox driver continues to produce `livox_ros_driver2/CustomMsg`, but its
point-cloud publication is remapped from `/livox/lidar` to
`/livox/lidar_raw`. A new ROS2 filter node subscribes to
`/livox/lidar_raw`, removes the forward sector, updates `point_num`, and
publishes the filtered `CustomMsg` to the original `/livox/lidar` topic.

```text
Livox driver -> /livox/lidar_raw -> front-sector filter -> /livox/lidar
                                                           |-> FAST-LIO
                                                           |-> SensorGateway/NavDP
```

FAST-LIO, SensorGateway, NavDP, readiness checks, and downstream topic names
remain unchanged. `/livox/imu` is not remapped or filtered. FAST-LIO's
registered cloud is consequently built only from filtered scans.

## Message Preservation

The filter preserves the input message's `header`, `timebase`, `lidar_id`,
reserved bytes, and the original ordering of retained points. Each retained
point keeps `offset_time`, XYZ coordinates, reflectivity, tag, and line. Only
the `points` sequence and its matching `point_num` are changed.

The filter republishes one output message for every input message, including
an empty output when every input point lies inside the masked sector. It does
not synthesize points, rotate coordinates, change timestamps, or apply range,
height, ground, or statistical filtering.

## Launch and Failure Behavior

The inference launcher starts the processes in this order:

1. Livox driver publishing `/livox/lidar_raw` and `/livox/imu`.
2. Front-sector filter publishing `/livox/lidar`.
3. FAST-LIO subscribing to `/livox/lidar` and `/livox/imu`.
4. SensorGateway subscribing to the same filtered `/livox/lidar`.

The driver is started with the same MID-360 parameters currently contained in
`msg_MID360_launch.py`, supplied through `ros2 run`, plus an explicit ROS2
topic remap. The external Livox and FAST-LIO source trees are not modified or
rebuilt.

If the filter is absent or exits, no publisher silently restores raw points
on `/livox/lidar`. Existing readiness and health checks therefore report a
missing/stale LiDAR stream and prevent or stop normal navigation rather than
running with unfiltered data. Parent-process cleanup terminates the driver,
filter, and FAST-LIO together with SensorGateway.

SLAM debug recording continues to record `/livox/lidar`, so its normal point
cloud is the exact filtered input seen by FAST-LIO. `/livox/lidar_raw` remains
available for targeted manual diagnostics but is not added to the default bag,
avoiding a substantial duplicate recording.

## Components

- `gear_sonic/scripts/run_livox_front_sector_filter.py`: pure mask helpers,
  ROS2 node, CLI topic/sector configuration, and clean shutdown.
- `gear_sonic/scripts/launch_inference.py`: constructs the remapped Livox
  driver command, starts the filter process, and includes it in cleanup.
- Existing FAST-LIO YAML, SensorGateway subscriptions, NavDP filtering, and
  runtime endpoint definitions retain `/livox/lidar`.

The runtime command passes a 90-degree sector explicitly. The helper accepts a
sector argument for isolated testing, but production uses exactly 90 degrees.

## Testing and Validation

Unit tests use literal point fixtures to prove that forward, diagonal-boundary,
rear, side, all-height, and empty-output cases follow the mask definition.
Message-level tests verify retained ordering, all metadata fields, per-point
fields, and corrected `point_num`.

Launcher tests prove that:

- the driver publishes `/livox/lidar_raw` through an explicit remap;
- the filter reads raw and publishes `/livox/lidar` with a 90-degree sector;
- FAST-LIO and SensorGateway still consume `/livox/lidar`;
- the filter PID participates in parent cleanup; and
- debug recording continues to capture the filtered topic.

Runtime validation compares simultaneous samples from `/livox/lidar_raw` and
`/livox/lidar`: raw data may contain points satisfying
`x > 0 and abs(y) <= x`, while filtered data must contain zero such points.
It then confirms FAST-LIO odometry and registered-cloud topics remain live and
SensorGateway receives the filtered stream at the expected rate.
