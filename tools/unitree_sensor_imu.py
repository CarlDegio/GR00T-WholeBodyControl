"""Local CycloneDDS Python IDL for sensor_msgs/msg/Imu.

The vendored unitree_sdk2_python tree in this repository includes PointCloud2
but not Imu. This mirrors the generated style used by that package so the
viewer can subscribe to rt/utlidar/imu_livox_mid360 without ROS2.
"""

from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Quaternion_, Vector3_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Imu_(idl.IdlStruct, typename="sensor_msgs.msg.dds_.Imu_"):
    header: Header_
    orientation: Quaternion_
    orientation_covariance: types.array[types.float64, 9]
    angular_velocity: Vector3_
    angular_velocity_covariance: types.array[types.float64, 9]
    linear_acceleration: Vector3_
    linear_acceleration_covariance: types.array[types.float64, 9]
