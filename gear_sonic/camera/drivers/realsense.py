"""Intel RealSense camera driver.

Requires the ``pyrealsense2`` SDK — install with::

    pip install pyrealsense2

See https://github.com/IntelRealSense/librealsense for hardware-specific instructions.
"""

import time
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

import pyrealsense2 as rs

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


class RealSenseConfig:
    """Configuration for the RealSense camera."""

    depth_image_dim: tuple[int, int] = (640, 480)
    color_image_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    mount_position: str = CameraMountPosition.EGO_VIEW.value
    enable_depth: bool = True


class RealSenseSensor(Sensor, SensorServer):
    """Sensor for Intel RealSense depth cameras."""

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: RealSenseConfig = RealSenseConfig(),
        id: int = 0,
        device_id: str | None = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
        devices = list(rs.context().query_devices())
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found")

        for device in devices:
            print(f"Device: {device.get_info(rs.camera_info.name)}")
            print(f"    Serial number: {device.get_info(rs.camera_info.serial_number)}")
            print(
                f"    Firmware version: {device.get_info(rs.camera_info.firmware_version)}"
            )

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.mount_position = mount_position
        devices = sorted(
            devices, key=lambda x: x.get_info(rs.camera_info.serial_number)
        )
        selected_serial = self._select_device_serial(
            devices=devices, device_id=device_id, id=id
        )
        self.config.enable_device(selected_serial)

        try:
            self.config.enable_stream(
                rs.stream.color,
                config.color_image_dim[0],
                config.color_image_dim[1],
                rs.format.rgb8,
                config.fps,
            )
            if config.enable_depth:
                self.config.enable_stream(
                    rs.stream.depth,
                    config.depth_image_dim[0],
                    config.depth_image_dim[1],
                    rs.format.z16,
                    config.fps,
                )
            self._pipeline_profile = self.pipeline.start(self.config)
            self._depth_aligner = None
            color_profile = self._pipeline_profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()
            intrinsics = color_profile.get_intrinsics()
            self._camera_info = {
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "cx": float(intrinsics.ppx),
                "cy": float(intrinsics.ppy),
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
            }
            if config.enable_depth:
                self._depth_aligner = rs.align(rs.stream.color)
                depth_scale_m = (
                    self._pipeline_profile.get_device()
                    .first_depth_sensor()
                    .get_depth_scale()
                )
                self._camera_info.update(
                    {
                        "depth_scale_m": float(depth_scale_m),
                        "depth_aligned_to": self.mount_position,
                    }
                )
        except Exception as e:
            raise RuntimeError(f"Failed to start RealSense pipeline: {e}")

        self._realsense_config = config
        self._run_as_server = run_as_server
        if self._run_as_server:
            self.start_server(port)
        print(
            f"Done initializing RealSense sensor for {mount_position}: {selected_serial}"
        )

    @staticmethod
    def _select_device_serial(
        devices: list[Any], device_id: str | None, id: int
    ) -> str:
        available_serials = [
            device.get_info(rs.camera_info.serial_number) for device in devices
        ]

        if device_id is not None:
            if device_id in available_serials:
                return device_id
            raise ValueError(
                f"RealSense device with serial '{device_id}' not found. "
                f"Available devices: {available_serials}"
            )

        if id < 0 or id >= len(devices):
            raise IndexError(
                f"RealSense device index {id} out of range for {len(devices)} devices: "
                f"{available_serials}"
            )
        return available_serials[id]

    def read(self) -> dict[str, Any] | None:
        try:
            frames = self.pipeline.wait_for_frames()
        except Exception as e:
            print(f"ERROR! Failed to wait for frames: {e}")
            return None

        if self._realsense_config.enable_depth:
            frames = self._depth_aligner.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = (
            frames.get_depth_frame() if self._realsense_config.enable_depth else None
        )

        if not color_frame:
            print("WARNING! No color frame")
            return None

        try:
            color_image = np.asanyarray(color_frame.get_data())
        except Exception as e:
            print(f"ERROR! Failed to convert color frame to numpy array: {e}")
            return None

        if color_image.size == 0:
            print("WARNING! Empty color image")
            return None

        current_time = time.time()
        timestamps = {self.mount_position: current_time}
        images = {self.mount_position: color_image}

        if self._realsense_config.enable_depth:
            if not depth_frame:
                print("WARNING! No depth frame")
                return None

            try:
                depth_image = np.asanyarray(depth_frame.get_data())
            except Exception as e:
                print(f"ERROR! Failed to convert depth frame to numpy array: {e}")
                return None

            if depth_image.size == 0:
                print("WARNING! Empty depth image")
                return None

            timestamps[f"{self.mount_position}_depth"] = current_time
            images[f"{self.mount_position}_depth"] = depth_image

        return {
            "timestamps": timestamps,
            "images": images,
            "camera_info": {self.mount_position: self._camera_info},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(
            timestamps=data["timestamps"],
            images=data["images"],
            camera_info=data.get("camera_info", {}),
        )
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        spaces = {
            "color_image": gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    self._realsense_config.color_image_dim[1],
                    self._realsense_config.color_image_dim[0],
                    3,
                ),
                dtype=np.uint8,
            )
        }
        if self._realsense_config.enable_depth:
            spaces["depth_image"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    self._realsense_config.depth_image_dim[1],
                    self._realsense_config.depth_image_dim[0],
                    1,
                ),
                dtype=np.uint16,
            )
        return gym.spaces.Dict(spaces)

    def close(self):
        if self._run_as_server:
            self.stop_server()
        self.pipeline.stop()

    def run_server(self):
        if not self._run_as_server:
            raise ValueError("run_as_server must be True to call run_server()")
        while True:
            read_result = self.read()
            if read_result is None:
                continue
            self.send_message({self.mount_position: self.serialize(read_result)})
