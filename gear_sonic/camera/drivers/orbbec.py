"""Orbbec RGB-D camera driver using the official Orbbec SDK v2.

Install the Python package and Linux USB permissions on the robot with::

    pip install --upgrade pyorbbecsdk2
    python $(python -c "import pyorbbecsdk, os; print(os.path.dirname(pyorbbecsdk.__file__))")/shared/setup_env.py

The package is named ``pyorbbecsdk2`` on PyPI and imported as
``pyorbbecsdk`` in Python.
"""

import time
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

import pyorbbecsdk as ob

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


class OrbbecConfig:
    """Configuration for an Orbbec RGB-D camera."""

    depth_image_dim: tuple[int, int] = (640, 480)
    color_image_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    mount_position: str = CameraMountPosition.EGO_VIEW.value
    enable_depth: bool = True
    frame_timeout_ms: int = 1000


class OrbbecSensor(Sensor, SensorServer):
    """Sensor for Orbbec cameras with a RealSense-compatible payload."""

    _ORBBEC_VENDOR_ID = 0x2BC5
    _GEMINI_345LG_PRODUCT_ID = 0x0813

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: OrbbecConfig = OrbbecConfig(),
        id: int = 0,
        device_id: str | None = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
        self._context = ob.Context()
        device_list = self._context.query_devices()
        device_count = device_list.get_count()
        if device_count == 0:
            raise RuntimeError("No Orbbec devices found")

        available_serials = []
        opened_devices = {}
        for index in range(device_count):
            serial = device_list.get_device_serial_number_by_index(index)
            if not serial:
                opened_device = device_list.get_device_by_index(index)
                serial = opened_device.get_device_info().get_serial_number()
                opened_devices[serial] = opened_device
            available_serials.append(serial)
        available_serials.sort()
        selected_serial = self._select_device_serial(
            available_serials=available_serials,
            device_id=device_id,
            id=id,
        )
        device = opened_devices.get(selected_serial)
        if device is None:
            device = device_list.get_device_by_serial_number(selected_serial)
        device_info = device.get_device_info()
        self.serial_number = device_info.get_serial_number()
        self.mount_position = mount_position
        self._orbbec_config = config
        self._run_as_server = run_as_server
        self._closed = False

        print(f"Device: {device_info.get_name()}")
        print(f"    Serial number: {self.serial_number}")
        print(f"    Firmware version: {device_info.get_firmware_version()}")

        self.pipeline = ob.Pipeline(device)
        self.config = ob.Config()
        self._align_filter = None
        self._color_undistortion_filter = None
        pipeline_started = False

        try:
            color_profiles = self.pipeline.get_stream_profile_list(
                ob.OBSensorType.COLOR_SENSOR
            )
            self._color_profile = color_profiles.get_video_stream_profile(
                config.color_image_dim[0],
                config.color_image_dim[1],
                ob.OBFormat.RGB,
                config.fps,
            )
            self.config.enable_stream(self._color_profile)

            if config.enable_depth:
                depth_profiles = self.pipeline.get_stream_profile_list(
                    ob.OBSensorType.DEPTH_SENSOR
                )
                self._depth_profile = depth_profiles.get_video_stream_profile(
                    config.depth_image_dim[0],
                    config.depth_image_dim[1],
                    ob.OBFormat.Y16,
                    config.fps,
                )
                self.config.enable_stream(self._depth_profile)
                self.config.set_frame_aggregate_output_mode(
                    ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
                )

            self.pipeline.start(self.config)
            pipeline_started = True

            intrinsics = self._color_profile.get_intrinsic()
            self._camera_info = {
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "cx": float(intrinsics.cx),
                "cy": float(intrinsics.cy),
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
                "depth_aligned_to": self.mount_position,
            }

            if config.enable_depth:
                self._align_filter = ob.AlignFilter(
                    align_to_stream=ob.OBStreamType.COLOR_STREAM
                )
                self._color_undistortion_filter = self._make_undistortion_filter(
                    device_info
                )
        except Exception as exc:
            if pipeline_started:
                self.pipeline.stop()
            raise RuntimeError(f"Failed to start Orbbec pipeline: {exc}") from exc

        if self._run_as_server:
            self.start_server(port)
        print(
            f"Done initializing Orbbec sensor for {mount_position}: "
            f"{self.serial_number}"
        )

    @staticmethod
    def _select_device_serial(
        available_serials: list[str], device_id: str | None, id: int
    ) -> str:
        if device_id is not None:
            if device_id in available_serials:
                return device_id
            raise ValueError(
                f"Orbbec device with serial '{device_id}' not found. "
                f"Available devices: {available_serials}"
            )

        if id < 0 or id >= len(available_serials):
            raise IndexError(
                f"Orbbec device index {id} out of range for "
                f"{len(available_serials)} devices: {available_serials}"
            )
        return available_serials[id]

    def _make_undistortion_filter(self, device_info: Any) -> Any | None:
        """Use the SDK's 345Lg color correction when the installed SDK has it."""
        if (
            device_info.get_vid() != self._ORBBEC_VENDOR_ID
            or device_info.get_pid() != self._GEMINI_345LG_PRODUCT_ID
        ):
            return None

        filter_type = getattr(ob, "UnDistortionFilter", None)
        if filter_type is None:
            return None
        return filter_type(ob.OBStreamType.COLOR_STREAM)

    @staticmethod
    def _as_frame_set(frame: Any) -> Any:
        as_frame_set = getattr(frame, "as_frame_set", None)
        return as_frame_set() if as_frame_set is not None else frame

    @staticmethod
    def _color_frame_to_numpy(color_frame: Any) -> np.ndarray:
        if color_frame.get_format() != ob.OBFormat.RGB:
            raise ValueError(f"Unsupported Orbbec color format: {color_frame.get_format()}")
        height = color_frame.get_height()
        width = color_frame.get_width()
        color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        expected_size = height * width * 3
        if color_data.size != expected_size:
            raise ValueError(
                f"Invalid Orbbec color buffer size {color_data.size}; "
                f"expected {expected_size}"
            )
        return color_data.reshape((height, width, 3)).copy()

    @staticmethod
    def _depth_frame_to_numpy(depth_frame: Any) -> np.ndarray:
        height = depth_frame.get_height()
        width = depth_frame.get_width()
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        expected_size = height * width
        if depth_data.size != expected_size:
            raise ValueError(
                f"Invalid Orbbec depth buffer size {depth_data.size}; "
                f"expected {expected_size}"
            )
        return depth_data.reshape((height, width)).copy()

    def read(self) -> dict[str, Any] | None:
        try:
            frames = self.pipeline.wait_for_frames(
                self._orbbec_config.frame_timeout_ms
            )
        except Exception as exc:
            print(f"ERROR! Failed to wait for Orbbec frames: {exc}")
            return None

        if not frames:
            print("WARNING! Timed out waiting for Orbbec frames")
            return None

        if self._orbbec_config.enable_depth:
            try:
                if self._color_undistortion_filter is not None:
                    frames = self._color_undistortion_filter.process(frames)
                    if not frames:
                        print("WARNING! Orbbec color undistortion returned no frames")
                        return None
                    frames = self._as_frame_set(frames)

                frames = self._align_filter.process(frames)
                if not frames:
                    print("WARNING! Orbbec depth-to-color alignment returned no frames")
                    return None
                frames = self._as_frame_set(frames)
            except Exception as exc:
                print(f"ERROR! Failed to align Orbbec depth to color: {exc}")
                return None

        color_frame = frames.get_color_frame()
        depth_frame = (
            frames.get_depth_frame() if self._orbbec_config.enable_depth else None
        )
        if not color_frame:
            print("WARNING! No Orbbec color frame")
            return None
        if self._orbbec_config.enable_depth and not depth_frame:
            print("WARNING! No Orbbec depth frame")
            return None

        try:
            color_image = self._color_frame_to_numpy(color_frame)
            depth_image = (
                self._depth_frame_to_numpy(depth_frame)
                if self._orbbec_config.enable_depth
                else None
            )
        except (TypeError, ValueError) as exc:
            print(f"ERROR! Failed to convert Orbbec frame to numpy array: {exc}")
            return None

        if color_image.size == 0:
            print("WARNING! Empty Orbbec color image")
            return None

        current_time = time.time()
        timestamps = {self.mount_position: current_time}
        images = {self.mount_position: color_image}
        camera_info: dict[str, dict[str, Any]] = {}

        if self._orbbec_config.enable_depth:
            if depth_image is None or depth_image.size == 0:
                print("WARNING! Empty Orbbec depth image")
                return None
            if depth_image.shape != color_image.shape[:2]:
                print(
                    "ERROR! Orbbec aligned depth shape "
                    f"{depth_image.shape} does not match color shape "
                    f"{color_image.shape[:2]}"
                )
                return None

            timestamps[f"{self.mount_position}_depth"] = current_time
            images[f"{self.mount_position}_depth"] = depth_image
            calibration = dict(self._camera_info)
            calibration["depth_scale_m"] = float(
                depth_frame.get_depth_scale()
            ) * 0.001
            camera_info[self.mount_position] = calibration

        return {
            "timestamps": timestamps,
            "images": images,
            "camera_info": camera_info,
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
                    self._orbbec_config.color_image_dim[1],
                    self._orbbec_config.color_image_dim[0],
                    3,
                ),
                dtype=np.uint8,
            )
        }
        if self._orbbec_config.enable_depth:
            spaces["depth_image"] = gym.spaces.Box(
                low=0,
                high=np.iinfo(np.uint16).max,
                shape=(
                    self._orbbec_config.depth_image_dim[1],
                    self._orbbec_config.depth_image_dim[0],
                    1,
                ),
                dtype=np.uint16,
            )
        return gym.spaces.Dict(spaces)

    def close(self):
        if self._closed:
            return
        if self._run_as_server:
            self.stop_server()
        self.pipeline.stop()
        self._closed = True

    def run_server(self):
        if not self._run_as_server:
            raise ValueError("run_as_server must be True to call run_server()")
        while True:
            read_result = self.read()
            if read_result is None:
                continue
            self.send_message({self.mount_position: self.serialize(read_result)})
