"""ZMQ PUB/SUB transport and image serialisation for the camera server."""

import base64
from concurrent.futures import Executor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import zmq

from gear_sonic.camera.constants import PRODUCTION_JPEG_QUALITY

CAMERA_SEND_HWM = 1


# =============================================================================
# Pose Message Schema
# =============================================================================
@dataclass
class PoseData:
    """Single pose data point with quaternion orientation and translation."""

    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "qw": self.qw,
            "tx": self.tx,
            "ty": self.ty,
            "tz": self.tz,
        }

    @staticmethod
    def from_dict(data: dict[str, float]) -> "PoseData":
        return PoseData(
            qx=data.get("qx", 0.0),
            qy=data.get("qy", 0.0),
            qz=data.get("qz", 0.0),
            qw=data.get("qw", 1.0),
            tx=data.get("tx", 0.0),
            ty=data.get("ty", 0.0),
            tz=data.get("tz", 0.0),
        )

    def to_array(self) -> np.ndarray:
        return np.array([self.qx, self.qy, self.qz, self.qw, self.tx, self.ty, self.tz])

    @staticmethod
    def from_array(arr: np.ndarray) -> "PoseData":
        return PoseData(
            qx=float(arr[0]),
            qy=float(arr[1]),
            qz=float(arr[2]),
            qw=float(arr[3]),
            tx=float(arr[4]),
            ty=float(arr[5]),
            tz=float(arr[6]),
        )


@dataclass
class PoseMessageSchema:
    """Standardized message schema for pose / positional data."""

    timestamp: float = 0.0
    device_id: str = "iphone"
    pose: PoseData = field(default_factory=PoseData)

    def serialize(self) -> bytes:
        data = {
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "pose": self.pose.to_dict(),
        }
        return msgpack.packb(data, use_bin_type=True)

    @staticmethod
    def deserialize(packed_data: bytes) -> "PoseMessageSchema":
        data = msgpack.unpackb(packed_data, object_hook=m.decode)
        return PoseMessageSchema(
            timestamp=data.get("timestamp", 0.0),
            device_id=data.get("device_id", "iphone"),
            pose=PoseData.from_dict(data.get("pose", {})),
        )

    def asdict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "pose": self.pose.to_dict(),
        }


# =============================================================================
# Image Message Schema
# =============================================================================
@dataclass
class ImageMessageSchema:
    """Standardized message schema for camera images.

    Production serialization emits msgpack binary values: software-encoded RGB
    images are JPEG bytes, existing device MJPEG bytes pass through unchanged,
    and uint16 depth images are lossless PNG bytes.

    ``deserialize`` retains ordinary-consumer support for legacy Base64 JPEG/PNG
    strings. That decoder-only compatibility is not a VLA rollout guarantee; the
    VLA encoded ingress accepts only the ``jpeg_bytes`` representation.
    """

    timestamps: dict[str, float]
    images: dict[str, Any]
    camera_info: dict[str, Any] = field(default_factory=dict)
    image_shapes: dict[str, list[int]] = field(default_factory=dict)

    @staticmethod
    def _encode_image_value(
        key: str,
        image: Any,
        jpeg_quality: int = PRODUCTION_JPEG_QUALITY,
    ) -> bytes:
        if key.endswith("_depth"):
            return ImageUtils.encode_depth_image(image)
        if isinstance(image, bytes | bytearray):
            return bytes(image)
        return ImageUtils.encode_image(image, quality=jpeg_quality)

    def serialize(
        self,
        executor: Executor | None = None,
        jpeg_quality: int = PRODUCTION_JPEG_QUALITY,
    ) -> dict[str, Any]:
        serialized_msg: dict[str, Any] = {
            "schema_version": 2,
            "timestamps": self.timestamps,
            "images": {},
            "camera_info": self.camera_info,
            "image_shapes": {
                **self.image_shapes,
                **{
                    key: list(image.shape)
                    for key, image in self.images.items()
                    if isinstance(image, np.ndarray)
                },
            },
        }
        if executor is None:
            encoded_images = [
                self._encode_image_value(key, image, jpeg_quality)
                for key, image in self.images.items()
            ]
        else:
            futures = [
                executor.submit(self._encode_image_value, key, image, jpeg_quality)
                for key, image in self.images.items()
            ]
            encoded_images = [future.result() for future in futures]
        serialized_msg["images"] = dict(
            zip(self.images, encoded_images, strict=True)
        )
        return serialized_msg

    @staticmethod
    def deserialize(data: dict[str, Any], decode_images: bool = True) -> "ImageMessageSchema":
        timestamps = data.get("timestamps", {})
        images = {}
        for key, value in data.get("images", {}).items():
            if not decode_images:
                images[key] = value
                continue

            if key.endswith("_depth") and isinstance(value, bytes | bytearray):
                images[key] = cv2.imdecode(
                    np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
            elif key.endswith("_depth") and isinstance(value, str):
                images[key] = ImageUtils.decode_depth_image(value)
            elif isinstance(value, bytes | bytearray):
                mat = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
                images[key] = mat[..., ::-1]  # BGR -> RGB
            elif isinstance(value, str):
                images[key] = ImageUtils.decode_image(value)
            elif isinstance(value, np.ndarray):
                images[key] = value
            elif isinstance(value, dict) and b"nd" in value:
                images[key] = m.decode(value)
            else:
                images[key] = value
        return ImageMessageSchema(
            timestamps=timestamps,
            images=images,
            camera_info=data.get("camera_info", {}),
            image_shapes=data.get("image_shapes", {}),
        )

    def asdict(self) -> dict[str, Any]:
        return {
            "timestamps": self.timestamps,
            "images": self.images,
            "camera_info": self.camera_info,
            "image_shapes": self.image_shapes,
        }


# =============================================================================
# ZMQ Server / Client
# =============================================================================
class SensorServer:
    """ZMQ PUB server that streams msgpack-encoded sensor payloads."""

    def start_server(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, CAMERA_SEND_HWM)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")
        print(f"Sensor server running at tcp://*:{port}")

        self.message_sent = 0
        self.message_dropped = 0

    def stop_server(self):
        self.socket.close()
        self.context.term()

    def send_message(self, data: dict[str, Any]):
        try:
            packed = msgpack.packb(data, use_bin_type=True)
            self.socket.send(packed, flags=zmq.NOBLOCK)
        except zmq.Again:
            self.message_dropped += 1
            print(f"[Warning] message dropped: {self.message_dropped}")
        self.message_sent += 1

        if self.message_sent % 100 == 0:
            print(
                f"[Sensor server] Message sent: {self.message_sent}, "
                f"message dropped: {self.message_dropped}"
            )


class SensorClient:
    """ZMQ SUB client that receives msgpack-encoded sensor payloads."""

    def start_client(self, server_ip: str, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.CONFLATE, True)
        self.socket.setsockopt(zmq.RCVHWM, 3)
        self.socket.connect(f"tcp://{server_ip}:{port}")

    def stop_client(self):
        self.socket.close()
        self.context.term()

    def receive_message(self):
        packed = self.socket.recv()
        return msgpack.unpackb(packed, object_hook=m.decode)

    def receive_message_nonblocking(self, timeout_ms: int = 0):
        if self.socket.poll(timeout_ms):
            packed = self.socket.recv()
            return msgpack.unpackb(packed, object_hook=m.decode)
        return None


# =============================================================================
# Helpers
# =============================================================================
class CameraMountPosition(Enum):
    EGO_VIEW = "ego_view"
    HEAD = "head"
    CHEST_VIEW = "chest_view"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"


class ImageUtils:
    @staticmethod
    def encode_image(image: np.ndarray, quality: int = PRODUCTION_JPEG_QUALITY) -> bytes:
        image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        ok, color_buffer = cv2.imencode(
            ".jpg",
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            raise RuntimeError("failed to encode RGB image as JPEG")
        return color_buffer.tobytes()

    @staticmethod
    def encode_depth_image(image: np.ndarray) -> bytes:
        if not isinstance(image, np.ndarray) or image.ndim != 2 or image.dtype != np.uint16:
            raise ValueError("depth image must be a 2D uint16 array")
        ok, depth_compressed = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("failed to encode depth image as PNG")
        return depth_compressed.tobytes()

    @staticmethod
    def decode_image(image: str) -> np.ndarray:
        color_data = base64.b64decode(image)
        color_array = np.frombuffer(color_data, dtype=np.uint8)
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)

    @staticmethod
    def decode_depth_image(image: str) -> np.ndarray:
        depth_data = base64.b64decode(image)
        depth_array = np.frombuffer(depth_data, dtype=np.uint8)
        return cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
