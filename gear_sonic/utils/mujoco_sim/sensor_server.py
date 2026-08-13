"""ZMQ PUB server for streaming JPEG-encoded camera images as msgpack payloads."""

import base64
from dataclasses import dataclass, field
from typing import Any, Dict

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import zmq

from gear_sonic.camera.constants import PRODUCTION_JPEG_QUALITY

CAMERA_SEND_HWM = 1


@dataclass
class ImageMessageSchema:
    """
    Standardized message schema for image data.
    """

    timestamps: Dict[str, float]
    images: Dict[str, np.ndarray]
    image_shapes: Dict[str, list[int]] = field(default_factory=dict)

    def serialize(self) -> Dict[str, Any]:
        serialized_msg = {
            "timestamps": self.timestamps,
            "images": {},
            "image_shapes": {
                **self.image_shapes,
                **{
                    key: list(image.shape)
                    for key, image in self.images.items()
                    if isinstance(image, np.ndarray)
                },
            },
        }
        for key, image in self.images.items():
            serialized_msg["images"][key] = ImageUtils.encode_image(image)
        return serialized_msg

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> "ImageMessageSchema":
        timestamps = data.get("timestamps", {})
        images = {}
        for key, value in data.get("images", {}).items():
            if isinstance(value, str):
                images[key] = ImageUtils.decode_image(value)
            elif isinstance(value, bytes | bytearray):
                mat = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
                images[key] = mat[..., ::-1]
            else:
                images[key] = value
        return ImageMessageSchema(
            timestamps=timestamps,
            images=images,
            image_shapes=data.get("image_shapes", {}),
        )


class SensorServer:
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

    def send_message(self, data: Dict[str, Any]):
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


class ImageUtils:
    @staticmethod
    def encode_image(image: np.ndarray) -> bytes:
        image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        ok, color_buffer = cv2.imencode(
            ".jpg",
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), PRODUCTION_JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError("failed to encode RGB image as JPEG")
        return color_buffer.tobytes()

    @staticmethod
    def decode_image(image: str) -> np.ndarray:
        color_data = base64.b64decode(image)
        color_array = np.frombuffer(color_data, dtype=np.uint8)
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)
