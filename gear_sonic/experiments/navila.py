"""Client for the inspected NaVILA MessagePack REQ/REP server."""

import math
import os
import re
import time

import cv2
import msgpack
import zmq


def parse_action(response, sequence, config):
    if not isinstance(response, dict) or response.get("type") != "navila_action" or response.get("version") != 1:
        raise ValueError(f"Invalid NaVILA response: {response!r}")
    if response.get("sequence") != sequence:
        raise ValueError("NaVILA response sequence mismatch")
    text = str(response.get("text", "")).strip().lower().rstrip(".")
    if text == "stop":
        return None
    turn = re.fullmatch(r"turn\s+(left|right)(?:\s+by)?\s+(\d+(?:\.\d+)?)\s*(?:degrees?|°)", text)
    forward = re.fullmatch(r"move\s+forward(?:\s+by)?\s+(\d+(?:\.\d+)?)\s*(cm|centimeters?|m|meters?)", text)
    if turn:
        degrees = float(turn[2])
        if degrees not in {15, 30, 45}:
            raise ValueError("Unsupported NaVILA turn angle")
        wz = float(config["wz"])
        return (0.0, 0.0, wz if turn[1] == "left" else -wz), math.radians(degrees) / wz
    if forward:
        distance = float(forward[1]) / (100.0 if forward[2].startswith("c") else 1.0)
        if not any(math.isclose(distance, x) for x in (0.25, 0.5, 0.75)):
            raise ValueError("Unsupported NaVILA forward distance")
        vx = float(config["vx"])
        return (vx, 0.0, 0.0), distance / vx
    raise ValueError(f"Unsupported NaVILA action: {text!r}")


class NavilaClient:
    def __init__(self, config):
        self.config = config
        self.sequence = 0

    def call(self, endpoint, data=None):
        # A fresh REQ socket also recovers cleanly after a request timeout.
        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        for option in (zmq.RCVTIMEO, zmq.SNDTIMEO):
            socket.setsockopt(option, int(1000 * self.config["timeout_s"]))
        socket.connect(f"tcp://{self.config['host']}:{self.config['port']}")
        request = {"endpoint": endpoint, "data": data}
        token = os.getenv(self.config.get("token_env", "NAVILA_API_TOKEN"))
        if token:
            request["api_token"] = token
        try:
            socket.send(msgpack.packb(request, use_bin_type=True))
            response = msgpack.unpackb(socket.recv(), raw=False)
        finally:
            socket.close()
        if not isinstance(response, dict) or response.get("error"):
            raise RuntimeError(f"NaVILA request failed: {response}")
        return response

    def reset(self):
        if self.call("ping").get("status") != "ok" or self.call("reset").get("status") != "ok":
            raise RuntimeError("NaVILA ping/reset failed")
        self.sequence = 0

    def action(self, image, instruction):
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.config["jpeg_quality"]])
        if not ok:
            raise ValueError("NaVILA JPEG encoding failed")
        self.sequence += 1
        response = self.call(
            "get_action",
            dict(
                type="navila_camera_frame",
                version=1,
                sequence=self.sequence,
                timestamp_ns=time.time_ns(),
                camera=self.config["camera"],
                instruction=instruction,
                encoding="jpeg",
                image_jpeg=jpeg.tobytes(),
            ),
        )
        return parse_action(response, self.sequence, self.config)
