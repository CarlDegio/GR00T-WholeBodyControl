#!/usr/bin/env python3
"""Publish metric chest depth estimated directly from RGB with Depth Anything V2."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

from gear_sonic.camera.calibration import load_camera_intrinsics
from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer
from gear_sonic.runtime.gateway.sensor_client import (
    SensorGatewayClient,
    SensorGatewayClientError,
)
from gear_sonic.runtime.profile import load_component_config, load_runtime_profile
from gear_sonic.runtime.gateway.control_client import ControlGatewaySubscriber
from gear_sonic.runtime.gateway.sensor import DEPTH_ANYTHING_STATUS_TYPE
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest, TimestampBasis
from gear_sonic.runtime.telemetry import configure_file_logging


LOGGER = logging.getLogger("sonic.depth_anything")
DEPTH_SOURCE = "depth-anything-v2-metric-hypersim-vitb"
RGB_STREAM = "camera/chest_view"
CAMERA_NAME = "chest_view"
DEPTH_NAME = "chest_view_depth"


@dataclass
class DepthAnythingConfig:
    sensor_gateway_request_timeout_ms: int
    sensor_gateway_max_age_ms: float
    root: str
    checkpoint: str
    camera_intrinsics_path: str
    encoder: str
    device: str
    inference_hz: float
    input_size: int
    model_max_depth_m: float
    publish_max_depth_m: float
    use_amp: bool
    idle_heartbeat_hz: float
    profile: str = ""
    overlay: tuple[str, ...] = ()


def load_depth_anything_config(
    profile: str = "", overlays: tuple[str, ...] = ()
) -> DepthAnythingConfig:
    return load_component_config(
        DepthAnythingConfig,
        "depth_anything",
        profile or None,
        overlays=overlays,
    )


class DepthAnythingInferenceGate:
    """Keep DA resident while allowing forward passes only for active consumers."""

    ACCEPTED_COMMANDS = {
        "start_navigation",
        "lavira_rgbd_captured",
        "navigation_goal",
        "navigation_status",
        "cancel_navigation",
    }
    TERMINAL_STATES = {"reached", "failed", "stopped"}

    def __init__(self) -> None:
        self.owner: str | None = None
        self.generation = -1

    @property
    def active(self) -> bool:
        return self.owner is not None

    def apply(self, name: str, parameters: dict[str, Any]) -> bool:
        previous = (self.owner, self.generation)
        generation = int(parameters.get("generation", -1))
        if name == "start_navigation" and generation >= self.generation:
            self.owner = "lavira"
            self.generation = generation
        elif (
            name in {"lavira_rgbd_captured", "navigation_goal"}
            and self.owner == "lavira"
            and generation == self.generation
        ):
            self.owner = None
        elif (
            name == "navigation_status"
            and self.owner == "lavira"
            and generation == self.generation
            and str(parameters.get("state", "")) in self.TERMINAL_STATES
        ):
            self.owner = None
        elif name == "cancel_navigation" and generation >= self.generation:
            self.owner = None
            self.generation = generation
        return previous != (self.owner, self.generation)


def depth_anything_status_payload(gate: DepthAnythingInferenceGate) -> dict[str, Any]:
    return {
        "type": DEPTH_ANYTHING_STATUS_TYPE,
        "version": 1,
        "active": gate.active,
        "owner": gate.owner or "idle",
        "generation": gate.generation,
        "timestamp": time.time(),
    }


class DepthAnythingSensorGatewayClient:
    """Read chest RGB only; raw sensor depth never enters this process."""

    def __init__(
        self,
        endpoint: str,
        *,
        request_timeout_ms: int = 100,
        max_age_ms: float = 1000.0,
        client: SensorGatewayClient | None = None,
    ) -> None:
        self.client = client or SensorGatewayClient(
            endpoint,
            request_timeout_ms=request_timeout_ms,
        )
        self._owns_client = client is None
        self.max_age_ms = float(max_age_ms)
        self._last_error = ""

    def read(self) -> tuple[np.ndarray, float, dict[str, Any]] | None:
        try:
            snapshot = self.client.read_snapshot(
                SnapshotRequest(
                    streams=(RGB_STREAM,),
                    max_age_ms=self.max_age_ms,
                    max_skew_ms=0.0,
                    timestamp_basis=TimestampBasis.SOURCE,
                ),
                retries=0,
            )
        except SensorGatewayClientError as exc:
            message = str(exc)
            if not self._last_error:
                LOGGER.warning("waiting for chest RGB: %s", message)
            self._last_error = message
            return None

        frame = snapshot.snapshot.frames[RGB_STREAM]
        rgb = np.asarray(snapshot.arrays[RGB_STREAM])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(
                f"Gateway chest RGB must be HxWx3 uint8, got {rgb.shape} {rgb.dtype}"
            )
        if self._last_error:
            LOGGER.info("chest RGB recovered")
            self._last_error = ""
        timestamp = (
            frame.source_timestamp_ns * 1.0e-9
            if frame.source_timestamp_ns > 0
            else time.time()
        )
        return rgb, timestamp, dict(frame.attributes.get("camera_info", {}))

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


def metric_depth_payload(
    depth_m: np.ndarray,
    camera_info: dict[str, Any],
    *,
    timestamp: float,
    publish_max_depth_m: float,
    inference_owner: str = "unknown",
    inference_generation: int = -1,
) -> ImageMessageSchema:
    """Encode metric depth as uint16 millimetres with an explicit metre scale."""

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth Anything output must be HxW, got {depth.shape}")
    valid = (
        np.isfinite(depth)
        & (depth > 0.0)
        & (depth <= float(publish_max_depth_m))
        & (depth < np.iinfo(np.uint16).max / 1000.0)
    )
    depth_mm = np.zeros(depth.shape, dtype=np.uint16)
    depth_mm[valid] = np.rint(depth[valid] * 1000.0).astype(np.uint16)
    info = dict(camera_info)
    info.update(
        {
            "width": int(depth.shape[1]),
            "height": int(depth.shape[0]),
            "depth_scale_m": 0.001,
            "depth_aligned_to": CAMERA_NAME,
            "depth_source": DEPTH_SOURCE,
            "depth_representation": "uint16_millimetres",
            "metric_model_output": True,
            "uses_raw_depth": False,
            "inference_owner": str(inference_owner),
            "inference_generation": int(inference_generation),
        }
    )
    return ImageMessageSchema(
        timestamps={DEPTH_NAME: float(timestamp)},
        images={DEPTH_NAME: depth_mm},
        camera_info={CAMERA_NAME: info},
    )


def mark_ready(path: str) -> None:
    marker = Path(path).expanduser()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ready\n", encoding="utf-8")


class DepthAnythingMetricEstimator:
    """Depth Anything V2 metric indoor Base model."""

    MODEL_CONFIGS = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    def __init__(self, config: DepthAnythingConfig) -> None:
        root = Path(config.root).expanduser().resolve()
        checkpoint = Path(config.checkpoint).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Depth Anything metric source not found: {root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Depth Anything checkpoint not found: {checkpoint}")
        if config.encoder not in self.MODEL_CONFIGS:
            raise ValueError(f"unsupported Depth Anything encoder: {config.encoder}")
        sys.path.insert(0, str(root))

        import torch
        import torch.nn.functional as functional
        from depth_anything_v2.dpt import DepthAnythingV2

        self.torch = torch
        self.functional = functional
        self.device = torch.device(config.device)
        self.input_size = int(config.input_size)
        self.use_amp = bool(config.use_amp and self.device.type == "cuda")
        model_args = dict(self.MODEL_CONFIGS[config.encoder])
        model_args["max_depth"] = float(config.model_max_depth_m)
        LOGGER.info("loading checkpoint=%s device=%s", checkpoint, self.device)
        model = DepthAnythingV2(**model_args)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        """Return the model's direct metric estimate in metres; no raw-depth fusion."""

        torch = self.torch
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        image, (height, width) = self.model.image2tensor(bgr, self.input_size)
        image = image.to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            depth = self.model(image)
            depth = self.functional.interpolate(
                depth[:, None],
                (height, width),
                mode="bilinear",
                align_corners=True,
            )[0, 0]
        return depth.float().cpu().numpy()


def _calibration_for_frame(
    configured: dict[str, Any], live: dict[str, Any], rgb: np.ndarray
) -> dict[str, Any]:
    info = dict(configured)
    info.update(live)
    expected = (int(info.get("height", 0)), int(info.get("width", 0)))
    if expected != rgb.shape[:2]:
        raise ValueError(
            f"chest calibration {expected[::-1]} does not match RGB "
            f"{rgb.shape[1]}x{rgb.shape[0]}"
        )
    return info


def main(config: DepthAnythingConfig, *, ready_file: str = "") -> None:
    configure_file_logging("depth_anything")
    profile = load_runtime_profile(config.profile or None, overlays=config.overlay)
    publish_port = profile.endpoint("depth_anything").port
    if config.inference_hz <= 0.0:
        raise ValueError("inference_hz must be positive")
    if config.publish_max_depth_m <= 0.0:
        raise ValueError("publish_max_depth_m must be positive")
    if config.idle_heartbeat_hz <= 0.0:
        raise ValueError("idle_heartbeat_hz must be positive")
    calibration = load_camera_intrinsics(config.camera_intrinsics_path)[CAMERA_NAME]
    configured_info = calibration.asdict()
    estimator = DepthAnythingMetricEstimator(config)
    client = DepthAnythingSensorGatewayClient(
        profile.endpoint_uri("sensor_gateway_metadata"),
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
    )
    publisher = SensorServer()
    publisher.start_server(publish_port)
    control = ControlGatewaySubscriber(
        profile.endpoint_uri("control_gateway_dispatch"),
        accepted_names=DepthAnythingInferenceGate.ACCEPTED_COMMANDS,
    )
    gate = DepthAnythingInferenceGate()
    if ready_file:
        mark_ready(ready_file)
    LOGGER.info(
        "READY publish_port=%s ready_file=%s",
        publish_port,
        ready_file or "-",
    )

    period = 1.0 / config.inference_hz
    heartbeat_period = 1.0 / config.idle_heartbeat_hz
    next_heartbeat = 0.0
    try:
        while True:
            started = time.monotonic()
            state_changed = False
            while True:
                command = control.read_command()
                if command is None:
                    break
                state_changed = (
                    gate.apply(command.name, dict(command.parameters))
                    or state_changed
                )
            if state_changed:
                LOGGER.info(
                    "inference=%s owner=%s generation=%s",
                    "active" if gate.active else "idle",
                    gate.owner or "none",
                    gate.generation,
                )
                publisher.send_message(depth_anything_status_payload(gate))
                next_heartbeat = started + heartbeat_period
            elif started >= next_heartbeat:
                publisher.send_message(depth_anything_status_payload(gate))
                next_heartbeat = started + heartbeat_period

            if not gate.active:
                time.sleep(min(0.02, max(0.0, next_heartbeat - time.monotonic())))
                continue
            packet = client.read()
            if packet is not None:
                rgb, timestamp, live_info = packet
                info = _calibration_for_frame(configured_info, live_info, rgb)
                depth_m = estimator.infer(rgb)
                publisher.send_message(
                    metric_depth_payload(
                        depth_m,
                        info,
                        timestamp=timestamp,
                        publish_max_depth_m=config.publish_max_depth_m,
                        inference_owner=gate.owner or "unknown",
                        inference_generation=gate.generation,
                    ).serialize()
                )
            remaining = period - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        control.close()
        client.close()
        publisher.stop_server()
        if ready_file:
            Path(ready_file).expanduser().unlink(missing_ok=True)
        LOGGER.info("STOPPED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--ready-file", default="")
    args = parser.parse_args()
    try:
        main(
            load_depth_anything_config(args.profile, tuple(args.overlay)),
            ready_file=args.ready_file,
        )
    except Exception:
        configure_file_logging("depth_anything")
        LOGGER.exception("FAILED")
        raise
