"""Asynchronous LingBot-Depth completion viewer for the chest RGB-D stream."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
import tyro

from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer
from gear_sonic.runtime.client import SensorGatewayClient, SensorGatewayClientError
from gear_sonic.runtime.control_client import ControlGatewaySubscriber
from gear_sonic.runtime.snapshot import SnapshotRequest
from gear_sonic.scripts.run_depth_camera_viewer import colorize_depth
from gear_sonic.runtime.visualization import VisualizationPublisher


@dataclass
class LingBotDepthViewerConfig:
    sensor_gateway_endpoint: str = "tcp://127.0.0.1:5560"
    sensor_gateway_request_timeout_ms: int = 100
    sensor_gateway_max_age_ms: float = 1000.0
    sensor_gateway_max_skew_ms: float = 5.0
    publish_port: int = 5564
    ready_file: str = ""
    model: str = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
    lingbot_root: str = "/home/user/Project/lingbot-depth"
    device: str = "cuda"
    inference_hz: float = 2.0
    display_hz: float = 20.0
    max_depth_m: float = 10.0
    resolution_level: int = 6
    use_fp16: bool = True
    visualize: bool = True
    visualization_gateway_endpoint: str = ""
    control_gateway_endpoint: str = "tcp://127.0.0.1:5565"


class LingBotSensorGatewayClient:
    """Read the chest RGB-D pair from Gateway shared memory only."""

    STREAMS = ("camera/chest_view", "camera/chest_view_depth")

    def __init__(
        self,
        endpoint: str,
        *,
        request_timeout_ms: int = 100,
        max_age_ms: float = 1000.0,
        max_skew_ms: float = 5.0,
        client: SensorGatewayClient | None = None,
    ) -> None:
        self.client = client or SensorGatewayClient(
            endpoint,
            request_timeout_ms=request_timeout_ms,
        )
        self._owns_client = client is None
        self.max_age_ms = float(max_age_ms)
        self.max_skew_ms = float(max_skew_ms)
        self._last_error = ""
        self._last_error_time = 0.0

    def read(self, blocking: bool = False, **_kwargs) -> dict[str, Any] | None:
        if blocking:
            raise ValueError("blocking Gateway RGB-D reads are unsupported")
        try:
            snapshot = self.client.read_snapshot(
                SnapshotRequest(
                    streams=self.STREAMS,
                    max_age_ms=self.max_age_ms,
                    max_skew_ms=self.max_skew_ms,
                ),
                retries=0,
            )
        except SensorGatewayClientError as exc:
            now = time.monotonic()
            message = str(exc)
            if message != self._last_error or now - self._last_error_time >= 2.0:
                print(f"[LingBotDepth] waiting for SensorGateway RGB-D: {message}")
                self._last_error = message
                self._last_error_time = now
            return None

        rgb_frame = snapshot.snapshot.frames[self.STREAMS[0]]
        depth_frame = snapshot.snapshot.frames[self.STREAMS[1]]
        rgb_timestamp = (
            rgb_frame.source_timestamp_ns * 1.0e-9
            if rgb_frame.source_timestamp_ns > 0
            else time.time()
        )
        depth_timestamp = (
            depth_frame.source_timestamp_ns * 1.0e-9
            if depth_frame.source_timestamp_ns > 0
            else rgb_timestamp
        )
        return {
            "images": {
                "chest_view": snapshot.arrays[self.STREAMS[0]],
                "chest_view_depth": snapshot.arrays[self.STREAMS[1]],
            },
            "timestamps": {
                "chest_view": rgb_timestamp,
                "chest_view_depth": depth_timestamp,
            },
            "camera_info": {
                "chest_view": dict(rgb_frame.attributes.get("camera_info", {}))
            },
        }

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


class LingBotInferenceModeGate:
    """Keep the model resident while admitting frame updates only in PLANNER mode."""

    def __init__(self) -> None:
        self.mode = "PLANNER"

    @property
    def inference_enabled(self) -> bool:
        return self.frame_updates_enabled

    @property
    def frame_updates_enabled(self) -> bool:
        """Return whether RGB-D reads and visualization updates may run."""
        return self.mode == "PLANNER"

    def accept(self, command_name: str) -> bool:
        previous = self.mode
        if command_name == "select_pose_mode":
            self.mode = "POSE"
        elif command_name == "select_planner_mode":
            self.mode = "PLANNER"
        return self.mode != previous


def _read_frame_if_enabled(client, mode_gate: LingBotInferenceModeGate):
    """Read one RGB-D frame unless POSE mode is freezing LingBot updates."""
    if not mode_gate.frame_updates_enabled:
        return None
    return client.read(blocking=False)


def configure_lingbot_runtime_environment() -> None:
    """Ensure the official nested-tensor path can load xFormers."""
    os.environ.pop("XFORMERS_DISABLED", None)


def mark_ready(path: str) -> None:
    marker = Path(path).expanduser()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ready\n", encoding="utf-8")


def prepare_depth_meters(
    raw_depth: np.ndarray, *, depth_scale_m: float, max_depth_m: float
) -> np.ndarray:
    depth = np.asarray(raw_depth)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected 2-D depth, got {depth.shape}")
    depth_m = depth.astype(np.float32) * float(depth_scale_m)
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_depth_m)
    return np.where(valid, depth_m, 0.0).astype(np.float32)


def normalized_intrinsics(
    camera_info: dict[str, Any], *, width: int, height: int
) -> np.ndarray:
    required = ("fx", "fy", "cx", "cy")
    missing = [name for name in required if name not in camera_info]
    if missing:
        raise ValueError(f"Missing camera intrinsics: {', '.join(missing)}")
    return np.array(
        [
            [float(camera_info["fx"]) / width, 0.0, float(camera_info["cx"]) / width],
            [0.0, float(camera_info["fy"]) / height, float(camera_info["cy"]) / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def conservative_depth_fusion(
    raw_depth_m: np.ndarray,
    completed_depth_m: np.ndarray,
    *,
    max_depth_m: float,
) -> np.ndarray:
    raw = np.asarray(raw_depth_m, dtype=np.float32)
    completed = np.asarray(completed_depth_m, dtype=np.float32)
    if raw.shape != completed.shape:
        completed = cv2.resize(
            completed, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    raw_valid = np.isfinite(raw) & (raw > 0.0) & (raw <= max_depth_m)
    completed_valid = (
        np.isfinite(completed) & (completed > 0.0) & (completed <= max_depth_m)
    )
    fused = np.zeros(raw.shape, dtype=np.float32)
    fused[raw_valid] = raw[raw_valid]
    only_completed = ~raw_valid & completed_valid
    fused[only_completed] = completed[only_completed]
    both = raw_valid & completed_valid
    fused[both] = np.minimum(raw[both], completed[both])
    return fused


def completed_depth_payload(
    rgb: np.ndarray,
    completed_depth_m: np.ndarray,
    camera_info: dict[str, Any],
    *,
    timestamp: float,
    max_depth_m: float,
) -> ImageMessageSchema:
    """Build a composed-camera message containing pure LingBot depth."""
    completed = np.asarray(completed_depth_m, dtype=np.float32)
    valid = (
        np.isfinite(completed) & (completed > 0.0) & (completed <= max_depth_m)
    )
    depth_mm = np.zeros(completed.shape, dtype=np.uint16)
    depth_mm[valid] = np.rint(completed[valid] * 1000.0).astype(np.uint16)
    info = dict(camera_info)
    info.update(
        {
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "depth_scale_m": 0.001,
            "depth_aligned_to": "chest_view",
            "depth_source": "lingbot-depth",
        }
    )
    return ImageMessageSchema(
        timestamps={"chest_view": timestamp, "chest_view_depth": timestamp},
        images={"chest_view": rgb, "chest_view_depth": depth_mm},
        camera_info={"chest_view": info},
    )


def resolve_local_model_path(
    model: str, *, download: Callable[..., str] | None = None
) -> str:
    """Resolve a model path strictly from disk without any metadata request."""
    direct_path = Path(model).expanduser()
    if direct_path.is_file():
        return str(direct_path.resolve())
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download
    return download(
        repo_id=model,
        repo_type="model",
        filename="model.pt",
        local_files_only=True,
    )


class LingBotDepthCompleter:
    def __init__(self, config: LingBotDepthViewerConfig):
        root = Path(config.lingbot_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"LingBot-Depth repository not found: {root}")
        sys.path.insert(0, str(root))
        configure_lingbot_runtime_environment()

        import torch
        from mdm.model.v2 import MDMModel

        self.torch = torch
        self.device = torch.device(config.device)
        model_path = resolve_local_model_path(config.model)
        print(f"[LingBotDepth] loading {model_path} on {self.device} ...")
        self.model = MDMModel.from_pretrained(model_path).to(self.device).eval()
        self.resolution_level = config.resolution_level
        self.use_fp16 = config.use_fp16

    def infer(
        self, rgb: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, float]:
        torch = self.torch
        started = time.monotonic()
        rgb_tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.astype(np.float32) / 255.0)
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)
        depth_tensor = torch.from_numpy(np.ascontiguousarray(depth_m)).unsqueeze(0).to(
            self.device
        )
        intrinsics_tensor = torch.from_numpy(intrinsics).unsqueeze(0).to(self.device)
        output = self.model.infer(
            rgb_tensor,
            depth_in=depth_tensor,
            intrinsics=intrinsics_tensor,
            resolution_level=self.resolution_level,
            use_fp16=self.use_fp16,
            apply_mask=False,
        )
        completed = output["depth"].squeeze(0).detach().float().cpu().numpy()
        return completed, time.monotonic() - started


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.putText(
        output, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return output


def main(config: LingBotDepthViewerConfig) -> None:
    completer = LingBotDepthCompleter(config)
    client = LingBotSensorGatewayClient(
        config.sensor_gateway_endpoint,
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        max_skew_ms=config.sensor_gateway_max_skew_ms,
    )
    publisher = SensorServer()
    publisher.start_server(config.publish_port)
    visualization_publisher = (
        VisualizationPublisher(config.visualization_gateway_endpoint)
        if config.visualization_gateway_endpoint
        else None
    )
    mode_commands = ControlGatewaySubscriber(
        config.control_gateway_endpoint,
        accepted_names={"select_pose_mode", "select_planner_mode"},
    )
    mode_gate = LingBotInferenceModeGate()
    if config.ready_file:
        mark_ready(config.ready_file)
        print(f"[LingBotDepth] model loaded; ready marker: {config.ready_file}")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lingbot-depth")
    pending: Future[tuple[np.ndarray, float]] | None = None
    pending_frame: tuple[np.ndarray, dict[str, Any], float] | None = None
    completed_depth: np.ndarray | None = None
    inference_seconds = 0.0
    published_frames = 0
    last_submit = 0.0
    period = 1.0 / max(config.display_hz, 1.0)
    inference_period = 1.0 / max(config.inference_hz, 0.01)
    window = "LingBot Chest Depth: raw | completed | conservative"

    try:
        while True:
            loop_start = time.monotonic()
            command = mode_commands.read_command()
            if command is not None and mode_gate.accept(command.name):
                state = "running" if mode_gate.inference_enabled else "paused"
                visualization_state = (
                    "running" if mode_gate.frame_updates_enabled else "frozen"
                )
                print(
                    f"[LingBotDepth] GPU inference {state}; "
                    f"visualization {visualization_state}; "
                    f"control mode={mode_gate.mode}, model remains loaded"
                )
            packet = _read_frame_if_enabled(client, mode_gate)
            images = packet.get("images", {}) if packet else {}
            rgb = images.get("chest_view")
            raw_depth = images.get("chest_view_depth")

            if pending is not None and pending.done():
                try:
                    completed_depth, inference_seconds = pending.result()
                    if pending_frame is not None:
                        frame_rgb, frame_info, frame_timestamp = pending_frame
                        publisher.send_message(
                            completed_depth_payload(
                                frame_rgb,
                                completed_depth,
                                frame_info,
                                timestamp=frame_timestamp,
                                max_depth_m=config.max_depth_m,
                            ).serialize()
                        )
                        published_frames += 1
                        if published_frames == 1 or published_frames % 10 == 0:
                            print(
                                f"[LingBotDepth] ready: published frame "
                                f"{published_frames} to tcp://127.0.0.1:"
                                f"{config.publish_port} "
                                f"({inference_seconds:.2f}s inference)"
                            )
                except Exception as exc:
                    print(f"[LingBotDepth] inference failed: {exc}")
                pending = None
                pending_frame = None

            if rgb is not None and raw_depth is not None:
                info = packet.get("camera_info", {}).get("chest_view", {})
                scale = float(info.get("depth_scale_m", 0.001))
                raw_m = prepare_depth_meters(
                    raw_depth,
                    depth_scale_m=scale,
                    max_depth_m=config.max_depth_m,
                )
                if rgb.shape[:2] != raw_m.shape:
                    rgb = cv2.resize(rgb, (raw_m.shape[1], raw_m.shape[0]))

                now = time.monotonic()
                if (
                    mode_gate.inference_enabled
                    and pending is None
                    and now - last_submit >= inference_period
                ):
                    try:
                        intrinsics = normalized_intrinsics(
                            info, width=raw_m.shape[1], height=raw_m.shape[0]
                        )
                    except ValueError as exc:
                        print(f"[LingBotDepth] {exc}")
                    else:
                        pending = executor.submit(
                            completer.infer, rgb.copy(), raw_m.copy(), intrinsics
                        )
                        pending_frame = (
                            rgb.copy(),
                            dict(info),
                            float(packet.get("timestamps", {}).get("chest_view", time.time())),
                        )
                        last_submit = now

                raw_color, _ = colorize_depth(
                    raw_m, max_depth_m=config.max_depth_m, depth_scale_m=1.0
                )
                tiles = [_label(raw_color, "RAW")]
                if completed_depth is not None:
                    fused = conservative_depth_fusion(
                        raw_m, completed_depth, max_depth_m=config.max_depth_m
                    )
                    completed_color, _ = colorize_depth(
                        completed_depth,
                        max_depth_m=config.max_depth_m,
                        depth_scale_m=1.0,
                    )
                    fused_color, _ = colorize_depth(
                        fused, max_depth_m=config.max_depth_m, depth_scale_m=1.0
                    )
                    tiles.extend(
                        [
                            _label(completed_color, f"LINGBOT {inference_seconds:.2f}s"),
                            _label(fused_color, "CONSERVATIVE"),
                        ]
                    )
                if not mode_gate.inference_enabled:
                    tiles[0] = _label(raw_color, "RAW | LINGBOT PAUSED IN POSE")
                visualization_frame = np.hstack(tiles)
                if visualization_publisher is not None:
                    visualization_publisher.publish(
                        "visualization/lingbot_depth", visualization_frame
                    )
                if config.visualize:
                    cv2.imshow(window, visualization_frame)

            if config.visualize and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            remaining = period - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        client.close()
        publisher.stop_server()
        if visualization_publisher is not None:
            visualization_publisher.close()
        mode_commands.close()
        if config.ready_file:
            Path(config.ready_file).expanduser().unlink(missing_ok=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(LingBotDepthViewerConfig))
