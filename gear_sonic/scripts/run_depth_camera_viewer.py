"""Display the chest depth stream from the composed camera server."""

from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np
import tyro

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


@dataclass(frozen=True)
class DepthStats:
    valid_ratio: float
    min_depth_m: float
    median_depth_m: float


@dataclass
class DepthViewerConfig:
    camera_host: str = "localhost"
    camera_port: int = 5555
    stream_name: str = "chest_view_depth"
    max_depth_m: float = 5.0
    fps: float = 20.0


def colorize_depth(
    depth: np.ndarray, *, max_depth_m: float, depth_scale_m: float = 0.001
) -> tuple[np.ndarray, DepthStats]:
    """Convert a raw depth image to a fixed-range BGR visualization."""
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth image, got shape {depth.shape}")

    depth_m = depth.astype(np.float32) * float(depth_scale_m)
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_depth_m)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        255.0 * depth_m[valid] / max_depth_m, 0.0, 255.0
    ).astype(np.uint8)
    color = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0

    values = depth_m[valid]
    stats = DepthStats(
        valid_ratio=float(valid.mean()),
        min_depth_m=float(values.min()) if values.size else 0.0,
        median_depth_m=float(np.median(values)) if values.size else 0.0,
    )
    return color, stats


def main(config: DepthViewerConfig) -> None:
    client = ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )
    window_name = "LaViRA Chest Depth (0-5 m)"
    period = 1.0 / max(config.fps, 1.0)
    print(
        f"[DepthViewer] waiting for {config.stream_name} at "
        f"{config.camera_host}:{config.camera_port}"
    )

    try:
        while True:
            started = time.monotonic()
            packet = client.read(blocking=False)
            images = packet.get("images", {}) if packet else {}
            depth = images.get(config.stream_name)
            if depth is not None:
                camera_info = packet.get("camera_info", {})
                stream_info = camera_info.get(config.stream_name, {})
                if not stream_info:
                    stream_info = camera_info.get("chest_view", {})
                scale = float(stream_info.get("depth_scale_m", 0.001))
                try:
                    canvas, stats = colorize_depth(
                        depth,
                        max_depth_m=config.max_depth_m,
                        depth_scale_m=scale,
                    )
                except ValueError as exc:
                    print(f"[DepthViewer] {exc}")
                else:
                    label = (
                        f"valid {stats.valid_ratio * 100:.1f}%  "
                        f"min {stats.min_depth_m:.2f} m  "
                        f"median {stats.median_depth_m:.2f} m"
                    )
                    cv2.putText(
                        canvas,
                        label,
                        (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, canvas)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(DepthViewerConfig))
