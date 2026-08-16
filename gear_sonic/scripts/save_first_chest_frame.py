"""Save the first valid ``chest_view`` RGB frame from a camera server."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Mapping

import cv2
import numpy as np

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


def extract_chest_rgb(message: Mapping[str, Any] | None) -> np.ndarray | None:
    """Return a validated RGB ``chest_view`` image, or ``None`` if absent."""
    if not message:
        return None
    images = message.get("images")
    if not isinstance(images, Mapping) or "chest_view" not in images:
        return None

    image = images["chest_view"]
    if not isinstance(image, np.ndarray):
        raise ValueError("chest_view is not a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"chest_view must have shape HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"chest_view must be uint8 RGB, got {image.dtype}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("chest_view is empty")
    return np.ascontiguousarray(image)


def save_rgb_png(
    image_rgb: np.ndarray, output_path: Path, overwrite: bool = False
) -> None:
    """Save an RGB array as a lossless PNG without swapping red and blue."""
    if output_path.suffix.lower() != ".png":
        raise ValueError("output_path must use a .png extension")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise OSError(f"failed to write image: {output_path}")


def capture_first_chest_frame(
    camera_host: str,
    camera_port: int,
    output_path: Path,
    timeout_sec: float,
    ready_file: Path | None = None,
    overwrite: bool = False,
) -> None:
    """Connect, announce subscriber readiness, and save the first valid frame."""
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be greater than zero")

    client = ComposedCameraClientSensor(
        server_ip=camera_host,
        port=camera_port,
        decode_images=True,
    )
    try:
        if ready_file is not None:
            ready_file.parent.mkdir(parents=True, exist_ok=True)
            ready_file.touch()

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            image = extract_chest_rgb(client.read(blocking=False))
            if image is not None:
                save_rgb_png(image, output_path, overwrite=overwrite)
                print(
                    f"Saved first chest_view RGB frame ({image.shape[1]}x{image.shape[0]}) "
                    f"to {output_path.resolve()}",
                    flush=True,
                )
                return
            time.sleep(0.01)
    finally:
        client.close()

    raise TimeoutError(
        f"no chest_view RGB frame received from {camera_host}:{camera_port} "
        f"within {timeout_sec:.1f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save the first valid chest_view RGB frame from a camera server."
    )
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="Touch this file after the ZMQ subscriber has connected.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_first_chest_frame(
        camera_host=args.camera_host,
        camera_port=args.camera_port,
        output_path=args.output_path,
        timeout_sec=args.timeout_sec,
        ready_file=args.ready_file,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
