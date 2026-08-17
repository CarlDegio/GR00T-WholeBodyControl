"""Validated storage for one-shot robot camera calibration captures."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import cv2
import numpy as np


REQUIRED_STREAMS = ("ego_view", "chest_view")
FORMAT_VERSION = 1
DEFAULT_CAMERA_INTRINSICS_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "camera_intrinsics.json"
)


class CameraCalibrationError(ValueError):
    """Raised when saved or live camera calibration is incomplete."""


@dataclass(frozen=True)
class CameraIntrinsics:
    stream_name: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_model: str
    distortion_coeffs: tuple[float, ...]
    distortion_coeff_order: tuple[str, ...]
    source_distortion_model: str
    source_distortion_coeffs: tuple[float, ...]
    rgb_undistorted: bool
    camera_type: str
    camera_serial: str
    color_image_dim: tuple[int, int]
    depth_image_dim: tuple[int, int]
    fps: int
    depth_scale_m: float
    depth_aligned_to: str

    @classmethod
    def from_mapping(
        cls, stream_name: str, value: Mapping[str, Any]
    ) -> "CameraIntrinsics":
        required = (
            "fx",
            "fy",
            "cx",
            "cy",
            "width",
            "height",
            "distortion_model",
            "distortion_coeffs",
            "distortion_coeff_order",
            "source_distortion_model",
            "source_distortion_coeffs",
            "rgb_undistorted",
            "camera_type",
            "camera_serial",
            "color_image_dim",
            "depth_image_dim",
            "fps",
            "depth_scale_m",
            "depth_aligned_to",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise CameraCalibrationError(
                f"{stream_name} calibration is missing {', '.join(missing)}"
            )
        try:
            fx, fy, cx, cy = (
                float(value[name]) for name in ("fx", "fy", "cx", "cy")
            )
            width = int(value["width"])
            height = int(value["height"])
            coefficients = tuple(float(item) for item in value["distortion_coeffs"])
            coefficient_order = tuple(
                str(item).strip() for item in value["distortion_coeff_order"]
            )
            source_coefficients = tuple(float(item) for item in value["source_distortion_coeffs"])
            color_dim = tuple(int(item) for item in value["color_image_dim"])
            depth_dim = tuple(int(item) for item in value["depth_image_dim"])
            fps = int(value["fps"])
            depth_scale = float(value["depth_scale_m"])
        except (TypeError, ValueError) as exc:
            raise CameraCalibrationError(
                f"{stream_name} calibration contains invalid numeric values"
            ) from exc
        numeric = (fx, fy, cx, cy, depth_scale, *coefficients, *source_coefficients)
        if not all(math.isfinite(item) for item in numeric):
            raise CameraCalibrationError(
                f"{stream_name} calibration values must be finite"
            )
        if fx <= 0.0 or fy <= 0.0 or depth_scale <= 0.0:
            raise CameraCalibrationError(
                f"{stream_name} focal lengths and depth_scale_m must be positive"
            )
        if width <= 0 or height <= 0 or color_dim != (width, height):
            raise CameraCalibrationError(
                f"{stream_name} color dimensions do not match width and height"
            )
        if len(depth_dim) != 2 or min(depth_dim) <= 0:
            raise CameraCalibrationError(
                f"{stream_name} depth_image_dim must contain two positive values"
            )
        if not coefficients or not source_coefficients:
            raise CameraCalibrationError(
                f"{stream_name} distortion coefficients must not be empty"
            )
        if len(coefficient_order) != len(coefficients) or not all(coefficient_order):
            raise CameraCalibrationError(
                f"{stream_name} distortion coefficient order is invalid"
            )
        distortion_model = str(value["distortion_model"]).strip()
        source_distortion_model = str(value["source_distortion_model"]).strip()
        camera_serial = str(value["camera_serial"]).strip()
        camera_type = str(value["camera_type"]).strip()
        aligned_to = str(value["depth_aligned_to"]).strip()
        rgb_undistorted = value["rgb_undistorted"]
        if not isinstance(rgb_undistorted, bool):
            raise CameraCalibrationError(
                f"{stream_name} rgb_undistorted must be a boolean"
            )
        if not all((distortion_model, source_distortion_model, camera_type, camera_serial)):
            raise CameraCalibrationError(
                f"{stream_name} distortion models, camera type and serial are required"
            )
        if aligned_to != stream_name:
            raise CameraCalibrationError(
                f"{stream_name} depth must be aligned to {stream_name}"
            )
        if fps <= 0:
            raise CameraCalibrationError(f"{stream_name} fps must be positive")
        return cls(
            stream_name=stream_name,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            distortion_model=distortion_model,
            distortion_coeffs=coefficients,
            distortion_coeff_order=coefficient_order,
            source_distortion_model=source_distortion_model,
            source_distortion_coeffs=source_coefficients,
            rgb_undistorted=rgb_undistorted,
            camera_type=camera_type,
            camera_serial=camera_serial,
            color_image_dim=(width, height),
            depth_image_dim=(depth_dim[0], depth_dim[1]),
            fps=fps,
            depth_scale_m=depth_scale,
            depth_aligned_to=aligned_to,
        )

    def asdict(self) -> dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "distortion_model": self.distortion_model,
            "distortion_coeffs": list(self.distortion_coeffs),
            "distortion_coeff_order": list(self.distortion_coeff_order),
            "source_distortion_model": self.source_distortion_model,
            "source_distortion_coeffs": list(self.source_distortion_coeffs),
            "rgb_undistorted": self.rgb_undistorted,
            "camera_type": self.camera_type,
            "camera_serial": self.camera_serial,
            "color_image_dim": list(self.color_image_dim),
            "depth_image_dim": list(self.depth_image_dim),
            "fps": self.fps,
            "depth_scale_m": self.depth_scale_m,
            "depth_aligned_to": self.depth_aligned_to,
        }


def _validated_capture(
    message: Mapping[str, Any],
) -> tuple[dict[str, CameraIntrinsics], dict[str, tuple[np.ndarray, np.ndarray]]]:
    images = message.get("images")
    info_map = message.get("camera_info")
    if not isinstance(images, Mapping):
        raise CameraCalibrationError("capture images mapping is missing")
    if not isinstance(info_map, Mapping):
        raise CameraCalibrationError("capture camera_info mapping is missing")
    calibrations: dict[str, CameraIntrinsics] = {}
    captured_images: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for stream_name in REQUIRED_STREAMS:
        depth_name = f"{stream_name}_depth"
        if stream_name not in images:
            raise CameraCalibrationError(f"capture is missing {stream_name}")
        if depth_name not in images:
            raise CameraCalibrationError(f"capture is missing {depth_name}")
        info = info_map.get(stream_name)
        if not isinstance(info, Mapping):
            raise CameraCalibrationError(
                f"capture is missing {stream_name} camera_info"
            )
        calibration = CameraIntrinsics.from_mapping(stream_name, info)
        rgb = np.asarray(images[stream_name])
        depth = np.asarray(images[depth_name])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise CameraCalibrationError(f"{stream_name} RGB must be uint8 HxWx3")
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise CameraCalibrationError(f"{depth_name} must be uint16 HxW")
        if rgb.shape[:2] != depth.shape:
            raise CameraCalibrationError(
                f"{stream_name} RGB and aligned depth shapes differ"
            )
        if (rgb.shape[1], rgb.shape[0]) != (
            calibration.width,
            calibration.height,
        ):
            raise CameraCalibrationError(
                f"{stream_name} images do not match calibration dimensions"
            )
        calibrations[stream_name] = calibration
        captured_images[stream_name] = (
            np.ascontiguousarray(rgb),
            np.ascontiguousarray(depth),
        )
    return calibrations, captured_images


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_png(path: Path, image: np.ndarray, *, rgb: bool = False) -> None:
    value = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb else image
    if not cv2.imwrite(str(path), value):
        raise OSError(f"failed to write calibration image {path}")


def persist_calibration_capture(
    message: Mapping[str, Any],
    *,
    active_path: str | Path,
    backup_root: str | Path,
    capture_id: str,
    robot_host: str,
) -> Path:
    calibrations, images = _validated_capture(message)
    backup_dir = Path(backup_root) / capture_id
    if backup_dir.exists():
        raise FileExistsError(f"calibration backup already exists: {backup_dir}")

    document = {
        "format_version": FORMAT_VERSION,
        "capture_id": capture_id,
        "robot_host": str(robot_host),
        "streams": {
            name: calibrations[name].asdict() for name in REQUIRED_STREAMS
        },
    }

    backup_dir.mkdir(parents=True, exist_ok=False)
    for stream_name, (rgb, depth) in images.items():
        _write_png(backup_dir / f"{stream_name}_rgb.png", rgb, rgb=True)
        _write_png(backup_dir / f"{stream_name}_depth_raw.png", depth)
    _atomic_write_json(backup_dir / "camera_intrinsics.json", document)
    _atomic_write_json(Path(active_path), document)
    return backup_dir


def load_camera_intrinsics(
    path: str | Path,
) -> dict[str, CameraIntrinsics]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraCalibrationError(
            f"failed to load camera calibration from {source}"
        ) from exc
    if document.get("format_version") != FORMAT_VERSION:
        raise CameraCalibrationError("unsupported camera calibration format_version")
    streams = document.get("streams")
    if not isinstance(streams, Mapping):
        raise CameraCalibrationError("camera calibration streams mapping is missing")
    result: dict[str, CameraIntrinsics] = {}
    for stream_name in REQUIRED_STREAMS:
        value = streams.get(stream_name)
        if not isinstance(value, Mapping):
            raise CameraCalibrationError(
                f"saved calibration is missing {stream_name}"
            )
        result[stream_name] = CameraIntrinsics.from_mapping(stream_name, value)
    return result

