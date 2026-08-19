"""Head-camera multimodal base-pose adjustment policy for Unitree G1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import msgpack
import numpy as np
import zmq

from gear_sonic.camera.calibration import (
    CameraCalibrationError,
    DEFAULT_CAMERA_INTRINSICS_PATH,
    load_camera_intrinsics,
)
from gear_sonic.camera.sensor_server import ImageMessageSchema


HTTP_PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
ALL_PROXY_KEYS = ("all_proxy", "ALL_PROXY")

DEFAULT_QWENVL_PLUS_MODEL = "qwen3-vl-plus"
DEFAULT_QWENVL_BASE_URL = (
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
class BasePoseCameraError(RuntimeError):
    """Raised when a head-camera observation is unavailable or malformed."""


class BasePoseValidationError(ValueError):
    """Raised when structured YOLOE grounding output is invalid."""


@dataclass(frozen=True)
class AlignedRGBDSnapshot:
    rgb: np.ndarray
    depth_raw: np.ndarray | None
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m: float | None
    depth_aligned_to: str | None
    depth_source: str | None
    timestamp: float


@dataclass(frozen=True)
class DualRGBDCapture:
    """Independently decoded RGB-D streams from one composed-camera packet."""

    snapshots: Mapping[str, AlignedRGBDSnapshot]
    errors: Mapping[str, str]

    def require(self, stream_name: str) -> AlignedRGBDSnapshot:
        snapshot = self.snapshots.get(stream_name)
        if snapshot is not None:
            return snapshot
        raise BasePoseCameraError(
            self.errors.get(stream_name, f"camera payload requires {stream_name} RGB-D")
        )


def _atomic_write_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}_", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: Any) -> None:
    contents = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_bytes(path, contents.encode("utf-8"))


def _write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


class AlignedRGBDCamera:
    """Read a fresh RGB or aligned RGB-D frame from a composed camera stream."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        host: str,
        port: int,
        *,
        stream_name: str = "ego_view",
        require_depth: bool = False,
        required_depth_source: str | None = None,
        timeout_ms: int = 15000,
        calibration_path: str | Path | None = DEFAULT_CAMERA_INTRINSICS_PATH,
    ):
        self.stream_name = stream_name
        self.depth_key = f"{stream_name}_depth"
        self.require_depth = bool(require_depth)
        self.required_depth_source = required_depth_source
        self.timeout_ms = int(timeout_ms)
        self._configured_camera_info: dict[str, Any] | None = None
        if calibration_path is not None:
            try:
                self._configured_camera_info = load_camera_intrinsics(
                    calibration_path
                )[stream_name].asdict()
            except CameraCalibrationError as exc:
                raise BasePoseCameraError(str(exc)) from exc
        self._last_timestamp: float | None = None
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{int(port)}")

    def decode_payload(self, payload: Mapping[str, Any]) -> AlignedRGBDSnapshot:
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise BasePoseCameraError("unsupported camera schema_version")
        decoded = ImageMessageSchema.deserialize(
            dict(payload), decode_images=True
        ).asdict()
        images = decoded.get("images", {})
        info_map = decoded.get("camera_info", {})
        timestamps = decoded.get("timestamps", {})
        if self.stream_name not in images:
            raise BasePoseCameraError(f"camera payload requires {self.stream_name} RGB")
        packet_info = info_map.get(self.stream_name)
        configured_info = getattr(self, "_configured_camera_info", None)
        if isinstance(configured_info, Mapping):
            info = dict(configured_info)
            if isinstance(packet_info, Mapping) and packet_info.get("depth_source"):
                info["depth_source"] = packet_info["depth_source"]
        else:
            info = packet_info
        if not isinstance(info, Mapping):
            raise BasePoseCameraError(f"{self.stream_name} camera_info is missing")
        rgb = np.asarray(images[self.stream_name])
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise BasePoseCameraError(f"{self.stream_name} must be an RGB image")
        try:
            fx, fy, cx, cy = (float(info[name]) for name in ("fx", "fy", "cx", "cy"))
            width, height = int(info["width"]), int(info["height"])
            timestamp = float(timestamps[self.stream_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise BasePoseCameraError("head camera calibration is incomplete") from exc
        if not all(math.isfinite(value) for value in (fx, fy, cx, cy, timestamp)):
            raise BasePoseCameraError("head camera calibration must be finite")
        if fx <= 0.0 or fy <= 0.0 or (width, height) != (rgb.shape[1], rgb.shape[0]):
            raise BasePoseCameraError("head camera calibration is invalid")

        depth_raw: np.ndarray | None = None
        depth_scale_m: float | None = None
        depth_aligned_to: str | None = None
        depth_source: str | None = None
        if self.require_depth and self.depth_key in images:
            depth_raw = np.asarray(images[self.depth_key])
            if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
                raise BasePoseCameraError("aligned depth must be a uint16 image")
            if depth_raw.shape != rgb.shape[:2]:
                raise BasePoseCameraError("aligned RGB and depth shapes do not match")
            try:
                depth_scale_m = float(info["depth_scale_m"])
            except (KeyError, TypeError, ValueError) as exc:
                raise BasePoseCameraError("depth scale is missing") from exc
            depth_aligned_to = str(info.get("depth_aligned_to", ""))
            depth_source = str(info.get("depth_source", "")) or None
            if depth_scale_m <= 0.0 or not math.isfinite(depth_scale_m):
                raise BasePoseCameraError("depth scale must be finite and positive")
            if depth_aligned_to != self.stream_name:
                raise BasePoseCameraError(
                    "depth is not aligned to the requested RGB stream"
                )
            if (
                self.required_depth_source is not None
                and depth_source != self.required_depth_source
            ):
                raise BasePoseCameraError(
                    f"depth_source must be {self.required_depth_source!r}, got {depth_source!r}"
                )
            if (
                self.required_depth_source is not None
                and self.required_depth_source.startswith("depth-anything-v2-metric-")
                and not math.isclose(
                    depth_scale_m, 0.001, rel_tol=0.0, abs_tol=1.0e-9
                )
            ):
                raise BasePoseCameraError(
                    "Depth Anything uint16 depth must use millimetre units"
                )
        elif self.require_depth:
            raise BasePoseCameraError(f"camera payload requires {self.depth_key}")

        return AlignedRGBDSnapshot(
            rgb=rgb,
            depth_raw=depth_raw,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale_m=depth_scale_m,
            depth_aligned_to=depth_aligned_to,
            depth_source=depth_source,
            timestamp=timestamp,
        )

    def capture(self) -> AlignedRGBDSnapshot:
        deadline = time.monotonic() + self.timeout_ms / 1000.0
        while True:
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            if not self._socket.poll(remaining_ms):
                raise BasePoseCameraError(
                    "timed out waiting for a fresh head-camera frame"
                )
            try:
                payload = msgpack.unpackb(self._socket.recv(), raw=False)
                snapshot = self.decode_payload(payload)
            except BasePoseCameraError:
                raise
            except Exception as exc:
                raise BasePoseCameraError(
                    "failed to decode head-camera message"
                ) from exc
            if (
                self._last_timestamp is None
                or snapshot.timestamp != self._last_timestamp
            ):
                self._last_timestamp = snapshot.timestamp
                return snapshot
            if time.monotonic() >= deadline:
                raise BasePoseCameraError(
                    "timed out waiting for a newer head-camera frame"
                )

    def close(self) -> None:
        self._socket.close()
        self._context.term()


class DualAlignedRGBDCamera:
    """Read multiple aligned RGB-D streams through one composed-camera socket."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        stream_names: Sequence[str] = ("ego_view", "chest_view"),
        timeout_ms: int = 15000,
        calibration_path: str | Path | None = DEFAULT_CAMERA_INTRINSICS_PATH,
    ):
        names = tuple(str(name) for name in stream_names)
        if (
            len(names) < 2
            or len(set(names)) != len(names)
            or any(not name for name in names)
        ):
            raise ValueError("dual RGB-D stream names must be distinct and non-empty")
        self.stream_names = names
        self.timeout_ms = int(timeout_ms)
        configured: Mapping[str, Any] = {}
        if calibration_path is not None:
            try:
                configured = load_camera_intrinsics(calibration_path)
            except CameraCalibrationError as exc:
                raise BasePoseCameraError(str(exc)) from exc
        self._decoders: dict[str, AlignedRGBDCamera] = {}
        for stream_name in names:
            decoder = object.__new__(AlignedRGBDCamera)
            decoder.stream_name = stream_name
            decoder.depth_key = f"{stream_name}_depth"
            decoder.require_depth = True
            decoder.required_depth_source = None
            camera_info = configured.get(stream_name)
            decoder._configured_camera_info = (
                None if camera_info is None else camera_info.asdict()
            )
            self._decoders[stream_name] = decoder
        self._last_timestamps: dict[str, float] = {}
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{int(port)}")

    def decode_payload(self, payload: Mapping[str, Any]) -> DualRGBDCapture:
        snapshots: dict[str, AlignedRGBDSnapshot] = {}
        errors: dict[str, str] = {}
        for stream_name in self.stream_names:
            try:
                snapshots[stream_name] = self._decoders[stream_name].decode_payload(
                    payload
                )
            except BasePoseCameraError as exc:
                errors[stream_name] = str(exc)
        return DualRGBDCapture(snapshots=snapshots, errors=errors)

    def capture(self) -> DualRGBDCapture:
        deadline = time.monotonic() + self.timeout_ms / 1000.0
        while True:
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            if not self._socket.poll(remaining_ms):
                raise BasePoseCameraError(
                    "timed out waiting for a fresh dual-camera frame"
                )
            try:
                payload = msgpack.unpackb(self._socket.recv(), raw=False)
                decoded = self.decode_payload(payload)
            except BasePoseCameraError:
                raise
            except Exception as exc:
                raise BasePoseCameraError(
                    "failed to decode dual-camera message"
                ) from exc
            fresh: dict[str, AlignedRGBDSnapshot] = {}
            errors = dict(decoded.errors)
            for stream_name, snapshot in decoded.snapshots.items():
                if self._last_timestamps.get(stream_name) != snapshot.timestamp:
                    fresh[stream_name] = snapshot
                else:
                    errors[stream_name] = (
                        f"timed out waiting for newer {stream_name} frame"
                    )
            if fresh:
                self._last_timestamps.update(
                    {name: snapshot.timestamp for name, snapshot in fresh.items()}
                )
                return DualRGBDCapture(snapshots=fresh, errors=errors)
            if time.monotonic() >= deadline:
                raise BasePoseCameraError(
                    "timed out waiting for newer dual-camera frames"
                )

    def close(self) -> None:
        self._socket.close()
        self._context.term()


class CodexStructuredVisionClient:
    """Execute a read-only Codex CLI request with one or more images."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "xhigh",
        fast: bool = True,
        timeout_seconds: float = 600.0,
        codex_bin: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.fast = bool(fast)
        self.timeout_seconds = float(timeout_seconds)
        self.codex_bin = codex_bin or os.environ.get("CODEX_BIN", "codex")
        self.runner = runner
        self._authenticated = False

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        child_env = os.environ.copy()
        http_proxy = os.environ.get("BASE_POSE_CODEX_HTTP_PROXY")
        all_proxy = os.environ.get("BASE_POSE_CODEX_ALL_PROXY")
        for keys, value in ((HTTP_PROXY_KEYS, http_proxy), (ALL_PROXY_KEYS, all_proxy)):
            if value is None:
                continue
            for key in keys:
                if value:
                    child_env[key] = value
                else:
                    child_env.pop(key, None)
        return child_env

    def _check_login(self) -> None:
        if self._authenticated:
            return
        result = self.runner(
            [self.codex_bin, "login", "status"],
            capture_output=True,
            text=True,
            timeout=min(15.0, self.timeout_seconds),
            check=False,
            env=self._subprocess_env(),
        )
        login_text = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        if result.returncode != 0 or "chatgpt" not in login_text:
            raise RuntimeError(
                "Codex CLI must be logged in with a ChatGPT subscription"
            )
        self._authenticated = True

    def run(
        self,
        *,
        prompt: str,
        image_paths: Sequence[str | Path],
        schema: Mapping[str, Any],
        schema_filename: str,
        cwd: str | Path,
    ) -> dict[str, Any]:
        self._check_login()
        resolved_images = [Path(path).resolve() for path in image_paths]
        for path in resolved_images:
            if not path.is_file():
                raise FileNotFoundError(f"model image input not found: {path}")
        workdir = Path(cwd).resolve()
        schema_path = workdir / schema_filename
        _write_json(schema_path, schema)
        command = [
            self.codex_bin,
            "--ask-for-approval",
            "never",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--model",
            self.model,
            "--config",
            f'model_reasoning_effort="{self.reasoning_effort}"',
        ]
        command.extend(
            ("--config", f"features.fast_mode={str(self.fast).lower()}")
        )
        if self.fast:
            command.extend(("--config", 'service_tier="fast"'))
        for path in resolved_images:
            command.extend(("--image", str(path)))
        command.extend(("--output-schema", str(schema_path), prompt))
        result = self.runner(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
            cwd=str(workdir),
            env=self._subprocess_env(),
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "unknown error").strip()
            raise RuntimeError(f"Codex base-pose policy failed: {message}")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise BasePoseValidationError("Codex output is not valid JSON") from exc
        if not isinstance(value, dict):
            raise BasePoseValidationError("Codex output must be a JSON object")
        return value


def _dashscope_api_key(env_file: str | Path | None = None) -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    path = Path(env_file) if env_file is not None else Path(sys.prefix) / ".env"
    if path.is_file():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "DASHSCOPE_API_KEY" and value.strip():
                return value.strip().strip("\"'")
    raise RuntimeError(
        "Qwen-VL requires DASHSCOPE_API_KEY in the environment or "
        f"{path}"
    )


class QwenVLStructuredVisionClient:
    """Call Qwen-VL Plus through DashScope's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_QWENVL_PLUS_MODEL,
        base_url: str = DEFAULT_QWENVL_BASE_URL,
        timeout_seconds: float = 600.0,
        thinking_budget: int = 500,
        enable_thinking: bool = True,
        api_key: str | None = None,
        env_file: str | Path | None = None,
        client: Any | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = float(timeout_seconds)
        self.thinking_budget = int(thinking_budget)
        self.enable_thinking = bool(enable_thinking)
        self.last_reasoning_content = ""
        self.last_answer_content = ""
        if self.enable_thinking and self.thinking_budget <= 0:
            raise ValueError("Qwen-VL thinking_budget must be positive")
        if client is not None:
            self.client = client
            return
        key = api_key or _dashscope_api_key(env_file)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Qwen-VL requires the openai package in .venv_inference"
            ) from exc
        import httpx

        proxy = (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
        )
        http_client = (
            httpx.Client(proxy=proxy) if proxy else httpx.Client(trust_env=False)
        )
        self.client = OpenAI(
            api_key=key,
            base_url=base_url,
            http_client=http_client,
        )

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        text = content.strip()
        if not text:
            raise BasePoseValidationError("Qwen-VL base-pose output is empty")
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BasePoseValidationError(
                "Qwen-VL base-pose output is not valid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise BasePoseValidationError(
                "Qwen-VL base-pose output must be a JSON object"
            )
        return value

    def run(
        self,
        *,
        prompt: str,
        image_paths: Sequence[str | Path],
        schema: Mapping[str, Any],
        schema_filename: str,
        cwd: str | Path,
    ) -> dict[str, Any]:
        resolved_images = [Path(path).resolve() for path in image_paths]
        for path in resolved_images:
            if not path.is_file():
                raise FileNotFoundError(f"model image input not found: {path}")
        workdir = Path(cwd).resolve()
        _write_json(workdir / schema_filename, schema)
        content: list[dict[str, Any]] = []
        for path in resolved_images:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            )
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        image_manifest = "\n".join(
            f"ATTACHED_IMAGE_{index} is image_url item {index} above."
            for index in range(1, len(resolved_images) + 1)
        )
        content.append(
            {
                "type": "text",
                "text": (
                    f"{image_manifest}\n\n{prompt}\n\n"
                    "Return only one JSON object matching this schema:\n"
                    f"{schema_text}"
                ),
            }
        )
        extra_body: dict[str, bool | int] = {
            "enable_thinking": self.enable_thinking,
        }
        if self.enable_thinking:
            extra_body["thinking_budget"] = self.thinking_budget
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            stream=True,
            timeout=self.timeout_seconds,
            extra_body=extra_body,
        )
        reasoning_parts: list[str] = []
        answer_parts: list[str] = []
        for chunk in completion:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            answer = getattr(delta, "content", None)
            if answer:
                answer_parts.append(answer)
        self.last_reasoning_content = "".join(reasoning_parts)
        self.last_answer_content = "".join(answer_parts)
        return self._parse_json_content(self.last_answer_content)
