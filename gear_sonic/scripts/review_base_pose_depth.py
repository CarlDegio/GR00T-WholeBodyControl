#!/usr/bin/env python3
"""Inspect lossless depth samples from the latest BasePose YOLOE review log."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
import warnings
import webbrowser

import cv2
import numpy as np


DEFAULT_OUTPUT_ROOT = Path("outputs/base_pose_adjustment")
WINDOW_NAME = "BasePose YOLOE depth review"


@dataclass(frozen=True)
class ReviewSample:
    """One depth artifact and the parameters recorded for it in the frame log."""

    frame_index: int
    depth_path: Path
    rgb_path: Path | None
    depth_scale_m: float
    camera_timestamp: float
    camera_stream: str | None
    attempt_id: int | None
    failover_stage: str | None
    perception_kind: str | None
    perception_error: str | None


@dataclass(frozen=True)
class PixelReading:
    x: int
    y: int
    raw: int
    depth_m: float

    @property
    def valid(self) -> bool:
        return self.raw > 0 and math.isfinite(self.depth_m)


def find_latest_run(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Return the run whose raw frame log was modified most recently."""
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BasePose output root does not exist: {root}")
    logs = [
        path
        for path in root.rglob("raw_servo_frames.jsonl")
        if (path.parent / "review_samples" / "depth").is_dir()
    ]
    if not logs:
        raise FileNotFoundError(
            f"no BasePose review log with saved depth images found under {root}"
        )
    return max(logs, key=lambda path: path.stat().st_mtime_ns).parent


def _resolve_artifact(run_dir: Path, relative: object, *, field: str) -> Path | None:
    if relative is None:
        return None
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid {field} artifact path in frame log: {relative!r}")
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"{field} artifact escapes run directory: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"logged {field} artifact does not exist: {path}")
    return path


def load_review_samples(run_dir: str | Path) -> list[ReviewSample]:
    """Index sampled depths, taking the unit scale only from raw_servo_frames.jsonl."""
    root = Path(run_dir).expanduser().resolve()
    log_path = root / "raw_servo_frames.jsonl"
    if not log_path.is_file():
        raise FileNotFoundError(f"BasePose frame log does not exist: {log_path}")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    samples: list[ReviewSample] = []
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_index == len(lines) - 1:
                warnings.warn(
                    f"ignoring an incomplete final line in active log {log_path}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            raise ValueError(f"invalid JSON at {log_path}:{line_index + 1}") from exc

        artifacts = record.get("review_artifacts")
        if not isinstance(artifacts, dict) or not artifacts.get("raw_depth"):
            continue
        scale = artifacts.get("depth_scale_m")
        if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or scale <= 0:
            raise ValueError(
                f"frame {record.get('frame_index')} has saved depth but no valid "
                "depth_scale_m in raw_servo_frames.jsonl"
            )
        depth_path = _resolve_artifact(root, artifacts["raw_depth"], field="depth")
        assert depth_path is not None
        samples.append(
            ReviewSample(
                frame_index=int(record["frame_index"]),
                depth_path=depth_path,
                rgb_path=_resolve_artifact(root, artifacts.get("raw_rgb"), field="RGB"),
                depth_scale_m=float(scale),
                camera_timestamp=float(record["camera_timestamp"]),
                camera_stream=(
                    None if record.get("camera_stream") is None else str(record["camera_stream"])
                ),
                attempt_id=(
                    None if record.get("attempt_id") is None else int(record["attempt_id"])
                ),
                failover_stage=(
                    None if record.get("failover_stage") is None else str(record["failover_stage"])
                ),
                perception_kind=(
                    None if record.get("perception_kind") is None else str(record["perception_kind"])
                ),
                perception_error=(
                    None if record.get("perception_error") is None else str(record["perception_error"])
                ),
            )
        )
    if not samples:
        raise ValueError(f"no sampled depth artifacts are recorded in {log_path}")
    samples.sort(key=lambda sample: sample.frame_index)
    return samples


def load_depth(path: str | Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise OSError(f"OpenCV could not read depth image: {path}")
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(
            f"review depth must be a 2-D uint16 PNG, got shape={depth.shape} dtype={depth.dtype}"
        )
    return depth


def read_pixel(depth: np.ndarray, *, x: int, y: int, depth_scale_m: float) -> PixelReading:
    height, width = depth.shape
    if not (0 <= x < width and 0 <= y < height):
        raise IndexError(f"pixel ({x}, {y}) is outside {width}x{height} depth image")
    raw = int(depth[y, x])
    return PixelReading(x=x, y=y, raw=raw, depth_m=raw * depth_scale_m)


def auto_color_limits(
    depth: np.ndarray,
    *,
    depth_scale_m: float,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[float, float]:
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("color percentiles must satisfy 0 <= lower < upper <= 100")
    valid = depth[depth > 0].astype(np.float64) * depth_scale_m
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return 0.0, max(depth_scale_m, 1.0)
    low, high = np.percentile(valid, (lower_percentile, upper_percentile))
    if high <= low:
        high = low + max(depth_scale_m, 1.0e-6)
    return float(low), float(high)


def colorize_depth(
    depth: np.ndarray,
    *,
    depth_scale_m: float,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    if not math.isfinite(min_depth_m) or not math.isfinite(max_depth_m):
        raise ValueError("color depth limits must be finite")
    if max_depth_m <= min_depth_m:
        raise ValueError("max color depth must be greater than min color depth")
    depth_m = depth.astype(np.float32) * float(depth_scale_m)
    normalized = np.clip(
        (depth_m - float(min_depth_m)) / (float(max_depth_m) - float(min_depth_m)),
        0.0,
        1.0,
    )
    colors = cv2.applyColorMap(np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    colors[depth == 0] = (24, 24, 24)
    return colors


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.52,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


_BROWSER_VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BasePose YOLOE depth review</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: #111318;
      color: #eef1f7;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 760px; background: #111318; }
    header {
      position: sticky; top: 0; z-index: 2; padding: 13px 18px 11px;
      background: rgba(18, 21, 27, .96); border-bottom: 1px solid #343a46;
    }
    h1 { margin: 0 0 7px; font-size: 20px; letter-spacing: .2px; }
    .toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    button {
      border: 1px solid #505969; border-radius: 6px; background: #272d38;
      color: #f3f5f8; padding: 6px 12px; cursor: pointer; font-weight: 600;
    }
    button:hover { background: #353d4a; }
    input[type=range] { width: min(440px, 40vw); accent-color: #44b7ff; }
    .tag {
      padding: 5px 9px; border-radius: 5px; background: #222833;
      border: 1px solid #3b4452; font: 13px ui-monospace, monospace;
    }
    #app { padding: 16px; }
    .layout { display: grid; grid-template-columns: minmax(640px, 1fr) auto; gap: 18px; }
    .card { background: #191d24; border: 1px solid #343a46; border-radius: 9px; padding: 12px; }
    #depthCanvas {
      display: block; width: 100%; max-width: 960px; height: auto; margin: auto;
      image-rendering: pixelated; background: #181818; cursor: crosshair;
      border: 1px solid #616a79;
    }
    .inspector { width: max-content; }
    #gridCanvas { display: block; background: #17191e; border: 1px solid #505969; }
    .selected {
      margin: 0 0 10px; padding: 9px 10px; min-height: 62px; border-radius: 6px;
      background: #10151c; border: 1px solid #374352; font: 14px/1.55 ui-monospace, monospace;
      white-space: pre-line;
    }
    .meta { margin-top: 10px; font: 13px/1.55 ui-monospace, monospace; color: #c7ced9; }
    .hint { color: #aab3c1; font-size: 13px; margin-top: 9px; }
    .color-row { display: flex; gap: 10px; align-items: center; margin: 9px 0 2px; }
    .colorbar {
      height: 13px; flex: 1; border: 1px solid #697383;
      background: linear-gradient(90deg, #30123b, #466be3, #1bcfd4, #6dfd71, #f9e721, #f47714, #7a0403);
    }
    .muted { color: #98a2b2; }
    .error { color: #ffb178; }
    #status { color: #8ed9ff; }
    @media (max-width: 1180px) {
      .layout { grid-template-columns: 1fr; }
      .inspector { width: 100%; overflow-x: auto; }
    }
  </style>
</head>
<body>
  <header>
    <h1>BasePose YOLOE depth review</h1>
    <div class="toolbar">
      <button id="previous" title="Previous sample (A or Left Arrow)">◀ Previous</button>
      <input id="frameSlider" type="range" min="0" max="0" value="0">
      <button id="next" title="Next sample (D or Right Arrow)">Next ▶</button>
      <button id="blend" title="Toggle RGB context (B)">RGB blend: off</button>
      <button id="reload" title="Reload a log that is still being written">Reload log</button>
      <span id="frameTag" class="tag">loading…</span>
      <span id="status">connecting…</span>
    </div>
  </header>
  <main id="app">
    <div class="layout">
      <section class="card">
        <canvas id="depthCanvas"></canvas>
        <div class="color-row">
          <span id="colorMin" class="tag">-</span>
          <div class="colorbar"></div>
          <span id="colorMax" class="tag">-</span>
        </div>
        <div id="source" class="meta"></div>
        <div class="hint">
          Hover to inspect. Left click pins a pixel; right click releases it. Raw zero is invalid.
        </div>
      </section>
      <aside class="card inspector">
        <div id="selected" class="selected">Move over the depth map…</div>
        <canvas id="gridCanvas"></canvas>
        <div id="metadata" class="meta"></div>
      </aside>
    </div>
  </main>
  <script>
    "use strict";
    const ui = {
      depth: document.getElementById("depthCanvas"), grid: document.getElementById("gridCanvas"),
      selected: document.getElementById("selected"), metadata: document.getElementById("metadata"),
      slider: document.getElementById("frameSlider"), previous: document.getElementById("previous"),
      next: document.getElementById("next"), blend: document.getElementById("blend"),
      reload: document.getElementById("reload"), frameTag: document.getElementById("frameTag"),
      status: document.getElementById("status"), source: document.getElementById("source"),
      colorMin: document.getElementById("colorMin"), colorMax: document.getElementById("colorMax"),
    };
    const state = {
      manifest: null, position: 0, frame: null, rawBytes: null, cursor: null,
      pinned: false, blend: false, colorImage: null, rgbImage: null, requestSerial: 0,
      colorCanvas: document.createElement("canvas"), explicitInitialApplied: false,
    };
    const depthCtx = ui.depth.getContext("2d");
    const gridCtx = ui.grid.getContext("2d");
    const colorCtx = state.colorCanvas.getContext("2d", {willReadFrequently: true});

    function decodeBytes(encoded) {
      const binary = atob(encoded);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; ++i) bytes[i] = binary.charCodeAt(i);
      return bytes;
    }
    function rawAt(x, y) {
      if (!state.frame || x < 0 || y < 0 || x >= state.frame.width || y >= state.frame.height) return null;
      const offset = 2 * (y * state.frame.width + x);
      return state.rawBytes[offset] | (state.rawBytes[offset + 1] << 8);
    }
    function imageFromBase64(encoded) {
      return new Promise((resolve, reject) => {
        if (!encoded) { resolve(null); return; }
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("could not decode server image"));
        image.src = "data:image/png;base64," + encoded;
      });
    }
    function pointerPixel(event) {
      const rect = ui.depth.getBoundingClientRect();
      const x = Math.floor((event.clientX - rect.left) * ui.depth.width / rect.width);
      const y = Math.floor((event.clientY - rect.top) * ui.depth.height / rect.height);
      return {x: Math.max(0, Math.min(ui.depth.width - 1, x)), y: Math.max(0, Math.min(ui.depth.height - 1, y))};
    }
    function renderMain() {
      if (!state.colorImage) return;
      depthCtx.clearRect(0, 0, ui.depth.width, ui.depth.height);
      if (state.blend && state.rgbImage) {
        depthCtx.drawImage(state.rgbImage, 0, 0);
        depthCtx.globalAlpha = 0.62;
        depthCtx.drawImage(state.colorImage, 0, 0);
        depthCtx.globalAlpha = 1;
      } else {
        depthCtx.drawImage(state.colorImage, 0, 0);
      }
      if (state.cursor) {
        const {x, y} = state.cursor;
        depthCtx.save();
        depthCtx.strokeStyle = "#fff"; depthCtx.lineWidth = 1;
        depthCtx.beginPath();
        depthCtx.moveTo(x - 9, y + .5); depthCtx.lineTo(x + 10, y + .5);
        depthCtx.moveTo(x + .5, y - 9); depthCtx.lineTo(x + .5, y + 10);
        depthCtx.stroke(); depthCtx.restore();
      }
    }
    function renderInspector() {
      if (!state.frame || !state.cursor) return;
      const {x: cx, y: cy} = state.cursor;
      const raw = rawAt(cx, cy);
      const depthM = raw * state.frame.depth_scale_m;
      const validity = raw === 0 ? "INVALID (raw zero)" : "valid";
      ui.selected.textContent =
        `SELECTED  x=${cx}  y=${cy}\n` +
        `raw=${raw}  ${depthM * 1000} mm  ${depthM.toFixed(6)} m  ${validity}` +
        `${state.pinned ? "  [PINNED]" : ""}`;
      const radius = state.manifest.zoom_radius, cell = state.manifest.cell_size;
      const diameter = 2 * radius + 1;
      ui.grid.width = diameter * cell; ui.grid.height = diameter * cell;
      gridCtx.font = `${Math.max(10, Math.floor(cell * .24))}px ui-monospace, monospace`;
      gridCtx.textBaseline = "middle";
      for (let row = 0; row < diameter; ++row) {
        for (let column = 0; column < diameter; ++column) {
          const x = cx + column - radius, y = cy + row - radius;
          const left = column * cell, top = row * cell;
          const value = rawAt(x, y);
          if (value === null) {
            gridCtx.fillStyle = "#14161a"; gridCtx.fillRect(left, top, cell, cell);
          } else {
            const rgba = colorCtx.getImageData(x, y, 1, 1).data;
            gridCtx.fillStyle = `rgb(${rgba[0]},${rgba[1]},${rgba[2]})`;
            gridCtx.fillRect(left, top, cell, cell);
            const light = .299 * rgba[0] + .587 * rgba[1] + .114 * rgba[2];
            gridCtx.fillStyle = light > 145 ? "#090909" : "#fff";
            gridCtx.fillText(String(value), left + 4, top + cell * .36);
            const label = value === 0 ? "invalid" : `${(value * state.frame.depth_scale_m).toFixed(3)}m`;
            gridCtx.fillText(label, left + 4, top + cell * .72);
          }
          gridCtx.strokeStyle = row === radius && column === radius ? "#fff" : "#555d69";
          gridCtx.lineWidth = row === radius && column === radius ? 3 : 1;
          gridCtx.strokeRect(left + .5, top + .5, cell - 1, cell - 1);
        }
      }
    }
    async function loadFrame(position) {
      if (!state.manifest || !state.manifest.samples.length) return;
      state.position = Math.max(0, Math.min(state.manifest.samples.length - 1, position));
      const summary = state.manifest.samples[state.position];
      ui.slider.value = state.position; ui.status.textContent = "loading frame…";
      const serial = ++state.requestSerial;
      try {
        const response = await fetch(`/api/frame/${summary.frame_index}`, {cache: "no-store"});
        if (!response.ok) throw new Error((await response.json()).error || response.statusText);
        const frame = await response.json();
        const [colorImage, rgbImage] = await Promise.all([
          imageFromBase64(frame.color_png_base64), imageFromBase64(frame.rgb_png_base64),
        ]);
        if (serial !== state.requestSerial) return;
        state.frame = frame; state.rawBytes = decodeBytes(frame.depth_u16_le_base64);
        state.colorImage = colorImage; state.rgbImage = rgbImage;
        ui.depth.width = frame.width; ui.depth.height = frame.height;
        state.colorCanvas.width = frame.width; state.colorCanvas.height = frame.height;
        colorCtx.drawImage(colorImage, 0, 0);
        if (!state.cursor) state.cursor = {x: Math.floor(frame.width / 2), y: Math.floor(frame.height / 2)};
        state.cursor.x = Math.min(state.cursor.x, frame.width - 1);
        state.cursor.y = Math.min(state.cursor.y, frame.height - 1);
        ui.frameTag.textContent =
          `frame ${frame.frame_index}  (${state.position + 1}/${state.manifest.samples.length})  ` +
          `${frame.camera_stream || "unknown"}`;
        ui.colorMin.textContent = `${frame.color_min_m.toFixed(3)} m`;
        ui.colorMax.textContent = `${frame.color_max_m.toFixed(3)} m`;
        ui.source.textContent =
          `run: ${state.manifest.run_name} / ${frame.depth_filename} | ` +
          `log depth_scale_m=${frame.depth_scale_m} ` +
          `(1 raw unit = ${frame.depth_scale_m * 1000} mm)`;
        const error = frame.perception_error ? `\nerror: ${frame.perception_error}` : "";
        ui.metadata.textContent =
          `attempt=${frame.attempt_id ?? "-"}  stage=${frame.failover_stage ?? "-"}\n` +
          `perception=${frame.perception_kind ?? "-"}  ` +
          `valid=${(frame.valid_ratio * 100).toFixed(2)}%${error}`;
        ui.metadata.className = "meta" + (frame.perception_error ? " error" : "");
        ui.status.textContent = state.pinned ? "pixel pinned" : "live hover";
        renderMain(); renderInspector();
      } catch (error) {
        ui.status.textContent = `error: ${error.message}`;
      }
    }
    async function refreshManifest(initial = false) {
      const old = state.manifest;
      const oldFrame = state.frame ? state.frame.frame_index : null;
      try {
        const response = await fetch("/api/manifest", {cache: "no-store"});
        if (!response.ok) throw new Error((await response.json()).error || response.statusText);
        state.manifest = await response.json();
        ui.slider.max = Math.max(0, state.manifest.samples.length - 1);
        let target = state.position;
        if (initial) {
          target = state.manifest.samples.findIndex(
            item => item.frame_index === state.manifest.initial_frame_index
          );
          if (target < 0) target = state.manifest.samples.length - 1;
        } else if (oldFrame !== null) {
          target = state.manifest.samples.findIndex(item => item.frame_index === oldFrame);
          if (target < 0) target = state.manifest.samples.length - 1;
          if (state.manifest.follow_latest && old && state.position === old.samples.length - 1) {
            target = state.manifest.samples.length - 1;
          }
        }
        const newLatest = old && old.samples.length !== state.manifest.samples.length &&
          state.manifest.follow_latest;
        if (initial || target !== state.position || newLatest) {
          await loadFrame(target);
        }
      } catch (error) {
        ui.status.textContent = `log reload failed: ${error.message}`;
      }
    }
    ui.depth.addEventListener("mousemove", event => {
      if (state.pinned || !state.frame) return;
      state.cursor = pointerPixel(event); renderMain(); renderInspector();
    });
    ui.depth.addEventListener("click", event => {
      state.cursor = pointerPixel(event); state.pinned = true; ui.status.textContent = "pixel pinned";
      renderMain(); renderInspector();
    });
    ui.depth.addEventListener("contextmenu", event => {
      event.preventDefault(); state.pinned = false; ui.status.textContent = "live hover"; renderInspector();
    });
    ui.previous.addEventListener("click", () => loadFrame(state.position - 1));
    ui.next.addEventListener("click", () => loadFrame(state.position + 1));
    ui.slider.addEventListener("input", () => loadFrame(Number(ui.slider.value)));
    ui.blend.addEventListener("click", () => {
      state.blend = !state.blend; ui.blend.textContent = `RGB blend: ${state.blend ? "on" : "off"}`; renderMain();
    });
    ui.reload.addEventListener("click", () => refreshManifest(false));
    window.addEventListener("keydown", event => {
      if (["ArrowLeft", "a", "A", "p", "P"].includes(event.key)) loadFrame(state.position - 1);
      if (["ArrowRight", "d", "D", "n", "N"].includes(event.key)) loadFrame(state.position + 1);
      if (["b", "B"].includes(event.key)) ui.blend.click();
    });
    refreshManifest(true);
    setInterval(() => refreshManifest(false), 3000);
  </script>
</body>
</html>
"""


class BrowserDepthReviewApp:
    """Serve an exact-depth browser UI without relying on OpenCV HighGUI."""

    def __init__(
        self,
        run_dir: Path,
        *,
        initial_frame_index: int,
        follow_latest: bool,
        zoom_radius: int,
        cell_size: int,
        lower_percentile: float,
        upper_percentile: float,
        min_depth_m: float | None,
        max_depth_m: float | None,
    ):
        self.run_dir = run_dir
        self.initial_frame_index = initial_frame_index
        self.follow_latest = follow_latest
        self.zoom_radius = zoom_radius
        self.cell_size = cell_size
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.fixed_min_depth_m = min_depth_m
        self.fixed_max_depth_m = max_depth_m

    def _samples(self) -> list[ReviewSample]:
        return load_review_samples(self.run_dir)

    def manifest(self) -> dict[str, Any]:
        samples = self._samples()
        return {
            "run_name": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "initial_frame_index": self.initial_frame_index,
            "follow_latest": self.follow_latest,
            "zoom_radius": self.zoom_radius,
            "cell_size": self.cell_size,
            "samples": [
                {
                    "frame_index": sample.frame_index,
                    "camera_stream": sample.camera_stream,
                    "depth_scale_m": sample.depth_scale_m,
                }
                for sample in samples
            ],
        }

    def frame_payload(self, frame_index: int) -> dict[str, Any]:
        samples = self._samples()
        try:
            sample = next(item for item in samples if item.frame_index == frame_index)
        except StopIteration as exc:
            raise KeyError(f"frame {frame_index} is not a saved review sample") from exc
        depth = load_depth(sample.depth_path)
        auto_min, auto_max = auto_color_limits(
            depth,
            depth_scale_m=sample.depth_scale_m,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
        )
        min_depth_m = auto_min if self.fixed_min_depth_m is None else self.fixed_min_depth_m
        max_depth_m = auto_max if self.fixed_max_depth_m is None else self.fixed_max_depth_m
        color = colorize_depth(
            depth,
            depth_scale_m=sample.depth_scale_m,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        encoded_ok, encoded_color = cv2.imencode(".png", color)
        if not encoded_ok:
            raise OSError(f"could not encode color depth for frame {frame_index}")
        rgb_encoded = None
        if sample.rgb_path is not None:
            rgb_encoded = base64.b64encode(sample.rgb_path.read_bytes()).decode("ascii")
        valid = depth > 0
        height, width = depth.shape
        return {
            "frame_index": sample.frame_index,
            "width": width,
            "height": height,
            "depth_filename": sample.depth_path.name,
            "depth_scale_m": sample.depth_scale_m,
            "camera_timestamp": sample.camera_timestamp,
            "camera_stream": sample.camera_stream,
            "attempt_id": sample.attempt_id,
            "failover_stage": sample.failover_stage,
            "perception_kind": sample.perception_kind,
            "perception_error": sample.perception_error,
            "valid_ratio": float(valid.mean()),
            "color_min_m": min_depth_m,
            "color_max_m": max_depth_m,
            "depth_u16_le_base64": base64.b64encode(
                depth.astype("<u2", copy=False).tobytes(order="C")
            ).decode("ascii"),
            "color_png_base64": base64.b64encode(encoded_color.tobytes()).decode("ascii"),
            "rgb_png_base64": rgb_encoded,
        }


class _DepthReviewRequestHandler(BaseHTTPRequestHandler):
    app: BrowserDepthReviewApp

    def _send_bytes(self, payload: bytes, *, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, value: Any, *, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send_bytes(payload, content_type="application/json; charset=utf-8", status=status)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/":
                self._send_bytes(
                    _BROWSER_VIEWER_HTML.encode("utf-8"),
                    content_type="text/html; charset=utf-8",
                )
                return
            if path == "/api/manifest":
                self._send_json(self.app.manifest())
                return
            if path.startswith("/api/frame/"):
                frame_index = int(path.removeprefix("/api/frame/"))
                self._send_json(self.app.frame_payload(frame_index))
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, OSError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def run_browser_viewer(
    app: BrowserDepthReviewApp,
    *,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    handler = type("DepthReviewRequestHandler", (_DepthReviewRequestHandler,), {"app": app})
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        if port == 0:
            raise
        print(f"[DepthReview] port {port} unavailable ({exc}); selecting a free port")
        server = ThreadingHTTPServer((host, 0), handler)
    actual_port = int(server.server_address[1])
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{display_host}:{actual_port}/"
    print(f"[DepthReview] browser viewer: {url}")
    print("[DepthReview] press Ctrl-C in this terminal to stop the viewer")
    if open_browser:
        try:
            opened = webbrowser.open(url)
        except webbrowser.Error:
            opened = False
        if not opened:
            print("[DepthReview] browser did not open automatically; open the URL above manually")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n[DepthReview] viewer stopped")
    finally:
        server.server_close()


class DepthReviewViewer:
    """OpenCV viewer with exact hover readings and a labeled local pixel grid."""

    def __init__(
        self,
        run_dir: Path,
        samples: list[ReviewSample],
        *,
        initial_index: int,
        zoom_radius: int,
        cell_size: int,
        lower_percentile: float,
        upper_percentile: float,
        min_depth_m: float | None,
        max_depth_m: float | None,
    ):
        if zoom_radius < 1:
            raise ValueError("zoom_radius must be at least 1")
        if cell_size < 38:
            raise ValueError("cell_size must be at least 38 so labels remain readable")
        self.run_dir = run_dir
        self.samples = samples
        self.sample_index = initial_index
        self.zoom_radius = zoom_radius
        self.cell_size = cell_size
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.fixed_min_depth_m = min_depth_m
        self.fixed_max_depth_m = max_depth_m
        self.blend_rgb = False
        self.pinned = False
        self.depth = np.empty((0, 0), dtype=np.uint16)
        self.depth_color = np.empty((0, 0, 3), dtype=np.uint8)
        self.rgb: np.ndarray | None = None
        self.cursor = (0, 0)
        self.min_depth_m = 0.0
        self.max_depth_m = 1.0
        self.image_origin = (16, 100)
        self._load_sample(initial=True)

    @property
    def sample(self) -> ReviewSample:
        return self.samples[self.sample_index]

    def _load_sample(self, *, initial: bool = False) -> None:
        previous = self.cursor
        self.depth = load_depth(self.sample.depth_path)
        self.rgb = None
        if self.sample.rgb_path is not None:
            candidate = cv2.imread(str(self.sample.rgb_path), cv2.IMREAD_COLOR)
            if candidate is not None and candidate.shape[:2] == self.depth.shape:
                self.rgb = candidate
        auto_min, auto_max = auto_color_limits(
            self.depth,
            depth_scale_m=self.sample.depth_scale_m,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
        )
        self.min_depth_m = auto_min if self.fixed_min_depth_m is None else self.fixed_min_depth_m
        self.max_depth_m = auto_max if self.fixed_max_depth_m is None else self.fixed_max_depth_m
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("the effective maximum color depth must exceed the minimum")
        self.depth_color = colorize_depth(
            self.depth,
            depth_scale_m=self.sample.depth_scale_m,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
        )
        height, width = self.depth.shape
        if initial:
            self.cursor = (width // 2, height // 2)
        else:
            self.cursor = (
                min(max(previous[0], 0), width - 1),
                min(max(previous[1], 0), height - 1),
            )

    def _main_image(self) -> np.ndarray:
        if not self.blend_rgb or self.rgb is None:
            return self.depth_color.copy()
        result = cv2.addWeighted(self.rgb, 0.42, self.depth_color, 0.58, 0.0)
        result[self.depth == 0] = self.rgb[self.depth == 0]
        return result

    def _draw_legend(self, canvas: np.ndarray, *, x: int, y: int, height: int) -> None:
        gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
        legend = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
        canvas[y : y + height, x : x + 22] = np.repeat(legend, 22, axis=1)
        cv2.rectangle(canvas, (x, y), (x + 21, y + height - 1), (220, 220, 220), 1)
        _put_text(canvas, f"{self.max_depth_m:.3f} m", (x + 29, y + 12), scale=0.40)
        _put_text(canvas, f"{self.min_depth_m:.3f} m", (x + 29, y + height), scale=0.40)
        _put_text(canvas, "invalid = dark", (x + 29, y + height // 2), scale=0.40)

    def _draw_pixel_grid(self, canvas: np.ndarray, *, origin_x: int, origin_y: int) -> None:
        diameter = 2 * self.zoom_radius + 1
        cursor_x, cursor_y = self.cursor
        _put_text(canvas, "LOCAL PIXELS (raw units / metres)", (origin_x, origin_y - 36), scale=0.48)
        reading = read_pixel(
            self.depth,
            x=cursor_x,
            y=cursor_y,
            depth_scale_m=self.sample.depth_scale_m,
        )
        validity = "valid" if reading.valid else "INVALID (raw zero)"
        _put_text(
            canvas,
            f"SELECTED x={cursor_x} y={cursor_y}  raw={reading.raw}  "
            f"{reading.depth_m * 1000.0:.1f} mm  {reading.depth_m:.6f} m  {validity}",
            (origin_x, origin_y - 12),
            scale=0.45,
            color=(255, 255, 255),
        )
        height, width = self.depth.shape
        for row in range(diameter):
            y = cursor_y + row - self.zoom_radius
            for column in range(diameter):
                x = cursor_x + column - self.zoom_radius
                left = origin_x + column * self.cell_size
                top = origin_y + row * self.cell_size
                right = left + self.cell_size - 1
                bottom = top + self.cell_size - 1
                if 0 <= x < width and 0 <= y < height:
                    color = tuple(int(value) for value in self.depth_color[y, x])
                    cv2.rectangle(canvas, (left, top), (right, bottom), color, -1)
                    raw = int(self.depth[y, x])
                    depth_m = raw * self.sample.depth_scale_m
                    foreground = (255, 255, 255) if sum(color) < 390 else (10, 10, 10)
                    _put_text(
                        canvas,
                        str(raw),
                        (left + 3, top + self.cell_size // 2 - 2),
                        scale=0.35,
                        color=foreground,
                    )
                    label = "invalid" if raw == 0 else f"{depth_m:.3f}m"
                    _put_text(
                        canvas,
                        label,
                        (left + 3, top + self.cell_size - 7),
                        scale=0.29,
                        color=foreground,
                    )
                else:
                    cv2.rectangle(canvas, (left, top), (right, bottom), (18, 18, 18), -1)
                border = (255, 255, 255) if row == column == self.zoom_radius else (75, 75, 75)
                thickness = 2 if row == column == self.zoom_radius else 1
                cv2.rectangle(canvas, (left, top), (right, bottom), border, thickness)

    def render(self) -> np.ndarray:
        height, width = self.depth.shape
        grid_size = (2 * self.zoom_radius + 1) * self.cell_size
        panel_width = max(grid_size, 500)
        header_height = self.image_origin[1]
        footer_height = 50
        legend_width = 122
        canvas_height = max(header_height + height + footer_height, header_height + grid_size + 120)
        canvas_width = 16 + width + legend_width + 18 + panel_width + 16
        canvas = np.full((canvas_height, canvas_width, 3), 30, dtype=np.uint8)

        stream = self.sample.camera_stream or "unknown stream"
        _put_text(
            canvas,
            f"BasePose YOLOE depth | frame {self.sample.frame_index} "
            f"({self.sample_index + 1}/{len(self.samples)}) | {stream}",
            (16, 27),
            scale=0.68,
            color=(255, 255, 255),
            thickness=2,
        )
        _put_text(
            canvas,
            f"LOG PARAMETER: depth_scale_m={self.sample.depth_scale_m:g} "
            f"(1 raw unit = {self.sample.depth_scale_m * 1000.0:g} mm) | "
            f"color P{self.lower_percentile:g}-P{self.upper_percentile:g}",
            (16, 53),
            scale=0.49,
            color=(170, 235, 255),
        )
        pin_state = "PINNED (right click to release)" if self.pinned else "live hover"
        _put_text(
            canvas,
            f"Mouse: inspect / left click pin / right click release | "
            f"A,D or arrows: frames | B: RGB blend | Q/Esc: quit | {pin_state}",
            (16, 78),
            scale=0.45,
        )

        image_x, image_y = self.image_origin
        main_image = self._main_image()
        cursor_x, cursor_y = self.cursor
        cv2.drawMarker(
            main_image,
            (cursor_x, cursor_y),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            17,
            1,
            cv2.LINE_AA,
        )
        canvas[image_y : image_y + height, image_x : image_x + width] = main_image
        cv2.rectangle(
            canvas,
            (image_x - 1, image_y - 1),
            (image_x + width, image_y + height),
            (180, 180, 180),
            1,
        )
        legend_x = image_x + width + 10
        self._draw_legend(canvas, x=legend_x, y=image_y, height=height)

        panel_x = image_x + width + legend_width + 18
        self._draw_pixel_grid(canvas, origin_x=panel_x, origin_y=image_y + 50)
        metadata_y = image_y + grid_size + 78
        _put_text(
            canvas,
            f"attempt={self.sample.attempt_id}  stage={self.sample.failover_stage}  "
            f"perception={self.sample.perception_kind}",
            (panel_x, metadata_y),
            scale=0.43,
        )
        if self.sample.perception_error:
            _put_text(
                canvas,
                f"error: {self.sample.perception_error[:72]}",
                (panel_x, metadata_y + 23),
                scale=0.40,
                color=(120, 190, 255),
            )
        _put_text(
            canvas,
            f"run: {self.run_dir.name} | {self.sample.depth_path.name}",
            (16, image_y + height + 28),
            scale=0.43,
            color=(190, 190, 190),
        )
        return canvas

    def _mouse_callback(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        image_x, image_y = self.image_origin
        height, width = self.depth.shape
        inside = image_x <= x < image_x + width and image_y <= y < image_y + height
        if event == cv2.EVENT_RBUTTONDOWN:
            self.pinned = False
        elif event == cv2.EVENT_LBUTTONDOWN and inside:
            self.cursor = (x - image_x, y - image_y)
            self.pinned = True
        elif event == cv2.EVENT_MOUSEMOVE and inside and not self.pinned:
            self.cursor = (x - image_x, y - image_y)

    def _change_frame(self, offset: int) -> None:
        target = min(max(self.sample_index + offset, 0), len(self.samples) - 1)
        if target != self.sample_index:
            self.sample_index = target
            self._load_sample()

    def run(self) -> None:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV could not create a window; use --point X Y for headless inspection"
            ) from exc
        previous_keys = {ord("a"), ord("A"), ord("p"), ord("P"), ord(","), 81, 65361, 2424832}
        next_keys = {ord("d"), ord("D"), ord("n"), ord("N"), ord("."), 83, 65363, 2555904}
        try:
            while True:
                cv2.imshow(WINDOW_NAME, self.render())
                key = cv2.waitKeyEx(20)
                if key in (27, ord("q"), ord("Q")):
                    return
                if key in previous_keys:
                    self._change_frame(-1)
                elif key in next_keys:
                    self._change_frame(1)
                elif key in (ord("b"), ord("B")) and self.rgb is not None:
                    self.blend_rgb = not self.blend_rgb
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        return
                except cv2.error:
                    return
        finally:
            cv2.destroyWindow(WINDOW_NAME)


def _select_sample_index(samples: list[ReviewSample], frame_index: int | None) -> int:
    if frame_index is None:
        return len(samples) - 1
    for index, sample in enumerate(samples):
        if sample.frame_index == frame_index:
            return index
    available = ", ".join(str(sample.frame_index) for sample in samples)
    raise ValueError(f"frame {frame_index} is not a saved review sample; available: {available}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively inspect lossless BasePose YOLOE review depth PNGs. "
            "With no run_dir, the most recently modified raw_servo_frames.jsonl is used."
        )
    )
    parser.add_argument("run_dir", nargs="?", type=Path, help="BasePose run directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"run discovery root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--frame-index", type=int, help="initial saved frame (default: latest)")
    parser.add_argument(
        "--point",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="print one exact pixel value and exit without opening a GUI",
    )
    parser.add_argument("--list", action="store_true", help="list sampled frames and exit")
    parser.add_argument("--zoom-radius", type=int, default=4, help="local grid radius (default: 4)")
    parser.add_argument("--cell-size", type=int, default=50, help="local grid cell pixels (default: 50)")
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--min-depth-m", type=float, help="optional fixed color-map lower limit")
    parser.add_argument("--max-depth-m", type=float, help="optional fixed color-map upper limit")
    parser.add_argument(
        "--ui",
        choices=("browser", "opencv"),
        default="browser",
        help="interactive frontend (default: browser; opencv requires GTK/HighGUI)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="browser viewer listen host")
    parser.add_argument("--port", type=int, default=8765, help="browser viewer listen port")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="ask the remote machine to open the browser URL automatically",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run_dir = (
        find_latest_run(args.output_root)
        if args.run_dir is None
        else args.run_dir.expanduser().resolve()
    )
    samples = load_review_samples(run_dir)
    selected_index = _select_sample_index(samples, args.frame_index)
    sample = samples[selected_index]
    print(f"[DepthReview] run: {run_dir}")
    print(f"[DepthReview] log: {run_dir / 'raw_servo_frames.jsonl'}")
    print(
        f"[DepthReview] {len(samples)} saved frames; selected frame={sample.frame_index} "
        f"stream={sample.camera_stream} depth_scale_m={sample.depth_scale_m:g}"
    )
    if args.list:
        for item in samples:
            print(
                f"frame={item.frame_index:06d} stream={item.camera_stream or '-':10s} "
                f"scale_m={item.depth_scale_m:g} file={item.depth_path.name}"
            )
        return
    if args.point is not None:
        depth = load_depth(sample.depth_path)
        reading = read_pixel(
            depth,
            x=args.point[0],
            y=args.point[1],
            depth_scale_m=sample.depth_scale_m,
        )
        validity = "valid" if reading.valid else "invalid"
        print(
            f"frame={sample.frame_index} x={reading.x} y={reading.y} raw={reading.raw} "
            f"depth_mm={reading.depth_m * 1000.0:.6f} "
            f"depth_m={reading.depth_m:.9f} {validity}"
        )
        return

    if args.zoom_radius < 1:
        raise ValueError("zoom_radius must be at least 1")
    if args.cell_size < 38:
        raise ValueError("cell_size must be at least 38 so labels remain readable")
    if not 0 <= args.port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if args.ui == "browser":
        browser_app = BrowserDepthReviewApp(
            run_dir,
            initial_frame_index=sample.frame_index,
            follow_latest=args.frame_index is None,
            zoom_radius=args.zoom_radius,
            cell_size=args.cell_size,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        run_browser_viewer(
            browser_app,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )
        return

    opencv_viewer = DepthReviewViewer(
        run_dir,
        samples,
        initial_index=selected_index,
        zoom_radius=args.zoom_radius,
        cell_size=args.cell_size,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )
    opencv_viewer.run()


if __name__ == "__main__":
    main()
