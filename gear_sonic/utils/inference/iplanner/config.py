"""Select the trajectory model without changing the shared controller profile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gear_sonic.runtime.profile import RuntimeProfile, load_component_config


def navigation_backend(profile: RuntimeProfile) -> str:
    settings = profile.components.get("navigation_policy", {})
    if set(settings) - {"backend"}:
        raise ValueError("components.navigation_policy only accepts backend")
    backend = settings.get("backend", "xnavdp")
    if backend not in {"xnavdp", "iplanner"}:
        raise ValueError("components.navigation_policy.backend must be xnavdp or iplanner")
    return backend


@dataclass(frozen=True)
class IPlannerServerConfig:
    root: str
    checkpoint: str
    model_config: str
    conda_env: str
    device: Literal["cpu", "cuda"]

    def asset_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else Path(self.root).expanduser() / path

    def required_assets(self) -> tuple[tuple[Path, str], ...]:
        return (
            (self.asset_path("iplanner_server.py"), "LaViRA iPlanner server"),
            (self.asset_path("../prompts.py"), "LaViRA navigation prompts"),
            (self.asset_path(self.checkpoint), "iPlanner checkpoint"),
            (self.asset_path(self.model_config), "iPlanner model config"),
        )


def load_iplanner_server_config(profile: RuntimeProfile) -> IPlannerServerConfig:
    config = load_component_config(
        IPlannerServerConfig,
        "iplanner",
        profile.source_files[0],
        overlays=profile.source_files[1:],
    )
    if not all((config.root.strip(), config.checkpoint.strip(),
                config.model_config.strip(), config.conda_env.strip())):
        raise ValueError("iPlanner source, checkpoint, model config, and conda_env are required")
    return config
