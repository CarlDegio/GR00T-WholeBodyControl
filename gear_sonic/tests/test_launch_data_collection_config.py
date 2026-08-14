from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

import gear_sonic.scripts.launch_data_collection as data_collection_launcher
from gear_sonic.scripts.launch_data_collection import DataCollectionLaunchConfig


def _load(path: str | Path | None = None) -> DataCollectionLaunchConfig:
    loader = getattr(
        data_collection_launcher,
        "load_data_collection_launch_config",
        None,
    )
    assert loader is not None, "data collection YAML loader is missing"
    return loader(path)


def _default_values() -> dict[str, object]:
    return {
        name: value
        for name, value in vars(DataCollectionLaunchConfig()).items()
        if name != "config"
    }


def _write_config(
    tmp_path: Path,
    values: dict[str, object],
    *,
    version: int = 1,
) -> Path:
    path = tmp_path / "launch_data_collection.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "sonic.data_collection_launch",
                "version": version,
                "launch_data_collection": values,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_default_yaml_contains_every_launch_parameter() -> None:
    default_path = getattr(
        data_collection_launcher,
        "default_data_collection_config_path",
        None,
    )
    assert default_path is not None, "default data collection YAML path is missing"

    loaded = _load()

    assert default_path().name == "launch_data_collection.yaml"
    assert loaded.config == str(default_path())
    assert loaded.runtime_profile == "gear_sonic/config/launch_inference.yaml"
    assert loaded.pico_video is True
    assert loaded.camera_viewer is True


def test_loader_rejects_unknown_launch_field(tmp_path: Path) -> None:
    values = _default_values()
    values["unexpected"] = True

    with pytest.raises(ValueError, match="unknown launch_data_collection YAML fields"):
        _load(_write_config(tmp_path, values))


def test_loader_rejects_missing_launch_field(tmp_path: Path) -> None:
    values = _default_values()
    values.pop("camera_port")

    with pytest.raises(ValueError, match="missing launch_data_collection YAML fields"):
        _load(_write_config(tmp_path, values))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sim", "false", "must be a boolean"),
        ("camera_port", True, "must be an integer"),
        ("task_prompt", 123, "must be a string"),
    ],
)
def test_loader_rejects_wrong_scalar_types(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    values = _default_values()
    values[field] = value

    with pytest.raises(ValueError, match=message):
        _load(_write_config(tmp_path, values))


def test_loader_rejects_wrong_schema_version(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _default_values(), version=2)

    with pytest.raises(ValueError, match="sonic.data_collection_launch version 1"):
        _load(path)


def test_parse_uses_selected_yaml_as_defaults_and_keeps_cli_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _default_values()
    values.update(task_prompt="yaml prompt", camera_port=6000, pico_video=True)
    path = _write_config(tmp_path, values)
    captured: dict[str, object] = {}

    def fake_cli(config_type, *, args, default):
        captured.update(config_type=config_type, args=args, default=default)
        parsed_values = vars(default).copy()
        parsed_values.update(task_prompt="CLI prompt", pico_video=False)
        return DataCollectionLaunchConfig(**parsed_values)

    monkeypatch.setattr(
        data_collection_launcher.tyro,
        "cli",
        fake_cli,
        raising=False,
    )
    parser = getattr(
        data_collection_launcher,
        "parse_data_collection_launch_config",
        None,
    )
    assert parser is not None, "two-stage data collection parser is missing"

    parsed = parser(
        [
            "--config",
            str(path),
            "--task-prompt",
            "CLI prompt",
            "--no-pico-video",
        ]
    )

    assert captured["config_type"] is DataCollectionLaunchConfig
    assert captured["args"] == [
        "--config",
        str(path),
        "--task-prompt",
        "CLI prompt",
        "--no-pico-video",
    ]
    yaml_defaults = captured["default"]
    assert isinstance(yaml_defaults, DataCollectionLaunchConfig)
    assert yaml_defaults.task_prompt == "yaml prompt"
    assert yaml_defaults.camera_port == 6000
    assert parsed.task_prompt == "CLI prompt"
    assert parsed.camera_port == 6000
    assert parsed.pico_video is False
