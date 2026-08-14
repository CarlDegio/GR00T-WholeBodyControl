from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import tomli
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
    version: object = 1,
) -> Path:
    path = tmp_path / "launch_data_collection.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "sonic.data_collection_launch",
                "version": version,
                "profile": "test_data_collection",
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


@pytest.mark.parametrize("version", [True, 1.0, 2])
def test_loader_rejects_non_integer_or_unsupported_schema_version(
    tmp_path: Path,
    version: object,
) -> None:
    path = _write_config(tmp_path, _default_values(), version=version)

    with pytest.raises(ValueError, match="sonic.data_collection_launch version 1"):
        _load(path)


def test_loader_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _default_values())
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["endpoints"] = {"camera_server": {"port": 5555}}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown data collection YAML root fields"):
        _load(path)


@pytest.mark.parametrize("profile", [None, "", 123])
def test_loader_requires_nonempty_string_profile(
    tmp_path: Path,
    profile: object,
) -> None:
    path = _write_config(tmp_path, _default_values())
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if profile is None:
        payload.pop("profile")
    else:
        payload["profile"] = profile
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="profile must be a non-empty string"):
        _load(path)


def test_loader_reports_unreadable_config_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(ValueError, match=r"cannot read launch YAML .*missing\.yaml"):
        _load(missing)


def test_bootstrap_dependency_probe_requires_tyro_and_yaml() -> None:
    probe = getattr(
        data_collection_launcher,
        "_launcher_dependencies_available",
        None,
    )
    assert probe is not None, "launcher dependency probe is missing"
    imported: list[str] = []

    def missing_yaml(module_name: str) -> object:
        imported.append(module_name)
        if module_name == "yaml":
            raise ImportError("PyYAML is unavailable")
        return object()

    assert probe(missing_yaml) is False
    assert imported == ["tyro", "yaml"]
    assert probe(lambda _module_name: object()) is True


def test_data_collection_extra_declares_pyyaml_directly() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomli.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["optional-dependencies"]["data_collection"]

    assert "pyyaml" in dependencies


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


def test_real_tyro_applies_cli_overrides_over_selected_yaml(tmp_path: Path) -> None:
    values = _default_values()
    values.update(task_prompt="yaml prompt", camera_port=6000, pico_video=True)
    path = _write_config(tmp_path, values)
    repo_root = Path(__file__).resolve().parents[2]
    program = """
import json
import sys
from gear_sonic.scripts.launch_data_collection import parse_data_collection_launch_config

config = parse_data_collection_launch_config([
    "--config", sys.argv[1],
    "--task-prompt", "CLI prompt",
    "--no-pico-video",
])
print(json.dumps({
    "task_prompt": config.task_prompt,
    "camera_port": config.camera_port,
    "pico_video": config.pico_video,
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", program, str(path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(completed.stdout) == {
        "task_prompt": "CLI prompt",
        "camera_port": 6000,
        "pico_video": False,
    }


def test_cli_reports_config_error_without_traceback(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "gear_sonic" / "scripts" / "launch_data_collection.py"
    missing = tmp_path / "missing.yaml"

    completed = subprocess.run(
        [sys.executable, str(script), "--config", str(missing)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert f"ERROR: cannot read launch YAML {missing}" in completed.stderr
    assert "Traceback" not in completed.stderr
