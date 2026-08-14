# Data Collection Launch YAML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated, strict YAML configuration entry point for `launch_data_collection.py` while continuing to use `launch_inference.yaml` as the shared Gateway runtime profile.

**Architecture:** `launch_data_collection.yaml` owns only `DataCollectionLaunchConfig` defaults. The launcher first discovers `--config`, validates that YAML, and then lets tyro apply explicit CLI overrides. Gateway endpoints and component settings remain centralized in `launch_inference.yaml`.

**Tech Stack:** Python 3.10+, dataclasses, argparse, tyro, PyYAML, pytest, YAML

## Global Constraints

- Do not duplicate `endpoints`, `ros_topics`, or `components` from `launch_inference.yaml`.
- Preserve all existing data-collection CLI flags and runtime behavior.
- Explicit CLI flags override YAML values.
- Reject unsupported schema versions, unknown fields, missing fields, and incorrect scalar types before tmux starts.
- Do not change the tmux layout or launched process commands.

---

### Task 1: Strict YAML model and loader

**Files:**
- Create: `gear_sonic/config/launch_data_collection.yaml`
- Modify: `gear_sonic/scripts/launch_data_collection.py`
- Create: `gear_sonic/tests/test_launch_data_collection_config.py`

**Interfaces:**
- Consumes: `DataCollectionLaunchConfig`.
- Produces: `default_data_collection_config_path() -> Path` and `load_data_collection_launch_config(path: str | Path | None = None) -> DataCollectionLaunchConfig`.

- [ ] **Step 1: Read the required test-quality reference**

Run:

```bash
sed -n '1,320p' /home/user/.codex/plugins/cache/openai-curated-remote/superpowers/6.2.0/skills/test-driven-development/writing-good-tests.md
```

Expected: the complete reference is read before editing tests.

- [ ] **Step 2: Write failing loader tests**

Create tests that assert:

```python
def test_default_yaml_contains_every_launch_parameter() -> None:
    loaded = load_data_collection_launch_config()
    assert default_data_collection_config_path().name == "launch_data_collection.yaml"
    assert loaded.config == str(default_data_collection_config_path())
    assert loaded.runtime_profile == "gear_sonic/config/launch_inference.yaml"
    assert loaded.pico_video is True
    assert loaded.camera_viewer is True

def test_loader_rejects_unknown_launch_field(tmp_path: Path) -> None:
    values = _default_values()
    values["unexpected"] = True
    with pytest.raises(ValueError, match="unknown launch_data_collection YAML fields"):
        load_data_collection_launch_config(_write_config(tmp_path, values))

def test_loader_rejects_missing_launch_field(tmp_path: Path) -> None:
    values = _default_values()
    values.pop("camera_port")
    with pytest.raises(ValueError, match="missing launch_data_collection YAML fields"):
        load_data_collection_launch_config(_write_config(tmp_path, values))

@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sim", "false", "must be a boolean"),
        ("camera_port", True, "must be an integer"),
        ("task_prompt", 123, "must be a string"),
    ],
)
def test_loader_rejects_wrong_scalar_types(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    values = _default_values()
    values[field] = value
    with pytest.raises(ValueError, match=message):
        load_data_collection_launch_config(_write_config(tmp_path, values))

def test_loader_rejects_wrong_schema_version(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _default_values(), version=2)
    with pytest.raises(ValueError, match="sonic.data_collection_launch version 1"):
        load_data_collection_launch_config(path)
```

The helpers write `schema: sonic.data_collection_launch`, `version: 1`, and a `launch_data_collection` mapping.

- [ ] **Step 3: Verify RED**

Run:

```bash
.venv_data_collection/bin/python -m pytest gear_sonic/tests/test_launch_data_collection_config.py -q
```

Expected: collection fails because the loader functions do not exist.

- [ ] **Step 4: Implement the minimal loader**

In `launch_data_collection.py`:

- add `argparse`, dataclass `fields`, PyYAML, and runtime type-introspection imports;
- add `default_data_collection_config_path()`;
- add final field `config: str = str(default_data_collection_config_path())`;
- validate exactly every dataclass field except `config`;
- accept only the declared `bool`, `int`, `float`, and `str` types;
- require schema `sonic.data_collection_launch` version `1`;
- construct `DataCollectionLaunchConfig(config=str(config_path), **values)`.

Create this complete default mapping:

```yaml
schema: sonic.data_collection_launch
version: 1
profile: agent_full_data_collection
launch_data_collection:
  sim: false
  deploy_input_type: zmq_manager
  deploy_zmq_host: localhost
  deploy_checkpoint: ''
  deploy_obs_config: ''
  deploy_planner: ''
  deploy_motion_data: ''
  deploy_output_type: ''
  pico_manager: true
  pico_vis_vr3pt: false
  pico_vis_smpl: false
  pico_waist_tracking: false
  pico_video: true
  task_prompt: demo
  dataset_name: ''
  runtime_profile: gear_sonic/config/launch_inference.yaml
  record_wrist_cameras: false
  record_chest_camera: false
  text_to_speech: true
  camera_viewer: true
  camera_host: localhost
  camera_port: 5555
```

- [ ] **Step 5: Verify GREEN**

Run the focused test module again and expect zero failures.

- [ ] **Step 6: Commit Task 1**

```bash
git add gear_sonic/config/launch_data_collection.yaml gear_sonic/scripts/launch_data_collection.py gear_sonic/tests/test_launch_data_collection_config.py
git commit -m "feat: add data collection launch yaml"
```

### Task 2: YAML defaults with CLI precedence

**Files:**
- Modify: `gear_sonic/scripts/launch_data_collection.py`
- Modify: `gear_sonic/tests/test_launch_data_collection_config.py`

**Interfaces:**
- Consumes: Task 1 loader and tyro.
- Produces: `parse_data_collection_launch_config(args: list[str] | None = None) -> DataCollectionLaunchConfig`.

- [ ] **Step 1: Write a failing precedence test**

Use a custom YAML with `task_prompt: yaml prompt`, `camera_port: 6000`, and `pico_video: true`. Monkeypatch `launch_data_collection.tyro.cli` to capture its `default` and simulate explicit `--task-prompt "CLI prompt" --no-pico-video` overrides. Assert the parsed result keeps YAML port `6000` but returns prompt `CLI prompt` and `pico_video is False`.

- [ ] **Step 2: Verify RED**

Run only the precedence test. Expected: FAIL because `parse_data_collection_launch_config` does not exist.

- [ ] **Step 3: Implement two-stage parsing**

Add exactly this interface:

```python
def parse_data_collection_launch_config(
    args: list[str] | None = None,
) -> DataCollectionLaunchConfig:
    argv = list(sys.argv[1:] if args is None else args)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        default=str(default_data_collection_config_path()),
    )
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    yaml_defaults = load_data_collection_launch_config(bootstrap_args.config)
    return tyro.cli(DataCollectionLaunchConfig, args=argv, default=yaml_defaults)
```

Replace the direct `tyro.cli(DataCollectionLaunchConfig)` call in `__main__` with this parser.

- [ ] **Step 4: Verify configuration and launcher compatibility**

Run:

```bash
.venv_data_collection/bin/python -m pytest   gear_sonic/tests/test_launch_data_collection_config.py   gear_sonic/tests/test_camera_viewer.py -q
```

Expected: all selected tests pass and existing command strings remain unchanged.

- [ ] **Step 5: Commit Task 2**

```bash
git add gear_sonic/scripts/launch_data_collection.py gear_sonic/tests/test_launch_data_collection_config.py
git commit -m "feat: load collection defaults from yaml"
```

### Task 3: Documentation and final verification

**Files:**
- Modify: `docs/source/tutorials/data_collection.md`
- Modify: `docs/source/tutorials/vla_workflow.md`

**Interfaces:**
- Consumes: the new default YAML and `--config`.
- Produces: documented zero-argument, custom-config, and CLI-override workflows.

- [ ] **Step 1: Update tutorials**

Document:

```bash
python gear_sonic/scripts/launch_data_collection.py
python gear_sonic/scripts/launch_data_collection.py   --config /path/to/my_collection.yaml   --task-prompt "pick up the cup"
```

State that CLI flags override YAML and `runtime_profile` continues to reference `gear_sonic/config/launch_inference.yaml`.

- [ ] **Step 2: Verify launcher help**

Run:

```bash
.venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py --help
```

Expected: exit 0 and help includes `--config`, `--task-prompt`, `--pico-video`, and `--camera-port`.

- [ ] **Step 3: Run the focused regression suite**

Run:

```bash
.venv_data_collection/bin/python -m pytest   gear_sonic/tests/test_launch_data_collection_config.py   gear_sonic/tests/test_camera_viewer.py   gear_sonic/tests/test_data_exporter_recording_controls.py -q
```

Expected: zero failures.

- [ ] **Step 4: Inspect the final patch**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Expected: no whitespace errors and only the plan, configuration, launcher, focused tests, and two tutorials changed.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/source/tutorials/data_collection.md docs/source/tutorials/vla_workflow.md
git commit -m "docs: explain data collection launch config"
```

