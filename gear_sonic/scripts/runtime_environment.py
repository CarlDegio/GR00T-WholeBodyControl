"""Resolve shared environments without changing the checkout supplying code."""

import os
from pathlib import Path
import shlex


def environment_root(repo_root: Path) -> Path:
    return (
        Path(os.environ.get("SONIC_ENV_ROOT", str(repo_root.parent / "GR00T-WholeBodyControl")))
        .expanduser()
        .absolute()
    )


def activate(repo_root: Path, name: str) -> str:
    return f"source {shlex.quote(str(environment_root(repo_root) / name / 'bin/activate'))} && "


def python_path(repo_root: Path, name: str = ".venv_inference") -> str:
    return shlex.quote(str(environment_root(repo_root) / name / "bin/python"))


def code_directory(repo_root: Path) -> str:
    root = shlex.quote(str(repo_root))
    return f"cd {root} && export PYTHONPATH={root}${{PYTHONPATH:+:$PYTHONPATH}} && "


def model_environment(repo_root: Path) -> str:
    path = shlex.quote(str(environment_root(repo_root) / ".env.local"))
    return f"set -a; [ ! -f {path} ] || . {path}; set +a; "
