"""Shared process-launcher primitives with no third-party dependencies."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any


def bootstrap_venv(
    dependencies_available: bool, *, repo_root: Path, venv_name: str,
    missing_message: str,
) -> None:
    if dependencies_available:
        return
    python = repo_root / venv_name / "bin" / "python"
    if not python.exists():
        print(missing_message)
        raise SystemExit(1)
    print(f"Re-launching with {python} ...")
    os.execv(str(python), [str(python), *sys.argv])


class TmuxSession:
    """Operate one named tmux session without owning its pane layout."""

    def __init__(
        self, name: str, *, runner: Callable[..., Any] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.name = str(name)
        self._run = runner
        self._sleep = sleeper

    def command(self, *args: str, output: bool = False, check: bool = True) -> str:
        result = self._run(
            ["tmux", *args], check=check, capture_output=output, text=output,
        )
        return result.stdout.strip() if output else ""

    def kill(self) -> None:
        self._run(
            ["tmux", "kill-session", "-t", self.name], capture_output=True,
        )

    def pane_id(self, target: str) -> str:
        return self.command(
            "display-message", "-p", "-t", target, "#{pane_id}", output=True,
        )

    def split_pane(
        self, target: str, shell: Sequence[str], *split_args: str,
    ) -> str:
        return self.command(
            "split-window", *split_args, "-t", target, "-P", "-F",
            "#{pane_id}", *shell, output=True,
        )

    def send(
        self, target: str, command: str, *, wait: float = 1.0, check: bool = True,
    ) -> None:
        self.command("send-keys", "-t", target, command, "C-m", check=check)
        self._sleep(wait)

    def pane_alive(self, target: str) -> bool:
        result = self._run(
            ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
            capture_output=True, text=True,
        )
        return result.stdout.strip() != "1"

    def attach(self) -> bool:
        """Attach and report whether the session remains afterward."""
        try:
            self._run(["tmux", "attach", "-t", self.name])
        except KeyboardInterrupt:
            pass
        result = self._run(
            ["tmux", "has-session", "-t", self.name], capture_output=True,
        )
        return result.returncode == 0
