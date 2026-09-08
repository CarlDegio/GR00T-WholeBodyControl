"""Terminal rendering for Control Gateway runtime events."""

from __future__ import annotations

from collections import deque
import re
import shutil
import sys
import textwrap
from typing import Any, Mapping, TextIO

from gear_sonic.runtime.telemetry import format_event


class EventPaneDisplay:
    """Keep the current LaViRA TODO above a bounded recent-event stream."""

    TERMINAL_CODES = {
        "TASK_COMPLETED", "TASK_FAILED", "TASK_CANCELLED",
        "TASK_CANCEL_REQUESTED",
    }

    def __init__(
        self,
        *,
        stream: TextIO = sys.stdout,
        interactive: bool | None = None,
        max_events: int = 500,
    ) -> None:
        self.stream = stream
        self.interactive = (
            bool(stream.isatty()) if interactive is None else bool(interactive)
        )
        self.events: deque[str] = deque(maxlen=max_events)
        self.todo_list: str | None = None
        self.todo_generation: int | None = None
        self.todo_step: int | None = None
        self.waiting_for_todo = False
        self.active = False

    def accept(self, payload: Mapping[str, Any]) -> None:
        line = format_event(payload, color=False)
        code = str(payload.get("code", ""))
        component = str(payload.get("component", ""))
        fields = payload.get("fields", {})
        fields = dict(fields) if isinstance(fields, Mapping) else {}
        generation_value = fields.get("generation")
        event_generation = (
            int(generation_value) if generation_value is not None else None
        )
        if component == "lavira" and code == "TODO_UPDATED":
            todo = fields.get("todo_list")
            if not isinstance(todo, str):
                raise ValueError("TODO_UPDATED requires string fields.todo_list")
            if (
                self.todo_generation is None
                or event_generation is None
                or event_generation >= self.todo_generation
            ):
                self.todo_list = todo
                self.todo_generation = event_generation
                self.todo_step = int(fields.get("step", 0))
                self.waiting_for_todo = False
        else:
            self.events.append(line)
            if code in {"START_NAVIGATION", "TASK_ACCEPTED"}:
                if (
                    self.todo_generation is None
                    or event_generation is None
                    or event_generation >= self.todo_generation
                ):
                    self.todo_list = None
                    self.todo_generation = event_generation
                    self.todo_step = None
                    self.waiting_for_todo = True
            elif (
                component in {"lavira", "control_gateway"}
                and code in self.TERMINAL_CODES
                and (
                    self.todo_generation is None
                    or event_generation is None
                    or event_generation == self.todo_generation
                )
            ):
                self.todo_list = None
                self.todo_generation = None
                self.todo_step = None
                self.waiting_for_todo = False
        if self.interactive:
            self.render()
        else:
            print(line, file=self.stream, flush=True)

    @staticmethod
    def _todo_items(todo_list: str) -> tuple[list[tuple[str, str]], int, int]:
        items: list[tuple[str, str]] = []
        completed = 0
        active_marked = False
        for raw_line in todo_list.splitlines():
            line = raw_line.strip()
            checked = re.match(r"^- \[[xX]\]\s*(.*)$", line)
            unchecked = re.match(r"^- \[ \]\s*(.*)$", line)
            if checked:
                completed += 1
                items.append(("✓", checked.group(1)))
            elif unchecked:
                marker = "▶" if not active_marked else "○"
                active_marked = True
                items.append((marker, unchecked.group(1)))
            elif line:
                items.append(("·", line))
        return items, completed, len(items)

    @staticmethod
    def _wrapped(prefix: str, text: str, width: int) -> list[str]:
        available = max(12, width - len(prefix))
        chunks = textwrap.wrap(
            text, width=available, replace_whitespace=True,
            drop_whitespace=True,
        ) or [""]
        continuation = " " * len(prefix)
        return [prefix + chunks[0], *(
            continuation + chunk for chunk in chunks[1:]
        )]

    def dashboard_text(self, *, width: int = 120, height: int = 40) -> str:
        width = max(40, int(width))
        height = max(12, int(height))
        if self.todo_list is None:
            if self.waiting_for_todo:
                generation = (
                    "?" if self.todo_generation is None
                    else str(self.todo_generation)
                )
                todo_lines = [
                    f"LA TODO · generation={generation}",
                    "  Waiting for the Language Agent to create the TODO…",
                ]
            else:
                todo_lines = [
                    "LA TODO · no active task",
                    "  Press N in the Control pane to start manipulation.",
                ]
        else:
            items, completed, total = self._todo_items(self.todo_list)
            todo_lines = [
                "LA TODO · "
                f"generation={self.todo_generation} · step={self.todo_step} · "
                f"progress={completed}/{total}"
            ]
            for marker, text in items:
                todo_lines.extend(self._wrapped(f"  {marker} ", text, width))

        # Reserve roughly the top quarter for TODO and the rest for events.
        max_todo_lines = max(2, height // 4)
        if len(todo_lines) > max_todo_lines:
            body = todo_lines[1:]
            visible_body_lines = max(0, max_todo_lines - 2)
            active_index = next(
                (index for index, line in enumerate(body) if line.startswith("  ▶ ")),
                0,
            )
            start = max(0, active_index - max(0, visible_body_lines // 2))
            start = min(start, max(0, len(body) - visible_body_lines))
            visible = body[start : start + visible_body_lines]
            hidden = len(body) - len(visible)
            todo_lines = [
                todo_lines[0],
                *visible,
                f"  … {hidden} additional TODO line(s)",
            ]
        separator = "─" * min(width, 120)
        fixed = [*todo_lines, separator, "RUNTIME EVENTS · latest"]
        event_capacity = max(1, height - len(fixed) - 1)
        event_lines = list(self.events)[-event_capacity:]
        lines = [*fixed, *(event_lines or ["[INFO] Waiting for runtime events…"])]
        return "\n".join(line[:width] for line in lines)

    def render(self) -> None:
        if not self.interactive:
            return
        if not self.active:
            self.stream.write("\x1b[?1049h\x1b[?25l")
            self.active = True
        size = shutil.get_terminal_size((120, 40))
        self.stream.write("\x1b[2J\x1b[H")
        self.stream.write(self.dashboard_text(width=size.columns, height=size.lines))
        self.stream.flush()

    def close(self) -> None:
        if self.active:
            self.stream.write("\x1b[?25h\x1b[?1049l")
            self.stream.flush()
            self.active = False
