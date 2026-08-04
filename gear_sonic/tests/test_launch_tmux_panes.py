from __future__ import annotations

import sys
import types

import pytest


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.scripts.launch_inference import _parse_pane_ids


def test_parse_pane_ids_returns_stable_ids_in_visual_index_order() -> None:
    output = "2 %8\n0 %3\n1 %5\n5 %13\n4 %11\n3 %9\n"

    assert _parse_pane_ids(output) == ["%3", "%5", "%8", "%9", "%11", "%13"]


def test_parse_pane_ids_rejects_incomplete_layout() -> None:
    with pytest.raises(RuntimeError, match="expected 6 tmux panes, found 5"):
        _parse_pane_ids("0 %1\n1 %2\n2 %3\n3 %4\n4 %5\n")
