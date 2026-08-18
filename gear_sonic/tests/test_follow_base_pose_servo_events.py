from __future__ import annotations

import os
from pathlib import Path

import pytest

from gear_sonic.scripts import follow_base_pose_servo_events as follower
from gear_sonic.scripts.follow_base_pose_servo_events import (
    find_latest_event_log,
    follow_event_logs,
)


def test_find_latest_event_log_selects_newest_raw_or_dual_run(tmp_path: Path) -> None:
    raw = tmp_path / "raw_yoloe_20260817_115857_g2" / "raw_servo_events.jsonl"
    dual = (
        tmp_path
        / "dual_raw_yoloe_20260817_120000_g3"
        / "raw_servo_events.jsonl"
    )
    ignored = tmp_path / "runtime_events.jsonl"
    raw.parent.mkdir()
    dual.parent.mkdir()
    raw.write_text('{"event":"old"}\n', encoding="utf-8")
    dual.write_text('{"event":"new"}\n', encoding="utf-8")
    ignored.write_text('{"event":"newest but global"}\n', encoding="utf-8")
    os.utime(raw, ns=(1, 1))
    os.utime(dual, ns=(2, 2))
    os.utime(ignored, ns=(3, 3))

    assert find_latest_event_log(tmp_path) == dual


def test_find_latest_event_log_ignores_missing_event_files(tmp_path: Path) -> None:
    (tmp_path / "raw_yoloe_20260817_115857_g2").mkdir()

    assert find_latest_event_log(tmp_path) is None


def test_follow_event_logs_tails_existing_then_switches_to_new_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    old_log = tmp_path / "raw_yoloe_20260817_115857_g2" / follower.LOG_NAME
    new_log = tmp_path / "raw_yoloe_20260817_120000_g3" / follower.LOG_NAME
    old_log.parent.mkdir()
    old_log.write_text('{"event":"historical"}\n', encoding="utf-8")
    sleep_count = 0

    class StopFollowing(Exception):
        pass

    def advance_logs(_seconds: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 1:
            with old_log.open("a", encoding="utf-8") as handle:
                handle.write('{"event":"appended"}\n')
        elif sleep_count == 2:
            new_log.parent.mkdir()
            new_log.write_text('{"event":"next_run"}\n', encoding="utf-8")
        else:
            raise StopFollowing

    monkeypatch.setattr(follower.time, "sleep", advance_logs)

    with pytest.raises(StopFollowing):
        follow_event_logs(tmp_path)

    output = capsys.readouterr().out
    assert '"historical"' not in output
    assert '"appended"' in output
    assert '"next_run"' in output
    assert str(old_log) in output
    assert str(new_log) in output
