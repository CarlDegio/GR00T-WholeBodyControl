"""Copy formal experiment trials into the user archive and refresh its reports.

Run from the repository root with .venv_inference/bin/python. Source logs and
assets remain untouched; latest annotations determine eligibility and results.
"""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gear_sonic.experiments.recording import read_events
from gear_sonic.experiments.results import METHOD_LABELS, export, rate, summarize, trial_rows

ARCHIVE = ROOT / "experiment data"
POLICY = ROOT / "outputs/experiments/formal_results/selection_policy.json"
TASKS = {
    "T1": "T1_药瓶入篮",
    "T2": "T2_垃圾袋入箱并搬箱",
    "T3": "T3_踩桶倒垃圾",
    "T4": "T4_提袋导航踩桶投放",
}
TZ = ZoneInfo("Asia/Shanghai")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_time(ns):
    return datetime.fromtimestamp(ns / 1e9, TZ).isoformat(timespec="seconds") if ns else None


def selected_events(events, trial_id):
    event_ids = {e["event_id"] for e in events if e.get("trial_id") == trial_id}
    return [
        e for e in events
        if e["type"] == "session"
        or e.get("trial_id") == trial_id
        or e.get("gate_event_id") in event_ids
    ]


def write_events(path, events):
    path.write_text("".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n" for e in events))


def metric(rows):
    sr = rate([r["success"] for r in rows])
    return {"trials": len(rows), "successes": sr["k"], "failures": sr["n"] - sr["k"],
            "labeled": sr["n"], "pending": sr["missing"], "sr": sr}


def sr_text(m):
    if not m["labeled"]:
        return "待标注"
    return f"{m['successes']}/{m['labeled']}（{100 * m['sr']['value']:.1f}%）"


def copy_trial(source, events, row, start):
    trial_id = row["trial_id"]
    dest = ARCHIVE / TASKS[row["task_id"]] / "trials" / trial_id
    dest.mkdir(parents=True, exist_ok=True)
    selected = selected_events(events, trial_id)
    files = []

    def copy_file(src):
        relative = src.relative_to(source.parent)
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        digest = sha256(src)
        if sha256(target) != digest:
            raise ValueError(f"Copy verification failed: {target}")
        files.append({"path": relative.as_posix(), "source": str(src),
                      "bytes": target.stat().st_size, "sha256": digest})

    copy_file(source.parent / "runtime.yaml")
    for name in ("checks", "snapshots"):
        folder = source.parent / name / trial_id
        for src in sorted(folder.rglob("*")):
            if src.is_file():
                copy_file(src)
    pulse_file = source.parent / "vy_pulses.jsonl"
    if pulse_file.exists():
        pulses = [e for e in read_events(pulse_file) if e.get("generation") == start["generation"]]
        write_events(dest / pulse_file.name, pulses)
        files.append({"path": pulse_file.name, "source": str(pulse_file),
                      "sha256": sha256(dest / pulse_file.name), "selection": "matching generation"})
    write_events(dest / "events.jsonl", selected)
    # Event fields retain original evidence paths. The identical directory suffix
    # below the source session resolves to the local copy under this trial folder.
    restored = trial_rows(read_events(dest / "events.jsonl"))
    if len(restored) != 1 or restored[0] != {**row, "repeat": 1}:
        raise ValueError(f"Archived result differs from source: {trial_id}")
    end = next((e for e in selected if e["type"] == "trial_end"), {})
    metadata = {
        "trial_id": trial_id, "task_id": row["task_id"], "method": row["method"],
        "case_id": row["case_id"], "condition": row["condition"],
        "start_time": local_time(start["wall_time_ns"]),
        "end_time": local_time(end.get("completed_wall_time_ns", end.get("wall_time_ns"))),
        "success": row["success"], "physical_success": row["physical_success"],
        "terminal_state": row["terminal_state"], "terminal_reason": row["terminal_reason"],
        "source_log": str(source), "source_log_sha256": sha256(source),
        "events_sha256": sha256(dest / "events.jsonl"),
        "source_path_prefix": str(source.parent), "archive_path_prefix": str(dest),
        "archive_dir": str(dest.relative_to(ARCHIVE)), "files": files,
    }
    write_json(dest / "archive.json", metadata)
    return metadata, selected


def main():
    policy = json.loads(POLICY.read_text())
    since = int(datetime.fromisoformat(policy["since_inclusive"]).timestamp() * 1e9)
    generated_at = datetime.now(TZ).isoformat(timespec="seconds")
    selected = []
    omitted = []
    seen = set()
    no_trials = []
    # Read all source records before writing any archive files.
    sources = set((ROOT / "outputs/experiments").glob("*/events.jsonl"))
    sources.update(Path(p).resolve() for p in policy.get("additional_source_logs", []))
    for source in sorted(sources):
        events = read_events(source)
        starts = {e["trial_id"]: e for e in events if e["type"] == "trial_start"}
        if not starts:
            no_trials.append(str(source))
        for row in trial_rows(events):
            tid = row["trial_id"]
            if tid in seen:
                raise ValueError(f"Duplicate source trial_id: {tid}")
            seen.add(tid)
            start = starts[tid]
            reason = "before_formal_window" if start["wall_time_ns"] < since else row["exclude_reason"]
            if not reason and row["method"] in policy.get("excluded_methods", []):
                reason = "excluded_method:" + row["method"]
            if reason:
                omitted.append({"trial_id": tid, "reason": reason, "source_log": str(source),
                                "start_time": local_time(start["wall_time_ns"])})
                continue
            if row["task_id"] not in TASKS:
                raise ValueError(f"Unknown task: {row['task_id']}")
            selected.append((source, events, row, start))
    selected.sort(key=lambda item: item[3]["wall_time_ns"])
    manifest = []
    event_sets = {task: [] for task in TASKS}
    for source, events, row, start in selected:
        entry, archived_events = copy_trial(source, events, row, start)
        manifest.append(entry)
        event_sets[row["task_id"]].append(archived_events)

    for task, folder in TASKS.items():
        task_dir = ARCHIVE / folder
        task_dir.mkdir(parents=True, exist_ok=True)
        active_ids = {m["trial_id"] for m in manifest if m["task_id"] == task}
        # A later smoke annotation withdraws a previous archive without deleting it.
        # Its log is renamed so a scan for events.jsonl cannot count the old result.
        for old in sorted((task_dir / "trials").glob("*")):
            if old.is_dir() and old.name not in active_ids:
                if not (old / "archive.json").exists():
                    raise ValueError(f"Unmanaged trial folder; review manually: {old}")
                withdrawal = next((o for o in omitted if o["trial_id"] == old.name),
                                  {"trial_id": old.name, "reason": "source_not_found"})
                history = task_dir / "withdrawn" / f"{old.name}_{datetime.now(TZ).strftime('%Y%m%d_%H%M%S_%f')}"
                history.parent.mkdir(parents=True, exist_ok=True)
                old.rename(history)
                (history / "events.jsonl").rename(history / "archived_events.jsonl")
                write_json(history / "withdrawal.json", withdrawal)
        report = summarize(event_sets[task])
        export(report, task_dir / "summary")
        rows = report["trials"]
        entries = [m for m in manifest if m["task_id"] == task]
        overview = {
            "generated_at": generated_at, "task_id": task, "since_inclusive": policy["since_inclusive"],
            "overall": metric(rows),
            "by_method": {m: metric([r for r in rows if r["method"] == m])
                          for m in sorted({r["method"] for r in rows})},
            "by_date": {date: metric([r for r in rows if r["trial_id"] in
                                       {e["trial_id"] for e in entries if e["start_time"][:10] == date}])
                        for date in sorted({e["start_time"][:10] for e in entries})},
            "by_method_case": {m: {c: metric([r for r in rows if r["method"] == m and r["case_id"] == c])
                                    for c in sorted({r["case_id"] for r in rows if r["method"] == m})}
                               for m in sorted({r["method"] for r in rows})},
        }
        write_json(task_dir / "summary/success_rates.json", overview)
        lines = [f"# {folder}", "", f"更新：{generated_at}；正式起点：{policy['since_inclusive']}。", "",
                 f"有效试验 {len(rows)} 次；已标注成功率 {sr_text(overview['overall'])}；待标注 {overview['overall']['pending']} 次。",
                 "", "成功率分母仅含已标注试验；待标注不当作失败。系统完成不能代替人工确认。", "",
                 "| 方法 | 有效试验 | 成功 | 失败 | 待标注 | 已标注成功率 |",
                 "|---|---:|---:|---:|---:|---|" ]
        for method, m in overview["by_method"].items():
            lines.append(f"| {METHOD_LABELS.get(method, method)} | {m['trials']} | {m['successes']} | {m['failures']} | {m['pending']} | {sr_text(m)} |")
        lines += ["", "[完整统计](summary/statistics.json) · [成功率统计](summary/success_rates.json) · [逐试验 CSV](summary/trials.csv)", "",
                  "| 序号 | 开始时间（北京） | 方法 | 条件 | 人工结果 | 试验数据 |",
                  "|---|---|---|---|---|---|"]
        for number, e in enumerate(entries, 1):
            label = "待标注" if e["success"] is None else "成功" if e["success"] else "失败"
            lines.append(f"| {number} | {e['start_time'][:19].replace('T', ' ')} | {e['method']} | {e['case_id']} | {label} | [{e['trial_id']}](trials/{e['trial_id']}/archive.json) |")
        (task_dir / "README.md").write_text("\n".join(lines) + "\n")
        print(task, json.dumps(overview["overall"], ensure_ascii=False))

    scope = {"generated_at": generated_at, "source_policy": str(POLICY), "policy": policy,
             "source_trials": len(seen), "archived_trials": len(manifest),
             "omitted_trials": omitted, "sessions_without_trial_start": no_trials,
             "rule": "Formal window, latest exclusions and corrections; include pending labels without guessing success."}
    write_json(ARCHIVE / "scope.json", scope)
    write_json(ARCHIVE / "manifest.json", {"generated_at": generated_at, "trials": manifest})
    with (ARCHIVE / "index.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["sequence", "start_time", "end_time", "task_id", "method", "case_id", "trial_id", "success", "terminal_state", "archive_dir", "source_log"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number, entry in enumerate(manifest, 1):
            writer.writerow({"sequence": number, **{key: entry[key] for key in fields if key != "sequence"}})


if __name__ == "__main__":
    main()
