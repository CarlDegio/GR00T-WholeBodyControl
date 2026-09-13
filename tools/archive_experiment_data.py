"""Rebuild the user archive from formal source trials without modifying sources."""

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

from gear_sonic.experiments.recording import read_events  # noqa: E402
from gear_sonic.experiments.results import (  # noqa: E402
    METHOD_LABELS, export, rate, summarize, trial_rows,
)

ARCHIVE = ROOT / "experiment data"
POLICY = ROOT / "outputs/experiments/formal_results/selection_policy.json"
TASKS = {
    "T1": "T1_药瓶入篮",
    "T2": "T2_垃圾袋入箱并搬箱",
    "T3": "T3_踩桶倒垃圾",
    "T4": "T4_柜顶取物放入抽屉",
}
TZ = ZoneInfo("Asia/Shanghai")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_events(path, events):
    path.write_text("".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n" for e in events))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_time(ns):
    return datetime.fromtimestamp(ns / 1e9, TZ).isoformat(timespec="seconds") if ns else None


def metric(rows):
    sr = rate([r["success"] for r in rows])
    return dict(trials=len(rows), successes=sr["k"], failures=sr["n"] - sr["k"],
                labeled=sr["n"], pending=sr["missing"], sr=sr)


def sr_text(m):
    if not m["trials"]:
        return "无样本"
    if not m["labeled"]:
        return "待标注"
    return f"{m['successes']}/{m['labeled']}（{100 * m['sr']['value']:.1f}%）"


def copy_trial(source, events, row, start):
    tid = row["trial_id"]
    dest = ARCHIVE / TASKS[row["task_id"]] / "trials" / tid
    dest.mkdir(parents=True, exist_ok=True)
    ids = {e["event_id"] for e in events if e.get("trial_id") == tid}
    selected = [e for e in events if e["type"] == "session" or e.get("trial_id") == tid
                or e.get("gate_event_id") in ids]
    end = next((e for e in selected if e["type"] == "trial_end"), {})
    write_events(dest / "events.jsonl", selected)
    files = []
    sources = [source.parent / "runtime.yaml"]
    for folder in ("checks", "snapshots"):
        sources.extend(p for p in sorted((source.parent / folder / tid).rglob("*")) if p.is_file())
    # These request files have no trial_id; retain requests timestamped inside
    # the trial's recorded execution interval and document this association.
    request_end = end.get("completed_wall_time_ns", end.get("wall_time_ns"))
    if request_end is not None:
        for request in sorted((source.parent / "lavira_requests").glob("*.json")):
            timestamp = json.loads(request.read_text()).get("timestamp_ns")
            if timestamp is not None and start["wall_time_ns"] <= timestamp <= request_end:
                sources.append(request)
    for src in sources:
        relative = src.relative_to(source.parent)
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        digest = sha256(src)
        if sha256(target) != digest:
            raise ValueError(f"Copy verification failed: {target}")
        files.append(dict(path=relative.as_posix(), source=str(src),
                          bytes=target.stat().st_size, sha256=digest))
    pulse = source.parent / "vy_pulses.jsonl"
    if pulse.exists():
        selected_pulses = [e for e in read_events(pulse) if e.get("generation") == start["generation"]]
        write_events(dest / pulse.name, selected_pulses)
        files.append(dict(path=pulse.name, source=str(pulse), selection="matching generation",
                          bytes=(dest / pulse.name).stat().st_size, sha256=sha256(dest / pulse.name)))
    restored = trial_rows(read_events(dest / "events.jsonl"))
    if restored != [{**row, "repeat": 1}]:
        raise ValueError(f"Archive differs from source result: {tid}")
    entry = {k: row[k] for k in ("trial_id", "task_id", "method", "case_id", "condition", "success",
                                "physical_success", "terminal_state", "terminal_reason", "progress",
                                "progress_numerator", "progress_denominator")}
    entry.update(start_time=local_time(start["wall_time_ns"]),
                 end_time=local_time(end.get("completed_wall_time_ns", end.get("wall_time_ns"))),
                 source_log=str(source), source_log_sha256=sha256(source),
                 events_sha256=sha256(dest / "events.jsonl"), source_path_prefix=str(source.parent),
                 archive_path_prefix=str(dest), archive_dir=str(dest.relative_to(ARCHIVE)), files=files)
    entry["request_selection"] = "lavira_requests.timestamp_ns within trial_start..trial_end (no trial_id in source)"
    write_json(dest / "archive.json", entry)
    return entry, selected


def main():
    policy = json.loads(POLICY.read_text())
    since = int(datetime.fromisoformat(policy["since_inclusive"]).timestamp() * 1e9)
    excluded_methods = set(policy.get("excluded_methods", []))
    generated = datetime.now(TZ).isoformat(timespec="seconds")
    selected, omitted, no_trials, seen = [], [], [], set()
    for source in sorted((ROOT / "outputs/experiments").glob("*/events.jsonl")):
        events = read_events(source)
        starts = {e["trial_id"]: e for e in events if e["type"] == "trial_start"}
        if not starts:
            no_trials.append(str(source))
        for row in trial_rows(events):
            tid = row["trial_id"]
            if tid in seen:
                raise ValueError(f"Duplicate source trial: {tid}")
            seen.add(tid)
            start = starts[tid]
            reason = "before_formal_window" if start["wall_time_ns"] < since else row["exclude_reason"]
            if not reason and row["method"] in excluded_methods:
                reason = f"method_excluded:{row['method']}"
            if reason:
                omitted.append(dict(trial_id=tid, reason=reason, source_log=str(source),
                                    start_time=local_time(start["wall_time_ns"])))
            else:
                if row["task_id"] not in TASKS:
                    raise ValueError(f"Unknown task: {row['task_id']}")
                selected.append((source, events, row, start))
    selected.sort(key=lambda item: item[3]["wall_time_ns"])
    manifest, event_sets = [], {task: [] for task in TASKS}
    for source, events, row, start in selected:
        entry, archived_events = copy_trial(source, events, row, start)
        manifest.append(entry)
        event_sets[row["task_id"]].append(archived_events)

    for task, folder in TASKS.items():
        task_dir = ARCHIVE / folder
        task_dir.mkdir(parents=True, exist_ok=True)
        entries = [e for e in manifest if e["task_id"] == task]
        active_ids = {e["trial_id"] for e in entries}
        for old in sorted((task_dir / "trials").glob("*")):
            if not old.is_dir() or old.name in active_ids:
                continue
            if not (old / "archive.json").exists():
                raise ValueError(f"Unmanaged archive folder: {old}")
            withdrawal = next((o for o in omitted if o["trial_id"] == old.name),
                              dict(trial_id=old.name, reason="source_not_found"))
            stamp = datetime.now(TZ).strftime("%Y%m%d_%H%M%S_%f")
            history = task_dir / "withdrawn" / f"{old.name}_{stamp}"
            history.parent.mkdir(parents=True, exist_ok=True)
            old.rename(history)
            (history / "events.jsonl").rename(history / "archived_events.jsonl")
            write_json(history / "withdrawal.json", withdrawal)
        report = summarize(event_sets[task])
        export(report, task_dir / "summary")
        rows = report["trials"]
        methods = sorted({r["method"] for r in rows})
        dates = {e["trial_id"]: e["start_time"][:10] for e in entries}
        overview = dict(
            generated_at=generated, task_id=task, since_inclusive=policy["since_inclusive"],
            overall=metric(rows),
            by_method={m: metric([r for r in rows if r["method"] == m]) for m in methods},
            by_date={d: metric([r for r in rows if dates[r["trial_id"]] == d])
                     for d in sorted(set(dates.values()))},
            by_method_case={m: {c: metric([r for r in rows if r["method"] == m and r["case_id"] == c])
                                for c in sorted({r["case_id"] for r in rows if r["method"] == m})}
                            for m in methods},
        )
        write_json(task_dir / "summary/success_rates.json", overview)
        total = overview["overall"]
        lines = [f"# {folder}", "", f"更新：{generated}；正式起点：{policy['since_inclusive']}。", "",
                 f"有效试验 {len(rows)} 次；已标注成功率 {sr_text(total)}；待标注 {total['pending']} 次。", "",
                 "成功率分母仅含已标注试验；待标注不当作失败，系统完成不能代替人工确认。", "",
                 "| 方法 | 有效试验 | 成功 | 失败 | 待标注 | 已标注成功率 |",
                 "|---|---:|---:|---:|---:|---|"]
        for method, m in overview["by_method"].items():
            lines.append(f"| {METHOD_LABELS.get(method, method)} | {m['trials']} | {m['successes']} | "
                         f"{m['failures']} | {m['pending']} | {sr_text(m)} |")
        lines += ["", "[完整统计](summary/statistics.json) · [成功率统计](summary/success_rates.json) · "
                  "[逐试验 CSV](summary/trials.csv)", "",
                  "| 序号 | 开始时间（北京） | 方法 | 条件 | 人工结果 | Progress | 试验数据 |",
                  "|---|---|---|---|---|---|---|"]
        for number, entry in enumerate(entries, 1):
            label = "待标注" if entry["success"] is None else "成功" if entry["success"] else "失败"
            tid = entry["trial_id"]
            start_time = entry["start_time"][:19].replace("T", " ")
            progress = (f"{entry['progress_numerator']}/{entry['progress_denominator']}"
                        if entry["progress_numerator"] is not None else "待标注")
            lines.append(f"| {number} | {start_time} | {entry['method']} | {entry['case_id']} | "
                         f"{label} | {progress} | [{tid}](trials/{tid}/archive.json) |")
        (task_dir / "README.md").write_text("\n".join(lines) + "\n")
        print(task, json.dumps(total, ensure_ascii=False))

    scope = dict(generated_at=generated, source_policy=str(POLICY), policy=policy,
                 source_trials=len(seen), archived_trials=len(manifest), omitted_trials=omitted,
                 sessions_without_trial_start=no_trials)
    write_json(ARCHIVE / "scope.json", scope)
    write_json(ARCHIVE / "manifest.json", dict(generated_at=generated, trials=manifest))
    with (ARCHIVE / "index.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["sequence", "start_time", "end_time", "task_id", "method", "case_id", "trial_id",
                  "success", "progress_numerator", "progress_denominator", "progress", "terminal_state",
                  "archive_dir", "source_log"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number, entry in enumerate(manifest, 1):
            writer.writerow(dict(sequence=number, **{k: entry[k] for k in fields if k != "sequence"}))


if __name__ == "__main__":
    main()
