"""Independent labels and reproducible table statistics from minimal logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from .config import merge, read_yaml
from .recording import Recorder, read_events

METHOD_LABELS = {
    "full_vln": "Full agent",
    "full_objectnav": "Full agent（目标搜索）",
    "nav_direct_vla": "导航后直接 VLA",
    "navila_basepose_vla": "NaVILA + BasePose + VLA",
    "geometric_vla": "简单几何接近/居中 + VLA",
    "near_dual": "BasePose + VLA（双相机）",
    "near_head": "BasePose + VLA（单头相机）",
    "near_vla": "纯 VLA",
}


def rate(values):
    known = [bool(v) for v in values if v is not None]
    n, k = len(known), sum(known)
    if not n:
        return dict(k=0, n=0, value=None, ci95=None, missing=len(values))
    p, z = k / n, 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return dict(
        k=k, n=n, value=p, ci95=[max(0.0, center - radius), min(1.0, center + radius)], missing=len(values) - n
    )


def complete_mean(values):
    return mean(values) if values and all(v is not None for v in values) else None


def trial_rows(events):
    session = next(e["experiment"] for e in events if e["type"] == "session")
    rows = []
    for start in (e for e in events if e["type"] == "trial_start"):
        trial = [e for e in events if e.get("trial_id") == start["trial_id"]]
        end = next((e for e in trial if e["type"] == "trial_end"), None)
        synthetic_end = end is not None and end.get("reason") == "operator_closed_log_offline"
        annotations = {}
        for event in trial:
            if event["type"] == "annotation":
                annotations = merge(annotations, event["values"])
        runtime = [e for e in trial if e["type"] == "runtime"]
        # Late model replies remain evidence but cannot create another completed skill.
        if end:
            runtime = [e for e in runtime if e["monotonic_ns"] <= end["monotonic_ns"]]
        skill_starts = [e for e in runtime if e["code"] == "SKILL_STARTED"]
        align_count = sum(e.get("skill") == "ALIGN" for e in skill_starts)
        nav_retries = 0
        recovery_needed = False
        for event in runtime:
            if event["code"] == "SKILL_STARTED" and event.get("skill") == "MOVE_TO" and recovery_needed:
                nav_retries += 1
                recovery_needed = False
            if event["code"] == "SKILL_COMPLETED" and event.get("skill") in {"ALIGN", "MOVE_TO"}:
                recovery_needed = event.get("controller_state") in {"failed", "target_not_found"} or event.get(
                    "va_result"
                ) in {"NOT_FOUND", "NOT_SATISFIED", "UNKNOWN"}
        active_events = [e for e in trial if end is None or e["monotonic_ns"] <= end["monotonic_ns"]]
        started_vla = any(e["type"] == "vla_first_action" for e in active_events)
        invoked = {e.get("skill_id") for e in active_events if e["type"] == "vla_first_action"}
        success = annotations.get("success")
        completion = annotations.get("completion_time_s")
        intervention = any(
            e["type"] == "intervention"
            and (completion is None or (e["monotonic_ns"] - start["monotonic_ns"]) / 1e9 < completion)
            for e in trial
        )
        if intervention and success is True:
            success = False
        progress = annotations.get("progress")
        progress = progress / len(session["task"]["milestones"]) if progress is not None else None
        duration = (
            session["task"]["total_timeout_s"] if success is False else completion if success is True else None
        )
        perturb = next((e for e in trial if e["type"] == "perturbation"), None)
        trigger = (perturb["monotonic_ns"] - start["monotonic_ns"]) / 1e9 if perturb else None
        recovery = None
        if perturb and success is False:
            recovery = session["task"]["recovery_budget_s"]
        elif perturb and success is True and completion is not None:
            recovery = max(0.0, completion - trigger)
        snapshots = [e for e in trial if e["type"] == "vla_snapshot"]
        inferred_ids = {e["skill_id"] for e in trial if e["type"] == "vla_first_inference"}
        snapshot_ids = {e["skill_id"] for e in snapshots}
        mask_ids = {e["skill_id"] for e in trial if e["type"] == "mask_enrichment"}
        condition = merge(session["condition"], annotations.get("condition", {}))

        def first_time(event_type, **criteria):
            event = next(
                (
                    e
                    for e in active_events
                    if e["type"] == event_type and all(e.get(k) == v for k, v in criteria.items())
                ),
                None,
            )
            return (event["monotonic_ns"] - start["monotonic_ns"]) / 1e9 if event else None

        row = dict(
            trial_id=start["trial_id"],
            method=session["method"],
            table=session["table"],
            repeat=1 + len(rows),
            entry_stage=session["entry_stage"],
            task_id=session["task_id"],
            case_id=session["case_id"],
            visibility=condition.get("visibility"),
            nearfield_group=condition.get("nearfield_group"),
            condition=condition,
            success=success,
            physical_success=annotations.get("success"),
            progress=progress,
            time_s=duration,
            completion_time_s=completion,
            actual_runtime_s=(end["monotonic_ns"] - start["monotonic_ns"]) / 1e9
            if end and not synthetic_end
            else None,
            nav_success=annotations.get("nav_success") if session["entry_stage"] == "navigation" else None,
            handoff_entered=any(e.get("skill") in {"ALIGN", "MANIPULATE"} for e in skill_starts) or started_vla,
            vla_started=started_vla,
            vla_activation_count=len(invoked),
            vla_requested_s=first_time("vla_command", command="start_vla_task"),
            vla_acknowledged_s=first_time("runtime", code="VLA_TASK_ACKNOWLEDGED"),
            vla_first_inference_s=first_time("vla_first_inference"),
            vla_first_action_s=first_time("vla_first_action"),
            align_attempts=align_count,
            align_retries=max(0, align_count - 1),
            nav_retries=nav_retries,
            alignment_timeout=any(
                ("timeout" in str(e.get("reason", "")) or "max_run" in str(e.get("reason", "")))
                and e.get("channel") == "base_pose_status"
                for e in runtime
            ),
            snapshot_count=len(snapshots),
            valid_pose_count=sum(bool(e.get("pose_valid")) for e in snapshots),
            missing_snapshot_count=len(inferred_ids - snapshot_ids),
            missing_depth_count=sum(2 - e.get("depth_count", 0) for e in snapshots),
            pending_mask_count=len(snapshot_ids - mask_ids),
            missing_annotations=[key for key in ("success", "progress") if annotations.get(key) is None]
            + (["completion_time_s"] if annotations.get("success") is True and completion is None else [])
            + (
                ["nav_success"]
                if session["entry_stage"] == "navigation" and annotations.get("nav_success") is None
                else []
            ),
            intervention=intervention,
            perturbed=perturb is not None,
            recovery_time_s=recovery,
            excluded=bool(annotations.get("exclude_reason")),
            exclude_reason=annotations.get("exclude_reason"),
            terminal_state=end.get("state") if end else None,
            terminal_reason=end.get("reason") if end else None,
            complete_log=end is not None and not synthetic_end,
            terminal_record_synthetic=synthetic_end,
        )
        rows.append(row)
    return rows


def task_metrics(rows):
    usable = [r for r in rows if not r["excluded"]]
    started = [r for r in usable if r["vla_started"]]
    return dict(
        n=len(usable),
        excluded=len(rows) - len(usable),
        sr=rate([r["success"] for r in usable]),
        progress=complete_mean([r["progress"] for r in usable]),
        time_s=complete_mean([r["time_s"] for r in usable]),
        nav_sr=rate([r["nav_success"] for r in usable if r["entry_stage"] == "navigation"]),
        handoff_rate=rate([r["handoff_entered"] for r in usable]),
        vla_start_rate=rate([r["vla_started"] for r in usable]),
        conditional_manipulation_sr=rate([r["success"] for r in started]),
        visible_sr=rate([r["success"] for r in usable if r["visibility"] == "visible"]),
        invisible_sr=rate([r["success"] for r in usable if r["visibility"] == "invisible"]),
        alignment_timeout_count=sum(r["alignment_timeout"] for r in usable),
        nav_retries=complete_mean([r["nav_retries"] for r in usable]),
        align_retries=complete_mean([r["align_retries"] for r in usable]),
        incomplete_logs=sum(not r["complete_log"] for r in usable),
        snapshots=sum(r["snapshot_count"] for r in usable),
        valid_poses=sum(r["valid_pose_count"] for r in usable),
        missing_snapshots=sum(r["missing_snapshot_count"] for r in usable),
        missing_depths=sum(r["missing_depth_count"] for r in usable),
        pending_masks=sum(r["pending_mask_count"] for r in usable),
        missing_annotation_fields=sum(len(r["missing_annotations"]) for r in usable),
        vla_not_started=sum(not r["vla_started"] for r in usable),
        trials_with_snapshot=sum(r["snapshot_count"] > 0 for r in usable),
        valid_pose_rate=rate([r["valid_pose_count"] > 0 for r in usable]),
        training_neighborhood_coverage=None,
    )


def summarize(event_sets):
    rows = [r for events in event_sets for r in trial_rows(events)]
    methods = {}
    for name in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == name]
        tasks = {
            task: task_metrics([r for r in selected if r["task_id"] == task]) for task in ("T1", "T2", "T3", "T4")
        }
        methods[name] = dict(
            table=selected[0]["table"],
            tasks=tasks,
            mean_sr=complete_mean([m["sr"]["value"] for m in tasks.values()]),
            mean_progress=complete_mean([m["progress"] for m in tasks.values()]),
            mean_time_s=complete_mean([m["time_s"] for m in tasks.values()]),
            nearfield_groups={
                group: {
                    task: task_metrics(
                        [r for r in selected if r["task_id"] == task and r["nearfield_group"] == group]
                    )
                    for task in ("T1", "T2", "T3", "T4")
                }
                for group in sorted({r["nearfield_group"] for r in selected if r["nearfield_group"]})
            },
        )
        rates = [m["sr"] for m in tasks.values()]
        methods[name]["mean_sr_ci95"] = None
        if all(r["n"] for r in rates):
            rng = np.random.default_rng(0)
            samples = sum(rng.binomial(r["n"], r["value"], 10000) / r["n"] for r in rates) / 4
            methods[name]["mean_sr_ci95"] = np.quantile(samples, [0.025, 0.975]).tolist()
    gate_metrics, semantic = {}, {}
    semantic_ids = set()
    for events in event_sets:
        session = next(e["experiment"] for e in events if e["type"] == "session")
        labels = {e["gate_event_id"]: e["truth"] for e in events if e["type"] == "gate_label"}
        for gate in (e for e in events if e["type"] == "gate"):
            if session["table"] != "gates" or gate["gate"] != session.get("gate_under_test"):
                continue
            if any(r["trial_id"] == gate.get("trial_id") and r["excluded"] for r in rows):
                continue
            end = next(
                (e for e in events if e["type"] == "trial_end" and e.get("trial_id") == gate.get("trial_id")), None
            )
            if gate["applied"] == "ignored_late_result" or end and gate["monotonic_ns"] > end["monotonic_ns"]:
                continue
            key = f"{session['method']}:{gate['gate']}"
            metric = gate_metrics.setdefault(
                key, {"false_allow": [], "false_block": [], "unlabeled": 0, "events": 0}
            )
            metric["events"] += 1
            truth = labels.get(gate["event_id"])
            disturbed = session["condition"].get("perturbation", {}).get("enabled", False)
            if disturbed and not any(
                e["type"] == "perturbation"
                and e.get("trial_id") == gate.get("trial_id")
                and e["monotonic_ns"] <= gate["monotonic_ns"]
                for e in events
            ):
                metric["events"] -= 1
                continue
            if truth is None:
                metric["unlabeled"] += 1
                metric["false_allow" if disturbed else "false_block"].append(None)
                continue
            if not truth and disturbed:
                metric["false_allow"].append(gate["applied"] == "allow")
            if truth and not disturbed:
                metric["false_block"].append(gate["applied"] != "allow")
        labels = {}
        for event in events:
            if event["type"] == "semantic_label":
                labels.setdefault(event["sample_id"], {}).update(event["values"])
        for sample in (e for e in events if e["type"] == "semantic_result"):
            if sample["sample_id"] in semantic_ids:
                raise ValueError(
                    f"Duplicate semantic sample across input logs: {sample['sample_id']}; "
                    "select one evaluation run"
                )
            semantic_ids.add(sample["sample_id"])
            semantic.setdefault(sample["task_id"], []).append(labels.get(sample["sample_id"], {}))
    for key, metric in gate_metrics.items():
        metric["false_allow"], metric["false_block"] = rate(metric["false_allow"]), rate(metric["false_block"])
        method = key.split(":")[0]
        recovered = [r for r in rows if r["method"] == method and r["perturbed"] and not r["excluded"]]
        metric["recovery_sr"] = rate([r["success"] for r in recovered])
        metric["recovery_time_s"] = complete_mean([r["recovery_time_s"] for r in recovered])
    semantic_metrics = {
        task: {
            **{field: rate([r.get(field) for r in labels]) for field in ("position", "yaw", "joint", "usable")},
            "n": len(labels),
        }
        for task, labels in semantic.items()
    }
    return dict(
        trials=rows,
        methods=methods,
        gates=gate_metrics,
        semantic=semantic_metrics,
        semantic_macro={
            field: complete_mean(
                [semantic_metrics.get(task, {}).get(field, {}).get("value") for task in ("T1", "T2", "T3", "T4")]
            )
            for field in ("position", "yaw", "joint", "usable")
        },
    )


def format_rate(value):
    if value["value"] is None:
        return "待标注" if value["missing"] else "不适用"
    lo, hi = value["ci95"]
    pending = f"；待标注 {value['missing']}" if value["missing"] else ""
    return (
        f"{value['k']}/{value['n']} ({100 * value['value']:.1f}%; 95% CI {100 * lo:.1f}–{100 * hi:.1f}%){pending}"
    )


def export(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    rows = report["trials"]
    with (output / "trials.csv").open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v for k, v in row.items()}
                for row in rows
            )

    def number(v, percent=False):
        return "—" if v is None else f"{v * (100 if percent else 1):.1f}" + ("%" if percent else "")

    lines = [
        "# ICRA 实验结果",
        "",
        "SR 分母为已独立标注的试验；缺失标注不作失败。四任务未齐的宏平均保留空缺。",
        "",
    ]
    for table, title in (("main", "表 1"), ("nearfield", "表 2")):
        heading = (
            "| 方法 | 输入条件 | T1 SR | T2 SR | T3 SR | T4 SR | 平均 SR | 平均 Progress | 平均耗时 (s) |"
            if table == "main"
            else "| 方法 | T1 SR | T2 SR | T3 SR | T4 SR | 平均 SR | 训练邻域覆盖率 |"
        )
        lines += [f"## {title}", "", heading, "|" + "---|" * (9 if table == "main" else 7)]
        for method, data in report["methods"].items():
            if data["table"] != table:
                continue
            cells = [METHOD_LABELS.get(method, method)] + (
                ["ObjectNav" if method == "full_objectnav" else "VLN"] if table == "main" else []
            )
            cells += [format_rate(data["tasks"][t]["sr"]) for t in ("T1", "T2", "T3", "T4")]
            macro = number(data["mean_sr"], True)
            if data["mean_sr_ci95"] is not None:
                lo, hi = data["mean_sr_ci95"]
                macro += f" (95% CI {100 * lo:.1f}–{100 * hi:.1f}%)"
            cells += [macro] + (
                [number(data["mean_progress"], True), number(data["mean_time_s"])]
                if table == "main"
                else ["未计算"]
            )
            lines.append("| " + " | ".join(cells) + " |")
        if table == "nearfield":
            lines += ["", "训练邻域覆盖率：未计算；首帧、深度与角色 mask 供后续分析。"]
        lines += [""]
    lines += [
        "## 副表 1",
        "",
        "| 方法与门控 | 无扰动错误阻挡率 | 扰动错误放行率 | 扰动后 SR | 恢复耗时 (s) | 未标注检查数 |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in report["gates"].items():
        lines.append(
            f"| {name} | {format_rate(m['false_block'])} | {format_rate(m['false_allow'])} "
            f"| {format_rate(m['recovery_sr'])} | {number(m['recovery_time_s'])} | {m['unlabeled']} |"
        )
    lines += [
        "",
        "## 副表 2",
        "",
        "| 任务 | 样本数 | 位置角色正确率 | yaw 角色正确率 | 联合正确率 | 检测可用率 |",
        "|---|---|---|---|---|---|",
    ]
    for task, m in report["semantic"].items():
        lines.append(
            "| "
            + " | ".join([task, str(m["n"])] + [format_rate(m[k]) for k in ("position", "yaw", "joint", "usable")])
            + " |"
        )
    lines.append(
        "| 四任务宏平均 | 不适用 | "
        + " | ".join(number(report["semantic_macro"][k], True) for k in ("position", "yaw", "joint", "usable"))
        + " |"
    )
    (output / "tables.md").write_text("\n".join(lines) + "\n")


def boolean(value):
    if value.lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Use true or false")
    return value.lower() == "true"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    result = subs.add_parser("result")
    result.add_argument("log", type=Path)
    result.add_argument("--trial", required=True)
    result.add_argument("--success", type=boolean)
    result.add_argument("--progress", type=int)
    result.add_argument("--completion-time", type=float)
    result.add_argument("--nav-success", type=boolean)
    result.add_argument("--exclude-reason")
    result.add_argument("--close-interrupted", action="store_true")
    result.add_argument(
        "--condition-file", type=Path, help="YAML mapping of independently verified condition metadata"
    )
    gate = subs.add_parser("gate")
    gate.add_argument("log", type=Path)
    gate.add_argument("--event", required=True)
    gate.add_argument("--truth", required=True, type=boolean)
    sem = subs.add_parser("semantic")
    sem.add_argument("log", type=Path)
    sem.add_argument("--sample", required=True)
    for name in ("position", "yaw", "joint", "usable"):
        sem.add_argument("--" + name, type=boolean)
    inspect = subs.add_parser("inspect")
    inspect.add_argument("log", type=Path)
    aggregate = subs.add_parser("summarize")
    aggregate.add_argument("logs", nargs="+", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        paths = [
            p for item in args.logs for p in (sorted(item.rglob("events.jsonl")) if item.is_dir() else [item])
        ]
        paths = list(dict.fromkeys(p.resolve() for p in paths))
        export(summarize([read_events(p) for p in paths]), args.output)
        return
    events = read_events(args.log)
    recorder = Recorder(path=args.log)
    session = next(e["experiment"] for e in events if e["type"] == "session")
    recorder.session_id = session["session_id"]
    if args.command == "inspect":
        for event in events:
            if event["type"] in {"trial_start", "trial_end", "gate", "semantic_result"}:
                print(json.dumps(event, ensure_ascii=False))
    elif args.command == "gate":
        if not any(e["type"] == "gate" and e["event_id"] == args.event for e in events):
            parser.error("Unknown gate event")
        recorder.write("gate_label", gate_event_id=args.event, truth=args.truth)
    elif args.command == "semantic":
        if not any(e["type"] == "semantic_result" and e["sample_id"] == args.sample for e in events):
            parser.error("Unknown sample")
        values = {
            k: getattr(args, k) for k in ("position", "yaw", "joint", "usable") if getattr(args, k) is not None
        }
        if not values:
            parser.error("Provide at least one independently evaluated field")
        combined = {}
        for event in events:
            if event["type"] == "semantic_label" and event["sample_id"] == args.sample:
                combined.update(event["values"])
        combined.update(values)
        if combined.get("joint") and not (combined.get("position") and combined.get("yaw")):
            parser.error("Joint correctness requires both individual roles correct")
        recorder.write("semantic_label", sample_id=args.sample, values=values)
    else:
        start = next((e for e in events if e["type"] == "trial_start" and e["trial_id"] == args.trial), None)
        if start is None:
            parser.error("Unknown trial")
        if args.progress is not None and not 0 <= args.progress <= len(session["task"]["milestones"]):
            parser.error("Progress is the valid completed milestone count")
        if (
            args.completion_time is not None
            and not 0 <= args.completion_time <= session["task"]["total_timeout_s"]
        ):
            parser.error("Completion time must be within the task budget")
        values = {
            k: v
            for k, v in dict(
                success=args.success,
                progress=args.progress,
                completion_time_s=args.completion_time,
                nav_success=args.nav_success,
                exclude_reason=args.exclude_reason,
            ).items()
            if v is not None
        }
        if args.condition_file:
            values["condition"] = read_yaml(args.condition_file)
        recorder.write("annotation", generation=start["generation"], values=values)
        if args.close_interrupted and not any(
            e["type"] == "trial_end" and e.get("trial_id") == args.trial for e in events
        ):
            recorder.write(
                "trial_end",
                generation=start["generation"],
                state="interrupted",
                reason="operator_closed_log_offline",
            )


if __name__ == "__main__":
    main()
