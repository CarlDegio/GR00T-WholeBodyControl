"""Resolve one experiment/task/condition into the existing runtime profile."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "config" / "experiments"


def merge(base, overlay):
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        out[key] = (
            merge(out[key], value)
            if isinstance(value, dict) and isinstance(out.get(key), dict)
            else copy.deepcopy(value)
        )
    return out


def read_yaml(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def settings(profile):
    return dict(profile.components.get("experiment", {}))


def resolve(path, task_id="T1", case_id=None):
    path = Path(path).resolve()
    spec = read_yaml(path)
    if spec.get("schema") != "sonic.experiment.v1":
        raise ValueError("Expected schema: sonic.experiment.v1")
    unknown = set(spec) - {
        "schema",
        "method",
        "table",
        "runtime_profile",
        "tasks_file",
        "cases_file",
        "default_case",
        "navigation",
        "navigation_backend",
        "entry_stage",
        "alignment",
        "gates",
        "runtime_overrides",
        "gate_under_test",
        "geometric",
        "navila",
    }
    if unknown:
        raise ValueError(f"Unknown experiment fields: {sorted(unknown)}")
    tasks = read_yaml(path.parent / spec["tasks_file"])
    task = merge(tasks["defaults"], tasks["tasks"][task_id])
    cases = read_yaml(path.parent / spec["cases_file"])["cases"]
    case_id = case_id or spec["default_case"]
    case = copy.deepcopy(cases[case_id])
    method = spec["method"]
    missing = []

    def required(value, label):
        if value is None or not str(value).strip() or str(value).startswith("REQUIRED"):
            missing.append(label)
        return value

    semantic = required(task.get("vla_prompt"), f"{task_id}.vla_prompt")
    trained = required(task.get("vla_trained_prompt"), f"{task_id}.vla_trained_prompt")
    required(task.get("policy_id"), f"{task_id}.policy_id")
    nav = spec.get("navigation", "vln")
    if spec["table"] == "semantic":
        missing = []  # Offline samples carry their own semantic instructions.
        mission = semantic
    elif spec.get("entry_stage", "navigation") != "navigation":
        mission = semantic
    elif nav == "object_nav":
        mission = task["objectnav_instruction"]
    else:
        instructions = case.get("navigation_instructions", {})
        mission = required(instructions.get(task_id), f"{case_id}.navigation_instructions.{task_id}")
    if spec["table"] != "semantic":
        for key in ("layout_id", "start_id"):
            required(case.get(key), f"{case_id}.{key}")
    runtime_path = (path.parent / spec["runtime_profile"]).resolve()
    runtime = merge(read_yaml(runtime_path), spec.get("runtime_overrides", {}))
    c = runtime["components"]
    c["lavira"].update(
        mission=mission or "REQUIRED",
        global_target=task["global_target"],
        manipulation_prompt=semantic or "REQUIRED",
        navigation_mode=nav,
        manipulation_timeout_seconds=float(task["vla_timeout_s"]),
        nav_handoff_max_depth_m=3.0,
    )
    c["vla"]["prompt"] = trained or "REQUIRED"
    c["launcher"].update(
        base_pose_enabled=spec.get("alignment", "dual") != "none", data_exporter=False, slam_debug=False
    )
    c["base_pose"]["raw_diagnostic_image_interval_frames"] = 0
    for name in ("total_timeout_s", "vla_timeout_s", "fixed_manipulation_s", "recovery_budget_s"):
        if not 0 < float(task[name]) <= 86400:
            raise ValueError(f"Invalid {name}")
    if float(task["fixed_manipulation_s"]) > float(task["vla_timeout_s"]):
        raise ValueError("fixed_manipulation_s cannot exceed vla_timeout_s")
    gates = merge({"nav": True, "align": True, "completion": True}, spec.get("gates", {}))
    if set(gates) != {"nav", "align", "completion"}:
        raise ValueError("Only nav, align, completion gates are supported")
    if any(type(v) is not bool for v in gates.values()):
        raise ValueError("Gate settings must be YAML booleans")
    experiment = dict(
        method=method,
        table=spec["table"],
        task_id=task_id,
        case_id=case_id,
        task=task,
        condition=case,
        gates=gates,
        entry_stage=spec.get("entry_stage", "navigation"),
        alignment=spec.get("alignment", "dual"),
        navigation_backend=spec.get("navigation_backend", "lavira"),
        geometric=merge(tasks.get("geometric", {}), spec.get("geometric", {})),
        navila=merge(tasks.get("navila", {}), spec.get("navila", {})),
        gate_under_test=spec.get("gate_under_test"),
        minimal_logging=True,
        source_config=str(path),
        models=dict(
            policy_id=task["policy_id"],
            la=c["lavira"]["la_model"],
            va=c["lavira"]["va_model"],
            navdp_checkpoint=c["navdp"]["checkpoint"],
            yoloe=c["base_pose"]["raw_yoloe_model_path"],
        ),
    )
    experiment["geometric"].setdefault("both_lost_frames", 20)
    experiment["geometric"]["target"] = task.get("geometry_target", "")
    if experiment["alignment"] not in {"dual", "head", "none", "geometric"} or experiment["entry_stage"] not in {
        "navigation",
        "alignment",
        "manipulation",
    }:
        raise ValueError("Unsupported alignment or entry_stage")
    if experiment["navigation_backend"] not in {"lavira", "navila"} or nav not in {"vln", "object_nav"}:
        raise ValueError("Unsupported navigation backend or mode")
    for group, names in (
        (
            "geometric",
            (
                "distance_m",
                "longitudinal_tolerance_m",
                "angle_tolerance_deg",
                "coarse_wz",
                "vx",
                "fine_wz",
                "timeout_s",
                "stable_frames",
                "both_lost_frames",
            ),
        ),
        ("navila", ("timeout_s", "vx", "wz", "max_actions")),
    ):
        for name in names:
            value = float(experiment[group][name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {group}.{name}")
    geo, nv = experiment["geometric"], experiment["navila"]
    for value in (geo["stable_frames"], geo["both_lost_frames"], nv["max_actions"], nv["port"]):
        if isinstance(value, bool) or not float(value).is_integer():
            raise ValueError("Frame counts, action budget and model port must be integers")
    if geo["vx"] > 0.4 or max(geo["coarse_wz"], geo["fine_wz"]) > 0.3 or nv["vx"] > 0.3 or nv["wz"] > 0.3:
        raise ValueError("Experiment speed exceeds the shared gateway envelope")
    if not 1 <= int(nv["port"]) <= 65535 or not str(nv["host"]).strip():
        raise ValueError("Invalid NaVILA model endpoint")
    disturbance = case.get("perturbation", {})
    if disturbance.get("enabled"):
        required(disturbance.get("description"), f"{case_id}.perturbation.description")
        required(disturbance.get("amplitude"), f"{case_id}.perturbation.amplitude")
    if experiment["alignment"] == "geometric":
        required(experiment["geometric"]["target"], f"{task_id}.geometry_target")
    c["experiment"] = experiment
    runtime["profile"] = f"icra_{method}_{task_id}_{case_id}"
    return runtime, missing


def launch(default_name, argv=None):
    parser = argparse.ArgumentParser(description="Start one ICRA condition; press n in the normal operator pane.")
    parser.add_argument("--config", type=Path, default=CONFIGS / f"{default_name}.yaml")
    parser.add_argument("--task", choices=("T1", "T2", "T3", "T4"), default="T1")
    parser.add_argument("--case")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, help="Offline semantic sample manifest")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "outputs" / "experiments")
    args = parser.parse_args(argv)
    profile, missing = resolve(args.config, args.task, args.case)
    # Validate the compiled worker profiles without connecting to any service.
    from gear_sonic.scripts.launch_inference import load_inference_launch_config
    from gear_sonic.utils.inference.base_pose.agent import load_base_pose_config
    from gear_sonic.utils.inference.lavira.service import load_lavira_config
    from gear_sonic.utils.inference.vla.service import load_inference_config

    with tempfile.TemporaryDirectory(prefix="sonic-experiment-validate-") as directory:
        validation_path = Path(directory) / "runtime.yaml"
        validation_path.write_text(yaml.safe_dump(profile))
        for loader in (
            load_inference_launch_config,
            load_base_pose_config,
            load_lavira_config,
            load_inference_config,
        ):
            loader(str(validation_path))
    exp = profile["components"]["experiment"]
    if exp["table"] == "semantic" and (args.manifest is None or not args.manifest.is_file()):
        missing.append("--manifest (existing offline sample manifest)")
    elif exp["table"] == "semantic":
        from .offline import load_samples

        samples = [s for s in load_samples(args.manifest) if s["task_id"] == args.task]
        if not samples:
            missing.append(f"--manifest: no {args.task} samples")
        for sample in samples:
            for name in ("rgb", "depth"):
                if sample.get(name) and not Path(sample[name]).is_file():
                    missing.append(f"{sample['sample_id']}.{name}: file not found")
    if args.dry_run:
        print(
            yaml.safe_dump(
                {
                    "ready": not missing,
                    "missing": missing,
                    "experiment": exp,
                    "startup": "offline semantic dataset; no robot process"
                    if exp["table"] == "semantic"
                    else "existing launch_inference.py; operator n starts each trial",
                },
                sort_keys=False,
                allow_unicode=True,
            )
        )
        return 0
    if missing:
        parser.error("Fill required experiment inputs before launch: " + ", ".join(missing))
    session_id = uuid.uuid4().hex[:12]
    run_dir = (args.output / f"{exp['method']}_{args.task}_{exp['case_id']}_{session_id}").resolve()
    run_dir.mkdir(parents=True)
    exp.update(session_id=session_id, run_dir=str(run_dir))
    exp["config_sha256"] = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    exp["git_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True
    ).stdout.strip()
    exp["git_dirty"] = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT.parent, capture_output=True, text=True
        ).stdout.strip()
    )
    resolved = run_dir / "runtime.yaml"
    exp["runtime_profile_path"] = str(resolved)
    resolved.write_text(yaml.safe_dump(profile, sort_keys=False, allow_unicode=True))
    from gear_sonic.runtime.profile import load_runtime_profile

    from .recording import Recorder

    Recorder(load_runtime_profile(resolved)).write("session", experiment=exp)
    if exp["table"] == "semantic":
        from .offline import evaluate_manifest

        evaluate_manifest(resolved, args.manifest)
        return 0
    from gear_sonic.scripts.launch_inference import load_inference_launch_config, main

    main(load_inference_launch_config(resolved))
    from .offline import enrich_snapshots

    # Detaching tmux may leave the robot running. Postprocessing is automatic
    # only after the inference session is gone; otherwise the offline CLI is used.
    if subprocess.run(["tmux", "has-session", "-t", "sonic_inference"], capture_output=True).returncode != 0:
        enrich_snapshots(resolved)
    else:
        print(f"首帧已归档。结束 inference 会话后生成 mask：python -m gear_sonic.experiments.offline {resolved}")
    return 0
