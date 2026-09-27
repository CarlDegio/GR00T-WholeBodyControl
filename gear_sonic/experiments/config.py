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


def resolve(path, task_id="T1", case_id=None, *, mission_path=None):
    path = Path(path).resolve()
    spec = read_yaml(path)
    if spec.get("schema") != "sonic.experiment.v1":
        raise ValueError("Expected schema: sonic.experiment.v1")
    unknown = set(spec) - {
        "schema",
        "method",
        "table",
        "runtime_profile",
        "runtime_overlays",
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
    mission_overlay = read_yaml(mission_path) if mission_path is not None else None
    # Named missions own their task identity, including when --task is omitted
    # or a stale command still passes another task number.
    identity = (mission_overlay or {}).get("components", {}).get("mission_metadata", {})
    if not isinstance(identity, dict):
        raise ValueError("mission_metadata must be a mapping")
    task_id = identity.get("task_id", task_id)
    if task_id not in {"T1", "T2", "T3", "T4"}:
        raise ValueError(f"Invalid mission task_id: {task_id!r}")
    tasks = read_yaml(path.parent / spec["tasks_file"])
    task = merge(tasks["defaults"], tasks["tasks"][task_id])
    cases = read_yaml(path.parent / spec["cases_file"])["cases"]
    case_id = case_id or spec["default_case"]
    case = copy.deepcopy(cases[case_id])
    method = spec["method"]
    missing = []
    if mission_overlay is not None and method not in {"full_vln", "nav_direct_vla", "geometric_vla", "baselinelavira", "unilm_nav"}:
        raise ValueError("--mission is supported by full_vln, nav_direct_vla, geometric_vla, baselinelavira and unilm_nav")

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
    overlays = spec.get("runtime_overlays", [])
    if not isinstance(overlays, list) or any(not isinstance(p, str) or not p.strip() for p in overlays):
        raise ValueError("runtime_overlays must be a list of YAML paths relative to the experiment config")
    runtime = read_yaml(runtime_path)
    for overlay in overlays:
        runtime = merge(runtime, read_yaml(path.parent / overlay))
    runtime = merge(runtime, spec.get("runtime_overrides", {}))
    if mission_overlay is not None:
        runtime = merge(runtime, mission_overlay)
    c = runtime["components"]
    flexible = c["lavira"].get("flexible_mission", False)
    if mission_overlay is not None and flexible is not True:
        raise ValueError("--mission requires flexible_mission: true")
    if flexible:
        # The whole mission and each skill's fixed prompt belong to the mission
        # overlay, not the legacy single-operation task/route templates.
        missing = [key for key in missing if key in {f"{case_id}.layout_id", f"{case_id}.start_id"}]
        task.update(vla_prompt=c["lavira"].get("mission", ""), vla_trained_prompt=None,
                    policy_id=None)
        c["lavira"]["manipulation_timeout_seconds"] = float(task["vla_timeout_s"])
    else:
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
        flexible_mission=flexible,
        source_config=str(path),
        models=dict(
            policy_id=task["policy_id"],
            la=c["lavira"]["la_model"],
            va=c["lavira"]["va_model"],
            navdp_checkpoint=c["navdp"]["checkpoint"],
            yoloe=c["base_pose"]["raw_yoloe_model_path"],
        ),
    )
    if flexible:
        experiment.update(
            nearfield_entries=copy.deepcopy(identity.get("nearfield_entries", {})),
            mission=c["lavira"]["mission"], skills=copy.deepcopy(c["lavira"].get("skills", {})),
            mission_source=str(Path(mission_path).resolve()) if mission_path else str(runtime_path),
        )
        experiment["models"]["vla_backends"] = copy.deepcopy(c["vla"].get("backends", {}))
    if experiment["alignment"] == "unilm_nav":
        experiment["models"].pop("yoloe", None)
        experiment["models"]["unilm_nav"] = copy.deepcopy(c.get("unilm_nav", {}))
    if c.get("navigation_policy", {}).get("backend") == "iplanner":
        checkpoint = Path(c["iplanner"]["checkpoint"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(c["iplanner"]["root"]).expanduser() / checkpoint
        experiment["models"].pop("navdp_checkpoint", None)
        experiment["models"].update(navigation_policy="iplanner", iplanner_checkpoint=str(checkpoint))
    experiment["geometric"].setdefault("both_lost_frames", 20)
    experiment["geometric"]["target"] = task.get("geometry_target", "")
    if experiment["alignment"] not in {"dual", "head", "none", "geometric", "unilm_nav"} or experiment["entry_stage"] not in {
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
    if experiment["alignment"] == "geometric" and not flexible:
        required(experiment["geometric"]["target"], f"{task_id}.geometry_target")
    c["experiment"] = experiment
    runtime["profile"] = f"icra_{method}_{task_id}_{case_id}"
    return runtime, missing


def launch(default_name, argv=None):
    parser = argparse.ArgumentParser(description="Start one ICRA condition; press n in the normal operator pane.")
    parser.add_argument("--config", type=Path, default=CONFIGS / f"{default_name}.yaml")
    parser.add_argument("--task", choices=("T1", "T2", "T3", "T4"), default="T1")
    parser.add_argument("--case")
    parser.add_argument("--mission", type=Path, help="Flexible mission overlay shared by the main methods")
    parser.add_argument("--demo", action="store_true", help="Cache demo video until trial end and save full LA/VA evidence")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, help="Offline semantic sample manifest")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "outputs" / "experiments")
    args = parser.parse_args(argv)
    profile, missing = resolve(args.config, args.task, args.case, mission_path=args.mission)
    if args.demo:
        if profile["components"]["experiment"]["table"] == "semantic":
            parser.error("--demo requires a live experiment, not semantic_roles")
        profile["components"]["experiment"]["demo"] = True
    # Validate the compiled worker profiles without connecting to any service.
    from gear_sonic.scripts.launch_inference import load_inference_launch_config
    from gear_sonic.utils.inference.base_pose.agent import load_base_pose_config
    from gear_sonic.utils.inference.lavira.service import load_lavira_config
    from gear_sonic.utils.inference.vla.service import load_inference_config
    from gear_sonic.runtime.profile import load_runtime_profile
    from gear_sonic.utils.inference.lavira.mission import validate_mission_profile
    from gear_sonic.utils.inference.iplanner.config import load_iplanner_server_config, navigation_backend

    with tempfile.TemporaryDirectory(prefix="sonic-experiment-validate-") as directory:
        validation_path = Path(directory) / "runtime.yaml"
        validation_path.write_text(yaml.safe_dump(profile))
        compiled = load_runtime_profile(validation_path)
        validate_mission_profile(compiled)
        for loader in (
            load_inference_launch_config,
            load_base_pose_config,
            load_lavira_config,
            load_inference_config,
        ):
            loader(str(validation_path))
        if profile["components"]["experiment"]["alignment"] == "unilm_nav":
            from gear_sonic.runtime.profile import load_component_config
            from gear_sonic.utils.inference.base_pose.unilm import UniLMConfig, robot_geometry

            unilm = load_component_config(UniLMConfig, "unilm_nav", str(validation_path))
            if not unilm.enabled:
                raise ValueError("unilm_nav experiment requires components.unilm_nav.enabled")
            robot_geometry(unilm.urdf_path, unilm.camera_height_m)
        if navigation_backend(compiled) == "iplanner":
            from gear_sonic.utils.inference.iplanner.control import load_navigation_config

            experiment = profile["components"]["experiment"]
            launcher = profile["components"]["launcher"]
            if (experiment["navigation_backend"] != "lavira" or experiment["entry_stage"] != "navigation"
                    or not launcher["keyboard_planner"] or launcher["planner_input"] != "lavira"):
                raise ValueError("iPlanner requires the LaViRA planner and a navigation entry stage")
            server = load_iplanner_server_config(compiled)
            load_navigation_config(compiled)
            missing.extend(f"{label}: {path}" for path, label in server.required_assets() if not path.is_file())
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
    run_dir = (args.output / f"{exp['method']}_{exp['task_id']}_{exp['case_id']}_{session_id}").resolve()
    run_dir.mkdir(parents=True)
    exp.update(session_id=session_id, run_dir=str(run_dir))
    if exp["alignment"] == "unilm_nav":
        profile["components"]["unilm_nav"]["output_root"] = str(run_dir / "unilm_nav")
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
