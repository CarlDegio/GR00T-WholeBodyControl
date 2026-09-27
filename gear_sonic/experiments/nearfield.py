"""Per-trial ALIGN → VLA entry, matching agent_full's near_dual experiments."""

import copy


def nearfield_experiment(experiment, *, start_key="m"):
    if start_key not in {"m", "t"}:
        raise ValueError("Unsupported nearfield key")
    if experiment.get("method") != "full_vln":
        raise ValueError("nearfield entry requires full_vln")
    result = copy.deepcopy(experiment)
    task = result["task"]
    if result.get("flexible_mission"):
        skills = result.get("skills", {})
        entries = result.get("nearfield_entries", {})
        entry = entries.get(start_key)
        if entry is not None:
            name = entry["skill"]
            if name not in skills:
                raise ValueError(f"Unknown nearfield skill: {name}")
            skill = skills[name]
        elif start_key == "m" and len(skills) == 1:
            name, skill = next(iter(skills.items()))
            entry = {}
        else:
            raise ValueError(f"No nearfield operation configured for key {start_key}")
        task.update(
            vla_prompt=skill.get("vla_prompt", skill.get("prompt")),
            vla_trained_prompt=skill["vla_trained_prompt"],
            global_target=skill.get("geometry_target", ""),
            geometry_target=skill.get("geometry_target", ""),
            vla_timeout_s=float(skill.get("timeout_seconds", task["vla_timeout_s"])),
        )
        result["nearfield_skill"] = name
        result["nearfield_base_pose"] = copy.deepcopy(skill.get("base_pose", {}))
        result["nearfield_expected_postcondition"] = entry.get(
            "expected_postcondition", skill.get("expected_postcondition", task["vla_prompt"]))
        if "milestones" in entry:
            task["milestones"] = copy.deepcopy(entry["milestones"])
        result["nearfield_vla_parameters"] = dict(
            skill_key=skill["backend"], exit_policy=skill["exit"],
            completion_settle_seconds=skill.get("completion_settle_seconds", 0.0),
            timeout_seconds=task["vla_timeout_s"], on_timeout="fail",
        )
    elif start_key != "m":
        raise ValueError(f"No nearfield operation configured for key {start_key}")
    if not task.get("vla_prompt") or not task.get("vla_trained_prompt"):
        raise ValueError("nearfield entry requires semantic and trained VLA prompts")
    result.update(parent_method="full_vln", method="near_dual", table="nearfield",
                  entry_stage="alignment", flexible_mission=False,
                  mission=task["vla_prompt"], start_key=start_key)
    return result
