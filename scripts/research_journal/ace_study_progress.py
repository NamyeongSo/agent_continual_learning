# SPDX-License-Identifier: Apache-2.0
"""Project registered ACE Train-only study progress without importing experiment runners."""

from __future__ import annotations

import json

STAGES = (
    ("initial_rollouts", "초기 추론", 500, "회"),
    ("reflection_update", "Reflection → Update", 500, "회"),
    ("updated_context_rollouts", "후보 Context 평가", 1500, "회"),
    ("all_context_training", "Context 학습", 30, "개"),
    ("train_final", "최종 Train 평가", 1500, "회"),
)


def projected_list(files, path, fields=None):
    """Cache counts or selected scalar fields, excluding prompts and reflection text."""
    key = (path, ("_ace_study_list", fields))
    previous = files.entries.get(key)
    try:
        stat = path.stat()
        stamp = stat.st_mtime_ns, stat.st_size
        if previous and previous[0] == stamp:
            return previous[1]
        data = json.loads(path.read_bytes())
        if not isinstance(data, list):
            raise ValueError("Expected a list of saved results")
        value = len(data) if fields is None else [
            {name: row[name] for name in fields if name in row} for row in data
        ]
        files.entries[key] = stamp, value
        return value
    except FileNotFoundError:
        return 0 if fields is None else []
    except (OSError, ValueError, TypeError, UnicodeError):
        files.issues += 1
        return previous[1] if previous else (0 if fields is None else [])


def ace_studies(base_dir, files, allowed, live_pid):
    studies = []
    for marker in sorted((base_dir / "outputs").glob("*/ace_study_monitor.json")):
        root = marker.parent
        if not allowed(root):
            continue
        monitor = files.read(marker, ("kind", "journal_id", "label", "initial_metrics_file"), {})
        if monitor.get("kind") != "ace_context_proposal_train_only_v1":
            continue
        plan = files.read(root / "plan.json", ("model", "modes", "workers", "training_contexts",
                                                "train_evaluations", "seeds"), {})
        state = files.read(root / "status.json", ("state", "phase", "updated_at", "completed", "total",
                                                   "running", "running_contexts", "active_contexts", "pending",
                                                   "worker_limit", "training_worker_limit", "by_mode", "errors"), {})
        owner = files.read(root / "control/process.json", ("pid",), {})
        alive = live_pid(owner.get("pid"), root)
        receipt = files.read(root / "completed.json", ("contexts", "train_evaluations"), {})
        final_scores = projected_list(files, root / "train_final_mean_std.json", (
            "mode", "strategy", "seeds", "pct_mean", "pct_std", "tgc_mean", "tgc_std",
        ))
        complete = state.get("state") == "completed" and (
            receipt.get("contexts") == plan.get("training_contexts", 30)
            and receipt.get("train_evaluations") == plan.get("train_evaluations", 1500)
            and len(final_scores) == 10
        )
        status = "completed" if complete else "error" if state.get("state") == "error" else (
            "running" if alive else "planned" if state.get("state") in {None, "planned", "scheduled"} else "stopped"
        )
        stages = []
        modes = plan.get("modes", ["thinking", "nonthinking"])
        for index, (name, label, total, unit) in enumerate(STAGES):
            if index < 3:
                done = sum(projected_list(files, root / mode / "reflection" / f"stage{index + 1}_results.json")
                           for mode in modes)
            elif name == "all_context_training":
                jobs = projected_list(files, root / "control/training_plan.json", ("path", "condition"))
                done = 0
                for job in jobs:
                    path = root / job["path"] / "conditions" / job["condition"]["id"] / "training_completed.json"
                    done += allowed(path) and path.is_file()
            else:
                saved = projected_list(files, root / "train_final_by_seed.json", ("tasks",))
                done = sum(row.get("tasks", 0) for row in saved)
            current = state.get("phase") == name and not complete
            if current and name != "all_context_training":
                done = max(done, state.get("completed", 0))
            running = state.get("running", state.get("active_contexts", state.get("running_contexts", 0)))
            running = running if current and alive else 0
            stages.append({"id": name, "label": label, "done": min(total, done), "total": total, "unit": unit,
                           "current": current, "running": running})
        initial_path = root / monitor.get("initial_metrics_file", "analysis/initial_metrics.json")
        initial = projected_list(files, initial_path, (
            "mode", "tasks", "rollouts", "pct_percent", "tgc_percent", "pass_at_5_percent", "pass_power_5_percent",
        )) if allowed(initial_path) else []
        by_mode = {mode: {key: values.get(key, 0) for key in ("completed", "total")}
                   for mode, values in state.get("by_mode", {}).items()}
        studies.append({"id": root.name, "journal_id": monitor.get("journal_id"),
                        "label": monitor.get("label", root.name), "model": plan.get("model"),
                        "state": status, "phase": state.get("phase"), "updated_at": state.get("updated_at"),
                        "controller_alive": alive, "workers": state.get(
                            "worker_limit", state.get("training_worker_limit", plan.get("workers"))),
                        "errors": len(state.get("errors", [])), "stages": stages, "by_mode": by_mode,
                        "initial_scores": initial, "final_scores": final_scores})
    return studies
