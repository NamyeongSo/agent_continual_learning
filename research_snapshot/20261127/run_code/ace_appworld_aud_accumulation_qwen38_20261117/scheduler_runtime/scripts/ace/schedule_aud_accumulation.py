# SPDX-License-Identifier: Apache-2.0
"""Queue matched AUD accumulation for thinking and non-thinking after both A runs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ace import run_context_proposal_bottleneck as base  # noqa: E402
from scripts.ace import run_reflection_accumulation as accumulation  # noqa: E402
from scripts.ace.schedule_nonthinking_replication import predecessor_ready, verify_sources  # noqa: E402

MATCHED_KEYS = ("model", "model_settings", "appworld_policy_sha256", "initial_playbook", "actor_limits",
                "turn_limit_scoring", "task_orders", "evaluation_repeats", "evaluation_workers",
                "training_context_workers", "selection_score", "tie_break")


def reuse_selected_bank(source, target):
    """Import only bank/choices, preserving exact ties and all A sampling decisions."""
    source, target = Path(source), Path(target)
    if not predecessor_ready(source):
        raise ValueError("A accumulation must be complete before importing matched selections")
    before, after = base.read(source / "protocol.json"), base.read(target / "protocol.json")
    if before["playbook_edit_mode"] != "add_only" or after["playbook_edit_mode"] != "add_update_delete":
        raise ValueError("Expected a matched A-to-AUD comparison")
    if any(before[key] != after[key] for key in MATCHED_KEYS):
        raise ValueError("A and AUD differ beyond the edit mode")
    if (target / "selection_reuse_manifest.json").exists():
        manifest = base.read(target / "selection_reuse_manifest.json")
        for name, expected in manifest["files"].items():
            if name == "bank_ready.json" and not (target / name).exists():
                if base.digest((source / name).read_bytes()) != expected:
                    raise ValueError("Source bank marker changed")
                shutil.copy2(source / name, target / name)
            if (base.digest((target / name).read_bytes()) != expected
                    or base.digest((source / name).read_bytes()) != expected):
                raise ValueError("Imported selection or source changed")
        return manifest
    if list((target / "conditions").glob("*/checkpoints/*.json")):
        raise ValueError("Cannot import selections into an already trained cohort")
    for path, expected in base.read(source / "bank_source_hashes.json").items():
        if base.digest(Path(path).read_bytes()) != expected:
            raise ValueError(f"Reflection-bank evidence changed: {path}")
    names = ["reflection_bank.json", "bank_source_hashes.json", "bank_ready.json"]
    for seed in (42, 43, 44):
        tasks = before["task_orders"][str(seed)]["train"]
        for condition in accumulation.conditions(seed):
            name = f"conditions/{condition['id']}/selections.json"
            choices = base.read(source / name)
            if [c["candidate"]["task_id"] for c in choices] != tasks:
                raise ValueError("Selection/task order mismatch")
            for choice in choices:
                c = choice["candidate"]
                if base.digest(c["reflection"]) != c["reflection_sha256"]:
                    raise ValueError("Selected Reflection changed")
            names.append(name)
    hashes = {}
    for name in names:
        if name == "bank_ready.json":
            continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
        hashes[name] = base.digest(destination.read_bytes())
    hashes["bank_ready.json"] = base.digest((source / "bank_ready.json").read_bytes())
    manifest = {"source": str(source.resolve()), "imported_at": base.now(), "files": hashes,
                "reflection_regeneration": False, "selection_regeneration": False,
                "learned_context_transfer": False}
    base.write(target / "selection_reuse_manifest.json", manifest)
    # Commit this marker last; the learner refuses to regenerate matched choices.
    shutil.copy2(source / "bank_ready.json", target / "bank_ready.json")
    return manifest


def prepare(thinking, nonthinking_pipeline, output):
    thinking, nonthinking_pipeline, output = map(Path.resolve, map(Path, (thinking, nonthinking_pipeline, output)))
    if output.exists():
        raise FileExistsError(output)
    nonthinking = Path(base.read(nonthinking_pipeline / "plan.json")["accumulation"])
    sources = {"thinking": thinking, "nonthinking": nonthinking}
    output.mkdir(parents=True, mode=0o700)
    runs = {}
    for mode, source in sources.items():
        previous = base.read(source / "protocol.json")
        if previous["playbook_edit_mode"] != "add_only":
            raise ValueError("Comparison source must remain ADD-only")
        target = output / mode
        accumulation.prepare(target, previous["source_reflection_run"], playbook_edit_mode="add_update_delete")
        protocol = base.read(target / "protocol.json")
        if any(previous[key] != protocol[key] for key in MATCHED_KEYS):
            raise ValueError(f"Unmatched {mode} AUD configuration")
        protocol.update(comparison_add_only_run=str(source), selection_source_accumulation_run=str(source),
                        selection_reuse="exact saved bank and seed-specific selections; no reranking",
                        reported_metrics=["PCT", "TGC"], tgc_metric="appworld_tgc_v1")
        base.write(target / "protocol.json", protocol)
        base.write(target / "status.json", {"state": "scheduled", "phase": "waiting_for_nonthinking_completion",
                                            "updated_at": base.now()})
        runs[mode] = {"source": str(source), "target": str(target),
                      "source_protocol_sha256": base.digest((source / "protocol.json").read_bytes())}
    runtime = output / "scheduler_runtime/scripts/ace"
    runtime.mkdir(parents=True)
    for name in (Path(__file__).name, "schedule_nonthinking_replication.py", "run_reflection_accumulation.py",
                 "run_context_proposal_bottleneck.py"):
        shutil.copy2(ROOT / "scripts/ace" / name, runtime / name)
    plan = {"created_at": base.now(), "nonthinking_pipeline": str(nonthinking_pipeline), "runs": runs,
            "phase_order": ["wait_for_all_nonthinking_A", "thinking_AUD", "nonthinking_AUD"],
            "planned_actor_trajectories": 5400, "planned_curator_steps": 1800,
            "planned_new_reflections": 0, "gpu_physical_id": 1, "port": 8006,
            "scheduler_source_hashes": {str(p.relative_to(output / "scheduler_runtime")): base.digest(p.read_bytes())
                                        for p in runtime.glob("*.py")}}
    base.write(output / "plan.json", plan)
    base.write(output / "status.json", {"state": "scheduled", "phase": "waiting_for_nonthinking_completion",
                                        "updated_at": base.now()})
    return plan


def run(output):
    from filelock import FileLock

    output = Path(output).resolve()
    with FileLock(str(output / "scheduler.lock"), timeout=0):
        plan = base.read(output / "plan.json")
        for name, expected in plan["scheduler_source_hashes"].items():
            if base.digest((output / "scheduler_runtime" / name).read_bytes()) != expected:
                raise ValueError("AUD scheduler source changed")
        base.write(output / "process.json", {"pid": os.getpid(), "started_at": base.now(), "command": sys.argv})
        phase = "waiting_for_nonthinking_completion"
        try:
            pipeline = Path(plan["nonthinking_pipeline"])
            while base.read(pipeline / "status.json").get("state") != "completed":
                if (output / "STOP").exists():
                    raise InterruptedError("STOP requested")
                base.write(output / "status.json", {"state": "scheduled", "phase": phase, "pid": os.getpid(),
                                                    "dependency": base.read(pipeline / "status.json"),
                                                    "updated_at": base.now()})
                time.sleep(30)
            for mode in ("thinking", "nonthinking"):
                phase = mode + "_AUD"
                if (output / "STOP").exists():
                    raise InterruptedError("STOP requested")
                source, target = (Path(plan["runs"][mode][k]) for k in ("source", "target"))
                if base.read(target / "status.json").get("state") == "completed":
                    if not predecessor_ready(target):
                        raise ValueError("AUD completion evidence missing")
                    continue
                if base.digest((source / "protocol.json").read_bytes()) != plan["runs"][mode]["source_protocol_sha256"]:
                    raise ValueError("A comparison protocol changed")
                verify_sources(target)
                reuse_selected_bank(source, target)
                base.write(output / f"endpoint_{mode}.json", base.check_endpoint(base.read(target / "protocol.json")))
                command = [sys.executable, str(target / "source_snapshot/scripts/ace/run_reflection_accumulation.py"),
                           "--run", "--output", str(target)]
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES="1", PYTHONUNBUFFERED="1")
                with (target / "scheduler.log").open("a") as log:
                    child = subprocess.Popen(command, cwd=pipeline.parent.parent, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    base.write(output / "status.json", {"state": "running", "phase": phase, "child_pid": child.pid,
                                                        "command": command, "updated_at": base.now()})
                    result = child.wait()
                if result != 0 or not predecessor_ready(target):
                    raise RuntimeError(f"{phase} did not complete; next experiment will not start")
            base.write(output / "status.json", {"state": "completed", "phase": "all_complete", "at": base.now()})
        except BaseException as exc:
            base.write(output / "status.json", {"state": "error", "phase": phase,
                                                "error": str(exc), "at": base.now()})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thinking", type=Path)
    parser.add_argument("--nonthinking-pipeline", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.thinking is None or args.nonthinking_pipeline is None:
            parser.error("--prepare requires --thinking and --nonthinking-pipeline")
        prepare(args.thinking, args.nonthinking_pipeline, args.output)
    else:
        run(args.output)
