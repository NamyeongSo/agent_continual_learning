# SPDX-License-Identifier: Apache-2.0
"""Freeze and queue the matched non-thinking Reflection + accumulation pipeline."""

from __future__ import annotations

import argparse
import copy
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


def nonthinking_settings(original):
    settings = copy.deepcopy(original)
    settings.pop("reasoning_effort", None)
    settings.update(temperature=0.7, top_p=0.80, presence_penalty=1.5)
    settings.setdefault("extra_body", {}).update(top_k=20, min_p=0.0, repetition_penalty=1.0)
    settings["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False, "preserve_thinking": False}
    return settings


def verify_sources(output):
    output = Path(output)
    for name, expected in base.read(output / "source_hashes.json").items():
        if base.digest((output / "source_snapshot" / name).read_bytes()) != expected:
            raise ValueError(f"Frozen source changed: {output.name}/{name}")


def freeze_reflection(source, output):
    """Copy immutable inputs/code only; never copy thinking outcomes or learned contexts."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    verify_sources(source)
    protocol = base.read(source / "protocol.json")
    if protocol.get("stage3_repeats_per_context") != 3 or len(protocol["task_ids"]) != 50:
        raise ValueError("Expected the approved train50, three-repeat reflection protocol")
    if protocol["turn_limit_scoring"] != "official_pct_at_limit":
        raise ValueError("The matched cohort must retain final-state turn-limit PCT")
    if protocol["appworld_policy_sha256"] != base.digest((source / "appworld_policy.txt").read_bytes()):
        raise ValueError("Environment policy changed")
    output.mkdir(parents=True, mode=0o700)
    for name in ("references.json", "task_splits.json", "appworld_policy.txt", "initial_playbook.txt",
                 "benchmark_dependency_hashes.json"):
        shutil.copy2(source / name, output / name)
    for name in ("source_snapshot", "benchmark_dependencies"):
        shutil.copytree(source / name, output / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Preserve the baseline's core bytes. Only the two orchestration modules now read
    # explicit role overrides from the protocol instead of hardcoding thinking/T1.
    shutil.copy2(ROOT / "scripts/ace/run_context_proposal_bottleneck.py",
                 output / "source_snapshot/scripts/ace/run_context_proposal_bottleneck.py")
    settings = nonthinking_settings(protocol["model_settings"])
    base.write(output / "model_settings.json", settings)
    for key in ("source_initial_bank", "initial_trajectories_regenerated", "rescored_initial_rollouts",
                "completed_proposals_reused"):
        protocol.pop(key, None)
    protocol.update(
        status="prepared", prepared_at=base.now(), protocol_revision=3, generation_mode="non_thinking",
        model_settings=settings, model_settings_source=str(output / "model_settings.json"),
        model_settings_source_sha256=base.digest((output / "model_settings.json").read_bytes()),
        benchmark_dependency_overlay=str(output / "benchmark_dependencies"),
        actor_trajectories_planned=1750, initial_trajectories_reused=0, proposals_reused=0,
        matched_thinking_reflection_run=str(source),
        matched_thinking_protocol_sha256=base.digest((source / "protocol.json").read_bytes()),
        generation_settings_authority="User instruction 2026-11-17: Instruct/non-thinking Qwen3.8-27B",
        reported_metrics=["PCT", "TGC"], tgc_metric="appworld_tgc_v1",
    )
    for stage, count in ((1, 250), (2, 500), (3, 1500)):
        jobs = base.jobs_for_stage(protocol["task_ids"], stage, 3)
        if len(jobs) != count or protocol["stages"][stage - 1]["total_jobs"] != count:
            raise ValueError("Queue sizes differ from the matched experiment")
        base.write(output / f"stage{stage}_queue.json", jobs)
    base.write(output / "protocol.json", protocol)
    base.write(output / "source_hashes.json", {
        str(p.relative_to(output / "source_snapshot")): base.digest(p.read_bytes())
        for p in (output / "source_snapshot").rglob("*") if p.is_file()
    })
    base.write(output / "status.json", {"state": "scheduled", "stage": 0,
                                        "phase": "waiting_for_thinking_pipeline", "updated_at": base.now()})


def predecessor_ready(output):
    """A process exit or one completed seed cannot release the dependency barrier."""
    output = Path(output)
    if base.read(output / "status.json").get("state") != "completed":
        return False
    protocol = base.read(output / "protocol.json")
    for seed in (42, 43, 44):
        marker = output / f"seed{seed}/completed.json"
        if not marker.exists() or base.read(marker).get("evaluations") != 900:
            raise ValueError(f"Completed predecessor missing seed{seed} completion evidence")
        for phase in ("train_final", "bwt_checkpoint", "val"):
            rows = base.read(output / f"seed{seed}/{phase}_results.json")
            split = "val" if phase == "val" else "train"
            expected = {(c["id"], t) for c in accumulation.conditions(seed)
                        for t in protocol["task_orders"][str(seed)][split]}
            keys = {(r["condition_id"], r["task_id"]) for r in rows}
            if len(rows) != 300 or len(keys) != 300 or keys != expected:
                raise ValueError(f"Incomplete predecessor seed{seed}/{phase}")
    return True


def prepare(predecessor, output):
    predecessor, output = Path(predecessor).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    original = base.read(predecessor / "protocol.json")
    source = Path(original["source_reflection_run"])
    if original["seeds"] != [42, 43, 44]:
        raise ValueError("Unexpected predecessor seeds")
    output.mkdir(parents=True, mode=0o700)
    reflection, learned = output / "reflection", output / "accumulation"
    freeze_reflection(source, reflection)
    accumulation.prepare(learned, reflection)
    new = base.read(learned / "protocol.json")
    for key in ("task_orders", "appworld_policy_sha256", "initial_playbook", "actor_limits", "turn_limit_scoring",
                "evaluation_repeats", "evaluation_workers", "training_context_workers", "playbook_edit_mode"):
        if new[key] != original[key]:
            raise ValueError(f"Replication changed matched setting: {key}")
    template = Path("/home/thskadud/chat_templates/qwen3_8-27B.jinja")
    plan = {"prepared_at": base.now(), "predecessor": str(predecessor),
            "predecessor_protocol_sha256": base.digest((predecessor / "protocol.json").read_bytes()),
            "reflection": str(reflection), "accumulation": str(learned),
            "scope": "fresh Reflection study followed by accumulation; no thinking outcomes reused",
            "model_settings": new["model_settings"], "gpu_physical_id": 1, "port": 8006,
            "server_chat_template": str(template), "server_chat_template_sha256": base.digest(template.read_bytes()),
            "planned_actor_trajectories": 4450, "planned_proposal_chains": 500, "planned_curator_steps": 900,
            "phase_order": ["wait_for_thinking_completion", "reflection", "accumulation"]}
    # Freeze the scheduler and its imports as well as both experiment runners.
    runtime = output / "scheduler_runtime/scripts/ace"
    runtime.mkdir(parents=True)
    for filename in (Path(__file__).name, "run_context_proposal_bottleneck.py", "run_reflection_accumulation.py"):
        shutil.copy2(ROOT / "scripts/ace" / filename, runtime / filename)
    plan["scheduler_source_hashes"] = {str(p.relative_to(output / "scheduler_runtime")): base.digest(p.read_bytes())
                                       for p in runtime.iterdir() if p.is_file()}
    base.write(output / "plan.json", plan)
    base.write(output / "status.json", {"state": "scheduled", "phase": "waiting_for_thinking_completion",
                                        "updated_at": base.now()})
    return plan


def run(output):
    from filelock import FileLock

    output = Path(output).resolve()
    with FileLock(str(output / "scheduler.lock"), timeout=0):
        plan = base.read(output / "plan.json")
        predecessor = Path(plan["predecessor"])
        for relative, expected in plan["scheduler_source_hashes"].items():
            if base.digest((output / "scheduler_runtime" / relative).read_bytes()) != expected:
                raise ValueError("Scheduler source changed")
        base.write(output / "process.json", {"pid": os.getpid(), "started_at": base.now(), "command": sys.argv})
        phase = "waiting_for_thinking_completion"
        try:
            while not predecessor_ready(predecessor):
                if (output / "STOP").exists():
                    raise InterruptedError("STOP requested")
                base.write(output / "status.json", {"state": "scheduled", "phase": phase,
                                                    "predecessor_status": base.read(predecessor / "status.json"),
                                                    "updated_at": base.now(), "pid": os.getpid()})
                time.sleep(30)
            if base.digest((predecessor / "protocol.json").read_bytes()) != plan["predecessor_protocol_sha256"]:
                raise ValueError("Predecessor protocol changed")
            if base.digest(Path(plan["server_chat_template"]).read_bytes()) != plan["server_chat_template_sha256"]:
                raise ValueError("Serving chat template changed")
            for phase in ("reflection", "accumulation"):
                if (output / "STOP").exists():
                    raise InterruptedError("STOP requested")
                target = Path(plan[phase])
                if base.read(target / "status.json").get("state") == "completed":
                    continue
                verify_sources(target)
                protocol = base.read(target / "protocol.json")
                if protocol["model_settings"] != plan["model_settings"]:
                    raise ValueError("Scheduled generation settings changed")
                base.write(output / f"endpoint_{phase}.json", base.check_endpoint(protocol))
                filename = ("run_context_proposal_bottleneck.py" if phase == "reflection"
                            else "run_reflection_accumulation.py")
                command = [sys.executable, str(target / "source_snapshot/scripts/ace" / filename),
                           "--run", "--output", str(target)]
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES="1", PYTHONUNBUFFERED="1")
                with (target / "scheduler.log").open("a") as log:
                    child = subprocess.Popen(command, cwd=predecessor.parent.parent, env=env,
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    base.write(output / "status.json", {"state": "running", "phase": phase, "child_pid": child.pid,
                                                        "command": command, "updated_at": base.now()})
                    accumulation.journal(target, "running", f"Non-thinking {phase} 시작: {command}")
                    result = child.wait()
                if result != 0 or base.read(target / "status.json").get("state") != "completed":
                    raise RuntimeError(f"{phase} did not complete (exit {result}); next phase will not start")
                accumulation.journal(target, "completed", f"Non-thinking {phase} 전체 완료.")
            base.write(output / "status.json", {"state": "completed", "phase": "all_complete", "at": base.now()})
        except BaseException as exc:
            base.write(output / "status.json", {"state": "error", "phase": phase,
                                                "error": str(exc), "at": base.now()})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.predecessor is None:
            parser.error("--prepare requires --predecessor")
        prepare(args.predecessor, args.output)
    else:
        run(args.output)
