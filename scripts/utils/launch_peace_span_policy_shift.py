# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Add matched fresh BFCL policy-shift runs, with per-seed past=9 -> 3 -> 6 lanes."""

from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAST_ORDER = (9, 3, 6)
SEEDS = (42, 43, 44)
OVERLAYS = (
    "src/exgentic/agents/premise/pipeline.py",
    "src/exgentic/agents/premise/selection.py",
    "src/exgentic/agents/premise/agent.py",
    "src/exgentic/agents/premise/instance.py",
    "src/exgentic/agents/premise/span_policy_shift.py",
    "src/exgentic/agents/premise/proposal_retrieval.py",
    "scripts/premise/run_experiment.py",
)
ACTOR_FILES = (
    "src/exgentic/agents/premise/llm.py",
    "src/exgentic/agents/ace/prompts/generator.py",
    "src/exgentic/agents/ace/prompts/curator.py",
    "src/exgentic/agents/ace/playbook_utils.py",
    "src/exgentic/agents/litellm_tool_calling/instance.py",
    "src/exgentic/core/experiment_limits.py",
    "scripts/utils/run_qwen_isolated_when_free.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replace_function(code, reference, name):
    own = next(node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.FunctionDef) and node.name == name)
    source = next(
        node for node in ast.walk(ast.parse(reference)) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    lines = code.splitlines(keepends=True)
    lines[own.lineno - 1 : own.end_lineno] = [ast.get_source_segment(reference, source) + "\n"]
    return "".join(lines)


def prepare_snapshot(source, target):
    """Keep baseline actors/prompts and fixed-order/algorithm-seed behavior byte-pinned."""
    target.mkdir(parents=True, exist_ok=False)
    for name in ("src", "scripts"):
        shutil.copytree(
            source / name,
            target / name,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    for name in (".venv", ".exgentic"):
        (target / name).symlink_to((source / name).resolve(), target_is_directory=True)
    for name in OVERLAYS:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target / name)
    path = target / "src/exgentic/agents/premise/pipeline.py"
    code = replace_function(path.read_text(), (source / path.relative_to(target)).read_text(), "manifest_order")
    old = "        self.rng = random.Random(seed)"
    if code.count(old) != 1:
        raise ValueError("Cannot preserve the baseline algorithm seed")
    code = code.replace(
        old,
        '        self.algorithm_seed = int(json.loads(self.manifest.read_text()).get("algorithm_seed", seed))\n'
        "        self.rng = random.Random(self.algorithm_seed)",
    )
    old = '            "count": self.count, "past_count": self.past_count, "model": self.model,'
    if code.count(old) != 1:
        raise ValueError("Cannot preserve baseline seed provenance")
    code = code.replace(old, old + '\n            "algorithm_seed": self.algorithm_seed,')
    path.write_text(code)
    audit = {name: {"baseline_sha256": sha(source / name), "run_sha256": sha(target / name)} for name in ACTOR_FILES}
    if any(item["baseline_sha256"] != item["run_sha256"] for item in audit.values()):
        raise ValueError("A baseline actor, prompt, mutation application, or budget policy changed")
    write(target.parent / "baseline_actor_audit.json", audit)
    return audit


def build_jobs(plan, output, snapshot, runner):
    originals = {
        job["seed"]: job
        for job in plan["jobs"]
        if job["benchmark"] == "bfcl" and job["variant"] == "peace_no_alignment_aud"
    }
    if set(originals) != set(SEEDS):
        raise ValueError("Expected all three matched PEACE seeds")
    jobs = []
    for past in PAST_ORDER:
        for seed in SEEDS:
            original = originals[seed]
            variant = f"peace_span_policy_shift_aud_p{past}"
            identity = f"{variant}_bfcl_s{seed}"
            destination = output / "runs" / variant / "bfcl" / f"s{seed}"
            job = copy.deepcopy(original)
            for key in ("source_train_dir", "source_upstream"):
                job.pop(key, None)
            previous = PAST_ORDER.index(past) - 1
            job.update(
                id=identity,
                variant=variant,
                output=str(destination),
                train_dir=str(destination / "train"),
                source_snapshot=str(snapshot),
                template_job_id=original["id"],
                template_source_plan=original["source_plan"],
                source_plan=str(output / "launch_manifest.json"),
                predecessor=f"peace_span_policy_shift_aud_p{PAST_ORDER[previous]}_bfcl_s{seed}"
                if previous >= 0
                else None,
                lane=seed - 42,
                past_count=past,
                current_count=1,
                candidate_evaluation="span_policy_shift",
                policy_shift_temperature=0.0,
                replay_workers=28,
                llm_calls=28,
                status="pending",
                temperature_all_roles=0.0,
                from_scratch=True,
                restored_tasks=0,
                epochs=2,
                train_count=100,
                epoch1_count=50,
                alignment_scoring=False,
                port_block_start=55000 + (seed - 42) * 500,
                release_delay_seconds=0,
            )
            job["train_ids"] = job["epoch1_order"] + job["epoch2_order"]
            if len(job["train_ids"]) != 100 or len(set(job["train_ids"])) != 50:
                raise ValueError("Expected the same fifty tasks in two complete epochs")
            job["source_spec"].update(
                id=hashlib.sha256(str(destination).encode()).hexdigest()[:20],
                root=str(output),
                output=str(destination),
                source_output=str(destination),
                train_dir=job["train_dir"],
                variant=output.name,
                run_variant=variant,
                snapshot=str(snapshot),
                temperature=0.0,
                train_total=100,
                candidate_evaluation="span_policy_shift",
                past_count=past,
            )
            job["train_command"] = [
                str(snapshot / ".venv/bin/python"),
                "-u",
                str(runner),
                "train",
                "--output",
                str(output),
                "--job",
                identity,
            ]
            job["command"] = [
                str(REPO / ".venv/bin/python"),
                "-u",
                str(runner),
                "worker",
                "--output",
                str(output),
                "--job",
                identity,
            ]
            jobs.append(job)
    return jobs


def prepare(output, source):
    output, source = output.resolve(), source.resolve()
    if (output / "launch_manifest.json").exists():
        saved = read(output / "launch_manifest.json")
        if saved["comparison_source"] != str(source):
            raise ValueError("Existing output belongs to a different comparison")
        return saved
    baseline = read(source / "launch_manifest.json")
    if not baseline.get("from_scratch") or baseline.get("temperature_all_roles") != 0 or baseline.get("epochs") != 2:
        raise ValueError("Comparison must be fresh two-epoch temperature-zero training")
    output.mkdir(parents=True, exist_ok=True)
    original = next(job for job in baseline["jobs"] if job["variant"] == "peace_no_alignment_aud")
    snapshot = output / "code_snapshot"
    prepare_snapshot(Path(original["source_snapshot"]), snapshot)
    manifest = output / "task_splits.json"
    shutil.copy2(baseline["manifest"], manifest)
    if sha(manifest) != baseline["manifest_sha256"]:
        raise ValueError("Comparison split changed")
    runner = output / "span_runner.py"
    shutil.copy2(__file__, runner)
    jobs = build_jobs(baseline, output, snapshot, runner)
    for job in jobs:
        directory = Path(job["output"])
        (directory / "train").mkdir(parents=True, exist_ok=False)
        write(
            directory / "fresh_start_provenance.json",
            {
                "from_scratch": True,
                "restored_tasks": 0,
                "copied_training_artifacts": [],
                "epoch1_order": job["epoch1_order"],
                "epoch2_order": job["epoch2_order"],
                "template_job_id": job["template_job_id"],
                "predecessor": job["predecessor"],
                "past_count": job["past_count"],
                "current_count": 1,
                "action_concurrency": 28,
                "temperature_all_roles": 0.0,
                "snapshot": str(snapshot),
            },
        )
    hashes = {
        str(path.relative_to(snapshot)): sha(path)
        for name in ("src", "scripts")
        for path in (snapshot / name).rglob("*.py")
    }
    write(output / "code_hashes.json", {"snapshot": hashes, "runner_sha256": sha(runner)})
    plan = {
        "created_at": now(),
        "comparison_source": str(source),
        "comparison_manifest_sha256": sha(source / "launch_manifest.json"),
        "model": baseline["model"],
        "mode": "isolated",
        "epochs": 2,
        "from_scratch": True,
        "temperature_all_roles": 0.0,
        "playbook_edit_mode": "add_update_delete",
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
        "base_url": baseline["base_url"],
        "environment": baseline["environment"],
        "max_parallel": 3,
        "past_order": list(PAST_ORDER),
        "completion_gate": "training and matching full final test with zero infrastructure errors",
        "final_evaluation_splits": ["train", "val", "test"],
        "jobs": jobs,
    }
    write(output / "launch_manifest.json", plan)
    write(output / "progress.json", {"jobs": {job["id"]: {"status": "pending"} for job in jobs}})
    write(output / "status.json", {"phase": "planned", "at": now()})
    public = {key: value for key, value in plan.items() if key != "environment"}
    write(output / "protocol_public.json", public)
    return plan


def train(plan, job, *, check=False):
    snapshot = Path(job["source_snapshot"])
    sys.path[:0] = [str(snapshot / "src"), str(snapshot)]
    original = next(row for row in read(job["template_source_plan"])["jobs"] if row["id"] == job["template_job_id"])
    argv = [arg for arg in original["train_command"][3:] if arg != "--resume-sessions"]
    for flag, value in (
        ("--output-dir", job["train_dir"]),
        ("--temperature", "0.0"),
        ("--task-split-manifest", plan["manifest"]),
        ("--base-url", job["base_url"]),
        ("--past", str(job["past_count"])),
        ("--replay-workers", "28"),
        ("--llm-calls", "28"),
    ):
        argv[argv.index(flag) + 1] = value
    argv.extend(["--candidate-evaluation", "span_policy_shift", "--policy-shift-temperature", "0.0"])
    if check:
        argv.append("--dry-run")
    from exgentic.agents.premise.pipeline import PremisePipeline

    restore = PremisePipeline._restore_or_initialize

    def restore_order(self):
        self.order = [("bfcl", task) for task in job["train_ids"]]
        return restore(self)

    PremisePipeline._restore_or_initialize = restore_order
    file = snapshot / "scripts/premise/run_experiment.py"
    sys.argv = [str(file), *argv]
    module_file("span_frozen_peace", file).main()


def environment(plan, job):
    env = dict(os.environ, **plan["environment"])
    env.update(
        OPENAI_API_BASE=job["base_url"],
        OPENAI_BASE_URL=job["base_url"],
        OPENAI_API_KEY="EMPTY",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        EXGENTIC_PORT_BLOCK_START=str(job["port_block_start"]),
    )
    env["PATH"] = str(Path(job["source_snapshot"]) / ".venv/bin") + os.pathsep + env.get("PATH", "")
    return env


def worker(plan, job):
    snapshot = Path(job["source_snapshot"])
    sys.path[:0] = [str(snapshot / "src"), str(snapshot)]
    os.environ.update(environment(plan, job))
    fixed = module_file("span_fixed_worker", snapshot / "scripts/utils/fixed_matrix_worker.py")
    try:
        fixed.run({**plan, "snapshot": str(snapshot)}, job)
    except BaseException as exc:
        write(Path(job["output"]) / "job_status.json", {"phase": "error", "at": now(), "error": repr(exc)})
        raise


def eligible_jobs(plan, progress, *, completed=None):
    """Release a seed's next past-count immediately after its own predecessor completes."""
    return [
        job
        for job in plan["jobs"]
        if progress[job["id"]]["status"] == "pending"
        and (job["predecessor"] is None or progress[job["predecessor"]]["status"] == "completed")
        and (not job.get("external_predecessor")
             or (completed is not None and completed(job["external_predecessor"])))
        and (not job.get("external_predecessors")
             or (completed is not None and all(completed(old) for old in job["external_predecessors"])))
    ]


def run(output, plan):
    snapshot = Path(plan["jobs"][0]["source_snapshot"])
    helper = module_file("span_process_helpers", snapshot / "scripts/utils/continue_epoch2.py")
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = read(output / "progress.json")
        progress, active = saved["jobs"], {}
        write(output / "controller_process.json", {"pid": os.getpid(), "output": str(output), "at": now()})
        for job in plan["jobs"]:
            entry = progress[job["id"]]
            if helper.completed_job(job):
                entry["status"] = "completed"
            elif entry["status"] == "running":
                adopted = helper.existing_worker(job, entry, output)
                if adopted is not None:
                    active[job["id"]] = (job, adopted)
                else:
                    entry.update(status="error", reason="worker disappeared without complete results")
        while True:
            for identity, (job, process) in list(active.items()):
                result = process.poll()
                if result is None:
                    continue
                good = result == 0 and helper.completed_job(job)
                progress[identity].update(status="completed" if good else "error", returncode=result, finished_at=now())
                del active[identity]
            for job in plan["jobs"]:
                entry, predecessor = progress[job["id"]], job["predecessor"]
                if (
                    entry["status"] == "pending"
                    and predecessor
                    and progress[predecessor]["status"] in {"error", "blocked"}
                ):
                    entry.update(status="blocked", reason="predecessor failed")
            for job in eligible_jobs(plan, progress, completed=helper.completed_job):
                if len(active) >= plan["max_parallel"]:
                    break
                with (Path(job["output"]) / "worker.log").open("a") as log:
                    process = subprocess.Popen(
                        job["command"],
                        cwd=REPO,
                        env=environment(plan, job),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                identity = helper.process_identity(process.pid)
                active[job["id"]] = (job, process)
                progress[job["id"]].update(
                    status="running",
                    pid=process.pid,
                    started_at=now(),
                    start_ticks=identity["start_ticks"] if identity else None,
                )
            write(output / "progress.json", saved)
            pending = sum(entry["status"] == "pending" for entry in progress.values())
            errors = sum(entry["status"] in {"error", "blocked"} for entry in progress.values())
            phase = "running" if active or pending else "blocked" if errors else "completed"
            if pending and not active and plan.get("after_output"):
                phase = "waiting_predecessors"
            write(
                output / "status.json",
                {
                    "phase": phase,
                    "at": now(),
                    "active": len(active),
                    "pending": pending,
                    "errors": errors,
                    "benchmark": "bfcl",
                },
            )
            if not active and not pending:
                break
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker", "train", "check"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        plan = prepare(output, args.source)
        print(json.dumps({"output": str(output), "jobs": len(plan["jobs"]), "past_order": plan["past_order"]}))
        return
    plan = read(output / "launch_manifest.json")
    if args.action == "run":
        run(output, plan)
    else:
        job = next(job for job in plan["jobs"] if job["id"] == args.job)
        if args.action == "worker":
            worker(plan, job)
        else:
            train(plan, job, check=args.action == "check")


if __name__ == "__main__":
    main()
