# SPDX-License-Identifier: Apache-2.0
"""Schedule curator-only accumulation after the matched reflection bank completes."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.ace import run_context_proposal_bottleneck as base  # noqa: E402

MODES = ("random", "best", "worst")
ARMS = ("without_gt", "with_gt")


def exact_reward(row):
    if row.get("budget_failure"):
        if row["pct"] != 0:
            raise ValueError("Nonzero budget failure")
        return Fraction(0)
    total, passed = row["total_tests"], row["passed_tests"]
    if total <= 0 or not 0 <= passed <= total or abs(row["pct"] - passed / total) > 1e-12:
        raise ValueError("Invalid official test reward")
    return Fraction(passed, total)


def select_candidate(candidates, mode, seed, task_id, arm):
    """Uniform selection over exact ties; independent of task shuffle and execution order."""
    scores = [Fraction(c["score_numerator"], c["score_denominator"]) for c in candidates]
    if mode == "random":
        tied = list(range(len(candidates)))
        namespace = "paired_random"  # Same trajectory index for both GT arms.
    elif mode in {"best", "worst"}:
        target = (max if mode == "best" else min)(scores)
        tied = [i for i, score in enumerate(scores) if score == target]
        namespace = arm
    else:
        raise ValueError(mode)
    rng_seed = int(base.digest(f"selection-v1|{seed}|{task_id}|{mode}|{namespace}"), 16)
    index = random.Random(rng_seed).choice(tied)
    return {"candidate": candidates[index], "candidate_index": index, "tie_indices": tied,
            "selection_rng_seed": str(rng_seed), "mode": mode}


def freeze_bank(output):
    """Require complete, unfiltered train50/G5/three-repeat evidence before any learning."""
    output = Path(output)
    protocol = base.read(output / "protocol.json")
    source = Path(protocol["source_reflection_run"])
    if base.read(source / "status.json").get("state") != "completed":
        raise ValueError("Reflection experiment has not completed")
    source_protocol = base.read(source / "protocol.json")
    if source_protocol.get("stage3_repeats_per_context") != 3:
        raise ValueError("Expected the user-approved three-repeat candidate evaluation")
    for key in ("model", "model_settings", "appworld_policy_sha256", "initial_playbook", "turn_limit_scoring"):
        if source_protocol[key] != protocol[key]:
            raise ValueError(f"Predecessor protocol changed: {key}")
    tasks = protocol["task_orders"]["42"]["train"]
    task_set = set(tasks)
    expected = {(arm, task, repeat) for arm in ARMS for task in tasks for repeat in range(5)}
    scores = {}
    evaluated_contexts = {}
    hashes = {}
    for path in sorted((source / "stage3").glob("*/result.json")):
        row = base.read(path)
        key = row["arm"], row["task_id"], row["source_repeat"]
        if key not in expected or row["appworld_policy_sha256"] != protocol["appworld_policy_sha256"]:
            raise ValueError("Candidate evaluation outside the pinned train/arm/policy scope")
        repeats = scores.setdefault(key, {})
        if row["evaluation_repeat"] in repeats:
            raise ValueError("Duplicate candidate evaluation")
        repeats[row["evaluation_repeat"]] = exact_reward(row)
        evaluated_contexts.setdefault(key, set()).add(row["initial_context_sha256"])
        hashes[str(path)] = base.digest(path.read_bytes())
    if set(scores) != expected or any(set(v) != {0, 1, 2} for v in scores.values()):
        raise ValueError("Incomplete candidate bank: require 500 contexts and 1500 evaluations")
    if len(list((source / "stage2").glob("*/result.json"))) != 500:
        raise ValueError("Incomplete source proposals")
    bank = {arm: {task: [] for task in tasks} for arm in ARMS}
    for arm, task, repeat in sorted(expected):
        source_id = f"{task}_r{repeat}"
        proposal_path = source / "stage2" / f"{arm}_{source_id}" / "result.json"
        proposal = base.read(proposal_path)
        initial = base.read(source / "stage1" / source_id / "result.json")
        if task not in task_set or proposal["task_id"] != task or proposal["source_id"] != source_id:
            raise ValueError("Reflection/source task mismatch")
        if evaluated_contexts[(arm, task, repeat)] != {proposal["updated_context_sha256"]}:
            raise ValueError("Candidate score refers to a different updated context")
        updated_path = Path(proposal["updated_context_path"])
        if base.digest(updated_path.read_bytes()) != proposal["updated_context_sha256"]:
            raise ValueError("Source candidate context changed after evaluation")
        reflection_path = Path(proposal["reflection_path"])
        reflection = base.read(reflection_path)
        capture_path = Path(initial["capture_path"])
        if reflection["source_capture_sha256"] != base.digest(capture_path.read_bytes()):
            raise ValueError("Reflection refers to obsolete grading or history")
        capture = base.read(capture_path)
        if capture["playbook"] != protocol["initial_playbook"]:
            raise ValueError("Initial bank contains learned knowledge")
        reward = sum(scores[(arm, task, repeat)].values(), Fraction(0)) / 3
        bank[arm][task].append({
            "task_id": task, "source_repeat": repeat, "source_id": source_id,
            "reflection": reflection["reflection_for_curator"],
            "reflection_sha256": base.digest(reflection["reflection_for_curator"]),
            "score_numerator": reward.numerator, "score_denominator": reward.denominator,
            "mean_pct": float(reward), "task": capture["task"], "context": capture["context"],
            "source_reflection_path": str(reflection_path), "source_capture_path": str(capture_path),
        })
        for path in (proposal_path, reflection_path, capture_path, updated_path):
            hashes[str(path)] = base.digest(path.read_bytes())
    for arm in ARMS:
        for task in tasks:
            bank[arm][task].sort(key=lambda r: r["source_repeat"])
    base.write(output / "reflection_bank.json", bank)
    base.write(output / "bank_source_hashes.json", hashes)
    return bank


def conditions(seed):
    return [{"id": f"s{seed}_{arm}_{mode}", "seed": seed, "arm": arm, "mode": mode}
            for arm in ARMS for mode in MODES]


def plan_selections(output, bank):
    protocol = base.read(Path(output) / "protocol.json")
    for seed in protocol["seeds"]:
        for condition in conditions(seed):
            chosen = [select_candidate(bank[condition["arm"]][task], condition["mode"], seed, task, condition["arm"])
                      for task in protocol["task_orders"][str(seed)]["train"]]
            base.write(Path(output) / "conditions" / condition["id"] / "selections.json", chosen)


def train_condition(output, condition):
    """Only native curator calls; no actor, reflector, or evaluation during learning."""
    from exgentic.agents.ace.ace_instance import ACEAgentInstance
    from exgentic.agents.ace.playbook_store import PlaybookStore
    from exgentic.core.types import ModelSettings
    from exgentic.utils.paths import SessionPaths

    output = Path(output)
    protocol = base.read(output / "protocol.json")
    if protocol["playbook_edit_mode"] != "add_only":
        raise ValueError("Unexpected mutator mode")
    base.configure_environment(protocol)
    directory = output / "conditions" / condition["id"]
    choices = base.read(directory / "selections.json")
    order = protocol["task_orders"][str(condition["seed"])]["train"]
    if [c["candidate"]["task_id"] for c in choices] != order:
        raise ValueError("Selection order differs from pinned seed order")
    store = PlaybookStore(condition["id"], protocol["initial_playbook"])
    cp_dir = directory / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    if not (cp_dir / "000.json").exists():
        base.write(cp_dir / "000.json", {"store_id": store.store_id, "playbook": store.playbook,
                                        "next_global_id": 1, "session_count": 0, "condition": condition})
        (cp_dir / "000.txt").write_text(store.playbook)
    completed = 0
    while (cp_dir / f"{completed + 1:03d}.json").exists():
        completed += 1
    store.load_checkpoint(str(cp_dir / f"{completed:03d}.json"))
    if store.session_count != completed:
        raise ValueError("Checkpoint/session counter mismatch")
    instance = ACEAgentInstance(
        session_id="curation", model=protocol["model"], curator_model=protocol["model"],
        model_settings=ModelSettings(**protocol["model_settings"]), **base.role_call_overrides(protocol),
        benchmark_id="appworld", training_time=True, playbook_edit_mode="add_only",
    )
    instance._store = store
    instance._paths = SessionPaths("curation", "training", directory)
    with (directory / "training.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            import litellm

            original_completion = litellm.completion
            call_state = {"step": 0, "count": 0}

            def record_completion(*args, **kwargs):
                call_state["count"] += 1
                path = directory / "calls" / f"step{call_state['step']:03d}_{uuid.uuid4().hex}.json"
                entry = {"at": base.now(), "role": "curator", "request": kwargs}
                started = time.monotonic()
                try:
                    response = original_completion(*args, **kwargs)
                    entry["response"] = response.model_dump(mode="json")
                    return response
                except Exception as exc:
                    entry["error"] = str(exc)
                    raise
                finally:
                    entry["seconds"] = time.monotonic() - started
                    base.write(path, entry)

            litellm.completion = record_completion
            try:
                for index in range(completed, len(order)):
                    if (output / "STOP").exists():
                        raise InterruptedError("STOP requested")
                    choice, step = choices[index], index + 1
                    candidate = choice["candidate"]
                    if base.digest(candidate["reflection"]) != candidate["reflection_sha256"]:
                        raise ValueError("Frozen selected reflection changed")
                    instance.task, instance.context = candidate["task"], candidate["context"]
                    parent_sha = base.digest(store.playbook)
                    store.increment_session()
                    call_state["step"] = step
                    before = instance.get_cost().model_dump(mode="json")
                    instance._active_llm_stage = "curator"
                    try:
                        instance._run_curator(candidate["reflection"])
                    except Exception:
                        # Reload the last durable checkpoint on resume; never skip a task.
                        base.write(directory / "training_error.json", {
                            "step": step, "task_id": order[index], "traceback": traceback.format_exc(),
                        })
                        raise
                    payload = {"store_id": store.store_id, "playbook": store.playbook,
                               "next_global_id": store.next_global_id, "session_count": store.session_count,
                               "condition": condition, "task_id": order[index], "step": step,
                               "parent_context_sha256": parent_sha, "context_sha256": base.digest(store.playbook),
                               "reflection_sha256": candidate["reflection_sha256"],
                               "selected_source_id": candidate["source_id"], "tie_indices": choice["tie_indices"],
                               "selection_rng_seed": choice["selection_rng_seed"], "completed_at": base.now(),
                               "cost_before": before, "cost_after": instance.get_cost().model_dump(mode="json")}
                    (cp_dir / f"{step:03d}.txt").write_text(store.playbook)
                    base.write(cp_dir / f"{step:03d}.json", payload)
                    base.write(directory / "training_status.json", {"completed": step, "total": len(order),
                                                                  "updated_at": base.now()})
            finally:
                litellm.completion = original_completion
    final = cp_dir / f"{len(order):03d}.json"
    base.write(directory / "training_completed.json", {"condition": condition, "tasks": len(order),
                                                      "final_checkpoint": str(final), "completed_at": base.now()})
    return {"ok": True, "condition": condition["id"]}


def eval_jobs(output, seed, phase):
    if phase not in {"train_final", "bwt_checkpoint", "val"}:
        raise ValueError(phase)
    output = Path(output)
    protocol = base.read(output / "protocol.json")
    seed_dir = output / f"seed{seed}"
    seed_dir.mkdir(exist_ok=True)
    base.write(seed_dir / "protocol.json", protocol)
    split = protocol["task_orders"][str(seed)]
    tasks = split["val"] if phase == "val" else split["train"]
    jobs = []
    for condition in conditions(seed):
        cp_dir = output / "conditions" / condition["id"] / "checkpoints"
        for index, task in enumerate(tasks):
            step = index + 1 if phase == "bwt_checkpoint" else len(split["train"])
            checkpoint = cp_dir / f"{step:03d}.json"
            checkpoint_data = base.read(checkpoint)
            if phase == "bwt_checkpoint" and checkpoint_data["task_id"] != task:
                raise ValueError("BWT checkpoint/task mismatch")
            context_path = cp_dir / f"{step:03d}.txt"
            if context_path.read_text() != checkpoint_data["playbook"]:
                raise ValueError("Checkpoint JSON/text mismatch")
            key = f"{condition['id']}_{phase}_{task}"
            base.write(seed_dir / "stage2" / key / "result.json", {
                "task_id": task, "source_id": key, "updated_context_path": str(context_path),
                "updated_context_sha256": base.digest(context_path.read_bytes()),
            })
            jobs.append({"id": key, "stage": 3, "task_id": task, "source_id": key, "source_repeat": 0,
                         "context_id": key, "evaluation_repeat": 0, "arm": condition["arm"],
                         "condition_id": condition["id"], "seed": seed, "phase": phase,
                         "checkpoint_step": step})
    return jobs


def validate_evaluation(seed_dir, job, protocol):
    row = base.read(base.job_dir(seed_dir, job) / "result.json")
    if any(row.get(key) != value for key, value in job.items()):
        raise ValueError("Evaluation metadata differs from the pinned job")
    parent = base.read(Path(seed_dir) / "stage2" / job["context_id"] / "result.json")
    if (row["initial_context_sha256"] != parent["updated_context_sha256"]
            or row["appworld_policy_sha256"] != protocol["appworld_policy_sha256"]):
        raise ValueError("Evaluation context or environment policy mismatch")
    exact_reward(row)
    return row


class Progress:
    def __init__(self, output):
        self.output = Path(output)

    def emit(self, seed, phase, completed, total, running=0, pending=0, errors=0, state="running"):
        status = {"updated_at": base.now(), "state": state, "seed": seed, "phase": phase,
                  "completed": completed, "total": total, "running": running, "pending": pending,
                  "errors": errors, "controller_pid": os.getpid()}
        base.write(self.output / "status.json", status)
        line = (f"[{status['updated_at']}] seed={seed} phase={phase} done={completed}/{total} "
                f"running={running} pending={pending} errors={errors} state={state}")
        print(line, flush=True)
        with (self.output / "progress.log").open("a") as stream:
            stream.write(line + "\n")


def journal(output, status, message):
    """Best-effort lifecycle recording, keeping failed payloads for manual recovery."""
    output = Path(output)
    record_path = output / "journal_record.json"
    if not record_path.exists():
        return
    helper = str(Path.home() / ".codex/skills/experiment-journal/scripts/journal.py")
    record_id = base.read(record_path)["id"]
    directory = output / "journal_events" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    try:
        result = subprocess.run([sys.executable, helper, "show", record_id], capture_output=True, text=True, check=True)
        record = json.loads(result.stdout)
        base.write(directory / "patch.json", {"status": status, "summary": message})
        subprocess.run([sys.executable, helper, "update", record_id, "--revision", str(record["revision"]),
                        "--file", str(directory / "patch.json")], capture_output=True, text=True, check=True)
        (directory / "note.txt").write_text(message + "\n")
        subprocess.run([sys.executable, helper, "log", record_id, "--file", str(directory / "note.txt")],
                       capture_output=True, text=True, check=True)
    except Exception as exc:
        base.write(directory / "recording_error.json", {"message": message, "error": str(exc)})


def training_phase(output, seed, progress):
    protocol = base.read(Path(output) / "protocol.json")
    pending = [c for c in conditions(seed)
               if not (Path(output) / "conditions" / c["id"] / "training_completed.json").exists()]
    progress.emit(seed, "curator_training", 300 - 50 * len(pending), 300, len(pending))
    journal(output, "running", f"seed {seed}: Curator-only Context6개 학습 시작. 모든 0~50 체크포인트 저장.")
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context("spawn")) as pool:
        active = {pool.submit(train_condition, str(output), c): c for c in pending}
        last = 0.0
        while active:
            finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()  # A failed learner must stop the seed barrier.
                active.pop(future)
            if time.monotonic() - last >= 30 or not active:
                count = sum(sum((Path(output) / "conditions" / c["id"] / "checkpoints" / f"{i:03d}.json").exists()
                                for i in range(1, 51)) for c in conditions(seed))
                progress.emit(seed, "curator_training", count, 300, len(active))
                last = time.monotonic()
    for condition in conditions(seed):
        path = Path(output) / "conditions" / condition["id"] / "checkpoints"
        if any(not (path / f"{step:03d}.json").exists() for step in range(51)):
            raise ValueError("Missing training checkpoint")
    if protocol["playbook_edit_mode"] != "add_only":
        raise ValueError("Unexpected mutator mode")


def evaluation_phase(output, seed, phase, progress):
    output = Path(output)
    protocol = base.read(output / "protocol.json")
    jobs = eval_jobs(output, seed, phase)
    seed_dir = output / f"seed{seed}"
    base.write(seed_dir / f"{phase}_queue.json", jobs)
    done = [j for j in jobs if (base.job_dir(seed_dir, j) / "result.json").exists()]
    for job in done:
        validate_evaluation(seed_dir, job, protocol)
    finished_ids = {j["id"] for j in done}
    pending = [j for j in jobs if j["id"] not in finished_ids]
    attempts = {j["id"]: len(list(base.job_dir(seed_dir, j).glob("attempt_*"))) for j in pending}
    errors, active = [], {}
    progress.emit(seed, phase, len(done), len(jobs), pending=len(pending))
    with ProcessPoolExecutor(max_workers=50, mp_context=multiprocessing.get_context("spawn")) as pool:
        last = 0.0
        while pending or active:
            if (output / "STOP").exists():
                raise InterruptedError("STOP requested")
            while pending and len(active) < 50:
                job = pending.pop(0)
                if attempts[job["id"]] >= 3:
                    errors.append(job)
                    continue
                attempts[job["id"]] += 1
                active[pool.submit(base.worker, str(seed_dir), job, attempts[job["id"]])] = job
            completed, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                job = active.pop(future)
                result = future.result()
                if result["ok"]:
                    validate_evaluation(seed_dir, job, protocol)
                    done.append(job)
                elif attempts[job["id"]] < 3:
                    pending.append(job)
                else:
                    errors.append(job)
                    base.write(seed_dir / "evaluation_errors" / f"{job['id']}.json", result)
            if time.monotonic() - last >= 30 or (not active and not pending):
                progress.emit(seed, phase, len(done), len(jobs), len(active), len(pending), len(errors))
                last = time.monotonic()
    if errors or len(done) != len(jobs):
        raise RuntimeError(f"Unresolved {phase} evaluations; next phase/seed will not start")
    rows = [validate_evaluation(seed_dir, j, protocol) for j in jobs]
    base.write(seed_dir / f"{phase}_results.json", rows)
    journal(output, "running", f"seed {seed}: {phase} 평가 {len(rows)}/300 완료.")


def summarize_seed(output, seed):
    directory = Path(output) / f"seed{seed}"
    phases = {p: base.read(directory / f"{p}_results.json") for p in ("train_final", "bwt_checkpoint", "val")}
    report = {}
    for condition in conditions(seed):
        selected = {p: [r for r in rows if r["condition_id"] == condition["id"]] for p, rows in phases.items()}
        if any(len(rows) != 50 for rows in selected.values()):
            raise ValueError("Incomplete seed summary")
        final = {r["task_id"]: r["pct"] for r in selected["train_final"]}
        pairs = [{"task_id": r["task_id"], "checkpoint_step": r["checkpoint_step"],
                  "after_learning_pct": r["pct"], "final_pct": final[r["task_id"]],
                  "difference": final[r["task_id"]] - r["pct"]} for r in selected["bwt_checkpoint"]]
        prior = [r for r in pairs if r["checkpoint_step"] < 50]
        report[condition["id"]] = {
            "train_mean_pct": sum(final.values()) / 50,
            "val_mean_pct": sum(r["pct"] for r in selected["val"]) / 50,
            "bwt_mean_all50": sum(r["difference"] for r in pairs) / 50,
            "bwt_mean_prior49": sum(r["difference"] for r in prior) / 49,
            "bwt_pairs": pairs,
        }
    base.write(directory / "summary.json", report)
    base.write(directory / "completed.json", {"seed": seed, "evaluations": 900, "completed_at": base.now()})


def prepare(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if (output / "protocol.json").exists():
        raise FileExistsError("Scheduler already prepared")
    protocol = base.read(source / "protocol.json")
    source_protocol_hash = base.digest((source / "protocol.json").read_bytes())
    splits = base.read(source / "task_splits.json")
    orders = {str(seed): splits[f"appworld_{seed}"] for seed in (42, 43, 44)}
    train = set(protocol["task_ids"])
    for split in orders.values():
        if (len(split["train"]) != 50 or set(split["train"]) != train
                or len(split["val"]) != 50 or train & (set(split["val"]) | set(split["test"]))):
            raise ValueError("Invalid original seed-specific train/val/test membership")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    snapshot = output / "source_snapshot"
    shutil.copytree(source / "source_snapshot/src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (snapshot / "scripts/ace").mkdir(parents=True)
    shutil.copy2(source / "source_snapshot/pyproject.toml", snapshot / "pyproject.toml")
    shutil.copy2(source / "source_snapshot/scripts/ace/run_context_proposal_bottleneck.py",
                 snapshot / "scripts/ace/run_context_proposal_bottleneck.py")
    shutil.copy2(__file__, snapshot / "scripts/ace" / Path(__file__).name)
    shutil.copytree(source / "benchmark_dependencies", output / "benchmark_dependencies",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for key in ("stages", "seed", "proposal_chains_planned", "source_initial_bank",
                "initial_trajectories_regenerated", "rescored_initial_rollouts", "completed_proposals_reused",
                "stage3_repeats_per_context"):
        protocol.pop(key, None)
    protocol.update(
        experiment="ace_reflection_accumulation", source_reflection_run=str(source),
        source_protocol_sha256=source_protocol_hash, source_candidate_repeats=3,
        seeds=[42, 43, 44], task_orders=orders, playbook_edit_mode="add_only",
        benchmark_dependency_overlay=str(output / "benchmark_dependencies"),
        training_contexts=18, curator_steps_per_context=50, planned_curator_steps=900,
        evaluation_repeats=1, evaluations_per_seed=900, planned_evaluations=2700, actor_trajectories_planned=2700,
        apply_policy_to=["train_final", "bwt_checkpoint", "val"], model_settings_apply_to=["actor", "curator"],
        seed_phase_order=["curator_training", "train_final", "bwt_checkpoint", "val"],
        progress_logging={"progress_log": "progress.log", "status_file": "status.json",
                          "heartbeat_interval_seconds": 30, "waiting_heartbeat_interval_seconds": 60},
        evaluation_workers=50, training_context_workers=6, selection_score="exact mean of three official PCT rewards",
        tie_break="uniform random among exact ties", source_worktree=str(ROOT), prepared_at=base.now(),
    )
    base.write(output / "protocol.json", protocol)
    base.write(output / "source_hashes.json", {
        str(p.relative_to(snapshot)): base.digest(p.read_bytes()) for p in snapshot.rglob("*") if p.is_file()
    })
    base.write(output / "status.json", {"state": "scheduled", "phase": "waiting_for_reflection_experiment",
                                        "updated_at": base.now()})


def run(output):
    from filelock import FileLock

    output = Path(output).resolve()
    with FileLock(str(output / "scheduler.lock"), timeout=0):
        protocol = base.read(output / "protocol.json")
        for relative, expected in base.read(output / "source_hashes.json").items():
            if base.digest((output / "source_snapshot" / relative).read_bytes()) != expected:
                raise ValueError(f"Frozen source changed: {relative}")
        base.write(output / "scheduler_process.json", {
            "pid": os.getpid(), "command": sys.argv, "started_at": base.now(),
        })
        progress = Progress(output)
        try:
            source = Path(protocol["source_reflection_run"])
            last = 0.0
            while base.read(source / "status.json").get("state") != "completed":
                if (output / "STOP").exists():
                    raise InterruptedError("STOP requested")
                if time.monotonic() - last >= 60:
                    progress.emit(None, "waiting_for_reflection_experiment", 0, 1, state="scheduled")
                    last = time.monotonic()
                time.sleep(15)
            base.configure_environment(protocol)
            base.write(output / "endpoint_verification.json", base.check_endpoint(protocol))
            if base.digest((source / "protocol.json").read_bytes()) != protocol["source_protocol_sha256"]:
                raise ValueError("Predecessor protocol changed while waiting")
            if not (output / "bank_ready.json").exists():
                bank = freeze_bank(output)
                plan_selections(output, bank)
                base.write(output / "bank_ready.json", {"contexts": 500, "evaluations": 1500, "at": base.now()})
            for seed in protocol["seeds"]:
                if (output / f"seed{seed}/completed.json").exists():
                    continue
                if (output / "STOP").exists():
                    raise InterruptedError("STOP requested")
                base.write(output / f"endpoint_seed{seed}.json", base.check_endpoint(protocol))
                training_phase(output, seed, progress)
                for phase in ("train_final", "bwt_checkpoint", "val"):
                    evaluation_phase(output, seed, phase, progress)
                summarize_seed(output, seed)
                journal(output, "running", f"seed {seed}: Context 6개 학습과 train/BWT/val 900회 완료.")
            progress.emit(44, "all_complete", 2700, 2700, state="completed")
            journal(output, "completed", "seed42→43→44 전체 완료: Curator 900단계, Actor 평가 2700회.")
        except BaseException as exc:
            base.write(output / "error.json", {
                "at": base.now(), "error": str(exc), "traceback": traceback.format_exc(),
            })
            previous = base.read(output / "status.json") if (output / "status.json").exists() else {}
            base.write(output / "status.json", {
                **previous, "state": "error", "error": str(exc), "updated_at": base.now(),
            })
            journal(output, "blocked", f"누적실험 중단: {exc}. 완료 결과/체크포인트 보존; 다음 seed로 진행하지 않음.")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.source is None:
            parser.error("--prepare requires --source")
        prepare(args.output, args.source)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
