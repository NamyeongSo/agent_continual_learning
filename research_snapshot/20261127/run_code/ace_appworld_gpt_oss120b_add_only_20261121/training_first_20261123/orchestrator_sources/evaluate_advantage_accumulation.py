# SPDX-License-Identifier: Apache-2.0
"""Evaluate the sign-filtered ACE cohort after all Contexts finish training."""

import argparse
import multiprocessing
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain
from pathlib import Path
from statistics import mean

from filelock import FileLock

from scripts.ace.plot_worst_reflection_repeats import RUNS
from scripts.ace.run_reflection_accumulation import Progress, base, journal, validate_evaluation

PHASES = {"train_final": 1200, "bwt_checkpoint": 582, "val": 1200}


def build_queue(output, plan, phase, runs=RUNS):
    batches = []
    for learner in sorted(plan, key=lambda job: job["condition"]["seed"]):
        directory = Path(learner["path"])
        condition = learner["condition"]
        cid, seed = condition["id"], condition["seed"]
        protocol = base.read(directory / "protocol.json")
        original = base.read(runs[directory.parent.name] / "protocol.json")["task_orders"][str(seed)]
        learned = protocol["task_orders"][str(seed)]["train"]
        assert len(learned) == learner["steps"]
        assert learned == [task for task in original["train"] if task in set(learned)]
        assert protocol["task_orders"][str(seed)]["val"] == original["val"]
        root = output / "evaluations" / directory.parent.name / cid
        base.write(root / "protocol.json", protocol)
        checkpoints = directory / "conditions" / cid / "checkpoints"
        assert (checkpoints.parent / "training_completed.json").exists()
        tasks = {"train_final": original["train"], "bwt_checkpoint": learned, "val": original["val"]}[phase]
        batch = []
        for index, task in enumerate(tasks):
            step = index + 1 if phase == "bwt_checkpoint" else len(learned)
            checkpoint = base.read(checkpoints / f"{step:03d}.json")
            assert phase != "bwt_checkpoint" or checkpoint["task_id"] == task
            context = checkpoints / f"{step:03d}.txt"
            assert context.read_text() == checkpoint["playbook"]
            key = f"{cid}_{phase}_{task}"
            job = {
                "id": key,
                "stage": 3,
                "task_id": task,
                "source_id": key,
                "context_id": key,
                "source_repeat": 0,
                "evaluation_repeat": 0,
                "arm": condition["arm"],
                "seed": seed,
                "phase": phase,
                "condition_id": cid,
                "checkpoint_step": step,
                "sampler": directory.parent.name,
            }
            base.write(
                root / "stage2" / key / "result.json",
                {
                    "task_id": task,
                    "source_id": key,
                    "updated_context_path": str(context),
                    "updated_context_sha256": base.digest(context.read_bytes()),
                },
            )
            batch.append((str(root), job))
        batches.append(batch)
    queue = list(chain.from_iterable(batches))
    assert len(queue) == phase_totals(plan)[phase]
    assert len({(root, job["id"]) for root, job in queue}) == len(queue)
    base.write(output / "evaluations" / f"{phase}_queue.json", queue)
    return queue


def evaluate(root, job):
    directory = base.job_dir(root, job)
    protocol = base.read(Path(root) / "protocol.json")
    if not (directory / "result.json").exists():
        for _ in range(3):
            attempt = max((int(p.name.split("_")[1]) for p in directory.glob("attempt_*")), default=0) + 1
            if base.worker(root, job, attempt)["ok"]:
                break
    return validate_evaluation(root, job, protocol)


def phase_totals(plan):
    return {"train_final": 50 * len(plan), "bwt_checkpoint": sum(j["steps"] for j in plan), "val": 50 * len(plan)}


def summarize(results):
    contexts = defaultdict(lambda: defaultdict(dict))
    for phase, rows in results.items():
        for row in rows:
            contexts[f"{row['sampler']}/{row['condition_id']}"][phase][row["task_id"]] = row
    summary = {}
    for context, phases in contexts.items():
        train, bwt, val = (phases[p] for p in PHASES)
        assert len(train) == len(val) == 50 and bwt.keys() <= train.keys()
        summary[context] = {
            "train_tasks": len(train),
            "bwt_tasks": len(bwt),
            "heldout_tasks": len(val),
            "train_pct": mean(r["pct"] for r in train.values()),
            "train_tgc": mean(r["success"] for r in train.values()),
            "heldout_pct": mean(r["pct"] for r in val.values()),
            "heldout_tgc": mean(r["success"] for r in val.values()),
            "bwt_pct": mean(train[t]["pct"] - r["pct"] for t, r in bwt.items()) if bwt else None,
            "bwt_tgc": mean(int(train[t]["success"]) - int(r["success"]) for t, r in bwt.items()) if bwt else None,
        }
    return summary


def run(output, runs=RUNS):
    progress = Progress(output / "evaluations")
    plan = base.read(output / "plan.json")
    totals = phase_totals(plan)
    (output / "evaluations").mkdir(exist_ok=True)
    while True:
        training = base.read(output / "status.json")
        if training["state"] == "completed":
            break
        if training["state"] == "error":
            raise RuntimeError("Training failed before the evaluation barrier")
        progress.emit("all", "waiting_for_training", training["completed_contexts"], len(plan), state="scheduled")
        time.sleep(30)
    assert training["completed_contexts"] == len(plan) and training["completed_steps"] == totals["bwt_checkpoint"]
    limits = set()
    for directory in {job["path"] for job in plan}:
        protocol = base.read(Path(directory) / "protocol.json")
        base.check_endpoint(protocol)
        limits.add(protocol.get("evaluation_workers", 50))
    if len(limits) != 1:
        raise ValueError("Expected one shared evaluation concurrency")
    worker_limit = limits.pop()
    if not isinstance(worker_limit, int) or worker_limit < 1:
        raise ValueError("Invalid evaluation concurrency")
    results = {}
    journal(output, "running", "Context 학습 완료. Train → BWT → Held-out 평가를 시작합니다.")
    partial = {}
    for phase, total in totals.items():
        progress.emit("all", phase, 0, total, pending=total, worker_limit=worker_limit)
        queue = build_queue(output, plan, phase, runs)
        rows = []
        expected = defaultdict(int)
        for root, _ in queue:
            expected[root] += 1
        by_context = defaultdict(list)
        train_scores = {(r["sampler"], r["condition_id"], r["task_id"]): r for r in results.get("train_final", [])}
        with ProcessPoolExecutor(max_workers=worker_limit, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(evaluate, root, job): root for root, job in queue}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                root = futures[future]
                batch = by_context[root]
                batch.append(row)
                if len(batch) == expected[root]:
                    key = f"{row['sampler']}/{row['condition_id']}"
                    scores = {
                        "tasks": len(batch),
                        "pct": mean(r["pct"] for r in batch),
                        "tgc": mean(r["success"] for r in batch),
                    }
                    if phase == "bwt_checkpoint":
                        pairs = [(train_scores[(r["sampler"], r["condition_id"], r["task_id"])], r) for r in batch]
                        scores.update(
                            bwt_pct=mean(a["pct"] - b["pct"] for a, b in pairs),
                            bwt_tgc=mean(int(a["success"]) - int(b["success"]) for a, b in pairs),
                        )
                    partial.setdefault(key, {})[phase] = scores
                    base.write(Path(root) / f"{phase}_results.json", batch)
                    base.write(progress.output / "partial_summary.json", partial)
                if len(rows) % 10 == 0 or len(rows) == total:
                    remaining = total - len(rows)
                    progress.emit(
                        "all", phase, len(rows), total, min(worker_limit, remaining),
                        max(0, remaining - worker_limit), worker_limit=worker_limit,
                    )
        results[phase] = rows
        base.write(progress.output / f"{phase}_results.json", rows)
    base.write(progress.output / "summary.json", summarize(results))
    progress.emit("all", "completed", sum(totals.values()), sum(totals.values()), state="completed")
    journal(output, "completed", "Train, BWT, Held-out 평가 완료. PCT와 공식 TGC를 기록했습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    (output / "evaluations").mkdir(exist_ok=True)
    with FileLock(str(output / "evaluations" / "run.lock"), timeout=0):
        try:
            run(output)
        except Exception as exc:
            path = output / "evaluations" / "status.json"
            status = base.read(path) if path.exists() else {}
            base.write(path, {**status, "state": "error", "error": str(exc), "updated_at": base.now()})
            raise
