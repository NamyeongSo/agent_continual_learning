# SPDX-License-Identifier: Apache-2.0
"""Finish all saved ACE learners before a shared, resumable evaluation queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import runpy
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("train_final", "bwt_checkpoint", "val")
MODES = ("high", "low")
_CACHE = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def frozen(source):
    if source not in _CACHE:
        _CACHE[source] = runpy.run_path(source)
        # Frozen actor/curator modules stay first; orchestration helpers are outside that bundle.
        for name, relative in (("scripts", "scripts"), ("scripts.ace", "scripts/ace")):
            package = sys.modules.get(name)
            if package is not None and str(ROOT / relative) not in package.__path__:
                package.__path__.append(str(ROOT / relative))
    return _CACHE[source]


def completed(job):
    directory = Path(job["path"]) / "conditions" / job["condition"]["id"]
    if not (directory / "training_completed.json").exists():
        return False
    checkpoint = read(directory / "checkpoints" / f"{job['steps']:03d}.json")
    if checkpoint["session_count"] != job["steps"] or checkpoint["condition"] != job["condition"]:
        raise ValueError(f"Invalid final checkpoint: {directory}")
    if (directory / "checkpoints" / f"{job['steps']:03d}.txt").read_bytes().decode("utf-8") != checkpoint["playbook"]:
        raise ValueError(f"Checkpoint text mismatch: {directory}")
    return True


def require_all_trained(jobs):
    missing = [j["condition"]["id"] for j in jobs if not completed(j)]
    if missing:
        raise RuntimeError(f"Evaluation barrier: {len(missing)} unfinished Contexts")


def emit(output, control, phase, **fields):
    status = {"state": "running", "phase": phase, "pid": os.getpid(), "updated_at": now(), **fields}
    write(control / "status.json", status)
    write(output / "status.json", {**status, "progress_directory": str(control)})
    line = f"[{status['updated_at']}] {phase} " + " ".join(
        f"{k}={v}" for k, v in fields.items() if k not in {"contexts", "errors", "by_phase", "by_context"}
    )
    print(line, flush=True)
    with (control / "progress.log").open("a") as stream:
        stream.write(line + "\n")


def prepare(output, control):
    jobs = []
    runs = {mode: output / mode / "accumulation" for mode in MODES}
    manifest_hashes = []
    for mode, directory in runs.items():
        hashes = read(directory / "source_hashes.json")
        manifest_hashes.append(hashes)
        for relative, expected in hashes.items():
            if digest(directory / "source_snapshot" / relative) != expected:
                raise ValueError(f"Frozen source changed: {directory}/{relative}")
        source = str(directory / "source_snapshot/scripts/ace/run_reflection_accumulation.py")
        module = frozen(source)
        protocol = read(directory / "protocol.json")
        predecessor = Path(protocol["source_reflection_run"])
        if digest(predecessor / "protocol.json") != protocol["source_protocol_sha256"]:
            raise ValueError("Source Reflection protocol changed")
        if protocol["playbook_edit_mode"] != "add_only" or protocol["model_settings"]["reasoning_effort"] != mode:
            raise ValueError("Unexpected mode or edit operation")
        module["base"].configure_environment(protocol)
        write(control / f"endpoint_{mode}.json", module["base"].check_endpoint(protocol))
        if not (directory / "bank_ready.json").exists():
            bank = module["freeze_bank"](directory)
            module["plan_selections"](directory, bank)
            write(directory / "bank_ready.json", {"contexts": 500, "evaluations": 1500, "at": now()})
        for seed in protocol["seeds"]:
            for condition in module["conditions"](seed):
                jobs.append({"path": str(directory), "condition": condition, "steps": 50,
                             "source": source, "sampler": mode, "kind": "full"})
    if manifest_hashes[0] != manifest_hashes[1]:
        raise ValueError("High/Low frozen actor and curator sources differ")
    advantage = output / "advantage"
    advantage.mkdir(exist_ok=True)
    helper = runpy.run_path(str(ROOT / "scripts/ace/run_advantage_accumulation.py"))
    plan_path = advantage / "plan.json"
    plan = read(plan_path) if plan_path.exists() else helper["prepare"](advantage, runs)
    for job in plan:
        mode = Path(job["path"]).parent.name
        jobs.append({**job, "source": str(runs[mode] / "source_snapshot/scripts/ace/run_reflection_accumulation.py"),
                     "sampler": mode, "kind": "advantage"})
    if len(jobs) != 60 or len({(j["path"], j["condition"]["id"]) for j in jobs}) != 60:
        raise ValueError("Expected 36 full and 24 advantage Contexts")
    old_plan = control / "training_plan.json"
    if old_plan.exists() and read(old_plan) != jobs:
        raise ValueError("Training plan changed during continuation")
    write(old_plan, jobs)
    bindings = {}
    for filename in ("run_training_first.py", "run_advantage_accumulation.py", "evaluate_advantage_accumulation.py"):
        path = ROOT / "scripts/ace" / filename
        bindings[str(path)] = digest(path)
    for job in jobs:
        directory = Path(job["path"]) / "conditions" / job["condition"]["id"]
        for path in (Path(job["path"]) / "protocol.json", directory / "selections.json"):
            bindings[str(path)] = digest(path)
    binding_path = control / "input_hashes.json"
    if binding_path.exists() and read(binding_path) != bindings:
        raise ValueError("Pinned protocol or selections changed")
    write(binding_path, bindings)
    return jobs


def train_one(job):
    return frozen(job["source"])["train_condition"](job["path"], job["condition"])


def retryable(error):
    return any(text in str(error).lower() for text in (
        "timeout", "connection", "unexpected tokens remaining in message header", "serviceunavailable",
        "temporarily unavailable", "rate limit", "ratelimit", "remoteprotocolerror",
    ))


def train_all(output, control, jobs):
    pending = deque(j for j in jobs if not completed(j))
    workers = min(54, len(pending))
    errors, attempts, active = [], Counter(), {}

    def report():
        counts = {}
        for job in jobs:
            directory = Path(job["path"]) / "conditions" / job["condition"]["id"]
            progress = directory / "training_status.json"
            counts[f"{job['sampler']}/{job['condition']['id']}"] = {
                "completed": read(progress)["completed"] if progress.exists() else 0,
                "total": job["steps"], "done": completed(job),
            }
        emit(output, control, "all_context_training", total_contexts=len(jobs),
             completed_contexts=sum(c["done"] for c in counts.values()), active_contexts=len(active),
             pending_contexts=len(pending), training_worker_limit=workers,
             completed_steps=sum(c["completed"] for c in counts.values()),
             total_steps=sum(c["total"] for c in counts.values()), contexts=counts, errors=errors)
        for kind, path in [("full", output / mode / "accumulation") for mode in MODES] + [
            ("advantage", output / "advantage")
        ]:
            selected = [j for j in jobs if j["kind"] == kind and (kind == "advantage" or j["path"] == str(path))]
            subset = {f"{j['sampler']}/{j['condition']['id']}": counts[f"{j['sampler']}/{j['condition']['id']}"]
                      for j in selected}
            write(path / "status.json", {"state": "completed" if all(c["done"] for c in subset.values()) else "running",
                  "phase": "curator_training" if kind == "full" else "all_context_training",
                  "pid": os.getpid(), "contexts": subset, "total_contexts": len(subset),
                  "completed_contexts": sum(c["done"] for c in subset.values()),
                  "completed_steps": sum(c["completed"] for c in subset.values()),
                  "total_steps": sum(c["total"] for c in subset.values()), "updated_at": now(),
                  "evaluation_state": "waiting_for_all_60_contexts", "errors": errors})

    if workers:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            last = 0.0
            while pending or active:
                if (output / "STOP").exists() or (control / "STOP").exists():
                    raise InterruptedError("STOP requested")
                while pending and len(active) < workers:
                    job = pending.popleft()
                    attempts[job["path"], job["condition"]["id"]] += 1
                    active[pool.submit(train_one, job)] = job
                finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = active.pop(future)
                    try:
                        future.result()
                        if not completed(job):
                            raise RuntimeError("Curator returned without a durable final checkpoint")
                    except Exception as exc:
                        if retryable(exc) and attempts[job["path"], job["condition"]["id"]] < 4:
                            pending.append(job)
                            print(f"{now()} curator retry {job['condition']['id']}: {exc}", flush=True)
                        else:
                            errors.append({"job": job, "error": repr(exc)})
                if time.monotonic() - last >= 30 or not (pending or active):
                    report()
                    last = time.monotonic()
    report()
    if errors:
        raise RuntimeError(f"{len(errors)} Context trainings failed; evaluation barrier remains closed")
    require_all_trained(jobs)
    write(control / "training_completed.json", {"completed_contexts": len(jobs), "completed_at": now()})


def queue_key(item):
    job = item["job"]
    return PHASES.index(job["phase"]), job["seed"], MODES.index(item["sampler"]), job["condition_id"], item["ordinal"]


def build_queue(output, control, jobs):
    require_all_trained(jobs)
    queue = []
    runs = {mode: output / mode / "accumulation" for mode in MODES}
    evaluator = runpy.run_path(str(ROOT / "scripts/ace/evaluate_advantage_accumulation.py"))
    for phase in PHASES:
        for mode, directory in runs.items():
            source = str(directory / "source_snapshot/scripts/ace/run_reflection_accumulation.py")
            module = frozen(source)
            for seed in read(directory / "protocol.json")["seeds"]:
                batch = module["eval_jobs"](directory, seed, phase)
                write(directory / f"seed{seed}" / f"{phase}_queue.json", batch)
                queue.extend({"root": str(directory / f"seed{seed}"), "job": job, "source": source,
                              "sampler": mode, "kind": "full", "ordinal": n} for n, job in enumerate(batch))
        plan = read(output / "advantage/plan.json")
        batch = evaluator["build_queue"](output / "advantage", plan, phase, runs)
        for ordinal, (root, job) in enumerate(batch):
            mode = job["sampler"]
            queue.append({"root": root, "job": job, "source": str(
                runs[mode] / "source_snapshot/scripts/ace/run_reflection_accumulation.py"),
                "sampler": mode, "kind": "advantage", "ordinal": ordinal})
    queue.sort(key=queue_key)
    if len({(r["root"], r["job"]["id"]) for r in queue}) != len(queue):
        raise ValueError("Duplicate evaluation in the shared queue")
    write(control / "evaluation_queue.json", queue)
    return queue


def result_path(item):
    return Path(item["root"]) / "stage3" / item["job"]["id"] / "result.json"


def validate_result(item):
    module = frozen(item["source"])
    return module["validate_evaluation"](item["root"], item["job"], read(Path(item["root"]) / "protocol.json"))


def evaluate_one(item):
    if result_path(item).exists():
        return validate_result(item)
    module = frozen(item["source"])
    directory = result_path(item).parent
    attempt = max((int(p.name.removeprefix("attempt_")) for p in directory.glob("attempt_*")), default=0)
    for index in range(6):
        attempt += 1
        response = module["base"].worker(item["root"], item["job"], attempt)
        if response["ok"]:
            return validate_result(item)
        if index >= 2 and not retryable(response.get("error", "")):
            break
    raise RuntimeError(f"Unresolved infrastructure failure: {item['root']}/{item['job']['id']}: {response}")


def evaluate_all(output, control, queue, workers):
    pending, done, active, errors = deque(), {}, {}, []
    expected = Counter((item["root"], item["job"]["phase"]) for item in queue)
    batches = defaultdict(list)
    contexts = defaultdict(list)
    partial = {}

    def collect(item, row):
        done[item["root"], item["job"]["id"]] = row
        key = item["root"], item["job"]["phase"]
        batches[key].append(row)
        if len(batches[key]) == expected[key]:
            write(Path(item["root"]) / f"{key[1]}_results.json", batches[key])
        key = item["sampler"], row["condition_id"], row["phase"]
        contexts[key].append(row)
        values = contexts[key]
        partial["/".join(key)] = {"tasks": len(values), "pct": sum(r["pct"] for r in values) / len(values),
                                  "tgc": sum(r["success"] for r in values) / len(values)}

    for item in queue:
        if result_path(item).exists():
            collect(item, validate_result(item))
        else:
            pending.append(item)
    reused = len(done)

    def report():
        by_phase = {phase: {"completed": sum(k[2] == phase and len(v) for k, v in contexts.items()),
                           "total": sum(item["job"]["phase"] == phase for item in queue)} for phase in PHASES}
        emit(output, control, "shared_evaluation", completed=len(done), total=len(queue), running=len(active),
             pending=len(pending), reused_evaluations=reused, evaluation_worker_limit=workers,
             by_phase=by_phase, errors=errors)
        write(control / "partial_summary.json", partial)

    report()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        last = time.monotonic()
        while pending or active:
            if (output / "STOP").exists() or (control / "STOP").exists():
                raise InterruptedError("STOP requested")
            while pending and len(active) < workers:
                item = pending.popleft()
                active[pool.submit(evaluate_one, item)] = item
            finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in finished:
                item = active.pop(future)
                try:
                    collect(item, future.result())
                except Exception as exc:
                    errors.append({"item": item, "error": repr(exc)})
            if time.monotonic() - last >= 30 or not (pending or active):
                report()
                last = time.monotonic()
    if errors or len(done) != len(queue):
        raise RuntimeError(f"{len(errors)} unresolved evaluations; successful outcomes preserved")
    for mode in MODES:
        directory = output / mode / "accumulation"
        module = frozen(str(directory / "source_snapshot/scripts/ace/run_reflection_accumulation.py"))
        for seed in read(directory / "protocol.json")["seeds"]:
            module["summarize_seed"](directory, seed)
        write(directory / "status.json", {"state": "completed", "phase": "all_complete", "seed": 44,
              "completed": 2700, "total": 2700, "running": 0, "pending": 0, "errors": 0, "updated_at": now()})
    results = {phase: [done[item["root"], item["job"]["id"]] for item in queue
                       if item["kind"] == "advantage" and item["job"]["phase"] == phase] for phase in PHASES}
    directory = output / "advantage/evaluations"
    for phase, rows in results.items():
        write(directory / f"{phase}_results.json", rows)
    evaluator = runpy.run_path(str(ROOT / "scripts/ace/evaluate_advantage_accumulation.py"))
    write(directory / "summary.json", evaluator["summarize"](results))
    write(directory / "status.json", {"state": "completed", "phase": "completed", "updated_at": now(),
          "completed": sum(map(len, results.values())), "total": sum(map(len, results.values()))})
    write(control / "evaluation_completed.json", {"completed": len(done), "reused": reused, "at": now()})
    emit(output, control, "all_complete", state="completed", completed=len(done), total=len(queue))


def execute(output, control, jobs, workers):
    train_all(output, control, jobs)
    require_all_trained(jobs)
    frozen(jobs[0]["source"])["journal"](
        output, "running", "60개 Context 학습 완료. 전체 미완료 평가를 공유 queue에 등록하여 동시성 100으로 평가합니다."
    )
    queue = build_queue(output, control, jobs)
    evaluate_all(output, control, queue, workers)
    frozen(jobs[0]["source"])["journal"](
        output, "completed", "60개 Context의 Train·BWT·Held-out 전체 평가 완료. 기존 완료 평가를 보존했습니다."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--evaluation-workers", type=int, default=100)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output, control = args.output.resolve(), args.control.resolve()
    control.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.evaluation_workers <= 100:
        raise ValueError("Expected at most 100 evaluation workers")
    with FileLock(str(output / "scheduler.lock"), timeout=0):
        try:
            jobs = prepare(output, control)
            if args.prepare_only:
                print(json.dumps({"total_contexts": len(jobs), "completed": sum(completed(j) for j in jobs),
                                  "remaining": sum(not completed(j) for j in jobs),
                                  "total_training_steps": sum(j["steps"] for j in jobs)}, indent=2))
                return
            write(control / "process.json", {"pid": os.getpid(), "started_at": now(), "command": sys.argv})
            execute(output, control, jobs, args.evaluation_workers)
        except Exception as exc:
            emit(output, control, "error", state="error", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
