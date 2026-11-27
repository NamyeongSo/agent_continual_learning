# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Evaluate completed final contexts through one durable fifty-task queue, without intermediate CPs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_WORKER = None


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_finished(job):
    folder = Path(job["train_dir"])
    marker = folder / "training_complete.json"
    if not marker.exists() or read(marker).get("completed_task_count") != job["train_count"]:
        return False
    if job["method"] == "premise":
        state = folder / "state.json"
        return state.exists() and read(state).get("cursor") == job["train_count"]
    return True


def register_context(queue, child, job, spec, policy_sha):
    from scripts.utils.run_qwen_isolated_when_free import frozen_evaluation_jobs

    if (job["method"] not in {"ace", "premise"} or job["benchmark"] != "appworld"
            or len(job["test_ids"]) != 50 or len(set(job["test_ids"])) != 50):
        raise ValueError("Expected fifty distinct AppWorld test tasks per final context")
    if spec["appworld_policy_sha256"] != policy_sha or spec["splits"]["test"] != list(map(str, job["test_ids"])):
        raise ValueError("Final evaluation policy or split differs from training")
    evaluations = frozen_evaluation_jobs(job)
    existing = queue.db.execute("SELECT id FROM checkpoints WHERE stream=? AND point=?",
                                (spec["id"], job["train_count"])).fetchone()
    if not existing and list((Path(job["output"]) / "evaluation/records").glob("*.json")):
        raise ValueError("Existing final results need explicit reconciliation before a new queue")
    frozen = read(evaluations[0]["frozen"])
    spec = {**spec, "splits": {"test": spec["splits"]["test"]}}
    queue.register(spec, {"point": job["train_count"], "content": frozen["playbook"],
                          "selection": "final_completed_training_context"})
    return {"stream": spec["id"], "method": job["method"], "source_root": child["root"],
            "source_output": job["output"], "journal_id": child["journal_id"],
            "variant": spec["playbook_edit_mode"], "seed": job["seed"], "evaluations": evaluations,
            "frozen": evaluations[0]["frozen"], "frozen_sha256": evaluations[0]["frozen_sha256"],
            "policy_sha256": hashlib.sha256(frozen["playbook"].encode()).hexdigest()}


def completed_sources(plan):
    from scripts.utils.checkpoint_sources import Reader, streams

    info = read(Path(plan["cohort"]) / "cohort.json")
    children = [child for child in info["children"] if child["kind"] == "legacy"]
    specs = {spec["output"]: spec for spec in streams([child["root"] for child in children], Reader())}
    existing = {c["source_output"] for c in plan["contexts"]}
    return [(child, job, specs[job["output"]], info["appworld_policy_sha256"])
            for child in children for job in read(Path(child["root"]) / "progress.json")["jobs"]
            if job["method"] in {"ace", "premise"} and job["output"] not in existing and training_finished(job)]


def collect_finished(queue, plan, output):
    added = []
    for child, job, spec, policy_sha in completed_sources(plan):
        context = register_context(queue, child, job, spec, policy_sha)
        plan["contexts"].append(context)
        added.append(context)
    if added:
        plan["tasks"] = queue.db.execute("SELECT count(*) FROM tasks").fetchone()[0]
        plan["last_registered_at"] = now()
        write(output / "final_test_queue_manifest.json", plan)
        export_tuples(queue, plan, output)
        for context in added:
            print(json.dumps({"event": "final_context_registered", "method": context["method"],
                              "variant": context["variant"], "seed": context["seed"], "tasks": 50}), flush=True)
    return added


def export_tuples(queue, plan, output):
    contexts = {c["stream"]: c for c in plan["contexts"]}
    temporary = output / "context_task_tuples.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tuple_id", "context_id", "method", "variant", "seed", "context_file",
                         "context_sha256", "task_id", "split"])
        for row in queue.db.execute("""SELECT t.id,t.task_id,t.split,c.stream,c.file,c.sha FROM tasks t
            JOIN checkpoints c ON c.id=t.cp ORDER BY t.ordinal,t.id"""):
            context = contexts[row["stream"]]
            writer.writerow([row["id"], row["stream"], context.get("method", "ace"), context["variant"],
                             context["seed"], row["file"], row["sha"], row["task_id"], row["split"]])
    temporary.replace(output / "context_task_tuples.csv")


def all_registered(plan):
    expected = {(method, mode, seed) for method in ("ace", "premise")
                for mode in ("add_only", "add_update_delete") for seed in (42, 43, 44)}
    return {(c.get("method", "ace"), c["variant"], c["seed"]) for c in plan["contexts"]} == expected


def enable_peace(output):
    path = output / "final_test_queue_manifest.json"
    plan = read(path)
    backup = output / "manifest_before_peace_watch.json"
    if not backup.exists():
        write(backup, plan)
    plan.update(watch_completed_training=True, expected_contexts=12, expected_tasks=600,
                registration_interval_seconds=5)
    plan.setdefault("runtime_revisions", []).append({"at": now(), "runner_sha256": sha(__file__),
                                                     "change": "Auto-enroll completed PEACE final contexts"})
    write(path, plan)
    return plan


def prepare(cohort, output, base_url):
    from scripts.utils.checkpoint_queue import Queue
    from scripts.utils.checkpoint_sources import Reader, streams
    from scripts.utils.run_qwen_isolated_when_free import frozen_evaluation_jobs

    if (output / "final_test_queue_manifest.json").exists():
        raise ValueError("Final test queue already prepared; use --run to resume")
    info = read(cohort / "cohort.json")
    children = [c for c in info["children"] if c["kind"] == "legacy"]
    specs = streams([Path(c["root"]) for c in children], Reader())
    contexts = []
    queue = Queue(output)
    for child in children:
        root = Path(child["root"])
        for job in read(root / "progress.json")["jobs"]:
            if job["method"] != "ace":
                continue
            if job["benchmark"] != "appworld" or len(job["test_ids"]) != 50:
                raise ValueError("Expected fifty AppWorld test tasks per final ACE context")
            if len(set(job["test_ids"])) != 50:
                raise ValueError("Duplicate test IDs")
            evaluations = frozen_evaluation_jobs(job)
            spec = next(s for s in specs if s["output"] == job["output"])
            if (spec["appworld_policy_sha256"] != info["appworld_policy_sha256"]
                    or spec["splits"]["test"] != list(map(str, job["test_ids"]))):
                raise ValueError("Final evaluation policy or split differs from training")
            if list((Path(job["output"]) / "evaluation/records").glob("*.json")):
                raise ValueError("Existing final results need explicit reconciliation before a new queue")
            frozen = read(evaluations[0]["frozen"])
            spec = {**spec, "splits": {"test": spec["splits"]["test"]}}
            queue.register(spec, {"point": job["train_count"], "content": frozen["playbook"],
                                  "selection": "final_completed_training_context"})
            contexts.append({"stream": spec["id"], "source_root": str(root), "source_output": job["output"],
                             "journal_id": child["journal_id"], "variant": spec["playbook_edit_mode"],
                             "seed": job["seed"], "evaluations": evaluations,
                             "frozen": evaluations[0]["frozen"], "frozen_sha256": evaluations[0]["frozen_sha256"],
                             "policy_sha256": hashlib.sha256(frozen["playbook"].encode()).hexdigest()})
    expected = {(mode, seed) for mode in ("add_only", "add_update_delete") for seed in (42, 43, 44)}
    if {(c["variant"], c["seed"]) for c in contexts} != expected or len(contexts) != 6:
        raise ValueError("Expected six distinct ACE final contexts")
    count = queue.db.execute("SELECT count(*) FROM tasks").fetchone()[0]
    if count != 300 or queue.db.execute("SELECT count(*) FROM tasks WHERE split!='test'").fetchone()[0]:
        raise ValueError("The final-only queue must contain exactly 300 test tuples")
    plan = {"created_at": now(), "cohort": str(cohort), "snapshot": info["snapshot"], "base_url": base_url,
            "workers": 50, "tasks": count, "contexts": contexts, "temperature": 0.7,
            "metric": "appworld_tgc_v1", "intermediate_checkpoint_evaluation": False,
            "checkpoint_queue_output": info["queue_output"], "max_infrastructure_attempts": 3,
            "runner_sha256": sha(__file__)}
    write(output / "final_test_queue_manifest.json", plan)
    queue.db.close()
    return plan


def worker_initialize(output, base_url, snapshot):
    global _WORKER
    snapshot = Path(snapshot)
    sys.path[:0] = [str(snapshot), str(snapshot / "src")]
    from scripts.utils import checkpoint_eval_worker

    if not Path(checkpoint_eval_worker.__file__).resolve().is_relative_to(snapshot.resolve()):
        raise RuntimeError("Evaluation must use the existing frozen runtime")
    checkpoint_eval_worker.initialize(output, base_url)
    _WORKER = checkpoint_eval_worker


def worker_evaluate(task):
    return _WORKER.evaluate(task)


def publish_attempt(output, context, task, result, terminal):
    if sha(context["frozen"]) != context["frozen_sha256"]:
        raise ValueError("Source final context changed during evaluation")
    original = next(j for j in context["evaluations"] if str(j["task_id"]) == str(task["task_id"]))
    record = {**original, **result, "attempt": task["attempt"], "final_test_queue": str(output)}
    folder = Path(context["source_output"]) / "evaluation"
    write(folder / "attempts" / f"{original['index']:03d}_{task['attempt']}.json", record)
    if terminal:
        write(folder / "records" / f"{original['index']:03d}.json", record)


def summaries(output, contexts, phase):
    for context in contexts:
        folder = Path(context["source_output"]) / "evaluation"
        records = [read(p) for p in (folder / "records").glob("*.json")]
        errors = sum(r["status"] == "error" for r in records)
        measured = [r["score"] for r in records if r["status"] == "completed"]
        complete = len(records) == len(context["evaluations"])
        score = sum(measured) / len(measured) if measured else None
        write(folder / "summary.json", {"complete": complete, "tasks": len(records), "expected_tasks": 50,
              "execution_errors": errors, "score": score if not errors else None,
              "temperature_all_roles": 0.7, "playbook_edit_mode": context["variant"],
              "score_metric": "appworld_tgc_v1", "final_test_queue": str(output)})
        write(folder / "live_evaluation.json", {"phase": "completed" if complete and not errors else (
              "error" if complete and errors else phase), "controller_pid": os.getpid(), "queue_output": str(output),
              "policy_sha256": context["policy_sha256"], "at": now(), "workers": 50})


def bounded_dispatch(queue, pool, active, limit=50):
    for row in queue.pending(max(0, limit - len(active))):
        task = queue.claim(row)
        active[pool.submit(worker_evaluate, task)] = task


def run(output):
    from scripts.utils.checkpoint_queue import Queue, stop_orphan_workers
    from scripts.utils.process_identity import stable_identity

    plan = read(output / "final_test_queue_manifest.json")
    watching = plan.get("watch_completed_training", False)
    if (plan["workers"] != 50 or not 300 <= plan["tasks"] <= 600 or plan["intermediate_checkpoint_evaluation"]
            or (watching and (plan["expected_contexts"] != 12 or plan["expected_tasks"] != 600))):
        raise ValueError("Unexpected final evaluation scope")
    with FileLock(str(output / "controller.lock"), timeout=0):
        stop_orphan_workers(output)
        write(output / "controller_process.json", {"pid": os.getpid(), "started_at": now(),
                                                  **stable_identity(os.getpid())})
        queue = Queue(output)
        with queue.db:
            queue.db.execute("UPDATE tasks SET status='pending',attempts=max(0,attempts-1) WHERE status='running'")
        contexts = {c["stream"]: c for c in plan["contexts"]}
        cp_streams = {r["id"]: r["stream"] for r in queue.db.execute("SELECT id,stream FROM checkpoints")}
        active = {}
        last_registration = 0
        summaries(output, plan["contexts"], "running")
        pool = ProcessPoolExecutor(max_workers=50, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=worker_initialize,
                                   initargs=(str(output), plan["base_url"], plan["snapshot"]))
        with pool:
            while True:
                if watching and time.monotonic() - last_registration >= plan["registration_interval_seconds"]:
                    collect_finished(queue, plan, output)
                    contexts = {c["stream"]: c for c in plan["contexts"]}
                    cp_streams = {r["id"]: r["stream"] for r in queue.db.execute("SELECT id,stream FROM checkpoints")}
                    last_registration = time.monotonic()
                for future, task in list(active.items()):
                    if not future.done():
                        continue
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"status": "error", "score": None, "error": f"{type(exc).__name__}: {exc}"}
                    if result.get("status") == "completed" and result.get("score") not in (0, 1):
                        result = {**result, "status": "error", "score": None, "error": "Missing official binary TGC"}
                    queue.finish(task, result)
                    terminal = queue.db.execute("SELECT status FROM tasks WHERE id=?", (task["id"],)).fetchone()[0]
                    publish_attempt(output, contexts[cp_streams[task["cp"]]], task, result,
                                    terminal in {"completed", "error"})
                    del active[future]
                bounded_dispatch(queue, pool, active)
                counts = {r[0]: r[1] for r in queue.db.execute("SELECT status,count(*) FROM tasks GROUP BY status")}
                awaiting = watching and not all_registered(plan)
                phase = "running" if active else "watching_training" if awaiting else "finished"
                write(output / "progress.json", {"at": now(), "phase": phase, "workers": 50,
                      "active": len(active), "task_counts": counts, "tasks": plan["tasks"],
                      "expected_tasks": plan.get("expected_tasks", plan["tasks"]),
                      "registered_contexts": len(contexts), "expected_contexts": plan.get("expected_contexts", 6)})
                summaries(output, plan["contexts"], "running" if active else "stopped")
                if not active and not counts.get("pending") and not awaiting:
                    break
                if active:
                    wait(active, timeout=5, return_when=FIRST_COMPLETED)
                else:
                    time.sleep(5)
        queue.db.close()
        return 1 if counts.get("error") else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:19007/v1")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--enable-peace", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.enable_peace:
        with FileLock(str(output / "controller.lock"), timeout=0):
            plan = enable_peace(output)
        print(json.dumps({"expected_contexts": plan["expected_contexts"], "expected_tasks": plan["expected_tasks"],
                          "workers": plan["workers"], "registered_contexts": len(plan["contexts"])}))
        return 0
    if args.prepare:
        if not args.cohort:
            parser.error("--prepare requires --cohort")
        plan = prepare(args.cohort.resolve(), output, args.base_url)
        print(json.dumps({"contexts": len(plan["contexts"]), "tasks": plan["tasks"], "workers": plan["workers"]}))
        return 0
    return run(output)


if __name__ == "__main__":
    raise SystemExit(main())
