# SPDX-License-Identifier: Apache-2.0
"""Watch completed epoch-two policies and evaluate unique train/val tasks.

Uses the checkpoint queue's durable records/live dashboard. Test results are
imported from the existing final-test runner, never dispatched by this queue.
Each inference subprocess imports the corresponding frozen training actor.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def training_finished(job):
    folder = Path(job["train_dir"])
    method = job["method"]
    if method == "ace":
        marker = folder / "training_complete.json"
        return marker.exists() and read(marker).get("completed_task_count") == job["train_count"]
    if method == "premise":
        return read(folder / "state.json").get("cursor") == job["train_count"]
    return (folder / "training_result.json").exists() and len(list((folder / "online").glob("*.json"))) == job[
        "train_count"
    ]


def final_snapshot(job, source_root):
    if not training_finished(job):
        return None
    folder = Path(job["train_dir"])
    method, benchmark = job["method"], job["benchmark"]
    if method == "ace":
        source = folder / f"playbook_ace_isolated_{benchmark}.json"
        content = read(source)["playbook"]
        selection = "ACE final post-curation playbook"
    elif method == "premise":
        source = folder / "state.json"
        content = read(source)["states"][benchmark]["beam"][0]["playbook"]
        selection = (
            f"PEACE strict-F1 span policy shift, past={job['past_count']}, final current/past Pareto rank 1"
            if job.get("candidate_evaluation") == "span_policy_shift"
            else "PEACE final selected beam rank 1, AUD without alignment"
        )
        if job.get("proposal_retrieval"):
            selection = f"PEACE proposal retrieval {job['proposal_retrieval']}, " + (
                f"strict-F1 span policy shift, past={job['past_count']}, current/past Pareto rank 1"
                if job.get("candidate_evaluation") == "span_policy_shift"
                else "real-replay official-score Pareto rank 1"
            )
        if job.get("grounded_proposals"):
            selection = ("PEACE reviewed evidence + strict A/B retrieval + conditional proposal, "
                         "real-replay Pareto rank 1")
    else:
        source = folder / "frozen_instruction.json"
        if not source.exists():
            return None  # The final-test entrypoint freezes this immediately after training.
        frozen = read(source)
        content = frozen["content"]
        if digest(content) != frozen["sha256"]:
            raise ValueError("Final instruction hash differs")
        selection = "Final policy selected by validation, identical to final-test policy"
    if method in ("ace", "premise"):
        test_policy = Path(job["output"]) / "evaluation/frozen_checkpoint.json"
        if test_policy.exists() and read(test_policy)["playbook"] != content:
            raise ValueError("Final train/val checkpoint differs from the final-test checkpoint")
    original = job["source_spec"]
    splits = {name: list(original["splits"][name]) for name in ("train", "val", "test")}
    for name, tasks in splits.items():
        if len(tasks) != len(set(tasks)):
            raise ValueError(f"Evaluation split contains repeated IDs: {name}")
    if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Evaluation split memberships overlap")
    orders = [job["epoch1_order"]] + ([job["epoch2_order"]] if job.get("epochs", 2) > 1 else [])
    if any(set(order) != set(splits["train"]) for order in orders):
        raise ValueError("Repeated epoch memberships differ from the evaluation train split")
    spec = {
        **original,
        "id": digest(str(Path(job["output"]).resolve()))[:20],
        "root": str(source_root),
        "source_output": job["output"],
        "train_dir": job["train_dir"],
        "output": job["train_dir"] if method in ("gepa", "skillopt") else job["output"],
        "variant": source_root.name,
        "run_variant": job["variant"],
        "snapshot": job["source_snapshot"],
        "model": "openai/" + original["model"].removeprefix("openai/"),
        "train_total": job["train_count"],
        "splits": splits,
    }
    if benchmark == "appworld":
        if digest(spec["appworld_policy_override"]) != spec["appworld_policy_sha256"]:
            raise ValueError("AppWorld environment policy changed")
        if spec["benchmark_kwargs"].get("interaction_mode") != "code_generation":
            raise ValueError("AppWorld must retain Original policy/code generation")
    snapshot = {
        "point": job["train_count"],
        "content": content,
        "selection": selection,
        "source": str(source),
        "source_sha256": digest(source.read_bytes()),
    }
    return spec, snapshot


def collect(queue, plan):
    source_root = Path(plan["source_output"])
    jobs = read(source_root / "launch_manifest.json")["jobs"]
    registered = 0
    issues = []
    for job in jobs:
        try:
            result = final_snapshot(job, source_root)
            if result is None:
                continue
            spec, snapshot = result
            queue.register(spec, snapshot)
            queue.import_final(spec)
            registered += 1
        except (OSError, ValueError, KeyError) as exc:
            issues.append({"job": job["id"], "error": f"{type(exc).__name__}: {exc}"})
    return registered, issues


def eligible_tasks(queue, limit):
    # Existing final-test jobs own test; this controller only imports their results.
    return queue.db.execute(
        """SELECT t.*, c.file AS checkpoint_file, c.sha AS content_sha256
        FROM tasks t JOIN checkpoints c ON c.id=t.cp
        WHERE t.status='pending' AND t.split IN ('train','val')
        ORDER BY t.ordinal,t.id LIMIT ?""",
        (limit,),
    ).fetchall()


def evaluate_subprocess(output, task, plan):
    payload = read(task["checkpoint_file"])
    snapshot = Path(payload["stream"]["snapshot"])
    folder = output / "dispatch" / f"{task['id']:07d}" / f"attempt_{task['attempt']}"
    request, response = folder / "request.json", folder / "response.json"
    write(
        request,
        {
            "task": task,
            "queue_output": str(output),
            "base_url": plan["base_url"],
            "snapshot": str(snapshot),
            "worker_module": plan["worker_module"],
        },
    )
    environment = dict(os.environ)
    environment.update(read(Path(plan["source_output"]) / "launch_manifest.json")["environment"])
    environment.update(
        OPENAI_API_BASE=plan["base_url"],
        OPENAI_BASE_URL=plan["base_url"],
        OPENAI_API_KEY="EMPTY",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    with (folder / "worker.log").open("a") as log:
        process = subprocess.run(
            [
                str(snapshot / ".venv/bin/python"),
                "-u",
                str(Path(__file__).resolve()),
                "task",
                "--request",
                str(request),
                "--response",
                str(response),
            ],
            cwd=snapshot,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode == 0 and response.exists():
        return read(response)
    return {"status": "error", "score": None, "error": f"Evaluation process exited {process.returncode}; see {folder}"}


def evaluate_request(request, response):
    snapshot = Path(request["snapshot"])
    sys.path[:0] = [str(snapshot / "src"), str(snapshot)]
    module_spec = importlib.util.spec_from_file_location("epoch2_frozen_checkpoint_worker", request["worker_module"])
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.initialize(request["queue_output"], request["base_url"])
    result = module.evaluate(request["task"])
    write(response, result)


def prepare(output, source, workers, base_url):
    from scripts.utils.checkpoint_queue import Queue

    if (output / "checkpoint_queue_manifest.json").exists():
        return read(output / "checkpoint_queue_manifest.json")
    output.mkdir(parents=True, exist_ok=True)
    import shutil

    frozen = output / "final_split_worker.py"
    shutil.copy2(__file__, frozen)
    worker = output / "checkpoint_eval_worker.py"
    shutil.copy2(ROOT / "scripts/utils/checkpoint_eval_worker.py", worker)
    plan = {
        "source_output": str(source.resolve()),
        "workers": workers,
        "base_url": base_url,
        "checkpoint_interval": 10,
        "share_initial": False,
        "source_journal_ids": ["exp-20261109-68f88079"],
        "scope_label": "Epoch2 최종 CP · frozen Train/Val · 기존 Test 재사용 · 동시성50",
        "evaluation_splits": ["train", "val"],
        "test_mode": "reuse_existing_only",
        "source_job_count": len(read(source / "launch_manifest.json")["jobs"]),
        "worker_module": str(worker),
        "worker_sha256": digest(worker.read_bytes()),
        "controller_module": str(frozen),
        "controller_sha256": digest(frozen.read_bytes()),
    }
    write(output / "checkpoint_queue_manifest.json", plan)
    queue = Queue(output, interval=10)
    count, issues = collect(queue, plan)
    dashboard = queue.dashboard(issues, workers=workers)
    dashboard.update(phase="planned", training_jobs_registered=count, training_jobs_total=plan["source_job_count"])
    write(output / "dashboard.json", dashboard)
    queue.db.close()
    return plan


def run(output):
    from scripts.utils.checkpoint_queue import Queue

    plan = read(output / "checkpoint_queue_manifest.json")
    if digest(Path(plan["controller_module"]).read_bytes()) != plan["controller_sha256"]:
        raise ValueError("Frozen final-split controller changed")
    if digest(Path(plan["worker_module"]).read_bytes()) != plan["worker_sha256"]:
        raise ValueError("Frozen evaluation worker changed")
    queue = Queue(output, interval=10)
    queue.recover_interrupted()
    registered, issues = collect(queue, plan)
    workers = plan["workers"]
    active = {}
    last_scan = 0
    ready = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            if time.monotonic() - last_scan >= 10:
                registered, issues = collect(queue, plan)
                try:
                    with urllib.request.urlopen(plan["base_url"].rstrip("/") + "/models", timeout=5) as response:
                        ready = any("Qwen/Qwen3.6-35B-A3B" in row["id"] for row in json.load(response)["data"])
                except (OSError, ValueError, KeyError):
                    ready = False
                last_scan = time.monotonic()
            for future, task in list(active.items()):
                if future.done():
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"status": "error", "score": None, "error": f"{type(exc).__name__}: {exc}"}
                    queue.finish(task, result)
                    del active[future]
            if ready and not (output / "pause_dispatch.json").exists():
                for row in eligible_tasks(queue, max(0, workers - len(active))):
                    task = queue.claim(row)
                    active[pool.submit(evaluate_subprocess, output, task, plan)] = task
            dashboard = queue.dashboard(issues, workers=workers)
            counts = dashboard["task_counts"]
            drained = (
                registered == plan["source_job_count"]
                and not issues
                and not active
                and not counts.get("pending")
                and not counts.get("running")
            )
            dashboard.update(
                training_jobs_registered=registered,
                training_jobs_total=plan["source_job_count"],
                phase=("error" if counts.get("error") else "completed")
                if drained
                else (
                    "waiting_model"
                    if not ready
                    else "paused"
                    if (output / "pause_dispatch.json").exists()
                    else "running"
                    if active
                    else "watching"
                ),
            )
            write(output / "dashboard.json", dashboard)
            if drained:
                break
            time.sleep(2)
    queue.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "task"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    args = parser.parse_args()
    if args.action == "task":
        evaluate_request(read(args.request), args.response)
    elif args.action == "prepare":
        prepare(args.output.resolve(), args.source.resolve(), args.workers, args.base_url)
    else:
        with (args.output / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args.output.resolve())


if __name__ == "__main__":
    main()
