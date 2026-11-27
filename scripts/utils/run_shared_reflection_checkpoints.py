# SPDX-License-Identifier: Apache-2.0
"""Evaluate the shared-reflector cohort at ten-task boundaries through one queue."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.utils import checkpoint_sources as sources  # noqa: E402
from scripts.utils.checkpoint_queue import Queue, now  # noqa: E402
from scripts.utils.run_shared_reflection_matrix import complete, read, write  # noqa: E402

_OUTPUT = None


def streams(source):
    plan = read(source / "launch_manifest.json")
    if sources.digest(Path(plan["inference_profile"]).read_bytes()) != plan["inference_profile_sha256"]:
        raise ValueError("Training inference profile changed")
    specs = []
    for job in plan["jobs"]:
        folder = Path(job["train_dir"])
        config = read(folder / ("state.json" if job["method"] == "premise" else
                                "experiment_config.json" if job["method"] == "ace" else "config.json"))
        if job["method"] == "premise":
            config = config["config"]
        kwargs = dict(config["benchmark_configs"][job["benchmark"]] if job["method"] in {"ace", "premise"}
                      else config["benchmark_kwargs"])
        kwargs["use_cache"] = False
        spec = {"id": sources.digest(str(Path(job["output"]).resolve()))[:20], "job_id": job["id"],
                "root": str(source), "source_output": job["output"], "train_dir": str(folder),
                "output": str(folder) if job["method"] in {"gepa", "skillopt"} else job["output"],
                "snapshot": plan["snapshot"], "variant": source.name, "run_variant": job["variant"],
                "method": job["method"], "benchmark": job["benchmark"], "seed": job["seed"],
                "model": "openai/" + plan["model"].removeprefix("openai/"),
                "temperature": 0.0, "actor_temperature": 0.0,
                "reasoning_effort": "medium", "inference_profile_sha256": plan["inference_profile_sha256"],
                "actor_context": config["actor_context"], "actor_context_metadata": config["actor_context_metadata"],
                "playbook_edit_mode": job.get("playbook_edit_mode", "add_only"),
                "train_total": job["train_count"], "benchmark_kwargs": kwargs,
                "split_sha256": plan["manifest_sha256"],
                "splits": {split: list(map(str, job[f"{split}_ids"])) for split in ("train", "val", "test")}}
        if spec["actor_context"] != "shared-empty-v1":
            raise ValueError("This evaluator requires the cohort's common actor context")
        split_sets = list(map(set, spec["splits"].values()))
        if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Checkpoint split memberships overlap")
        if job["benchmark"] == "appworld":
            policy = Path(plan["appworld_policy"]).read_text()
            if sources.digest(policy) != plan["appworld_policy_sha256"] or kwargs.get("policy_override") != policy:
                raise ValueError("Checkpoint AppWorld policy differs from training")
            spec.update(appworld_policy_override=policy, appworld_policy_sha256=plan["appworld_policy_sha256"])
        specs.append(spec)
    return plan, specs


def reuse_rollouts(queue, spec):
    """Only reuse matching frozen prompts, task membership, and inference settings."""
    from exgentic.actor_context import shared_actor_prompt

    checkpoints = queue.db.execute("SELECT * FROM checkpoints WHERE stream=? AND canonical=id",
                                   (spec["id"],)).fetchall()
    by_policy = {cp["sha"]: cp for cp in checkpoints}
    found = {}
    source_config = read(Path(spec["train_dir"]) / "config.json")
    for path in sorted((Path(spec["train_dir"]) / "rollouts").glob("*/request.json")):
        request = read(path)
        policy_sha = request.get("prompt_sha256")  # Runtime cache identity hashes the learned context.
        if policy_sha not in by_policy:
            continue
        content = json.loads(by_policy[policy_sha]["payload"])["content"]
        expected_prompt = shared_actor_prompt(spec["benchmark"], content)
        prompt_sha = sources.digest(expected_prompt)
        item = request.get("item") or {}
        split, task_id = item.get("split"), str(item.get("task_id"))
        if not (task_id in spec["splits"].get(split, []) and request.get("actor_temperature") == 0.0
                and request.get("reasoning_effort") == "medium" and request.get("model") == spec["model"]
                and request.get("actor_context") == spec["actor_context"]
                and request.get("actor_context_metadata") == spec["actor_context_metadata"]
                and request.get("benchmark_kwargs") == source_config["benchmark_kwargs"]
                and request.get("system_prompt") == expected_prompt
                and sources.digest(request.get("learned_policy", "")) == policy_sha):
            continue
        result_path = path.with_name("result.json")
        if not result_path.exists():
            continue
        result = read(result_path)
        if not (result.get("id") == task_id and type(result.get("hard")) is int and result["hard"] in {0, 1}
                and result.get("prompt_sha256") == prompt_sha and result.get("run_id") and "error" not in result):
            continue
        key = (by_policy[policy_sha]["id"], split, task_id)
        entry = {"status": "completed", "score": result["hard"], "partial_score": result.get("soft"),
                 "reused_from": str(result_path), "source_request": str(path), "prompt_sha256": prompt_sha,
                 "inference_profile_sha256": spec["inference_profile_sha256"]}
        if key in found and found[key] is not None and found[key]["score"] != entry["score"]:
            found[key] = None  # Conflicting repeated measurements require a fresh evaluation.
        elif key not in found:
            found[key] = entry
    with queue.db:
        for (cp, split, task_id), result in found.items():
            if result is not None:
                queue.db.execute("""UPDATE tasks SET status='completed',result=?,ended=?
                    WHERE cp=? AND split=? AND task_id=? AND status='pending'""",
                                 (json.dumps(result), now(), cp, split, task_id))


def reuse_native_test(queue, spec, job, plan):
    from exgentic.actor_context import shared_actor_prompt

    if not complete(job, plan):
        return
    frozen = read(Path(spec["source_output"]) / "evaluation/frozen_checkpoint.json")
    if frozen.get("actor_context_metadata") != spec["actor_context_metadata"]:
        raise ValueError("Final evaluation actor context differs from the checkpoint")
    sha = sources.digest(frozen["playbook"])
    prompt_sha = sources.digest(shared_actor_prompt(spec["benchmark"], frozen["playbook"]))
    for cp in queue.db.execute("SELECT * FROM checkpoints WHERE stream=? AND sha=? AND canonical=id",
                               (spec["id"], sha)).fetchall():
        with queue.db:
            for path in sorted((Path(spec["source_output"]) / "evaluation/records").glob("*.json")):
                row = read(path)
                if (str(row.get("task_id")) not in spec["splits"]["test"] or row.get("infrastructure_error")
                        or type(row.get("success")) is not bool or row.get("score") != int(row["success"])
                        or row.get("actor_prompt_sha256") != prompt_sha
                        or not row.get("session_id") or not row.get("run_id")):
                    continue
                result = {**row, "status": "completed", "reused_from": str(path),
                          "inference_profile_sha256": spec["inference_profile_sha256"]}
                queue.db.execute("""UPDATE tasks SET status='completed',result=?,ended=?
                    WHERE cp=? AND split='test' AND task_id=? AND status='pending'""",
                                 (json.dumps(result), now(), cp["id"], str(row["task_id"])))


def collect(queue, source, *, reuse=True):
    plan, specs = streams(source)
    reader, issues = sources.Reader(), []
    jobs = {job["id"]: job for job in plan["jobs"]}
    for spec in specs:
        try:
            snapshots = [row for row in sources.checkpoints(spec, reader) if row["point"] % 10 == 0]
            job = jobs[spec["job_id"]]
            if complete(job, plan):
                if not snapshots or snapshots[-1]["point"] != spec["train_total"]:
                    raise ValueError("Completed stream is missing its final checkpoint")
                final = read(Path(spec["train_dir"]) / "frozen_instruction.json")["content"] if spec["method"] in {
                    "gepa", "skillopt"} else read(Path(spec["source_output"]) / "evaluation/frozen_checkpoint.json")[
                        "playbook"]
                if snapshots[-1]["content"] != final:
                    raise ValueError("Recovered final checkpoint differs from the evaluated final policy")
            for snapshot in snapshots:
                queue.register(spec, snapshot)
            if reuse:
                if spec["method"] in {"gepa", "skillopt"}:
                    reuse_rollouts(queue, spec)
                else:
                    reuse_native_test(queue, spec, job, plan)
        except (OSError, ValueError, KeyError, EOFError) as exc:
            issues.append({"job": spec["job_id"], "error": f"{type(exc).__name__}: {exc}"})
    return specs, issues


def initialize(output):
    global _OUTPUT
    _OUTPUT = Path(output)
    os.environ["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("docker") is None:
        raise RuntimeError("The evaluation environment is missing the training Docker executable")
    os.environ.pop("EXGENTIC_PORT_BLOCK_START", None)


def evaluate(task):
    from scripts.baselines.actor import run as actor
    from scripts.utils.run_shared_reflection_matrix import evaluate_task

    payload = read(task["checkpoint_file"])
    spec, content = payload["stream"], payload["content"]
    if sources.digest(content) != task["content_sha256"]:
        raise ValueError("Frozen checkpoint policy changed")
    if os.environ.get("EXGENTIC_INFERENCE_PROFILE_SHA256") != spec["inference_profile_sha256"]:
        raise ValueError("Checkpoint inference profile mismatch")
    folder = _OUTPUT / "sessions" / f"task_{task['id']:07d}" / f"attempt_{task['attempt']}"
    try:
        if spec["method"] in {"ace", "premise"}:
            from exgentic.agents.ace.playbook_store import PlaybookStore

            PlaybookStore.reset_all()
            job = {**spec, "id": f"{spec['job_id']}_cp{payload['point']:03d}_{task['split']}", "output": str(folder)}
            row = evaluate_task((job, {"benchmark_configs": {spec["benchmark"]: spec["benchmark_kwargs"]}},
                                 content, task["task_id"], task["id"], task["attempt"]))
            return {**row, "status": "completed"}
        from exgentic.actor_context import shared_actor_prompt

        result = actor({"method": spec["method"], "benchmark": spec["benchmark"], "model": spec["model"],
                        "benchmark_kwargs": spec["benchmark_kwargs"], "actor_context": spec["actor_context"],
                        "system_prompt": shared_actor_prompt(spec["benchmark"], content), "learned_policy": content,
                        "actor_temperature": 0.0, "reasoning_effort": "medium",
                        "item": {"task_id": task["task_id"], "split": task["split"]}, "session_output": str(folder)})
        return {"status": "completed", "score": result["hard"], "partial_score": result["soft"],
                "run_id": result["run_id"], "prompt_sha256": result["prompt_sha256"],
                "session_dir": result["trace"].get("session_dir")}
    except Exception as exc:
        return {"status": "error", "score": None, "error": f"{type(exc).__name__}: {exc}"}


def run(output, source, *, once=False):
    plan = read(output / "checkpoint_queue_manifest.json")
    queue = Queue(output, interval=10)
    specs, issues = collect(queue, source)
    write(output / "collection.json", {"at": now(), "streams": len(specs), "issues": issues})
    if once:
        write(output / "dashboard.json", queue.dashboard(issues, workers=plan["workers"]))
        return
    if issues:
        raise ValueError("Checkpoint discovery failed; inspect collection.json")
    queue.recover_interrupted()
    queue.db.execute("UPDATE tasks SET priority=point FROM checkpoints WHERE checkpoints.id=tasks.cp")
    queue.db.commit()
    from scripts.utils.process_identity import stable_identity

    write(output / "controller_process.json", {"pid": os.getpid(), "started_at": now(), **stable_identity(os.getpid())})
    active, last_scan, last_reuse = {}, time.monotonic(), time.monotonic()
    initialize(output)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=plan["workers"], mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize, initargs=(str(output),),
    ) as pool:
        while True:
            for future, task in list(active.items()):
                if future.done():
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"status": "error", "score": None, "error": f"{type(exc).__name__}: {exc}"}
                    queue.finish(task, result)
                    del active[future]
            if time.monotonic() - last_scan >= 60:
                do_reuse = time.monotonic() - last_reuse >= 1800
                specs, issues = collect(queue, source, reuse=do_reuse)
                last_scan = time.monotonic()
                if do_reuse:
                    last_reuse = last_scan
            if not (output / "pause_dispatch.json").exists() and not issues:
                for row in queue.pending(max(0, plan["workers"] - len(active))):
                    task = queue.claim(row)
                    active[pool.submit(evaluate, task)] = task
            dashboard = queue.dashboard(issues, workers=plan["workers"])
            dashboard["expected_checkpoints"] = sum(spec["train_total"] // 10 for spec in specs)
            dashboard["scope_label"] = plan["scope_label"]
            dashboard["gpu_queries"] = router_status(plan["query_router_url"])
            drained = (not active and not issues and dashboard["checkpoints"] == dashboard["expected_checkpoints"]
                       and not any(dashboard["task_counts"].get(s) for s in ("pending", "running")))
            if drained:
                dashboard["phase"] = "error" if dashboard["task_counts"].get("error") else "completed"
            elif (output / "pause_dispatch.json").exists():
                dashboard["phase"] = "paused"
            write(output / "dashboard.json", dashboard)
            write(output / "heartbeat.json", {"at": now(), "active": len(active), "workers": plan["workers"]})
            if drained:
                break
            time.sleep(2)


def router_status(url):
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError):
        return {"unavailable": True}


def main():
    from filelock import FileLock

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    with FileLock(str(args.output / "controller.lock"), timeout=0):
        run(args.output.resolve(), args.source.resolve(), once=args.collect_only)


if __name__ == "__main__":
    main()
