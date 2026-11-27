# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Schedule exported GEPA/SkillOpt AppWorld test checkpoints after an existing queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
import statistics
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.utils import evaluate_isolated_checkpoint_queue as shared  # noqa: E402

_OUTPUT: Path | None = None
_PROMPTS: dict[str, str] = {}


def exported_prompt(method: str, raw: bytes) -> str:
    if method == "gepa":
        prompt = json.loads(raw)["best_candidate"]["system_prompt"]
    elif method == "skillopt":
        prompt = raw.decode("utf-8")
    else:
        raise ValueError(f"Unsupported exported method: {method}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Checkpoint has no system prompt")
    return prompt


def prepare(source: Path, output: Path, upstream: Path, workers: int, base_url: str) -> dict:
    from exgentic.utils.task_splits import validate_task_split

    upstream_raw = (upstream / "launch_manifest.json").read_bytes()
    dependency = json.loads(upstream_raw)
    split_raw = (upstream / "task_split_manifest.json").read_bytes()
    if shared.digest(split_raw) != dependency["manifest_sha256"]:
        raise ValueError("Upstream task split hash differs")
    splits = json.loads(split_raw)
    models = {group["model"] for group in dependency["groups"]}
    if len(models) != 1 or dependency["stages"] != ["appworld", "bfcl"]:
        raise ValueError("Expected one model and an AppWorld -> BFCL predecessor")
    model = models.pop()
    groups, identities, covered = [], set(), set()
    for provenance_path in sorted(source.rglob("provenance.json")):
        provenance = shared.read_json(provenance_path)
        method, mode, seed, point = (provenance[key] for key in ("method", "mode", "seed", "checkpoint"))
        identity = (method, mode, seed, point)
        if identity in identities:
            raise ValueError(f"Duplicate checkpoint identity: {identity}")
        identities.add(identity)
        filename = "candidate.json" if method == "gepa" else "skill.md"
        path = provenance_path.parent / filename
        raw = path.read_bytes()
        if shared.digest(raw) != provenance["sha256"]:
            raise ValueError(f"Exported checkpoint hash differs: {path}")
        prompt = exported_prompt(method, raw)
        covered.add(path.resolve())
        if method == "gepa":
            prompt_path = path.with_name("prompt.md")
            if prompt_path.read_text().strip() != prompt.strip():
                raise ValueError(f"GEPA prompt differs from its candidate: {path}")
            covered.add(prompt_path.resolve())
        frozen_dir = output / "frozen" / method / mode / f"s{seed}" / f"cp{point:03d}"
        frozen_dir.mkdir(parents=True, exist_ok=True)
        for original in sorted(path.parent.iterdir()):
            if original.is_file():
                (frozen_dir / original.name).write_bytes(original.read_bytes())
        task_ids = validate_task_split(splits[f"appworld_{seed}"], "appworld")["test"]
        if len(task_ids) != 50:
            raise ValueError("Expected 50 frozen test tasks per seed")
        groups.append({
            "benchmark": "appworld", "method": method, "mode": mode, "seed": seed,
            "checkpoint": point, "split": "test", "task_ids": task_ids,
            "group": f"appworld/{method}/{mode}/s{seed}/cp{point:03d}/test",
            "checkpoint_path": str(frozen_dir / filename), "checkpoint_sha256": shared.digest(raw),
            "prompt_sha256": shared.digest(prompt.encode()), "source": str(path), "provenance": provenance,
            "model": model, "temperature": 0.0, "subset": "test_challenge",
        })
    aliases = []
    for path in sorted(source.rglob("cp*_candidate.json")):
        canonical = path.parent / (path.name.split("_")[0] + "_shared") / "candidate.json"
        if canonical.resolve() not in covered or path.read_bytes() != canonical.read_bytes():
            raise ValueError(f"Unresolved checkpoint outside an exported directory: {path}")
        covered.add(path.resolve())
        aliases.append({"source": str(path), "canonical": str(canonical)})
    for pattern in ("candidate.json", "skill.md", "prompt.md"):
        uncovered = [str(path) for path in source.rglob(pattern) if path.resolve() not in covered]
        if uncovered:
            raise ValueError(f"Checkpoints missing provenance: {uncovered}")
    if not groups:
        raise ValueError("No checkpoints found")
    groups.sort(key=lambda group: (group["checkpoint"], group["seed"], group["method"], group["mode"]))
    (output / "task_split_manifest.json").write_bytes(split_raw)
    counts = {method: sum(group["method"] == method for group in groups) for method in ("gepa", "skillopt")}
    return {
        "created_at": shared.timestamp(), "source": str(source), "upstream": str(upstream),
        "upstream_launch_sha256": shared.digest(upstream_raw), "workers": workers, "base_url": base_url,
        "checkpoint_count": len(groups), "checkpoint_counts": counts, "total_tasks": len(groups) * 50,
        "manifest_sha256": shared.digest(split_raw), "max_tokens": shared.MAX_TOKENS,
        "output_length_additional_retries": shared.LENGTH_RETRIES, "output_length_max_attempts": 3,
        "transport_additional_retries": 0, "learning": False, "max_steps": 50, "max_actions": 100,
        "policy": "Original AppWorld benchmark policy; no teammate override",
        "prompt_application": "Exact exported full system prompt through frozen PREMiSE tool-calling runtime",
        "enable_tool_shortlisting": True, "max_selected_tools": 30, "temperature": 0.0,
        "stage_transition": "Wait for terminal AppWorld/BFCL results and predecessor controller shutdown",
        "runner_sha256": shared.digest(Path(__file__).read_bytes()),
        "shared_runner_sha256": shared.digest(Path(shared.__file__).read_bytes()),
        "aliases": aliases, "groups": groups,
    }


def predecessor_ready(upstream: Path, expected_hash: str) -> tuple[bool, str, dict]:
    if shared.digest((upstream / "launch_manifest.json").read_bytes()) != expected_hash:
        raise ValueError("Predecessor launch manifest changed")
    status_path = upstream / "status.json"
    status = shared.read_json(status_path) if status_path.exists() else {}
    for stage in ("appworld", "bfcl"):
        marker = upstream / f"{stage}_complete.json"
        if not marker.exists():
            return False, f"Waiting for {stage} to finish", status
        completion = shared.read_json(marker)
        if (completion.get("all_tasks_terminal"), completion.get("tasks"), completion.get("groups")) != (
            True, 6000, 120,
        ):
            raise ValueError(f"Invalid predecessor completion marker: {stage}")
    if status.get("phase") != "complete" or status.get("processed_tasks") != 12000:
        return False, "Waiting for predecessor controller shutdown", status
    with (upstream / "controller.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False, "Waiting for predecessor worker processes to exit", status
    return True, "Both predecessor stages finished", status


def priority_ready(upstream: Path, dependency: dict) -> tuple[bool, str, dict]:
    if shared.digest((upstream / "launch_manifest.json").read_bytes()) != dependency["launch_sha256"]:
        raise ValueError("Priority launch manifest changed")
    status = shared.read_json(upstream / "status.json")
    marker = upstream / "appworld_test_complete.json"
    if not marker.exists():
        return False, "Waiting for team CP130 priority evaluation", status
    completion = shared.read_json(marker)
    if (completion.get("all_tasks_terminal"), completion.get("processed_tasks"),
        completion.get("checkpoint_count")) != (True, dependency["total_tasks"], dependency["checkpoint_count"]):
        raise ValueError("Invalid priority completion marker")
    if status.get("phase") != "complete":
        return False, "Waiting for priority controller shutdown", status
    with (upstream / "controller.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False, "Waiting for priority controller to release its lock", status
    return True, "Priority evaluation finished", status


def worker_init(output: str, base_url: str) -> None:
    global _OUTPUT
    _OUTPUT = Path(output)
    shared.worker_init(output, base_url)


def task_config(job: dict, directory: Path, run_id: str):
    from exgentic.core.types import RunConfig

    benchmark = job.get("benchmark", "appworld")
    if benchmark not in ("appworld", "bfcl"):
        raise ValueError(f"Unsupported exported-prompt benchmark: {benchmark}")
    max_tokens = job.get("max_tokens", shared.MAX_TOKENS)
    num_retries = job.get("model_num_retries", 0)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("Output limit must be a positive integer")
    if isinstance(num_retries, bool) or not isinstance(num_retries, int) or num_retries < 0:
        raise ValueError("Request retries must be a nonnegative integer")
    path = Path(job["checkpoint_path"])
    if job["group"] not in _PROMPTS:
        raw = path.read_bytes()
        if shared.digest(raw) != job["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed")
        prompt = exported_prompt(job["method"], raw)
        if shared.digest(prompt.encode()) != job["prompt_sha256"]:
            raise ValueError("Frozen prompt changed")
        _PROMPTS[job["group"]] = prompt
    subset = job.get("subset", "test_challenge" if benchmark == "appworld" else "multi_turn_base")
    benchmark_kwargs = {"subset": subset, "use_cache": False}
    if benchmark == "appworld":
        benchmark_kwargs["env_kwargs"] = {"experiment_name": run_id}
    return RunConfig(
        benchmark=benchmark, agent="premise", model=job["model"], task_ids=[job["task_id"]],
        output_dir=str(directory), run_id=run_id, max_workers=1, max_steps=50, max_actions=100,
        benchmark_kwargs=benchmark_kwargs,
        agent_kwargs={
            "model": job["model"], "runner": "direct", "enable_thinking": False,
            "playbook": "", "system_prompt_override": _PROMPTS[job["group"]],
            "enable_tool_shortlisting": benchmark == "appworld", "max_selected_tools": 30,
            "model_settings": {"temperature": job["temperature"], "max_tokens": max_tokens,
                               "timeout": 600, "num_retries": num_retries,
                               "retry_after": job.get("transport_retry_after", 0.5),
                               "retry_strategy": "exponential_backoff_retry"},
        },
    )


def evaluate_task(job: dict) -> dict:
    from exgentic.interfaces.lib.api import execute

    key, index = job["group"], job["index"]
    run_id = f"team_{job['method']}_{job['mode']}_s{job['seed']}_cp{job['checkpoint']:03d}_{index:03d}"
    if job.get("evaluation_label"):
        run_id += f"_{job['split']}_{job['evaluation_label']}"
    directory = _OUTPUT / "sessions" / key
    audit_path = _OUTPUT / "request_audits" / key / f"{index:03d}.json"
    audit = {"task_id": job["task_id"], "group": key, "worker_pid": os.getpid(),
             "audit_path": str(audit_path), "model_calls": 0, "output_length_errors": 0,
             "output_length_retries": 0, "length_exhausted": False,
             "max_tokens": job.get("max_tokens", shared.MAX_TOKENS),
             "transport_additional_retries": job.get("transport_additional_retries", 0),
             "transport_retry_after": job.get("transport_retry_after", 0.5), "transport_retries": 0}
    shared._AUDIT = audit
    shared.write_json(_OUTPUT / "active_workers" / f"{os.getpid()}.json", {
        "phase": "running", "started_at": shared.timestamp(), **job, "run_id": run_id,
    })
    row = {**job, "worker_pid": os.getpid(), "started_at": shared.timestamp(), "run_id": run_id}
    try:
        result = execute(config=task_config(job, directory, run_id))
        if len(result.session_results) != 1:
            raise ValueError("Expected exactly one session result")
        session = result.session_results[0]
        session_dir = directory / run_id / "sessions" / session.session_id
        row.update(session_id=session.session_id, session_dir=str(session_dir), session_status=session.status.value)
        if audit["length_exhausted"]:
            raise shared.OutputLengthRetriesExhaustedError("Output length failed after two additional retries")
        if session.status.value in ("error", "cancelled") or session.score is None:
            raise RuntimeError(str(session.details))
        if job.get("benchmark", "appworld") == "appworld":
            scores = shared.read_json(session_dir / "benchmark/scores.json")
            if not isinstance(scores.get("success"), bool):
                raise ValueError("Missing official AppWorld success flag")
            hard = float(scores["success"])
        else:
            hard = float(session.score)
            if hard not in (0.0, 1.0):
                raise ValueError("Expected the binary official BFCL task score")
        row.update(status="completed", score=hard, partial_score=float(session.score),
                   steps=session.steps)
    except Exception as exc:
        row.update(status="error", score=0.0, partial_score=0.0, error_type=type(exc).__name__,
                   error=str(exc), output_length_failure=audit["length_exhausted"])
    row.update(completed_at=shared.timestamp(), model_calls=audit["model_calls"],
               output_length_retries=audit["output_length_retries"], transport_retries=audit["transport_retries"])
    shared.write_json(audit_path, audit)
    shared.write_json(_OUTPUT / "results" / key / f"{index:03d}.json", row)
    shared.write_json(_OUTPUT / "active_workers" / f"{os.getpid()}.json", {
        "phase": "idle", "completed_at": shared.timestamp(), "last_group": key,
    })
    return row


def save_aggregates(output: Path, groups: list[dict], records: dict[str, list]) -> None:
    aggregates = []
    identities = sorted({(group["method"], group["mode"], group["checkpoint"]) for group in groups})
    for method, mode, point in identities:
        selected = [group for group in groups if (group["method"], group["mode"], group["checkpoint"]) == (
            method, mode, point,
        )]
        summaries = [shared.summarize(group, records[group["group"]]) for group in selected]
        complete = all(row["complete"] for row in summaries)
        values = [row["score_with_errors_zero"] for row in summaries]
        errors = sum(row["execution_errors"] for row in summaries)
        mean = statistics.mean(values) if complete else None
        std = statistics.stdev(values) if complete and len(values) > 1 else None
        aggregates.append({
            "benchmark": "appworld", "method": method, "mode": mode, "checkpoint": point, "split": "test",
            "complete": complete, "execution_errors": errors, "seed_count": len(selected),
            "seeds": [group["seed"] for group in selected], "std_ddof": 1,
            "score_by_seed": {str(group["seed"]): summary["score_with_errors_zero"]
                              for group, summary in zip(selected, summaries, strict=True)} if complete else {},
            "mean_with_errors_zero": mean, "std_with_errors_zero": std,
            "mean": mean if not errors else None, "std": std if not errors else None,
        })
    shared.write_json(output / "scores.json", {"updated_at": shared.timestamp(), "aggregates": aggregates})


def run(output: Path, launch: dict, *, wait_for_predecessor: bool = True) -> int:
    if wait_for_predecessor:
        dependency = launch.get("priority_dependency")
        upstream = Path(dependency["output"] if dependency else launch["upstream"])
        expected_hash = dependency["launch_sha256"] if dependency else launch["upstream_launch_sha256"]
        while True:
            if dependency:
                ready, reason, predecessor_status = priority_ready(upstream, dependency)
            else:
                ready, reason, predecessor_status = predecessor_ready(upstream, expected_hash)
            if ready:
                break
            shared.write_json(output / "status.json", {
                "controller_pid": os.getpid(), "phase": "waiting", "reason": reason,
                "checkpoint_count": launch["checkpoint_count"], "total_tasks": launch["total_tasks"],
                "workers": launch["workers"], "active_tasks": 0, "processed_tasks": 0,
                "upstream": str(upstream), "upstream_status": predecessor_status, "updated_at": shared.timestamp(),
            })
            time.sleep(10)
        shared.write_json(output / "dependency_complete.json", {
            "verified_at": shared.timestamp(), "upstream_status": predecessor_status,
            "upstream_launch_sha256": expected_hash,
        })
    print(f"Starting {launch['checkpoint_count']} checkpoints; wait_for_predecessor={wait_for_predecessor}", flush=True)
    groups = launch["groups"]
    records = {group["group"]: [None] * 50 for group in groups}
    by_key = {group["group"]: group for group in groups}
    for group in groups:
        for index, task_id in enumerate(group["task_ids"]):
            path = output / "results" / group["group"] / f"{index:03d}.json"
            if path.exists():
                row = shared.read_json(path)
                if (row["task_id"], row["checkpoint_sha256"], row["prompt_sha256"]) != (
                    task_id, group["checkpoint_sha256"], group["prompt_sha256"],
                ) or row["status"] not in ("completed", "error"):
                    raise ValueError("Existing result differs from frozen evaluation inputs")
                records[group["group"]][index] = row
    queue = shared.queue_jobs(groups, records)
    pending = {}
    workers = launch["workers"]
    print(f"START appworld test: groups={len(groups)} tasks={len(queue)} concurrency={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init, initargs=(str(output), launch["base_url"])) as pool:
        while queue or pending:
            while queue and len(pending) < workers:
                job = queue.popleft()
                pending[pool.submit(evaluate_task, job)] = job
            observed = [row for rows in records.values() for row in rows if row is not None]
            shared.write_json(output / "status.json", {
                "controller_pid": os.getpid(), "phase": "running", "stage": "appworld_test",
                "workers": workers, "active_tasks": len(pending), "queued_tasks": len(queue),
                "processed_tasks": len(observed), "total_tasks": launch["total_tasks"],
                "checkpoint_count": len(groups), "execution_errors": sum(row["status"] == "error" for row in observed),
                "updated_at": shared.timestamp(),
            })
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                row = future.result()
                records[job["group"]][job["index"]] = row
                summary = shared.summarize(by_key[job["group"]], records[job["group"]])
                shared.write_json(output / "summaries" / job["group"] / "summary.json", summary)
                if summary["complete"]:
                    save_aggregates(output, groups, records)
                print(f"DONE {job['group']} task={job['index']:02d} status={row['status']}", flush=True)
    save_aggregates(output, groups, records)
    errors = sum(row["status"] == "error" for rows in records.values() for row in rows)
    completion = {"controller_pid": os.getpid(), "phase": "complete", "all_tasks_terminal": True,
                  "processed_tasks": launch["total_tasks"], "checkpoint_count": len(groups),
                  "execution_errors": errors, "updated_at": shared.timestamp()}
    shared.write_json(output / "appworld_test_complete.json", completion)
    shared.write_json(output / "status.json", completion)
    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output, source, upstream = args.output.resolve(), args.source.resolve(), args.after.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "automatic_resume_disabled.json").exists() and not args.dry_run:
            state = shared.read_json(output / "status.json") if (output / "status.json").exists() else {}
            state.update(phase="paused", active_tasks=0, automatic_resume_disabled=True,
                         last_resume_skipped_at=shared.timestamp(), skipped_controller_pid=os.getpid())
            shared.write_json(output / "status.json", state)
            print("Team evaluation remains paused by the user's request; no workers started.", flush=True)
            return 0
        manifest = output / "launch_manifest.json"
        if manifest.exists():
            launch = shared.read_json(manifest)
            if (launch["source"], launch["upstream"], launch["workers"], launch["base_url"]) != (
                str(source), str(upstream), args.workers, args.base_url,
            ):
                raise ValueError("Resume settings differ from the scheduled evaluation")
        else:
            launch = prepare(source, output, upstream, args.workers, args.base_url)
            shared.write_json(manifest, launch)
        if args.dry_run:
            print(json.dumps({key: value for key, value in launch.items() if key != "groups"}, indent=2))
            return 0
        return run(output, launch)


if __name__ == "__main__":
    raise SystemExit(main())
