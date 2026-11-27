# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Evaluate frozen isolated checkpoints on train/test through one bounded worker queue."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
import random
import statistics
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
SOURCE = ROOT / "outputs/isolated_baselines_20260930"
APPWORLD_COLLECTION = ROOT / "outputs/appworld_isolated_checkpoints_cp005_050_ace_premise_20261008"
SPLITS = ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
METHODS = ("ace", "premise")
SEEDS = (42, 43, 44)
POINTS = tuple(range(5, 51, 5))
MAX_TOKENS = 2048
LENGTH_RETRIES = 2
_AUDIT: dict = {}
_FROZEN: dict = {}
_OUTPUT: Path | None = None


class OutputLengthRetriesExhaustedError(RuntimeError):
    """One unchanged model request reached its output limit three times."""


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def is_length_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return isinstance(exc, OutputLengthRetriesExhaustedError) or "finish_reason=length" in text or (
        "truncated" in text and ("output" in text or "completion" in text)
    )


def completion_with_length_retries(call, kwargs: dict, audit: dict, *, actor_budget=None):
    """Keep separate budgets for request errors and two output-length retries."""
    if audit.get("length_exhausted"):
        raise OutputLengthRetriesExhaustedError("Output length retry budget already exhausted for this task")
    max_tokens = audit.get("max_tokens", MAX_TOKENS)
    if actor_budget is not None:
        max_tokens = min(max_tokens or MAX_TOKENS, kwargs.get("max_tokens") or MAX_TOKENS)
        audit["max_tokens"] = min(audit.get("max_tokens") or MAX_TOKENS, actor_budget.limits["output_tokens"])
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("Output limit must be a positive integer")
    transport_limit = audit.get("transport_additional_retries", 0)
    if isinstance(transport_limit, bool) or not isinstance(transport_limit, int) or transport_limit < 0:
        raise ValueError("Request retry budget must be a nonnegative integer")
    request = {**kwargs, "max_tokens": max_tokens, "max_retries": 0, "num_retries": 0}
    length_failures = request_failures = 0
    for _ in range(LENGTH_RETRIES + transport_limit + 1):
        audit["model_calls"] = audit.get("model_calls", 0) + 1
        try:
            response = call(**request)
            if any(choice.get("finish_reason") == "length" for choice in response["choices"]):
                if actor_budget is not None:
                    audit["output_length_errors"] = audit.get("output_length_errors", 0) + 1
                    return response  # The shared guard records usage and fails the trajectory.
                raise OutputLengthRetriesExhaustedError("Completion truncated (finish_reason=length)")
            return response
        except Exception as exc:
            if actor_budget is not None and is_length_error(exc):
                actor_budget.fail("output_truncated", actor_budget.limits["output_tokens"], 0)
            if not is_length_error(exc):
                if request_failures >= transport_limit:
                    raise
                from exgentic.agents.litellm_tool_calling.instance import NonRetryableCompletionError
                from exgentic.utils.context_errors import is_context_length_error

                if isinstance(exc, NonRetryableCompletionError) or is_context_length_error(exc):
                    raise
                delay = audit.get("transport_retry_after", 0.5) * 2 ** request_failures
                request_failures += 1
                audit["transport_retries"] = audit.get("transport_retries", 0) + 1
                audit.setdefault("transport_events", []).append({
                    "at": timestamp(), "retry": request_failures, "error_type": type(exc).__name__,
                    "delay_seconds": delay,
                })
                if audit.get("audit_path"):
                    write_json(Path(audit["audit_path"]), audit)
                if delay:
                    time.sleep(delay)
                continue
            length_failures += 1
            retry = length_failures <= LENGTH_RETRIES
            audit["output_length_errors"] = audit.get("output_length_errors", 0) + 1
            audit["output_length_retries"] = audit.get("output_length_retries", 0) + int(retry)
            event = {"at": timestamp(), "attempt": length_failures, "max_tokens": max_tokens, "retry": retry}
            audit.setdefault("length_events", []).append(event)
            if audit.get("audit_path"):
                write_json(Path(audit["audit_path"]), audit)
            if not retry:
                audit["length_exhausted"] = True
                raise OutputLengthRetriesExhaustedError(
                    f"Output length failure after 3 attempts at max_tokens={max_tokens} (2 additional retries)"
                ) from exc
    raise AssertionError("Unreachable completion retry state")


def actor_request_completion(instance, request):
    import litellm

    from exgentic.core.experiment_limits import actor_completion

    budget = getattr(instance, "_actor_turn_budget", None)
    return actor_completion(instance, request, completion_fn=lambda **kwargs: completion_with_length_retries(
        litellm.completion, kwargs, _AUDIT, actor_budget=budget,
    ))


def ace_completion(instance, **kwargs):
    from exgentic.core.context import get_context

    request = instance._model_settings.model_dump(
        exclude_none=True, exclude={"num_retries", "retry_after", "retry_strategy"},
    )
    request.update(kwargs)
    body = dict(request.get("extra_body") or {})
    body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), "enable_thinking": False}
    request["extra_body"] = body
    request.setdefault("metadata", {})["context"] = get_context()
    if request.get("tools") is None:
        request.pop("tools", None)
    return actor_request_completion(instance, request)


def premise_completion(instance, request):
    response = actor_request_completion(instance, request)
    instance._raise_if_invalid_completion(response)
    return response


def worker_init(output: str, base_url: str) -> None:
    global _OUTPUT
    _OUTPUT = Path(output)
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["EXGENTIC_LLM_TRACE_FORMAT"] = "dedup_v1"
    os.environ.pop("EXGENTIC_PORT_BLOCK_START", None)
    from exgentic.agents.ace.ace_instance import ACEAgentInstance
    from exgentic.agents.premise.instance import PremiseAgentInstance
    from exgentic.agents.premise.llm import configure_llm_slots

    ACEAgentInstance._completion = ace_completion
    PremiseAgentInstance._completion_with_retries = premise_completion
    configure_llm_slots(1)


def group_key(benchmark: str, method: str, seed: int, point: int, split: str) -> str:
    return f"{benchmark}/{method}/s{seed}/cp{point:03d}/{split}"


def evaluate_task(job: dict) -> dict:
    global _AUDIT
    from exgentic.agents.ace.playbook_store import PlaybookStore
    from exgentic.core.types import RunConfig
    from exgentic.interfaces.lib.api import execute

    key = job["group"]
    index = job["index"]
    directory = _OUTPUT / "sessions" / key
    run_id = f"queue_{job['benchmark']}_{job['method']}_s{job['seed']}_cp{job['checkpoint']:03d}"
    run_id += f"_{job['split']}_{index:03d}"
    audit_path = _OUTPUT / "request_audits" / key / f"{index:03d}.json"
    _AUDIT = {"task_id": job["task_id"], "group": key, "worker_pid": os.getpid(),
              "audit_path": str(audit_path), "model_calls": 0, "output_length_errors": 0,
              "output_length_retries": 0, "length_exhausted": False}
    write_json(_OUTPUT / "active_workers" / f"{os.getpid()}.json", {
        "phase": "running", "started_at": timestamp(), **job, "run_id": run_id,
    })
    started = timestamp()
    row = {**job, "worker_pid": os.getpid(), "started_at": started, "run_id": run_id}
    try:
        frozen_key = (job["benchmark"], job["method"], job["seed"], job["checkpoint"])
        if frozen_key not in _FROZEN:
            raw = Path(job["checkpoint_path"]).read_bytes()
            if digest(raw) != job["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint changed")
            snapshot = json.loads(raw)
            _FROZEN[frozen_key] = (
                snapshot["playbook"] if job["method"] == "ace"
                else snapshot["states"][job["benchmark"]]["beam"][0]["playbook"]
            )
        playbook = _FROZEN[frozen_key]
        PlaybookStore.reset_all()
        kwargs = {
            "model": job["model"], "runner": "direct", "enable_thinking": False,
            "enable_tool_shortlisting": job["benchmark"] == "appworld", "max_selected_tools": 30,
            "model_settings": {"temperature": job["temperature"], "max_tokens": MAX_TOKENS,
                               "timeout": 600, "num_retries": 0},
        }
        if job["method"] == "ace":
            kwargs.update(initial_playbook=playbook, shuffle_mode="isolated", benchmark_id=job["benchmark"],
                          evaluation_mode=True, training_time=False, curator_model=job["model"])
        else:
            kwargs["playbook"] = playbook
        benchmark_kwargs = {"subset": job["subset"], "use_cache": False}
        if job["benchmark"] == "appworld":
            benchmark_kwargs["env_kwargs"] = {"experiment_name": run_id}
        config = RunConfig(
            benchmark=job["benchmark"], agent=job["method"], model=job["model"],
            task_ids=[job["task_id"]], output_dir=str(directory), run_id=run_id,
            agent_kwargs=kwargs, benchmark_kwargs=benchmark_kwargs, max_workers=1,
            max_steps=50, max_actions=100,
        )
        result = execute(config=config)
        if len(result.session_results) != 1:
            raise ValueError("Expected exactly one session result")
        session = result.session_results[0]
        session_dir = directory / run_id / "sessions" / session.session_id
        row.update(session_id=session.session_id, session_dir=str(session_dir), session_status=session.status.value)
        if _AUDIT["length_exhausted"]:
            raise OutputLengthRetriesExhaustedError("Output length failed after two additional retries")
        if session.status.value in ("error", "cancelled") or session.score is None:
            raise RuntimeError(str(session.details))
        if job["method"] == "ace":
            for store in PlaybookStore.list_stores().values():
                if store.playbook != playbook or store.session_count != 0:
                    raise RuntimeError("Frozen ACE playbook changed during evaluation")
        if job["benchmark"] == "appworld":
            scores = read_json(session_dir / "benchmark/scores.json")
            if not isinstance(scores.get("success"), bool):
                raise ValueError("Missing official AppWorld success flag")
            score = float(scores["success"])
        else:
            score = float(session.score)
        row.update(status="completed", score=score, partial_score=float(session.score), steps=session.steps)
    except Exception as exc:
        row.update(status="error", score=0.0, partial_score=0.0,
                   error_type=type(exc).__name__, error=str(exc), output_length_failure=_AUDIT["length_exhausted"])
    row.update(completed_at=timestamp(), model_calls=_AUDIT["model_calls"],
               output_length_retries=_AUDIT["output_length_retries"])
    write_json(audit_path, _AUDIT)
    write_json(_OUTPUT / "results" / key / f"{index:03d}.json", row)
    write_json(_OUTPUT / "active_workers" / f"{os.getpid()}.json", {
        "phase": "idle", "completed_at": timestamp(), "last_group": key,
    })
    return row


def collect_bfcl(output: Path) -> list[dict]:
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
    from exgentic.agents.ace.playbook_utils import update_bullet_counts
    from exgentic.agents.premise.selection import sample_past
    from scripts.ace.checkpoint_recovery import completed_prefix

    checkpoints = []
    for seed in SEEDS:
        train = SOURCE / "bfcl/ace" / f"s{seed}/train"
        config = read_json(train / "experiment_config.json")
        prefix = completed_prefix(train, config["task_order"])
        if len(prefix) != 50:
            raise ValueError("Incomplete BFCL ACE training prefix")
        for point in POINTS:
            entry = prefix[point - 1]
            if entry["checkpoint"]["session_count"] != point:
                raise ValueError("ACE checkpoint task ordinal differs")
            raw = Path(entry["checkpoint_path"]).read_bytes()
            path = output / "frozen/bfcl/ace" / f"s{seed}" / f"checkpoint_{point:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            checkpoints.append({"benchmark": "bfcl", "method": "ace", "seed": seed, "checkpoint": point,
                                "checkpoint_path": str(path), "checkpoint_sha256": digest(raw),
                                "model": config["model"], "source": entry["checkpoint_path"]})
        run = SOURCE / "bfcl/premise" / f"s{seed}/run"
        saved = read_json(run / "checkpoints/checkpoint_050.json")
        if saved["config"]["order"] != config["task_order"]:
            raise ValueError("BFCL method task orders differ")
        initial = {"playbook": DEFAULT_PLAYBOOK, "next_global_id": 1}
        state = {"beam": [copy.deepcopy(initial), copy.deepcopy(initial)], "memory": []}
        rng = random.Random(seed)
        snapshots = {}
        for index, (benchmark, task_id) in enumerate(saved["config"]["order"], 1):
            artifact = read_json(run / "iterations" / f"iter_{index:04d}.json")
            if (artifact["iteration"], artifact["benchmark"], artifact["task_id"]) != (index, benchmark, task_id):
                raise ValueError("BFCL iteration differs from the training order")
            for rank, parent in enumerate(artifact["parents"]):
                if parent:
                    parent_config = read_json(Path(parent["session_dir"]) / "config.json")
                    if parent_config["agent_kwargs"]["playbook"] != state["beam"][rank]["playbook"]:
                        raise ValueError("Reconstructed BFCL beam differs from the next parent")
            past = sample_past(state["memory"], saved["config"]["past_count"], rng)
            sampled_tasks = [{"benchmark": item["benchmark"], "task_id": item["task_id"]} for item in past]
            if sampled_tasks != artifact["past_sample"]:
                raise ValueError("BFCL replay samples differ from recorded samples")
            candidates = {item["id"]: item for item in artifact["candidates"]}
            selected = [candidates[name] for name in artifact["selection"]["selected"]]
            state["beam"] = [copy.deepcopy(item["playbook"]) for item in selected]
            for rank, candidate in enumerate(selected):
                tags = [{"id": item.get("entry_id", item.get("id")), "tag": item.get("tag", "neutral")}
                        for item in candidate["reflection"].get("entry_assessments", [])]
                state["beam"][rank]["playbook"] = update_bullet_counts(state["beam"][rank]["playbook"], tags)
            primary = selected[0]
            for sampled, result in zip(past, primary.get("past_results", []), strict=False):
                entry = next(item for item in state["memory"] if item["memory_id"] == sampled["memory_id"])
                entry.update(latest_selected_replay_official_score=result["official_score"],
                             latest_selected_replay_candidate_id=primary["id"], latest_selected_replay_iteration=index)
            if primary["applied"]:
                state["memory"].append({
                    "memory_id": f"{index}|{benchmark}|{task_id}", "benchmark": benchmark, "task_id": task_id,
                    "trace_artifact": f"iterations/iter_{index:04d}.json", "parent_index": primary["parent_rank"] - 1,
                    "reflection": primary["reflection"], "accepted_candidate_id": primary["id"],
                    "selected_official_score": primary["current_official_score"],
                    "latest_selected_replay_official_score": primary["current_official_score"],
                })
            if index in POINTS:
                snapshots[index] = {"config": saved["config"], "cursor": index,
                                    "states": {"bfcl": copy.deepcopy(state)}, "rng_state": repr(rng.getstate())}
        if snapshots[50] != saved:
            raise ValueError("Reconstructed full BFCL checkpoint 50 differs from the original")
        for point, snapshot in snapshots.items():
            path = output / "frozen/bfcl/premise" / f"s{seed}" / f"checkpoint_{point:03d}.json"
            write_json(path, snapshot)
            checkpoints.append({"benchmark": "bfcl", "method": "premise", "seed": seed, "checkpoint": point,
                                "checkpoint_path": str(path), "checkpoint_sha256": digest(path.read_bytes()),
                                "model": saved["config"]["model"], "reconstructed": point != 50,
                                "source_run": str(run), "full_checkpoint_050_verified": True})
    return checkpoints


def prepare(output: Path, workers: int, base_url: str) -> dict:
    from exgentic.utils.task_splits import validate_task_split

    split_bytes = SPLITS.read_bytes()
    splits = json.loads(split_bytes)
    collection = read_json(APPWORLD_COLLECTION / "manifest.json")
    if collection["manifest_sha256"] != digest(split_bytes) or collection["checkpoint_count"] != 60:
        raise ValueError("AppWorld collection differs from frozen task splits")
    checkpoints = []
    for entry in collection["checkpoints"]:
        raw = (APPWORLD_COLLECTION / entry["path"]).read_bytes()
        if digest(raw) != entry["sha256"]:
            raise ValueError("AppWorld checkpoint changed after collection")
        path = output / "frozen/appworld" / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        cfg = read_json(APPWORLD_COLLECTION / "source_configs" / f"{entry['method']}_s{entry['seed']}.json")
        checkpoints.append({"benchmark": "appworld", "method": entry["method"], "seed": entry["seed"],
                            "checkpoint": entry["checkpoint"], "checkpoint_path": str(path),
                            "checkpoint_sha256": digest(raw), "model": cfg["model"],
                            "source": str(APPWORLD_COLLECTION / entry["path"])})
    checkpoints.extend(collect_bfcl(output))
    groups = []
    for checkpoint in sorted(checkpoints, key=lambda row: (
        row["benchmark"], row["checkpoint"], row["seed"], row["method"],
    )):
        benchmark, method, seed, point = (checkpoint[key] for key in ("benchmark", "method", "seed", "checkpoint"))
        split = validate_task_split(splits[f"{benchmark}_{seed}"], benchmark)
        for partition in ("train", "test"):
            ids = split[partition]
            if len(ids) != 50:
                raise ValueError("Expected 50 tasks in every train and test partition")
            groups.append({**checkpoint, "split": partition, "task_ids": ids,
                           "group": group_key(benchmark, method, seed, point, partition),
                           "temperature": 1.0 if method == "ace" else 0.0,
                           "subset": "test_challenge" if benchmark == "appworld" else "multi_turn_base"})
    if len(groups) != 240 or len({group["group"] for group in groups}) != 240:
        raise ValueError("Expected exactly 120 groups per benchmark")
    return {
        "created_at": timestamp(), "stages": ["appworld", "bfcl"], "workers": workers,
        "groups_per_stage": 120, "tasks_per_stage": 6000, "total_groups": 240, "total_tasks": 12000,
        "max_tokens": MAX_TOKENS, "output_length_additional_retries": LENGTH_RETRIES,
        "output_length_max_attempts": 3, "retry_scope": "Unchanged model request; task fails on exhaustion",
        "transport_additional_retries": 0, "base_url": base_url, "learning": False, "premise_beam_rank": 1,
        "policy": "Original benchmark policy; no teammate override", "checkpoint_source": str(APPWORLD_COLLECTION),
        "manifest_source": str(SPLITS), "manifest_sha256": digest(split_bytes),
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "worker_architecture": "One shared bounded queue and 100 reusable isolated worker processes",
        "stage_transition": "Start BFCL after every AppWorld task has a terminal completed/error result",
        "groups": groups,
    }


def summarize(group: dict, rows: list[dict | None]) -> dict:
    observed = [row for row in rows if row is not None]
    errors = sum(row["status"] == "error" for row in observed)
    total = len(rows)
    complete = len(observed) == total
    score = sum(row["score"] for row in observed) / total
    return {"identity": group, "results": rows, "processed": len(observed), "total": total,
            "complete": complete, "execution_errors": errors, "score_with_errors_zero": score,
            "score": score if complete and not errors else None,
            "partial_score_with_errors_zero": sum(row["partial_score"] for row in observed) / total,
            "updated_at": timestamp()}


def queue_jobs(groups: list[dict], records: dict[str, list]) -> deque:
    jobs = deque()
    for index in range(50):
        for group in groups:
            if records[group["group"]][index] is None:
                task = {key: value for key, value in group.items() if key != "task_ids"}
                jobs.append({**task, "index": index, "task_id": group["task_ids"][index]})
    return jobs


def save_aggregates(output: Path, groups: list[dict], records: dict[str, list]) -> None:
    aggregates = []
    for benchmark in ("appworld", "bfcl"):
        for method in METHODS:
            for point in POINTS:
                for partition in ("train", "test"):
                    selected = [group for group in groups if (
                        group["benchmark"], group["method"], group["checkpoint"], group["split"]
                    ) == (benchmark, method, point, partition)]
                    summaries = [summarize(group, records[group["group"]]) for group in selected]
                    complete = len(summaries) == 3 and all(row["complete"] for row in summaries)
                    values = [row["score_with_errors_zero"] for row in summaries]
                    errors = sum(row["execution_errors"] for row in summaries)
                    by_seed = {str(group["seed"]): summary["score_with_errors_zero"]
                               for group, summary in zip(selected, summaries, strict=True)} if complete else {}
                    aggregates.append({"benchmark": benchmark, "method": method, "checkpoint": point,
                        "split": partition, "complete": complete, "execution_errors": errors,
                        "score_by_seed": by_seed,
                        "mean_with_errors_zero": statistics.mean(values) if complete else None,
                        "std_with_errors_zero": statistics.stdev(values) if complete else None,
                        "mean": statistics.mean(values) if complete and not errors else None,
                        "std": statistics.stdev(values) if complete and not errors else None, "std_ddof": 1})
    write_json(output / "scores.json", {"updated_at": timestamp(), "aggregates": aggregates})


def run(output: Path, workers: int, base_url: str) -> int:
    launch_path = output / "launch_manifest.json"
    if launch_path.exists():
        launch = read_json(launch_path)
        if launch["workers"] != workers or launch["base_url"] != base_url:
            raise ValueError("Resume settings differ from the frozen evaluation")
    else:
        launch = prepare(output, workers, base_url)
        write_json(launch_path, launch)
        (output / "task_split_manifest.json").write_bytes(SPLITS.read_bytes())
    groups = launch["groups"]
    records = {group["group"]: [None] * 50 for group in groups}
    for group in groups:
        for index, task_id in enumerate(group["task_ids"]):
            path = output / "results" / group["group"] / f"{index:03d}.json"
            if path.exists():
                row = read_json(path)
                if row["task_id"] != task_id or row["checkpoint_sha256"] != group["checkpoint_sha256"]:
                    raise ValueError("Existing task result differs from frozen evaluation inputs")
                records[group["group"]][index] = row
        write_json(output / "summaries" / group["group"] / "summary.json", summarize(group, records[group["group"]]))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init, initargs=(str(output), base_url)) as pool:
        for stage in launch["stages"]:
            stage_groups = [group for group in groups if group["benchmark"] == stage]
            queue = queue_jobs(stage_groups, records)
            pending = {}
            print(f"START {stage}: groups=120 tasks={len(queue)} concurrency={workers}", flush=True)

            def save_status(stage=stage, stage_groups=stage_groups, pending=pending, queue=queue):
                done = [row for group in stage_groups for row in records[group["group"]] if row is not None]
                write_json(output / "status.json", {
                    "controller_pid": os.getpid(), "phase": "running", "stage": stage,
                    "workers": workers, "active_tasks": len(pending), "queued_tasks": len(queue),
                    "processed_tasks": len(done), "stage_total_tasks": 6000,
                    "execution_errors": sum(row["status"] == "error" for row in done),
                    "remaining_stages": launch["stages"][launch["stages"].index(stage) + 1:],
                    "updated_at": timestamp(),
                })

            while queue or pending:
                while queue and len(pending) < workers:
                    job = queue.popleft()
                    pending[pool.submit(evaluate_task, job)] = job
                save_status()
                done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    row = future.result()
                    records[job["group"]][job["index"]] = row
                    group = next(group for group in stage_groups if group["group"] == job["group"])
                    summary = summarize(group, records[group["group"]])
                    write_json(output / "summaries" / group["group"] / "summary.json", summary)
                    print(f"DONE {job['group']} task={job['index']:02d} status={row['status']}", flush=True)
            save_aggregates(output, groups, records)
            errors = sum(row["status"] == "error" for group in stage_groups for row in records[group["group"]])
            write_json(output / f"{stage}_complete.json", {
                "completed_at": timestamp(), "tasks": 6000, "groups": 120, "execution_errors": errors,
                "all_tasks_terminal": True,
            })
            print(f"FINISH {stage}: 6000 tasks, errors={errors}", flush=True)
    errors = sum(row["status"] == "error" for rows in records.values() for row in rows)
    write_json(output / "status.json", {"controller_pid": os.getpid(), "phase": "complete",
               "processed_tasks": 12000, "execution_errors": errors, "updated_at": timestamp()})
    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "automatic_resume_disabled.json").exists() and not args.dry_run:
            state = read_json(output / "status.json") if (output / "status.json").exists() else {}
            state.update(phase="paused", active_tasks=0, automatic_resume_disabled=True,
                         last_resume_skipped_at=timestamp(), skipped_controller_pid=os.getpid())
            write_json(output / "status.json", state)
            print("Isolate evaluation remains paused by the user's request; no workers started.", flush=True)
            return 0
        if args.dry_run:
            launch = prepare(output, args.workers, args.base_url)
            print(json.dumps({key: value for key, value in launch.items() if key != "groups"}, indent=2))
            return 0
        return run(output, args.workers, args.base_url)


if __name__ == "__main__":
    raise SystemExit(main())
