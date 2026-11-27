# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Generate single-parent PREMiSE proposals, then evaluate all BFCL pairs in one queue."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import multiprocessing
import os
import random
import statistics
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.utils import evaluate_isolated_checkpoint_queue as execution  # noqa: E402

CHECKPOINTS = ((42, 10), (43, 20), (44, 30), (42, 40), (43, 40))
SOURCE = ROOT / "outputs/isolated_checkpoint_train_test_queue100_20261008"
SPLITS = ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
_CONTEXTS: dict = {}
_OUTPUT: Path | None = None
_SETTINGS: dict = {}


def save_immutable(path: Path, value: dict) -> None:
    if path.exists():
        if execution.read_json(path) != value:
            raise ValueError(f"Frozen inputs differ: {path}")
    else:
        execution.write_json(path, value)


def checkpoint_parent(snapshot: dict, point: int) -> dict:
    """Accept isolated states or a sequential prefix containing only BFCL tasks."""
    config = snapshot["config"]
    if snapshot["cursor"] != point:
        raise ValueError("Checkpoint cursor differs from the requested point")
    if config["mode"] == "isolated":
        state = snapshot["states"]["bfcl"]
    elif config["mode"] == "sequential":
        prefix = config["order"][:point + 1]
        if len(prefix) != point + 1 or any(benchmark != "bfcl" for benchmark, _ in prefix):
            raise ValueError("Sequential source must contain only BFCL before and at the proposal task")
        state = snapshot["states"]["global"]
        if any(item["benchmark"] != "bfcl" for item in state["memory"]):
            raise ValueError("Sequential source memory contains another benchmark")
    else:
        raise ValueError("Expected an isolated checkpoint or a sequential BFCL-only prefix")
    return state["beam"][0]


def prepare(output: Path, source: Path, workers: int, base_url: str, max_steps: int,
            checkpoints: tuple[tuple[int, int], ...] = CHECKPOINTS,
            reasoning_effort: str | None = None, model_override: str | None = None) -> dict:
    """Validate provenance before freezing samples or issuing any model requests."""
    from exgentic.agents.premise.llm import public_tool_contract
    from exgentic.utils.task_splits import validate_task_split

    source_launch = execution.read_json(source / "launch_manifest.json")
    split_bytes = SPLITS.read_bytes()
    splits = json.loads(split_bytes)
    if source_launch["manifest_sha256"] != execution.digest(split_bytes):
        raise ValueError("Source checkpoint collection uses a different task split")
    if not checkpoints or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Checkpoints must be nonempty and distinct")
    points = []
    for seed, point in checkpoints:
        group = next(row for row in source_launch["groups"] if (
            row["benchmark"], row["method"], row["seed"], row["checkpoint"]
        ) == ("bfcl", "premise", seed, point))
        raw = Path(group["checkpoint_path"]).read_bytes()
        if execution.digest(raw) != group["checkpoint_sha256"]:
            raise ValueError("Source checkpoint hash differs")
        snapshot = json.loads(raw)
        parent = checkpoint_parent(snapshot, point)
        iteration_path = Path(group["source_run"]) / "iterations" / f"iter_{point + 1:04d}.json"
        iteration = execution.read_json(iteration_path)
        if (iteration["benchmark"], iteration["task_id"]) != tuple(snapshot["config"]["order"][point]):
            raise ValueError("Proposal task differs from the next training task")
        recorded = iteration.get("parents", [None])[0]
        trace = copy.deepcopy(recorded)
        if trace is not None:
            parent_session = Path(trace["session_dir"])
            parent_config = execution.read_json(parent_session / "config.json")
            if parent_config["agent_kwargs"]["playbook"] != parent["playbook"]:
                raise ValueError("Recorded proposal trajectory does not use the checkpoint parent")
            original_result = execution.read_json(parent_session / "results.json")
            public_session = execution.read_json(parent_session / "session.json")
            trace["public_tool_contract"] = public_tool_contract(public_session["actions"], trace["actions"])
        else:
            original_result = {"steps": None}
            trace = {"benchmark": "bfcl", "task_id": iteration["task_id"], "source_parent_trace_unavailable": True}
        rerun_proposal = recorded is None or original_result["steps"] > max_steps or (
            model_override is not None and model_override != snapshot["config"]["model"]
        )
        split = validate_task_split(splits[f"bfcl_{seed}"], "bfcl")
        if any(len(split[name]) != 50 for name in ("train", "val")):
            raise ValueError("This experiment expects 50 train and 50 validation tasks")
        seen = [task for benchmark, task in snapshot["config"]["order"][:point] if benchmark == "bfcl"]
        if len(seen) != point or not set(seen).issubset(split["train"]):
            raise ValueError("Training prefix does not match the frozen train split")
        key = f"s{seed}_cp{point:03d}"
        rng = random.Random(f"bfcl_alignment_20261011:{key}")
        past = rng.sample(seen, 10)
        validation = rng.sample(split["val"], 10)
        frozen = output / "frozen" / key
        save_immutable(frozen / "checkpoint.json", snapshot)
        trace_name = ("proposal_source_unavailable.json" if recorded is None else
                      "recorded_proposal_trace.json" if rerun_proposal else "proposal_trace.json")
        save_immutable(frozen / trace_name, trace)
        save_immutable(frozen / "parent.json", {"id": "parent", "playbook": parent})
        points.append({
            "key": key, "seed": seed, "checkpoint": point, "model": model_override or snapshot["config"]["model"],
            "parent_rank": 1, "source_checkpoint": str(group["checkpoint_path"]),
            "source_checkpoint_sha256": execution.digest(raw), "source_iteration": str(iteration_path),
            "source_iteration_sha256": execution.digest(iteration_path.read_bytes()),
            "proposal_task": trace["task_id"], "proposal_recorded_steps": original_result["steps"],
            "train_task_ids": split["train"], "seen_task_ids": seen,
            "past_task_ids": past, "validation_task_ids": validation,
            "validation_full_task_ids": validation + [task for task in split["val"] if task not in validation],
        })
        if rerun_proposal:
            points[-1]["proposal_trace_mode"] = "rerun_parent_with_requested_step_budget"
        if recorded is None:
            points[-1]["proposal_recorded_parent_available"] = False
        if snapshot["config"]["mode"] != "isolated":
            points[-1]["source_mode"] = snapshot["config"]["mode"]
        if model_override is not None:
            points[-1]["source_model"] = snapshot["config"]["model"]
    candidate_count = len(points) * 5
    plan = {
        "schema_version": 1, "benchmark": "bfcl", "method": "premise", "mode": "isolated",
        "source": str(source), "manifest_sha256": execution.digest(split_bytes),
        "workers": workers, "base_url": base_url, "max_steps": max_steps, "max_tokens": 8192,
        "temperature": 0.0, "request_timeout": 600, "transport_additional_retries": 5,
        "output_length_additional_retries": 2, "candidates_per_checkpoint": 5,
        "candidate_count": candidate_count, "candidate_evaluations": candidate_count * 100,
        "parent_evaluations": len(points) * 50,
        "total_evaluations": len(points) * 550, "generation_parent_count": 1,
        "generation": "Five independent PREMiSE reflections and ADD mutations of the same rank-1 parent",
        "past_score": "Reuse the shared train evaluations on ten randomly sampled previously seen tasks",
        "validation_score": "Evaluate all 50 validation tasks; report the shared random-ten score separately",
        "ground_truth": "All 50 train tasks; also report seen-prefix and overlap-excluded improvements",
        "sampling_seed": "bfcl_alignment_20261011:<seed_checkpoint>", "checkpoints": points,
    }
    if any(point.get("source_mode") == "sequential" for point in points):
        plan["source_context_scope"] = "Sequential BFCL-only prefix; no other benchmark has been encountered"
        plan["enable_thinking"] = False
    if reasoning_effort is not None:
        plan["reasoning_effort"] = reasoning_effort
    if model_override is not None:
        plan["model_override"] = model_override
        plan["model_context_transfer"] = any(point["source_model"] != point["model"] for point in points)
    save_immutable(output / "launch_manifest.json", plan)
    target = output / "task_split_manifest.json"
    if target.exists() and target.read_bytes() != split_bytes:
        raise ValueError("Saved task split differs")
    target.write_bytes(split_bytes)
    return plan


def materialize_proposal_traces(output: Path, plan: dict) -> None:
    """Replay over-budget source trajectories without conditioning checkpoint choice on their outcomes."""
    from exgentic.agents.premise.pipeline import trajectory_from_session

    jobs = []
    for point in plan["checkpoints"]:
        frozen = output / "frozen" / point["key"]
        if (frozen / "proposal_trace.json").exists():
            continue
        if point.get("proposal_trace_mode") != "rerun_parent_with_requested_step_budget":
            raise ValueError("Missing recorded proposal trajectory")
        parent_path = frozen / "parent.json"
        jobs.append({
            "key": f"{point['key']}/parent/proposal/000", "group": f"{point['key']}/parent/proposal",
            "index": 0, "checkpoint_key": point["key"], "seed": point["seed"],
            "checkpoint": point["checkpoint"], "context": "parent", "partition": "proposal",
            "task_id": point["proposal_task"], "context_path": str(parent_path),
            "context_sha256": execution.digest(parent_path.read_bytes()), "model": point["model"],
            "max_steps": plan["max_steps"],
        })
    if not jobs:
        return
    settings = {key: plan[key] for key in ("base_url", "max_tokens", "temperature", "request_timeout",
                                         "transport_additional_retries")}
    if "reasoning_effort" in plan:
        settings["reasoning_effort"] = plan["reasoning_effort"]
    rollouts = output / "proposal_rollouts"
    execution.write_json(output / "status.json", {
        "phase": "proposal_trajectories", "controller_pid": os.getpid(),
        "trajectory_count": len(jobs), "updated_at": execution.timestamp(),
    })
    with ProcessPoolExecutor(max_workers=min(25, len(jobs)), mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init, initargs=(str(rollouts), settings)) as pool:
        futures = {}
        for job in jobs:
            saved = rollouts / "results" / f"{job['key']}.json"
            if saved.exists():
                row = execution.read_json(saved)
                if any(row.get(key) != value for key, value in job.items()):
                    raise ValueError("Proposal rollout does not match its frozen checkpoint")
                futures[job["key"]] = row
            else:
                futures[job["key"]] = pool.submit(evaluate_pair, job)
        for job in jobs:
            pending = futures[job["key"]]
            row = pending if isinstance(pending, dict) else pending.result()
            model_error = (
                "gpt-oss" in row["model"] and row.get("session_id") is not None
                and any(marker in row.get("error", "") for marker in (
                    "LiteLLM completion missing assistant content",
                    "unexpected tokens remaining in message header",
                ))
            )
            if row["status"] != "completed" and not model_error:
                raise RuntimeError(f"Proposal parent rollout failed: {row}")
            run_id = f"alignment_{job['checkpoint_key']}_parent_proposal_000"
            trace = trajectory_from_session(
                rollouts / "sessions" / job["group"], run_id,
                SimpleNamespace(session_id=row["session_id"], score=row["score"], details={}), "bfcl",
            )
            if model_error:
                trace["agent_protocol_error"] = row["error"]
            save_immutable(output / "frozen" / job["checkpoint_key"] / "proposal_trace.json", trace)
            print(f"PROPOSAL_TRACE {job['checkpoint_key']} steps={row['steps']}", flush=True)


def generate_candidates(output: Path, plan: dict) -> None:
    from exgentic.agents.premise.llm import PremiseLLM
    from exgentic.agents.premise.pipeline import PremisePipeline

    materialize_proposal_traces(output, plan)
    candidate_count = len(plan["checkpoints"]) * 5
    models = {point["model"] for point in plan["checkpoints"]}
    if len(models) != 1:
        raise ValueError("All checkpoints must use the same model")
    proposer = PremisePipeline.__new__(PremisePipeline)
    proposer.llm = PremiseLLM(
        models.pop(), plan["base_url"], max_calls=25, timeout=plan["request_timeout"], num_retries=5,
        trace_dir=output / "llm_calls",
        reasoning_effort=plan.get("reasoning_effort"),
    )
    specs = []
    for point in plan["checkpoints"]:
        frozen = output / "frozen" / point["key"]
        parent = execution.read_json(frozen / "parent.json")["playbook"]
        trace = execution.read_json(frozen / "proposal_trace.json")
        for index in range(1, 6):
            path = frozen / f"candidate_{index:02d}.json"
            if path.exists():
                saved = execution.read_json(path)
                if not saved.get("valid") or saved["parent_rank"] != 1:
                    raise ValueError(f"Invalid saved candidate: {path}")
            else:
                specs.append((point, index, path, parent, trace))

    def one(spec):
        point, index, path, parent, trace = spec
        reflection_path = path.with_name(f"reflection_{index:02d}.json")
        if reflection_path.exists():
            reflection = execution.read_json(reflection_path)
        else:
            reflection = proposer._one_reflection(1, index, copy.deepcopy(trace), copy.deepcopy(parent))
            save_immutable(reflection_path, reflection)
        candidate = None
        for attempt in range(3):
            error_path = path.with_name(
                f"candidate_{index:02d}_error.json" if attempt == 0
                else f"candidate_{index:02d}_attempt_{attempt + 1:02d}_error.json"
            )
            if error_path.exists():
                candidate = execution.read_json(error_path)
            else:
                candidate = proposer._one_candidate(reflection, copy.deepcopy(trace), copy.deepcopy(parent))
                candidate.update(id=f"candidate_{index:02d}", checkpoint_key=point["key"])
            if candidate["valid"]:
                if attempt:
                    candidate["generation_additional_attempts"] = attempt
                break
            save_immutable(error_path, candidate)
            if not any("LLM response has no JSON object" in error for error in candidate["errors"]):
                break
            print(f"RETRY_MUTATION {point['key']}/{index} format_attempt={attempt + 1}", flush=True)
        if not candidate["valid"]:
            raise ValueError(f"Invalid mutation at {point['key']} candidate {index}: {candidate['errors']}")
        save_immutable(path, candidate)
        return f"{point['key']}/{candidate['id']}"

    execution.write_json(output / "status.json", {
        "phase": "generating", "candidate_count": candidate_count,
        "completed_candidates": candidate_count - len(specs),
        "controller_pid": os.getpid(), "updated_at": execution.timestamp(),
    })
    with ThreadPoolExecutor(max_workers=25) as pool:
        futures = [pool.submit(one, spec) for spec in specs]
        for done, future in enumerate(as_completed(futures), candidate_count - len(specs) + 1):
            print("GENERATED", future.result(), flush=True)
            execution.write_json(output / "status.json", {
                "phase": "generating", "candidate_count": candidate_count, "completed_candidates": done,
                "controller_pid": os.getpid(), "updated_at": execution.timestamp(),
            })
    counts = {}
    for point in plan["checkpoints"]:
        texts = [execution.read_json(output / "frozen" / point["key"] / f"candidate_{index:02d}.json")
                 ["playbook"]["playbook"] for index in range(1, 6)]
        counts[point["key"]] = len(set(texts))
    save_immutable(output / "generation_complete.json", {
        "candidates": candidate_count, "unique_candidates_by_checkpoint": counts,
    })


def make_jobs(output: Path, plan: dict) -> list[dict]:
    if not (output / "generation_complete.json").exists():
        raise ValueError("Generate all candidates before constructing the evaluation queue")
    jobs = []
    for point in plan["checkpoints"]:
        for context in ("parent", *(f"candidate_{index:02d}" for index in range(1, 6))):
            path = output / "frozen" / point["key"] / f"{context}.json"
            sha = execution.digest(path.read_bytes())
            partitions = {"train": point["train_task_ids"]}
            if context != "parent":
                partitions["val"] = point["validation_full_task_ids"]
            for partition, tasks in partitions.items():
                group = f"{point['key']}/{context}/{partition}"
                for index, task_id in enumerate(tasks):
                    jobs.append({
                        "key": f"{group}/{index:03d}", "group": group, "index": index,
                        "checkpoint_key": point["key"], "seed": point["seed"],
                        "checkpoint": point["checkpoint"], "context": context, "partition": partition,
                        "task_id": task_id, "context_path": str(path), "context_sha256": sha,
                        "model": point["model"], "max_steps": plan["max_steps"],
                    })
    # Mix all checkpoints, candidates and partitions in the same bounded queue.
    random.Random("bfcl_alignment_queue_20261011").shuffle(jobs)
    return jobs


def worker_init(output: str, settings: dict) -> None:
    global _OUTPUT, _SETTINGS
    _OUTPUT, _SETTINGS = Path(output), settings
    os.environ["OPENAI_API_BASE"] = settings["base_url"]
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["EXGENTIC_LLM_TRACE_FORMAT"] = "dedup_v1"
    os.environ.pop("EXGENTIC_PORT_BLOCK_START", None)
    from exgentic.agents.premise.instance import PremiseAgentInstance
    from exgentic.agents.premise.llm import configure_llm_slots

    PremiseAgentInstance._completion_with_retries = execution.premise_completion
    configure_llm_slots(1)


def evaluate_pair(job: dict) -> dict:
    from exgentic.core.types import RunConfig
    from exgentic.interfaces.lib.api import execute

    started = time.monotonic()
    row = {**job, "started_at": execution.timestamp(), "worker_pid": os.getpid()}
    audit = _OUTPUT / "request_audits" / f"{job['key']}.json"
    execution._AUDIT = {
        "audit_path": str(audit), "max_tokens": _SETTINGS["max_tokens"],
        "transport_additional_retries": _SETTINGS["transport_additional_retries"],
        "model_calls": 0, "output_length_retries": 0,
    }
    execution.write_json(_OUTPUT / "active_workers" / f"{os.getpid()}.json", {**row, "phase": "running"})
    try:
        path = Path(job["context_path"])
        if job["context_sha256"] not in _CONTEXTS:
            raw = path.read_bytes()
            if execution.digest(raw) != job["context_sha256"]:
                raise ValueError("Frozen candidate or parent changed")
            _CONTEXTS[job["context_sha256"]] = json.loads(raw)["playbook"]["playbook"]
        run_id = f"alignment_{job['checkpoint_key']}_{job['context']}_{job['partition']}_{job['index']:03d}"
        config = RunConfig(
            benchmark="bfcl", agent="premise", model=job["model"], task_ids=[job["task_id"]],
            output_dir=str(_OUTPUT / "sessions" / job["group"]), run_id=run_id,
            agent_kwargs={
                "model": job["model"], "playbook": _CONTEXTS[job["context_sha256"]],
                "runner": "direct", "enable_thinking": False,
                "model_settings": {"temperature": _SETTINGS["temperature"],
                                   "max_tokens": _SETTINGS["max_tokens"],
                                   "timeout": _SETTINGS["request_timeout"], "num_retries": 0,
                                   **({"reasoning_effort": _SETTINGS["reasoning_effort"]}
                                      if "reasoning_effort" in _SETTINGS else {})},
            },
            benchmark_kwargs={"subset": "multi_turn_base", "use_cache": False},
            max_workers=1, max_steps=job["max_steps"], max_actions=100, overwrite_sessions=True,
        )
        result = execute(config=config)
        if len(result.session_results) != 1:
            raise ValueError("Expected exactly one session result")
        session = result.session_results[0]
        row.update(session_id=session.session_id, session_status=session.status.value, steps=session.steps,
                   execution_time=session.execution_time)
        if session.status.value in ("error", "cancelled") or session.score is None:
            raise RuntimeError(str(session.details))
        if session.steps > job["max_steps"]:
            raise ValueError("Session exceeded the frozen step budget")
        row.update(status="completed", score=float(session.score))
    except Exception as exc:
        row.update(status="error", score=0.0, error_type=type(exc).__name__, error=str(exc))
    row.update(completed_at=execution.timestamp(), wall_seconds=time.monotonic() - started,
               model_calls=execution._AUDIT["model_calls"],
               output_length_retries=execution._AUDIT["output_length_retries"])
    execution.write_json(audit, execution._AUDIT)
    execution.write_json(_OUTPUT / "results" / f"{job['key']}.json", row)
    execution.write_json(_OUTPUT / "active_workers" / f"{os.getpid()}.json", {
        "phase": "idle", "completed_at": row["completed_at"], "last_key": job["key"],
    })
    return row


def correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    xm, ym = statistics.mean(x), statistics.mean(y)
    xx, yy = sum((v - xm) ** 2 for v in x), sum((v - ym) ** 2 for v in y)
    return sum((a - xm) * (b - ym) for a, b in zip(x, y, strict=True)) / math.sqrt(xx * yy) if xx * yy else None


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [statistics.mean(index + 1 for index, item in enumerate(ordered) if item == value) for value in values]


def selected_improvement(scores: list[float], improvements: list[float]) -> float:
    """Report the expectation of uniform selection among tied best candidates."""
    best = max(scores)
    return statistics.mean(delta for score, delta in zip(scores, improvements, strict=True) if score == best)


def analyze(output: Path, plan: dict, records: dict) -> dict:
    all_rows, checkpoints = [], []
    for point in plan["checkpoints"]:
        key = point["key"]
        parents = {row["task_id"]: row["score"] for row in records.values()
                   if row["checkpoint_key"] == key and row["context"] == "parent"}
        subsets = {"train": set(point["train_task_ids"]), "seen": set(point["seen_task_ids"]),
                   "train_without_past": set(point["train_task_ids"]) - set(point["past_task_ids"]),
                   "seen_without_past": set(point["seen_task_ids"]) - set(point["past_task_ids"])}
        candidates = []
        for index in range(1, 6):
            context = f"candidate_{index:02d}"
            rows = [row for row in records.values() if row["checkpoint_key"] == key and row["context"] == context]
            train = {row["task_id"]: row["score"] for row in rows if row["partition"] == "train"}
            val = {row["task_id"]: row["score"] for row in rows if row["partition"] == "val"}
            candidate = {"checkpoint_key": key, "candidate": context,
                         "s_exp": statistics.mean(train[task] for task in point["past_task_ids"]),
                         "s_val": statistics.mean(val[task] for task in point["validation_task_ids"]),
                         "s_val_full": statistics.mean(val.values()),
                         "execution_errors": sum(row["status"] == "error" for row in rows)}
            for name, tasks in subsets.items():
                candidate[f"delta_{name}"] = (
                    statistics.mean(train[task] - parents[task] for task in tasks) if tasks else None
                )
            candidates.append(candidate)
            all_rows.append(candidate)
        exp, val = [c["s_exp"] for c in candidates], [c["s_val"] for c in candidates]
        val_full = [c["s_val_full"] for c in candidates]
        measures = {}
        for name in subsets:
            delta = [c[f"delta_{name}"] for c in candidates]
            if any(value is None for value in delta):
                measures[name] = None
                continue
            picked_exp, picked_val = selected_improvement(exp, delta), selected_improvement(val, delta)
            rho_exp, rho_val = correlation(delta, exp), correlation(delta, val)
            picked_val_full = selected_improvement(val_full, delta)
            measures[name] = {
                "rho_exp": rho_exp, "rho_val": rho_val,
                "rho_val_full": correlation(delta, val_full),
                "rho_difference": rho_exp - rho_val if rho_exp is not None and rho_val is not None else None,
                "spearman_exp": correlation(ranks(delta), ranks(exp)),
                "spearman_val": correlation(ranks(delta), ranks(val)),
                "spearman_val_full": correlation(ranks(delta), ranks(val_full)),
                "selected_improvement_exp": picked_exp, "selected_improvement_val": picked_val,
                "selected_improvement_val_full": picked_val_full,
                "selection_difference": picked_exp - picked_val,
                "selection_gain_exp": picked_exp - statistics.mean(delta),
                "selection_gain_val": picked_val - statistics.mean(delta),
                "regret_exp": max(delta) - picked_exp, "regret_val": max(delta) - picked_val,
                "regret_val_full": max(delta) - picked_val_full,
            }
        checkpoints.append({"checkpoint_key": key, "seed": point["seed"], "checkpoint": point["checkpoint"],
                            "candidates": candidates, "measures": measures})
    summary = {"complete": True, "total_evaluations": len(records),
               "execution_errors": sum(row["status"] == "error" for row in records.values()),
               "tie_policy": "Uniform random tie-breaking expectation", "checkpoints": checkpoints,
               "zero_variance_correlations": sum(cp["measures"]["train"]["rho_exp"] is None or
                                                 cp["measures"]["train"]["rho_val"] is None for cp in checkpoints),
               "error_policy": "Execution errors contribute zero and are counted separately",
               "interpretation": f"{len(checkpoints)} checkpoint groups; descriptive comparisons "
                                 "without iid-candidate inference"}
    execution.write_json(output / "summary.json", summary)
    with (output / "candidate_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    return summary


def run_queue(output: Path, plan: dict) -> None:
    jobs = make_jobs(output, plan)
    save_immutable(output / "evaluation_queue.json", {"jobs": jobs})
    records = {}
    for job in jobs:
        path = output / "results" / f"{job['key']}.json"
        if path.exists():
            row = execution.read_json(path)
            if any(row.get(key) != value for key, value in job.items()):
                raise ValueError("Saved result does not match its frozen candidate/task pair")
            records[job["key"]] = row
    queued = deque(job for job in jobs if job["key"] not in records)
    pending = {}
    peak = 0
    started = execution.timestamp()

    def status(phase="evaluating"):
        execution.write_json(output / "status.json", {
            "phase": phase, "controller_pid": os.getpid(), "workers": plan["workers"],
            "completed_candidates": len(plan["checkpoints"]) * 5, "candidate_count": len(plan["checkpoints"]) * 5,
            "total_evaluations": len(jobs),
            "processed_tasks": len(records), "active_tasks": len(pending), "queued_tasks": len(queued),
            "peak_active_tasks": peak, "execution_errors": sum(row["status"] == "error" for row in records.values()),
            "max_steps": plan["max_steps"], "started_at": started, "updated_at": execution.timestamp(),
        })

    print(f"START evaluation: pairs={len(jobs)} concurrency={plan['workers']} "
          f"max_steps={plan['max_steps']}", flush=True)
    settings = {key: plan[key] for key in ("base_url", "max_tokens", "temperature", "request_timeout",
                                         "transport_additional_retries")}
    if "reasoning_effort" in plan:
        settings["reasoning_effort"] = plan["reasoning_effort"]
    with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init, initargs=(str(output), settings)) as pool:
        while queued or pending:
            while queued and len(pending) < plan["workers"]:
                job = queued.popleft()
                pending[pool.submit(evaluate_pair, job)] = job
            peak = max(peak, len(pending))
            status()
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                row = future.result()
                records[job["key"]] = row
                print(f"DONE {len(records)}/{len(jobs)} {job['key']} status={row['status']}", flush=True)
    summary = analyze(output, plan, records)
    status("complete" if not summary["execution_errors"] else "complete_with_errors")
    print(f"FINISH evaluations={len(records)} errors={summary['execution_errors']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument("--model", help="Override the model; proposal trajectories are rerun under this model.")
    parser.add_argument("--workers", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--checkpoint", action="append", metavar="SEED:STEP",
                        help="Checkpoint to include; repeat for each point. Defaults to the original five.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.max_steps < 1:
        parser.error("workers and max-steps must be positive")
    try:
        checkpoints = (tuple(tuple(map(int, item.split(":"))) for item in args.checkpoint)
                       if args.checkpoint else CHECKPOINTS)
        if any(len(item) != 2 or item[1] < 10 for item in checkpoints):
            raise ValueError
    except ValueError:
        parser.error("checkpoint must have the form SEED:STEP with STEP >= 10")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["OPENAI_API_BASE"] = args.base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["EXGENTIC_LLM_TRACE_FORMAT"] = "dedup_v1"
    os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")))
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            plan = prepare(output, args.source.resolve(), args.workers, args.base_url, args.max_steps, checkpoints,
                           args.reasoning_effort, args.model)
            print(f"PREPARED checkpoints={len(checkpoints)} candidates={plan['candidate_count']} "
                  f"evaluations={plan['total_evaluations']} "
                  f"max_steps={args.max_steps}", flush=True)
            if args.prepare_only:
                return
            generate_candidates(output, plan)
            if not args.generate_only:
                run_queue(output, plan)
        except Exception as exc:
            execution.write_json(output / "controller_error.json", {
                "error_type": type(exc).__name__, "error": str(exc), "at": execution.timestamp(),
            })
            raise


if __name__ == "__main__":
    main()
