# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Probe BFCL calls under identical histories at each five-task PREMiSE checkpoint."""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import csv
import fcntl
import json
import multiprocessing
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from scripts.premise.collect_bfcl_prefix_checkpoints import recover  # noqa: E402
from scripts.premise.run_bfcl_alignment import (  # noqa: E402
    correlation,
    evaluate_pair,
    ranks,
    save_immutable,
    worker_init,
)
from scripts.utils.evaluate_isolated_checkpoint_queue import digest, read_json, timestamp, write_json  # noqa: E402

POINTS = tuple(range(0, 51, 5))
METRICS = tuple(f"{mode}_{metric}" for mode in ("loose", "strict")
                for metric in ("accuracy", "precision", "recall", "f1"))


def decode_record(value):
    """Decode legacy LiteLLM reprs as data, without evaluating constructor calls."""
    if not isinstance(value, str):
        return value

    def visit(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [visit(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {visit(key): visit(item) for key, item in zip(node.keys, node.values, strict=True)}
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"Choices", "Message", "Function", "ChatCompletionMessageToolCall"}
                and not node.args and all(item.arg is not None for item in node.keywords)):
            return {item.arg: visit(item.value) for item in node.keywords}
        raise ValueError("Unsupported trace representation")

    return visit(ast.parse(value, mode="eval").body)


def response_message(response: dict) -> dict:
    choice = decode_record(response["choices"][0])
    if choice.get("finish_reason") == "length":
        raise ValueError("Output truncated")
    message = decode_record(choice["message"])
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("Missing assistant message")
    if not message.get("tool_calls") and not message.get("content"):
        raise ValueError("Empty assistant response")
    return message


def normalize_calls(message: dict) -> list[dict]:
    calls = []
    for item in message.get("tool_calls") or []:
        function = decode_record(decode_record(item)["function"])
        name, arguments = function["name"], function["arguments"]
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid tool name")
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        # IDs, content, reasoning text and object-key order do not define an action.
        calls.append({"name": name, "arguments": arguments})
    return calls


def call_keys(calls: list[dict], strict: bool) -> list[str]:
    return [json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if strict else call["name"] for call in calls]


def score_calls(predicted: list[dict], reference: list[dict]) -> dict:
    result = {}
    for mode in ("loose", "strict"):
        pred, ref = call_keys(predicted, mode == "strict"), call_keys(reference, mode == "strict")
        matched = sum((Counter(pred) & Counter(ref)).values())
        precision = matched / len(pred) if pred else float(not ref)
        recall = matched / len(ref) if ref else float(not pred)
        result.update({f"{mode}_accuracy": float(pred == ref), f"{mode}_precision": precision,
                       f"{mode}_recall": recall,
                       f"{mode}_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    return result


def task_balanced(rows: list[dict]) -> dict:
    tasks = defaultdict(list)
    for row in rows:
        tasks[row["task_id"]].append(row)
    return {metric: statistics.mean(statistics.mean(row[metric] for row in task_rows)
                                    for task_rows in tasks.values()) if tasks else None for metric in METRICS}


def resume_config(source: dict) -> dict:
    """Retain the exact BFCL order and sampler settings in an isolated continuation."""
    config = copy.deepcopy(source)
    config.update(benchmarks=["bfcl"], mode="isolated",
                  order=[item for item in config["order"] if item[0] == "bfcl"],
                  benchmark_configs={"bfcl": config["benchmark_configs"]["bfcl"]})
    config["max_steps"]["bfcl"] = 20
    if len(config["order"]) != 50:
        raise ValueError("Expected exactly fifty frozen BFCL training tasks")
    return config


def pipeline_for(run: Path, model: dict):
    from exgentic.agents.premise.pipeline import PremisePipeline

    config = read_json(run / "state.json")["config"]
    return PremisePipeline(
        output=run, manifest=Path(config["manifest"]), benchmarks=config["benchmarks"],
        mode=config["mode"], seed=config["seed"], count=config["count"], past_count=config["past_count"],
        model=config["model"], base_url=model["base_url"], max_steps=config["max_steps"],
        replay_retries=config["replay_retries"], reasoning_effort=model.get("reasoning_effort"),
        llm_calls=50, llm_timeout=600, llm_retries=5, tool_calling_max_tokens=8192,
        benchmark_limits={"bfcl": None}, execution_tag="same_history_cp5_resume",
    )


def prepare_model(output: Path, model: dict, split_hash: str) -> dict:
    """Verify full source states and fork continuations without issuing model requests."""
    prepared = output / "prepared.json"
    if prepared.exists():
        prior = read_json(prepared)
        if prior["model"] != model or prior["manifest_sha256"] != split_hash:
            raise ValueError("Prepared model configuration changed")
        return prior
    runs = []
    for item in model["sources"]:
        seed, source = item["seed"], Path(item["run"])
        raw = (source / "state.json").read_bytes()
        saved = json.loads(raw)
        if saved["config"]["model"] != model["model"] or saved["config"].get("reasoning_effort") != (
            model.get("reasoning_effort")
        ):
            raise ValueError("Contexts must be learned by the probed model with the specified reasoning setting")
        if not 5 <= saved["cursor"] <= 50:
            raise ValueError("Source training must cover a BFCL-only prefix of five to fifty tasks")
        recover(source, output / "source_verified" / f"s{seed}", (saved["cursor"],), seed,
                split_hash, minimum_point=1, allow_terminal=True)
        run = source
        if saved["cursor"] < 50:
            run = output / "training" / f"s{seed}"
            branch = copy.deepcopy(saved)
            branch["config"] = resume_config(saved["config"])
            key = "bfcl" if saved["config"]["mode"] == "isolated" else "global"
            branch["states"] = {"bfcl": copy.deepcopy(saved["states"][key])}
            # Preparation is restartable, and never rewinds an already advancing branch.
            if not (run / "state.json").exists():
                save_immutable(run / "state.json", branch)
            for index in range(1, saved["cursor"] + 1):
                save_immutable(run / "iterations" / f"iter_{index:04d}.json",
                               read_json(source / "iterations" / f"iter_{index:04d}.json"))
            pipeline_for(run, model)  # Exact configuration/order/RNG restore preflight; no inference.
        if (source / "state.json").read_bytes() != raw:
            raise ValueError("Source changed during continuation preparation")
        runs.append({"seed": seed, "source": str(source), "run": str(run), "resume_cursor": saved["cursor"],
                     "source_state_sha256": digest(raw), "source_max_steps": saved["config"]["max_steps"]["bfcl"],
                     "continued_max_steps": 20 if saved["cursor"] < 50 else None,
                     "beam_memory_rng_preserved": True})
    result = {"model": model, "manifest_sha256": split_hash, "runs": runs, "checkpoints": list(POINTS)}
    save_immutable(prepared, result)
    return result


def finish_training(output: Path, prepared: dict) -> None:
    pending = [item for item in prepared["runs"] if read_json(Path(item["run"]) / "state.json")["cursor"] < 50]

    def status(active_seeds):
        progress = {str(item["seed"]): read_json(Path(item["run"]) / "state.json")["cursor"]
                    for item in prepared["runs"]}
        write_json(output / "status.json", {
            "phase": "own_model_training", "updated_at": timestamp(), "pending_seeds": active_seeds,
            "seed_cursors": progress, "target_cursor": 50, "max_steps": 20, "concurrency": 150,
            "model": prepared["model"]["model"], "base_url": prepared["model"]["base_url"],
            "controller_pid": os.getpid(), "new_training_attempted":
                sum(progress[str(item["seed"])] - item["resume_cursor"] for item in prepared["runs"]),
            "new_training_tasks": sum(50 - item["resume_cursor"] for item in prepared["runs"]),
        })

    status([item["seed"] for item in pending])

    from exgentic.agents.premise.llm import configure_llm_slots

    pipelines = [pipeline_for(Path(item["run"]), prepared["model"]) for item in pending]
    # Constructors reset the semaphore; finish all constructors before any seed starts.
    configure_llm_slots(150)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(pipeline.run, stop_after=50): item["seed"]
                   for pipeline, item in zip(pipelines, pending, strict=True)}
        while futures:
            done, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
                del futures[future]
            status(list(futures.values()))


def collect_contexts(output: Path, prepared: dict) -> list[dict]:
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK

    contexts = []
    for item in prepared["runs"]:
        seed, run = item["seed"], Path(item["run"])
        if read_json(run / "state.json")["cursor"] != 50:
            raise ValueError("Cannot probe an incomplete fifty-task training run")
        folder = output / "frozen" / f"s{seed}"
        groups = recover(run, folder / "states", POINTS[1:], seed, prepared["manifest_sha256"],
                         minimum_point=1, allow_terminal=True)
        for point in POINTS:
            playbook = DEFAULT_PLAYBOOK
            if point:
                group = next(group for group in groups if group["checkpoint"] == point)
                state = read_json(Path(group["checkpoint_path"]))
                playbook = next(iter(state["states"].values()))["beam"][0]["playbook"]
            path = folder / f"context_{point:03d}.json"
            save_immutable(path, {"playbook": {"playbook": playbook}, "seed": seed, "checkpoint": point})
            contexts.append({"seed": seed, "checkpoint": point, "context_path": str(path),
                             "context_sha256": digest(path.read_bytes())})
    save_immutable(output / "contexts.json", {"contexts": contexts})
    return contexts


def reference_jobs(prepared: dict, contexts: list[dict]) -> list[dict]:
    jobs = []
    for item in prepared["runs"]:
        config = read_json(Path(item["run"]) / "state.json")["config"]
        context = next(context for context in contexts if context["seed"] == item["seed"] and not context["checkpoint"])
        for index, (_, task) in enumerate(config["order"]):
            key = f"s{item['seed']}_{index:03d}"
            jobs.append({"key": key, "seed": item["seed"], "task_id": task, "model": config["model"],
                         "context_path": context["context_path"], "context_sha256": context["context_sha256"],
                         "checkpoint_key": f"s{item['seed']}_cp000", "context": "parent", "partition": "reference",
                         "index": index, "group": f"s{item['seed']}", "max_steps": 20})
    return jobs


def build_reference_rollouts(output: Path, prepared: dict, contexts: list[dict], workers: int) -> list[dict]:
    reference = output / "reference"
    model = prepared["model"]
    settings = {"base_url": model["base_url"], "temperature": 0.0, "max_tokens": 8192,
                "transport_additional_retries": 5, "request_timeout": 600}
    if model.get("reasoning_effort"):
        settings["reasoning_effort"] = model["reasoning_effort"]
    jobs = reference_jobs(prepared, contexts)
    records = {path.stem: read_json(path) for path in (reference / "results").glob("*.json")}
    pending = deque(job for job in jobs if job["key"] not in records)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init, initargs=(str(reference), settings)) as pool:
        active = {}
        while pending or active:
            while pending and len(active) < workers:
                job = pending.popleft()
                active[pool.submit(evaluate_pair, job)] = job
            write_json(output / "status.json", {"phase": "initial_context_reference_rollouts",
                                               "processed": len(records), "total": len(jobs), "active": len(active),
                                               "updated_at": timestamp()})
            if active:
                done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    job = active.pop(future)
                    records[job["key"]] = future.result()
    return [records[job["key"]] for job in jobs]


def freeze_history_bank(output: Path, rollouts: list[dict]) -> list[dict]:
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
    from exgentic.agents.ace.prompts.generator import build_generator_system_prompt
    from exgentic.integrations.litellm.trace_logger import iter_expanded_trace_rows

    bank_path = output / "history_bank.json"
    if bank_path.exists():
        bank = read_json(bank_path)["anchors"]
        for anchor in bank:
            if digest(Path(anchor["path"]).read_bytes()) != anchor["sha256"]:
                raise ValueError("Frozen history bank changed")
        return bank
    anchors, audit = [], []
    for rollout in rollouts:
        run_id = (f"alignment_{rollout['checkpoint_key']}_{rollout['context']}_"
                  f"{rollout['partition']}_{rollout['index']:03d}")
        session_root = output / "reference/sessions" / rollout["group"] / run_id / "sessions"
        paths = sorted(session_root.glob("*/agent/litellm/trace.jsonl"))
        if len(paths) != 1:
            audit.append({"rollout": rollout["key"], "status": "missing_trace", "matches": len(paths)})
            continue
        trace = paths[0]
        raw_hash = digest(trace.read_bytes())
        for index, row in enumerate(iter_expanded_trace_rows(trace)):
            request = row.get("request") or {}
            # Shortlisting calls are excluded. The acting call's actual tools are frozen.
            if not request.get("tools"):
                continue
            try:
                if request["messages"][0] != {"role": "system", "content": build_generator_system_prompt(
                    DEFAULT_PLAYBOOK
                )}:
                    raise ValueError("Reference does not use the initial generator context")
                reference_calls = normalize_calls(response_message(row["response"]))
                if not reference_calls:
                    audit.append({"rollout": rollout["key"], "trace_row": index, "status": "no_reference_calls"})
                    continue
                key = f"{rollout['key']}_a{index:04d}"
                path = output / "histories" / f"{key}.json"
                data = {"key": key, "seed": rollout["seed"], "task_id": rollout["task_id"],
                        "messages": request["messages"], "tools": request["tools"],
                        "reference_calls": reference_calls, "source_trace": str(trace), "source_trace_row": index,
                        "source_trace_sha256": raw_hash, "reference_rollout_status": rollout["status"],
                        "reference_rollout_score": rollout["score"]}
                save_immutable(path, data)
                anchors.append({"key": key, "seed": rollout["seed"], "task_id": rollout["task_id"],
                                "path": str(path), "sha256": digest(path.read_bytes())})
            except (ValueError, KeyError, TypeError, IndexError, SyntaxError) as exc:
                audit.append({"rollout": rollout["key"], "trace_row": index, "status": "invalid_reference",
                              "error": str(exc)})
    if not anchors:
        raise ValueError("No valid initial-context action histories found")
    save_immutable(output / "reference_audit.json", {
        "rollouts": rollouts, "excluded_acting_requests": audit, "anchor_count": len(anchors),
        "task_counts_by_seed": {str(seed): len({a['task_id'] for a in anchors if a['seed'] == seed})
                                for seed in (42, 43, 44)},
    })
    save_immutable(bank_path, {"anchors": anchors, "source": "Own-model initial-context BFCL train50 rollouts",
                               "frozen_fields": ["messages excluding first system context", "tools"],
                               "reference_max_steps": 20})
    return anchors


def probe_request(anchor: dict, playbook: str, model: dict) -> dict:
    from exgentic.agents.ace.prompts.generator import build_generator_system_prompt

    messages = copy.deepcopy(anchor["messages"])
    if messages[0]["role"] != "system":
        raise ValueError("Missing generator system message")
    messages[0]["content"] = build_generator_system_prompt(playbook)
    body = {"model": model["model"].removeprefix("openai/"), "messages": messages,
            "tools": copy.deepcopy(anchor["tools"]), "temperature": 0.0, "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": False}}
    if model.get("reasoning_effort"):
        body["reasoning_effort"] = model["reasoning_effort"]
    return body


async def run_probes(output: Path, anchors: list[dict], contexts: list[dict], model: dict, workers: int) -> None:
    import httpx

    jobs = [(anchor, context) for anchor in anchors for context in contexts if anchor["seed"] == context["seed"]]
    random.Random("bfcl_policy_shift_cp5").shuffle(jobs)
    context_books = {}
    for context in contexts:
        path = Path(context["context_path"])
        if digest(path.read_bytes()) != context["context_sha256"]:
            raise ValueError("Frozen context changed")
        context_books[(context["seed"], context["checkpoint"])] = read_json(path)["playbook"]["playbook"]
    queue = asyncio.Queue()
    completed, errors, active = 0, 0, 0
    for anchor, context in jobs:
        key = f"{anchor['key']}_cp{context['checkpoint']:03d}"
        path = output / "probe_results" / f"{key}.json"
        if path.exists():
            old = read_json(path)
            if (old["anchor_sha256"], old["context_sha256"]) != (anchor["sha256"], context["context_sha256"]):
                raise ValueError("Probe provenance changed")
            if old["status"] == "completed":
                completed += 1
                continue
        queue.put_nowait((anchor, context, key, path))

    def status():
        write_json(output / "status.json", {"phase": "same_history_probes", "total": len(jobs),
                                           "processed": completed, "errors": errors, "active": active,
                                           "queued": queue.qsize(), "concurrency": workers,
                                           "checkpoints": list(POINTS), "updated_at": timestamp()})

    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=workers,
                                                               max_keepalive_connections=workers)) as client:
        async def worker():
            nonlocal completed, errors, active
            while not queue.empty():
                anchor, context, key, path = queue.get_nowait()
                active += 1
                started = time.monotonic()
                if digest(Path(anchor["path"]).read_bytes()) != anchor["sha256"]:
                    raise ValueError("History changed before probing")
                history = read_json(Path(anchor["path"]))
                body = probe_request(history, context_books[(context["seed"], context["checkpoint"])], model)
                row = {"key": key, "anchor_key": anchor["key"], "anchor_sha256": anchor["sha256"],
                       "context_sha256": context["context_sha256"], "request_sha256": digest(json.dumps(
                           body, ensure_ascii=False, sort_keys=True).encode()), "seed": anchor["seed"],
                       "task_id": anchor["task_id"], "checkpoint": context["checkpoint"], "started_at": timestamp()}
                attempts = []
                for attempt in range(6):
                    try:
                        response = await client.post(model["base_url"].rstrip("/") + "/chat/completions", json=body,
                                                     headers={"Authorization": "Bearer EMPTY"})
                        response.raise_for_status()
                        payload = response.json()
                        predicted = normalize_calls(response_message(payload))
                        row.update(status="completed", predicted_calls=predicted,
                                   metrics=score_calls(predicted, history["reference_calls"]), response=payload)
                        break
                    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
                        detail = str(exc)
                        if isinstance(exc, httpx.HTTPStatusError):
                            detail += " " + exc.response.text[:4000]
                        attempts.append({"attempt": attempt + 1, "type": type(exc).__name__, "error": detail})
                        row.update(status="error", error=detail)
                        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 401, 403, 404):
                            break
                        if attempt < 5:
                            await asyncio.sleep(min(2 ** attempt, 30))
                row.update(attempts=attempts, wall_seconds=time.monotonic() - started, completed_at=timestamp())
                write_json(path, row)
                completed += 1
                errors += row["status"] != "completed"
                active -= 1
                queue.task_done()
                if completed % 50 == 0:
                    status()

        async def heartbeat():
            while True:
                status()
                await asyncio.sleep(30)

        ticker = asyncio.create_task(heartbeat())
        try:
            await asyncio.gather(*(worker() for _ in range(min(workers, queue.qsize()))))
        finally:
            ticker.cancel()
        status()


def summarize_predictions(anchors: list[dict], records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    valid = {(row["anchor_key"], row["checkpoint"]): row for row in records if row["status"] == "completed"}
    common = [anchor for anchor in anchors if all((anchor["key"], point) in valid for point in POINTS)]
    coverage = {"total_anchors": len(anchors), "common_anchors_all_checkpoints": len(common),
                "common_tasks_by_seed": {str(seed): len({a['task_id'] for a in common if a['seed'] == seed})
                                         for seed in (42, 43, 44)},
                "total_probes": len(anchors) * len(POINTS), "successful_probes": len(valid),
                "failed_or_missing_probes": len(anchors) * len(POINTS) - len(valid)}
    checkpoint_rows, pair_rows = [], []
    for seed in (42, 43, 44):
        seed_common = [anchor for anchor in common if anchor["seed"] == seed]
        for point in POINTS:
            rows = [{"task_id": anchor["task_id"], **valid[(anchor["key"], point)]["metrics"]}
                    for anchor in seed_common]
            available = [{"task_id": anchor["task_id"], **valid[(anchor["key"], point)]["metrics"]}
                         for anchor in anchors if anchor["seed"] == seed and (anchor["key"], point) in valid]
            checkpoint_rows.append({"seed": seed, "checkpoint": point, "common_anchors": len(rows),
                                    "common_tasks": len({row["task_id"] for row in rows}),
                                    "available_anchors": len(available), **task_balanced(rows),
                                    **{f"available_{k}": v for k, v in task_balanced(available).items()},
                                    **{f"step_weighted_{k}": statistics.mean(row[k] for row in rows)
                                       if rows else None for k in METRICS}})
        for i, left in enumerate(POINTS):
            for right in POINTS[i + 1:]:
                rows = [{"task_id": anchor["task_id"], **score_calls(
                    valid[(anchor["key"], left)]["predicted_calls"],
                    valid[(anchor["key"], right)]["predicted_calls"])} for anchor in seed_common]
                pair_rows.append({"seed": seed, "left": left, "right": right, "distance": right - left,
                                  "common_anchors": len(rows), **task_balanced(rows)})
    return checkpoint_rows, pair_rows, coverage


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(output: Path, anchors: list[dict], model: dict, *, benchmark: str = "BFCL",
            training_audit: list[dict] | None = None, initial_reference: bool = True) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [read_json(path) for path in (output / "probe_results").glob("*.json")]
    checkpoint_rows, pair_rows, coverage = summarize_predictions(anchors, records)
    folder = output / "analysis"
    write_csv(folder / "checkpoint_metrics.csv", checkpoint_rows)
    write_csv(folder / "pairwise_metrics.csv", pair_rows)
    write_json(folder / "coverage.json", coverage)
    valid = {(row["anchor_key"], row["checkpoint"]): row for row in records if row["status"] == "completed"}
    common = [anchor for anchor in anchors if all((anchor["key"], point) in valid for point in POINTS)]
    per_task = []
    for seed in (42, 43, 44):
        for task in sorted({anchor["task_id"] for anchor in common if anchor["seed"] == seed}):
            task_anchors = [a for a in common if a["seed"] == seed and a["task_id"] == task]
            for point in POINTS:
                per_task.append({"seed": seed, "task_id": task, "checkpoint": point, "anchors": len(task_anchors),
                                 **{metric: statistics.mean(valid[(a["key"], point)]["metrics"][metric]
                                                            for a in task_anchors) for metric in METRICS}})
    if per_task:
        write_csv(folder / "task_metrics.csv", per_task)
    aggregate = []
    for point in POINTS:
        rows = [row for row in checkpoint_rows if row["checkpoint"] == point]
        data = {"checkpoint": point, "seeds": sum(row["strict_accuracy"] is not None for row in rows)}
        for metric in METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            data[metric] = statistics.mean(values) if values else None
            data[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        aggregate.append(data)
    write_csv(folder / "seed_summary.csv", aggregate)
    correlations = []
    for seed in (*range(42, 45), "mean"):
        source = aggregate if seed == "mean" else [row for row in checkpoint_rows if row["seed"] == seed]
        for metric in METRICS:
            usable = [row for row in source if row["checkpoint"] and row[metric] is not None]
            x, y = [row["checkpoint"] for row in usable], [row[metric] for row in usable]
            correlations.append({"seed": seed, "metric": metric, "points": len(usable),
                                 "pearson": correlation(x, y) if len(x) > 1 else None,
                                 "spearman": correlation(ranks(x), ranks(y)) if len(x) > 1 else None})
    write_csv(folder / "checkpoint_correlations.csv", correlations)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS, strict=True):
        available = [row for row in aggregate if row[metric] is not None]
        x, y = [row["checkpoint"] for row in available], [row[metric] for row in available]
        sd = [row[metric + "_std"] for row in available]
        axis.plot(x, y, marker="o", markersize=3)
        axis.fill_between(x, [v - s for v, s in zip(y, sd, strict=True)],
                          [v + s for v, s in zip(y, sd, strict=True)], alpha=0.15)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xticks(POINTS)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Training checkpoint")
    reference_label = "initial-policy calls" if initial_reference else "recorded training calls"
    axes[0, 0].set_ylabel("Agreement with " + reference_label)
    axes[1, 0].set_ylabel("Agreement with " + reference_label)
    fig.suptitle(f"{benchmark} · {model['key']} · task-balanced mean ± seed SD\n"
                 f"Same complete histories across CP 0/5/…/50: "
                 f"{coverage['common_anchors_all_checkpoints']}/{coverage['total_anchors']} anchors")
    for extension in ("pdf", "png"):
        fig.savefig(folder / f"checkpoint_agreement.{extension}", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True, constrained_layout=True)
    for axis, metric in zip(axes, ("strict_accuracy", "strict_f1"), strict=True):
        x, y, spread = [], [], []
        for distance in POINTS[1:]:
            values = [1 - row[metric] for row in pair_rows if row["distance"] == distance and row[metric] is not None]
            if values:
                x.append(distance)
                y.append(statistics.mean(values))
                spread.append(statistics.stdev(values) if len(values) > 1 else 0.0)
        axis.plot(x, y, marker="o", markersize=3)
        axis.fill_between(x, [v - s for v, s in zip(y, spread, strict=True)],
                          [v + s for v, s in zip(y, spread, strict=True)], alpha=0.15)
        axis.set_title("1 - " + metric.replace("strict_", "").upper())
        axis.set_xlabel("Checkpoint distance (tasks)")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Same-history action disagreement")
    fig.suptitle(f"{model['key']} · within-seed pairs · mean ± pair SD")
    for extension in ("pdf", "png"):
        fig.savefig(folder / f"pairwise_policy_shift.{extension}", dpi=200)
    plt.close(fig)
    prepared = read_json(output / "prepared.json")
    if training_audit is None:
        training_audit = []
        for item in prepared["runs"]:
            artifacts = [read_json(Path(item["run"]) / "iterations" / f"iter_{index:04d}.json")
                         for index in range(1, 51)]
            training_audit.append({"seed": item["seed"], "attempted_tasks": 50,
                                   "skipped_tasks": sum(a.get("status") == "skipped" for a in artifacts),
                                   "successful_selection_tasks": sum("selection" in a for a in artifacts),
                                   "source_max_steps": item["source_max_steps"],
                                   "continued_max_steps": item["continued_max_steps"]})
    summary = {"benchmark": benchmark, "model": model, "checkpoints": list(POINTS), "coverage": coverage,
               "primary_aggregation": "Equal task weights within seed, then equal seed weights; sample seed SD",
               "accuracy": "Exact ordered sequence of function names (loose) or names and parsed arguments (strict)",
               "precision_recall_f1": "Multiset overlap of tool calls; duplicates count, order ignored",
               "reference": ("Own-model initial-context actions at saved train50 history prefixes" if initial_reference
                             else "Observed ACE training actions at recorded histories with task-specific contexts"),
               "cp0": ("Independent re-query of initial context; estimates repeatability" if initial_reference
                       else "Initial-context query on recorded training histories; not a repeatability control"),
               "errors": "Missing/invalid responses are reported as errors, never imputed as zero accuracy",
               "pairwise": "Within-seed predictions on the same all-checkpoint complete history bank",
               "checkpoint_unit": "Attempted training tasks; skipped updates remain on the frozen order",
               "correlations": "Descriptive Pearson/Spearman across CP5..50; dependent checkpoints, no iid p-values",
               "training_audit": training_audit, "provenance": prepared, "finished_at": timestamp()}
    write_json(folder / "summary.json", summary)
    return summary


def run(plan: dict, key: str, prepare_only: bool = False, base_url: str | None = None) -> None:
    if plan["checkpoints"] != list(POINTS) or plan["workers"] != 150 or plan["reference_max_steps"] != 20:
        raise ValueError("This study requires CP0/5/…/50, concurrency150, and twenty-step reference rollouts")
    if digest(Path(plan["manifest"]).read_bytes()) != plan["manifest_sha256"]:
        raise ValueError("Frozen task split changed")
    declared_model = next(model for model in plan["models"] if model["key"] == key)
    model = copy.deepcopy(declared_model)
    if {item["seed"] for item in model["sources"]} != {42, 43, 44} or len(model["sources"]) != 3:
        raise ValueError("Expected exactly three distinct seed sources")
    output = Path(plan["output"]) / key
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime_path = output / "runtime_manifest.json"
        if base_url:
            model["base_url"] = base_url
        elif runtime_path.exists():
            model["base_url"] = read_json(runtime_path)["model"]["base_url"]
        os.environ["OPENAI_API_BASE"] = model["base_url"]
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        # Endpoint routing is runtime state; preserve the immutable source/model preparation.
        prepared = prepare_model(output, declared_model, plan["manifest_sha256"])
        if prepare_only:
            print(f"PREPARED {key}: {[item['resume_cursor'] for item in prepared['runs']]}", flush=True)
            return
        runtime = {"model": model, "declared_base_url": declared_model["base_url"],
                   "controller_pid": os.getpid(), "controller_start_ticks":
                       Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
                   "output": str(output.resolve()), "started_at": timestamp(), "workers": plan["workers"],
                   "checkpoints": plan["checkpoints"], "max_steps": plan["reference_max_steps"],
                   "source_preparation_preserved": True}
        write_json(runtime_path, runtime)
        save_immutable(output / "runtime_manifests" / f"{os.getpid()}.json", runtime)
        prepared = copy.deepcopy(prepared)
        prepared["model"] = model
        print(f"START {key}: endpoint={model['base_url']} CP0/5/…/50 concurrency={plan['workers']}", flush=True)
        try:
            finish_training(output, prepared)
            contexts = collect_contexts(output, prepared)
            rollouts = build_reference_rollouts(output, prepared, contexts, plan["workers"])
            anchors = freeze_history_bank(output, rollouts)
            asyncio.run(run_probes(output, anchors, contexts, model, plan["workers"]))
            summary = analyze(output, anchors, model)
            write_json(output / "status.json", {"phase": "complete" if not summary["coverage"][
                "failed_or_missing_probes"] else "complete_with_errors", "updated_at": timestamp(),
                "coverage": summary["coverage"], "controller_pid": os.getpid()})
        except Exception as exc:
            write_json(output / "status.json", {"phase": "error", "error": str(exc), "updated_at": timestamp()})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--base-url", help="Override only runtime routing; retain frozen source/model preparation")
    args = parser.parse_args()
    run(read_json(args.plan), args.model_key, args.prepare_only, args.base_url)


if __name__ == "__main__":
    main()
