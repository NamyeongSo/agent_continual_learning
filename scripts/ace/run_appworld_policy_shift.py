# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Restore own-model ACE inputs and probe contexts without launching AppWorld environments."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from scripts.premise.run_bfcl_policy_shift import (  # noqa: E402
    POINTS,
    analyze,
    normalize_calls,
    response_message,
    run_probes,
    save_immutable,
)
from scripts.utils.evaluate_isolated_checkpoint_queue import digest, read_json, timestamp, write_json  # noqa: E402

MODEL = "openai/Qwen/Qwen3.6-35B-A3B"
SEEDS = (42, 43, 44)
HISTORY_LIMIT = 20


def source_model_matches(trace_model: str, learned_model: str) -> bool:
    # LiteLLM removes its provider prefix when logging requests. For OSS the
    # actual model ID also starts with openai/, so stripping both sides loses it.
    return trace_model in {learned_model, learned_model.removeprefix("openai/")}


def prepare(output: Path, bundle: Path, base_url: str, workers: int, model_name: str = MODEL,
            reasoning_effort: str | None = None) -> dict:
    """Validate exported ACE contexts against their completed, immutable training sources."""
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
    from exgentic.utils.task_splits import validate_task_split

    manifest = read_json(bundle / "manifest.json")
    if manifest["benchmark"] != "appworld" or manifest["mode"] != "isolated":
        raise ValueError("Expected an isolated AppWorld checkpoint collection")
    splits_path = bundle / "task_split_manifest.json"
    if digest(splits_path.read_bytes()) != manifest["manifest_sha256"]:
        raise ValueError("Frozen task split changed")
    splits = read_json(splits_path)
    model_key = "qwen_ace" if model_name == MODEL else model_name.rsplit("/", 1)[-1].replace("-", "_") + "_ace"
    model = {"key": model_key, "model": model_name, "base_url": base_url, "enable_thinking": False,
             "method": "ace", "reasoning_effort": reasoning_effort}
    runs, contexts, audit = [], [], []
    entries = {(entry["seed"], entry["checkpoint"]): entry for entry in manifest["checkpoints"]
               if entry["method"] == "ace"}
    if set(entries) != {(seed, point) for seed in SEEDS for point in POINTS[1:]}:
        raise ValueError("Expected all thirty ACE checkpoints at CP5..50 for three seeds")
    for seed in SEEDS:
        source_entry = next(item for item in manifest["runs"] if item["method"] == "ace" and item["seed"] == seed)
        source = Path(source_entry["source_run"])
        config_path = source / "experiment_config.json"
        config = read_json(config_path)
        if (config["model"] != model_name or config["agent"] != "ace" or config["seed"] != seed
                or config["mode"] != "isolated" or config["benchmarks"] != ["appworld"]
                or not config["disable_thinking"]):
            raise ValueError("Contexts must come from the requested model's isolated AppWorld ACE learning")
        if reasoning_effort is not None and config.get("reasoning_effort") != reasoning_effort:
            raise ValueError("Learning and evaluation reasoning effort differ")
        if (config != read_json(bundle / "source_configs" / f"ace_s{seed}.json")
                or config["task_split_manifest_sha256"] != manifest["manifest_sha256"]):
            raise ValueError("Exported configuration differs from source learning")
        task_ids = [task for benchmark, task in config["task_order"] if benchmark == "appworld"]
        train_ids = validate_task_split(splits[f"appworld_{seed}"], "appworld")["train"]
        if len(task_ids) != 50 or len(set(task_ids)) != 50 or set(task_ids) != set(train_ids):
            raise ValueError("Expected exactly the fifty frozen training tasks")
        metrics_path = source / "online_metrics.jsonl"
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        completion_path = source / "training_complete.json"
        if (read_json(completion_path)["completed_task_count"] != 50 or len(metrics) != 50
                or [row["task_id"] for row in metrics] != task_ids):
            raise ValueError("Source ACE training is incomplete or its task order changed")
        source_checks = {str(path): digest(path.read_bytes()) for path in
                         (config_path, metrics_path, completion_path)}
        for point in POINTS:
            playbook = DEFAULT_PLAYBOOK
            if point:
                entry = entries[(seed, point)]
                checkpoint = bundle / entry["path"]
                if digest(checkpoint.read_bytes()) != entry["sha256"]:
                    raise ValueError("Exported ACE checkpoint changed")
                snapshot = read_json(checkpoint)
                if snapshot["store_id"] != "ace_isolated_appworld":
                    raise ValueError("Checkpoint belongs to another benchmark")
                original = Path(entry["source"])
                if digest(original.read_bytes()) != entry["source_sha256"]:
                    raise ValueError("Original ACE checkpoint changed")
                source_checks[str(checkpoint)] = entry["sha256"]
                source_checks[str(original)] = entry["source_sha256"]
                playbook = snapshot["playbook"]
                if point == 50:
                    final_path = source / "playbook_ace_isolated_appworld.json"
                    if read_json(final_path) != snapshot:
                        raise ValueError("Exported CP50 differs from the final learned ACE state")
                    source_checks[str(final_path)] = digest(final_path.read_bytes())
            context = output / "frozen" / f"s{seed}" / f"context_{point:03d}.json"
            save_immutable(context, {"seed": seed, "checkpoint": point, "playbook": {"playbook": playbook}})
            contexts.append({"seed": seed, "checkpoint": point, "context_path": str(context),
                             "context_sha256": digest(context.read_bytes())})
        runs.append({"seed": seed, "source": str(source), "run": str(source), "source_checks": source_checks,
                     "source_max_steps": config["benchmark_max_turns"]["appworld"], "continued_max_steps": None,
                     "task_ids": task_ids})
        audit.append({"seed": seed, "attempted_tasks": 50, "source_max_steps":
                      config["benchmark_max_turns"]["appworld"], "continued_max_steps": None,
                      "recorded_training_errors": sum(row["status"] == "error" for row in metrics),
                      "new_training_tasks": source_entry.get("study_training_tasks", 0),
                      "training_method": "ACE with training-time feedback"})
    prepared = {"benchmark": "appworld", "model": model, "runs": runs, "checkpoints": list(POINTS),
                "manifest_sha256": manifest["manifest_sha256"], "bundle": str(bundle),
                "bundle_manifest_sha256": digest((bundle / "manifest.json").read_bytes()),
                "workers": workers, "history_limit_per_task": HISTORY_LIMIT, "max_tokens": 8192, "learning": False,
                "reference_source": "Recorded ACE training actions", "environment_execution": False,
                "training_audit": audit}
    save_immutable(output / "prepared.json", prepared)
    save_immutable(output / "contexts.json", {"contexts": contexts})
    return prepared


def recorded_trace_definitions(prepared: dict, source_checks: dict) -> dict:
    """Recover content-addressed definitions from saved traces and initial task inputs."""
    definitions = {}
    for run in prepared["runs"]:
        sessions = sorted(Path(run["source"]).glob(f"ace_isolated_s{run['seed']}_appworld*/sessions/*"))
        for session in sessions:
            trace = session / "agent/litellm/trace.jsonl"
            if trace.is_file():
                source_checks[str(trace)] = digest(trace.read_bytes())
                for line in trace.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("record_type") != "definition":
                        continue
                    encoded = json.dumps(row["value"], ensure_ascii=False, sort_keys=True, default=str)
                    if row["id"].split(":", 1)[1] != digest(encoded.encode()):
                        raise ValueError("Recorded trace definition does not match its content hash")
                    definitions[row["id"]] = row["value"]
            metadata_path = session / "session.json"
            if not metadata_path.is_file():
                continue
            metadata = read_json(metadata_path)
            task, context = metadata.get("task"), metadata.get("context")
            if not isinstance(task, str) or not isinstance(context, dict):
                continue
            if any(isinstance(v, dict) and v.get("type") == "image_url" for v in context.values()):
                continue
            # Mirror ACE.start; the reference SHA must match before this can be used.
            text = task + "\n" + "".join(f"\n<{k}>\n{v}\n</{k}>" for k, v in context.items())
            message = {"role": "user", "content": text}
            encoded = json.dumps(message, ensure_ascii=False, sort_keys=True, default=str)
            definitions["messages:" + digest(encoded.encode())] = message
            source_checks[str(metadata_path)] = digest(metadata_path.read_bytes())
    return definitions


def restore_history_bank(output: Path, prepared: dict) -> list[dict]:
    """Decode saved model requests; no benchmark construction or environment calls."""
    from exgentic.integrations.litellm.trace_logger import iter_expanded_trace_rows

    bank_path = output / "history_bank.json"
    if bank_path.exists():
        bank = read_json(bank_path)
        for path, expected in bank["source_checks"].items():
            if digest(Path(path).read_bytes()) != expected:
                raise ValueError("Original training history changed")
        for anchor in bank["anchors"]:
            if digest(Path(anchor["path"]).read_bytes()) != anchor["sha256"]:
                raise ValueError("Frozen history changed")
        return bank["anchors"]
    anchors, audit, source_checks = [], [], {}
    fallback_definitions = None
    for run in prepared["runs"]:
        seed = run["seed"]
        sessions = sorted(Path(run["source"]).glob(f"ace_isolated_s{seed}_appworld*/sessions/*"))
        by_task = {}
        for session in sessions:
            metadata_path = session / "session.json"
            if not metadata_path.is_file():
                continue
            metadata = read_json(metadata_path)
            task = metadata["task_id"]
            if task in by_task:
                raise ValueError("Ambiguous duplicate source training session")
            by_task[task] = session
            source_checks[str(metadata_path)] = digest(metadata_path.read_bytes())
        for task_index, task in enumerate(run["task_ids"]):
            key_prefix = f"s{seed}_{task_index:03d}"
            session = by_task.get(task)
            trace = session / "agent/litellm/trace.jsonl" if session else None
            if trace is None or not trace.is_file():
                audit.append({"seed": seed, "task_id": task, "status": "missing_training_trace"})
                continue
            trace_hash = digest(trace.read_bytes())
            source_checks[str(trace)] = trace_hash
            used = 0
            try:
                rows = list(iter_expanded_trace_rows(trace))
            except KeyError as exc:
                if fallback_definitions is None:
                    fallback_definitions = recorded_trace_definitions(prepared, source_checks)
                try:
                    rows = list(iter_expanded_trace_rows(trace, definition_fallback=fallback_definitions))
                except KeyError as missing:
                    audit.append({"seed": seed, "task_id": task, "status": "missing_training_definition",
                                  "error": str(missing), "source_trace": str(trace)})
                    continue
                audit.append({"seed": seed, "task_id": task, "status": "recovered_training_definitions",
                              "first_missing_definition": str(exc), "source_trace": str(trace),
                              "recovery": "Exact content hashes from saved traces and session task/context"})
            for index, row in enumerate(rows):
                request = row.get("request") or {}
                if not request.get("tools"):
                    continue  # Tool shortlisting is not an acting-policy request.
                try:
                    messages = request["messages"]
                    if (not messages or messages[0]["role"] != "system" or not isinstance(
                            messages[0]["content"], str) or not messages[0]["content"].startswith(
                            "You are an expert agent that completes tasks using available tools.")):
                        raise ValueError("Not an ACE generator request")
                    if not source_model_matches(request["model"], prepared["model"]["model"]):
                        raise ValueError("Training history comes from a different model")
                    reference_calls = normalize_calls(response_message(row["response"]))
                    if not reference_calls:
                        audit.append({"seed": seed, "task_id": task, "trace_row": index,
                                      "status": "no_recorded_api_calls"})
                        continue
                    key = f"{key_prefix}_a{index:04d}"
                    path = output / "histories" / f"{key}.json"
                    data = {"key": key, "seed": seed, "task_id": task,
                            "messages": messages, "tools": request["tools"], "reference_calls": reference_calls,
                            "source_trace": str(trace), "source_trace_row": index,
                            "source_trace_sha256": trace_hash, "source_training_task_index": task_index,
                            "source_system_context_sha256": digest(messages[0]["content"].encode()),
                            "reference_policy": "Recorded task-specific ACE training policy"}
                    save_immutable(path, data)
                    anchors.append({"key": key, "seed": seed, "task_id": task,
                                    "path": str(path), "sha256": digest(path.read_bytes())})
                    used += 1
                    if used == HISTORY_LIMIT:
                        break
                except (ValueError, KeyError, TypeError, IndexError, SyntaxError) as exc:
                    audit.append({"seed": seed, "task_id": task, "trace_row": index,
                                  "status": "invalid_training_action", "error": str(exc)})
            if not used:
                audit.append({"seed": seed, "task_id": task, "status": "no_usable_training_history"})
            write_json(output / "status.json", {"phase": "restoring_training_inputs", "controller_pid": os.getpid(),
                                               "histories": len(anchors), "last_seed": seed,
                                               "last_task_index": task_index, "environment_execution": False,
                                               "updated_at": timestamp()})
    if not anchors:
        raise ValueError("No usable saved training inputs")
    save_immutable(output / "history_restore_audit.json", {"excluded": audit, "history_count": len(anchors),
        "history_limit_per_task": HISTORY_LIMIT, "source_checks": source_checks,
        "task_counts_by_seed": {str(seed): len({a['task_id'] for a in anchors if a['seed'] == seed})
                                for seed in SEEDS}})
    save_immutable(bank_path, {"anchors": anchors, "source_checks": source_checks,
        "source": f"Saved {prepared['model']['model']} ACE AppWorld training requests and observed next API calls",
        "history_limit_per_task": HISTORY_LIMIT, "environment_execution": False,
        "frozen_fields": ["messages excluding first system context", "tools"],
        "reference_policy": "Task-specific learned contexts in the original ACE training"})
    return anchors


def run(output: Path, bundle: Path, base_url: str, workers: int, prepare_only: bool = False,
        model_name: str = MODEL, reasoning_effort: str | None = None) -> None:
    import httpx

    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared = prepare(output, bundle, base_url, workers, model_name, reasoning_effort)
        anchors = restore_history_bank(output, prepared)
        if prepare_only:
            write_json(output / "status.json", {"phase": "prepared", "histories": len(anchors),
                                               "total_probes": len(anchors) * len(POINTS),
                                               "environment_execution": False, "updated_at": timestamp()})
            print(f"PREPARED: {len(anchors)} restored histories, {len(anchors) * len(POINTS)} probes, "
                  "CP0/5/…/50, no environments", flush=True)
            return
        response = httpx.get(base_url.rstrip("/") + "/models", timeout=15)
        response.raise_for_status()
        if model_name.removeprefix("openai/") not in {row["id"] for row in response.json()["data"]}:
            raise ValueError("Endpoint does not serve the learned model")
        runtime = {"model": prepared["model"], "output": str(output), "started_at": timestamp(),
                   "controller_pid": os.getpid(), "controller_start_ticks":
                   Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
                   "workers": workers, "history_limit_per_task": HISTORY_LIMIT, "learning": False,
                   "environment_execution": False}
        write_json(output / "runtime_manifest.json", runtime)
        save_immutable(output / "runtime_manifests" / f"{os.getpid()}.json", runtime)
        print(f"START AppWorld {model_name} ACE: {base_url}, CP0/5/…/50, concurrency={workers}, "
              "restored inputs only", flush=True)
        try:
            contexts = read_json(output / "contexts.json")["contexts"]
            asyncio.run(run_probes(output, anchors, contexts, prepared["model"], workers))
            summary = analyze(output, anchors, prepared["model"], benchmark="AppWorld",
                              training_audit=prepared["training_audit"], initial_reference=False)
            summary["restored_histories"] = read_json(output / "history_restore_audit.json")
            summary["environment_execution"] = False
            write_json(output / "analysis/summary.json", summary)
            write_json(output / "status.json", {"phase": "complete" if not summary["coverage"][
                "failed_or_missing_probes"] else "complete_with_errors", "updated_at": timestamp(),
                "controller_pid": os.getpid(), "coverage": summary["coverage"], "learning": False,
                "environment_execution": False})
        except Exception as exc:
            write_json(output / "status.json", {"phase": "error", "error": str(exc), "updated_at": timestamp()})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "outputs"
                        / "appworld_isolated_checkpoints_cp005_050_ace_premise_20261008")
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument("--workers", type=int, default=150)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive")
    run(args.output.resolve(), args.bundle.resolve(), args.base_url, args.workers,
        args.prepare_only, args.model, args.reasoning_effort)


if __name__ == "__main__":
    main()
