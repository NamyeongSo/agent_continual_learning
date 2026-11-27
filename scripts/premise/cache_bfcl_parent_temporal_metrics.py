# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Cache archived BFCL parent/candidate success and turns without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

CONTEXTS = ("parent", *(f"candidate_{number:02d}" for number in range(1, 6)))
SCHEMA_VERSION = "bfcl_parent_temporal_metrics_v1"


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive_integer_steps(value: object) -> int | None:
    """Reject absent, nonintegral, nonfinite, boolean, and nonpositive turns."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 1 or int(value) != value:
        return None
    return int(value)


def read_context(
    source: Path,
    key: str,
    context: str,
    train_order: list[str],
    expected_rewards: list[float],
    expected_errors: list[bool],
    expected_context_sha256: str,
) -> dict:
    """Join on task identity, never the manifest file index or sorted filename."""
    if len(train_order) != len(set(train_order)) or len(expected_rewards) != len(train_order):
        raise ValueError("Training task identities must be unique and match expected reward count")
    if len(expected_errors) != len(train_order):
        raise ValueError("Expected infrastructure flags must match the training task count")
    directory = source / "results" / key / context / "train"
    by_task = {}
    for path in sorted(directory.glob("*.json")):
        data = path.read_bytes()
        row = json.loads(data)
        if (row.get("checkpoint_key"), row.get("context"), row.get("partition")) != (key, context, "train"):
            raise ValueError(f"Result identity mismatch: {path}")
        task_id = row.get("task_id")
        if task_id not in train_order or task_id in by_task:
            raise ValueError(f"Unexpected or duplicate task identity: {path}")
        if row.get("context_sha256") != expected_context_sha256:
            raise ValueError(f"Frozen context digest mismatch: {path}")
        if isinstance(row.get("score"), bool) or row.get("score") not in (0, 1):
            raise ValueError(f"Archived success must be binary: {path}")
        by_task[task_id] = (path, row, hashlib.sha256(data).hexdigest())
    missing = sorted(set(train_order) - set(by_task))
    if missing:
        raise ValueError(f"Missing archived training results for {key}/{context}: {missing}")
    values = {name: [] for name in ("success", "turns", "infrastructure_flags", "result_paths", "result_sha256")}
    missing_success_turns = []
    counter = Counter()
    for index, task_id in enumerate(train_order):
        path, row, digest = by_task[task_id]
        success = float(row["score"])
        infrastructure = row.get("status") == "error"
        if success != expected_rewards[index] or infrastructure != bool(expected_errors[index]):
            raise ValueError(f"Frozen reward/infrastructure flags disagree: {path}")
        turns = positive_integer_steps(row.get("steps"))
        values["success"].append(success)
        values["turns"].append(turns)
        values["infrastructure_flags"].append(infrastructure)
        values["result_paths"].append(str(path.resolve()))
        values["result_sha256"].append(digest)
        counter["results"] += 1
        counter["successes"] += success == 1
        counter["infrastructure_errors"] += infrastructure
        counter["missing_turns"] += turns is None
        counter["max_steps_20"] += row.get("max_steps") == 20
        if success == 1 and turns is None:
            missing_success_turns.append({"task_id": task_id, "result_path": str(path), "steps": row.get("steps")})
    return values | {"counts": dict(counter), "missing_success_turns": missing_success_turns}


def cache_point(label: str, point: dict, parent: dict) -> dict:
    order = point["train_order"]
    if order != parent["train_order"] or point["key"] != parent["key"]:
        raise ValueError("Frozen parent training order/checkpoint mismatch")
    if not 0 <= point["checkpoint"] < len(order) or point["current"] != [order[point["checkpoint"]]]:
        raise ValueError("Checkpoint/current-task identity mismatch")
    expected = [parent["parent_rewards"], *point["train_rewards"]]
    errors = [parent["parent_error_flags"], *point["train_error_flags"]]
    hashes = [parent["parent_context_sha256"], *point["context_sha256"]]
    contexts = [read_context(Path(point["source"]), point["key"], name, order, rewards, flags, digest)
                for name, rewards, flags, digest in zip(CONTEXTS, expected, errors, hashes, strict=True)]
    mean_parent = sum(contexts[0]["success"]) / len(order)
    if not math.isclose(mean_parent, point["parent_scores"]["full_train"], abs_tol=1e-12):
        raise ValueError("Parent full-training mean disagrees with the frozen target summary")
    for index, context in enumerate(contexts[1:]):
        if not math.isclose(sum(context["success"]) / len(order), point["scores"]["full_train"][index], abs_tol=1e-12):
            raise ValueError("Candidate full-training mean disagrees with the frozen target summary")
    counts = sum((Counter(context["counts"]) for context in contexts), Counter())
    missing = [{"context": name, **row} for name, context in zip(CONTEXTS, contexts, strict=True)
               for row in context["missing_success_turns"]]
    return {"model": label, "key": point["key"], "seed": point["seed"], "checkpoint": point["checkpoint"],
            "current_index": point["checkpoint"], "train_order": order, "contexts": list(CONTEXTS),
            **{field: [context[field] for context in contexts]
               for field in ("success", "turns", "infrastructure_flags")},
            "successful_turns_complete": not missing, "missing_success_turns": missing, "counts": dict(counts),
            "targets": point["scores"], "parent_scores": point["parent_scores"],
            "provenance": {"source": point["source"], "context_sha256": hashes,
                           "result_paths": [context["result_paths"] for context in contexts],
                           "result_sha256": [context["result_sha256"] for context in contexts]}}


def audit_feature_cache(path: Path) -> dict:
    counts = Counter()
    mismatches = []
    for line in path.open():
        group = json.loads(line)
        for trace in group["traces"]:
            if not trace["available"]:
                counts["unavailable"] += 1
                continue
            counts["available"] += 1
            features = trace["features"]
            if features["environment_steps"] == features["reported_steps"]:
                counts["environment_equals_reported_steps"] += 1
            else:
                counts["mismatches"] += 1
                mismatches.append({"group": group["id"], "context": trace["context"],
                                   "environment_steps": features["environment_steps"],
                                   "reported_steps": features["reported_steps"]})
    if not counts["available"] or mismatches:
        raise ValueError("Archived reported turns failed the observed environment-action audit")
    return {"path": str(path.resolve()), "sha256": digest_file(path), "counts": dict(counts),
            "mismatch_examples": mismatches}


def build_cache(raw_path: Path, parent_path: Path, features_path: Path, output: Path, workers: int = 8) -> dict:
    raw = json.loads(raw_path.read_text())
    parents = json.loads(parent_path.read_text())
    parent_models = {model["label"]: {point["key"]: point for point in model["points"]}
                     for model in parents["models"] if model["label"] in ("Qwen", "Gemma")}
    audit = audit_feature_cache(features_path)
    models = []
    aggregate = Counter()
    combined = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for model in raw["models"]:
            label = model["label"]
            if label not in ("Qwen", "Gemma"):
                continue
            jobs = [(label, point, parent_models[label][point["key"]]) for point in model["points"]]
            points = list(executor.map(lambda values: cache_point(*values), jobs))
            models.append({"label": label, "points": points})
            for point in points:
                aggregate.update(point["counts"])
                for hashes in point["provenance"]["result_sha256"]:
                    for digest in hashes:
                        combined.update(digest.encode())
            print(f"Cached {label}: {len(points)} checkpoints, {len(points) * 6 * 50} archived results", flush=True)
    result = {"schema_version": SCHEMA_VERSION, "created_at": datetime.now(UTC).isoformat(),
              "new_model_calls": 0, "new_environment_calls": 0, "contexts": list(CONTEXTS),
              "turn_definition": "Archived result.steps; audited equal to observed environment action count",
              "missing_turn_policy": "None if absent, nonpositive, or nonintegral; only successful turns needed",
              "historical_actor_max_steps": 20, "original_infrastructure_errors_keep_score_zero": True,
              "models": models, "counts": dict(aggregate),
              "successful_turns_complete": all(point["successful_turns_complete"]
                                               for model in models for point in model["points"]),
              "provenance": {"raw_rewards_path": str(raw_path.resolve()), "raw_rewards_sha256": digest_file(raw_path),
                             "parent_cache_path": str(parent_path.resolve()),
                             "parent_cache_sha256": digest_file(parent_path), "turn_count_audit": audit,
                             "result_digest_model_checkpoint_context_train_order": combined.hexdigest(),
                             "script_sha256": digest_file(Path(__file__))}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-rewards", type=Path, required=True)
    parser.add_argument("--parent-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    values = build_cache(args.raw_rewards, args.parent_cache, args.feature_cache, args.output, args.workers)
    print(json.dumps({key: values[key] for key in ("counts", "successful_turns_complete")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
