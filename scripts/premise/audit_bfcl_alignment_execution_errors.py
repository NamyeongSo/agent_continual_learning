# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Audit saved BFCL evaluation failures and selection sensitivity without new inference."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def error_category(row: dict) -> str:
    """Categorize recorded execution errors, separate from ordinary benchmark failures."""
    if row["status"] != "error":
        return "none"
    error = row.get("error", "")
    if "missing assistant content" in error:
        return "missing_assistant_content"
    if ("unexpected tokens remaining in message header" in error or "Harmony" in error
            or ("Unknown role:" in error and "<|channel|>" in error)):
        return "harmony_header_protocol"
    if "Venv service failed to start" in error:
        return "isolated_environment_startup"
    if "Output length failure" in error:
        return "actor_output_length"
    if "Timeout" in error or "timed out" in error:
        return "request_timeout"
    if "Connection refused" in error or "BadGatewayError" in error or "ConnectionError" in error:
        return "server_transport"
    return "other_execution_error"


def compact_error(row: dict) -> str:
    try:
        return str(ast.literal_eval(row.get("error", "{}"))["session_metadata"]["error"])
    except (ValueError, SyntaxError, KeyError, TypeError):
        return str(row.get("error", ""))


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    if x.size < 2 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mean_defined(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def stats(rows: list[dict]) -> dict:
    errors = sum(row["status"] == "error" for row in rows)
    return {
        "evaluations": len(rows), "errors": errors, "error_fraction": errors / len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "session_status_counts": dict(Counter(row["session_status"] for row in rows)),
        "error_categories": dict(Counter(row["error_category"] for row in rows if row["status"] == "error")),
        "mean_score": float(np.mean([row["score"] for row in rows])),
        "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in rows])),
        "median_steps": float(np.median([row["steps"] for row in rows if row["steps"] is not None])),
    }


def write_csv(path: Path, rows: list[dict]):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_result(path: Path) -> dict:
    blob = path.read_bytes()
    row = json.loads(blob)
    if row["status"] not in {"completed", "error"}:
        raise ValueError(f"Unexpected result status: {path}")
    if row["status"] == "error" and float(row["score"]) != 0:
        raise ValueError(f"Error has a nonzero score: {path}")
    keep = ("key", "checkpoint_key", "seed", "checkpoint", "context", "partition", "task_id", "score",
            "status", "session_status", "steps", "wall_seconds", "model_calls", "output_length_retries",
            "started_at", "completed_at")
    return {**{key: row.get(key) for key in keep}, "path": str(path.resolve()),
            "sha256": hashlib.sha256(blob).hexdigest(), "error_category": error_category(row),
            "error_message": compact_error(row)}


def analyze_model(model: dict, workers: int, policies: list[dict]) -> dict:
    paths = sorted(path for point in model["points"]
                   for path in (Path(point["source"]) / "results" / point["key"]).glob("*/*/*.json"))
    if len(paths) != 27500:
        raise ValueError(f"Expected 27500 evaluations, got {len(paths)}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(load_result, paths))
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["sha256"].encode())
    error_rows = [row for row in rows if row["status"] == "error"]
    if len(error_rows) != model["original_errors"]:
        raise ValueError("Saved summary error count differs")
    groups = defaultdict(list)
    for row in rows:
        role = "parent" if row["context"] == "parent" else "candidate"
        for group in (f"{role}/{row['partition']}", f"seed_{row['seed']}"):
            groups[group].append(row)
    contexts = defaultdict(list)
    for row in rows:
        contexts[(row["checkpoint_key"], row["context"], row["partition"])].append(row)
    context_rows = []
    for (key, context, partition), values in sorted(contexts.items()):
        s = stats(values)
        context_rows.append({"model": model["label"], "key": key, "context": context, "partition": partition,
                             **{k: v for k, v in s.items() if not isinstance(v, dict)}})
    cp_rows, sensitivity = [], []
    for point in model["points"]:
        key = point["key"]
        candidate_rows = {partition: [contexts[(key, f"candidate_{i:02d}", partition)] for i in range(1, 6)]
                          for partition in ("train", "val")}
        metrics = {}
        for partition, values in candidate_rows.items():
            errors = np.array([sum(r["status"] == "error" for r in context) / 50 for context in values])
            scores = np.array([np.mean([r["score"] for r in context]) for context in values])
            common = set.intersection(*[{r["task_id"] for r in context if r["status"] != "error"}
                                        for context in values])
            complete_scores = np.array([np.mean([r["score"] for r in context if r["task_id"] in common])
                                        for context in values]) if common else np.full(5, np.nan)
            metrics[partition] = {"errors": errors, "scores": scores, "common": common,
                                  "complete_scores": complete_scores}
        train, val = metrics["train"], metrics["val"]
        cp_rows.append({
            "model": model["label"], "key": key, "seed": point["seed"], "checkpoint": point["checkpoint"],
            "train_error_fraction": float(train["errors"].mean()),
            "val_error_fraction": float(val["errors"].mean()),
            "train_candidate_error_range": float(np.ptp(train["errors"])),
            "val_candidate_error_range": float(np.ptp(val["errors"])),
            "common_train_tasks_no_candidate_error": len(train["common"]),
            "common_val_tasks_no_candidate_error": len(val["common"]),
            "rho_train_val_error_rates": pearson(train["errors"], val["errors"]),
            "rho_train_error_val_score": pearson(train["errors"], val["scores"]),
            "rho_full_train_full_val_score": pearson(train["scores"], val["scores"]),
            "rho_common_train_common_val_score": pearson(train["complete_scores"], val["complete_scores"]),
        })
        for policy in policies:
            if policy["model"] != model["label"] or policy["key"] != key:
                continue
            probability = np.array([policy[f"candidate_{i}_probability"] for i in range(1, 6)])
            target = val["complete_scores"]
            sensitivity.append({
                "model": model["label"], "key": key, "seed": point["seed"], "policy": policy["policy"],
                "matched36": policy["matched36"], "common_val_tasks": len(val["common"]),
                "original_validation_lift": float(probability @ val["scores"] - val["scores"].mean()),
                "same_choices_common_val_lift": float(probability @ target - target.mean()),
                "selected_val_error_fraction_minus_proposal": float(probability @ val["errors"]
                                                                      - val["errors"].mean()),
            })
    examples = {}
    for row in error_rows:
        examples.setdefault(row["error_category"], {"path": row["path"], "error": row["error_message"]})
    return {"label": model["label"], "source_result_digest_sorted_path_order": digest.hexdigest(),
            "overall": stats(rows), "groups": {key: stats(value) for key, value in groups.items()},
            "examples": examples, "context_rows": context_rows, "checkpoint_rows": cp_rows,
            "selection_sensitivity": sensitivity,
            "most_common_exact_errors": Counter(row["error_message"] for row in error_rows).most_common(10)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extremes", type=Path)
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    blob, policy_blob = args.source.read_bytes(), args.hypotheses.read_bytes()
    source, hypotheses = json.loads(blob), json.loads(policy_blob)
    policies = [p for p in hypotheses["checkpoint_scores"]
                if p["policy"] in {"uniform_a050", "similar_a050", "near6_uniform3_a050"}]
    if args.extremes:
        extra = json.loads(args.extremes.read_text())
        policies += [{**p, "policy": f"near{p['near']}_far{p['far']}_{p['weight']}"}
                     for p in extra["checkpoint_scores"]
                     if p["weight"] == "period_mean" and p["near"] in {0, 5, 9}]
    models = []
    for model in source["models"]:
        print(f"Auditing {model['label']}...", flush=True)
        models.append(analyze_model(model, args.workers, policies))
        print(json.dumps(models[-1]["overall"]), flush=True)
    summary = {
        "created_at": datetime.now(UTC).isoformat(), "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(blob).hexdigest(), "hypotheses": str(args.hypotheses.resolve()),
        "hypotheses_sha256": hashlib.sha256(policy_blob).hexdigest(), "new_model_calls": 0,
        "scope": "All 50 checkpoints/model, five candidate train+val50 and parent train50.",
        "sensitivity_limitations": "Post hoc complete-case diagnostic, not an unbiased correction. Choices stay fixed; "
        "validation tasks failing for any candidate are removed. Changed support can be systematically easier and "
        "model/context dependent. Ordinary unsuccessful/limit_reached trajectories are retained.",
        "models": models,
    }
    if args.extremes:
        summary["extremes_source"] = str(args.extremes.resolve())
        summary["extremes_sha256"] = hashlib.sha256(args.extremes.read_bytes()).hexdigest()
    if args.contexts:
        with args.contexts.open() as handle:
            context_metadata = list(csv.DictReader(handle))
        duplicate_rows = []
        for model in models:
            metrics = {(r["key"], r["context"], r["partition"]): r for r in model["context_rows"]}
            by_playbook = defaultdict(list)
            for row in context_metadata:
                if row["model"] == model["label"]:
                    by_playbook[(row["key"], row["playbook_sha256"])].append(row)
            for (key, digest), contexts in by_playbook.items():
                if len(contexts) < 2:
                    continue
                for partition in ("train", "val"):
                    rows = [metrics[(key, f"candidate_{int(r['candidate']):02d}", partition)] for r in contexts]
                    duplicate_rows.append({
                        "model": model["label"], "key": key, "playbook_sha256": digest,
                        "candidate_count": len(rows), "partition": partition,
                        "score_range": max(r["mean_score"] for r in rows) - min(r["mean_score"] for r in rows),
                        "error_fraction_range": max(r["error_fraction"] for r in rows)
                        - min(r["error_fraction"] for r in rows),
                    })
        summary["duplicate_context_source"] = str(args.contexts.resolve())
        summary["duplicate_context_groups"] = duplicate_rows
        write_csv(args.output / "duplicate_context_error_ranges.csv", duplicate_rows)
    (args.output / "error_audit_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name in ("context_rows", "checkpoint_rows", "selection_sensitivity"):
        write_csv(args.output / f"{name}.csv", [row for model in models for row in model[name]])
    lines = ["# BFCL saved execution-error audit", "", summary["scope"], "", summary["sensitivity_limitations"], ""]
    for model in models:
        lines += [f"## {model['label']}", "", f"Recorded errors: {model['overall']['errors']}/27500.",
                  "", f"Categories: `{model['overall']['error_categories']}`.",
                  "", f"Session status: `{model['overall']['session_status_counts']}`.", ""]
        for category, example in model["examples"].items():
            lines += [f"- {category}: `{example['error']}`", f"  Source: `{example['path']}`."]
        cp = model["checkpoint_rows"]
        lines += ["", "Common validation tasks without a recorded error for any of five candidates: "
                  f"mean {np.mean([r['common_val_tasks_no_candidate_error'] for r in cp]):.2f}/50.",
                  "", "Mean within-checkpoint correlation of candidate train/validation error rates: "
                  f"{mean_defined([r['rho_train_val_error_rates'] for r in cp])}.", ""]
        lines += ["| Fixed training-only selection rule | Original val50 lift (pp) | Common-valid val lift (pp) |",
                  "| --- | ---: | ---: |"]
        for policy in dict.fromkeys(row["policy"] for row in model["selection_sensitivity"]):
            rows = [r for r in model["selection_sensitivity"] if r["policy"] == policy]
            before = 100 * np.mean([r["original_validation_lift"] for r in rows])
            after = 100 * np.mean([r["same_choices_common_val_lift"] for r in rows])
            lines.append(f"| {policy} | {before:+.3f} | {after:+.3f} |")
        lines += [""]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
