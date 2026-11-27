# SPDX-License-Identifier: Apache-2.0
"""Export the frozen one-epoch study's numeric results, without trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CP = ROOT / "outputs/qwen_cp10_original_appworld_noalign_dual100_20261105"
SHIFT = ROOT / "outputs/qwen_cp10_policy_shift_own_history_temp0_20261105"
MAIN = ROOT / "outputs/qwen_main_original_appworld_peace_no_alignment_20261105"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_checkpoints(stage):
    with sqlite3.connect(CP.joinpath("queue.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        specs = {row["id"]: json.loads(row["spec"]) for row in db.execute("SELECT * FROM streams")}
        assert len(specs) == 54
        assert db.execute("SELECT count(*) FROM stream_invalidations").fetchone()[0] == 0
        checkpoints = db.execute("SELECT * FROM checkpoints ORDER BY stream,point").fetchall()
        physical = {}
        for row in db.execute("SELECT id,cp,split,task_id,status,attempts,result FROM tasks ORDER BY id"):
            value = json.loads(row["result"]) if row["result"] else {}
            score = value.get("score") if row["status"] == "completed" else None
            if score is not None:
                assert isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1
            error = value.get("error") or ""
            entry = {
                "physical_task_id": row["id"],
                "canonical_checkpoint_id": row["cp"],
                "split": row["split"],
                "task_id": row["task_id"],
                "status": row["status"],
                "score": score,
                "partial_score": value.get("partial_score") if row["status"] == "completed" else None,
                "attempts": row["attempts"],
                "error_category": "timeout" if "timeout" in error.lower() else error.split(":", 1)[0],
                "metric": value.get("score_metric") or "accuracy",
            }
            physical.setdefault((row["cp"], row["split"]), []).append(entry)
        logical, scores, mappings = [], [], []
        for cp in checkpoints:
            spec = specs[cp["stream"]]
            common = {
                "stream": spec["id"],
                "variant": spec["run_variant"],
                "benchmark": spec["benchmark"],
                "seed": spec["seed"],
                "checkpoint": cp["point"],
                "checkpoint_id": cp["id"],
                "canonical": cp["canonical"],
            }
            mappings.append({**common, "content_sha256": cp["sha"], "reused": cp["id"] != cp["canonical"]})
            assert spec["train_total"] == (30 if spec["benchmark"] == "terminalbench2" else 50)
            for split in ("train", "val", "test"):
                tasks = physical[cp["canonical"], split]
                assert {row["task_id"] for row in tasks} == set(spec["splits"][split])
                measured = [row["score"] for row in tasks if row["status"] == "completed"]
                scores.append(
                    {
                        **common,
                        "split": split,
                        "score": statistics.mean(measured) if measured else None,
                        "measured": len(measured),
                        "total": len(tasks),
                        "errors": sum(row["status"] == "error" for row in tasks),
                    }
                )
                logical.extend({**common, **row} for row in tasks)
        physical_rows = sorted(
            (row for rows in physical.values() for row in rows), key=lambda row: row["physical_task_id"]
        )
    write_csv(stage / "checkpoint/task_scores_logical.csv", logical)
    write_csv(stage / "checkpoint/task_scores_physical.csv", physical_rows)
    write_csv(stage / "checkpoint/checkpoint_aliases.csv", mappings)
    write_csv(stage / "checkpoint/checkpoint_scores_verified.csv", scores)
    stream_rows, split_rows, seen_splits = [], [], set()
    for spec in specs.values():
        stream_rows.append(
            {
                "stream": spec["id"],
                "variant": spec["run_variant"],
                "method": spec["method"],
                "benchmark": spec["benchmark"],
                "seed": spec["seed"],
                "train_total": spec["train_total"],
                "model": spec["model"],
                "temperature": spec["temperature"],
                "split_sha256": spec["split_sha256"],
                "appworld_policy_sha256": spec.get("appworld_policy_sha256", ""),
                **spec["actor_limits"],
            }
        )
        for split, ids in spec["splits"].items():
            key = (spec["benchmark"], spec["seed"], split)
            if key not in seen_splits:
                seen_splits.add(key)
                split_rows.extend(
                    {
                        "benchmark": spec["benchmark"],
                        "seed": spec["seed"],
                        "split": split,
                        "ordinal": ordinal,
                        "task_id": task,
                    }
                    for ordinal, task in enumerate(ids)
                )
    write_csv(stage / "checkpoint/streams.csv", stream_rows)
    write_csv(stage / "checkpoint/split_membership.csv", split_rows)
    with (MAIN / "learning_curves/checkpoint_seed_scores.csv").open() as stream:
        existing = list(csv.DictReader(stream))

    def key(row):
        return (row["variant"], row["benchmark"], int(row["seed"]), int(row["checkpoint"]), row["split"])

    lookup = {key(row): row for row in scores}
    assert len(existing) == len(scores)
    for row in existing:
        current = lookup[key(row)]
        assert math.isclose(float(row["score"]), current["score"], abs_tol=1e-12)
        assert all(int(row[name]) == current[name] for name in ("measured", "total", "errors"))
        assert row["canonical"] == current["canonical"]
    return {
        "streams": len(specs),
        "logical_checkpoints": len(checkpoints),
        "reused_checkpoint_rows": sum(row["reused"] for row in mappings),
        "physical_contexts": len(checkpoints) - sum(row["reused"] for row in mappings),
        "physical_task_status_counts": dict(Counter(row["status"] for row in physical_rows)),
        "logical_task_rows": len(logical),
        "logical_task_status_counts": dict(Counter(row["status"] for row in logical)),
        "checkpoint_seed_split_points": len(scores),
        "matches_original_figure_seed_scores": True,
    }


def export(output):
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "epoch1_checkpoint_policy_shift"
    if stage.exists():
        raise ValueError("Refuse to overwrite an existing export")
    stage.mkdir()
    provenance = []

    def copy(source, target):
        target = stage / target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        provenance.append({"file": str(target.relative_to(stage)), "source": str(source), "sha256": digest(source)})

    cp_summary = export_checkpoints(stage)
    for name in ("checkpoint_seed_scores.csv", "checkpoint_seed_mean_scores.csv", "checkpoint_split_correlations.csv"):
        copy(MAIN / "learning_curves" / name, "checkpoint/" + name)
    for name in ("checkpoint_learning_curves_seed_mean", "checkpoint_split_correlations"):
        for suffix in ("pdf", "png"):
            copy(MAIN / "learning_curves" / f"{name}.{suffix}", f"figures/checkpoint/{name}.{suffix}")
    copy(MAIN / "learning_curves/figure_manifest.json", "metadata/checkpoint_figure_manifest.json")
    for directory in ("stability_gap_cp10", "wc_acc_cp10"):
        for source in sorted((MAIN / directory).iterdir()):
            if source.suffix in (".csv", ".md"):
                copy(source, f"checkpoint/derived/{directory}/{source.name}")
            elif source.name in (
                "stability_gap_cp10.png",
                "stability_gap_cp10.pdf",
                "observed_wc_acc_cp10.png",
                "observed_wc_acc_cp10.pdf",
            ):
                copy(source, f"figures/checkpoint/{source.name}")
    copy(MAIN / "offline_early_selections.csv", "checkpoint/derived/offline_early_selections.csv")
    latest = Path(load(SHIFT / "analysis/latest_plot.json")["output"])
    original_manifest = load(latest / "manifest.json")
    policy_files = (
        "distance_metrics.csv",
        "pairwise_metrics.csv",
        "pairwise_task_metrics.csv",
        "correlations.csv",
        "coverage_counts.csv",
        "cp0_repeat_control_summary.csv",
        "coverage.json",
        "summary.json",
    )
    for name in policy_files:
        if name in original_manifest["files"]:
            assert digest(latest / name) == original_manifest["files"][name]
        copy(latest / name, "policy_shift/" + name)
    copy(SHIFT / "analysis/repeatability_metrics.csv", "policy_shift/repeatability_metrics.csv")
    copy(latest / "manifest.json", "metadata/policy_shift_figure_manifest.json")
    copy(latest / "validation.json", "metadata/policy_shift_original_validation.json")
    for name in ("policy_shift_delta_cp", "policy_shift_delta_cp_f1"):
        for suffix in ("png", "pdf"):
            copy(latest / f"{name}.{suffix}", f"figures/policy_shift/{name}.{suffix}")
    safe_policy_keys = (
        "model",
        "temperature",
        "checkpoint_mode",
        "comparison",
        "history_bank",
        "aggregation",
        "history_sampling_temperature",
        "environment_execution",
        "learning",
        "per_gpu_query_limit",
    )
    original_plan = load(SHIFT / "plan.json")
    protocol = {
        "study": "Original one-epoch six-setting study only; no epoch-two data",
        "method_labels": original_manifest["method_labels"],
        "seeds": [42, 43, 44],
        "benchmarks": ["bfcl", "appworld", "terminalbench2"],
        "checkpoint_points": {
            "bfcl": [0, 10, 20, 30, 40, 50],
            "appworld": [0, 10, 20, 30, 40, 50],
            "terminalbench2": [0, 10, 20, 30],
        },
        "checkpoint_evaluation_temperature": 0.7,
        "policy_shift": {key: original_plan[key] for key in safe_policy_keys},
        "journal_ids": {
            "checkpoint": "exp-20261104-783f6a10",
            "policy_shift": "exp-20261105-391724e9",
            "main_table": "exp-20261104-92f935c2",
        },
        "excluded": [
            "trajectories",
            "history prefix text",
            "action outputs",
            "learned contexts/prompts",
            "raw databases",
            "environment variables",
            "credentials",
            "epoch-two results",
        ],
    }
    write_json(stage / "metadata/protocol.json", protocol)
    validation = {
        "checkpoint": cp_summary,
        "policy_shift": {
            key: original_manifest[key]
            for key in (
                "job_counts",
                "common_history_scope",
                "common_history_records",
                "cp_pair_rows",
                "task_pair_metric_rows",
            )
        },
        "original_policy_plot_file_hashes_verified": True,
        "new_model_calls": 0,
    }
    write_json(stage / "metadata/validation.json", validation)
    copy(Path(__file__), "scripts/export_epoch1_results.py")
    write_json(stage / "metadata/source_files.json", provenance)
    return stage, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage, summary = export(args.output.resolve())
    print(json.dumps({"stage": str(stage), "validation": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
