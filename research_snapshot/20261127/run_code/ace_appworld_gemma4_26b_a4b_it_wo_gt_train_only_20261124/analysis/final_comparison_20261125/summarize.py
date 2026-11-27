# SPDX-License-Identifier: Apache-2.0
"""Validate saved Gemma Train receipts and compare actual selected reflection policies."""

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.ace import schedule_gemma4_train_only as schedule  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[2]
TARGET = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    provenance = {}
    sibling = OUTPUT.parent / "ace_appworld_gemma4_31b_it_wo_gt_train_only_20261124"
    for cohort in (OUTPUT, sibling):
        for path, expected in read(cohort / "input_hashes.json").items():
            assert digest(Path(path)) == expected, path
        for mode in ("thinking", "nonthinking"):
            for phase in ("reflection", "accumulation"):
                folder = cohort / mode / phase
                for relative, expected in read(folder / "source_hashes.json").items():
                    path = folder / "source_snapshot" / relative
                    assert digest(path) == expected, str(path)
    assert schedule.predecessor_complete(OUTPUT)
    training = read(OUTPUT / "control/training_plan.json")
    schedule.study.training.require_all_trained(training)
    assert len(training) == 30
    queue = read(OUTPUT / "control/evaluation_queue.json")
    results = defaultdict(list)
    for item in queue:
        job = item["job"]
        assert job["arm"] == "without_gt" and job["phase"] == "train_final"
        path = Path(item["root"]) / "stage3" / job["id"] / "result.json"
        row = read(path)
        assert type(row["success"]) is bool and math.isfinite(row["pct"])
        assert 0 <= row["pct"] <= 1
        results[item["sampler"], row["seed"], row["condition_id"]].append(row)
        provenance[str(path)] = digest(path)
    per_seed = read(OUTPUT / "train_final_by_seed.json")
    for record in per_seed:
        key = record["mode"], record["seed"], record["condition_id"]
        rows = results[key]
        protocol = read(OUTPUT / record["mode"] / "reflection/protocol.json")
        assert len(rows) == 50 and {r["task_id"] for r in rows} == set(protocol["task_ids"])
        assert math.isclose(record["pct"], mean(r["pct"] for r in rows))
        assert math.isclose(record["tgc"], mean(r["success"] for r in rows))
    assert len(results) == 30 and len(queue) == 1500
    initial_path = OUTPUT / "analysis/initial_performance_20261124/initial_metrics.json"
    independent_path = OUTPUT / "analysis/available_results_20261125/independent_by_seed.json"
    initial = {r["mode"]: r for r in read(initial_path)}
    independent = {(r["mode"], r["seed"], r["strategy"]): r for r in read(independent_path)}
    accumulated = {(r["mode"], r["seed"], r["strategy"]): r for r in per_seed}
    table = []
    metrics = []
    for group in read(OUTPUT / "train_final_mean_std.json"):
        mode, strategy = group["mode"], group["strategy"]
        rows = [accumulated[mode, seed, strategy] for seed in (42, 43, 44)]
        candidates = [independent[mode, seed, strategy] for seed in (42, 43, 44)]
        row = {"mode": mode, "strategy": strategy, "seeds": "42,43,44", "evaluation_tasks": 50,
               "learned_tasks": candidates[0]["learned_tasks"]}
        for metric in ("pct", "tgc"):
            baseline = initial[mode][f"{metric}_percent"]
            observed = [r[metric] * 100 for r in rows]
            expected = [r[f"independent_{metric}"] * 100 for r in candidates]
            paired_deltas = [a - b for a, b in zip(observed, expected, strict=True)]
            assert math.isclose(mean(observed), group[f"{metric}_mean"] * 100)
            assert math.isclose(stdev(observed), group[f"{metric}_std"] * 100)
            row.update({f"initial_{metric}_percent": baseline,
                        f"independent_{metric}_mean_percent": mean(expected),
                        f"independent_{metric}_std_pp": stdev(expected),
                        f"accumulated_{metric}_mean_percent": mean(observed),
                        f"accumulated_{metric}_std_pp": stdev(observed),
                        f"accumulated_minus_initial_{metric}_pp": mean(observed) - baseline,
                        f"accumulated_minus_independent_{metric}_mean_pp": mean(paired_deltas),
                        f"accumulated_minus_independent_{metric}_std_pp": stdev(paired_deltas)})
            metrics.append({"name": f"{mode} {strategy} accumulated {metric.upper()}",
                            "value": f"{mean(observed):.2f} ± {stdev(observed):.2f}",
                            "unit": "%", "split": "train50",
                            "source": str(TARGET / "comparison.csv") +
                                      "; mean and sample SD of seed42/43/44 Train means; 50 official outcomes per seed"})
        table.append(row)
    with (TARGET / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (TARGET / "comparison.json").write_text(json.dumps(table, indent=2) + "\n")
    for path in (initial_path, independent_path, OUTPUT / "train_final_by_seed.json",
                 OUTPUT / "train_final_mean_std.json", OUTPUT / "control/evaluation_queue.json",
                 OUTPUT / "completed.json", Path(__file__)):
        provenance[str(path)] = digest(path)
    receipt = {"checked_at": datetime.now(UTC).isoformat(), "contexts": 30, "train_evaluations": 1500,
               "source_bindings_valid": True, "all_checkpoint_texts_valid": True,
               "sampling_or_policy_changed": False, "no_bwt_or_heldout_evaluation": True,
               "population": "same Train50; 3 seeds; independent score from selection bank, not new repeats",
               "standard_deviation": "sample SD across seed-level Train50 averages, ddof=1",
               "source_sha256": provenance}
    (TARGET / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    patch = {"status": "completed", "metrics": metrics,
             "summary": "26B 전 단계 완료: 초기500, Reflection/Update500, 후보1500, Context30, Train1500. "
                        "Thinking best PCT90.82±2.87/TGC72.67±6.11; Non-thinking best_adv PCT87.42±1.65/TGC61.33±1.15.",
             "next_steps": "예약대로 31B 자동 전환 완료; 31B Train-only 실험을 2시간마다 점검."}
    (TARGET / "journal_patch.json").write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
