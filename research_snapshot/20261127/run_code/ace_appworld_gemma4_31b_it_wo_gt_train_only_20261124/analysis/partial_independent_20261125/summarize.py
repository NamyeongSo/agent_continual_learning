# SPDX-License-Identifier: Apache-2.0
"""Summarize the paused candidate bank without treating missing evaluations as failures."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from statistics import mean

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[1]
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ace.run_gemma4_train_only import validate_reflection  # noqa: E402
from scripts.ace.run_reflection_accumulation import exact_reward  # noqa: E402

SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def csv_file(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    paused = read(RUN / "status.json")
    assert paused["state"] == "paused" and (RUN / "STOP").exists()
    summaries, coverage, candidates, selected = [], [], [], []
    total_completed, total_used = 0, 0
    for mode in ("thinking", "nonthinking"):
        folder = RUN / mode / "reflection"
        protocol = read(folder / "protocol.json")
        tasks = protocol["task_ids"]
        assert len(tasks) == len(set(tasks)) == 50 and protocol["arms"] == ["without_gt"]
        initial = defaultdict(list)
        for job in read(folder / "stage1_queue.json"):
            row = validate_reflection({"root": str(folder), "job": job})
            read(folder / "stage1" / job["id"] / "result.json")
            initial[row["task_id"]].append(row)
        assert set(initial) == set(tasks) and all(len(rows) == 5 for rows in initial.values())
        evaluations = defaultdict(dict)
        for job in read(folder / "stage3_queue.json"):
            path = folder / "stage3" / job["id"] / "result.json"
            if not path.exists():
                continue
            row = validate_reflection({"root": str(folder), "job": job})
            read(path)
            assert row["arm"] == "without_gt" and row["task_id"] in tasks
            key = row["task_id"], row["source_repeat"]
            assert row["evaluation_repeat"] not in evaluations[key]
            evaluations[key][row["evaluation_repeat"]] = row
        total_completed += sum(len(rows) for rows in evaluations.values())
        complete = defaultdict(list)
        for task in tasks:
            completed_candidates = 0
            for index in range(5):
                rows = evaluations.get((task, index), {})
                full = set(rows) == {0, 1, 2}
                if full:
                    pct = sum((exact_reward(r) for r in rows.values()), Fraction()) / 3
                    tgc = Fraction(sum(r["success"] for r in rows.values()), 3)
                    complete[task].append({"index": index, "pct": pct, "tgc": tgc})
                    completed_candidates += 1
                    total_used += 3
                candidates.append({"mode": mode, "task_id": task, "source_repeat": index,
                                   "completed_evaluations": len(rows), "included": full,
                                   "pct": float(pct) if full else "", "tgc": float(tgc) if full else ""})
            coverage.append({"mode": mode, "task_id": task, "fully_evaluated_candidates": completed_candidates,
                             "planned_candidates": 5,
                             "completed_rollouts": sum(len(evaluations.get((task, i), {})) for i in range(5))})
        eligible = [t for t in tasks if len(complete[t]) >= 2]
        assert len(eligible) == 50, "Use an explicitly matched baseline if task coverage changes"
        baseline = {t: {"pct": sum((exact_reward(r) for r in initial[t]), Fraction()) / 5,
                        "tgc": Fraction(sum(r["success"] for r in initial[t]), 5)} for t in eligible}
        counts = Counter(len(complete[t]) for t in eligible)
        baseline_means = {m: 100 * mean(float(baseline[t][m]) for t in eligible) for m in ("pct", "tgc")}
        for strategy in ("best", "random", "worst"):
            scores = []
            for task in eligible:
                group = complete[task]
                target = (max if strategy == "best" else min)(c["pct"] for c in group)
                pool = group if strategy == "random" else [c for c in group if c["pct"] == target]
                values = {metric: sum((c[metric] for c in pool), Fraction()) / len(pool)
                          for metric in ("pct", "tgc")}
                scores.append(values)
                selected.append({"mode": mode, "task_id": task, "strategy": strategy,
                                 "available_candidates": len(group), "selection_pool": json.dumps(
                                     sorted(c["index"] for c in pool)),
                                 "pool_size": len(pool), "pct": float(values["pct"]),
                                 "tgc": float(values["tgc"]),
                                 "initial_pct": float(baseline[task]["pct"]),
                                 "initial_tgc": float(baseline[task]["tgc"])})
            summary = {"model": "google/gemma-4-31B-it", "mode": mode, "strategy": strategy,
                       "arm": "without_gt", "tasks": len(eligible),
                       "fully_evaluated_candidates": sum(len(complete[t]) for t in eligible),
                       "tasks_with_2_candidates": counts[2], "tasks_with_3_candidates": counts[3],
                       "full_G5_tasks": counts[5], "status": "provisional_observed_candidate_pool",
                       **{f"{m}_percent": 100 * mean(float(v[m]) for v in scores) for m in ("pct", "tgc")},
                       **{f"initial_{m}_percent": baseline_means[m] for m in ("pct", "tgc")}}
            summary.update({f"{m}_gain_pp": summary[f"{m}_percent"] - baseline_means[m] for m in ("pct", "tgc")})
            summaries.append(summary)
    assert total_completed == paused["completed"] == 778
    assert total_used == 753 and len(selected) == 300
    assert all(r["full_G5_tasks"] == 0 for r in summaries)
    csv_file("summary.csv", summaries)
    csv_file("task_coverage.csv", coverage)
    csv_file("candidate_coverage_and_scores.csv", candidates)
    csv_file("strategy_task_scores.csv", selected)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    provenance = {"generated_at": datetime.now(UTC).isoformat(), "run_state": "paused",
                  "measured_rollouts": total_completed, "used_rollouts": total_used,
                  "excluded_incomplete_candidate_rollouts": total_completed - total_used,
                  "candidate_repeats_required": 3, "tasks_per_mode": 50, "fully_evaluated_G5_tasks": 0,
                  "best_worst": "Exact mean PCT extrema among currently complete candidates; uniform exact ties",
                  "random": "Analytic uniform expectation over currently complete candidates, task-balanced",
                  "uncertainty": "Point expectations, not seed-level mean +/- SD; not final G5 selection scores",
                  "missing_outcomes": "Excluded, never scored as zero; incomplete candidates not used for selection",
                  "new_model_calls": 0, "source_sha256": SOURCES,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    assert json.loads((RUN / "status.json").read_bytes())["state"] == "paused" and (RUN / "STOP").exists()
    print(json.dumps({"summary": summaries, "measured_rollouts": total_completed, "used_rollouts": total_used}))


if __name__ == "__main__":
    main()
