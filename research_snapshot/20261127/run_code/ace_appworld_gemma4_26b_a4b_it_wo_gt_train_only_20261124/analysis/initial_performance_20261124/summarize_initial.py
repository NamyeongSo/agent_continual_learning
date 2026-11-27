# SPDX-License-Identifier: Apache-2.0
"""Compute the complete Gemma initial Train50 baseline from five official rollouts per task."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from math import comb
from pathlib import Path
from statistics import mean, pstdev

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[1]
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ace.run_reflection_accumulation import exact_reward  # noqa: E402

POLICY = "082147d2215b74ce52131e6ef30baa342aeaf4099e2e475f0bbb97fed9fe195d"


def csv_file(filename, rows):
    with (OUT / filename).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    summary, task_rows, source_hashes = [], [], {}
    task_set = None
    for mode in ("thinking", "nonthinking"):
        directory = RUN / mode / "reflection"
        protocol = json.loads((directory / "protocol.json").read_text())
        tasks = set(protocol["task_ids"])
        assert len(tasks) == 50
        if task_set is None:
            task_set = tasks
        assert tasks == task_set
        assert protocol["model"] == "openai/google/gemma-4-26B-A4B-it"
        assert protocol["history_policy_version"] == "gemma-tool-reasoning-history-v1"
        assert protocol["turn_limit_scoring"] == "official_pct_at_limit"
        context_sha = hashlib.sha256(protocol["initial_playbook"].encode()).hexdigest()
        queue = json.loads((directory / "stage1_queue.json").read_text())
        assert len(queue) == 250
        groups = defaultdict(list)
        rows = []
        for job in queue:
            path = directory / "stage1" / job["id"] / "result.json"
            if not path.exists():
                raise RuntimeError(f"Initial baseline is incomplete: {mode}/{job['id']}")
            raw = path.read_bytes()
            row = json.loads(raw)
            assert all(row.get(k) == v for k, v in job.items())
            assert type(row["success"]) is bool
            assert row["initial_context_sha256"] == context_sha
            assert row["appworld_policy_sha256"] == POLICY
            exact_reward(row)
            source_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            groups[row["task_id"]].append(row)
            rows.append(row)
        assert len(rows) == 250 and set(groups) == tasks
        for task, group in sorted(groups.items()):
            assert len(group) == 5 and {r["source_repeat"] for r in group} == set(range(5))
            values = [float(exact_reward(r)) for r in group]
            task_rows.append({
                "mode": mode, "task_id": task, "rollouts": 5, "mean_pct": mean(values),
                "pct_population_std": pstdev(values), "successful_rollouts": sum(r["success"] for r in group),
                "tgc": mean(r["success"] for r in group),
            })
        successes = [sum(r["success"] for r in group) for group in groups.values()]
        summary.append({
            "mode": mode, "tasks": 50, "rollouts": 250,
            "pct_percent": 100 * mean(float(exact_reward(r)) for r in rows),
            "tgc_percent": 100 * mean(r["success"] for r in rows),
            "pass_at_5_percent": 100 * mean(s > 0 for s in successes),
            "pass_power_5_percent": 100 * mean(s == 5 for s in successes),
            "pass_at_3_percent": 100 * mean(1 - comb(5 - s, 3) / comb(5, 3) for s in successes),
            "pass_power_3_percent": 100 * mean(comb(s, 3) / comb(5, 3) for s in successes),
            "successful_rollouts": sum(successes), "tasks_with_any_success": sum(s > 0 for s in successes),
            "tasks_with_all_success": sum(s == 5 for s in successes),
            "turn_limited_rollouts": sum(bool(r.get("turn_limit_reached")) for r in rows),
            "budget_failed_rollouts": sum(bool(r.get("budget_failure")) for r in rows),
            "budget_failure_reasons_json": json.dumps(dict(Counter(
                r["budget_failure"]["reason"] for r in rows if r.get("budget_failure")
            )), sort_keys=True),
        })
    assert len(source_hashes) == 500
    csv_file("initial_metrics.csv", summary)
    csv_file("initial_task_metrics.csv", task_rows)
    (OUT / "initial_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    provenance = {
        "generated_at": datetime.now(UTC).isoformat(), "run": str(RUN), "valid_initial_rollouts": 500,
        "pct": "Mean official individual-test pass fraction over Train50 x 5 rollouts; "
               "retain protocol-defined budget failure zeros",
        "tgc": "Mean recorded official boolean success and protocol-defined budget failure false values; "
               "no PCT thresholding",
        "pass_at_5": "Fraction of 50 tasks with at least one official success in the five rollouts",
        "pass_power_5": "Fraction of 50 tasks with all five rollouts officially successful",
        "pass_at_3_and_pass_power_3": "Combinatorial estimate over the 10 subsets of three saved rollouts",
        "history_policy": "Gemma tool reasoning history v1; earlier invalid nine-rollout cohort excluded",
        "actor_calls": "No additional inference for this analysis",
        "source_sha256": source_hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
