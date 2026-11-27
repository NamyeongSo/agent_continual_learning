# SPDX-License-Identifier: Apache-2.0
"""Audit saved official AppWorld answer-test cases across completed ACE Train evaluations."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

from scripts.ace.plot_worst_reflection_repeats import ARMS, ROOT, RUNS, SEEDS

OUTPUT = Path(__file__).resolve().parent
MODELS = {
    "qwen38": RUNS,
    "qwen36": {mode: ROOT / "outputs/ace_appworld_qwen36_add_only_20261118" / mode / "accumulation" for mode in RUNS},
    "qwen35": {mode: OUTPUT.parents[1] / mode / "accumulation" for mode in RUNS},
}
NULL = re.compile(r"""==\s*(['"])null\1""")
SUMMARY_WORDS = re.compile(
    r"\b(?:complet|process|successfully|order|appl|forward|creat|sent|purchas|relabel|renam|mark|verif|add|updat)",
    re.I,
)


def read(path):
    return json.loads(Path(path).read_text())


def audit(rows, policy):
    counts = Counter()
    findings = []
    for row in rows:
        assert row["appworld_policy_sha256"] == policy
        assert type(row["success"]) is bool
        counts["rollouts"] += 1
        counts["successes"] += int(row["success"])
        counts["turn_caps"] += int(bool(row.get("turn_limit_reached")))
        if row.get("budget_failure"):
            counts["budget_failures"] += 1
            continue
        path = Path(row["session_dir"]) / "benchmark" / "scores.json"
        if not path.is_file():
            counts["missing_official_tests"] += 1
            continue
        tracker = read(path)["test_tracker"]
        assert tracker["success"] == row["success"], row["id"]
        assert len(tracker["passes"]) == row["passed_tests"] and tracker["num_tests"] == row["total_tests"], row["id"]
        counts["official_tests_recorded"] += 1
        failures = tracker["failures"]
        answer = [f for f in failures if "answers match" in f["requirement"].lower()]
        if not answer:
            continue
        counts["any_answer_failure"] += 1
        if len(failures) != 1:
            continue
        counts["only_answer_failure"] += 1
        trace = answer[0]["trace"]
        match = NULL.search(trace)
        if "<<not_given>>" in trace:
            category = "only_answer_missing"
        elif match:
            prefix = trace[:match.start()].rsplit("AssertionError:", 1)[-1].strip()
            category = "only_null_wrong_value"
            if SUMMARY_WORDS.search(prefix):
                counts["null_wrong_value_with_summary_language"] += 1
        else:
            category = "only_answer_other"
        counts[category] += 1
        findings.append({"task": row["task_id"], "rollout_id": row["id"], "pct": row["pct"],
                         "category": category, "turn_cap": bool(row.get("turn_limit_reached"))})
    assert sum(counts[key] for key in ("only_answer_missing", "only_null_wrong_value", "only_answer_other")) == (
        counts["only_answer_failure"]
    )
    return counts, findings


def main():
    combined = []
    case_rows = []
    fields = ("rollouts", "successes", "turn_caps", "official_tests_recorded", "budget_failures",
              "missing_official_tests", "any_answer_failure", "only_answer_failure", "only_null_wrong_value",
              "only_answer_missing", "only_answer_other", "null_wrong_value_with_summary_language")

    def add(model, mode, arm, strategy, seed, rows, policy):
        counts, findings = audit(rows, policy)
        tag = {"model": model, "mode": mode, "arm": arm, "strategy": strategy, "seed": seed}
        combined.append({**tag, **{key: counts[key] for key in fields}})
        case_rows.extend({**tag, **case} for case in findings)

    for model, runs in MODELS.items():
        for mode, run in runs.items():
            protocol = read(run / "protocol.json")
            policy = protocol["appworld_policy_sha256"]
            source = Path(protocol["source_reflection_run"])
            initial = [read(f) for f in (source / "stage1").glob("*/result.json")]
            assert len(initial) == 250 and len({(r["task_id"], r["source_repeat"]) for r in initial}) == 250
            add(model, mode, "-", "initial", "-", initial, policy)
            for seed in SEEDS:
                path = run / f"seed{seed}" / "train_final_results.json"
                if model == "qwen35" and not path.exists():
                    continue
                rows = read(path)
                assert len(rows) == 300 and len({(r["condition_id"], r["task_id"]) for r in rows}) == 300
                tasks = set(protocol["task_orders"][str(seed)]["train"])
                for arm in ARMS:
                    for strategy in ("best", "random", "worst"):
                        matched = [r for r in rows if r["condition_id"] == f"s{seed}_{arm}_{strategy}"]
                        assert {r["task_id"] for r in matched} == tasks and len(matched) == 50
                        add(model, mode, arm, strategy, seed, matched, policy)
    for filename, rows in (("by_condition.csv", combined), ("answer_only_cases.csv", case_rows)):
        with (OUTPUT / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
