# SPDX-License-Identifier: Apache-2.0
"""Compare matched independent-reflection expectations with accumulated PCT/TGC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from scripts.ace.analyze_context_proposal_bottleneck import bootstrap_interval
from scripts.ace.analyze_reflection_accumulation import official_tgc, validate_phase
from scripts.ace.run_reflection_accumulation import conditions, exact_reward


def candidate_metrics(rows):
    if len(rows) != 3 or {r["evaluation_repeat"] for r in rows} != {0, 1, 2}:
        raise ValueError("Independent candidates require three distinct saved evaluations")
    pct = sum((exact_reward(r) for r in rows), Fraction(0)) / 3
    flags = [official_tgc(r) for r in rows]
    measured = [v for v in flags if v is not None]
    return {"pct": float(pct), "pct_numerator": pct.numerator, "pct_denominator": pct.denominator,
            "tgc": statistics.mean(measured) if measured else None, "tgc_successes": sum(measured),
            "tgc_outcomes": len(measured)}


def selected_metrics(choice, arm, task, bank):
    candidate = choice["candidate"]
    if candidate["task_id"] != task:
        raise ValueError("Selected reflection/task mismatch")
    result = candidate_metrics(bank[(arm, task, candidate["source_repeat"])])
    if Fraction(result["pct_numerator"], result["pct_denominator"]) != Fraction(
            candidate["score_numerator"], candidate["score_denominator"]):
        raise ValueError("Independent score differs from the frozen selected reflection")
    return result


def compare(run, seed):
    run = Path(run).resolve()
    protocol = json.loads((run / "protocol.json").read_text())
    source = Path(protocol["source_reflection_run"])
    source_protocol = json.loads((source / "protocol.json").read_text())
    if source_protocol["appworld_policy_sha256"] != protocol["appworld_policy_sha256"]:
        raise ValueError("Different environment policy in independent and accumulated evaluations")
    bank, hashes = defaultdict(list), {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for path in sorted((source / "stage3").glob("*/result.json")):
        row = read(path)
        if row["appworld_policy_sha256"] != protocol["appworld_policy_sha256"]:
            raise ValueError("Independent candidate policy mismatch")
        bank[(row["arm"], row["task_id"], row["source_repeat"])].append(row)
    tasks = protocol["task_orders"][str(seed)]["train"]
    expected = {(arm, task, i) for arm in ("without_gt", "with_gt") for task in tasks for i in range(5)}
    if set(bank) != expected:
        raise ValueError("Independent bank must cover exactly train50/G5/two arms")
    for rows in bank.values():
        candidate_metrics(rows)
    train = validate_phase(run, protocol, seed, "train_final", read(run / f"seed{seed}/train_final_results.json"))
    val_path = run / f"seed{seed}/val_results.json"
    val = validate_phase(run, protocol, seed, "val", read(val_path)) if val_path.exists() else None
    initial = [read(p) for p in sorted((source / "stage1").glob("*/result.json"))]
    initial_flags = [official_tgc(r) for r in initial]
    measured = [v for v in initial_flags if v is not None]
    result = {"seed": seed, "selection_metric": "PCT (unchanged)", "conditions": {},
              "reported_metrics": ["PCT", "TGC"], "tgc_metric": "appworld_tgc_v1",
              "baseline_empty_context": {"pct": statistics.mean(float(exact_reward(r)) for r in initial),
                                         "tgc": statistics.mean(measured) if measured else None,
                                         "outcomes": len(measured)},
              "definition": "Independent expectation uses the exact selected reflection per train task and its 3 "
                            "saved independent evaluations; actual uses accumulated C50, one rollout/task.",
              "limitations": ["Same-bank best/worst selection bias; 3 versus 1 evaluation repeats",
                              "Difference includes Curator generation, accumulation, noise and budget effects",
                              "No independent-reflection or empty-context held-out baseline was measured"],
              "source_sha256": hashes}
    details = []
    for condition in conditions(seed):
        cid = condition["id"]
        choices = read(run / "conditions" / cid / "selections.json")
        if [c["candidate"]["task_id"] for c in choices] != tasks:
            raise ValueError("Selection task order mismatch")
        pairs = []
        for task, choice in zip(tasks, choices, strict=True):
            expected_metric = selected_metrics(choice, condition["arm"], task, bank)
            actual = train[(cid, task)]
            row = {"condition": cid, "task_id": task, "selected_source_repeat": choice["candidate"]["source_repeat"],
                   "expected_pct": expected_metric["pct"], "actual_pct": float(exact_reward(actual)),
                   "expected_tgc": expected_metric["tgc"], "actual_tgc": official_tgc(actual),
                   "expected_tgc_successes": expected_metric["tgc_successes"],
                   "expected_tgc_outcomes": expected_metric["tgc_outcomes"]}
            row["delta_pct"] = row["actual_pct"] - row["expected_pct"]
            row["delta_tgc"] = (row["actual_tgc"] - row["expected_tgc"]
                                if row["actual_tgc"] is not None and row["expected_tgc"] is not None else None)
            pairs.append(row)
        valid = [r for r in pairs if r["actual_tgc"] is not None and r["expected_tgc"] is not None]
        actual_flags = [r["actual_tgc"] for r in pairs if r["actual_tgc"] is not None]
        expected_count = sum(r["expected_tgc_outcomes"] for r in pairs)
        summary = {"tasks": len(pairs), "expected_independent_pct": statistics.mean(r["expected_pct"] for r in pairs),
                   "actual_accumulated_pct": statistics.mean(r["actual_pct"] for r in pairs),
                   "expected_independent_tgc": (sum(r["expected_tgc_successes"] for r in pairs) / expected_count
                                                if expected_count else None),
                   "actual_accumulated_tgc": statistics.mean(actual_flags) if actual_flags else None,
                   "expected_tgc_outcomes": expected_count, "actual_tgc_outcomes": len(actual_flags),
                   "tgc_paired_tasks": len(valid),
                   "delta_pct": statistics.mean(r["delta_pct"] for r in pairs),
                   "delta_tgc": statistics.mean(r["delta_tgc"] for r in valid) if valid else None,
                   "delta_pct_task_bootstrap_ci95": bootstrap_interval([r["delta_pct"] for r in pairs]),
                   "delta_tgc_task_bootstrap_ci95": bootstrap_interval([r["delta_tgc"] for r in valid]) if valid
                   else None}
        if val is not None:
            rs = [val[(cid, task)] for task in protocol["task_orders"][str(seed)]["val"]]
            flags = [official_tgc(r) for r in rs]
            flags = [v for v in flags if v is not None]
            summary["heldout_actual_pct"] = statistics.mean(float(exact_reward(r)) for r in rs)
            summary["heldout_actual_tgc"] = statistics.mean(flags) if flags else None
            summary["heldout_tgc_successes"] = sum(flags)
            summary["heldout_tgc_outcomes"] = len(flags)
        result["conditions"][cid] = summary
        details.extend(pairs)
    result["uniform_random_expectation"] = {}
    for arm in ("without_gt", "with_gt"):
        rows = [r for (a, _, _), rs in bank.items() if a == arm for r in rs]
        flags = [official_tgc(r) for r in rows]
        flags = [v for v in flags if v is not None]
        result["uniform_random_expectation"][arm] = {
            "pct": statistics.mean(float(exact_reward(r)) for r in rows),
            "tgc": statistics.mean(flags) if flags else None,
        }
    output = run / "analysis"
    label = f"seed{seed}_independent_vs_accumulated"
    result["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / f"{label}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (output / f"{label}_tasks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    plot(result, output, label)
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))
    return result


def plot(result, output, label):
    import matplotlib.pyplot as plt
    import numpy as np

    items = list(result["conditions"].items())
    names = [n.removeprefix(f"s{result['seed']}_").replace("without_gt_", "No GT\n")
             .replace("with_gt_", "GT\n") for n, _ in items]
    x = np.arange(len(items))
    def percent(value):
        return value * 100 if value is not None else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, metric in zip(axes, ("pct", "tgc"), strict=True):
        expected = [percent(r[f"expected_independent_{metric}"]) for _, r in items]
        actual = [percent(r[f"actual_accumulated_{metric}"]) for _, r in items]
        ax.bar(x - 0.18, expected, width=0.34, label="Independent: selected reflection, 3 repeats", color="#94a3b8")
        ax.bar(x + 0.18, actual, width=0.34, label="Accumulated C50: 1 repeat", color="#3977ba")
        ax.set(xticks=x, xticklabels=names, ylim=(0, 105), ylabel=f"{metric.upper()} (%)",
               title=f"Train50: expected independent vs actual accumulated {metric.upper()}")
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Seed {result['seed']} | matched selected reflections | best/worst selected by PCT")
    fig.savefig(output / f"{label}.png", dpi=200)
    plt.close(fig)
    if all("heldout_actual_tgc" in r for _, r in items):
        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        ax.bar(x - 0.18, [r["heldout_actual_pct"] * 100 for _, r in items], width=0.34,
               label="PCT", color="#3977ba")
        ax.bar(x + 0.18, [percent(r["heldout_actual_tgc"]) for _, r in items], width=0.34,
               label="TGC (official success)", color="#d78837")
        ax.set(xticks=x, xticklabels=names, ylim=(0, 105), ylabel="Score (%)",
               title=f"Seed {result['seed']}: held-out actual PCT and TGC | val50, 1 rollout/task")
        ax.legend(loc="lower right")
        fig.savefig(output / f"seed{result['seed']}_heldout_pct_tgc.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--seed", type=int, required=True, choices=[42, 43, 44])
    args = parser.parse_args()
    compare(args.run, args.seed)
