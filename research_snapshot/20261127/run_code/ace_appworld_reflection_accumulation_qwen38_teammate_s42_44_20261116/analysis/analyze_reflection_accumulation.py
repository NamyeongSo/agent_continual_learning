# SPDX-License-Identifier: Apache-2.0
"""Summarize only completed accumulation evaluation phases, without new inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from scripts.ace.analyze_context_proposal_bottleneck import bootstrap_interval
from scripts.ace.run_reflection_accumulation import conditions, exact_reward

PHASES = ("train_final", "bwt_checkpoint", "val")


def official_tgc(row):
    """Never infer full-task success by thresholding a rounded PCT score."""
    flag = row.get("success")
    if flag is None:
        return None
    if type(flag) is not bool:
        raise ValueError("TGC requires a recorded boolean official success flag")
    return int(flag)


def tgc_summary(rows):
    values = [official_tgc(row) for row in rows]
    measured = [value for value in values if value is not None]
    return {"tgc_rate": statistics.mean(measured) if measured else None,
            "tgc_successes": sum(measured), "tgc_recorded_outcomes": len(measured),
            "tgc_missing_outcomes": len(values) - len(measured),
            "tgc_task_bootstrap_ci95": bootstrap_interval(measured) if measured else None}


def plot_summary(result, output, label):
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(result["conditions"])
    labels = [name.removeprefix(f"s{result['seed']}_").replace("without_gt_", "No GT\n")
              .replace("with_gt_", "GT\n") for name in names]
    x = np.arange(len(names))
    has_bwt = all("bwt" in result["conditions"][name] for name in names)
    fig, axes = plt.subplots(1, 2 if has_bwt else 1, figsize=(12 if has_bwt else 7, 4.8),
                             layout="constrained", squeeze=False)
    ax = axes[0, 0]
    train = [result["conditions"][n]["train_final"]["mean_pct"] * 100 for n in names]
    if "val" in result["completed_phases"]:
        val = [result["conditions"][n]["val"]["mean_pct"] * 100 for n in names]
        ax.bar(x - 0.18, train, width=0.34, label="Train", color="#3977ba")
        ax.bar(x + 0.18, val, width=0.34, label="Held-out", color="#d78837")
    else:
        ax.bar(x, train, color="#3977ba", label="Train (held-out pending)")
    ax.set(xticks=x, xticklabels=labels, ylabel="PCT (%)", ylim=(0, 103), title="Final C50 performance")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    if has_bwt:
        ax = axes[0, 1]
        bwt = [result["conditions"][n]["bwt"] for n in names]
        means = [r["all50"] * 100 for r in bwt]
        prior = [r["prior49"] * 100 for r in bwt]
        low = [r["all50_task_bootstrap_ci95"][0] * 100 for r in bwt]
        high = [r["all50_task_bootstrap_ci95"][1] * 100 for r in bwt]
        ax.vlines(x - 0.08, low, high, color="#3977ba", linewidth=1.5)
        ax.scatter(x - 0.08, means, color="#3977ba", label="All50 + task bootstrap 95%")
        ax.scatter(x + 0.08, prior, marker="x", color="#d78837", label="Prior49 (point estimate)")
        ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
        ax.set(xticks=x, xticklabels=labels, ylabel="Final PCT - post-learning PCT (pp)",
               title="BWT includes one-rollout evaluation noise")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"ACE accumulation seed {result['seed']} | 50 tasks/context, 1 independent rollout/task")
    fig.savefig(output / f"{label}.png", dpi=200)
    plt.close(fig)


def validate_phase(run, protocol, seed, phase, rows):
    split = protocol["task_orders"][str(seed)]
    tasks = split["val"] if phase == "val" else split["train"]
    expected = {(c["id"], t) for c in conditions(seed) for t in tasks}
    by_key = {}
    for row in rows:
        key = row["condition_id"], row["task_id"]
        if key not in expected or key in by_key:
            raise ValueError("Wrong task/condition or duplicate result")
        if (row["phase"] != phase or row["seed"] != seed or row["evaluation_repeat"] != 0
                or row["appworld_policy_sha256"] != protocol["appworld_policy_sha256"]):
            raise ValueError("Evaluation protocol mismatch")
        step = split["train"].index(row["task_id"]) + 1 if phase == "bwt_checkpoint" else 50
        checkpoint = Path(run) / "conditions" / row["condition_id"] / "checkpoints" / f"{step:03d}.json"
        cp = json.loads(checkpoint.read_text())
        if (row["checkpoint_step"] != step or cp["context_sha256"] != row["initial_context_sha256"]
                or cp["context_sha256"] != hashlib.sha256(cp["playbook"].encode()).hexdigest()):
            raise ValueError("Evaluation/checkpoint mismatch")
        if phase == "bwt_checkpoint" and cp["task_id"] != row["task_id"]:
            raise ValueError("BWT uses the wrong post-learning task checkpoint")
        exact_reward(row)
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError("Incomplete evaluation phase")
    return by_key


def analyze(run, seed):
    run = Path(run).resolve()
    protocol = json.loads((run / "protocol.json").read_text())
    phases, source_hashes = {}, {}
    for phase in PHASES:
        path = run / f"seed{seed}" / f"{phase}_results.json"
        if path.exists():
            raw = path.read_bytes()
            phases[phase] = validate_phase(run, protocol, seed, phase, json.loads(raw))
            source_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    if "train_final" not in phases:
        raise ValueError("Final training evaluation is not complete")
    split = protocol["task_orders"][str(seed)]
    result = {"seed": seed, "completed_phases": list(phases), "conditions": {},
              "reported_metrics": ["PCT", "TGC"], "tgc_metric": "appworld_tgc_v1",
              "task_bootstrap_samples": 10000, "bootstrap_seed": 42,
              "limitations": "One rollout/task/context. Task bootstrap does not estimate within-task repeatability.",
              "source_sha256": source_hashes,
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    task_rows = []
    for condition in conditions(seed):
        cid = condition["id"]
        summary = {}
        for phase, lookup in phases.items():
            tasks = split["val"] if phase == "val" else split["train"]
            rows = [lookup[(cid, task)] for task in tasks]
            values = [float(exact_reward(row)) for row in rows]
            tgc = tgc_summary(rows)
            summary[phase] = {
                "tasks": len(rows), "mean_pct": statistics.mean(values),
                "task_bootstrap_ci95": bootstrap_interval(values),
                "official_success_count": tgc["tgc_successes"], **tgc,
                "turn_cap_rollouts": sum(bool(row.get("turn_limit_reached")) for row in rows),
                "budget_failure_reasons": dict(Counter(row["budget_failure"]["reason"] for row in rows
                                                       if row.get("budget_failure"))),
            }
            for row in rows:
                task_rows.append({"condition": cid, "phase": phase, "task_id": row["task_id"],
                                  "checkpoint_step": row["checkpoint_step"], "pct": float(exact_reward(row)),
                                  "success": row.get("success"), "tgc": official_tgc(row),
                                  "budget_reason": (row.get("budget_failure") or {}).get("reason", "")})
        if "bwt_checkpoint" in phases:
            pairs = [{"task_id": task, "step": index + 1,
                      "final_pct": float(exact_reward(phases["train_final"][(cid, task)])),
                      "learned_pct": float(exact_reward(phases["bwt_checkpoint"][(cid, task)])),
                      "final_tgc": official_tgc(phases["train_final"][(cid, task)]),
                      "learned_tgc": official_tgc(phases["bwt_checkpoint"][(cid, task)])}
                     for index, task in enumerate(split["train"])]
            deltas = [r["final_pct"] - r["learned_pct"] for r in pairs]
            summary["bwt"] = {"all50": statistics.mean(deltas), "prior49": statistics.mean(deltas[:-1]),
                              "all50_task_bootstrap_ci95": bootstrap_interval(deltas),
                              "prior49_task_bootstrap_ci95": bootstrap_interval(deltas[:-1]),
                              "pairs": pairs}
            valid = [r for r in pairs if r["final_tgc"] is not None and r["learned_tgc"] is not None]
            prior = [r for r in valid if r["step"] < 50]
            for name, rows in [("all50", valid), ("prior49", prior)]:
                values = [r["final_tgc"] - r["learned_tgc"] for r in rows]
                summary["bwt"][f"tgc_{name}"] = statistics.mean(values) if values else None
                summary["bwt"][f"tgc_{name}_paired_tasks"] = len(values)
                summary["bwt"][f"tgc_{name}_task_bootstrap_ci95"] = bootstrap_interval(values) if values else None
        result["conditions"][cid] = summary
    comparisons = {}
    for phase, lookup in phases.items():
        tasks = split["val"] if phase == "val" else split["train"]
        contrasts = {}
        for arm in ("without_gt", "with_gt"):
            best, worst = f"s{seed}_{arm}_best", f"s{seed}_{arm}_worst"
            deltas = [float(exact_reward(lookup[(worst, task)]) - exact_reward(lookup[(best, task)]))
                      for task in tasks]
            contrasts[arm + "_worst_minus_best"] = {
                "mean_difference_pct": statistics.mean(deltas), "task_bootstrap_ci95": bootstrap_interval(deltas),
            }
        comparisons[phase] = contrasts
    result["paired_comparisons"] = comparisons
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    label = f"seed{seed}_" + ("complete" if len(phases) == 3 else "_".join(phases))
    (output / f"{label}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (output / f"{label}_tasks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    plot_summary(result, output, label)
    compact = {name: {key: {k: v for k, v in values.items() if k != "pairs"} for key, values in data.items()}
               for name, data in result["conditions"].items()}
    print(json.dumps({"seed": seed, "completed_phases": list(phases), "conditions": compact,
                      "paired_comparisons": comparisons}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=[42, 43, 44])
    args = parser.parse_args()
    analyze(args.run, args.seed)
