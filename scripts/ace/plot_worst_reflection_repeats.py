# SPDX-License-Identifier: Apache-2.0
"""Plot the three saved independent PCT evaluations of each selected worst Reflection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from scripts.ace.run_reflection_accumulation import exact_reward

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "thinking": ROOT / "outputs/ace_appworld_reflection_accumulation_qwen38_teammate_s42_44_20261116",
    "nonthinking": ROOT / "outputs/ace_appworld_nonthinking_replication_qwen38_20261117/accumulation",
}
ARMS = ("without_gt", "with_gt")
SEEDS = (42, 43, 44)


def analyze(output, runs=RUNS):
    output.mkdir(parents=True, exist_ok=True)
    hashes, records, summaries = {}, [], {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for mode, run in runs.items():
        protocol = read(run / "protocol.json")
        source = Path(protocol["source_reflection_run"])
        tasks = set(protocol["task_orders"]["42"]["train"])
        assert len(tasks) == 50
        bank = defaultdict(dict)
        for path in sorted((source / "stage3").glob("*/result.json")):
            row = read(path)
            assert row["task_id"] in tasks
            assert row["appworld_policy_sha256"] == protocol["appworld_policy_sha256"]
            key = row["arm"], row["task_id"], row["source_repeat"]
            assert row["evaluation_repeat"] not in bank[key]
            bank[key][row["evaluation_repeat"]] = exact_reward(row)
        assert len(bank) == 500
        assert all(set(v) == {0, 1, 2} for v in bank.values())
        means = {key: sum(values.values(), Fraction()) / 3 for key, values in bank.items()}
        for arm in ARMS:
            reference = {}
            for seed in SEEDS:
                path = run / f"conditions/s{seed}_{arm}_worst/selections.json"
                choices = read(path)
                assert len(choices) == 50
                assert {choice["candidate"]["task_id"] for choice in choices} == tasks
                aud = ROOT / f"outputs/ace_appworld_aud_accumulation_qwen38_20261117/{mode}"
                aud_path = aud / path.relative_to(run)
                if runs is RUNS and aud_path.exists():
                    assert hashlib.sha256(aud_path.read_bytes()).hexdigest() == hashes[str(path)]
                selected = []
                for choice in choices:
                    candidate = choice["candidate"]
                    task = candidate["task_id"]
                    key = arm, task, candidate["source_repeat"]
                    mean = means[key]
                    minimum = min(means[(arm, task, i)] for i in range(5))
                    assert mean == minimum
                    assert mean == Fraction(candidate["score_numerator"], candidate["score_denominator"])
                    assert choice["mode"] == "worst"
                    tied = [i for i in range(5) if means[(arm, task, i)] == minimum]
                    assert choice["tie_indices"] == tied
                    if seed == 42:
                        reference[task] = mean
                    else:
                        assert reference[task] == mean
                    scores = [float(bank[key][i]) for i in range(3)]
                    selected.append(
                        {
                            "mode": mode,
                            "arm": arm,
                            "selection_seed": seed,
                            "task_id": task,
                            "source_id": candidate["source_id"],
                            "source_repeat": candidate["source_repeat"],
                            "pct_run1": scores[0],
                            "pct_run2": scores[1],
                            "pct_run3": scores[2],
                            "mean_pct": float(mean),
                            "mean_numerator": mean.numerator,
                            "mean_denominator": mean.denominator,
                            "within_reflection_sample_std": statistics.stdev(scores),
                            "minimum_tie_count": len(tied),
                        }
                    )
                selected.sort(
                    key=lambda r: (
                        Fraction(r["mean_numerator"], r["mean_denominator"]),
                        r["task_id"],
                    )
                )
                for i, row in enumerate(selected, 1):
                    row["reflection_rank"] = i
                records.extend(selected)
                values = [r["mean_pct"] for r in selected]
                summaries[f"{mode}/{arm}/seed{seed}"] = {
                    "reflections": 50,
                    "saved_evaluation_scores": 150,
                    "mean_pct": statistics.mean(values),
                    "median_pct": statistics.median(values),
                    "sample_std_pct": statistics.stdev(values),
                    "min_pct": min(values),
                    "max_pct": max(values),
                    "all_three_pct_zero": values.count(0),
                    "all_three_pct_one": values.count(1),
                    "all_five_candidates_tied": sum(r["minimum_tie_count"] == 5 for r in selected),
                }
    result = {
        "definition": "Each task's worst candidate minimizes the exact mean of its three independent saved PCT runs.",
        "plot_order": "Ascending three-run mean, task ID breaks display ties; rank maps to task/source ID in CSV.",
        "selection_seed_note": "Seeds only change the choice within tied minima; each task's mean is identical. "
        "These are reused source-bank evaluations, not new trajectories or independent seed replications.",
        "aud_selection_note": "Confirmed byte-identical worst selections wherever matched AUD selections exist. "
        "Pending Non-thinking AUD has not copied its selections yet; no new AUD candidate scores were measured.",
        "turn_limit_scoring": "Keep official final-state PCT for 40-turn terminations and original other budget zeros.",
        "summaries": summaries,
        "source_sha256": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (output / "reflection_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for seed in SEEDS:
        plot(records, seed, output, modes=tuple(runs))
    return result


def mode_label(mode):
    return {"thinking": "Thinking", "nonthinking": "Non-thinking"}.get(mode, f"Reasoning effort: {mode}")


def plot(records, seed, output, *, modes=tuple(RUNS)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(modes), len(ARMS), figsize=(23, 11), layout="constrained",
                             sharex=True, sharey=True, squeeze=False)
    colors = ("#3478b8", "#dd8a2c", "#329662")
    offsets = (-0.23, 0, 0.23)
    for row, mode in enumerate(modes):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            selected = [r for r in records if r["mode"] == mode and r["arm"] == arm and r["selection_seed"] == seed]
            assert len(selected) == 50
            x = [r["reflection_rank"] for r in selected]
            y = [100 * r["mean_pct"] for r in selected]
            ax.plot(x, y, color="#222222", linewidth=1.6, label="Mean of 3 runs", zorder=2)
            for i, (color, offset) in enumerate(zip(colors, offsets, strict=True), 1):
                ax.scatter(
                    [v + offset for v in x],
                    [100 * r[f"pct_run{i}"] for r in selected],
                    s=23,
                    color=color,
                    alpha=0.9,
                    label=f"Run {i}",
                    zorder=3,
                )
            grand_mean = statistics.mean(y)
            ax.axhline(
                grand_mean,
                color="#777777",
                linestyle="--",
                linewidth=1,
                label=f"Overall mean: {grand_mean:.2f}%",
                zorder=1,
            )
            n_perfect = sum(v == 100 for v in y)
            label = mode_label(mode)
            ax.set_title(
                f"{label} | {'w/o GT' if arm == 'without_gt' else 'w/ GT'} | "
                f"50 worst Reflections; 3/3 PCT=100%: {n_perfect}/50",
                fontsize=12,
            )
            ax.set_xlabel("Reflection rank (sorted by three-run mean; task/source IDs in CSV)")
            ax.set_ylabel("PCT (%)")
            ax.set_xlim(0.3, 50.7)
            ax.set_ylim(-3, 106)
            ax.set_xticks([1, *range(5, 51, 5)])
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_yticks(range(0, 101, 20))
            ax.grid(axis="y", alpha=0.2)
            ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.suptitle(
        f"Worst-selected Reflection: three independent PCT runs and their mean | selection seed {seed}\n"
        "Before accumulation; reused independent candidate evaluations; "
        "horizontal jitter separates overlapping dots",
        fontsize=14,
    )
    fig.savefig(output / f"worst_reflection_three_runs_seed{seed}.png", dpi=180)
    fig.savefig(output / f"worst_reflection_three_runs_seed{seed}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.output.resolve())
    print(json.dumps(result["summaries"], ensure_ascii=False, indent=2))
