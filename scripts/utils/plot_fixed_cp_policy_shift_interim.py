# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Read-only interim policy-shift figures for the six-setting CP10 own-history experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils import plot_pairwise_policy_shift_interim as analysis  # noqa: E402
from scripts.utils.plot_checkpoint_learning_curves import BENCHMARKS, COLORS, LABELS, panel_limits  # noqa: E402
from scripts.utils.policy_shift_metrics import write_json  # noqa: E402


def repeat_controls(root):
    path = root / "analysis/repeatability_metrics.csv"
    if not path.exists():
        return []
    groups = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            groups[row["benchmark"], row["method"], row["metric"]].append(row)
    return [{"benchmark": benchmark, "method": method, "metric": metric,
             "mean_repeat_shift": statistics.mean(float(row["policy_shift"]) for row in rows),
             "seeds": len(rows)} for (benchmark, method, metric), rows in sorted(groups.items())]


def plot(summary, output, *, tool_metric="strict_accuracy"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    correlations = analysis.curve_correlations(summary["scores"])
    lookup = {(r["benchmark"], r["method"], r["metric"]): r["pearson_r"] for r in correlations}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.1))
    for ax, (benchmark, title) in zip(axes, BENCHMARKS.items(), strict=True):
        metric = "accuracy" if benchmark == "terminalbench2" else tool_metric
        for (method, label), color in zip(LABELS.items(), COLORS, strict=True):
            group = sorted((r for r in summary["scores"] if (r["benchmark"], r["method"], r["metric"]) ==
                            (benchmark, method, metric)), key=lambda r: r["checkpoint_distance"])
            if not group:
                continue
            x = [r["checkpoint_distance"] for r in group]
            y = [r["mean_shift"] for r in group]
            sd = [r["pair_sample_sd"] for r in group]
            r = lookup[benchmark, method, metric]
            coefficient = "N/A" if r is None else f"{r:.2f}"
            ax.plot(x, y, color=color, marker="o", markersize=4, linewidth=1.9,
                    linestyle="--" if "train_as_val" in method else "-", label=f"{label} (r={coefficient})")
            ax.fill_between(x, [max(0, a-b) for a, b in zip(y, sd, strict=True)],
                            [min(1, a+b) for a, b in zip(y, sd, strict=True)], color=color, alpha=.08)
        histories = sum(r["all_checkpoint_common_histories"] for r in summary["coverage"]
                        if r["benchmark"] == benchmark)
        tasks = sum(r["compared_tasks"] for r in summary["coverage"] if r["benchmark"] == benchmark)
        lower, upper = panel_limits([{"mean": r["mean_shift"], "sample_sd": r["pair_sample_sd"]}
                                    for r in summary["scores"] if (r["benchmark"], r["metric"]) == (benchmark, metric)])
        ax.set(title=f"{title}\n{histories:,} history records · {tasks} task records",
               xlabel="Checkpoint distance (ΔCP)", ylabel="Policy shift (1 - action agreement)",
               ylim=(lower / 100 - .01, upper / 100 + .01),
               xticks=sorted({r["checkpoint_distance"] for r in summary["scores"] if r["benchmark"] == benchmark}))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7.5, loc="best", framealpha=.85)
    timestamp = summary["generated_at"][:16].replace("T", " ") + " KST"
    phase = "Interim" if summary["phase"] == "interim" else "Final observed"
    total = summary["effective_requests"]
    valid = summary["completed_requests"]
    metric_title = tool_metric.replace("_", " ")
    fig.suptitle(f"{phase} ΔCP policy shift · temperature 0 · {valid:,}/{total:,} valid responses\n"
                 f"{timestamp} · {metric_title} (BFCL/AppWorld)",
                 fontsize=16, y=.98)
    notes = [
        "Each baseline uses its own CP0 train histories; histories are shared only within that baseline's variations. "
        "AppWorld uses the Original policy + code generation.",
        "Use only histories with completed primary responses at ALL CPs within each stream. "
        "Unavailable/error responses are excluded; coverage is recorded in the companion CSV/JSON files.",
        ("BFCL/AppWorld: strict tool-call accuracy (arguments included). " if tool_metric == "strict_accuracy" else
         "BFCL/AppWorld: strict tool-call F1 (arguments included). ") +
        "Terminal-Bench: exact command-bundle accuracy. Shade = pair SD; r = Pearson(ΔCP, mean shift).",
        "Histories are averaged within task, then tasks equally within CP pair; "
        "pair/seed rows are averaged at each ΔCP. Unchanged CP contexts reuse responses. "
        "Counts repeat stream membership; y ranges vary by panel.",
    ]
    controls = summary.get("cp0_repeat_control", [])
    if controls:
        ranges = []
        for benchmark, label in BENCHMARKS.items():
            metric = "accuracy" if benchmark == "terminalbench2" else tool_metric
            values = [r["mean_repeat_shift"] for r in controls if
                      (r["benchmark"], r["metric"]) == (benchmark, metric)]
            if values:
                ranges.append(f"{label} {100 * min(values):.1f}-{100 * max(values):.1f}%")
        notes.append("Same-CP0 repeat disagreement: " + "; ".join(ranges) +
                     ". Serving repeat variation is present; no control subtraction is applied.")
    for i, note in enumerate(notes):
        fig.text(.015, .14 - i*.035, note, fontsize=9, color="#526271")
    fig.tight_layout(rect=(0, .25, 1, .88))
    suffix_ = "_f1" if tool_metric == "strict_f1" else ""
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"policy_shift_delta_cp{suffix_}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return correlations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Active policy-shift study")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    plan = json.loads((root / "plan.json").read_text())
    if (set(plan["method_labels"]) != set(LABELS) or plan["temperature"] != 0
            or plan["comparison"] != "all_within_seed_checkpoint_pairs_by_distance"):
        raise ValueError("Expected the fixed six-setting, temperature-zero, pairwise CP10 study")
    stamp = analysis.now().replace("-", "").replace(":", "").split(".")[0].replace("T", "_")
    output = args.out_dir.resolve() if args.out_dir else root / "analysis/plots" / stamp
    output.mkdir(parents=True, exist_ok=False)
    summary, pairs, tasks = analysis.analyze_partial(root / "policy_shift.sqlite3", require_all_checkpoints=True)
    if not summary["job_counts"].get("pending") and not summary["job_counts"].get("running"):
        summary["phase"] = "completed_with_missing" if summary["job_counts"].get("error") else "completed"
        summary["cp0_repeat_control"] = repeat_controls(root)
    summary.update(model=plan["model"], temperature=plan["temperature"], method_labels=plan["method_labels"],
                   history_bank=plan["history_bank"], workers=plan["workers"], source=str(root),
                   comparison=plan["comparison"], query_url=plan["base_url"])
    if not summary["scores"]:
        raise ValueError("No all-checkpoint common histories are complete yet")
    write_json(output / "summary.json", summary)
    write_json(output / "coverage.json", summary["coverage"])
    analysis.write_csv(output / "distance_metrics.csv", summary["scores"])
    analysis.write_csv(output / "pairwise_metrics.csv", pairs)
    analysis.write_csv(output / "pairwise_task_metrics.csv", tasks)
    correlations = plot(summary, output)
    plot(summary, output, tool_metric="strict_f1")
    analysis.write_csv(output / "correlations.csv", correlations)
    analysis.write_csv(output / "cp0_repeat_control_summary.csv", summary.get("cp0_repeat_control", []))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
              (Path(__file__).resolve(), Path(analysis.__file__).resolve(),
               ROOT / "scripts/utils/policy_shift_metrics.py")}
    manifest = {"generated_at": summary["generated_at"], "journal_id": plan["journal_id"],
                "source": str(root), "mode": "read_only_consistent_sqlite_snapshot",
                "method_labels": LABELS, "code_hashes": hashes, "job_counts": summary["job_counts"],
                "common_history_scope": summary["common_history_scope"],
                "common_history_records": sum(r["all_checkpoint_common_histories"] for r in summary["coverage"]),
                "cp_pair_rows": len(pairs), "task_pair_metric_rows": len(tasks),
                "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()
                          if path.is_file()}, "inference_unchanged": True}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "counts": summary["job_counts"],
                      "completion_fraction": summary["completion_fraction"],
                      "common_histories": manifest["common_history_records"], "cp_pair_rows": len(pairs)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
