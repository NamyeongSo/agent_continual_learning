# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Combine existing policy-shift results in a model-by-benchmark overview."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.plot_qwen_policy_shift import METRICS, aggregate  # noqa: E402

MODELS = (("Qwen", "openai/Qwen/Qwen3.6-35B-A3B", "qwen"),
          ("Gemma", "openai/google/gemma-4-26B-A4B-it", "gemma"),
          ("OSS", "openai/openai/gpt-oss-20b", "oss"))
LEGACY_RUNS = ("ace_offline_90p_20260612-024204", "ace_offline_90p_seed124_20260612-042309",
               "ace_offline_90p_seed128_20260612-084057", "ace_offline_90p_seed129_20260612-085428",
               "ace_offline_90p_seed130_20260612-092852")
LEGACY_POINTS = tuple(range(5, 51, 5))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def call_agreement(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[float, float]:
    overlap = sum((Counter(left) & Counter(right)).values())
    size = len(left) + len(right)
    return float(left == right), 2 * overlap / size if size else 1.0


def legacy_gemma_pairwise(root: Path, output: Path) -> tuple[list[dict], dict]:
    """Compare saved predictions, retaining the legacy parser's strict call keys."""
    pair_rows, runs = [], []
    for run_id in LEGACY_RUNS:
        folder = root / "results/metrics" / f"appworld_same_history_{run_id}_pb005_to_pb090_regenerated"
        config = json.loads((folder / "run_config.json").read_text())
        if config["model"] != "google/gemma-4-26B-A4B-it":
            raise ValueError("Legacy AppWorld predictions must come from Gemma")
        records = folder / "records.jsonl"
        bank, signatures, observed = defaultdict(dict), {}, set()
        failures = 0
        with records.open() as stream:
            for line in stream:
                record = json.loads(line)
                point = int(record["checkpoint"])
                if point not in LEGACY_POINTS:
                    continue
                key = (record["task_id"], int(record["message_index"]), int(record["step"]))
                if (key, point) in observed:
                    raise ValueError("Duplicate legacy prediction; resolve before plotting")
                observed.add((key, point))
                signature = (record["trajectory_path"], record["history_message_count"], record["reference_content"])
                if key in signatures and signatures[key] != signature:
                    raise ValueError("Legacy checkpoint predictions do not share the same history")
                signatures[key] = signature
                if record["status"] != "success":
                    failures += 1
                    continue
                calls = record["strict"]["candidate"]
                if not isinstance(calls, list) or not all(isinstance(call, str) for call in calls):
                    raise ValueError("Expected cached strict Python API-call keys")
                bank[key][point] = tuple(calls)
        common = {key: predictions for key, predictions in bank.items() if set(predictions) == set(LEGACY_POINTS)}
        if not common:
            raise ValueError("No all-checkpoint complete legacy histories")
        for index, left in enumerate(LEGACY_POINTS):
            for right in LEGACY_POINTS[index + 1:]:
                tasks = defaultdict(list)
                for key, predictions in common.items():
                    tasks[key[0]].append(call_agreement(predictions[left], predictions[right]))
                accuracy = statistics.mean(statistics.mean(v[0] for v in values) for values in tasks.values())
                f1 = statistics.mean(statistics.mean(v[1] for v in values) for values in tasks.values())
                pair_rows.append({"run_id": run_id, "left": left, "right": right, "distance": right - left,
                                  "common_anchors": len(common), "common_tasks": len(tasks),
                                  "strict_accuracy": accuracy, "strict_f1": f1})
        runs.append({"run_id": run_id, "model": config["model"], "checkpoints": list(LEGACY_POINTS),
                     "available_histories": len(bank), "common_histories": len(common),
                     "common_tasks": len({key[0] for key in common}), "failed_predictions": failures,
                     "records": str(records), "records_sha256": sha256(records),
                     "config": str(folder / "run_config.json"), "config_sha256": sha256(folder / "run_config.json")})
        print(f"Legacy {run_id}: {len(common)} common histories", flush=True)
    write_csv(output / "gemma_appworld_pairwise_metrics.csv", pair_rows)
    data = []
    for distance in LEGACY_POINTS[:-1]:
        group = [row for row in pair_rows if row["distance"] == distance]
        for metric, *_ in METRICS:
            values = [1 - row[metric] for row in group]
            data.append({"benchmark": "AppWorld", "checkpoint_distance": distance, "metric": metric,
                         "mean_shift": statistics.mean(values), "pair_sample_sd": statistics.stdev(values),
                         "checkpoint_pairs": len(values), "seeds": len(LEGACY_RUNS)})
    return data, {"model": "Gemma", "benchmark": "AppWorld", "origin": "Legacy saved predictions",
                  "checkpoints": list(LEGACY_POINTS), "runs": runs,
                  "aggregation": "Mean within task, equal tasks per run, equal within-run checkpoint pairs",
                  "call_matching": "Original cached strict call keys; ordered exact ACC and multiset F1",
                  "new_inference_requests": 0}


def native_panel(analysis: Path, benchmark: str, display: str, model: str) -> tuple[list[dict], dict]:
    data, summary = aggregate(analysis, benchmark, model)
    return data, {"model": display, "benchmark": benchmark, "origin": "Existing pairwise metrics",
                  "checkpoints": summary["checkpoints"], "coverage": summary["coverage"],
                  "analysis": str(analysis), "source_hashes": {name: sha256(analysis / name)
                      for name in ("pairwise_metrics.csv", "summary.json")}}


def run(output: Path, legacy_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    panels, provenance, combined = {}, [], []
    base = ROOT / "outputs/bfcl_same_history_policy_shift_cp5_steps20_queue150_20261011"
    for display, model, key in MODELS:
        data, source = native_panel(base / key / "analysis", "BFCL", display, model)
        panels[(display, "BFCL")] = data
        provenance.append(source)
    data, source = native_panel(ROOT / "outputs/appworld_ace_qwen_policy_shift_cp5_restored_history_queue150_20261013"
                                / "analysis", "AppWorld", "Qwen", MODELS[0][1])
    panels[("Qwen", "AppWorld")] = data
    provenance.append(source)
    data, source = legacy_gemma_pairwise(legacy_root, output)
    panels[("Gemma", "AppWorld")] = data
    provenance.append(source)
    data, source = native_panel(ROOT / "outputs/appworld_ace_oss_policy_shift_cp5_steps20_queue150_20261013"
                                / "analysis", "AppWorld", "OSS", MODELS[2][1])
    panels[("OSS", "AppWorld")] = data
    provenance.append(source)
    for (model, _), data in panels.items():
        combined.extend({"model": model, **row} for row in data)
    write_csv(output / "policy_shift_overview.csv", combined)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.titlesize": 10, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.linewidth": 0.65})
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.125, right=0.995, bottom=0.145, top=0.91, wspace=0.16, hspace=0.22)
    maximum = max(row["mean_shift"] + row["pair_sample_sd"] for row in combined)
    upper = min(1.0, max(0.6, math.ceil((maximum + 0.035) * 10) / 10))
    correlations = []
    for row_index, benchmark in enumerate(("BFCL", "AppWorld")):
        for column, (display, _model, _key) in enumerate(MODELS):
            axis = axes[row_index, column]
            axis.set_xlim(1, 52)
            axis.set_ylim(0, upper)
            axis.set_xticks([10, 20, 30, 40, 50])
            axis.set_yticks([tick / 10 for tick in range(0, round(upper * 10) + 1, 2)])
            axis.tick_params(length=3, width=0.7, pad=3)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_axisbelow(True)
            if row_index == 0:
                axis.set_title(display.upper(), pad=6)
            data = panels.get((display, benchmark))
            if data is None:
                axis.set_facecolor("#F5F5F5")
                axis.text(0.5, 0.5, "In progress", transform=axis.transAxes, ha="center", va="center",
                          color="#888888", fontsize=9, fontstyle="italic")
                continue
            for metric, label, color, marker, line in METRICS:
                group = [item for item in data if item["metric"] == metric]
                x = [item["checkpoint_distance"] for item in group]
                y = [item["mean_shift"] for item in group]
                sd = [item["pair_sample_sd"] for item in group]
                r = statistics.correlation(x, y) if len(set(y)) > 1 else None
                value = f"{r:.4f}" if r is not None else r"\mathrm{undefined}"
                correlations.append({"model": display, "benchmark": benchmark, "metric": metric,
                                     "points": len(x), "pearson_r": r})
                axis.fill_between(x, [max(0, mean - spread) for mean, spread in zip(y, sd, strict=True)],
                                  [min(1, mean + spread) for mean, spread in zip(y, sd, strict=True)],
                                  color=color, alpha=0.12, linewidth=0)
                axis.plot(x, y, color=color, linewidth=1.15, marker=marker, markersize=2.8,
                          markerfacecolor="white", markeredgewidth=0.75, linestyle=line,
                          label=label[:-1] + rf",\ r={value}$")
            axis.grid(axis="y", color="#D5D5D5", linewidth=0.55, alpha=0.6)
            axis.legend(loc="lower right", frameon=False, fontsize=7, handlelength=1.3,
                        handletextpad=0.4, borderaxespad=0.35, borderpad=0.1, labelspacing=0.25)
        position = axes[row_index, 0].get_position()
        fig.text(0.064, (position.y0 + position.y1) / 2, benchmark.upper(), va="center", ha="center",
                 rotation=90, fontsize=9, fontweight="semibold")
    fig.supylabel("Policy shift", x=0.009, fontsize=9)
    fig.supxlabel("Checkpoint distance (training tasks)", y=0.012, fontsize=9)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output / f"policy_shift_overview.{extension}", dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    write_csv(output / "policy_shift_overview_correlations.csv", correlations)
    summary = {"panel_layout": {"rows": ["BFCL", "AppWorld"], "columns": [model[0] for model in MODELS]},
               "figure_size_inches": [6.8, 3.1], "latex_placement": "figure* [t], full text width",
               "pending_panel": None,
               "metrics": "Strict 1 - ACC and 1 - F1 under identical histories",
               "aggregation": "Task-balanced within run, then mean across within-run checkpoint pairs at each distance",
               "bands": "One sample SD across checkpoint pairs and runs, not confidence intervals",
               "histories": "Successful-history intersection at all checkpoints, separately within each study",
               "correlations": correlations, "correlation_definition": "Pearson r versus distance-averaged shift",
               "checkpoint_windows": "Native studies CP0..50; legacy Gemma AppWorld CP5..50 (CP0 not recorded)",
               "legacy_note": "Five legacy runs with Python API parsing; native studies use three seeds and tool calls",
               "sources": provenance, "created_at": datetime.now(UTC).isoformat()}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    caption = (
        "Policy shift on BFCL (top) and AppWorld (bottom) for Qwen, Gemma, and OSS (columns). "
        "Lines show strict $1-\\mathrm{ACC}$ and $1-\\mathrm{F1}$ on identical histories, averaging "
        "steps within task and tasks equally per run. Bands show one sample standard deviation across "
        "within-run checkpoint pairs and runs; legends report Pearson $r$ between distance and mean shift. "
        "Each study uses histories with valid predictions at every checkpoint. Gemma AppWorld uses five "
        "historical runs at CP5--50; the other panels use three seeds at CP0--50."
    )
    (output / "policy_shift_overview.tex").write_text(
        "\\begin{figure*}[t]\n  \\centering\n"
        "  \\includegraphics[width=\\textwidth]{figures/policy_shift_overview.pdf}\n"
        f"  \\caption{{{caption}}}\n  \\label{{fig:policy-shift-overview}}\n\\end{{figure*}}\n"
    )
    print(f"Saved {output / 'policy_shift_overview.pdf'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/policy_shift_overview_20261013")
    parser.add_argument("--legacy-root", type=Path, default=Path("/home/thskadud/agent_context"))
    args = parser.parse_args()
    run(args.output.resolve(), args.legacy_root.resolve())


if __name__ == "__main__":
    main()
