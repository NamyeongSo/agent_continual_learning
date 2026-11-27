# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Two-axis analysis of cached selections: retaining experience and generalizing.

Net success gain can conceal lost parent successes offset by newly solved tasks.
Retention therefore also measures the fraction of the parent's successes that
survive selection. Validation metrics use tasks outside the training partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_temporal_parent_preservation import CASES, MODELS, WEIGHTS  # noqa: E402
from scripts.premise.analyze_bfcl_temporal_selection import write_csv  # noqa: E402
from scripts.premise.bfcl_trajectory_selection_rewards import choice_probabilities  # noqa: E402


def outcome_vectors(success: np.ndarray, current: int, targets: dict) -> dict[str, np.ndarray]:
    """Compute target vectors after selection, keeping retention separate from repair."""
    success = np.asarray(success, dtype=float)
    if success.shape != (6, 50) or not np.isin(success, (0, 1)).all():
        raise ValueError("Expected binary parent and five-candidate results on fifty training tasks")
    if not isinstance(current, int) or not 1 <= current < 50:
        raise ValueError("Need at least one previously trained task")
    output = {}
    for name, stop in (("past", current), ("past_current", current + 1)):
        parent, candidate = success[0, :stop], success[1:, :stop]
        parent_passes = parent == 1
        count = int(parent_passes.sum())
        regression = ((candidate == 0) & parent_passes).mean(axis=1)
        repair = ((candidate == 1) & ~parent_passes).mean(axis=1)
        output[f"retention_{name}"] = (candidate[:, parent_passes].mean(axis=1)
                                      if count else np.full(5, np.nan))
        output[f"regression_{name}"] = regression
        output[f"repair_{name}"] = repair
        output[f"gain_{name}"] = candidate.mean(axis=1) - parent.mean()
        if not np.allclose(output[f"gain_{name}"], repair - regression, rtol=0, atol=1e-12):
            raise ValueError("Gain must decompose into repairs minus regressions")
    for source, name in (("full_validation", "validation50"), ("validation_remaining40", "validation40")):
        score = np.asarray(targets[source], dtype=float)
        if score.shape != (5,) or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
            raise ValueError("Validation targets require five candidate success rates")
        output[f"score_{name}"] = score
        output[f"lift_{name}"] = score - score.mean()
    return output


def evaluate_choice(probabilities: np.ndarray, outcomes: dict) -> dict:
    """Read frozen choice probabilities; evaluation outcomes cannot alter selection."""
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.shape != (5,) or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-10)):
        raise ValueError("Expected the frozen choice distribution over five candidates")
    probabilities = probabilities / probabilities.sum()
    return {name: float(probabilities @ vector) if np.isfinite(vector).all() else None
            for name, vector in outcomes.items()}


def aggregate_rows(rows: list[dict], metrics: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        for scope in ("all", row["seed"]):
            grouped[row["model"], row["case"], row["weight"], row["n"], scope].append(row)
    output = []
    for (model, case, weight, n, scope), group in grouped.items():
        row = {"model": model, "case": case, "weight": weight, "n": n,
               "scope": scope, "checkpoints": len(group)}
        for metric in metrics:
            values = np.asarray([p[metric] for p in group if p[metric] is not None], dtype=float)
            row[metric] = {"mean": float(values.mean()) if len(values) else None,
                           "checkpoint_std": float(values.std(ddof=1)) if len(values) > 1 else None,
                           "defined_checkpoints": len(values)}
        output.append(row)
    return output


def calculate(source: Path) -> dict:
    """Reuse every success_mean choice from the original seven-sampler figure."""
    old_path, cache_path = source / "summary.json", source / "full_train_metrics.json"
    old, cache = json.loads(old_path.read_text()), json.loads(cache_path.read_text())
    if hashlib.sha256(cache_path.read_bytes()).hexdigest() != old["source_sha256"]:
        raise ValueError("Cached outcomes differ from the original selection experiment")
    keys = old["fixed_checkpoint_keys"]
    points = {(m["label"], p["key"]): p for m in cache["models"] if m["label"] in MODELS
              for p in m["points"] if p["key"] in keys}
    if len(points) != len(keys) * len(MODELS):
        raise ValueError("Both models require the complete original checkpoint cohort")
    outcomes, baseline, refs, metrics = {}, [], [], []
    for (model, key), point in points.items():
        success = np.asarray(point["success"])
        t = point["current_index"]
        outcomes[model, key] = vectors = outcome_vectors(success, t, point["targets"])
        metrics = list(vectors)
        identity = {"model": model, "key": key, "seed": point["seed"], "checkpoint": t}
        baseline.append({**identity, "case": "proposal", "weight": "none", "n": 0,
                         **evaluate_choice(np.full(5, .2), vectors)})
        val_choice = choice_probabilities(np.asarray(point["targets"]["val10"])[None])[0]
        refs.append({**identity, "case": "val10", "weight": "none", "n": 10,
                     **evaluate_choice(val_choice, vectors)})
    rows, differences = [], []
    for previous in old["checkpoint_rows"]:
        if previous["method"] != "success_mean" or previous["model"] not in MODELS:
            continue
        identity = {k: previous[k] for k in ("model", "key", "seed", "checkpoint", "case", "weight", "n",
                                            "actual_n")}
        row = {**identity, **evaluate_choice(previous["choice_probabilities"],
                                            outcomes[previous["model"], previous["key"]])}
        for original, calculated in (("validation_lift", "lift_validation50"),
                                     ("validation_remaining40_lift", "lift_validation40")):
            differences.append(abs(previous[original] - row[calculated]))
        rows.append(row)
    maximum = max(differences)
    if maximum > 1e-12:
        raise ValueError("Validation selection outcomes changed")
    return {
        "models": list(MODELS), "cases": list(CASES), "weights": list(WEIGHTS), "method": "success_mean",
        "checkpoints_per_model": len(keys), "fixed_checkpoint_keys": keys, "trials": old["trials"],
        "sampling_seed": old["sampling_seed"], "new_model_calls": 0, "new_environment_calls": 0,
        "parent_validation_scores_available": False,
        "retention": "Within each checkpoint, fraction of parent-successful tasks still solved by the selected "
                     "candidate, then average equally across checkpoints. No parent successes: undefined.",
        "past": "Previously trained tasks train_order[:current_index]. The current task generated the proposal "
                "and is not yet in the parent's training history.",
        "past_current": "Original requested experience target train_order[:current_index+1]. Both its gain and "
                        "parent-success retention are also saved, without silently replacing the original target.",
        "gain": "Success-rate change relative to parent; equals repair rate minus regression rate. "
                "Positive net gain can conceal lost parent successes.",
        "generalization": "Candidate performance on the separate validation partition. Store absolute scores "
                          "and proposal-relative selection lifts, never a fabricated parent-relative val gain.",
        "primary_validation_comparison": "Common remaining validation40, excluding the archived val10 used "
                                         "by the validation-selection reference. Validation50 is also saved.",
        "uncertainty": "Frozen task-sampling and uniform tie expectations. Curves show checkpoint means; "
                       "checkpoint std is saved separately. No confidence bands or significance claims.",
        "selection": "Exactly the original seven success_mean samplers and both task/period weightings; "
                     "candidate choice is fixed before any retention or validation outcomes are accessed.",
        "source_files": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in (old_path, cache_path)],
        "aggregate": aggregate_rows(rows, metrics), "checkpoint_rows": rows,
        "proposal_reference": aggregate_rows(baseline, metrics),
        "validation_reference": aggregate_rows(refs, metrics),
        "validation_replay_audit": {"metrics_compared": len(differences), "maximum_absolute_difference": maximum},
    }


def flatten(rows: list[dict]) -> list[dict]:
    return [{**{k: v for k, v in row.items() if not isinstance(v, dict)},
             **{f"{k}_{field}": value for k, item in row.items() if isinstance(item, dict)
                for field, value in item.items()}} for row in rows]


def plot(summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from scripts.premise.plot_bfcl_temporal_parent_preservation import (
        CASE_LABELS,
        COLORS,
        MARKERS,
        STYLES,
        WEIGHT_LABELS,
    )

    output.mkdir(parents=True, exist_ok=True)
    indexed = {(r["model"], r["case"], r["weight"], r["n"]): r
               for r in summary["aggregate"] if r["scope"] == "all"}
    references = {r["model"]: r for r in summary["validation_reference"] if r["scope"] == "all"}
    proposals = {r["model"]: r for r in summary["proposal_reference"] if r["scope"] == "all"}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                         "axes.labelsize": 9.5, "axes.titlesize": 10, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .65})
    handles = [Line2D([0], [0], color=COLORS[c], linestyle=STYLES[c], marker=MARKERS[c],
                      markersize=3, linewidth=1.2, label=CASE_LABELS[c]) for c in CASES]
    handles += [Line2D([0], [0], color="#777777", linestyle=(0, (4, 3)), label="Proposal mean"),
                Line2D([0], [0], color="#111111", marker="*", linestyle="none", markersize=7, label="Val 10")]
    files = []
    validation_panel = ("lift_validation40", "Generalization", r"$s_{\rm val}^{i^\star}-\bar{s}_{\rm val}$ (pp)")
    retention_panel = ("retention_past", "Retention", "Past retention (%)")
    past_gain_panel = ("gain_past", "Past gain", r"$\delta_{t,\mathrm{past}}^{i^\star}$ (pp)")
    plot_specs = (
        ("retention_generalization", (retention_panel, validation_panel)),
        ("experience_gain_generalization", (
            ("gain_past_current", "Past + current gain", r"$\delta_t^{i^\star}$ (pp)"), validation_panel)),
        ("past_current_retention_generalization", (
            ("retention_past_current", "Retention", "Past + current retained (%)"), validation_panel)),
        ("past_gain_generalization", (past_gain_panel, validation_panel)),
        ("retention_past_gain_generalization", (retention_panel, past_gain_panel, validation_panel)),
    )
    for weight in WEIGHTS:
        for stem, panels in plot_specs:
            columns = len(panels)
            fig, axes = plt.subplots(2, columns, figsize=(3.7 * columns, 4.45), sharex=True, squeeze=False)
            fig.subplots_adjust(left=.10 if columns == 2 else .065, right=.985, bottom=.115,
                                top=.735, wspace=.34, hspace=.43)
            for r, model in enumerate(MODELS):
                for c, (metric, title, ylabel) in enumerate(panels):
                    ax = axes[r, c]
                    for case in CASES:
                        measured = [indexed[model, case, weight, n] for n in range(1, 11)
                                    if (model, case, weight, n) in indexed]
                        x = [row["n"] for row in measured]
                        y = [100 * row[metric]["mean"] for row in measured]
                        ax.plot(x, y, color=COLORS[case], linestyle=STYLES[case], marker=MARKERS[case],
                                linewidth=1.2, markersize=3, markeredgewidth=.55,
                                zorder=5 if case == "current_past" else 3)
                    ax.axhline(100 * proposals[model][metric]["mean"], color="#777777",
                               linestyle=(0, (4, 3)), linewidth=.9)
                    ax.scatter([10], [100 * references[model][metric]["mean"]], marker="*", color="#111111",
                               s=55, edgecolor="white", linewidth=.5, clip_on=False, zorder=7)
                    if metric.startswith("retention"):
                        ax.axhline(100, color="#BBBBBB", linewidth=.55)
                        ax.set_ylim(top=100.3)
                    else:
                        ax.axhline(0, color="#BBBBBB", linewidth=.55)
                    ax.set_title(f"{model} · {title}", pad=4)
                    ax.set_ylabel(ylabel)
                    ax.set(xlim=(.85, 10.25), xticks=(1, 2, 4, 6, 8, 10))
                    ax.grid(axis="y", linewidth=.4, alpha=.2)
                    ax.spines[["top", "right"]].set_visible(False)
                    if r == 1:
                        ax.set_xlabel(r"Sample size $n$")
            fig.suptitle(WEIGHT_LABELS[weight], fontsize=10, y=.995)
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .94), ncol=3,
                       frameon=False, fontsize=7.6, handlelength=2.2, columnspacing=1.2, labelspacing=.5)
            for extension in ("pdf", "png"):
                fig.savefig(output / f"{stem}_{weight}.{extension}", dpi=260, bbox_inches="tight", pad_inches=.035)
            plt.close(fig)
            files.append(f"{stem}_{weight}")
    (output / "manifest.json").write_text(json.dumps({"files": files, "confidence_bands": False}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = calculate(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "checkpoint_metrics.csv", summary["checkpoint_rows"])
    write_csv(args.output / "aggregate_metrics.csv", flatten(summary["aggregate"]))
    write_csv(args.output / "n10_metrics.csv", flatten([r for r in summary["aggregate"]
                                                      if r["n"] == 10 and r["scope"] == "all"]))
    plot(summary, args.output / "figures")
    print(f"Saved retention/generalization analysis to {args.output}", flush=True)


if __name__ == "__main__":
    main()
