# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Resample the frozen Train10/Val10 pools to plot BFCL selection for n=1,...,10."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TARGETS = ("full_train", "full_validation")
SELECTORS = ("train10", "val10")
COLORS = ("#287C85", "#A34F65")
SHADE_COLORS = ("#EAF4F5", "#F9EEF1")


def load_model(label: str, summary_path: Path) -> dict:
    raw = summary_path.read_bytes()
    summary = json.loads(raw)
    if "partial_progress" in summary or not summary["checkpoints"]:
        raise ValueError(f"Expected a final, nonempty summary: {summary_path}")
    plans, points = {}, []
    for original in sorted(summary["checkpoints"], key=lambda point: point["key"]):
        if not original["evaluation_complete"]:
            raise ValueError(f"Incomplete evaluation: {original['key']}")
        source = Path(original["source"])
        if source not in plans:
            manifest = source / "launch_manifest.json"
            data = manifest.read_bytes()
            plans[source] = {"sha256": hashlib.sha256(data).hexdigest(),
                             "checkpoints": {point["key"]: point for point in json.loads(data)["checkpoints"]}}
        frozen = plans[source]["checkpoints"][original["key"]]
        samples, task_ids = {}, {}
        for selector, partition, pool_key, full_key in (
            ("train10", "train", "past_task_ids", "train_task_ids"),
            ("val10", "val", "validation_task_ids", "validation_full_task_ids"),
        ):
            pool = frozen[pool_key]
            if len(pool) != 10 or len(set(pool)) != 10:
                raise ValueError(f"Expected ten distinct frozen tasks: {original['key']}/{selector}")
            matrix = []
            for index in range(1, 6):
                context = f"candidate_{index:02d}"
                rewards = []
                for task in pool:
                    task_index = frozen[full_key].index(task)
                    path = source / "results" / original["key"] / context / partition / f"{task_index:03d}.json"
                    row = json.loads(path.read_text())
                    if (row["task_id"] != task or row["checkpoint_key"] != original["key"]
                            or row["context"] != context or row["partition"] != partition
                            or row["context_sha256"] != original["context_sha256"][index - 1]
                            or row["status"] not in ("completed", "error")):
                        raise ValueError(f"Frozen result identity differs: {path}")
                    rewards.append(row["score"])
                matrix.append(rewards)
            if not np.allclose(np.mean(matrix, axis=1), original["scores"][selector], atol=1e-12, rtol=0):
                raise ValueError(f"Ten-task score differs from the final summary: {original['key']}/{selector}")
            samples[selector], task_ids[selector] = matrix, pool
        points.append({"key": original["key"], "seed": original["seed"], "checkpoint": original["checkpoint"],
                       "scores": original["scores"], "parent_scores": original["parent_scores"],
                       "samples": samples, "task_ids": task_ids})
    errors = len(summary["candidate_execution_errors"]) + len(summary["parent_execution_errors"])
    evaluations = sum(point["coverage"]["parent_full_train"] + sum(point["coverage"]["full_train"])
                      + sum(point["coverage"]["full_validation"]) for point in summary["checkpoints"])
    return {"label": label, "source": str(summary_path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "source_manifests": {str(source): plan["sha256"] for source, plan in plans.items()},
            "points": points, "reference": summary["aggregate"], "errors": errors, "evaluations": evaluations}


def correlation_table(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Rows are checkpoint/subset; Pearson r is across the same five candidates."""
    centered = scores - scores.mean(axis=-1, keepdims=True)
    target_centered = targets - targets.mean(axis=-1, keepdims=True)
    numerator = (centered * target_centered[:, None, :]).sum(axis=-1)
    denominator = np.sqrt((centered ** 2).sum(axis=-1) * (target_centered ** 2).sum(axis=-1)[:, None])
    defined = (np.ptp(scores, axis=-1) > 0) & (np.ptp(targets, axis=-1)[:, None] > 0)
    values = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=values, where=defined & (denominator > 0))
    return values


def selection_table(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Integrate uniform tie-breaking instead of choosing an arbitrary candidate."""
    tied = scores == scores.max(axis=-1, keepdims=True)
    return (tied * targets[:, None, :]).sum(axis=-1) / tied.sum(axis=-1)


def subset_draws(keys: list[str], selector: str, n: int, count: int, trials: int, seed: int) -> np.ndarray:
    """Stable draws match across models, but train and validation draws are independent."""
    draws = []
    for key in keys:
        entropy = hashlib.sha256(f"{seed}:{n}:{key}:{selector}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(entropy[:16], "big"))
        draws.append(rng.integers(count, size=trials))
    return np.column_stack(draws)


def sampling_range(values: np.ndarray) -> list[float] | None:
    valid = values[np.isfinite(values)]
    return np.quantile(valid, [0.025, 0.975]).tolist() if valid.size else None


def sampling_std(values: np.ndarray) -> float | None:
    valid = values[np.isfinite(values)]
    if not valid.size:
        return None
    return float(np.std(valid)) if np.ptp(valid) > 0 else 0.0


def calculate_model(model: dict, trials: int, seed: int) -> dict:
    points = model["points"]
    keys = [point["key"] for point in points]
    columns = np.arange(len(points))[None, :]
    sizes, checkpoint_rows = [], []
    baselines = {}
    for target in TARGETS:
        values = np.array([point["scores"][target] for point in points])
        parents = [point["parent_scores"][target] for point in points]
        baselines[target] = {"proposal_mean_performance": float(values.mean()),
                             "parent_mean_performance": float(np.mean(parents))
                             if all(value is not None for value in parents) else None}
        parent_mean = baselines[target]["parent_mean_performance"]
        baselines[target]["proposal_mean_gain"] = (
            float(values.mean()) - parent_mean if parent_mean is not None else None
        )
    for n in range(1, 11):
        subsets = np.array(list(itertools.combinations(range(10), n)))
        tables, draws = {}, {}
        for selector in SELECTORS:
            # One task subset scores all five candidates. Never resample separately per candidate.
            tables[selector] = np.array([
                np.asarray(point["samples"][selector])[:, subsets].mean(axis=-1).T for point in points
            ])
            draws[selector] = subset_draws(keys, selector, n, len(subsets), trials, seed)
        entry = {"n": n, "subsets_per_pool": len(subsets), "targets": {}}
        for target in TARGETS:
            values = np.array([point["scores"][target] for point in points])
            proposal_mean = baselines[target]["proposal_mean_performance"]
            parent_mean = baselines[target]["parent_mean_performance"]
            correlation_tables = {selector: correlation_table(tables[selector], values) for selector in SELECTORS}
            sampled = {selector: correlation_tables[selector][columns, draws[selector]] for selector in SELECTORS}
            paired = np.isfinite(sampled["train10"]) & np.isfinite(sampled["val10"])
            counts = paired.sum(axis=-1)
            aggregate = {"paired_checkpoint_count_mean": float(counts.mean()),
                         "paired_checkpoint_count_range": sampling_range(counts),
                         "paired_checkpoint_count_std": sampling_std(counts),
                         "defined_trials": int((counts > 0).sum()), "total_trials": trials}
            for selector in SELECTORS:
                correlations = np.full(trials, np.nan)
                numerator = np.where(paired, sampled[selector], 0).sum(axis=-1)
                np.divide(numerator, counts, out=correlations, where=counts > 0)
                selected = selection_table(tables[selector], values)
                selected_mean = float(selected.mean())  # Exact uniform expectation over every n-of-10 subset.
                selected_trials = selected[columns, draws[selector]].mean(axis=-1)
                corr_table = correlation_tables[selector]
                valid = np.isfinite(corr_table)
                aggregate[selector] = {
                    "mean_pearson": float(np.nanmean(correlations)) if (counts > 0).any() else None,
                    "pearson_sampling_range": sampling_range(correlations),
                    "pearson_std": sampling_std(correlations),
                    "individually_defined_checkpoint_count_mean": float(valid.mean(axis=-1).sum()),
                    "mean_selected_performance": selected_mean,
                    "selected_performance_sampling_range": sampling_range(selected_trials),
                    "selected_performance_std": sampling_std(selected_trials),
                    "mean_selected_gain": selected_mean - parent_mean if parent_mean is not None else None,
                    "selected_gain_sampling_range": sampling_range(selected_trials - parent_mean)
                    if parent_mean is not None else None,
                    "selected_gain_std": sampling_std(selected_trials) if parent_mean is not None else None,
                    "mean_selection_gain": selected_mean - proposal_mean,
                    "selection_gain_sampling_range": sampling_range(selected_trials - proposal_mean),
                    "selection_gain_std": sampling_std(selected_trials),
                    "mean_regret": float(values.max(axis=-1).mean()) - selected_mean,
                }
                for index, point in enumerate(points):
                    defined = corr_table[index][valid[index]]
                    performance = float(selected[index].mean())
                    parent = point["parent_scores"][target]
                    checkpoint_rows.append({
                        "model": model["label"], "checkpoint": point["key"], "seed": point["seed"],
                        "n": n, "target": target, "selector": selector, "subsets": len(subsets),
                        "defined_correlation_subsets": int(defined.size),
                        "mean_individually_defined_pearson": float(defined.mean()) if defined.size else None,
                        "mean_selected_performance": performance,
                        "proposal_mean_performance": float(values[index].mean()),
                        "mean_selected_gain": performance - parent if parent is not None else None,
                        "mean_selection_gain": performance - float(values[index].mean()),
                    })
            entry["targets"][target] = aggregate
        sizes.append(entry)
    result = {key: value for key, value in model.items() if key not in ("reference", "points")}
    result.update(checkpoints=points, baselines=baselines, sizes=sizes, checkpoint_scores=checkpoint_rows)
    validate_ten(result, model["reference"])
    result["n10_matches_existing_summary"] = True
    return result


def validate_ten(result: dict, reference: dict) -> None:
    for target in TARGETS:
        actual, expected = result["sizes"][-1]["targets"][target], reference[target]
        if actual["paired_checkpoint_count_mean"] != expected["paired_checkpoints"]:
            raise ValueError(f"n=10 paired cohort changed: {result['label']}/{target}")
        for selector in SELECTORS:
            for new, old in (("mean_pearson", "mean_pearson"), ("mean_selected_gain", "mean_selected_gain"),
                             ("mean_selection_gain", "mean_selection_gain"),
                             ("mean_selected_performance", "mean_selected_performance")):
                left, right = actual[selector][new], expected[selector][old]
                if (left is None) != (right is None) or (left is not None and not math.isclose(
                    left, right, abs_tol=1e-10, rel_tol=1e-10
                )):
                    raise ValueError(f"n=10 differs from existing summary: {result['label']}/{target}/{selector}/{new}")


def curve(axis, result: dict, target: str, metric: str) -> None:
    sizes = result["sizes"]
    x = [entry["n"] for entry in sizes]
    if metric == "correlation":
        key, std_key, scale = "mean_pearson", "pearson_std", 1
    elif target == "full_train":
        key, std_key, scale = "mean_selected_gain", "selected_gain_std", 100
    else:
        key, std_key, scale = "mean_selection_gain", "selection_gain_std", 100
    for selector, color, shade_color in zip(SELECTORS, COLORS, SHADE_COLORS, strict=True):
        summaries = [entry["targets"][target][selector] for entry in sizes]
        y = np.array([row[key] * scale if row[key] is not None else np.nan for row in summaries])
        std = np.array([row[std_key] * scale if row[std_key] is not None else np.nan for row in summaries])
        axis.fill_between(x, y - std, y + std, color=shade_color, linewidth=0, zorder=1)
        axis.plot(x, y, color=color, marker="o", markersize=2.6, linewidth=1.5, zorder=3,
                  label="Train" if selector == "train10" else "Validation")
    axis.axhline(0, color="#8B8B8B", linewidth=0.6, zorder=2)
    if metric == "gain" and target == "full_train":
        baseline = result["baselines"][target]["proposal_mean_gain"] * 100
        axis.axhline(baseline, color="#687581", linewidth=1.1, linestyle="--", label=r"$\mu_\delta$", zorder=2)
    axis.set(xlim=(0.7, 10.3))
    axis.set_xticks(range(1, 11))
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)


def save_figure(figure, output: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")


def plot_model_overview(result: dict, output: Path) -> None:
    """Keep text legible when the four panels are inserted at one-column width."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    panels = [("full_train", "correlation", "Full train", r"$\rho$"),
              ("full_validation", "correlation", "Full validation", r"$\rho$"),
              ("full_train", "gain", "Full train", r"$\delta_t^{i^\star}$ (pp)"),
              ("full_validation", "gain", "Full validation", r"$s_t^{i^\star}-\mu_\delta$ (pp)")]
    style = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.5,
             "font.weight": "normal", "axes.labelweight": "normal", "axes.titleweight": "normal",
             "mathtext.fontset": "dejavuserif", "mathtext.default": "it",
             "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65,
             "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        figure, axes = plt.subplots(2, 2, figsize=(3.35, 2.75), sharex=True)
        figure.subplots_adjust(left=0.155, right=0.99, top=0.855, bottom=0.16, wspace=0.47, hspace=0.31)
        for axis, (target, metric, target_label, ylabel) in zip(axes.flat, panels, strict=True):
            curve(axis, result, target, metric)
            axis.set_title(target_label, fontsize=7.5, pad=3)
            axis.set_ylabel(ylabel, fontsize=8, labelpad=2)
            axis.set_xticks([1, 3, 5, 7, 10])
            axis.set_xticks([2, 4, 6, 8, 9], minor=True)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.tick_params(labelsize=6.5, length=2.5, width=0.6, pad=2)
            axis.tick_params(axis="x", which="minor", length=1.6, width=0.6)
            axis.grid(axis="y", color="#D5D5D5", linewidth=0.45, alpha=0.6)
            for line in axis.lines[:2]:
                line.set_linewidth(1.2)
                line.set_markersize(2.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        extra_handles, extra_labels = axes[1, 0].get_legend_handles_labels()
        for handle, label in zip(extra_handles, extra_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=len(labels),
                      frameon=False, prop={"size": 7, "weight": "normal"},
                      handlelength=1.4, handletextpad=0.4, columnspacing=0.9)
        figure.supxlabel(r"Number of evaluation tasks ($n$)", fontsize=7.5, y=0.03)
        slug = "".join(char.lower() if char.isalnum() else "_" for char in result["label"]).strip("_")
        for suffix in ("pdf", "png"):
            figure.savefig(output / f"{slug}_overview.{suffix}", dpi=600)
        plt.close(figure)


def plot_combined_overview(results: list[dict], output: Path) -> None:
    """Compare all models using the accepted paper typography and common row scales."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    order = {label: index for index, label in enumerate(("qwen", "gemma", "oss"))}
    models = sorted(results, key=lambda result: order.get(result["label"].lower(), len(order)))
    panels = [("full_train", "correlation", "Full train", r"$\rho$"),
              ("full_validation", "correlation", "Full validation", r"$\rho$"),
              ("full_train", "gain", "Full train", r"$\delta_t^{i^\star}$ (pp)"),
              ("full_validation", "gain", "Full validation", r"$s_t^{i^\star}-\mu_\delta$ (pp)")]
    style = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.5,
             "font.weight": "normal", "axes.labelweight": "normal", "axes.titleweight": "normal",
             "mathtext.fontset": "dejavuserif", "mathtext.default": "it",
             "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65,
             "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        figure, axes = plt.subplots(4, len(models), figsize=(2.33 * len(models), 5.6),
                                    sharex=True, sharey="row", squeeze=False)
        figure.subplots_adjust(left=0.105, right=0.99, top=0.91, bottom=0.085, wspace=0.15, hspace=0.23)
        for row, (target, metric, target_label, ylabel) in enumerate(panels):
            for column, model in enumerate(models):
                axis = axes[row, column]
                curve(axis, model, target, metric)
                axis.set_xticks([1, 3, 5, 7, 10])
                axis.set_xticks([2, 4, 6, 8, 9], minor=True)
                axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
                axis.tick_params(labelsize=7, length=2.5, width=0.6, pad=2)
                axis.tick_params(axis="x", which="minor", length=1.6, width=0.6)
                axis.grid(axis="y", color="#D5D5D5", linewidth=0.45, alpha=0.6)
                for line in axis.lines[:2]:
                    line.set_linewidth(1.2)
                    line.set_markersize(2.5)
                if row == 0:
                    axis.set_title(model["label"], fontsize=8.5, pad=5)
                if column == 0:
                    axis.set_ylabel(target_label + "\n" + ylabel, fontsize=8, labelpad=3)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        extra_handles, extra_labels = axes[2, 0].get_legend_handles_labels()
        for handle, label in zip(extra_handles, extra_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(labels),
                      frameon=False, prop={"size": 7.5, "weight": "normal"},
                      handlelength=1.8, handletextpad=0.45, columnspacing=1.4)
        figure.supxlabel(r"Number of evaluation tasks ($n$)", fontsize=8, y=0.025)
        for suffix in ("pdf", "png"):
            figure.savefig(output / f"overview.{suffix}", dpi=600)
        plt.close(figure)


def plot_results(results: list[dict], output: Path, trials: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    panels = [("full_train", "correlation", "Full train", r"$\rho$"),
              ("full_validation", "correlation", "Full validation", r"$\rho$"),
              ("full_train", "gain", "Full train", r"$\delta_t^{i^\star}$ (pp)"),
              ("full_validation", "gain", "Full validation", r"$s_t^{i^\star}-\mu_\delta$ (pp)")]
    xlabel = r"Number of evaluation tasks ($n$)"

    def legend(figure, axis, y, baseline_axis=None):
        handles, labels = axis.get_legend_handles_labels()
        if baseline_axis is not None:
            extra_handles, extra_labels = baseline_axis.get_legend_handles_labels()
            for handle, label in zip(extra_handles, extra_labels, strict=True):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
                      ncol=len(labels), frameon=False, fontsize=8.5)

    for target, metric, target_label, ylabel in panels:
        fig, axes = plt.subplots(1, len(results), figsize=(2.37 * len(results), 2.45), sharey=True, squeeze=False)
        flat = list(axes.flat)
        fig.subplots_adjust(left=0.10, right=0.99, top=0.78, bottom=0.22, wspace=0.15)
        for axis, result in zip(flat, results, strict=True):
            curve(axis, result, target, metric)
            axis.set_title(result["label"], fontsize=9.5, pad=7)
        flat[0].set_ylabel(target_label + "\n" + ylabel, labelpad=3)
        fig.supxlabel(xlabel, fontsize=9, y=0.025)
        legend(fig, flat[0], 1.01)
        save_figure(fig, output, f"{metric}_{target}")
        plt.close(fig)

    plot_combined_overview(results, output)

    fig, axes = plt.subplots(1, len(results), figsize=(2.37 * len(results), 2.45), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.78, bottom=0.22, wspace=0.15)
    for axis, result in zip(axes.flat, results, strict=True):
        for target, color, shade_color, marker in zip(TARGETS, COLORS, SHADE_COLORS, ("o", "x"), strict=True):
            entries = [size["targets"][target] for size in result["sizes"]]
            x = range(1, 11)
            y = np.array([entry["paired_checkpoint_count_mean"] for entry in entries])
            std = np.array([entry["paired_checkpoint_count_std"] for entry in entries])
            axis.fill_between(x, y - std, y + std, color=shade_color, linewidth=0, zorder=1)
            axis.plot(x, y, color=color, linewidth=1.5, zorder=3,
                      marker=marker, markersize=2.6,
                      label="Full train" if target == "full_train" else "Full validation")
        axis.set(ylim=(0, len(result["checkpoints"])))
        axis.set_title(result["label"], fontsize=9.5, pad=7)
        axis.set_xticks(range(1, 11))
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    axes[0, 0].set_ylabel(r"Checkpoints with defined $\rho$", labelpad=3)
    fig.supxlabel(xlabel, fontsize=9, y=0.025)
    legend(fig, axes[0, 0], 1.01)
    save_figure(fig, output, "paired_checkpoint_coverage")
    plt.close(fig)

    for result in results:
        plot_model_overview(result, output)


def write_results(results: list[dict], output: Path, trials: int, seed: int) -> None:
    rows, checkpoint_rows = [], []
    for result in results:
        checkpoint_rows.extend(result["checkpoint_scores"])
        for size in result["sizes"]:
            for target in TARGETS:
                aggregate = size["targets"][target]
                for selector in SELECTORS:
                    row = {"model": result["label"], "n": size["n"], "target": target, "selector": selector,
                           "paired_checkpoint_count_mean": aggregate["paired_checkpoint_count_mean"],
                           **result["baselines"][target]}
                    for key, value in aggregate[selector].items():
                        if key.endswith("_sampling_range"):
                            row[f"{key}_low"], row[f"{key}_high"] = value if value is not None else (None, None)
                        else:
                            row[key] = value
                    rows.append(row)
    for name, records in (("scores.csv", rows), ("checkpoint_scores.csv", checkpoint_rows)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    payload = {"sampling_seed": seed, "random_subset_trials": trials,
               "sampling": "Uniform n-of-10 subsets without replacement within each frozen pool. "
                           "One subset scores all five candidates; separate train/validation draws. "
                           "Identical draws across models at the same checkpoint.",
               "gain": "Exact expectation over all combinations; uniform expectation among tied argmax candidates. "
                       "Average over all checkpoint groups, including undefined-correlation checkpoints.",
               "correlation": "Mean of within-checkpoint Pearson r on paired-defined checkpoint groups per trial; "
                              "then mean over nonempty trials. Undefined correlations are excluded, never set to zero. "
                              "The paired checkpoint set varies with n and the sampled subset.",
               "shading": "Mean plus/minus one population standard deviation across random-subset trials "
                          "with checkpoints fixed. Undefined correlation trials excluded. "
                          "Original 95% sampling ranges retained in JSON/CSV, but not plotted.",
               "figure_style": {"train_line": COLORS[0], "validation_line": COLORS[1],
                                "train_std_fill": SHADE_COLORS[0], "validation_std_fill": SHADE_COLORS[1],
                                "std_multiplier": 1, "png_dpi": 300,
                                "validation_gain_axis": r"s_t^{i^\star}-\mu_\delta"},
               "targets": "Fixed full50 train and validation targets, including the original score pools.",
               "validation_gain": "Selection lift over proposal mean. No parent validation evaluations exist.",
               "errors": "Preserves recorded error rewards of zero for consistency with the original figures.",
               "models": [{key: value for key, value in result.items() if key != "checkpoint_scores"}
                          for result in results]}
    (output / "sample_size_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    text = ["# BFCL: n=1…10개의 task로 후보 선택", "",
            "기존 각 checkpoint의 Train10과 Val10에서 각각 n개를 비복원 무작위 추출합니다.",
            "같은 n개 task로 후보 5개 모두를 평가하며, 두 조건은 서로 독립적으로 추출합니다.",
            f"모델 간 추출은 동일합니다. 난수 seed={seed}, 반복={trials:,}회입니다.", "",
            "- `overview.png` / `overview.pdf`: 세 모델의 correlation 두 개와 gain 두 개를 한 장에 표시.",
            "- `correlation_full_train.*`, `correlation_full_validation.*`: Train n·Val n의 Pearson r.",
            "- `gain_full_train.*`: 선택 후보의 부모 대비 gain, 점선은 proposal 평균 gain.",
            "- `gain_full_validation.*`: proposal 평균 대비 selection gain. Validation 부모 점수는 미측정.",
            "- `*_overview.*`: 각 모델만 모은 네 패널.",
            "- `scores.csv`: 그래프의 집계 수치. Gain 단위는 reward이며 그림에서는 100을 곱해 %p 표시.",
            "- `checkpoint_scores.csv`: CP별 전체 부분집합을 계산한 결과.",
            "- `paired_checkpoint_coverage.*`: 두 상관이 모두 정의되는 CP 수와 추출에 따른 표준편차.",
            "- `sample_size_summary.json`: 원본 경로·해시·task ID·원시 10개 점수 및 방법 기록.", "",
            "Gain 중심선은 가능한 모든 n개 부분집합을 균등하게 선택했을 때의 정확한 기대값입니다.",
            "동점이면 최고 점수 후보 사이의 균등 선택 기대값을 사용합니다.",
            "ρ는 각 반복에서 두 조건의 상관이 모두 정의되는 동일 CP들을 평균한 뒤 반복 평균합니다.",
            "상수 점수/상수 target의 상관은 미정의이며 0으로 대체하지 않습니다. n에 따라 유효 CP 수가 달라집니다.",
            "유효 CP가 0개인 반복은 correlation 평균에서 제외합니다. 유효 반복 수는 summary에 기록했습니다.",
            "음영은 무작위 task 추출에 따른 집계 값의 평균 ±1 표준편차입니다(10,000회, ddof=0).",
            "기존 95% 추출 범위도 JSON/CSV에 보존했습니다. 그림에는 ±1 표준편차만 표시합니다.",
            "축 표기: correlation은 rho, train gain은 delta_t^i*, validation은 s_t^i* - mu_delta.",
            "Validation의 mu_delta 표기는 사용자가 지정한 축 기호입니다.",
            "실제 차감량은 proposal 평균 validation 점수이며 부모 대비 개선량의 평균은 미측정입니다.",
            "전체 target은 각 50개 task로 고정했고 점수 계산에 쓰인 task와 중복됩니다.", "",
            "| 모델 | CP 수 | 원본 실행 오류 | n=10 기존 결과 일치 |", "|---|---:|---:|---|",
            *[f"| {r['label']} | {len(r['checkpoints'])} | {r['errors']:,}/{r['evaluations']:,} | 확인 |"
              for r in results],
            "", "원본 실행 오류의 0점은 그대로 유지했습니다. 특히 OSS는 오류 비율이 높아 해석에 주의가 필요합니다.", ""]
    (output / "README.md").write_text("\n".join(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs=2, action="append", required=True, metavar=("LABEL", "PATH"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("Trials must be positive")
    models = [load_model(label, Path(path).resolve()) for label, path in args.summary]
    if len({model["label"] for model in models}) != len(models):
        raise ValueError("Model labels must be distinct")
    identities = [(point["key"], point["task_ids"]) for point in models[0]["points"]]
    if any([(point["key"], point["task_ids"]) for point in model["points"]] != identities for model in models[1:]):
        raise ValueError("Models must use identical checkpoint groups and frozen ten-task pools")
    results = []
    for model in models:
        print(f"Computing {model['label']}: {len(model['points'])} checkpoints", flush=True)
        results.append(calculate_model(model, args.trials, args.seed))
        print(f"Verified n=10 against the existing {model['label']} summary", flush=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_results(results, output, args.trials, args.seed)
    plot_results(results, output, args.trials)
    print(f"Saved all figures and scores to {output}", flush=True)


if __name__ == "__main__":
    main()
