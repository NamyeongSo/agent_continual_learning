# SPDX-License-Identifier: Apache-2.0
"""Plot weighted score distributions of individual ACE Reflection candidates."""

import csv
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator, PercentFormatter

OUTPUT = Path(__file__).resolve().parent
RUN = OUTPUT.parents[1]
ARCHIVE = OUTPUT.parent / "notion_consolidation_20261120"
MODELS = {"qwen38": "Qwen3.8-27B", "qwen36": "Qwen3.6-35B-A3B", "qwen35": "Qwen3.5-9B"}
MODES = {"thinking": "Thinking", "nonthinking": "Non-thinking"}
ARMS = {"without_gt": "w/o GT", "with_gt": "w/ GT"}
SELECTORS = {"best": max, "random": None, "worst": min}
COLORS = {"best": "#17765c", "random": "#376ca6", "worst": "#b55643"}
GAIN_GRID = np.linspace(-100, 100, 801)
GAIN_BANDWIDTH = 7


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def archive_reward(row):
    if row.get("budget_failure"):
        assert float(row["pct"]) == 0
        return Fraction(0)
    pct = Fraction(int(row["passed_tests"]), int(row["total_tests"]))
    assert abs(float(pct) - float(row["pct"])) < 1e-12
    return pct


def read_banks():
    trials = defaultdict(dict)
    initial = defaultdict(dict)
    for row in read_csv(ARCHIVE / "task_scores.csv"):
        model, mode, task = (row[key] for key in ("model", "mode", "task_id"))
        if row["variant"] != "reflection" or model not in ("qwen38", "qwen36"):
            continue
        assert row["success"] in {"True", "False"}
        success = int(row["success"] == "True")
        if row["phase"] == "initial":
            repeat = int(row["source_repeat"])
            assert repeat not in initial[model, mode, task]
            initial[model, mode, task][repeat] = (archive_reward(row), success)
        elif row["phase"] == "independent":
            key = model, mode, row["arm"], task, int(row["source_repeat"])
            repeat = int(row["evaluation_repeat"])
            assert repeat not in trials[key]
            trials[key][repeat] = (archive_reward(row), success)
    assert len(trials) == 2000 and all(set(rows) == {0, 1, 2} for rows in trials.values())
    assert all(set(rows) == set(range(5)) for rows in initial.values())
    banks = defaultdict(dict)
    for (model, mode, arm, task, candidate), rows in trials.items():
        banks[model, mode, arm, task][candidate] = tuple(mean(v) for v in zip(*rows.values(), strict=True))
    for mode in MODES:
        run = RUN / mode / "reflection"
        assert json.loads((run / "status.json").read_text())["state"] == "completed"
        for row in read_csv(run / "analysis/reflection_final/candidate_effects.csv"):
            pct = Fraction(int(row["score_numerator"]), int(row["score_denominator"]))
            tgc = Fraction(int(row["official_success_count"]), 3)
            group = banks["qwen35", mode, row["arm"], row["task_id"]]
            repeat = int(row["source_repeat"])
            assert repeat not in group
            group[repeat] = (pct, tgc)
        for path in (run / "stage1").glob("*/result.json"):
            row = json.loads(path.read_text())
            group = initial["qwen35", mode, row["task_id"]]
            repeat = row["source_repeat"]
            assert repeat not in group and isinstance(row["success"], bool)
            group[repeat] = (archive_reward(row), int(row["success"]))
    assert all(set(rows) == set(range(5)) for rows in initial.values())
    task_ids = {task for model, mode, task in initial if (model, mode) == ("qwen38", "thinking")}
    assert len(task_ids) == 50 and len(banks) == 600
    initial_scores = {key: tuple(mean(v) for v in zip(*rows.values(), strict=True)) for key, rows in initial.items()}
    baselines = {}
    for model in MODELS:
        for mode in MODES:
            assert {t for m, md, t in initial_scores if (m, md) == (model, mode)} == task_ids
            baselines[model, mode] = tuple(
                mean(v) for v in zip(*(initial_scores[model, mode, task] for task in task_ids), strict=True)
            )
            for arm in ARMS:
                assert {t for m, md, a, t in banks if (m, md, a) == (model, mode, arm)} == task_ids
    assert all(set(rows) == set(range(5)) for rows in banks.values())
    return banks, baselines, initial_scores


def candidate_weights(group, selector):
    if selector is None:
        return list(group.values())
    target = selector(pct for pct, _ in group.values())
    return [scores for scores in group.values() if scores[0] == target]


def describe(distribution):
    ordered = sorted(distribution.items())
    total = sum(distribution.values())
    thresholds = iter((Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)))
    threshold = next(thresholds, None)
    quartiles = []
    running = Fraction(0)
    for score, weight in ordered:
        running += weight
        while threshold is not None and running >= threshold * total:
            quartiles.append(score)
            threshold = next(thresholds, None)
    return {key: float(value * 100) for key, value in zip(
        ("min", "q1", "median", "q3", "max", "mean", "perfect_share"),
        (ordered[0][0], *quartiles, ordered[-1][0],
         sum(score * weight for score, weight in ordered) / total, distribution.get(Fraction(1), 0) / total),
        strict=True,
    )}


def score_distributions(banks, *, strategies=tuple(SELECTORS), initial_scores=None):
    distributions = defaultdict(lambda: defaultdict(Fraction))
    for (model, mode, arm, task), group in banks.items():
        for name in strategies:
            selected = candidate_weights(group, SELECTORS[name])
            for pct, tgc in selected:
                scores = (("PCT", pct), ("TGC", tgc)) if initial_scores is None else (
                    ("PCT gain", pct - initial_scores[model, mode, task][0]),
                )
                weight = Fraction(1, len(selected))
                for metric, value in scores:
                    distributions[model, mode, arm, name, metric][value] += weight
                    distributions[model, "pooled", "pooled", name, metric][value] += weight
    return distributions


def summarize(distributions, *, filtered=False):
    result = {}
    for key, distribution in distributions.items():
        task_count = sum(distribution.values())
        if not filtered:
            assert task_count == (200 if key[1] == "pooled" else 50)
        stats = describe(distribution)
        if key[-1] == "PCT gain":
            del stats["perfect_share"]
        result[key] = {"task_conditions": int(task_count), **stats}
    return result


def candle(ax, x, stats, color):
    lo, q1, median, q3, hi = (stats[k] for k in ("min", "q1", "median", "q3", "max"))
    width = 0.49
    ax.vlines(x, lo, hi, color=color, linewidth=1.25, zorder=2)
    ax.hlines([lo, hi], x - 0.11, x + 0.11, color=color, linewidth=1.25, zorder=2)
    if q1 != q3:
        ax.add_patch(Rectangle((x - width / 2, q1), width, q3 - q1,
                               facecolor=color, alpha=0.23, edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((x - width / 2, q1), width, q3 - q1,
                               fill=False, edgecolor=color, linewidth=1.35, zorder=4))
    ax.hlines(median, x - width / 2, x + width / 2, color=color, linewidth=2.25, zorder=5)
    ax.scatter(x, stats["mean"], s=28, facecolor="white", edgecolor=color, linewidth=1.2, zorder=6)


def violin(ax, x, scores, color):
    # For PCT, every best/worst tie has the same score and contributes one Task.
    assert all(weight.denominator == 1 for weight in scores.values())
    if len(scores) > 1:
        observations = [float(score * 100) for score, weight in scores.items() for _ in range(int(weight))]
        parts = ax.violinplot(observations, positions=[x], widths=0.68, showextrema=False)
        for body in parts["bodies"]:
            body.set(facecolor=color, edgecolor=color, alpha=0.4, linewidth=1.4, zorder=3)
    ax.scatter(x, describe(scores)["median"], s=28, facecolor="white", edgecolor=color, linewidth=1.2, zorder=5)


def plot(values, baselines, metric, *, style="candles", cohort_counts=None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "semibold"})
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 12.1), sharey=True)
    filtered = cohort_counts is not None
    gain = metric == "PCT gain"
    strategies = ("best", "worst") if filtered else tuple(SELECTORS)
    width = len(strategies)
    positions = (*range(width), *range(width + 1, 2 * width + 1))
    centers = ((width - 1) / 2, (3 * width + 1) / 2)
    if gain:
        if style != "candles":
            lo = min(float(min(scores) * 100) for scores in values.values())
            hi = max(float(max(scores) * 100) for scores in values.values())
        else:
            lo = min(scores["min"] for scores in values.values())
            hi = max(scores["max"] for scores in values.values())
        limits = (lo - 5, hi + 5)
    else:
        limits = (-5, 106)
    for row, (model, model_label) in enumerate(MODELS.items()):
        for col, (mode, mode_label) in enumerate(MODES.items()):
            ax = axes[row, col]
            i = 0
            for arm in ARMS:
                for strategy in strategies:
                    scores = values.get((model, mode, arm, strategy, metric))
                    if scores and style == "dots":
                        ordered = sorted(scores.items(), reverse=True)
                        ax.scatter(
                            [positions[i]] * len(ordered), [float(pct * 100) for pct, _ in ordered],
                            s=[float(weight * 9) for _, weight in ordered], color=COLORS[strategy],
                            alpha=0.68, edgecolors="none", zorder=3,
                        )
                    elif scores and style == "violin":
                        violin(ax, positions[i], scores, COLORS[strategy])
                    elif scores:
                        candle(ax, positions[i], scores, COLORS[strategy])
                    i += 1
            if not filtered:
                initial = baselines[model, mode][0 if metric == "PCT" else 1] * 100
                ax.axhline(initial, linestyle=(0, (3, 3)), color="#737987", linewidth=1.1, zorder=1)
            elif gain:
                ax.axhline(0, linestyle=(0, (3, 3)), color="#737987", linewidth=1.1, zorder=1)
            ax.axvline(width, color="#e6e8ed", linewidth=1)
            ax.set(xlim=(-0.57, positions[-1] + 0.57), ylim=limits,
                   title=f"{model_label} · {mode_label}", xticks=positions, xticklabels=strategies * 2)
            ax.yaxis.set_major_locator(MultipleLocator(25))
            if gain:
                if col == 0:
                    ax.set_ylabel("PCT gain (pp)", color="#333946")
            else:
                ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.grid(axis="y", color="#e6e8ed", linewidth=0.8)
            ax.tick_params(axis="both", length=0, pad=5)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            for x, (arm, arm_label) in zip(centers, ARMS.items(), strict=True):
                if filtered:
                    arm_label += f" · n={cohort_counts.get((model, mode, arm), 0)}/50"
                ax.text(x, -0.18, arm_label, transform=ax.get_xaxis_transform(), ha="center",
                        va="top", color="#333946", weight="semibold")
    title = f"AppWorld Train · Single Reflection {metric}"
    subtitle = (
        "Tasks with nonzero candidate PCT variance · 5 candidates per task · 3 rollouts per candidate"
        if filtered else "50 tasks per panel and GT arm · 5 candidates per task · 3 rollouts per candidate"
    )
    if gain:
        subtitle = "Candidate mean PCT minus same-task initial mean PCT · Only nonzero-advantage groups"
    fig.suptitle(title, y=0.98, fontsize=19, weight="bold")
    fig.text(0.5, 0.946, subtitle, ha="center", color="#51576a", fontsize=10.5)
    if style == "dots":
        handles = [
            Line2D([0], [0], color="#51576a", alpha=0.68, linestyle="none", marker="o",
                   markersize=(9 * weight) ** 0.5, label=f"{weight} task{'s' if weight != 1 else ''}")
            for weight in (1, 5, 25)
        ]
    elif style == "violin":
        handles = [
            Patch(facecolor="#d7dfec", edgecolor="#376ca6", label="Smoothed density"),
            Line2D([0], [0], color="#376ca6", linestyle="none", marker="o", markerfacecolor="white", label="Median"),
        ]
    else:
        handles = [
            Patch(facecolor="#d7dfec", edgecolor="#376ca6", label="25th to 75th percentile"),
            Line2D([0], [0], color="#376ca6", linewidth=2.25, label="Median"),
            Line2D([0], [0], color="#376ca6", linewidth=1.25, marker="|", label="Min to max"),
            Line2D([0], [0], color="#376ca6", linestyle="none", marker="o", markerfacecolor="white", label="Mean"),
        ]
    if not filtered:
        handles.append(Line2D([0], [0], color="#737987", linestyle=(0, (3, 3)), label="Initial mean"))
    elif gain:
        handles.append(Line2D([0], [0], color="#737987", linestyle=(0, (3, 3)), label="No change"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.018))
    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.086, right=0.975, hspace=0.48, wspace=0.17)
    suffix = "nonzero_advantage_" + style if filtered else style
    metric_name = metric.lower().replace(" ", "_")
    fig.savefig(OUTPUT / f"reflection_{metric_name}_{suffix}.png", dpi=190, facecolor="white")
    plt.close(fig)


def write_summary(path, summary):
    fields = ("model", "mode", "arm", "strategy", "metric")
    rows = [{**dict(zip(fields, key, strict=True)), **value} for key, value in sorted(summary.items())]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def density_per_50(scores, grid, bandwidth):
    """Give each original task equal mass, including zero mass for filtered tasks."""
    gains = np.array([float(score * 100) for score in scores])
    weights = np.array([float(weight) for weight in scores.values()])
    offsets = (grid[:, None] - gains[None, :]) / bandwidth
    kernels = np.where(np.abs(offsets) <= 4, np.exp(-0.5 * offsets**2), 0)
    kernels /= np.trapz(kernels, grid, axis=0)
    density = kernels @ (weights / 50)
    assert np.isclose(np.trapz(density, grid), weights.sum() / 50, atol=1e-12)
    return density


def plot_split_gain(gains, densities, scale, *, filtered, show_random=False):
    grid = GAIN_GRID
    hatches = {"qwen38": "", "qwen36": "///", "qwen35": "..."}
    labels = {"qwen38": "3.8-27B", "qwen36": "3.6-35B", "qwen35": "3.5-9B"}
    plt.rcParams["hatch.linewidth"] = 0.65
    fig, ax = plt.subplots(figsize=(16.7, 7.5))
    positions, ticks = [], []
    random_fill = "#dfe2e6"
    random_edge = "#aab1ba"
    triplet_spacing = 3.90
    for group, (mode, arm) in enumerate((m, a) for m in MODES for a in ARMS):
        first = group * triplet_spacing
        ax.text(first + 1, 1.025, f"{MODES[mode]} · {ARMS[arm]}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=12, weight="semibold")
        if group:
            ax.axvline(first - (triplet_spacing - 2) / 2, color="#d8dfe8", linewidth=1)
        for idx, model in enumerate(MODELS):
            center = first + idx
            count = int(sum(gains.get((model, mode, arm, "best", "PCT gain"), {}).values()))
            positions.append(center)
            ticks.append(f"{labels[model]}\nn={count}/50")
            if show_random and count:
                density = densities[model, mode, arm, "random", "PCT gain"]
                assert np.isclose(np.trapz(density, grid), count / 50, atol=1e-12)
                width = scale * density
                ax.fill_betweenx(grid, center - width, center + width, where=density > 0, interpolate=True,
                                 facecolor=random_fill, edgecolor=random_edge, linewidth=0.75,
                                 alpha=0.66, zorder=1)
            for strategy, side in (("best", -1), ("worst", 1)):
                key = model, mode, arm, strategy, "PCT gain"
                if key not in densities:
                    continue
                color = COLORS[strategy]
                density = densities[key]
                assert np.isclose(np.trapz(density * scale, grid) / scale, count / 50, atol=1e-12)
                ax.fill_betweenx(grid, center, center + side * scale * density,
                                 where=density > 0, interpolate=True,
                                 facecolor=to_rgba(color, 0.2), edgecolor=color, linewidth=0.85,
                                 hatch=hatches[model], zorder=3)
                median = describe(gains[key])["median"]
                width_at_median = np.interp(median, grid, density) * scale
                ax.scatter(center + side * width_at_median * 0.50, median, s=16,
                           facecolors="white", edgecolors=color, linewidths=0.9, zorder=4)
    ax.axhline(0, color="#555e69", linewidth=1.25, linestyle=(0, (4, 3)), zorder=2)
    ax.set(xlim=(-0.69, positions[-1] + 0.69), ylim=(-100, 100),
           ylabel="PCT gain (percentage points)", xticks=positions, xticklabels=ticks)
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.grid(axis="y", color="#e6ebf1", linewidth=0.85)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.suptitle("AppWorld Train · Single Reflection PCT gain", y=0.971, fontsize=20, weight="bold")
    subtitle = "Best on the left  |  Worst on the right  ·  Half-violin area proportional to n / 50"
    if show_random:
        subtitle = "Best on the left  |  Worst on the right  |  Random in gray  ·  Half area proportional to n / 50"
    fig.text(0.5, 0.913, subtitle,
             ha="center", color="#333946", fontsize=11.5)
    handles = [
        Patch(facecolor="#eceff4", edgecolor="#555e69", hatch=hatches[model], label=label)
        for model, label in MODELS.items()
    ]
    handles += [
        Patch(facecolor=to_rgba(COLORS[strategy], 0.2), edgecolor=COLORS[strategy], label=strategy.capitalize())
        for strategy in ("best", "worst")
    ]
    if show_random:
        handles.append(Patch(facecolor=random_fill, edgecolor=random_edge, label="Random (same tasks)"))
    handles += [
        Line2D([0], [0], color="#555e69", linestyle="none", marker="o", markerfacecolor="white", label="Median"),
        Line2D([0], [0], color="#555e69", linestyle=(0, (4, 3)), label="No change"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.066))
    cohort = "All-zero advantage groups excluded" if filtered else "All 50 train tasks per condition"
    fig.text(0.5, 0.038,
             f"{cohort} · Gain measured against each task's initial mean · "
             f"Shared Gaussian bandwidth: {GAIN_BANDWIDTH} pp",
             ha="center", color="#555e69", fontsize=9)
    fig.subplots_adjust(left=0.073, right=0.982, top=0.83, bottom=0.20)
    stem = "reflection_pct_gain_split_violin" if filtered else "reflection_pct_gain_split_violin_all_tasks"
    suffix = "_with_random" if show_random else ""
    fig.savefig(OUTPUT / f"{stem}_n_over_50{suffix}.png", dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    banks, baselines, initial_scores = read_banks()
    distributions = score_distributions(banks)
    results = summarize(distributions)
    write_summary(OUTPUT / "distribution_summary.csv", results)
    for metric in ("PCT", "TGC"):
        plot(results, baselines, metric)
    plot(distributions, baselines, "PCT", style="dots")
    nonzero_groups = {key: group for key, group in banks.items() if len({pct for pct, _ in group.values()}) > 1}
    cohort_counts = defaultdict(int)
    for model, mode, arm, _task in nonzero_groups:
        cohort_counts[model, mode, arm] += 1
    nonzero_distributions = score_distributions(nonzero_groups, strategies=("best", "worst"))
    nonzero_summary = summarize(nonzero_distributions, filtered=True)
    write_summary(OUTPUT / "nonzero_advantage_summary.csv", nonzero_summary)
    plot(nonzero_distributions, None, "PCT", style="dots", cohort_counts=cohort_counts)
    plot(nonzero_summary, None, "PCT", cohort_counts=cohort_counts)
    gains = score_distributions(nonzero_groups, strategies=("best", "worst"), initial_scores=initial_scores)
    gain_summary = summarize(gains, filtered=True)
    write_summary(OUTPUT / "nonzero_advantage_gain_summary.csv", gain_summary)
    plot(gains, None, "PCT gain", style="dots", cohort_counts=cohort_counts)
    plot(gain_summary, None, "PCT gain", cohort_counts=cohort_counts)
    plot(gains, None, "PCT gain", style="violin", cohort_counts=cohort_counts)
    all_gains = score_distributions(banks, initial_scores=initial_scores)
    filtered_random = score_distributions(nonzero_groups, strategies=("random",), initial_scores=initial_scores)
    filtered_gains = {**gains, **filtered_random}
    all_densities, filtered_densities = (
        {key: density_per_50(scores, GAIN_GRID, GAIN_BANDWIDTH)
         for key, scores in distribution.items() if key[1] != "pooled"}
        for distribution in (all_gains, filtered_gains)
    )
    for key, density in filtered_densities.items():
        assert (density <= all_densities[key] + 1e-12).all()
    scale = 0.43 / max(density.max() for density in all_densities.values())
    plot_split_gain(filtered_gains, filtered_densities, scale, filtered=True, show_random=True)
    plot_split_gain(all_gains, all_densities, scale, filtered=False, show_random=True)
    with (OUTPUT / "pooled_summary.json").open("w") as stream:
        json.dump({"/".join(key): value for key, value in results.items() if key[1] == "pooled"}, stream, indent=2)
        stream.write("\n")
