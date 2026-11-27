# SPDX-License-Identifier: Apache-2.0
"""Compare answer-only failure rates across completed ACE contexts."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, PercentFormatter

OUTPUT = Path(__file__).resolve().parent
MODELS = {"qwen38": "Qwen3.8-27B", "qwen36": "Qwen3.6-35B-A3B", "qwen35": "Qwen3.5-9B"}
MODES = {"thinking": "Thinking", "nonthinking": "Non-thinking"}
ARMS = {"without_gt": "w/o GT", "with_gt": "w/ GT"}
STRATEGIES = {"best": "#17765c", "random": "#65717c", "worst": "#b55643"}


def load_rates(*, seed_matched=False):
    with (OUTPUT / "by_condition.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    bins = defaultdict(list)
    for row in rows:
        if row["strategy"] == "initial":
            continue
        if seed_matched and (row["mode"], row["seed"]) != ("thinking", "42"):
            continue
        key = tuple(row[field] for field in ("model", "mode", "arm", "strategy"))
        bins[key].append(row)
    result = {}
    for key, values in bins.items():
        assert len(values) == len({r["seed"] for r in values})
        assert all(int(r["rollouts"]) == 50 for r in values)
        total, answer_only, unexpected_value = (
            sum(int(r[column]) for r in values)
            for column in ("rollouts", "only_answer_failure", "only_null_wrong_value")
        )
        assert 0 <= unexpected_value <= answer_only <= total
        result[key] = {
            "n": total, "seeds": tuple(sorted(r["seed"] for r in values)), "answer_only": 100 * answer_only / total,
            "unexpected_value": 100 * unexpected_value / total,
        }
    return result


def draw(rates, filename, *, seed_matched):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "semibold"})
    if seed_matched:
        fig, grid = plt.subplots(1, 3, figsize=(14.0, 6.5), sharey=True)
        axes = {(model, "thinking"): ax for model, ax in zip(MODELS, grid, strict=True)}
    else:
        fig, grid = plt.subplots(3, 2, figsize=(12.7, 11.5), sharey=True)
        axes = {(model, mode): grid[i, j] for i, model in enumerate(MODELS) for j, mode in enumerate(MODES)}
    y_top = max(40, 5 * ((max(r["answer_only"] for r in rates.values()) + 7) // 5))
    positions = (0, 1, 2, 4, 5, 6)
    for (model, mode), ax in axes.items():
        panel = [rates.get((model, mode, arm, method)) for arm in ARMS for method in STRATEGIES]
        valid = [r for r in panel if r is not None]
        ax.set_title(f"{MODELS[model]} · {MODES[mode]}", fontsize=11.7, pad=18)
        ax.set(xlim=(-0.65, 6.65), ylim=(0, y_top))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.tick_params(axis="both", length=0)
        ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e8edf1", linewidth=0.8)
        ax.set_axisbelow(True)
        if not valid:
            ax.set_xticks([])
            ax.text(0.5, 0.50, "Accumulated Train results pending",
                    transform=ax.transAxes, ha="center", va="center", color="#65717c")
            continue
        assert len(valid) == len(panel)
        assert len({(r["seeds"], r["n"]) for r in valid}) == 1
        ax.set(xticks=positions, xticklabels=list(STRATEGIES) * 2)
        seeds = ", ".join(valid[0]["seeds"])
        prefix = "Seed" if len(valid[0]["seeds"]) == 1 else "Seeds"
        ax.text(0.5, 1.015, f"{prefix} {seeds} · n={valid[0]['n']} per bar", ha="center", va="bottom",
                transform=ax.transAxes, color="#65717c", fontsize=8.8)
        ax.axvline(3, color="#dfe5ec", linewidth=0.9)
        for x, method, row in zip(positions, list(STRATEGIES) * 2, panel, strict=True):
            color = STRATEGIES[method]
            total = row["answer_only"]
            focus = row["unexpected_value"]
            ax.bar(x, total, width=0.63, color=to_rgba(color, 0.22), edgecolor=color, linewidth=0.65)
            if focus:
                ax.bar(x, focus, width=0.63, color=color)
            if focus >= 4:
                ax.text(x, focus / 2, f"{focus:.1f}%" if not focus.is_integer() else f"{focus:g}%",
                        ha="center", va="center", color="white", fontsize=7.4, weight="semibold")
            ax.text(x, total + 0.7, f"{total:g}%" if total.is_integer() else f"{total:.1f}%",
                    ha="center", va="bottom", color="#263546", fontsize=8)
        for x, label in zip((1, 5), ARMS.values(), strict=True):
            ax.text(x, -0.125, label, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    color="#344356", fontsize=9, weight="semibold")
    fig.suptitle("AppWorld Train · Completion answer errors after accumulation", fontsize=17, weight="bold", y=0.98)
    subtitle = "Matched Thinking seed 42" if seed_matched else "All completed seeds for each model and mode"
    fig.text(0.5, 0.916 if seed_matched else 0.94, subtitle, ha="center", color="#65717c")
    handles = [
        Patch(facecolor=to_rgba("#65717c", 0.22), edgecolor="#65717c", label="Full bar: only answer check failed"),
        Patch(facecolor="#65717c", label="Dark segment: submitted a value where null was expected"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025), frameon=False, ncol=2, fontsize=9)
    fig.subplots_adjust(top=0.80 if seed_matched else 0.86, bottom=0.21 if seed_matched else 0.10, left=0.07,
                        right=0.98, hspace=0.58, wspace=0.18)
    fig.savefig(OUTPUT / filename, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw(load_rates(), "completion_error_rates_by_selection.png", seed_matched=False)
    draw(load_rates(seed_matched=True), "completion_error_rates_seed42_matched.png", seed_matched=True)
