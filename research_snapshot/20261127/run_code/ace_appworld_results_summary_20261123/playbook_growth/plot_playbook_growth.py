# SPDX-License-Identifier: Apache-2.0
"""Plot saved playbook character counts and effective update counts by Curator iteration."""

import json
import runpy
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

OUT = Path(__file__).resolve().parent
SPECS = runpy.run_path(str(OUT.parent / "build_report.py"))
SEEDS = SPECS["SEEDS"]
STRATEGIES = SPECS["STRATEGIES"]
MODEL_RUNS = SPECS["MODEL_RUNS"]
ADV_RUNS = SPECS["ADV_RUNS"]
ARMS = SPECS["ARMS"]
COLORS = {"best": "#17765c", "random": "#77818e", "worst": "#b55643",
          "best_advantage": "#269ba4", "worst_advantage": "#ca903c"}


def checkpoint_curve(directory):
    paths = sorted(directory.glob("[0-9][0-9][0-9].json"))
    if not paths:
        return None
    sizes, updates = [], []
    previous = None
    changed_count = 0
    for step, path in enumerate(paths):
        assert path.stem == f"{step:03d}", f"Checkpoint gap at {path}"
        row = json.loads(path.read_text())
        assert row["session_count"] == step
        playbook = row["playbook"]
        assert isinstance(playbook, str)
        if previous is not None:
            changed = playbook != previous
            changed_count += int(changed)
            assert len(playbook) >= len(previous), f"ADD-only playbook shrank at {path}"
            if "context_sha256" in row:
                assert changed == (row["context_sha256"] != row["parent_context_sha256"])
        sizes.append(len(playbook))
        updates.append(changed_count)
        previous = playbook
    return np.array([sizes, updates], dtype=float)


def collect(model):
    curves = {}
    advantage_run = ADV_RUNS.get(model)
    plans = json.loads((advantage_run / "plan.json").read_text()) if advantage_run else []
    for mode, run in MODEL_RUNS[model].items():
        for arm in ARMS:
            for strategy in STRATEGIES:
                seeds = []
                for seed in SEEDS:
                    cid = f"s{seed}_{arm}_{strategy}"
                    if strategy.endswith("_advantage"):
                        matches = [p for p in plans if p["condition"]["id"] == cid and
                                   Path(p["path"]).parent.name == mode]
                        if not matches:
                            continue
                        assert len(matches) == 1
                        path = Path(matches[0]["path"]) / "conditions" / cid / "checkpoints"
                    else:
                        path = run / "conditions" / cid / "checkpoints"
                    curve = checkpoint_curve(path)
                    if curve is not None:
                        seeds.append(curve)
                if seeds:
                    assert len(seeds) == 3 and len({c.shape for c in seeds}) == 1
                    curves[mode, arm, strategy] = np.stack(seeds)
    return curves


def plot(model, curves):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "semibold"})
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=True)
    max_size = max(float(a[:, 0, :].max()) for a in curves.values())
    for mode_index, mode in enumerate(MODEL_RUNS[model]):
        for col, arm in enumerate(ARMS):
            char_ax, count_ax = axes[2 * mode_index, col], axes[2 * mode_index + 1, col]
            char_ax.set_title(f"{SPECS['mode_label'](mode)} · {ARMS[arm]}", fontsize=13, pad=12)
            count_ax.text(0.025, 0.965, "Final updates · mean ± SD / iterations", transform=count_ax.transAxes,
                          ha="left", va="top", fontsize=8, color="#475363")
            for strategy_index, strategy in enumerate(STRATEGIES):
                samples = curves.get((mode, arm, strategy))
                if samples is None:
                    continue
                means, stds = samples.mean(axis=0), samples.std(axis=0, ddof=1)
                x = np.arange(samples.shape[-1])
                color = COLORS[strategy]
                linestyle = "--" if strategy.endswith("_advantage") else "-"
                count_ax.text(0.025, 0.895 - 0.058 * strategy_index,
                              f"{strategy}  {means[1, -1]:.2f} ± {stds[1, -1]:.2f} / {samples.shape[-1] - 1}",
                              transform=count_ax.transAxes, ha="left", va="top", fontsize=8, color=color)
                for metric_index, ax in enumerate((char_ax, count_ax)):
                    center, spread = means[metric_index], stds[metric_index]
                    ax.fill_between(x, np.maximum(0, center - spread), center + spread,
                                    facecolor=color, alpha=0.10, linewidth=0)
                    ax.plot(x, center, color=color, linewidth=2.0, linestyle=linestyle, zorder=3)
                    ax.scatter(x[-1], center[-1], s=24, color=color, zorder=4)
            char_ax.set_ylim(0, max_size * 1.05)
            char_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k" if v else "0"))
            count_ax.plot([0, 50], [0, 50], linestyle=":", color="#c1c7d0", linewidth=1.2, zorder=1)
            count_ax.set_ylim(0, 52)
            count_ax.yaxis.set_major_locator(MultipleLocator(10))
            char_ax.set_ylabel("Playbook size (characters)")
            count_ax.set_ylabel("Cumulative context updates")
            count_ax.set_xlabel("Curator iteration")
            for ax in (char_ax, count_ax):
                ax.set_xlim(0, 50)
                ax.xaxis.set_major_locator(MultipleLocator(10))
                ax.grid(color="#e4e9ef", linewidth=0.7, zorder=0)
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["left", "bottom"]].set_color("#c3cad3")
                ax.tick_params(length=0, labelbottom=True)
    handles = [Line2D([0], [0], color=COLORS[s], linewidth=2,
                      linestyle="--" if s.endswith("_advantage") else "-", label=s) for s in STRATEGIES]
    handles.append(Line2D([0], [0], color="#c1c7d0", linestyle=":", label="One update per iteration"))
    fig.suptitle(f"{SPECS['LABELS'][model]} · ADD-only playbook growth", fontsize=21, weight="bold", y=0.985)
    fig.text(0.5, 0.958, "Mean ± 1 sample SD across seeds 42, 43, 44", ha="center", fontsize=12, color="#475363")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.text(0.5, 0.018,
             "Iteration 0 is the initial playbook · Each changed playbook counts as one update · "
             "Advantage curves end after their selected training tasks",
             ha="center", fontsize=9, color="#586574")
    fig.subplots_adjust(left=0.085, right=0.98, top=0.885, bottom=0.065, hspace=0.32, wspace=0.22)
    target = OUT / f"{model}_playbook_size_and_updates_by_iteration.png"
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)
    return target


def main():
    for model in MODEL_RUNS:
        curves = collect(model)
        if not curves:
            print(json.dumps({"model": model, "state": "no accumulation checkpoints"}))
            continue
        target = plot(model, curves)
        final = {"/".join(key): {"steps": samples.shape[-1] - 1,
                                "chars_mean": round(float(samples[:, 0, -1].mean()), 1),
                                "updates_mean": round(float(samples[:, 1, -1].mean()), 2)}
                 for key, samples in curves.items()}
        print(json.dumps({"model": model, "figure": str(target), "final": final}))


if __name__ == "__main__":
    main()
