# SPDX-License-Identifier: Apache-2.0
"""TGC counterpart of the matched initial/pass@5/independent/accumulated PCT bars."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
ORIGINAL = OUT.parent
REFERENCE = ORIGINAL / "plot_model_strategy_bars.py"
SPEC = importlib.util.spec_from_file_location("pct_bar_reference", REFERENCE)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
BASE.OUT = OUT
NAME = "model_strategy_initial_pass5_independent_accumulated_tgc_4panel"
TICK_LABELS = ("Initial\nTGC", "Pass\n@5", "Best", "Random", "Worst", "Best\nadv.", "Worst\nadv.")


def verify_cached(initial, by_seed, summary):
    with (ORIGINAL / "metrics_by_seed.csv").open(newline="") as stream:
        cached_seed = {(r["model"], r["mode"], r["arm"], r["strategy"], int(r["seed"])): r
                       for r in csv.DictReader(stream)}
    assert len(cached_seed) == len(by_seed) == 300
    for row in by_seed:
        cached = cached_seed[row["model"], row["mode"], row["arm"], row["strategy"], row["seed"]]
        for field in ("independent_tgc", "accumulated_tgc"):
            assert abs(row[field] - float(cached[field])) < 1e-12
    with (ORIGINAL / "metrics_mean_std.csv").open(newline="") as stream:
        cached_summary = {(r["model"], r["mode"], r["arm"], r["strategy"]): r for r in csv.DictReader(stream)}
    assert len(cached_summary) == len(summary) == 100
    for key, row in summary.items():
        for variant in ("independent", "accumulated"):
            for stat in ("mean", "std"):
                field = f"{variant}_tgc_{stat}"
                assert abs(row[field] - float(cached_summary[key][field])) < 1e-12
    with (ORIGINAL / "initial_metrics.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            for field in ("tgc", "pass_at_5"):
                assert abs(initial[row["model"], row["mode"]][field] - float(row[field])) < 1e-12


def draw(initial, summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titleweight": "semibold", "hatch.linewidth": 0.7,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(43, 17.5), sharey=True)
    plotted = []
    for row in range(2):
        for col, arm in enumerate(BASE.ARMS):
            ax = axes[row, col]
            ticks, labels = [], []
            for index, model in enumerate(BASE.MODELS):
                mode = tuple(BASE.DATA.MODEL_RUNS[model])[row]
                start = index * BASE.STRIDE
                if index % 2 == 0:
                    ax.axvspan(start - 0.65, start + 9.50, color="#f3f5f8", alpha=0.8, zorder=0)
                if index:
                    ax.axvline(start - 0.83, color="#d3dbe5", linewidth=0.85, linestyle=(0, (3, 3)), zorder=1)
                baseline = initial[model, mode]
                for category, offset in (("initial", BASE.TICK_OFFSETS[0]), ("pass_at_5", BASE.TICK_OFFSETS[1])):
                    metric = "tgc" if category == "initial" else "pass_at_5"
                    value = baseline[metric] * 100
                    BASE.bar(ax, start + offset, value, BASE.COLORS[category])
                    plotted.append({"row": row, "arm": arm, "model": model, "mode": mode,
                                    "category": category, "variant": "baseline", "metric": metric,
                                    "x": start + offset, "mean_percent": value, "std_percent": "",
                                    "n_training_seeds": "", "initial_rollouts_per_task": 5})
                for number, strategy in enumerate(BASE.STRATEGIES):
                    record = summary[model, mode, arm, strategy]
                    for variant, delta in (("independent", -BASE.WIDTH / 2), ("accumulated", BASE.WIDTH / 2)):
                        x = start + BASE.TICK_OFFSETS[number + 2] + delta
                        value, std = (100 * record[f"{variant}_tgc_{name}"] for name in ("mean", "std"))
                        BASE.bar(ax, x, value, BASE.COLORS[strategy], std=std, independent=variant == "independent")
                        plotted.append({"row": row, "arm": arm, "model": model, "mode": mode,
                                        "category": strategy, "variant": variant, "metric": "tgc", "x": x,
                                        "mean_percent": value, "std_percent": std, "n_training_seeds": 3,
                                        "initial_rollouts_per_task": ""})
                ticks.extend(start + np.array(BASE.TICK_OFFSETS))
                labels.extend(TICK_LABELS)
                ax.text(start + 4.40, -0.18, BASE.LABELS[model], transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=14, fontweight="bold", color="#263244")
            ax.set_xticks(ticks, labels, fontsize=8.7)
            ax.tick_params(axis="x", length=0, pad=8)
            ax.tick_params(axis="y", labelsize=11, length=0, pad=6, labelleft=True)
            ax.set_xlim(-0.70, (len(BASE.MODELS) - 1) * BASE.STRIDE + 9.55)
            ax.set_ylim(0, 114)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.set_ylabel("Official TGC / pass@5 (%)", fontsize=13, labelpad=10)
            ax.grid(axis="y", color="#dfe5ec", linewidth=0.8, zorder=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#c5ceda")
            label = "w/o GT" if arm == "without_gt" else "w/ GT"
            letter = "ABCD"[row * 2 + col]
            ax.set_title(f"{letter}   {BASE.MODE_TITLES[row]}  ·  {label}", loc="left", fontsize=19, pad=19)
    handles = [Patch(facecolor=BASE.COLORS["initial"], label="Initial TGC"),
               Patch(facecolor=BASE.COLORS["pass_at_5"], label="Initial pass@5 (official TGC)"),
               Patch(facecolor=BASE.tinted("#657488"), edgecolor="#657488", hatch="///",
                     label="Independent: selected Reflection"),
               Patch(facecolor="#657488", label="Accumulated: final Context")]
    fig.suptitle("AppWorld Train · TGC before learning and by Reflection selection strategy",
                 y=0.982, fontsize=26, fontweight="bold", color="#1f2c3e")
    fig.text(0.5, 0.951, "50 matched training tasks  ·  Each model: initial TGC, pass@5, "
             "then five independent / accumulated TGC pairs",
             ha="center", fontsize=14, color="#536176")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.933), ncol=4,
               frameon=False, fontsize=12, handlelength=2.6, columnspacing=3.0)
    fig.text(0.5, 0.078,
             "Strategy bars: mean ± 1 sample SD across seeds 42, 43, 44; every seed uses all 50 Train tasks. "
             "Initial TGC: official success rate over 250 rollouts. pass@5: tasks with ≥1 official success / 50.",
             ha="center", fontsize=12, color="#536176")
    fig.text(0.5, 0.055,
             "Independent: actual seed-selected Reflection, measured on its saved 3-rollout bank. "
             "Accumulated: one final-Context rollout per task per seed. "
             "Advantage strategies: skipped zero-signal tasks retain their initial TGC in the independent expectation.",
             ha="center", fontsize=11.5, color="#536176")
    fig.text(0.5, 0.033,
             "TGC uses official boolean success. Candidate selection is still based on PCT, "
             "with the same saved choices as the PCT figure. "
             "Initial and pass@5 are shared by both GT panels and have no error bars. adv. = advantage.",
             ha="center", fontsize=11.5, color="#536176")
    fig.subplots_adjust(left=0.035, right=0.99, top=0.855, bottom=0.20, hspace=0.55, wspace=0.055)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{NAME}.{extension}", dpi=200, facecolor="white")
    fig.savefig(OUT / f"{NAME}_preview.png", dpi=90, facecolor="white")
    plt.close(fig)
    assert len(plotted) == 240
    for row in range(2):
        for arm in BASE.ARMS:
            subset = [r for r in plotted if r["row"] == row and r["arm"] == arm]
            assert len(subset) == 60
            assert [r["model"] for r in subset[::12]] == list(BASE.MODELS)
            assert all(0 <= r["mean_percent"] <= 100 + 1e-9 for r in subset)
    BASE.export_csv("plotted_bars.csv", plotted)


def main():
    original_files = [ORIGINAL / ("model_strategy_initial_pass5_independent_accumulated_pct_4panel." + ext)
                      for ext in ("png", "pdf", "svg")]
    unchanged = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in original_files}
    initial, by_seed, summary = BASE.collect()
    verify_cached(initial, by_seed, summary)
    BASE.export_csv("initial_metrics.csv", list(initial.values()))
    BASE.export_csv("metrics_by_seed.csv", by_seed)
    BASE.export_csv("metrics_mean_std.csv", list(summary.values()))
    draw(initial, summary)
    for path, sha in unchanged.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(), "model_order": list(BASE.MODELS),
        "strategy_order": list(BASE.STRATEGIES), "panel_rows": list(BASE.MODE_TITLES),
        "panel_columns": list(BASE.ARMS), "initial_banks": 10, "complete_seed_contexts": 300,
        "strategy_groups": 100, "plotted_bars": 240,
        "metric": "Official boolean success, never inferred by rounding or thresholding PCT",
        "initial": "Mean official success over 250 initial rollouts: 50 Train tasks, 5 repeats",
        "pass_at_5": "Fraction of 50 Train tasks with any official success in 5 initial rollouts",
        "strategy": "Task macro mean over Train50, then mean over seeds 42/43/44",
        "std": "Sample SD across three seed-level Train means, ddof=1; strategy bars only",
        "selection": "Unchanged saved selections based on the original candidate mean PCT",
        "independent": "Actual seed-selected candidate's three saved official success outcomes; "
                       "skipped advantage tasks use their initial five-rollout TGC",
        "accumulated": "One measured final-Context Train rollout per task per seed",
        "counterfactual": False, "new_model_calls": 0,
        "validation": "Recomputed from saved outcomes and checked against all 300 cached TGC seed results",
        "original_pct_images_unchanged": unchanged,
        "source_sha256": BASE.DATA.SOURCES,
        "helper_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / (NAME + ".png")), "seed_contexts": 300, "bars": 240}))


if __name__ == "__main__":
    main()
