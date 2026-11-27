# SPDX-License-Identifier: Apache-2.0
"""Add all-candidate PCT gains and matched overlap/gap scatters to the Gemma31 bundle."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.dont_write_bytecode = True
ROOT = Path('/home/thskadud/Sico/labs/AgentStream/exgentic')
SUMMARY = ROOT / "outputs/ace_appworld_results_summary_20261123"
OUT = SUMMARY / "qwen_gemma26b_31b_figures_20261127"
OLD = SUMMARY / "overlap_vs_accumulation_gap_20261124/qwen_gemma26b_wo_gt_20261125"
MODELS = ("qwen38", "qwen36", "qwen35", "gemma26", "gemma31")
MODES = ("thinking", "nonthinking")
COLORS = dict(zip(MODELS, ("#3973ac", "#218a69", "#d98b2b", "#8b67b1", "#b95c58"), strict=True))
MARKERS = {"thinking": "o", "nonthinking": "s"}
SOURCES = {}
FIGURES = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    SOURCES[str(path)] = digest(path)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read_csv(path):
    SOURCES[str(path)] = digest(path)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=200, facecolor="white")
        FIGURES.append(path.name)
    path = OUT / f"{name}_preview.png"
    fig.savefig(path, dpi=100, facecolor="white")
    FIGURES.append(path.name)
    plt.close(fig)


def collect_candidates(base):
    candidates, distributions = [], defaultdict(lambda: {s: [] for s in ("best", "worst", "random")})
    for model in MODELS:
        for mode in MODES:
            if model.startswith("gemma"):
                base.load_gemma(model, mode)
            else:
                base.REF.load_reflection(model, mode, base.REF.MODEL_RUNS[model][mode])
            tasks = sorted(base.REF.INITIAL[model, mode])
            assert len(tasks) == 50 and tasks == sorted(base.REF.INITIAL["qwen38", "thinking"])
            for task in tasks:
                before = base.REF.INITIAL[model, mode][task]
                assert len(before) == 5 and all(type(r["success"]) is bool for r in before)
                initial_pct = sum((base.REF.exact_reward(r) for r in before), Fraction()) / 5
                initial_tgc = Fraction(sum(r["success"] for r in before), 5)
                bank = base.REF.BANKS[model, mode]["without_gt", task]
                pcts = [bank[i]["exact_pct"] for i in range(5)]
                tgcs = []
                for i in range(5):
                    rows = bank[i]["rows"]
                    assert len(rows) == 3 and all(type(r["success"]) is bool for r in rows)
                    assert {r["evaluation_repeat"] for r in rows} == {0, 1, 2}
                    tgcs.append(Fraction(sum(r["success"] for r in rows), 3))
                center = sum(tgcs, Fraction()) / 5
                sigma = math.sqrt(float(sum(((x - center) ** 2 for x in tgcs), Fraction()) / 5))
                advs = [float(x - center) / (sigma + 1e-8) for x in tgcs]
                assert abs(sum(advs)) < 1e-12
                for i in range(5):
                    candidates.append({"model": model, "mode": mode, "task_id": task, "source_repeat": i,
                                       "candidate_mean_pct": float(pcts[i]), "initial_mean_pct": float(initial_pct),
                                       "pct_gain_pp": 100 * float(pcts[i] - initial_pct),
                                       "pct_gain_is_exact_zero": pcts[i] == initial_pct,
                                       "candidate_mean_tgc": float(tgcs[i]), "initial_mean_tgc": float(initial_tgc),
                                       "tgc_gain_pp": 100 * float(tgcs[i] - initial_tgc),
                                       "tgc_grpo_advantage": advs[i]})
                pools = {"best": [i for i, x in enumerate(pcts) if x == max(pcts)],
                         "worst": [i for i, x in enumerate(pcts) if x == min(pcts)], "random": list(range(5))}
                for strategy, pool in pools.items():
                    for i in pool:
                        distributions["gain", mode, model][strategy].append(
                            (100 * float(tgcs[i] - initial_tgc), 1 / len(pool)))
                        distributions["advantage", mode, model][strategy].append((advs[i], 1 / len(pool)))
    assert len(candidates) == 2500
    existing = read_csv(OUT / "data/selected_reflection/candidate_scores_and_advantages.csv")
    indexed = {(r["model"], r["mode"], r["task_id"], r["source_repeat"]): r for r in candidates}
    for row in existing:
        other = indexed[row["model"], row["mode"], row["task_id"], int(row["source_repeat"])]
        assert abs(float(row["mean_pct"]) - other["candidate_mean_pct"]) < 1e-12
    write_csv(OUT / "data/all_candidate_gain/candidate_rewards_and_gains.csv", candidates)
    return candidates, distributions


def all_gain(base, candidates):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    grid = base.OVERLAP.GRID
    groups = {(m, mode): [r for r in candidates if r["model"] == m and r["mode"] == mode]
              for m in MODELS for mode in MODES}
    densities = {key: base.REF.density([(r["pct_gain_pp"], .2) for r in rows], grid)
                 for key, rows in groups.items()}
    assert all(len(v) == 250 for v in groups.values())
    scale = .43 / max(v.max() for v in densities.values())
    fig, axes = plt.subplots(1, 2, figsize=(21, 6.5), sharey=True)
    summaries = []
    positions = np.arange(5) * 1.35
    for ax, mode in zip(axes, MODES, strict=True):
        ax.axhline(0, color="#a77d40", lw=1, linestyle=(0, (4, 3)), zorder=1)
        for x, model in zip(positions, MODELS, strict=True):
            rows = groups[model, mode]
            values = np.array([r["pct_gain_pp"] for r in rows])
            density = densities[model, mode]
            assert np.isclose(np.trapz(density, grid), 1)
            width = density * scale
            ax.fill_betweenx(grid, x - width, x + width, where=width > 0,
                             facecolor="#487f9b", edgecolor="#2d536b",
                             alpha=.84, lw=.9, zorder=3)
            positive = float((values >= 0).mean()) * 100
            zero = sum(r["pct_gain_is_exact_zero"] for r in rows) / 250 * 100
            ax.text(x, 119, f"Gain ≥ 0: {positive:.1f}%", ha="center", fontsize=9.6,
                    color="#285e4a", fontweight="semibold")
            ax.text(x, 108, f"Gain = 0: {zero:.1f}%", ha="center", fontsize=9.1, color="#795b2e")
            summaries.append({"model": model, "mode": mode, "tasks": 50, "candidates": 250,
                              "mean_gain_pp": float(values.mean()), "median_gain_pp": float(np.median(values)),
                              "nonnegative_percent": positive, "exact_zero_percent": zero})
        ax.set(xlim=(-.65, positions[-1] + .65), ylim=(-107, 128))
        ax.set_yticks(np.arange(-100, 101, 25))
        ax.set_xticks(positions, [base.LABELS[m] for m in MODELS], fontsize=11)
        ax.tick_params(length=0, labelleft=True, pad=7)
        ax.set_ylabel("PCT gain (percentage points)", fontsize=12)
        ax.grid(axis="y", color="#e3e9ef", lw=.75)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#c9d3dd")
        ax.set_title(("Thinking" if mode == "thinking" else "Non-thinking") + " · w/o GT",
                     fontsize=15, fontweight="semibold", loc="left", pad=14)
    fig.suptitle("AppWorld · PCT gain distribution of generated Reflections", fontsize=21,
                 fontweight="bold", y=.985, color="#233348")
    fig.text(.5, .915, "All 250 Reflections per model / panel · 50 matched tasks, 5 candidates each · "
             "Every candidate included once", ha="center", fontsize=11.5, color="#526176")
    fig.text(.5, .074, "Gain = candidate three-rollout mean PCT - the same task's initial five-rollout mean PCT. "
             "Exact zero gains retained; no duplication across selection seeds.",
             ha="center", fontsize=9.8, color="#526176")
    fig.text(.5, .043, "Shared density-to-width scale; Gaussian bandwidth 7 pp; "
             "bounded KDE on [-100, 100]. Both Gemma models have w/o GT data only.",
             ha="center", fontsize=9.8, color="#526176")
    fig.subplots_adjust(left=.065, right=.985, top=.80, bottom=.23, wspace=.19)
    save(fig, "reflection_pct_gain_all_candidates_qwen_gemma26b_31b_wo_gt_2panel")
    write_csv(OUT / "data/all_candidate_gain/distribution_summary.csv", summaries)
    np.savez_compressed(OUT / "data/all_candidate_gain/densities.npz", grid=grid,
                        **{f"{m}_{mode}": d for (m, mode), d in densities.items()})


def tgc_overlaps(base, distributions):
    results, arrays = [], {}
    for (metric, mode, model), points in distributions.items():
        grid = base.ADV.GRID if metric == "advantage" else base.OVERLAP.GRID
        ds = {s: base.ADV.density(p) if metric == "advantage" else base.REF.density(p, grid)
              for s, p in points.items()}
        assert all(np.isclose(sum(w for _, w in p), 50) for p in points.values())
        assert all(np.isclose(np.trapz(d, grid), 1) for d in ds.values())
        overlap = 100 * sum(base.OVERLAP.intersection(ds[s], ds["random"], grid) for s in ("best", "worst")) / (
            2 * base.OVERLAP.area(ds["random"], grid))
        results.append({"model": model, "mode": mode, "metric": metric,
                        "overlap_of_random_percent": overlap, "reward": "TGC", "selection": "PCT unchanged"})
        arrays.update({f"tgc_{metric}_{model}_{mode}_{s}": d for s, d in ds.items()})
    old = SUMMARY / "reflection_gain_with_oss120b_20261123/qwen_gemma26b_wo_gt_tgc_reward_20261125/source_densities.npz"
    SOURCES[str(old)] = digest(old)
    with np.load(old) as data:
        checked = 0
        for key in data.files:
            if key in arrays:
                assert np.allclose(data[key], arrays[key], rtol=0, atol=1e-12), key
                checked += 1
    assert checked == 48
    folder = OUT / "data/overlap_scatter"
    folder.mkdir(exist_ok=True)
    write_csv(folder / "tgc_overlap_and_means.csv", results)
    np.savez_compressed(folder / "tgc_overlap_densities.npz", **arrays)
    return {(r["model"], r["mode"], r["metric"]): r["overlap_of_random_percent"] for r in results}


def gaps(metric):
    rows = read_csv(OUT / "data/accumulated/metrics_by_seed.csv")
    indexed = {(r["model"], r["mode"], int(r["seed"]), r["strategy"]): r
               for r in rows if r["arm"] == "without_gt"}
    settings, pairs = {}, []
    for model in MODELS:
        for mode in MODES:
            values = []
            for seed in (42, 43, 44):
                selected = [indexed[model, mode, seed, s] for s in ("best", "worst")]
                assert all(int(r["train_tasks"]) == int(r["learned_tasks"]) == 50 for r in selected)
                best, worst = [100 * float(r[f"accumulated_{metric}"]) for r in selected]
                values.append(best - worst)
                pairs.append({"model": model, "mode": mode, "seed": seed,
                              "best_percent": best, "worst_percent": worst, "gap_pp": best - worst})
            settings[model, mode] = {"gap_mean_pp": mean(values), "gap_std_pp": stdev(values)}
    assert len(pairs) == 30
    write_csv(OUT / f"data/overlap_scatter/{metric}_paired_seed_gaps.csv", pairs)
    return settings


def scatter(reference, labels, overlaps, setting_gaps, x_reward, y_metric, name, old_csv):
    points = [{"model": model, "mode": mode, "arm": "without_gt", "overlap_metric": metric,
               "overlap_percent": overlaps[model, mode, metric], **setting_gaps[model, mode],
               "x_reward": x_reward, "y_metric": y_metric, "selection": "PCT unchanged"}
              for model in MODELS for mode in MODES for metric in ("gain", "advantage")]
    indexed = {(p["model"], p["mode"], p["overlap_metric"]): p for p in points}
    for old in read_csv(old_csv):
        if old["model"] in ('gemma26',):
            continue  # Updated cohorts are checked against new outcomes.
        p = indexed[old["model"], old["mode"], old["overlap_metric"]]
        for field in ("overlap_percent", "gap_mean_pp", "gap_std_pp"):
            assert abs(p[field] - float(old[field])) < 1e-10, (field, old, p)
    write_csv(OUT / f"data/overlap_scatter/{name}_points.csv", points)
    correlations = {}
    for metrics, suffix in [(("gain", "advantage"), "2panel"), (("gain",), "gain"),
                            (("advantage",), "grpo")]:
        fig, axes = plt.subplots(1, len(metrics), figsize=(18 if len(metrics) == 2 else 11, 8.3),
                                 squeeze=False, sharey=True)
        low = min(p["gap_mean_pp"] - p["gap_std_pp"] for p in points)
        high = max(p["gap_mean_pp"] + p["gap_std_pp"] for p in points)
        pad = .15 * (high - low)
        for metric, ax in zip(metrics, axes[0], strict=True):
            ps = [p for p in points if p["overlap_metric"] == metric]
            stats = reference.ORIGINAL.correlation(ps)
            correlations[metric] = stats
            for p in ps:
                color = COLORS[p["model"]]
                ax.errorbar(p["overlap_percent"], p["gap_mean_pp"], yerr=p["gap_std_pp"], fmt="none",
                            color=color, alpha=.4, elinewidth=1.4, capsize=3.5, zorder=2)
                ax.scatter(p["overlap_percent"], p["gap_mean_pp"], s=125, marker=MARKERS[p["mode"]],
                           color=color, edgecolor="white", linewidth=.9, zorder=4)
            left, right = min(p["overlap_percent"] for p in ps) - 3, max(p["overlap_percent"] for p in ps) + 3
            xs = np.linspace(left, right, 200)
            ax.plot(xs, stats["ols_intercept_gap_pp"] + stats["ols_slope_gap_pp_per_overlap_percentage_point"] * xs,
                    color="#29384b", lw=1.8, linestyle=(0, (5, 3)))
            ax.axhline(0, color="#99a6b6", lw=1.05)
            label = f"{x_reward} gain overlap" if metric == "gain" else f"{x_reward}-reward GRPO overlap"
            ax.set(xlim=(left, right), ylim=(low - pad, high + pad), xlabel=label + " with Random (%)",
                   title=label)
            ax.grid(color="#e4e9f0", lw=.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#c9d1dc")
            ax.text(.97, .97, f"Pearson r = {stats['pearson_r']:+.3f}\n"
                    f"Spearman $\\rho$ = {stats['spearman_rho']:+.3f}\n10 settings / 5 models",
                    transform=ax.transAxes, ha="right", va="top", fontsize=11.5,
                    bbox={"facecolor": "white", "edgecolor": "#dbe3ec", "boxstyle": "round,pad=.55"})
        axes[0, 0].set_ylabel(f"Best - Worst accumulated Train {y_metric} (percentage points)", labelpad=12)
        handles = [Line2D([0], [0], marker="o", linestyle="none", color=COLORS[m], label=labels[m], markersize=8)
                   for m in MODELS]
        modes = [Line2D([0], [0], marker=MARKERS[m], linestyle="none", color="#4a586b", markersize=8,
                        label="Thinking" if m == "thinking" else "Non-thinking") for m in MODES]
        modes.append(Line2D([0], [0], color="#29384b", linestyle="--", label="OLS descriptive trend"))
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .112))
        fig.legend(handles=modes, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .070))
        fig.suptitle(f"{x_reward} reflection overlap vs. accumulation {y_metric} gap · w/o GT",
                     fontsize=20 if len(metrics) == 2 else 15.5, weight="bold", y=.976, color="#233247")
        fig.text(.5, .918, "Qwen3.8 / Qwen3.6 / Qwen3.5 / Gemma26 / Gemma31 · "
                 "PCT-selected plain Best / Worst contexts held fixed", ha="center", fontsize=10.5, color="#526176")
        fig.text(.5, .041, "Y = mean paired seed gap (42, 43, 44) ±1 sample SD · Train50 per seed · "
                 "Same Y on both overlap axes", ha="center", fontsize=9.3, color="#526176")
        fig.text(.5, .016, 'Gemma26/31 include 115/52 selected-case reruns; other outcomes retained. Selection bias applies; correlations are descriptive.', ha="center", fontsize=9.3, color="#526176")
        fig.subplots_adjust(left=.075 if len(metrics) == 2 else .11, right=.985,
                            top=.835, bottom=.255, wspace=.12)
        save(fig, name + "_" + suffix)
    return correlations


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    originals = {str(p): digest(p) for p in OLD.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    baseline = json.loads((OUT / "verification.json").read_text())["figures"]
    previous_figures = {str(OUT / name): digest(OUT / name) for name in baseline}
    base = module("completed_five_model_reflections", OUT / "source_snapshot/plot_selected_reflection.py")
    reference = module("original_qwen_gemma_overlap_scatter", OLD / "plot_scatter.py")
    candidates, distributions = collect_candidates(base)
    all_gain(base, candidates)
    print("All-candidate gain violin rendered", flush=True)
    tgc_x = tgc_overlaps(base, distributions)
    pct_x = {(r["model"], r["mode"], r["metric"]): float(r["overlap_of_random_percent"])
             for r in read_csv(OUT / "data/selected_reflection/overlap_and_means.csv")}
    pct_y, tgc_y = gaps("pct"), gaps("tgc")
    labels = {**reference.LABELS, "gemma31": "Gemma4-31B"}
    families = [
        (pct_x, pct_y, "PCT", "PCT", "pct_overlap_vs_accumulation_pct_gap", OLD / "scatter_points.csv"),
        (pct_x, tgc_y, "PCT", "TGC", "pct_overlap_vs_accumulation_tgc_gap", OLD / "tgc/scatter_points.csv"),
        (tgc_x, tgc_y, "TGC", "TGC", "tgc_overlap_vs_accumulation_tgc_gap",
         OLD / "tgc_reward_overlap/scatter_points.csv"),
    ]
    correlations = {}
    for overlaps, ys, x, y, name, old_csv in families:
        correlations[name] = scatter(reference, labels, overlaps, ys, x, y, name, old_csv)
        print("Rendered " + name, flush=True)
    assert all(digest(Path(p)) == sha for p, sha in originals.items())
    assert all(digest(Path(p)) == sha for p, sha in previous_figures.items())
    snapshot = OUT / "source_snapshot" / Path(__file__).name
    shutil.copy2(__file__, snapshot)
    manifest = {"generated_at": datetime.now(UTC).isoformat(), "models": list(MODELS),
                "candidates": len(candidates), "reflections_per_model_mode": 250, "tasks_per_model": 50,
                "figures": FIGURES, "correlations_descriptive": correlations,
                "old_scatter_points_unchanged": "All x, y, paired gap SDs for the old8 settings checked per variant",
                "old_tgc_density_arrays_matched": 48, "error_bars": "Sample SD of three paired Best-Worst seed gaps",
                "selection": "Existing exact PCT best/worst pools and uniform ties, never reselected by TGC",
                "all_candidate_gain": "Every G5 candidate counted once; exact Fraction PCT difference "
                                      "from task initialG5",
                "new_inference_calls": 0, "original_files_preserved": originals,
                "previous_bundle_figures_preserved": previous_figures,
                "source_sha256": {**base.REF.SOURCES, **base.SOURCES, **SOURCES},
                "script_sha256": digest(Path(__file__))}
    (OUT / "scatter_and_gain_verification.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    note = "\nAdded: all-candidate PCT gain violin, and overlap-gap scatter families " \
           "PCT/PCT, PCT/TGC, TGC/TGC (gain+GRPO combined and separate).\n"
    if note not in (OUT / "README.txt").read_text():
        with (OUT / "README.txt").open("a") as stream:
            stream.write(note)
    print(json.dumps({"output": str(OUT), "figures_added": len(FIGURES), "verification": True}, indent=2))


if __name__ == "__main__":
    main()
