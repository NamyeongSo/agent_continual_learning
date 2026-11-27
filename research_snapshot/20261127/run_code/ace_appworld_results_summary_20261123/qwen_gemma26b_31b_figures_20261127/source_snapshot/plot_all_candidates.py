# SPDX-License-Identifier: Apache-2.0
"""Replace OSS with Gemma26 in saved all-candidate GRPO advantage violins without inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path('/home/thskadud/Sico/labs/AgentStream/exgentic')
MODELS = ("qwen38", "qwen36", "qwen35", "gemma26", "gemma31")
MODES = ("thinking", "nonthinking")
ARMS = ("without_gt", "with_gt")
LABELS = {"qwen38": "Qwen3.8\n27B", "qwen36": "Qwen3.6\n35B-A3B", "qwen35": "Qwen3.5\n9B",
          "gemma26": "Gemma4\n26B-A4B", "gemma31": "Gemma4\n31B"}
EPSILON = 1e-8
BANDWIDTH = .12
GRID = np.linspace(-2, 2, 1001)
COLOR = "#487f9b"


def density(values):
    kernel = np.zeros_like(GRID)
    for centers in (values, -4 - values, 4 - values):
        z = (GRID[:, None] - centers[None, :]) / BANDWIDTH
        kernel += np.exp(-.5 * z * z).sum(axis=1)
    result = kernel / (len(values) * BANDWIDTH * math.sqrt(2 * math.pi))
    result /= np.trapz(result, GRID)
    assert np.isclose(np.trapz(result, GRID), 1, atol=1e-12)
    return result


def reward(row):
    if row.get("budget_failure"):
        assert row["pct"] == 0
        return Fraction()
    assert row["total_tests"] > 0
    value = Fraction(row["passed_tests"], row["total_tests"])
    assert abs(float(value) - row["pct"]) < 1e-12
    return value


def csv_file(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(original, gemma, gemma31):
    sources = {}

    def data(path, as_csv=False):
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return list(csv.DictReader(io.StringIO(raw.decode()))) if as_csv else json.loads(raw)

    candidates = []
    original_rows = data(original / "reflection_advantages.csv", as_csv=True)
    for row in original_rows:
        if row["model"] not in MODELS[:3]:
            continue
        candidates.append({**{k: row[k] for k in ("model", "mode", "arm", "task_id")},
                           "source_repeat": int(row["source_repeat"]),
                           **{k: float(row[k]) for k in ("mean_pct", "pct_run0", "pct_run1", "pct_run2", "advantage")},
                           "zero_signal_group": row["zero_signal_group"] == "True"})
    assert len(candidates) == 3000
    tasks = {r["task_id"] for r in candidates}
    assert len(tasks) == 50
    checks = 0
    for model, cohort in (("gemma26", gemma), ("gemma31", gemma31)):
        for mode in MODES:
            folder = cohort / mode / "reflection"
            protocol = data(folder / "protocol.json")
            assert protocol["arms"] == ["without_gt"] and set(protocol["task_ids"]) == tasks
            assert protocol["turn_limit_scoring"] == "official_pct_at_limit"
            rows = data(folder / "stage3_results.json")
            assert len(rows) == 750
            bank = defaultdict(dict)
            for row in rows:
                assert row["arm"] == "without_gt" and row["task_id"] in tasks
                assert row["appworld_policy_sha256"] == protocol["appworld_policy_sha256"]
                group = bank[row["task_id"], row["source_repeat"]]
                assert row["evaluation_repeat"] not in group
                group[row["evaluation_repeat"]] = reward(row)
            assert set(bank) == {(task, repeat) for task in tasks for repeat in range(5)}
            assert all(set(evals) == {0, 1, 2} for evals in bank.values())
            previous = data(cohort / "advantage" / f"{mode}_advantage_groups.json")["without_gt"]
            for task in sorted(tasks):
                means = [sum(bank[task, i].values(), Fraction()) / 3 for i in range(5)]
                center = sum(means, Fraction()) / 5
                sigma = math.sqrt(float(sum(((m - center) ** 2 for m in means), Fraction()) / 5))
                advantages = [float(m - center) / (sigma + EPSILON) for m in means]
                assert np.allclose(advantages, previous[task]["advantages"], atol=1e-12, rtol=0)
                assert previous[task]["zero_signal_group"] == (sigma == 0)
                checks += 1
                for i in range(5):
                    candidates.append({"model": model, "mode": mode, "arm": "without_gt", "task_id": task,
                                       "source_repeat": i, "mean_pct": float(means[i]),
                                       **{f"pct_run{j}": float(bank[task, i][j]) for j in range(3)},
                                       "advantage": advantages[i], "zero_signal_group": sigma == 0})
    panels = defaultdict(list)
    task_groups = defaultdict(list)
    for row in candidates:
        panels[row["model"], row["mode"], row["arm"]].append(row["advantage"])
        task_groups[row["model"], row["mode"], row["arm"], row["task_id"]].append(row)
    assert len(panels) == 16 and len(task_groups) == 800 and len(candidates) == 4000
    groups = []
    for key, rows in task_groups.items():
        assert len(rows) == 5 and {r["source_repeat"] for r in rows} == set(range(5))
        assert abs(sum(r["advantage"] for r in rows)) < 1e-12
        groups.append({**dict(zip(("model", "mode", "arm", "task_id"), key, strict=True)),
                       "zero_signal_group": rows[0]["zero_signal_group"],
                       "mean_candidate_pct": float(np.mean([r["mean_pct"] for r in rows]))})
    summaries = []
    for key, values in panels.items():
        values = np.array(values)
        assert len(values) == 250 and abs(values.mean()) < 1e-12 and abs(values).max() <= 2 + 1e-12
        summaries.append({**dict(zip(("model", "mode", "arm"), key, strict=True)),
                          "tasks": 50, "reflections": 250, "mean_advantage": float(values.mean()),
                          "median_advantage": float(np.median(values)), "population_std_advantage": float(values.std()),
                          "nonnegative_count": int((values >= 0).sum()), "exact_zero_count": int((values == 0).sum()),
                          "nonnegative_fraction": float((values >= 0).mean())})
    return {k: np.array(v) for k, v in panels.items()}, candidates, groups, summaries, sources, checks


def draw(panels, summaries, output, *, wo_only=False):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})

    densities = {key: density(values) for key, values in panels.items()}
    scale = .43 / max(values.max() for values in densities.values())
    details = {(r["model"], r["mode"], r["arm"]): r for r in summaries}
    fig, axes = plt.subplots(1, 2, figsize=(21, 6.5), sharey=True) if wo_only else \
        plt.subplots(2, 2, figsize=(22, 11.5), sharey=True)
    positions = np.arange(len(MODELS)) * 1.35
    for row, mode in enumerate(MODES):
        for col, arm in enumerate(("without_gt",) if wo_only else ARMS):
            ax = axes[row] if wo_only else axes[row, col]
            ax.axhline(0, color="#a77d40", lw=1, linestyle=(0, (4, 3)), zorder=1)
            for x, model in zip(positions, MODELS, strict=True):
                key = model, mode, arm
                if key not in densities:
                    assert model in ("gemma26", "gemma31") and arm == "with_gt"
                    ax.axvspan(x - .48, x + .48, ymin=.01, ymax=.89, color="#eef1f4", alpha=.7, zorder=0)
                    ax.text(x, .45, "Not measured\n(w/o GT only)", ha="center", va="center",
                            fontsize=10, color="#7a8795")
                    ax.text(x, 2.35, "A ≥ 0: N/A", ha="center", va="center", fontsize=9.6, color="#8a929b")
                    ax.text(x, 2.13, "A = 0: N/A", ha="center", va="center", fontsize=9.1, color="#8a929b")
                    continue
                width = densities[key] * scale
                ax.fill_betweenx(GRID, x - width, x + width, facecolor=COLOR, edgecolor="#2d536b",
                                 alpha=.84, lw=.9, zorder=3)
                zero = details[key]["exact_zero_count"] / 250 * 100
                nonnegative = details[key]["nonnegative_fraction"] * 100
                ax.text(x, 2.35, f"A ≥ 0: {nonnegative:.1f}%", ha="center", va="center", fontsize=9.6,
                        fontweight="semibold", color="#285e4a")
                ax.text(x, 2.13, f"A = 0: {zero:.1f}%", ha="center", va="center", fontsize=9.1, color="#795b2e")
            ax.set(xlim=(positions[0] - .65, positions[-1] + .65), ylim=(-2.15, 2.55))
            ax.set_yticks(np.arange(-2, 2.01, .5))
            ax.set_xticks(positions, [LABELS[m] for m in MODELS], fontsize=11)
            ax.tick_params(length=0, labelleft=True, pad=7)
            ax.set_ylabel("GRPO advantage", fontsize=12, labelpad=8)
            ax.grid(axis="y", color="#e3e9ef", lw=.75, zorder=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#c9d3dd")
            label = "Thinking" if mode == "thinking" else "Non-thinking"
            label += " · w/o GT" if arm == "without_gt" else " · w/ GT"
            ax.set_title(label, fontsize=15, fontweight="semibold", loc="left", pad=14)
    fig.suptitle("AppWorld · Advantage distribution of generated Reflections", fontsize=21 if wo_only else 23,
                 fontweight="bold", y=.985 if wo_only else .98, color="#233348")
    fig.text(.5, .915 if wo_only else .937,
             "All 250 Reflections per measured model / panel · 50 matched tasks, 5 candidates each · "
             "Reward: three-rollout mean PCT", ha="center", fontsize=11.5, color="#526176")
    fig.text(.5, .074, "A = (candidate reward - task-group mean) / (task-group population SD + 1e-8). "
             "All-tied groups retained as exact zeros; no duplication across selection seeds.",
             ha="center", fontsize=9.8, color="#526176")
    fig.text(.5, .043, "Shared density-to-width scale; Gaussian bandwidth 0.12; reflected KDE on [-2, 2]. "
             "Both Gemma models have w/o GT data only.", ha="center", fontsize=9.8, color="#526176")
    if wo_only:
        fig.subplots_adjust(left=.065, right=.985, top=.80, bottom=.23, wspace=.19)
    else:
        fig.subplots_adjust(left=.065, right=.985, top=.855, bottom=.15, hspace=.40, wspace=.19)
    name = "reflection_advantage_all_candidates_qwen_gemma26b_31b_" + ("wo_gt_2panel" if wo_only else "4panel")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200, facecolor="white")
    fig.savefig(output / f"{name}_preview.png", dpi=110, facecolor="white")
    plt.close(fig)
    return densities, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--gemma31", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    originals = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.original.iterdir() if p.is_file()}
    panels, candidates, groups, summaries, sources, checked = collect(args.original, args.gemma, args.gemma31)
    csv_file(args.output / "reflection_advantages.csv", candidates)
    csv_file(args.output / "task_group_statistics.csv", groups)
    csv_file(args.output / "distribution_summary.csv", summaries)
    densities, scale = draw(panels, summaries, args.output)
    draw(panels, summaries, args.output, wo_only=True)
    old = defaultdict(list)
    raw = (args.original / "violin_density_grid.csv").read_bytes()
    sources[str(args.original / "violin_density_grid.csv")] = hashlib.sha256(raw).hexdigest()
    for row in csv.DictReader(io.StringIO(raw.decode())):
        if row["model"] in MODELS:
            old[row["model"], row["mode"], row["arm"]].append(float(row["density"]))
    for key, values in old.items():
        assert np.allclose(densities[key], values, atol=1e-12, rtol=0), key
    assert len(old) == 14
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == h for path, h in originals.items())
    csv_file(args.output / "violin_density_grid.csv", [
        {"model": key[0], "mode": key[1], "arm": key[2], "advantage": float(y), "density": float(d),
         "half_width": float(d * scale)} for key, values in densities.items()
        for y, d in zip(GRID, values, strict=True)])
    receipt = {"created_at": datetime.now(UTC).isoformat(), "models": list(MODELS), "arms": list(ARMS),
               "gemma_measured_arms": ["without_gt"], "missing_panels": "Both Gemma w/ GT cohorts shown as not measured",
               "candidates": len(candidates), "groups": len(groups), "measured_violins": len(panels),
               "candidate_reward": "Exact mean PCT of three saved candidate evaluations, not TGC",
               "advantage": "(candidate reward - G5 task mean)/(G5 population SD +1e-8)",
               "gemma_groups_verified": checked, "qwen_original_densities_verified": len(old),
               "common_half_width_scale": float(scale), "bandwidth": BANDWIDTH,
               "originals_preserved": originals, "source_sha256": sources,
               "new_inference_calls": 0, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "provenance.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2) + "\n")
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("source_sha256","originals_preserved")}, indent=2))


if __name__ == "__main__":
    main()
