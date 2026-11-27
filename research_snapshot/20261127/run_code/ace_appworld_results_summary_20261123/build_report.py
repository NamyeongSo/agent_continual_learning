# SPDX-License-Identifier: Apache-2.0
"""Build seed-level ACE result tables and task-normalized gain violins from saved outcomes."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from math import comb
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.ace.run_reflection_accumulation import exact_reward  # noqa: E402

OUT = Path(__file__).resolve().parent
BASE = ROOT / "outputs"
SEEDS = (42, 43, 44)
ARMS = {"without_gt": "w/o GT", "with_gt": "w/ GT"}
STRATEGIES = ("best", "random", "worst", "best_advantage", "worst_advantage")
LABELS = {"qwen38": "Qwen3.8-27B", "qwen36": "Qwen3.6-35B-A3B", "qwen35": "Qwen3.5-9B",
          "oss20": "GPT-OSS-20B", "oss120": "GPT-OSS-120B"}
SHORT = {"qwen38": "Qwen3.8\n27B", "qwen36": "Qwen3.6\n35B-A3B", "qwen35": "Qwen3.5\n9B",
         "oss20": "GPT-OSS\n20B", "oss120": "GPT-OSS\n120B"}
HATCHES = {"qwen38": "", "qwen36": "///", "qwen35": "...", "oss20": "xx", "oss120": "---"}
MODEL_RUNS = {
    "qwen38": {
        "thinking": BASE / "ace_appworld_reflection_accumulation_qwen38_teammate_s42_44_20261116",
        "nonthinking": BASE / "ace_appworld_nonthinking_replication_qwen38_20261117/accumulation",
    },
    "qwen36": {m: BASE / f"ace_appworld_qwen36_add_only_20261118/{m}/accumulation"
               for m in ("thinking", "nonthinking")},
    "qwen35": {m: BASE / f"ace_appworld_qwen35_9b_add_only_20261119/{m}/accumulation"
               for m in ("thinking", "nonthinking")},
    "oss20": {m: BASE / f"ace_appworld_gpt_oss20b_add_only_20261120/{m}/accumulation" for m in ("high", "low")},
    "oss120": {m: BASE / f"ace_appworld_gpt_oss120b_add_only_20261121/{m}/accumulation" for m in ("high", "low")},
}
ADV_RUNS = {
    "qwen38": BASE / "ace_appworld_advantage_add_20261118",
    "qwen36": BASE / "ace_appworld_qwen36_add_only_20261118/advantage",
    "qwen35": BASE / "ace_appworld_qwen35_9b_add_only_20261119/advantage",
    "oss20": BASE / "ace_appworld_gpt_oss20b_add_only_20261120/advantage",
}
POLICY_SHA = "082147d2215b74ce52131e6ef30baa342aeaf4099e2e475f0bbb97fed9fe195d"
SOURCES = {}
RAW = []
SEED_ROWS = []
BASELINE_ROWS = []
INITIAL = {}
BANKS = {}
ACC_TASKS = {}
ADV_TASKS = {}
COVERAGE = []
GENERATED = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def read(path):
    payload = Path(path).read_bytes()
    SOURCES[str(Path(path).relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def mode_label(mode):
    return {"thinking": "Thinking", "nonthinking": "Non-thinking", "high": "High", "low": "Low"}[mode]


def validate(rows):
    assert all(type(r["success"]) is bool and r["appworld_policy_sha256"] == POLICY_SHA for r in rows)
    assert all(float(exact_reward(r)) == r["pct"] for r in rows)


def metrics(rows):
    return {"pct": mean(float(exact_reward(r)) for r in rows), "tgc": mean(int(r["success"]) for r in rows)}


def record_raw(rows, model, mode, stage):
    validate(rows)
    for r in rows:
        RAW.append({"model": model, "mode": mode, "stage": stage,
                    **{k: r.get(k) for k in ("id", "task_id", "seed", "condition_id", "arm", "source_repeat",
                                            "evaluation_repeat", "pct", "success", "passed_tests", "total_tests",
                                            "turn_limit_reached", "attempt", "completed_at")},
                    "budget_failure": (r.get("budget_failure") or {}).get("reason", "")})


def table_stat(values):
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else None, "n": len(values)}


def cell(values):
    values = list(values)
    if not values:
        return "평가 대기"
    stat = table_stat(values)
    shown_mean = 0.0 if abs(stat["mean"]) < 0.00005 else 100 * stat["mean"]
    if stat["std"] is None:
        return f"{shown_mean:.2f} (n=1)"
    return f"{shown_mean:.2f} ± {100 * stat['std']:.2f}"


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def load_reflection(model, mode, run):
    protocol = read(run / "protocol.json")
    assert protocol["playbook_edit_mode"] == "add_only"
    assert protocol["appworld_policy_sha256"] == POLICY_SHA
    bank_run = Path(protocol["source_reflection_run"])
    split = protocol["task_orders"]["42"]
    assert len(split["train"]) == len(split["val"]) == 50
    assert set(split["train"]).isdisjoint(split["val"])
    for seed in SEEDS:
        assert set(protocol["task_orders"][str(seed)]["train"]) == set(split["train"])
        assert set(protocol["task_orders"][str(seed)]["val"]) == set(split["val"])
    status = read(bank_run / "status.json")
    initial_rows = read(bank_run / "stage1_results.json") if (bank_run / "stage1_results.json").exists() else []
    initial = defaultdict(list)
    if initial_rows:
        record_raw(initial_rows, model, mode, "initial")
        for row in initial_rows:
            initial[row["task_id"]].append(row)
        assert set(initial) == set(split["train"])
        assert all(len(group) == 5 and {r["source_repeat"] for r in group} == set(range(5))
                   for group in initial.values())
        INITIAL[model, mode] = initial
        repeated = [metrics([r for r in initial_rows if r["source_repeat"] == repeat]) for repeat in range(5)]
        successes = [sum(r["success"] for r in group) for group in initial.values()]
        BASELINE_ROWS.append({"model": model, "mode": mode, "tasks": 50, "rollouts": 250,
                              "pct_mean": mean(r["pct"] for r in repeated), "pct_std": stdev(r["pct"] for r in repeated),
                              "tgc_mean": mean(r["tgc"] for r in repeated), "tgc_std": stdev(r["tgc"] for r in repeated),
                              "pass_at_3": mean(1 - comb(5 - s, 3) / comb(5, 3) for s in successes),
                              "pass_power_3": mean(comb(s, 3) / comb(5, 3) for s in successes)})
    rows = read(bank_run / "stage3_results.json") if (bank_run / "stage3_results.json").exists() else []
    if not rows and initial_rows:
        queue = read(bank_run / "stage3_queue.json") if (bank_run / "stage3_queue.json").exists() else []
        rows = [read(bank_run / "stage3" / job["id"] / "result.json")
                for job in queue if (bank_run / "stage3" / job["id"] / "result.json").exists()]
    candidates = defaultdict(dict)
    if rows:
        record_raw(rows, model, mode, "independent")
        for row in rows:
            key = row["arm"], row["task_id"], row["source_repeat"]
            assert row["task_id"] in split["train"] and row["evaluation_repeat"] not in candidates[key]
            candidates[key][row["evaluation_repeat"]] = row
    banks = defaultdict(dict)
    for (arm, task, source_repeat), evaluations in candidates.items():
        if set(evaluations) == {0, 1, 2}:
            group = list(evaluations.values())
            banks[arm, task][source_repeat] = {**metrics(group), "rows": group,
                "exact_pct": sum(exact_reward(r) for r in group) / 3}
    BANKS[model, mode] = {key: group for key, group in banks.items() if set(group) == set(range(5))}
    stage2_rows = read(bank_run / "stage2_results.json") if (bank_run / "stage2_results.json").exists() else []
    COVERAGE.append({"model": model, "mode": mode, "initial": len(initial_rows), "proposals": len(stage2_rows),
                     "independent": len(rows), "full_task_arm_groups": len(BANKS[model, mode]),
                     "reflection_state": status.get("state"), "reflection_run": str(bank_run.relative_to(ROOT))})
    return protocol, candidates


def independent(model, mode, seed, arm, strategy, selected, candidates):
    chosen = {item["candidate"]["task_id"]: item["candidate"]["source_repeat"] for item in selected}
    assert len(chosen) == len(selected)
    scores = []
    for task, initial in INITIAL[model, mode].items():
        rows = list(candidates[arm, task, chosen[task]].values()) if task in chosen else initial
        assert len(rows) == (3 if task in chosen else 5)
        scores.append(metrics(rows))
    return {"model": model, "mode": mode, "seed": seed, "arm": arm, "strategy": strategy,
            "learned_tasks": len(selected), "independent_pct": mean(s["pct"] for s in scores),
            "independent_tgc": mean(s["tgc"] for s in scores)}


def add_condition(model, mode, run, seed, arm, strategy, candidates, phase_rows):
    cid = f"s{seed}_{arm}_{strategy}"
    selection_path = run / "conditions" / cid / "selections.json"
    if not selection_path.exists():
        return
    selected = read(selection_path)
    record = independent(model, mode, seed, arm, strategy, selected, candidates)
    initial_tasks = set(INITIAL[model, mode])
    selection_tasks = {r["candidate"]["task_id"] for r in selected}
    assert selection_tasks <= initial_tasks
    local = {}
    protocol = read(run / "protocol.json")
    for phase, rows in phase_rows.items():
        condition_rows = [r for r in rows if r["condition_id"] == cid and r["seed"] == seed]
        expected = set(protocol["task_orders"][str(seed)]["val"]) if phase == "val" else initial_tasks
        if phase == "bwt_checkpoint":
            expected = selection_tasks
        if len(condition_rows) != len(expected):
            continue
        assert len({r["task_id"] for r in condition_rows}) == len(expected)
        assert {r["task_id"] for r in condition_rows} == expected
        local[phase] = {r["task_id"]: r for r in condition_rows}
        if phase != "bwt_checkpoint":
            label = "train" if phase == "train_final" else "heldout"
            for metric, value in metrics(condition_rows).items():
                record[f"{label}_{metric}"] = value
            if phase == "train_final":
                target = ADV_TASKS if strategy.endswith("_advantage") else ACC_TASKS
                target[model, mode, arm, strategy, seed] = {r["task_id"]: r["pct"] for r in condition_rows}
    if "train_final" in local and "bwt_checkpoint" in local:
        after = local["train_final"]
        before = local["bwt_checkpoint"]
        record["bwt_pct"] = mean(after[t]["pct"] - r["pct"] for t, r in before.items())
        record["bwt_tgc"] = mean(int(after[t]["success"]) - int(r["success"]) for t, r in before.items())
    SEED_ROWS.append(record)


def collect():
    for model, modes in MODEL_RUNS.items():
        for mode, run in modes.items():
            _, candidates = load_reflection(model, mode, run)
            if (model, mode) not in INITIAL or len(BANKS[model, mode]) != 100:
                continue
            phases = {}
            for seed in SEEDS:
                for phase in ("train_final", "bwt_checkpoint", "val"):
                    path = run / f"seed{seed}" / f"{phase}_results.json"
                    if path.exists():
                        rows = read(path)
                        record_raw(rows, model, mode, phase)
                        phases[phase] = rows
                for arm in ARMS:
                    for strategy in STRATEGIES[:3]:
                        add_condition(model, mode, run, seed, arm, strategy, candidates, phases)
            advantage_run = ADV_RUNS.get(model)
            if not advantage_run or not (advantage_run / "plan.json").exists():
                continue
            plan = read(advantage_run / "plan.json")
            evaluation = advantage_run / "evaluations"
            adv_phases = {}
            for phase in ("train_final", "bwt_checkpoint", "val"):
                path = evaluation / f"{phase}_results.json"
                rows = read(path) if path.exists() else []
                rows = [r for r in rows if r["sampler"] == mode]
                record_raw(rows, model, mode, "advantage_" + phase)
                adv_phases[phase] = rows
            for job in plan:
                job_run = Path(job["path"])
                if job_run.parent.name != mode:
                    continue
                c = job["condition"]
                add_condition(model, mode, job_run, c["seed"], c["arm"], c["mode"], candidates, adv_phases)


def gain_points(model, mode, arm, *, accumulated=False, filtered=False, advantage=False):
    initial = INITIAL.get((model, mode), {})
    bank = BANKS.get((model, mode), {})
    tasks = [task for a, task in bank if a == arm]
    if filtered or advantage:
        tasks = [t for t in tasks if len({v["exact_pct"] for v in bank[arm, t].values()}) > 1]
    result = {s: [] for s in ("best", "random", "worst")}
    if accumulated:
        target = ADV_TASKS if advantage else ACC_TASKS
        for task in tasks:
            for strategy in result:
                selected_strategy = strategy + "_advantage" if advantage and strategy != "random" else strategy
                source = target if selected_strategy.endswith("_advantage") else ACC_TASKS
                samples = [source.get((model, mode, arm, selected_strategy, seed), {}).get(task) for seed in SEEDS]
                if any(v is None for v in samples):
                    continue
                result[strategy].append(((mean(samples) - metrics(initial[task])["pct"]) * 100, 1.0))
    else:
        for task in tasks:
            group = bank[arm, task]
            base = metrics(initial[task])["pct"]
            rewards = [v["exact_pct"] for v in group.values()]
            center = sum(rewards) / 5
            pools = {
                "best": [v for v in group.values() if v["exact_pct"] == max(rewards)],
                "random": list(group.values()),
                "worst": [v for v in group.values() if v["exact_pct"] == min(rewards)],
            }
            if advantage:
                pools["best"] = [v for v in group.values() if v["exact_pct"] >= center]
                pools["worst"] = [v for v in group.values() if v["exact_pct"] <= center]
            for strategy, pool in pools.items():
                result[strategy].extend(((v["pct"] - base) * 100, 1 / len(pool)) for v in pool)
    return result


def density(points, grid):
    values = np.array([v for v, _ in points])
    weights = np.array([w for _, w in points])
    offsets = (grid[:, None] - values) / 7
    kernels = np.where(np.abs(offsets) <= 4, np.exp(-0.5 * offsets ** 2), 0)
    kernels /= np.trapz(kernels, grid, axis=0)
    result = kernels @ (weights / 50)
    assert np.isclose(np.trapz(result, grid), weights.sum() / 50, atol=1e-12)
    return result


def plot_gain(name, *, accumulated=False, filtered=False, advantage=False):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "hatch.linewidth": 0.7})
    grid = np.linspace(-100, 100, 801)
    colors = {"best": "#17765c", "worst": "#b55643"}
    cached = {}
    for model in LABELS:
        for mode in MODEL_RUNS[model]:
            for arm in ARMS:
                points = gain_points(model, mode, arm, accumulated=accumulated, filtered=filtered, advantage=advantage)
                cached[model, mode, arm] = (points, {s: density(p, grid) for s, p in points.items() if p})
    max_density = max((float(d.max()) for _, ds in cached.values() for d in ds.values()), default=1)
    scale = 0.43 / max_density
    fig, axes = plt.subplots(2, 2, figsize=(16.4, 11), sharey=True)
    for row in range(2):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            ticks = []
            for position, model in enumerate(LABELS):
                mode = tuple(MODEL_RUNS[model])[row]
                points, densities = cached[model, mode, arm]
                count = round(sum(w for _, w in points["best"]))
                partial = count and not accumulated and len(BANKS.get((model, mode), {})) != 100
                ticks.append(SHORT[model] + (f"\nn={count}/50" if count else "\npending") + (" · partial" if partial else ""))
                if points["random"]:
                    width = scale * densities["random"]
                    ax.fill_betweenx(grid, position - width, position + width, where=width > 0,
                                     facecolor="#dfe2e6", edgecolor="#aab1ba", alpha=0.72, linewidth=0.7, zorder=1)
                for strategy, side in (("best", -1), ("worst", 1)):
                    if not points[strategy]:
                        continue
                    d = densities[strategy]
                    ax.fill_betweenx(grid, position, position + side * scale * d, where=d > 0,
                                     facecolor=to_rgba(colors[strategy], 0.22), edgecolor=colors[strategy],
                                     linewidth=0.85, hatch=HATCHES[model], zorder=3)
                    center = np.average([v for v, _ in points[strategy]], weights=[w for _, w in points[strategy]])
                    width = float(np.interp(center, grid, d)) * scale
                    ax.plot([position, position + side * width], [center, center], color=colors[strategy], linewidth=2.0,
                            zorder=4)
            ax.axhline(0, color="#555e69", linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
            ax.set(title=("Thinking / High" if row == 0 else "Non-thinking / Low") + " · " + ARMS[arm],
                   xticks=range(5), xticklabels=ticks, xlim=(-0.65, 4.65), ylim=(-100, 100))
            ax.set_yticks(range(-100, 101, 25))
            ax.grid(axis="y", color="#e6ebf1", linewidth=0.8)
            ax.tick_params(length=0)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("PCT gain (percentage points)")
    cohort = "Nonzero-advantage tasks" if filtered or advantage else "All train tasks"
    prefix = "Accumulated final context" if accumulated else "Single reflection"
    fig.suptitle(f"AppWorld Train · {prefix} PCT gain", fontsize=19, weight="bold", y=0.98)
    fig.text(0.5, 0.943, "Best on the left · Worst on the right · Random in gray · Shared area proportional to n / 50",
             ha="center", fontsize=11)
    handles = [Patch(facecolor=to_rgba(c, 0.22), edgecolor=c, label=s + ("_advantage" if advantage else ""))
               for s, c in colors.items()]
    handles += [Patch(facecolor="#dfe2e6", edgecolor="#aab1ba", label="Random"),
                Line2D([0], [0], color="#555e69", linewidth=2, label="Mean"),
                Line2D([0], [0], color="#555e69", linestyle="--", label="Zero gain")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.045))
    basis = "Final-context PCT averaged across seeds 42, 43, 44" if accumulated else "Three-rollout candidate mean · Uniform candidate selection pools"
    fig.text(0.5, 0.025, f"{cohort} · {basis} · Task-specific initial mean · Gaussian bandwidth 7 pp",
             ha="center", fontsize=9, color="#555e69")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.14, hspace=0.39, wspace=0.10)
    fig.savefig(OUT / f"{name}.png", dpi=190, facecolor="white")
    plt.close(fig)


def baseline_table(models=tuple(LABELS)):
    lines = ["| 모델 | 모드 | 초기 PCT | 초기 TGC | pass@3 | pass³ |",
             "| --- | --- | --- | --- | --- | --- |"]
    for model in models:
        for mode in MODEL_RUNS[model]:
            row = next((r for r in BASELINE_ROWS if r["model"] == model and r["mode"] == mode), None)
            if row:
                lines.append(f"| {LABELS[model]} | {mode_label(mode)} | {row['pct_mean'] * 100:.2f} ± {row['pct_std'] * 100:.2f} | "
                             f"{row['tgc_mean'] * 100:.2f} ± {row['tgc_std'] * 100:.2f} | "
                             f"{row['pass_at_3'] * 100:.2f} | {row['pass_power_3'] * 100:.2f} |")
            else:
                lines.append(f"| {LABELS[model]} | {mode_label(mode)} | 평가 대기 | 평가 대기 | 평가 대기 | 평가 대기 |")
    return "\n".join(lines)


def model_page(model):
    lines = ["모든 점수의 단위는 %이며 BWT는 %p입니다. 표준편차는 ddof=1로 계산했습니다.",
             "초기 성능은 50개 Task를 포함한 평가 5회의 벤치마크 점수에 대한 평균 ± 표준편차입니다. "
             "초기 pass@3·pass³는 각 Task의 5회 중 가능한 3회 조합 10개를 평균했습니다.",
             baseline_table((model,)),
             "비축적은 각 seed의 실제 Reflection 선정 내역과 후보별 3회 평가를 사용한 기대 점수입니다. "
             "축적 Train·Held-out·BWT는 seed 42·43·44 각각의 점수 세 개에 대한 평균 ± 표본 표준편차입니다. "
             "Advantage에서 학습을 건너뛴 Task의 비축적 기대에는 초기 Context의 5회 평균을 적용했습니다."]
    if model == "oss120":
        lines += ["## 진행 상황",
                  "High 초기 평가 250건과 Reflection·Context Update 500건을 완료했습니다. "
                  "후보 재평가가 진행 중이며, 모델별 바이올린에는 평가 대기 위치를 표시했습니다. "
                  "Low와 누적 Train·Held-out·BWT는 평가 대기 상태입니다."]
        return "\n\n".join(lines)
    for mode in MODEL_RUNS[model]:
        lines += [f"## {mode_label(mode)} · 비축적과 축적 Train",
                  "| GT | Reflection 선정 | 학습 Task | 비축적 PCT | 비축적 TGC | Train PCT | Train TGC |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for arm in ARMS:
            for strategy in STRATEGIES:
                selected = [r for r in SEED_ROWS if (r["model"], r["mode"], r["arm"], r["strategy"]) ==
                            (model, mode, arm, strategy)]
                n = str(selected[0]["learned_tasks"]) if selected else "대기"
                cells = [cell(r[key] for r in selected if key in r)
                         for key in ("independent_pct", "independent_tgc", "train_pct", "train_tgc")]
                lines.append(f"| {ARMS[arm]} | {strategy.replace('_', r'\_')} | {n} | " + " | ".join(cells) + " |")
        lines += [f"## {mode_label(mode)} · Held-out과 BWT",
                  "| GT | Reflection 선정 | Held-out PCT | Held-out TGC | BWT PCT | BWT TGC |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for arm in ARMS:
            for strategy in STRATEGIES:
                selected = [r for r in SEED_ROWS if (r["model"], r["mode"], r["arm"], r["strategy"]) ==
                            (model, mode, arm, strategy)]
                cells = [cell(r[key] for r in selected if key in r)
                         for key in ("heldout_pct", "heldout_tgc", "bwt_pct", "bwt_tgc")]
                lines.append(f"| {ARMS[arm]} | {strategy.replace('_', r'\_')} | " + " | ".join(cells) + " |")
    lines += ["## 지표와 선정 규칙",
              "best와 worst는 후보 5개의 3회 평균 PCT가 각각 최대·최소인 후보를 고릅니다. "
              "동점은 저장된 seed별 무작위 선정 내역을 따랐습니다. random도 저장된 실제 선정 내역을 사용했습니다.",
              "best\\_advantage는 Advantage ≥ 0인 후보 중 균등 선정하고, worst\\_advantage는 Advantage ≤ 0인 후보 중 균등 선정합니다. "
              "후보 5개의 Advantage가 모두 0인 Task는 학습에서 건너뜁니다.",
              "BWT는 최종 Context 성능에서 해당 Task 학습 직후 Context 성능을 뺀 값입니다. "
              "전체 학습은 50개, Advantage 학습은 실제 학습한 Task를 대상으로 계산했습니다.",
              "비축적 후보의 선정과 점수 계산은 같은 3회 평가를 사용했습니다. best·worst의 극값 선택 효과가 포함됩니다. "
              "같은 후보 bank를 seed별로 재사용하므로 비축적 best·worst PCT의 seed 표준편차는 0일 수 있습니다. "
              "Held-out의 학습 전 초기 Context 평가는 저장되어 있지 않습니다."]
    return "\n\n".join(join_blocks(lines))


def join_blocks(lines):
    """Keep consecutive table rows in a single Markdown block."""
    blocks = []
    for line in lines:
        if line.startswith("|") and blocks and blocks[-1].startswith("|"):
            blocks[-1] += "\n" + line
        else:
            blocks.append(line)
    return blocks


def main():
    collect()
    assert len(BASELINE_ROWS) >= 9
    assert len(SEED_ROWS) == 240
    unique = {(r["model"], r["mode"], r["seed"], r["arm"], r["strategy"]) for r in SEED_ROWS}
    assert len(unique) == len(SEED_ROWS)
    assert all(all(k in r for k in ("independent_pct", "independent_tgc", "train_pct", "train_tgc",
                                   "heldout_pct", "heldout_tgc", "bwt_pct", "bwt_tgc")) for r in SEED_ROWS)
    write_csv("initial_metrics.csv", BASELINE_ROWS)
    write_csv("metrics_by_seed.csv", SEED_ROWS)
    write_csv("official_outcomes.csv", RAW)
    write_csv("coverage.csv", COVERAGE)
    summary = []
    for model in LABELS:
        for mode in MODEL_RUNS[model]:
            for arm in ARMS:
                for strategy in STRATEGIES:
                    rows = [r for r in SEED_ROWS if (r["model"], r["mode"], r["arm"], r["strategy"]) ==
                            (model, mode, arm, strategy)]
                    if rows:
                        record = {"model": model, "mode": mode, "arm": arm, "strategy": strategy,
                                  "seeds": "42,43,44", "learned_tasks": rows[0]["learned_tasks"]}
                        for key in ("independent_pct", "independent_tgc", "train_pct", "train_tgc", "heldout_pct",
                                    "heldout_tgc", "bwt_pct", "bwt_tgc"):
                            stat = table_stat([r[key] for r in rows])
                            record[key + "_mean"], record[key + "_std"] = stat["mean"], stat["std"]
                        summary.append(record)
    write_csv("strategy_metrics_mean_std.csv", summary)
    for model in LABELS:
        (OUT / f"{model}.md").write_text(model_page(model))
    (OUT / "initial_table.md").write_text(baseline_table())
    for accumulated in (False, True):
        prefix = "accumulated" if accumulated else "independent"
        for filtered in (False, True):
            suffix = "nonzero_advantage" if filtered else "all_tasks"
            plot_gain(f"{prefix}_pct_gain_{suffix}_split_violin_n_over_50_with_random", accumulated=accumulated,
                      filtered=filtered)
        plot_gain(f"{prefix}_advantage_sign_pct_gain_split_violin_n_over_50_with_random", accumulated=accumulated,
                  advantage=True)
    audit = {"generated_at_kst": GENERATED, "seed_rows": len(SEED_ROWS), "raw_outcomes": len(RAW),
             "models": LABELS, "coverage": COVERAGE, "aggregation": "benchmark score per seed, sample SD ddof=1",
             "initial_aggregation": "five complete 50-task evaluation rounds; pass@3 and pass^3 from 10 combinations",
             "violin_bandwidth_pp": 7, "violin_half_area_denominator": 50,
             "candidate_violin_selection": "uniform over ties / all five candidates / signed eligible pool",
             "accumulated_violin": "per-task PCT mean across three learned contexts minus task initial mean",
             "source_sha256": SOURCES}
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("generated_at_kst", "seed_rows", "raw_outcomes", "coverage")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
