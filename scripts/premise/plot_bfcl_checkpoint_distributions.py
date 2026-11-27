# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Show checkpoint heterogeneity in cached BFCL retention, past gain, and validation lift."""

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

from scripts.premise.analyze_bfcl_retention_generalization import (  # noqa: E402
    evaluate_choice,
    outcome_vectors,
)
from scripts.premise.analyze_bfcl_temporal_parent_preservation import allocation  # noqa: E402
from scripts.premise.analyze_bfcl_temporal_selection import write_csv  # noqa: E402
from scripts.premise.bfcl_trajectory_selection_rewards import choice_probabilities  # noqa: E402
from scripts.premise.plot_bfcl_temporal_parent_preservation import (  # noqa: E402
    CASE_LABELS,
    CASES,
    COLORS,
    FILLS,
    MODELS,
    WEIGHT_LABELS,
    WEIGHTS,
)

METRICS = ("retention_past", "gain_past", "lift_validation40")
SEEDS = (42, 43, 44)
SEED_MARKERS = {42: "o", 43: "^", 44: "s"}
EXTRA_LABELS = {"proposal": "Proposal mean", "val10": "Val 10"}
ZERO_TOLERANCE_PP = 1e-10


def distribution_stats(values: list[float], metric: str) -> dict:
    """Describe CP outcomes in percent/pp, distinguishing numerical zero from gain."""
    a = np.asarray(values, dtype=float) * 100
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Distributions need finite checkpoint outcomes")
    q = np.quantile(a, [.1, .25, .5, .75, .9])
    result = {"unit": "%" if metric == "retention_past" else "pp", "checkpoints": len(a),
              "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else None,
              "minimum": float(a.min()), "p10": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
              "q75": float(q[3]), "p90": float(q[4]), "maximum": float(a.max()),
              "perfect_retention_checkpoints": None, "regression_checkpoints": None,
              "positive_checkpoints": None, "zero_checkpoints": None, "negative_checkpoints": None}
    if metric == "retention_past":
        if ((a < -ZERO_TOLERANCE_PP) | (a > 100 + ZERO_TOLERANCE_PP)).any():
            raise ValueError("Retention must be a fraction")
        result["perfect_retention_checkpoints"] = int((np.abs(a - 100) <= ZERO_TOLERANCE_PP).sum())
        result["regression_checkpoints"] = int((a < 100 - ZERO_TOLERANCE_PP).sum())
    else:
        result["positive_checkpoints"] = int((a > ZERO_TOLERANCE_PP).sum())
        result["zero_checkpoints"] = int((np.abs(a) <= ZERO_TOLERANCE_PP).sum())
        result["negative_checkpoints"] = int((a < -ZERO_TOLERANCE_PP).sum())
    return result


def validated_rows(summary: dict) -> list[dict]:
    """Require the same complete, seed-balanced cohort for every feasible selector."""
    keys = set(summary["fixed_checkpoint_keys"])
    if len(keys) != summary["checkpoints_per_model"]:
        raise ValueError("Checkpoint cohort size does not match its keys")
    grouped = defaultdict(list)
    for row in summary["checkpoint_rows"]:
        grouped[row["model"], row["weight"], row["n"], row["case"]].append(row)
    expected = {(m, w, n, c) for m in MODELS for w in WEIGHTS for n in range(1, 11)
                for c in CASES if allocation(c, n) is not None}
    if set(grouped) != expected:
        raise ValueError("Missing or unexpected sampling conditions")
    for (_, _, n, case), rows in grouped.items():
        if len(rows) != len(keys) or {r["key"] for r in rows} != keys:
            raise ValueError("Each selection condition must use the full identical checkpoint cohort")
        for seed in SEEDS:
            if sum(r["seed"] == seed for r in rows) != len(keys) // len(SEEDS):
                raise ValueError("Checkpoint cohorts must be balanced across the three seeds")
        for row in rows:
            if row["actual_n"] != sum(allocation(case, n)):
                raise ValueError("Recorded selection budget differs from temporal allocation")
            if any(row[k] is None or not np.isfinite(row[k]) for k in METRICS):
                raise ValueError("Every checkpoint requires all three outcomes")
    return [dict(row) for row in summary["checkpoint_rows"]]


def reference_rows(summary: dict, cache_path: Path) -> list[dict]:
    """Reconstruct CP reference outcomes, auditing against already reported aggregates."""
    expected_hash = next((r["sha256"] for r in summary["source_files"]
                          if Path(r["path"]).name == cache_path.name), None)
    if expected_hash is None or hashlib.sha256(cache_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Reference cache hash differs from the original analysis")
    cache = json.loads(cache_path.read_text())
    keys = set(summary["fixed_checkpoint_keys"])
    references, identities = [], set()
    for model in cache["models"]:
        label = model["label"]
        if label not in MODELS:
            continue
        for point in model["points"]:
            if point["key"] not in keys:
                continue
            identity = (label, point["key"])
            if identity in identities:
                raise ValueError("Duplicated checkpoint in reference cache")
            identities.add(identity)
            outcomes = outcome_vectors(np.asarray(point["success"]), point["current_index"], point["targets"])
            probabilities = {"proposal": np.full(5, .2),
                             "val10": choice_probabilities(np.asarray(point["targets"]["val10"])[None])[0]}
            for case, prob in probabilities.items():
                references.append({"model": label, "key": point["key"], "seed": point["seed"],
                                   "checkpoint": point["current_index"], "case": case,
                                   "actual_n": 0 if case == "proposal" else 10,
                                   **evaluate_choice(prob, outcomes)})
    if identities != {(m, key) for m in MODELS for key in keys}:
        raise ValueError("Missing reference checkpoints")
    for source in ("proposal_reference", "validation_reference"):
        for aggregate in summary[source]:
            rows = [r for r in references if r["model"] == aggregate["model"] and r["case"] == aggregate["case"]
                    and (aggregate["scope"] == "all" or r["seed"] == aggregate["scope"])]
            for metric in METRICS:
                values = np.array([r[metric] for r in rows])
                target = aggregate[metric]
                if (len(values) != target["defined_checkpoints"]
                        or not np.isclose(values.mean(), target["mean"], rtol=0, atol=1e-12)
                        or not np.isclose(values.std(ddof=1), target["checkpoint_std"], rtol=0, atol=1e-12)):
                    raise ValueError("Reconstructed references changed the reported results")
    return [{**r, "weight": w, "n": n} for r in references for w in WEIGHTS
            for n in (range(1, 11) if r["case"] == "proposal" else (10,))]


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        for scope in ("all", row["seed"]):
            groups[row["model"], row["weight"], row["n"], row["case"], scope].append(row)
    return [{"model": m, "weight": w, "n": n, "case": c, "scope": scope, "metric": metric,
             "actual_n": group[0]["actual_n"],
             **distribution_stats([r[metric] for r in group], metric)}
            for (m, w, n, c, scope), group in groups.items() for metric in METRICS]


def jitter(row: dict) -> float:
    """Keep a checkpoint at the same vertical offset across methods, panels, and n."""
    digest = hashlib.sha256(row["key"].encode()).digest()
    return (SEEDS.index(row["seed"]) - 1) * .12 + (int.from_bytes(digest[:4], "big") / (2**32 - 1) - .5) * .065


def plot(rows: list[dict], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, MaxNLocator

    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["weight"], row["n"], row["case"]].append(row)
    limits = []
    for metric in METRICS:
        values = np.array([r[metric] * 100 for r in rows])
        lower, upper = min(100 if metric == "retention_past" else 0, values.min()), max(100, values.max())
        if metric != "retention_past":
            upper = max(0, values.max())
        padding = max((upper - lower) * .07, .1)
        limits.append((lower - padding, upper + padding))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                         "axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .6})
    metric_labels = ("Past retention (%)", "Past gain (pp)", "Validation lift (pp)")
    handles = [Line2D([], [], marker=SEED_MARKERS[s], linestyle="none", color="#555555", markersize=4,
                      label=f"Seed {s}") for s in SEEDS]
    handles += [Line2D([], [], marker="D", linestyle="none", color="#111111", markersize=4, label="Mean"),
                Line2D([], [], marker="|", linestyle="none", color="#111111", markersize=8, label="Median")]
    files = []
    for n in range(1, 11):
        cases = [c for c in CASES if allocation(c, n) is not None] + ["proposal"] + (["val10"] if n == 10 else [])
        fig, axes = plt.subplots(4, 3, figsize=(13, 12), squeeze=False)
        fig.subplots_adjust(left=.21, right=.99, top=.91, bottom=.105, hspace=.49, wspace=.13)
        for row_index, (model, weight) in enumerate((m, w) for m in MODELS for w in WEIGHTS):
            for column, metric in enumerate(METRICS):
                ax = axes[row_index, column]
                for position, case in enumerate(cases):
                    points = sorted(groups[model, weight, n, case], key=lambda r: r["key"])
                    values = np.array([r[metric] * 100 for r in points])
                    color = COLORS.get(case, "#777777" if case == "proposal" else "#A34F65")
                    fill = FILLS.get(case, "#F2F2F2" if case == "proposal" else "#F9EEF1")
                    ax.boxplot([values], positions=[position], widths=.52, vert=False, patch_artist=True,
                               manage_ticks=False, showfliers=False, whis=1.5,
                               boxprops={"facecolor": fill, "edgecolor": color, "linewidth": .8},
                               medianprops={"color": "#111111", "linewidth": 1.1},
                               whiskerprops={"color": color, "linewidth": .7},
                               capprops={"color": color, "linewidth": .7})
                    for seed in SEEDS:
                        subset = [r for r in points if r["seed"] == seed]
                        ax.scatter([r[metric] * 100 for r in subset], [position + jitter(r) for r in subset],
                                   c=color, marker=SEED_MARKERS[seed], s=13, alpha=.75, linewidths=.25,
                                   edgecolors="white", zorder=4)
                    ax.scatter([values.mean()], [position], marker="D", s=22, color="#111111",
                               edgecolor="white", linewidth=.4, zorder=6)
                ax.set(yticks=np.arange(len(cases)), ylim=(len(cases) - .35, -.65), xlim=limits[column])
                if column == 0:
                    ax.set_yticklabels([CASE_LABELS[c] if c in CASE_LABELS else EXTRA_LABELS[c] for c in cases])
                else:
                    ax.set_yticklabels([])
                ax.set_title(f"{model} · {WEIGHT_LABELS[weight]}" if column == 0 else "", loc="left", pad=7)
                ax.set_xlabel(metric_labels[column], labelpad=3)
                ax.axvline(100 if metric == "retention_past" else 0, color="#AAAAAA", linewidth=.65, zorder=0)
                ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 101, 5)) if metric == "retention_past"
                                          else MaxNLocator(nbins=5))
                ax.grid(axis="x", linewidth=.4, alpha=.23)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.tick_params(axis="y", length=0, pad=5)
        fig.suptitle(f"Checkpoint distributions · $n={n}$", y=.977, fontsize=14)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.6, .954), ncol=5,
                   frameon=False, fontsize=8.5, columnspacing=1.6, handletextpad=.45)
        fig.text(.21, .018, "Each point: one checkpoint's sampling/tie expectation (36 checkpoints; 12 per seed).\n"
                 "Box: interquartile range; whiskers: 1.5 x IQR. Current uses 1 task at every n. "
                 "Validation: remaining 40 tasks, relative to proposal mean.", fontsize=8, va="bottom")
        stem = f"checkpoint_distributions_n{n:02d}"
        for extension in ("pdf", "png"):
            fig.savefig(output / f"{stem}.{extension}", dpi=210, bbox_inches="tight", pad_inches=.06)
        plt.close(fig)
        files.append(stem)
        print(f"Saved {stem}.pdf", flush=True)
    return files


def write_index(output: Path) -> None:
    buttons = "".join(f'<button type="button" data-n="{n}">n = {n}</button>' for n in range(1, 11))
    html = """<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BFCL checkpoint 분포</title>
<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 18px;color:#222}
button{padding:8px 14px;margin:3px;border:1px solid #287C85;background:#EAF4F5;border-radius:5px;cursor:pointer}
button[aria-pressed=true]{background:#287C85;color:white}img{width:100%;height:auto}p{line-height:1.6}</style>
<h1>Checkpoint별 retention · past gain · generalization</h1>
<p>각 점은 한 checkpoint에서 task sampling과 candidate 동점을 평균낸 기대값입니다.
Qwen·Gemma 각각 동일한 36 checkpoint(세 seed당 12개)를 비교합니다. 박스는 25-75% 범위이며,
중앙선은 중앙값, 검은 마름모는 평균입니다. Validation lift는 parent 대비가 아니라 5개 proposal 평균 대비입니다.</p>
<nav>BUTTONS</nav><p><a id="pdf" href="figures/checkpoint_distributions_n01.pdf">현재 n의 PDF</a>
 · <a href="README.md">분석 설명</a> · <a href="distribution_statistics.csv">분포 요약 CSV</a>
 · <a href="checkpoint_values.csv">Checkpoint별 값 CSV</a></p>
<img id="figure" src="figures/checkpoint_distributions_n01.png" alt="n=1 checkpoint distributions">
<script>function select(n){const stem='figures/checkpoint_distributions_n'+String(n).padStart(2,'0');
document.querySelector('#figure').src=stem+'.png';document.querySelector('#figure').alt='n='+n+' distributions';
document.querySelector('#pdf').href=stem+'.pdf';
document.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.n)===n)));}
document.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>select(Number(b.dataset.n))));
select(1);</script></html>
""".replace("BUTTONS", buttons)
    (output / "index.html").write_text(html)


def write_report(output: Path, statistics: list[dict], summary: dict) -> None:
    lookup = {(r["model"], r["weight"], r["case"], r["metric"]): r for r in statistics
              if r["scope"] == "all" and r["n"] == 10}
    lines = ["# BFCL checkpoint별 지표 분포", "",
             "n=1~10을 각각 별도의 PDF/PNG로 저장했다. 파일 하나에 4행(Qwen/Gemma x task/period mean), "
             "3열(past retention, past gain, validation lift)을 표시한다.", "",
             "- [n을 바꾸며 보기](index.html)",
             "- 모든 점은 같은 36개 checkpoint(각 seed 42/43/44당 12개)의 값이다.",
             f"- 각 점은 기존 {summary['trials']:,}회 task sampling 및 균등 동점 처리에 대한 기대값이다. "
             "개별 rollout이나 MC draw를 독립 checkpoint처럼 세지 않는다.",
             "- 박스는 IQR(25-75%), 중앙선은 중앙값, 검은 마름모는 평균이다. 수염은 1.5 x IQR이며 "
             "모든 checkpoint를 점으로 표시하므로 수염 밖 값도 빠지지 않는다.",
             "- Seed는 원(42), 삼각형(43), 사각형(44)으로 구분한다. 같은 checkpoint의 수직 오프셋은 "
             "방법·지표·n에 걸쳐 동일하다. 모든 n에서 각 지표의 축 범위를 고정했다.",
             "- Past는 현재 task를 제외한 이미 학습한 task이다. Retention은 parent가 성공한 과거 task 중 "
             "선택한 candidate도 성공한 비율, past gain은 과거 전체의 parent 대비 성공률 변화이다.",
             "- Generalization은 공통 validation remaining40에서 proposal 5개 평균 대비 selection lift(pp)이다. "
             "Parent validation 평가가 없어 parent 대비 gain으로 해석할 수 없다.",
             "- Proposal mean은 무선택 기준으로 모든 n에, 고정 Val10 선택 기준은 실제 측정한 n=10에만 표시했다. "
             "현재 task만 쓰는 방법은 모든 n에서 실제 sample 수가 1이다. "
             "n=1의 두 시점 혼합, n<3의 세 시점 혼합은 불가능하므로 표시하지 않았다.",
             "- Future 포함 방법은 retrospective 비교 조건이다. 과거 task를 선정에 재사용한 지표에는 "
             "선정·측정 task overlap이 있으므로 retention/past gain을 비편향 일반화 추정으로 해석하지 않는다.",
             "- CP 분포와 seed별 표는 기술통계다. 같은 학습 경로의 checkpoint들은 독립 반복이 아니며 "
             "36개의 점으로 통계적 유의성을 주장하지 않는다.",
             "- Gemma의 parent context는 Qwen에서 가져온 것이며 candidate 생성과 평가는 Gemma를 사용했다.",
             "- LLM 및 환경 호출 없이 기존 캐시만 사용했다.", "", "## n=10: Current + past", "",
             "| Model | Weight | Metric | Mean | Median | Min-max | Positive / zero / negative CP |",
             "|---|---|---|---:|---:|---:|---|" ]
    for model in MODELS:
        for weight in WEIGHTS:
            for metric in METRICS:
                row = lookup[model, weight, "current_past", metric]
                sign = (f"{row['positive_checkpoints']} / {row['zero_checkpoints']} / {row['negative_checkpoints']}"
                        if metric != "retention_past" else
                        f"100% retention: {row['perfect_retention_checkpoints']}/36")
                lines.append(f"| {model} | {weight} | {metric} ({row['unit']}) | {row['mean']:+.3f} | "
                             f"{row['median']:+.3f} | {row['minimum']:+.3f} ~ {row['maximum']:+.3f} | {sign} |")
    lines += ["", "## 파일", "", "- `figures/checkpoint_distributions_n01.pdf` ~ `n10.pdf`: 각 n별 12개 패널.",
              "- `checkpoint_values.csv`: 그림에 사용한 checkpoint별 값. Metric은 rate 단위와 %/pp 열을 모두 저장.",
              "- `distribution_statistics.csv`: 방법별 평균·표준편차·분위수·범위·양/음수 CP 수, 전체 및 seed별.",
              "- `manifest.json`: 입력/코드 hash, cohort, 호출 수, 해석 및 숫자 오차 기준.", ""]
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.input.read_text())
    cache = args.cache or Path(next(r["path"] for r in summary["source_files"]
                                   if Path(r["path"]).name == "full_train_metrics.json"))
    rows = validated_rows(summary) + reference_rows(summary, cache)
    statistics = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    columns = ("model", "weight", "n", "actual_n", "case", "key", "seed", "checkpoint")
    values = [{**{k: r[k] for k in columns}, **{k: r[k] for k in METRICS},
               "retention_past_pct": 100 * r["retention_past"], "gain_past_pp": 100 * r["gain_past"],
               "lift_validation40_pp": 100 * r["lift_validation40"]} for r in rows]
    write_csv(args.output / "checkpoint_values.csv", values)
    write_csv(args.output / "distribution_statistics.csv", statistics)
    files = plot(rows, args.output / "figures")
    write_index(args.output)
    write_report(args.output, statistics, summary)
    manifest = {"inputs": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in (args.input, cache)],
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "new_model_calls": 0, "new_environment_calls": 0, "checkpoints_per_model": 36,
                "fixed_checkpoint_keys": summary["fixed_checkpoint_keys"], "rows": len(rows),
                "figures": files, "statistics_rows": len(statistics), "metrics": list(METRICS),
                "zero_tolerance_pp": ZERO_TOLERANCE_PP,
                "point_unit": "One checkpoint's frozen task-sampling and uniform tie expectation",
                "uncertainty": "Descriptive CP distributions. CPs within seed are dependent; no significance claim.",
                "reference_audit": "Both reference means and sample SDs reproduce original all/seed aggregates.",
                "reference_budget": "Proposal at all n; measured fixed val10 only at n=10; current always 1 task.",
                "generalization": "Remaining validation40 success lift relative to the five-proposal mean, "
                                  "not relative to parent; parent validation scores unavailable."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(files)} distributions and {len(statistics)} descriptive summaries to {args.output}")


if __name__ == "__main__":
    main()
