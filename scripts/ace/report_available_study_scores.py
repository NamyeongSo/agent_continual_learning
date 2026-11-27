# SPDX-License-Identifier: Apache-2.0
"""Organize validated completed ACE phases; no benchmark execution or inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from exgentic.agents.ace.playbook_utils import get_playbook_stats
from scripts.ace.analyze_reflection_accumulation import PHASES, official_tgc

ROOT = Path(__file__).resolve().parents[2]
SPECS = {
    "thinking_A": ("Thinking ADD-only", "ace_appworld_reflection_accumulation_qwen38_teammate_s42_44_20261116"),
    "nonthinking_A": ("Non-thinking ADD-only", "ace_appworld_nonthinking_replication_qwen38_20261117/accumulation"),
    "thinking_AUD": ("Thinking AUD", "ace_appworld_aud_accumulation_qwen38_20261117/thinking"),
    "nonthinking_AUD": ("Non-thinking AUD", "ace_appworld_aud_accumulation_qwen38_20261117/nonthinking"),
}
METRICS = {
    "train_pct": ("train_final", "mean_pct"),
    "train_tgc": ("train_final", "tgc_rate"),
    "heldout_pct": ("val", "mean_pct"),
    "heldout_tgc": ("val", "tgc_rate"),
    "bwt_pct_all50": ("bwt", "all50"),
    "bwt_pct_prior49": ("bwt", "prior49"),
    "bwt_tgc_all50": ("bwt", "tgc_all50"),
    "bwt_tgc_prior49": ("bwt", "tgc_prior49"),
}
CONDITIONS = [f"{a}_{m}" for a in ("without_gt", "with_gt") for m in ("random", "best", "worst")]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(records):
    if not records:
        return {}
    result = {}
    for condition in CONDITIONS:
        selected = [r for r in records if r["condition"] == condition]
        result[condition] = {}
        for metric in (*METRICS, "expected_pct", "expected_tgc", "delta_pct", "delta_tgc", "context_chars"):
            values = [r[metric] for r in selected]
            if any(v is None for v in values):
                raise ValueError(f"Missing complete-seed metric: {condition}/{metric}")
            result[condition][metric] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "seeds": [r["seed"] for r in selected],
                "values": values,
            }
    return result


def collect(output):
    result = {
        "generated_at_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
        "cohorts": {},
        "matched_aud": {},
        "reflection": {},
        "source_sha256": {},
    }
    all_records = []
    for key, (title, relative) in SPECS.items():
        run = ROOT / "outputs" / relative
        protocol, status = read(run / "protocol.json"), read(run / "status.json")
        result["source_sha256"][str(run / "protocol.json")] = sha(run / "protocol.json")
        records, complete_seeds, phase_coverage = [], [], {}
        for seed in (42, 43, 44):
            phases = [p for p in PHASES if (run / f"seed{seed}/{p}_results.json").exists()]
            phase_coverage[str(seed)] = phases
            if "train_final" not in phases:
                continue
            label = "complete" if len(phases) == 3 else "_".join(phases)
            analysis_path = run / f"analysis/seed{seed}_{label}.json"
            expected_path = run / f"analysis/seed{seed}_independent_vs_accumulated.json"
            analysis, expected = read(analysis_path), read(expected_path)
            for source, digest in analysis["source_sha256"].items():
                if sha(source) != digest:
                    raise ValueError(f"Analysis is stale: {analysis_path}")
            if analysis["completed_phases"] != phases:
                raise ValueError("Analysis phase coverage differs from the saved outcomes")
            complete = len(phases) == 3 and (run / f"seed{seed}/completed.json").exists()
            if complete:
                complete_seeds.append(seed)
            for path in (analysis_path, expected_path):
                result["source_sha256"][str(path)] = sha(path)
            for condition in CONDITIONS:
                cid = f"s{seed}_{condition}"
                a, e = analysis["conditions"][cid], expected["conditions"][cid]
                cp = read(run / f"conditions/{cid}/checkpoints/050.json")
                record = {
                    "cohort": key,
                    "seed": seed,
                    "condition": condition,
                    "complete_seed": complete,
                    **{m: a.get(p, {}).get(f) for m, (p, f) in METRICS.items()},
                    "expected_pct": e["expected_independent_pct"],
                    "expected_tgc": e["expected_independent_tgc"],
                    "delta_pct": e["delta_pct"],
                    "delta_tgc": e["delta_tgc"],
                    "context_chars": len(cp["playbook"]),
                    "context_bullets": get_playbook_stats(cp["playbook"])["total_bullets"],
                }
                for phase in PHASES:
                    v = a.get(phase, {})
                    if v and (v["tasks"] != 50 or v["tgc_recorded_outcomes"] != 50):
                        raise ValueError("This table requires complete 50-task official outcomes")
                    record[phase + "_budget_failures"] = sum(v["budget_failure_reasons"].values()) if v else None
                records.append(record)
        final_records = [r for r in records if r["complete_seed"]]
        result["cohorts"][key] = {
            "title": title,
            "run": str(run),
            "status": status["state"],
            "complete_seeds": complete_seeds,
            "phase_coverage": phase_coverage,
            "execution_order": protocol.get("execution_order", "seed_major"),
            "model_settings": protocol["model_settings"],
            "records": records,
            "aggregate": aggregate(final_records),
        }
        all_records.extend(records)
    for sampler in ("thinking", "nonthinking"):
        a, aud = (result["cohorts"][f"{sampler}_{mode}"] for mode in ("A", "AUD"))
        seeds = sorted(set(a["complete_seeds"]) & set(aud["complete_seeds"]))
        if not seeds:
            continue
        pa, pb = (read(Path(c["run"]) / "protocol.json") for c in (a, aud))
        for field in ("model_settings", "appworld_policy_sha256", "actor_limits", "task_orders", "initial_playbook"):
            if pa[field] != pb[field]:
                raise ValueError(f"Unmatched A/AUD: {sampler}/{field}")
        for seed in seeds:
            for condition in CONDITIONS:
                relative = f"conditions/s{seed}_{condition}/selections.json"
                if sha(Path(a["run"]) / relative) != sha(Path(aud["run"]) / relative):
                    raise ValueError("A/AUD selections differ")
        ra = [r for r in a["records"] if r["seed"] in seeds]
        rb = [r for r in aud["records"] if r["seed"] in seeds]
        result["matched_aud"][sampler] = {
            "seeds": seeds,
            "A": aggregate(ra),
            "AUD": aggregate(rb),
            "same_selection_bytes_verified": True,
            "budget_failures": {
                mode: {p: sum(r[p + "_budget_failures"] for r in rows) for p in PHASES}
                for mode, rows in (("A", ra), ("AUD", rb))
            },
        }
    for sampler in ("thinking", "nonthinking"):
        run = Path(result["cohorts"][f"{sampler}_A"]["run"])
        source = Path(read(run / "protocol.json")["source_reflection_run"])
        initial = [read(p) for p in sorted((source / "stage1").glob("*/result.json"))]
        updated = [read(p) for p in sorted((source / "stage3").glob("*/result.json"))]
        if len(initial) != 250 or len(updated) != 1500:
            raise ValueError("Incomplete reflection bank")
        result["reflection"][sampler] = {}
        for arm in ("initial", "without_gt", "with_gt"):
            rows = initial if arm == "initial" else [r for r in updated if r["arm"] == arm]
            flags = [official_tgc(r) for r in rows]
            if any(f is None for f in flags):
                raise ValueError("Missing official Reflection-evaluation outcomes")
            result["reflection"][sampler][arm] = {
                "pct": statistics.mean(r["pct"] for r in rows),
                "tgc": statistics.mean(flags),
                "successes": sum(flags),
                "outcomes": len(rows),
            }
    result["analysis_script_sha256"] = sha(__file__)
    output.mkdir(parents=True, exist_ok=True)
    (output / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (output / "scores_by_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    render(result, output)
    plot(result, output)
    return result


def name(condition):
    return condition.replace("without_gt_", "w/o GT · ").replace("with_gt_", "w/ GT · ")


def value(metric):
    return f"{metric['mean'] * 100:.2f} ± {metric['std'] * 100:.2f}" if metric["std"] is not None else "—"


def render(result, output):
    lines = [
        f"분석 기준: {result['generated_at_kst']}.",
        "완료된 phase만 사용했습니다. PCT는 공식 개별 test 통과 비율, TGC는 공식 success flag입니다. "
        "ADD-only의 두 모드는 최종 결과이며, Thinking AUD는 완료된 seed만의 중간 결과입니다.",
        "## 실험 및 집계 범위",
        "| 실험 | 완료 seed | 집계 상태 | 실행 순서 |",
        "| --- | --- | --- | --- |",
    ]
    for c in result["cohorts"].values():
        seeds = ", ".join(map(str, c["complete_seeds"])) or "없음"
        state = (
            "최종 · n=3"
            if c["status"] == "completed" and len(c["complete_seeds"]) == 3
            else (f"중간 · n={len(c['complete_seeds'])}" if c["complete_seeds"] else "점수 대기")
        )
        lines.append(f"| {c['title']} | {seeds} | {state} | {c['execution_order']} |")
    a = result["cohorts"]["thinking_A"]["aggregate"]["with_gt_best"]
    b = result["cohorts"]["nonthinking_A"]["aggregate"]["with_gt_best"]
    lines += [
        "## 주요 관찰",
        "1. 독립 Reflection 업데이트에서는 두 모델 설정 모두 초기 대비 PCT·TGC가 상승했습니다. "
        "아래 수치는 저장된 평가의 기술통계이며, task별 효과의 유의성은 별도 분석과 함께 해석합니다.",
        f"2. w/ GT·best의 누적 Train PCT는 Thinking {a['train_pct']['mean'] * 100:.2f}%, "
        f"Non-thinking {b['train_pct']['mean'] * 100:.2f}%입니다. "
        f"TGC는 각각 {a['train_tgc']['mean'] * 100:.2f}%, {b['train_tgc']['mean'] * 100:.2f}%로 "
        "PCT 차이보다 격차가 큽니다. 부분 test 통과와 task 전체 성공을 함께 확인해야 합니다.",
        "3. 독립 평가의 best가 누적 후에도 가장 좋다는 순위는 일관되지 않습니다. "
        "best/worst 후보는 PCT로 선정했고, 동일 bank의 관측 최댓값·최솟값에는 선택 편향이 있습니다.",
        "4. Thinking AUD는 조건에 따라 A 대비 개선·하락이 함께 나타납니다. "
        "완료된 seed42·43만 비교한 중간 결과이므로 세 seed 최종 결론은 보류합니다.",
    ]
    lines += [
        "",
        "## 1. 독립 Reflection 업데이트: PCT·TGC",
        "| 모델 설정 | 초기 PCT / TGC | w/o GT PCT / TGC | w/ GT PCT / TGC |",
        "| --- | --- | --- | --- |",
    ]
    for mode, stages in result["reflection"].items():
        cells = [
            f"{stages[a]['pct'] * 100:.2f}% / {stages[a]['tgc'] * 100:.2f}%"
            for a in ("initial", "without_gt", "with_gt")
        ]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "초기는 50 task x 5회, 각 GT arm의 업데이트 평가는 50 task x 5후보 x 3회입니다. "
        "각 모드는 새 trajectory·Reflection을 생성했습니다. "
        "Thinking과 non-thinking은 여러 sampling 설정이 함께 달라집니다.",
        "## 2. ADD-only 최종 누적 성능: 두 모드 비교",
        "평균 ± seed 간 표본 표준편차(n=3)입니다. "
        "같은 train50·val50을 반복한 자료이며 150 unique task가 아닙니다. 오차막대는 신뢰구간이 아닙니다.",
    ]
    for metric, heading in (("pct", "PCT (%)"), ("tgc", "TGC (%)")):
        lines += [
            f"### {heading}",
            "| 조건 | Thinking Train | Non-thinking Train | Thinking Held-out | Non-thinking Held-out |",
            "| --- | --- | --- | --- | --- |",
        ]
        for condition in CONDITIONS:
            a, b = (result["cohorts"][k]["aggregate"][condition] for k in ("thinking_A", "nonthinking_A"))
            cells = [
                value(x[m])
                for x, m in (
                    (a, "train_" + metric),
                    (b, "train_" + metric),
                    (a, "heldout_" + metric),
                    (b, "heldout_" + metric),
                )
            ]
            lines.append(f"| {name(condition)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "![ADD-only 최종 비교](add_only_comparison.png)",
        "### BWT prior49 (pp)",
        "| 조건 | Thinking PCT | Non-thinking PCT | Thinking TGC | Non-thinking TGC |",
        "| --- | --- | --- | --- | --- |",
    ]
    for condition in CONDITIONS:
        a, b = (result["cohorts"][k]["aggregate"][condition] for k in ("thinking_A", "nonthinking_A"))
        cells = [
            value(x[m])
            for x, m in ((a, "bwt_pct_prior49"), (b, "bwt_pct_prior49"), (a, "bwt_tgc_prior49"), (b, "bwt_tgc_prior49"))
        ]
        lines.append(f"| {name(condition)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "BWT는 실제 누적 Ct와 최종 C50의 대응 차이입니다. 마지막 task의 독립 반복을 제외한 prior49를 본문에, "
        "all50은 종합 CSV·JSON에, task bootstrap은 각 seed의 분석 JSON에 보존합니다. "
        "양수는 후속 학습 뒤 성능 증가, 음수는 성능 감소를 뜻하며, 단회 평가 잡음도 포함합니다.",
        "## 3. 비누적 기대 vs 누적 실제",
    ]
    for key in ("thinking_A", "nonthinking_A"):
        c = result["cohorts"][key]
        lines += [
            f"### {c['title']} · seed42/43/44 평균",
            "| 조건 | 기대 PCT → 실제 PCT | 차이(pp) | 기대 TGC → 실제 TGC | 차이(pp) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for condition, v in c["aggregate"].items():
            lines.append(
                f"| {name(condition)} | {v['expected_pct']['mean'] * 100:.2f}% → {v['train_pct']['mean'] * 100:.2f}% "
                f"| {v['delta_pct']['mean'] * 100:+.2f} | {v['expected_tgc']['mean'] * 100:.2f}% → "
                f"{v['train_tgc']['mean'] * 100:.2f}% | {v['delta_tgc']['mean'] * 100:+.2f} |"
            )
    lines += [
        "",
        "![비누적 기대와 누적 실제](independent_vs_accumulated.png)",
        "기대값은 실제 선정 Reflection의 독립 ADD-only 업데이트 3회 점수입니다. "
        "실제값은 같은 task의 C50 평가 1회입니다. "
        "같은 bank로 best/worst를 선정하는 편향, 반복 수 차이, Curator 생성·예산·평가 잡음이 포함됩니다. "
        "이 차이를 순수한 축적·망각으로 단정하지 않습니다. 독립 Reflection Held-out 기준은 측정하지 않았습니다.",
        "## 4. AUD 중간 결과: 동일 seed의 A와 비교",
    ]
    for mode, pair in result["matched_aud"].items():
        lines += [
            f"### {mode} · 완료 seed {pair['seeds']}만 비교",
            "A의 전체3 seed 평균과 AUD의2 seed 평균을 섞지 않고, "
            "양쪽 모두 완료된 동일 seed만 대응했습니다. "
            "task 순서·모델·정책·예산과 정확한 선정 파일 SHA256을 확인했습니다.",
            "| 조건 | Train PCT A → AUD | Train TGC A → AUD | Held-out PCT A → AUD | Held-out TGC A → AUD |",
            "| --- | --- | --- | --- | --- |",
        ]
        for condition in CONDITIONS:
            a, b = pair["A"][condition], pair["AUD"][condition]
            cells = [
                f"{a[m]['mean'] * 100:.2f}% → {b[m]['mean'] * 100:.2f}%"
                for m in ("train_pct", "train_tgc", "heldout_pct", "heldout_tgc")
            ]
            lines.append(f"| {name(condition)} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "기타 예산 종료(Train / Ct / Held-out): "
            + " · ".join(
                f"{m} {v['train_final']}/{v['bwt_checkpoint']}/{v['val']}회" for m, v in pair["budget_failures"].items()
            ),
        ]
    lines += [
        "",
        "![AUD와 같은 seed의 ADD-only 비교](aud_matched_seed_comparison.png)",
        "AUD의 미완료 seed와 미완료 cohort는 최종 순위에 넣지 않습니다. "
        "AUD는 native 프롬프트와 허용 연산이 함께 달라지는 비교이며, "
        "예산 종료·독립 평가 잡음도 함께 반영됩니다. Non-thinking AUD는 아직 점수가 없습니다.",
        "## 5. seed42 Held-out 점수",
        "각 행은 val50에 대해 task당 1회 평가한 결과입니다. 다른 seed의 원자료도 CSV에 함께 보존했습니다.",
        "| 실험 | 조건 | PCT (%) | TGC (%) |",
        "| --- | --- | --- | --- |",
    ]
    for cohort in result["cohorts"].values():
        for row in cohort["records"]:
            if row["seed"] == 42 and row["heldout_pct"] is not None:
                lines.append(
                    f"| {cohort['title']} | {name(row['condition'])} "
                    f"| {row['heldout_pct'] * 100:.2f} | {row['heldout_tgc'] * 100:.2f} |"
                )
    lines += [
        "## 재현 자료",
        "scores_by_seed.csv는 조건·seed별 원자료, scores.json은 집계와 source SHA256입니다. "
        "각 run의 seed*_complete.json 및 seed*_independent_vs_accumulated.json에 "
        "task 대응과 원본 provenance를 보존했습니다. "
        "C0~C50·원래 task 분할·모델·teammate policy는 그대로 유지했습니다. "
        "40턴 종료는 실제 공식 PCT, 나머지 승인된 예산 실패는 0점입니다.",
        "분석은 저장 결과를 CPU로 처리했으며 추가 학습·추론 호출이 없습니다.",
    ]
    (output / "REPORT.md").write_text("\n\n".join(lines).replace("|\n\n|", "|\n|") + "\n")


def plot(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [n.replace("without_gt_", "No GT\n").replace("with_gt_", "GT\n") for n in CONDITIONS]
    panels = [
        ("train_pct", "Train PCT (%)"),
        ("heldout_pct", "Held-out PCT (%)"),
        ("bwt_pct_prior49", "PCT BWT prior49 (pp)"),
        ("train_tgc", "Train TGC (%)"),
        ("heldout_tgc", "Held-out TGC (%)"),
        ("bwt_tgc_prior49", "TGC BWT prior49 (pp)"),
    ]

    def performance(groups, title, filename):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
        for ax, (metric, panel) in zip(axes.flat, panels, strict=True):
            for label, data, offset, color in groups:
                for i, c in enumerate(CONDITIONS):
                    v = data[c][metric]
                    ax.errorbar(
                        i + offset,
                        v["mean"] * 100,
                        yerr=None if v["std"] is None else v["std"] * 100,
                        fmt="o",
                        color=color,
                        capsize=3,
                        label=label if i == 0 else None,
                    )
            ax.set_xticks(range(6), labels, fontsize=8)
            ax.set_title(panel)
            ax.grid(axis="y", alpha=0.2)
            if metric.startswith("bwt"):
                ax.axhline(0, color="#555555", lw=0.8)
            ax.legend(fontsize=7)
        fig.suptitle(title)
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    performance(
        [
            ("Thinking", result["cohorts"]["thinking_A"]["aggregate"], -0.12, "#2563a6"),
            ("Non-thinking", result["cohorts"]["nonthinking_A"]["aggregate"], 0.12, "#c8792d"),
        ],
        "ADD-only final: seeds42/43/44; mean +/- sample SD (n=3), not confidence intervals",
        "add_only_comparison.png",
    )
    if "thinking" in result["matched_aud"]:
        pair = result["matched_aud"]["thinking"]
        performance(
            [
                ("ADD-only, matched seeds", pair["A"], -0.12, "#2563a6"),
                ("AUD, matched seeds", pair["AUD"], 0.12, "#27815a"),
            ],
            f"Thinking AUD INTERIM vs matched ADD-only: seeds {pair['seeds']}; mean +/- sample SD",
            "aud_matched_seed_comparison.png",
        )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for row, key in enumerate(("thinking_A", "nonthinking_A")):
        for col, metric in enumerate(("pct", "tgc")):
            ax = axes[row, col]
            data = result["cohorts"][key]["aggregate"]
            for i, c in enumerate(CONDITIONS):
                a, b = (data[c][m]["mean"] * 100 for m in ("expected_" + metric, "train_" + metric))
                ax.plot([i - 0.12, i + 0.12], [a, b], color="#aaaaaa")
                ax.scatter(i - 0.12, a, marker="s", color="#888888", label="Selected independent" if i == 0 else None)
                ax.scatter(i + 0.12, b, color="#2563a6", label="Accumulated actual" if i == 0 else None)
            ax.set_xticks(range(6), labels, fontsize=8)
            ax.set_title(f"{result['cohorts'][key]['title']}: Train {metric.upper()} (%)")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Independent expectation vs accumulated actual: same-bank selection bias, 3 vs 1 repeats")
    fig.savefig(output / "independent_vs_accumulated.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.output.resolve())
    print(json.dumps({k: c["complete_seeds"] for k, c in report["cohorts"].items()}))
