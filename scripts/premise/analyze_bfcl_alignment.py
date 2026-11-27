# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Plot Train-10 and Val-10 correlation with the full train and validation targets."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.premise.run_bfcl_alignment import correlation, ranks, selected_improvement  # noqa: E402

TARGETS = ("full_train", "full_validation", "train_remaining40", "validation_remaining40")
SELECTORS = ("train10", "val10")


def calculate(points: list[dict]) -> dict:
    """Use the same candidates and checkpoint groups for both bars in each comparison."""
    per_checkpoint = []
    for point in points:
        scores = point["scores"]
        measures, proposals = {}, {}
        for target in TARGETS:
            values = scores[target]
            parent = point.get("parent_scores", {}).get(target)
            proposals[target] = {
                "mean_performance": statistics.mean(values), "parent_performance": parent,
                "mean_gain": statistics.mean(values) - parent if parent is not None else None,
            }
            measures[target] = {selector: {
                "pearson": correlation(scores[selector], values),
                "spearman": correlation(ranks(scores[selector]), ranks(values)),
                "selected_performance": selected_improvement(scores[selector], values),
                "selected_gain": selected_improvement(scores[selector], values) - parent
                if parent is not None else None,
                "selection_gain": selected_improvement(scores[selector], values) - statistics.mean(values),
                "regret": max(values) - selected_improvement(scores[selector], values),
            } for selector in SELECTORS}
        per_checkpoint.append({**point, "measures": measures, "proposal_statistics": proposals})
    aggregate = {}
    for target in TARGETS:
        paired = [p for p in per_checkpoint if all(
            p["measures"][target][s]["pearson"] is not None for s in SELECTORS)]
        summary = {"total_checkpoints": len(points), "paired_checkpoints": len(paired),
                   "paired_checkpoint_keys": [p["key"] for p in paired],
                   "paired_seeds": sorted({p["seed"] for p in paired})}
        parents = [p["proposal_statistics"][target]["parent_performance"] for p in per_checkpoint]
        complete_baselines = all(value is not None for value in parents)
        summary.update(
            proposal_mean_performance=statistics.mean(p["proposal_statistics"][target]["mean_performance"]
                                                      for p in per_checkpoint),
            parent_baseline_available_checkpoints=sum(value is not None for value in parents),
            mean_parent_performance=statistics.mean(parents) if complete_baselines else None,
            proposal_mean_gain=statistics.mean(p["proposal_statistics"][target]["mean_gain"]
                                               for p in per_checkpoint) if complete_baselines else None,
            gain_checkpoint_count=len(points),
        )
        for selector in SELECTORS:
            valid = [p for p in per_checkpoint if p["measures"][target][selector]["pearson"] is not None]
            summary[selector] = {
                "mean_pearson": statistics.mean(p["measures"][target][selector]["pearson"] for p in paired)
                if paired else None,
                "mean_spearman": statistics.mean(p["measures"][target][selector]["spearman"] for p in paired)
                if paired else None,
                "individually_valid_checkpoints": len(valid),
                "individually_valid_mean_pearson": statistics.mean(
                    p["measures"][target][selector]["pearson"] for p in valid) if valid else None,
                "mean_selected_performance": statistics.mean(
                    p["measures"][target][selector]["selected_performance"] for p in per_checkpoint),
                "mean_selected_gain": statistics.mean(
                    p["measures"][target][selector]["selected_gain"] for p in per_checkpoint)
                if complete_baselines else None,
                "mean_selection_gain": statistics.mean(
                    p["measures"][target][selector]["selection_gain"] for p in per_checkpoint),
                "mean_regret": statistics.mean(p["measures"][target][selector]["regret"] for p in per_checkpoint),
            }
        summary["mean_selected_difference_train10_minus_val10"] = statistics.mean(
            p["measures"][target]["train10"]["selected_performance"] -
            p["measures"][target]["val10"]["selected_performance"] for p in per_checkpoint)
        if paired:
            seeds = summary["paired_seeds"]
            summary["seed_equal_pearson"] = {s: statistics.mean(
                statistics.mean(p["measures"][target][s]["pearson"] for p in paired if p["seed"] == seed)
                for seed in seeds) for s in SELECTORS}
        aggregate[target] = summary
    return {"checkpoints": per_checkpoint, "aggregate": aggregate}


def load_points(inputs: list[Path], *, allow_partial: bool = False) -> tuple[list[dict], list[dict]]:
    points, errors, seen = [], [], set()
    for source in inputs:
        plan = json.loads((source / "launch_manifest.json").read_text())
        rows = defaultdict(list)
        for path in list((source / "results").glob("*/*/*/*.json")):
            row = json.loads(path.read_text())
            if row["partition"] in ("train", "val"):
                rows[(row["checkpoint_key"], row["context"], row["partition"])].append(row)
        for point in plan["checkpoints"]:
            key = point["key"]
            if key in seen:
                raise ValueError(f"Duplicate checkpoint group: {key}")
            seen.add(key)
            past, val10 = set(point["past_task_ids"]), set(point["validation_task_ids"])
            parent_rows = rows[(key, "parent", "train")]
            parent = {r["task_id"]: r["score"] for r in parent_rows}
            if len(parent_rows) != len(parent) or not set(parent).issubset(point["train_task_ids"]):
                raise ValueError(f"Invalid parent task identities at {key}")
            if len(parent_rows) != 50 or set(parent) != set(point["train_task_ids"]):
                if not allow_partial:
                    raise ValueError(f"Incomplete parent baseline at {key}")
            if not parent or not parent.keys() - past:
                continue
            parent_scores = {
                "full_train": statistics.mean(parent.values()),
                "train_remaining40": statistics.mean(parent[t] for t in parent.keys() - past),
                "full_validation": None, "validation_remaining40": None,
            }
            point_errors = [{"source": str(source), "checkpoint_key": key, "candidate_index": None,
                             "context": "parent", "partition": "train", "task_id": r["task_id"],
                             "score": r["score"], "in_score_sample": r["task_id"] in past}
                            for r in parent_rows if r["status"] == "error"]
            scores = {name: [] for name in (*TARGETS, *SELECTORS)}
            hashes = []
            coverage = {"parent_full_train": len(parent),
                        "parent_train_remaining40": len(parent.keys() - past),
                        **{name: [] for name in (*TARGETS, *SELECTORS)}}
            usable = True
            for index in range(1, 6):
                context = f"candidate_{index:02d}"
                train_rows, val_rows = rows[(key, context, "train")], rows[(key, context, "val")]
                if len(train_rows) != 50 or len(val_rows) != 50:
                    if not allow_partial:
                        raise ValueError(f"Incomplete checkpoint {key}/{context}: {len(train_rows)}/{len(val_rows)}")
                train = {r["task_id"]: r["score"] for r in train_rows}
                val = {r["task_id"]: r["score"] for r in val_rows}
                if len(train_rows) != len(train) or len(val_rows) != len(val):
                    raise ValueError(f"Duplicate task results at {key}/{context}")
                assert set(train).issubset(point["train_task_ids"])
                assert set(val).issubset(point["validation_full_task_ids"])
                assert all(r.get("steps", 0) <= plan["max_steps"] for r in (*train_rows, *val_rows))
                observed_past, observed_val10 = train.keys() & past, val.keys() & val10
                if not all((train, val, observed_past, observed_val10, train.keys() - past, val.keys() - val10)):
                    usable = False
                    break
                scores["train10"].append(statistics.mean(train[t] for t in observed_past))
                scores["val10"].append(statistics.mean(val[t] for t in observed_val10))
                scores["full_train"].append(statistics.mean(train.values()))
                scores["full_validation"].append(statistics.mean(val.values()))
                scores["train_remaining40"].append(statistics.mean(train[t] for t in train.keys() - past))
                scores["validation_remaining40"].append(statistics.mean(val[t] for t in val.keys() - val10))
                hashes.append(train_rows[0]["context_sha256"])
                for name, count in (("full_train", len(train)), ("full_validation", len(val)),
                                    ("train10", len(observed_past)), ("val10", len(observed_val10)),
                                    ("train_remaining40", len(train.keys() - past)),
                                    ("validation_remaining40", len(val.keys() - val10))):
                    coverage[name].append(count)
                point_errors.extend({"source": str(source), "checkpoint_key": key, "candidate_index": index - 1,
                                     "context": context, "partition": r["partition"],
                                     "task_id": r["task_id"], "score": r["score"],
                                     "in_score_sample": r["task_id"] in (past if r["partition"] == "train" else val10)}
                                    for r in (*train_rows, *val_rows) if r["status"] == "error")
            if not usable:
                continue
            errors.extend(point_errors)
            points.append({"key": key, "seed": point["seed"], "checkpoint": point["checkpoint"],
                           "source": str(source), "scores": scores, "parent_scores": parent_scores,
                           "context_sha256": hashes, "coverage": coverage,
                           "evaluation_complete": len(parent) == 50 and
                           all(n == 50 for name in TARGETS[:2] for n in coverage[name])})
    return points, errors


def error_sensitivity(points: list[dict], errors: list[dict]) -> dict:
    """Bound unknown binary rewards without changing recorded results or issuing model calls."""
    if len(errors) > 10:
        return {"skipped": "More than 10 failed candidate evaluations; exhaustive binary bounds omitted"}
    scenarios = []
    for outcomes in itertools.product((0, 1), repeat=len(errors)):
        modified = json.loads(json.dumps(points))
        by_key = {p["key"]: p for p in modified}
        for error, outcome in zip(errors, outcomes, strict=True):
            point = by_key[error["checkpoint_key"]]
            scores = point["scores"]
            candidate, change = error["candidate_index"], outcome - error["score"]
            if candidate is None:
                point["parent_scores"]["full_train"] = round(
                    point["parent_scores"]["full_train"] + change / 50, 12)
                if not error["in_score_sample"]:
                    point["parent_scores"]["train_remaining40"] = round(
                        point["parent_scores"]["train_remaining40"] + change / 40, 12)
                continue
            full = "full_train" if error["partition"] == "train" else "full_validation"
            scores[full][candidate] = round(scores[full][candidate] + change / 50, 12)
            if error["in_score_sample"]:
                selector = "train10" if error["partition"] == "train" else "val10"
                scores[selector][candidate] = round(scores[selector][candidate] + change / 10, 12)
            else:
                remaining = "train_remaining40" if error["partition"] == "train" else "validation_remaining40"
                scores[remaining][candidate] = round(scores[remaining][candidate] + change / 40, 12)
        scenarios.append(calculate(modified)["aggregate"])
    bounds = {target: {selector: {
        "mean_pearson_bounds": [min(values), max(values)] if values else None,
        "paired_checkpoint_counts": sorted({s[target]["paired_checkpoints"] for s in scenarios}),
    } for selector in SELECTORS
        for values in [[s[target][selector]["mean_pearson"] for s in scenarios
                        if s[target][selector]["mean_pearson"] is not None]]} for target in TARGETS}
    for target in TARGETS:
        gains = [s[target]["proposal_mean_gain"] for s in scenarios if s[target]["proposal_mean_gain"] is not None]
        bounds[target]["proposal_mean_gain_bounds"] = [min(gains), max(gains)] if gains else None
        for selector in SELECTORS:
            gains = [s[target][selector]["mean_selected_gain"] for s in scenarios
                     if s[target][selector]["mean_selected_gain"] is not None]
            bounds[target][selector]["mean_selected_gain_bounds"] = [min(gains), max(gains)] if gains else None
    return bounds


def draw_correlation(axis, summary: dict, title: str) -> None:
    values = [summary[s]["mean_pearson"] for s in SELECTORS]
    axis.bar([0, 1], [v if v is not None else 0 for v in values], width=0.52,
             color=["#138A72", "#DE8F05"])
    for index, value in enumerate(values):
        axis.text(index, value + (0.05 if value >= 0 else -0.05) if value is not None else 0,
                  f"{value:+.3f}" if value is not None else "Undefined", ha="center",
                  va="bottom" if value is None or value >= 0 else "top", fontsize=12)
    axis.axhline(0, color="#888888", lw=0.8)
    axis.set(title=title, ylabel="Mean within-checkpoint Pearson correlation", ylim=(-1.12, 1.12))
    axis.set_xticks([0, 1], ["Train 10", "Val 10"])
    axis.text(0.5, 0.02, f"Both correlations defined: {summary['paired_checkpoints']}/"
              f"{summary['total_checkpoints']} checkpoints", transform=axis.transAxes,
              ha="center", va="bottom", fontsize=9, color="#555555")


def draw_gain(axis, summary: dict, target: str) -> None:
    gain = summary["proposal_mean_gain"]
    if gain is not None:
        title = "Selected candidate gain on full train"
        values = [summary[s]["mean_selected_gain"] * 100 for s in SELECTORS]
        baseline = gain * 100
        ylabel = "Gain over parent (percentage points)"
        label = f"Proposal mean gain: {baseline:+.2f} pp"
    else:
        title = "Selection gain on full validation"
        values = [summary[s]["mean_selection_gain"] * 100 for s in SELECTORS]
        baseline = 0
        ylabel = "Gain over mean proposal (percentage points)"
        label = f"Mean proposal: {summary['proposal_mean_performance'] * 100:.2f}% reward"
    axis.bar([0, 1], values, width=0.52, color=["#138A72", "#DE8F05"])
    span = max([0, baseline, *values]) - min([0, baseline, *values])
    margin = max(span * 0.35, 0.4)
    lower, upper = min([0, baseline, *values]) - margin, max([0, baseline, *values]) + margin
    axis.set_ylim(lower, upper)
    axis.axhline(0, color="#888888", lw=0.8)
    axis.axhline(baseline, color="#444444", lw=1.4, linestyle="--", label=label)
    for index, value in enumerate(values):
        axis.text(index, value + (margin * 0.12 if value >= 0 else -margin * 0.12), f"{value:+.2f}",
                  ha="center", va="bottom" if value >= 0 else "top", fontsize=13)
    axis.set(title=title, ylabel=ylabel)
    axis.set_xticks([0, 1], ["Train 10", "Val 10"])
    axis.legend(loc="upper right", frameon=False, fontsize=9)
    note = f"All {summary['gain_checkpoint_count']} checkpoints"
    if gain is None:
        note += " · Parent-relative gain not measured"
    else:
        additional = [summary[s]["mean_selection_gain"] * 100 for s in SELECTORS]
        note += f" · Selection adds {additional[0]:+.2f} / {additional[1]:+.2f} pp"
    axis.text(0.5, 0.02, note, transform=axis.transAxes, ha="center", va="bottom", fontsize=9, color="#555555")


def plot_results(result: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    titles = ("Correlation with full train experience", "Correlation with full validation experience")
    progress = result.get("partial_progress")

    def label_progress(figure, axes):
        if progress is None:
            return
        figure.suptitle(f"BFCL · PROVISIONAL · {progress['usable_checkpoints']}/"
                       f"{progress['planned_checkpoints']} checkpoints", fontsize=13)
        for axis in axes:
            axis.set_xticks([0, 1], ["Train 10\n(observed subset)", "Val 10\n(observed subset)"])
        coverage = progress["candidate_coverage"]
        counts = " · ".join(f"{label}: {coverage[name]['mean']:.1f}/{total}"
                            for name, label, total in (("train10", "Train 10", 10), ("val10", "Val 10", 10),
                                                       ("full_train", "Train target", 50),
                                                       ("full_validation", "Val target", 50)))
        figure.supxlabel("Means use completed evaluations only; task subsets differ across candidates.\n"
                          f"Average coverage — {counts}\n"
                          f"Fully evaluated checkpoints: {progress['complete_checkpoints']}/"
                          f"{progress['planned_checkpoints']} · Snapshot {progress['snapshot_at'][:19]} UTC",
                          fontsize=8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for index, (target, title) in enumerate(zip(TARGETS[:2], titles, strict=True)):
        summary = result["aggregate"][target]
        draw_correlation(axes[0, index], summary, title)
        draw_gain(axes[1, index], summary, target)
    fig.suptitle(f"BFCL · {len(result['checkpoints'])} checkpoints · 5 proposals per checkpoint", fontsize=14)
    fig.supxlabel("Each full target has 50 tasks and includes its own 10-task score sample.", fontsize=9)
    label_progress(fig, axes.flat)
    fig.savefig(output / "correlation_train_validation.png", dpi=180)
    fig.savefig(output / "correlation_train_validation.pdf")
    plt.close(fig)
    for target, title in zip(TARGETS[:2], titles, strict=True):
        summary = result["aggregate"][target]
        fig, axes = plt.subplots(2, 1, figsize=(6.5, 8), layout="constrained")
        draw_correlation(axes[0], summary, title)
        draw_gain(axes[1], summary, target)
        label_progress(fig, axes)
        fig.savefig(output / f"correlation_{target}.png", dpi=180)
        fig.savefig(output / f"correlation_{target}.pdf")
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(7, 5), layout="constrained")
        draw_gain(axis, summary, target)
        label_progress(fig, [axis])
        fig.savefig(output / f"gain_{target}.png", dpi=180)
        fig.savefig(output / f"gain_{target}.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Plot provisional means using only observed tasks; never fill missing rewards with zero.")
    args = parser.parse_args()
    points, errors = load_points([p.resolve() for p in args.input], allow_partial=args.allow_partial)
    if not points:
        raise ValueError("No checkpoint has observations for all five candidates, both score samples, and targets")
    result = calculate(points)
    result.update(candidate_execution_errors=[e for e in errors if e["context"] != "parent"],
                  parent_execution_errors=[e for e in errors if e["context"] == "parent"],
                  error_sensitivity=error_sensitivity(points, errors) if not args.allow_partial else
                  {"skipped": "Incomplete evaluations require bounds for missing tasks as well as execution errors"},
                  aggregation="Arithmetic mean of within-checkpoint correlations, paired-defined groups for both bars",
                  target_interpretation="Empirical performance on 50 tasks; a parent baseline is constant within a "
                                        "checkpoint and does not change correlation",
                  overlap="Full train includes Train-10; full validation includes Val-10",
                  proposal_gain="Mean of all five candidate performances minus the parent performance; "
                                "averaged over all checkpoints, including those with undefined correlation",
                  selection_gain="Selected candidate performance minus the uniform proposal mean; "
                                 "parent-relative selected gain equals proposal mean gain plus selection gain",
                  inference="Descriptive results; checkpoints from the same seed are dependent")
    if args.allow_partial:
        planned = [p for source in args.input
                   for p in json.loads((source / "launch_manifest.json").read_text())["checkpoints"]]
        candidate_coverage = {}
        for name in (*TARGETS, *SELECTORS):
            counts = [n for point in points for n in point["coverage"][name]]
            candidate_coverage[name] = {"mean": statistics.mean(counts), "min": min(counts), "max": max(counts)}
        used_keys = {p["key"] for p in points}
        result["partial_progress"] = {
            "snapshot_at": datetime.now(UTC).isoformat(), "planned_checkpoints": len(planned),
            "usable_checkpoints": len(points), "complete_checkpoints": sum(p["evaluation_complete"] for p in points),
            "omitted_checkpoint_keys": [p["key"] for p in planned if p["key"] not in used_keys],
            "candidate_coverage": candidate_coverage,
            "parent_train_coverage_mean": statistics.mean(p["coverage"]["parent_full_train"] for p in points),
            "method": "Observed-task means estimate the fixed Train10/Val10 scores and full50 targets. "
                      "Each candidate can have a different observed subset; completion need not be random. "
                      "Missing rewards are excluded, and observed execution failures retain their recorded zero score.",
            "source_status": {str(source): json.loads((source / "status.json").read_text())
                              for source in args.input if (source / "status.json").exists()},
        }
        result["target_interpretation"] = "Provisional estimates of 50-task target performance from observed tasks"
        result["proposal_gain"] = "Provisional observed candidate mean minus observed parent mean"
        result["inference"] += "; incomplete tasks and completion-time bias can change correlations and gains"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "correlation_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    with (args.output / "checkpoint_correlations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["checkpoint", "target", *SELECTORS])
        writer.writeheader()
        for point in result["checkpoints"]:
            for target in TARGETS:
                writer.writerow({"checkpoint": point["key"], "target": target,
                                 **{s: point["measures"][target][s]["pearson"] for s in SELECTORS}})
    plot_results(result, args.output)
    text = [f"BFCL: {len(points)} checkpoints x 5 candidates", "",
            "각 CP에서 후보 5개에 대한 Pearson correlation을 구하고 CP별 값을 평균했습니다.",
            "두 막대에는 두 상관계수가 모두 정의되는 동일한 CP 집합을 사용했습니다.",
            "", "| 평가 대상 | Train 10 | Val 10 | 유효 CP |", "|---|---:|---:|---:|"]
    if args.allow_partial:
        progress = result["partial_progress"]
        text[:0] = ["중간 추정 결과 — 전체 평가 완료 전", "",
                    f"반영 CP {len(points)}/{progress['planned_checkpoints']}, "
                    f"평가가 모두 끝난 CP {progress['complete_checkpoints']}개.",
                    "Train10/Val10과 full50 target은 현재 관측된 task만으로 평균했습니다.",
                    "후보별 관측 task가 다르며, 미평가 task를 0점 처리하지 않았습니다.",
                    "빠르게 완료되는 평가만 먼저 반영되어 최종 결과와 차이가 날 수 있습니다.",
                    f"Task 관측 수: {json.dumps(progress['candidate_coverage'], ensure_ascii=False)}", ""]
    for target in TARGETS:
        summary = result["aggregate"][target]
        values = [f"{summary[s]['mean_pearson']:+.3f}" if summary[s]["mean_pearson"] is not None else "미정의"
                  for s in SELECTORS]
        text.append(f"| {target} | {values[0]} | {values[1]} | "
                    f"{summary['paired_checkpoints']}/{summary['total_checkpoints']} |")
    train = result["aggregate"]["full_train"]
    text.extend(["", f"Proposal 및 selection gain은 전체 {len(points)}개 CP에서 평균했습니다.",
                 "상관계수의 유효 CP 집합과 다르며, 후보 5개 전체를 선택 전 평균한 값입니다.", "",
                 "| Full train 기준 (%p) | Train 10 | Val 10 |", "|---|---:|---:|",
                 f"| Proposal 평균 gain | {train['proposal_mean_gain'] * 100:+.2f} | "
                 f"{train['proposal_mean_gain'] * 100:+.2f} |",
                 f"| Selection 추가 gain | {train['train10']['mean_selection_gain'] * 100:+.2f} | "
                 f"{train['val10']['mean_selection_gain'] * 100:+.2f} |",
                 f"| 선택한 후보의 부모 대비 gain | {train['train10']['mean_selected_gain'] * 100:+.2f} | "
                 f"{train['val10']['mean_selected_gain'] * 100:+.2f} |", "",
                 "선택한 후보의 gain = Proposal 평균 gain + Selection 추가 gain.",
                 "Validation의 부모 대비 proposal gain은 부모 평가가 없어 미측정입니다.",
                 "Validation 그림에는 proposal 평균 성공률과 무작위 후보 대비 selection 추가 gain을 표시했습니다.",
                 "", "full train/validation은 각각 점수를 계산한 10개를 포함한 50개 task입니다.",
                 "따라서 높은 상관의 일부는 task 중복으로 발생할 수 있습니다. 나머지 40개 결과도 JSON에 기록했습니다.",
                 "validation 부모 평가 없이도 상관은 계산할 수 있습니다. "
                 "부모 성능은 CP 내 후보 모두에 같은 상수입니다.",
                 f"부모 및 후보 평가에서 실행 오류 {len(errors)}개가 있었습니다. "
                 "원래 결과의 0점 처리와 가정적 0/1 민감도를 기록했습니다.",
                 "동일 seed의 CP는 독립 표본이 아니며, 위 수치는 기술 통계입니다.", ""])
    if args.allow_partial:
        text = [line for line in text if "가정적 0/1 민감도" not in line]
        text.append(f"관측된 실행 오류 {len(errors)}건은 원래 0점을 유지했습니다. 미평가 task 민감도는 미계산입니다.")
    (args.output / "correlation_analysis_ko.md").write_text("\n".join(text))
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
