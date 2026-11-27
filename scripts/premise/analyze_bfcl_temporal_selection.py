# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Compare six chronological training-task selectors using saved BFCL candidate rewards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.plot_bfcl_sample_size_contours import ellipse_points, gaussian_moments  # noqa: E402

CASES = ("current", "current_past", "past", "future", "current_future", "current_past_future")
LABELS = ("Current (1 task)", "Current + past", "Past", "Future", "Current + future", "Current + past + future")
WEIGHTS = ("task_mean", "period_mean")
WEIGHT_LABELS = ("Task mean", "Period mean")
TARGETS = ("full_train", "full_validation")
COLORS = ("#6A6A6A", "#15555D", "#287C85", "#A34F65", "#78618E", "#AF8238")
FILLS = ("#F2F2F2", "#E5EFF0", "#EAF4F5", "#F9EEF1", "#F3EFF7", "#F8F2E7")
MODE_COLORS = ("#287C85", "#A34F65")
MODE_FILLS = ("#EAF4F5", "#F9EEF1")
VALIDATION_COLOR = "#A34F65"
VALIDATION_FILL = "#F9EEF1"


def allocation(case: str, n: int) -> tuple[int, int, int] | None:
    if n < 1 or case not in CASES:
        raise ValueError("Expected a known selector and a positive total task count")
    if case == "current":
        return 1, 0, 0
    if case == "past":
        return 0, n, 0
    if case == "future":
        return 0, 0, n
    if case == "current_past" and n >= 2:
        return 1, n - 1, 0
    if case == "current_future" and n >= 2:
        return 1, 0, n - 1
    if case == "current_past_future" and n >= 3:
        return 1, n // 2, (n - 1) // 2
    return None


def relocated(path: str | Path, archive: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    marker = "outputs"
    if marker not in path.parts:
        raise FileNotFoundError(path)
    return archive.joinpath(*path.parts[path.parts.index(marker) + 1:])


def ordered_pools(point: dict, frozen: dict, archive: Path) -> dict:
    source = relocated(point["source"], archive)
    snapshot = source / "frozen" / point["key"] / "checkpoint.json"
    raw = snapshot.read_bytes()
    if hashlib.sha256(raw).hexdigest() != frozen["source_checkpoint_sha256"]:
        raise ValueError(f"Frozen checkpoint digest differs: {snapshot}")
    order = [task for benchmark, task in json.loads(raw)["config"]["order"] if benchmark == "bfcl"]
    t = point["checkpoint"]
    if len(order) != 50 or len(set(order)) != 50 or set(order) != set(frozen["train_task_ids"]):
        raise ValueError("Expected one complete, unique 50-task training pass")
    if order[:t] != frozen["seen_task_ids"] or order[t] != frozen["proposal_task"]:
        raise ValueError("Current/past task identity differs from the frozen manifest")
    if set(order) & set(frozen["validation_full_task_ids"]):
        raise ValueError("Training and validation tasks overlap")
    return {"train_order": order, "current": [order[t]], "past": order[:t], "future": order[t + 1:],
            "snapshot_sha256": hashlib.sha256(raw).hexdigest()}


def load_model(reference: dict, archive: Path, io_workers: int) -> dict:
    source = relocated(reference["source"], archive)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError(f"Original correlation summary digest differs: {source}")
    summary = json.loads(raw)
    plans, points = {}, []
    for old, expected in reference["source_manifests"].items():
        folder = relocated(old, archive)
        manifest = (folder / "launch_manifest.json").read_bytes()
        if hashlib.sha256(manifest).hexdigest() != expected:
            raise ValueError(f"Original manifest digest differs: {folder}")
        plans[old] = {point["key"]: point for point in json.loads(manifest)["checkpoints"]}
    for original in sorted(summary["checkpoints"], key=lambda point: point["key"]):
        if not original["evaluation_complete"]:
            raise ValueError(f"Incomplete checkpoint: {original['key']}")
        frozen = plans[original["source"]][original["key"]]
        pools = ordered_pools(original, frozen, archive)
        points.append({"key": original["key"], "seed": original["seed"], "checkpoint": original["checkpoint"],
                       "source": str(relocated(original["source"], archive)), "scores": original["scores"],
                       "parent_scores": original["parent_scores"], "context_sha256": original["context_sha256"],
                       "manifest_train_order": frozen["train_task_ids"], **pools})

    def read_result(job):
        p, candidate, index, task = job
        path = Path(p["source"]) / "results" / p["key"] / f"candidate_{candidate + 1:02d}" / "train"
        path = path / f"{index:03d}.json"
        data = path.read_bytes()
        row = json.loads(data)
        identity = (row["checkpoint_key"], row["context"], row["task_id"], row["partition"], row["context_sha256"])
        expected = (p["key"], f"candidate_{candidate + 1:02d}", task, "train", p["context_sha256"][candidate])
        if identity != expected or row["status"] not in ("completed", "error"):
            raise ValueError(f"Result identity/status differs: {path}")
        score = float(row["score"])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid reward: {path}")
        return p["key"], candidate, task, score, row["status"] == "error", hashlib.sha256(data).hexdigest()

    jobs = [(p, c, i, task) for p in points for c in range(5) for i, task in enumerate(p["manifest_train_order"])]
    lookup = {p["key"]: p for p in points}
    matrices = {p["key"]: np.zeros((5, 50)) for p in points}
    errors = {p["key"]: np.zeros((5, 50), dtype=bool) for p in points}
    indices = {p["key"]: {task: i for i, task in enumerate(p["train_order"])} for p in points}
    digest = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        for key, candidate, task, score, error, result_digest in pool.map(read_result, jobs):
            index = indices[key][task]
            matrices[key][candidate, index] = score
            errors[key][candidate, index] = error
            digest.update(result_digest.encode())
    for key, p in lookup.items():
        matrix = matrices[key]
        if not np.allclose(matrix.mean(axis=1), p["scores"]["full_train"], atol=1e-12, rtol=0):
            raise ValueError(f"Full-train raw rewards differ from the original summary: {key}")
        for selector, pool_key in (("train10", "past_task_ids"),):
            frozen = next(plan[key] for plan in plans.values() if key in plan)
            columns = [indices[key][task] for task in frozen[pool_key]]
            if not np.allclose(matrix[:, columns].mean(axis=1), p["scores"][selector], atol=1e-12, rtol=0):
                raise ValueError(f"Original Train10 rewards differ: {key}")
        p["train_rewards"] = matrix.tolist()
        p["train_error_flags"] = errors[key].tolist()
        p.pop("manifest_train_order")
    return {"label": reference["label"], "source": str(source.resolve()), "source_sha256": reference["sha256"],
            "original_errors": reference["errors"], "original_evaluations": reference["evaluations"],
            "loaded_candidate_train_results": len(jobs), "result_digest_in_manifest_order": digest.hexdigest(),
            "points": points}


def fixed_cohort(models: list[dict], maximum_n: int) -> list[str]:
    identities = [[(p["key"], p["train_order"], p["checkpoint"]) for p in model["points"]] for model in models]
    if any(values != identities[0] for values in identities[1:]):
        raise ValueError("Models must share checkpoint identities and training task order")
    return [p["key"] for p in models[0]["points"] if len(p["past"]) >= maximum_n and len(p["future"]) >= maximum_n]


def sample_prefix_sums(matrix: np.ndarray, trials: int, maximum_n: int, rng: np.random.Generator) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != 5 or matrix.shape[1] < maximum_n:
        raise ValueError("Insufficient temporal pool for the maximum requested task count")
    order = np.argsort(rng.random((trials, matrix.shape[1])), axis=1)[:, :maximum_n]
    values = matrix[:, order].transpose(1, 2, 0)
    return np.concatenate((np.zeros((trials, 1, 5)), np.cumsum(values, axis=1)), axis=1)


def score_cases(current: np.ndarray, past: np.ndarray, future: np.ndarray, n: int) -> np.ndarray:
    """Same subsets for every candidate and aggregation; prefix subsets also pair the six cases."""
    scores = np.full((2, len(CASES), past.shape[0], 5), np.nan)
    for index, case in enumerate(CASES):
        counts = allocation(case, n)
        if counts is None:
            continue
        c, p, f = counts
        sums = [np.broadcast_to(current, (past.shape[0], 5))] if c else []
        means = list(sums)
        for count, values in ((p, past), (f, future)):
            if count:
                sums.append(values[:, count])
                means.append(values[:, count] / count)
        scores[0, index] = sum(sums) / sum(counts)
        scores[1, index] = sum(means) / len(means)
    return scores


def correlations(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=-1, keepdims=True)
    target = targets - targets.mean()
    numerator = np.einsum("...i,i->...", centered, target)
    denominator = np.sqrt((centered ** 2).sum(axis=-1) * (target ** 2).sum())
    defined = (np.ptp(scores, axis=-1) > 1e-12) & (np.ptp(targets) > 1e-12)
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=defined & (denominator > 0))
    return np.clip(result, -1, 1)


def selected_performance(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    valid = np.isfinite(scores).all(axis=-1)
    maxima = scores.max(axis=-1, keepdims=True)
    tied = np.isclose(scores, maxima, atol=1e-12, rtol=0)
    result = np.full(valid.shape, np.nan)
    np.divide(np.einsum("...i,i->...", tied, targets), tied.sum(axis=-1), out=result, where=valid)
    return result


def describe(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"mean": None, "std": None, "mc_standard_error": None, "q025": None, "q975": None,
                "defined_trials": 0}
    std = float(np.std(finite)) if np.ptp(finite) > 1e-14 else 0.0
    low, high = np.quantile(finite, [0.025, 0.975])
    return {"mean": float(finite.mean()), "std": std, "mc_standard_error": std / math.sqrt(finite.size),
            "q025": float(low), "q975": float(high), "defined_trials": int(finite.size)}


def calculate(model: dict, keys: list[str], trials: int, seed: int, maximum_n: int) -> tuple[dict, list[dict]]:
    points = [point for point in model["points"] if point["key"] in keys]
    shape = (maximum_n, 2, len(CASES), trials)
    selected = {target: np.zeros(shape) for target in TARGETS}
    rho_sums = {kind: {target: np.zeros(shape) for target in TARGETS} for kind in ("common", "individual")}
    rho_counts = {kind: {target: np.zeros(shape, dtype=np.int32) for target in TARGETS}
                  for kind in ("common", "individual")}
    checkpoint_rows = []
    for point in points:
        matrix = np.asarray(point["train_rewards"])
        t = point["checkpoint"]
        entropy = hashlib.sha256(f"{seed}:{point['key']}".encode()).digest()
        generators = np.random.SeedSequence(int.from_bytes(entropy[:16], "big")).spawn(2)
        past = sample_prefix_sums(matrix[:, :t], trials, maximum_n, np.random.default_rng(generators[0]))
        future = sample_prefix_sums(matrix[:, t + 1:], trials, maximum_n, np.random.default_rng(generators[1]))
        for n in range(1, maximum_n + 1):
            scores = score_cases(matrix[:, t], past, future, n)
            active = [index for index, case in enumerate(CASES) if allocation(case, n) is not None]
            for target in TARGETS:
                targets = np.asarray(point["scores"][target])
                rho = correlations(scores, targets)
                performance = selected_performance(scores, targets)
                selected[target][n - 1] += np.nan_to_num(performance)
                valid = np.isfinite(rho)
                # One identical CP cohort per trial for every active selector and both aggregation schemes.
                common = valid[:, active].all(axis=(0, 1))
                common = np.broadcast_to(common, valid.shape) & np.isfinite(scores).all(axis=-1)
                for kind, mask in (("common", common), ("individual", valid)):
                    rho_sums[kind][target][n - 1] += np.where(mask, rho, 0)
                    rho_counts[kind][target][n - 1] += mask
                for mode, weight in enumerate(WEIGHTS):
                    for index in active:
                        finite = rho[mode, index][valid[mode, index]]
                        mean_performance = float(performance[mode, index].mean())
                        parent = point["parent_scores"][target]
                        checkpoint_rows.append({"model": model["label"], "key": point["key"], "seed": point["seed"],
                            "checkpoint": t, "n": n, "case": CASES[index], "weight": weight, "target": target,
                            "selected_performance": mean_performance,
                            "proposal_mean": float(targets.mean()), "parent_performance": parent,
                            "selection_lift": mean_performance - float(targets.mean()),
                            "parent_gain": mean_performance - parent if parent is not None else None,
                            "individual_rho": float(finite.mean()) if finite.size else None,
                            "rho_defined_fraction": float(valid[mode, index].mean())})
    baselines = {}
    for target in TARGETS:
        proposal = float(np.mean([p["scores"][target] for p in points]))
        parents = [p["parent_scores"][target] for p in points]
        parent = float(np.mean(parents)) if all(value is not None for value in parents) else None
        baselines[target] = {"proposal_mean": proposal, "parent_performance": parent,
                             "proposal_mean_gain": proposal - parent if parent is not None else None}
    rows = []
    for n in range(1, maximum_n + 1):
        for mode, weight in enumerate(WEIGHTS):
            for case_index, case in enumerate(CASES):
                counts = allocation(case, n)
                if counts is None:
                    continue
                for target in TARGETS:
                    performance = selected[target][n - 1, mode, case_index] / len(points)
                    baseline = baselines[target]
                    row = {"model": model["label"], "n": n, "case": case, "weight": weight, "target": target,
                           "current_tasks": counts[0], "past_tasks": counts[1], "future_tasks": counts[2],
                           "actual_tasks": sum(counts), "checkpoints": len(points),
                           "selected_performance": describe(performance),
                           "selection_lift": describe(performance - baseline["proposal_mean"]),
                           "parent_gain": describe(performance - baseline["parent_performance"])
                           if baseline["parent_performance"] is not None else None}
                    for kind in ("common", "individual"):
                        total = rho_sums[kind][target][n - 1, mode, case_index]
                        count = rho_counts[kind][target][n - 1, mode, case_index]
                        values = np.full(trials, np.nan)
                        np.divide(total, count, out=values, where=count > 0)
                        row[f"rho_{kind}"] = describe(values)
                        row[f"rho_{kind}_checkpoints"] = describe(count)
                    rows.append(row)
    return {"label": model["label"], "baselines": baselines, "scores": rows,
            "checkpoint_keys": keys, "seeds": dict(Counter(p["seed"] for p in points)),
            "original_errors": model["original_errors"],
            "original_evaluations": model["original_evaluations"]}, checkpoint_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flat_scores(models: list[dict]) -> list[dict]:
    result = []
    for model in models:
        for row in model["scores"]:
            flat = {}
            for key, value in row.items():
                if isinstance(value, dict) or key == "parent_gain":
                    statistics = value if value is not None else dict.fromkeys(describe(np.array([])))
                    flat.update({f"{key}_{stat}": number for stat, number in statistics.items()})
                else:
                    flat[key] = value
            result.append(flat)
    return result


def write_report(summary: dict, output: Path) -> None:
    allocations = []
    for n in range(1, 11):
        for case in CASES:
            counts = allocation(case, n)
            allocations.append({"n": n, "case": case, "valid": counts is not None,
                                "current": counts[0] if counts else None,
                                "past": counts[1] if counts else None, "future": counts[2] if counts else None})
    write_csv(output / "task_allocations.csv", allocations)
    moments, coverage = [], []
    for model in summary["models"]:
        for row in model["scores"]:
            moments.append({key: row[key] for key in ("model", "n", "case", "weight", "target")}
                           | {"gaussian": gaussian_moments(row["rho_common"]["mean"])})
            if row["case"] == "current" and row["weight"] == "task_mean":
                coverage.append({"model": model["label"], "n": row["n"], "target": row["target"],
                                 "fixed_checkpoints": row["checkpoints"],
                                 "jointly_defined_cp_mean": row["rho_common_checkpoints"]["mean"],
                                 "jointly_defined_cp_std": row["rho_common_checkpoints"]["std"],
                                 "nonempty_trials": row["rho_common"]["defined_trials"],
                                 "total_trials": summary["trials"]})
    write_csv(output / "correlation_coverage.csv", coverage)
    (output / "bivariate_moments.json").write_text(json.dumps({"normality_tested": False,
        "contour_probability_mass": [0.5, 0.9], "models": moments}, indent=2, allow_nan=False) + "\n")
    lines = ["# Training-task timing and validation selection", "",
        "6개 시점 구성 x 두 점수 집계 방식 x n=1…10을 기존 BFCL 후보 평가로 비교했습니다.",
        "추가 모델 호출/환경 실행은 0회입니다. 모든 그림은 `figures/` 안의 PDF와 PNG입니다.", "",
        "## 설정", "",
        "- 현재: checkpoint parent에서 후보를 생성하는 데 사용한 다음 training task.",
        "- 과거: parent가 이미 학습한 task 전체. 미래: 현재 task 이후 training task 전체. "
        "Validation task는 선정에 사용하지 않습니다.",
        "- 시간 순서는 원래 Qwen/PREMiSE checkpoint 기준입니다. "
        "Gemma/OSS도 원래 alignment 실험의 Qwen context transfer 조건을 그대로 재분석합니다.",
        "- 과거/미래는 기존 frozen Train10에서 뽑는 것이 아니라 전체 시간 구간에서 균등 비복원 추출합니다.",
        "- 총 n개 중 현재는 1개. 현재+과거+미래는 (1, ceil((n-1)/2), floor((n-1)/2))개입니다.",
        "- Task mean: 선택 task 전체의 평균. Period mean: 각 시점 내 평균을 낸 뒤 존재하는 시점을 동일 가중합니다.",
        "- 같은 추출을 후보 5개, 두 집계 방식, 조건별 중첩 부분집합, n 및 세 모델에서 공유합니다.",
        f"- 무작위 추출 {summary['trials']:,}회, seed={summary['sampling_seed']}. "
        "동점은 균등 선택 기대값으로 적분합니다.",
        "- 모든 n/조건에 공통으로 가능한 36 CP를 고정했습니다: seed 42/43/44 각각 12개, CP10…38.",
        "- 각 CP에서 과거/미래가 각각 10개 이상 있어야 합니다. "
        "CP40 이후는 미래만 n=10을 구성할 수 없어 주 비교에서 제외합니다.",
        "- 현재만은 항상 1 task 기준선이며 모든 n에 반복 표시했습니다. 두 시점은 n≥2, 세 시점은 n≥3입니다.",
        "- Gain은 36 CP 전체를 사용합니다. Train gain은 부모 대비, "
        "validation lift는 해당 CP의 후보 5개 평균 대비입니다.",
        "- Validation 부모 점수는 없으므로 validation gain을 부모 대비 실제 개선량으로 해석하지 않습니다.",
        "- 그림의 validation μ_delta는 기존 사용자 지정 축 기호를 유지한 것입니다. "
        "실제 차감량은 proposal 평균 validation reward입니다.",
        "- 전체 task 수 50개 중 training selector task가 full-train target에 포함됩니다. "
        "Full validation 50개는 training과 disjoint입니다.", "",
        "## 그림", "",
        "- `figures/qwen_overview.pdf`: Qwen validation rho 및 선택 이득; 왼쪽 task mean, 오른쪽 period mean.",
        "- `figures/gemma_overview.pdf`, `figures/oss_overview.pdf`: 같은 구성의 모델별 결과.",
        "- `figures/correlation_full_validation.pdf`, `figures/gain_full_validation.pdf`: 세 모델의 validation 비교.",
        "- `figures/correlation_full_train.pdf`, `figures/gain_full_train.pdf`: full train에 대한 보조 분석.",
        "- `figures/correlation_individual_full_validation.pdf` 및 train 버전: "
        "각 조건에서 개별적으로 유효한 CP를 이용한 rho 보조 분석.",
        "- `figures/{qwen,gemma,oss}_bivariate_{full_train,full_validation}.pdf`: "
        "시점 구성 6행 x n=1…10 열, 두 집계 방식의 등고선.",
        "- `figures/bivariate_overview_full_validation_n10.pdf` 및 train 버전: n=10의 3모델 x6조건 비교.", "",
        "## n=10: validation 선택 이득 (pp)", "",
        "Current-only는 실제 1-task 기준선입니다. 다른 조건은 총 10개 task입니다.", "",
        "| Model | Case | Task mean | Period mean | Task-mean rho | Period-mean rho |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in summary["models"]:
        for case, label in zip(CASES, LABELS, strict=True):
            rows = [next(row for row in model["scores"] if row["n"] == 10 and row["case"] == case
                         and row["weight"] == weight and row["target"] == "full_validation") for weight in WEIGHTS]
            rhos = [f"{row['rho_common']['mean']:+.3f}" if row["rho_common"]["mean"] is not None else "N/A"
                    for row in rows]
            lines.append(f"| {model['label']} | {label} | {100 * rows[0]['selection_lift']['mean']:+.4f} | "
                         f"{100 * rows[1]['selection_lift']['mean']:+.4f} | {rhos[0]} | {rhos[1]} |")
    lines.extend(["",
        "n=10에서 Qwen과 Gemma는 두 집계 방식 모두 현재+과거의 validation lift가 가장 높습니다.",
        "Qwen의 현재+과거는 Task mean +0.2432pp, Period mean +0.2564pp입니다. "
        "미래만 사용하면 -0.2729pp이며, 현재+과거+미래는 +0.0114/-0.0168pp입니다.",
        "Gemma의 현재+과거는 +0.4177/+0.4329pp입니다. "
        "OSS는 모든 n=10 조건의 lift가 음수이며 원래 실행 오류율이 높은 조건입니다.",
        "이 숫자는 관측된 선택 이득이며, 집계 방식 간 우열에 대한 통계적 유의성 주장은 아닙니다."])
    lines.extend(["", "## Correlation의 유효 CP와 해석", "",
        "주 rho와 등고선은 각 추출에서 모든 활성 조건 및 두 집계의 rho가 함께 정의되는 동일 CP들만 사용합니다.",
        "상수 점수의 rho는 0이 아니라 미정의입니다. "
        "하나의 현재 task에서 후보 점수가 모두 같은 CP가 많아 유효 CP가 감소합니다.",
        "각 추출의 공동 유효 CP 평균 수와 비어 있지 않은 추출 수는 다음과 같습니다 (validation).", "",
        "| Model | n | Mean jointly valid CP | Nonempty trials |", "|---|---:|---:|---:|"])
    for row in coverage:
        if row["target"] == "full_validation" and row["n"] in (1, 3, 10):
            lines.append(f"| {row['model']} | {row['n']} | {row['jointly_defined_cp_mean']:.4f}/36 | "
                         f"{row['nonempty_trials']}/{row['total_trials']} |")
    lines.extend(["",
        "현재-only의 score와 gain은 n에 따라 변하지 않습니다. "
        "주 rho는 공동 유효 CP 구성이 n/추출에 따라 달라져 소폭 변할 수 있습니다.",
        "모든 36 CP를 대상으로 각 조건의 개별 유효 rho를 계산한 보조 그림/수치도 저장했습니다. "
        "이 경우 조건별 rho의 유효 CP는 다릅니다.",
        "따라서 validation 성능 비교의 우선 지표는 모든 36 CP를 유지하는 selection lift입니다.", "",
        "## 불확실성 및 등고선", "",
        "음영은 고정 CP에서 task 추출을 반복한 평균 ±1 population SD이며 seed/CP 불확실성의 신뢰구간이 아닙니다.",
        "Gain 중심선은 전체 부분집합의 정확 적분이 아니라 Monte Carlo 평균입니다. "
        "각 draw 내 동점만 정확하게 적분합니다.",
        "Monte Carlo 표준오차와 2.5/97.5 percentile도 scores.csv/JSON에 저장했습니다. "
        "새로운 유의성 검정은 수행하지 않았습니다.",
        "등고선은 CP별 표준화된 pre-selection 후보의 rho를 갖는 이변량 정규근사입니다. "
        "실제 분포의 정규성을 주장하지 않습니다.",
        "실선+채움은 50%, 점선은 90% Gaussian probability mass이며 SD나 CI band가 아닙니다.",
        "등고선의 T/P는 Task mean/Period mean이고, 각 패널에 해당 rho를 표기합니다.", "",
        "## 재현 및 오류", "",
        "모델별 원래 평가 오류 Qwen 32/27,500, Gemma 2/27,500, OSS 7,456/27,500의 0점 처리를 유지했습니다.",
        "특히 OSS의 경향은 높은 실행 실패율과 함께 해석해야 합니다. "
        "이 분석은 원래 20-step 평가를 재사용하며 새 actor를 실행하지 않습니다.",
        "이전 그래프와 달리 36 CP로 고정하고 전체 과거 구간을 사용하므로 "
        "기존 frozen Train10/50-CP 결과와 숫자가 다릅니다.", "",
        "- `temporal_candidate_rewards.json`: 검증한 전체 train 점수 행렬, 시간순 task ID, 오류 flag, 출처 digest.",
        "- `temporal_selection_summary.json`, `scores.csv`: 전체 모델/조건/n/집계/target 수치와 방법.",
        "- `checkpoint_scores.csv`: CP별 성능, proposal 기준선 및 개별 유효 rho.",
        "- `correlation_coverage.csv`: 공동 유효 rho의 CP 수 및 반복 수.",
        "- `task_allocations.csv`: n별 6개 조건의 현재/과거/미래 수.",
        "- `bivariate_moments.json`: 등고선의 평균·공분산·rho.", "",
        "```bash",
        ".venv/bin/python scripts/premise/analyze_bfcl_temporal_selection.py \\",
        "  --reference outputs_arr_oct/bfcl_alignment_sample_size_n1_10_20261013/sample_size_summary.json \\",
        "  --archive outputs_arr_oct \\",
        "  --output outputs_arr_oct/bfcl_alignment_sample_size_n1_10_20261013/temporal_selection_20261021",
        "```", "",
        "저장된 통계만으로 figure를 다시 생성하려면 위 명령에 --replot을 추가합니다.", ""])
    (output / "README.md").write_text("\n".join(lines))


def plot_lines(axis, model: dict, target: str, weight: str, metric: str) -> None:
    for index, case in enumerate(CASES):
        rows = [row for row in model["scores"] if row["target"] == target and row["weight"] == weight
                and row["case"] == case]
        x = np.asarray([row["n"] for row in rows])
        scale = 1 if metric.startswith("rho") else 100
        y = np.asarray([row[metric]["mean"] if row[metric]["mean"] is not None else np.nan for row in rows]) * scale
        std = np.asarray([row[metric]["std"] if row[metric]["std"] is not None else np.nan for row in rows]) * scale
        axis.fill_between(x, y - std, y + std, facecolor=FILLS[index], edgecolor="none", zorder=1)
        axis.plot(x, y, color=COLORS[index], label=LABELS[index], linewidth=1.15, marker="o", markersize=2,
                  linestyle="--" if case == "current" else "-", zorder=3)
    if target == "full_validation" and metric == "selection_lift" and model.get("validation_reference"):
        rows = model["validation_reference"]
        x = np.asarray([row["n"] for row in rows])
        y = np.asarray([row["selection_lift"]["mean"] for row in rows]) * 100
        std = np.asarray([row["selection_lift"]["std"] for row in rows]) * 100
        axis.fill_between(x, y - std, y + std, facecolor=VALIDATION_FILL, edgecolor="none", zorder=1)
        axis.plot(x, y, color=VALIDATION_COLOR, linestyle="--", linewidth=1.9, marker="D",
                  markersize=2.8, markerfacecolor="white", markeredgewidth=0.8,
                  label="Validation", zorder=4)
    axis.axhline(0, color="#AAAAAA", linewidth=0.55, zorder=2)
    if metric == "parent_gain":
        axis.axhline(model["baselines"][target]["proposal_mean_gain"] * 100,
                     color="#999999", linewidth=0.7, linestyle=":", zorder=2)
    axis.set(xlim=(0.8, 10.2), xticks=[1, 3, 5, 7, 10])
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.4)
    axis.tick_params(labelsize=7, length=2.2, width=0.6, pad=2)


def plot_contour(axis, model: dict, target: str, case: str, n: int) -> None:
    for index, weight in enumerate(WEIGHTS):
        row = next((r for r in model["scores"] if r["target"] == target and r["weight"] == weight
                    and r["case"] == case and r["n"] == n), None)
        rho = row["rho_common"]["mean"] if row else None
        label = rf"$\rho_{{{'T' if index == 0 else 'P'}}}={rho:+.3f}$" if rho is not None else r"$\rho=\mathrm{N/A}$"
        axis.text(0.03, 0.97 - index * 0.18, label, transform=axis.transAxes, color=MODE_COLORS[index],
                  fontsize=5.8, va="top", zorder=5,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.1})
        if rho is None or abs(rho) >= 1:
            continue
        covariance = np.array([[1, rho], [rho, 1]])
        inner, outer = (ellipse_points(covariance, mass, count=121) for mass in (0.5, 0.9))
        axis.fill(inner[:, 0], inner[:, 1], color=MODE_FILLS[index], linewidth=0, zorder=1)
        axis.plot(inner[:, 0], inner[:, 1], color=MODE_COLORS[index], linewidth=0.85, zorder=3)
        axis.plot(outer[:, 0], outer[:, 1], color=MODE_COLORS[index], linewidth=0.65, linestyle="--", zorder=3)
    axis.axhline(0, color="#DDDDDD", linewidth=0.4, zorder=2)
    axis.axvline(0, color="#DDDDDD", linewidth=0.4, zorder=2)
    axis.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xticks=[-2, 0, 2], yticks=[-2, 0, 2])
    axis.set_aspect("equal")
    axis.tick_params(labelsize=5.7, length=1.8, pad=1.4)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_color("#B5B5B5")


def plot(models: list[dict], output: Path, validation_gain_only: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.5,
                         "mathtext.fontset": "dejavuserif", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42})

    def save(figure, stem):
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=400, bbox_inches="tight", pad_inches=0.04)
        plt.close(figure)
        print(f"Saved {stem}.pdf", flush=True)

    handles = [Line2D([0], [0], color=COLORS[i], linewidth=1.3,
                      linestyle="--" if i == 0 else "-", label=LABELS[i]) for i in range(len(CASES))]
    validation_handles = handles
    if any(model.get("validation_reference") for model in models):
        validation_handles = [*handles, Line2D([0], [0], color=VALIDATION_COLOR, linewidth=1.9,
                                              linestyle="--", marker="D", markersize=2.8,
                                              markerfacecolor="white", label="Validation")]
    for target in TARGETS:
        for stem, metric, ylabel in (
            ("correlation", "rho_common", r"$\rho$"),
            ("correlation_individual", "rho_individual", r"$\rho$"),
            ("gain", "parent_gain" if target == "full_train" else "selection_lift",
             r"$\delta_t^{i^\star}$ (pp)" if target == "full_train" else r"$s_t^{i^\star}-\mu_\delta$ (pp)"),
        ):
            if validation_gain_only and (target != "full_validation" or stem != "gain"):
                continue
            figure, axes = plt.subplots(2, len(models), figsize=(2.55 * len(models), 4.25),
                                        sharex=True, sharey=True, squeeze=False)
            figure.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.12, hspace=0.22, wspace=0.14)
            for column, model in enumerate(models):
                for mode, weight in enumerate(WEIGHTS):
                    plot_lines(axes[mode, column], model, target, weight, metric)
                    if mode == 0:
                        axes[mode, column].set_title(model["label"], fontsize=8.5, pad=4)
                    if column == 0:
                        axes[mode, column].set_ylabel(WEIGHT_LABELS[mode] + "\n" + ylabel, fontsize=8)
            legend_handles = validation_handles if target == "full_validation" and stem == "gain" else handles
            figure.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                          frameon=False, fontsize=7, handlelength=1.5, columnspacing=1.3)
            figure.supxlabel(r"Number of evaluation tasks ($n$)", fontsize=8, y=0.015)
            save(figure, f"{stem}_{target}")

    for model in models:
        slug = model["label"].lower()
        figure, axes = plt.subplots(2, 2, figsize=(6.7, 3.9), sharex=True, sharey="row")
        figure.subplots_adjust(left=0.12, right=0.99, top=0.79, bottom=0.14, hspace=0.28, wspace=0.16)
        for column, weight in enumerate(WEIGHTS):
            for row, metric in enumerate(("rho_common", "selection_lift")):
                plot_lines(axes[row, column], model, "full_validation", weight, metric)
                if row == 0:
                    axes[row, column].set_title(WEIGHT_LABELS[column], fontsize=8.5, pad=4)
                if column == 0:
                    axes[row, column].set_ylabel(r"$\rho$" if row == 0 else r"$s_t^{i^\star}-\mu_\delta$ (pp)",
                                                fontsize=8)
        figure.legend(handles=validation_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                      frameon=False, fontsize=7, handlelength=1.5, columnspacing=1.3)
        figure.supxlabel(r"Number of evaluation tasks ($n$)", fontsize=8, y=0.015)
        save(figure, f"{slug}_overview")

    if validation_gain_only:
        return

    contour_handles = [Line2D([0], [0], color=color, linewidth=1.2, label=label)
                       for color, label in zip(MODE_COLORS, WEIGHT_LABELS, strict=True)]
    xlabels = {"full_train": r"$z(\delta_t^i)$", "full_validation": r"$z(s_{t,\mathrm{val50}}^i)$"}
    for target in TARGETS:
        for model in models:
            figure, axes = plt.subplots(6, 10, figsize=(13.6, 8.9), sharex=True, sharey=True)
            figure.subplots_adjust(left=0.09, right=0.995, top=0.94, bottom=0.055, hspace=0.20, wspace=0.16)
            for row, case in enumerate(CASES):
                for column, n in enumerate(range(1, 11)):
                    plot_contour(axes[row, column], model, target, case, n)
                    if row == 0:
                        axes[row, column].set_title(rf"$n={n}$", fontsize=7.5, pad=3)
                    if column == 0:
                        label = LABELS[row].replace(" + ", " +\n")
                        axes[row, column].set_ylabel(label, fontsize=7, labelpad=5)
            figure.legend(handles=contour_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                          ncol=2, frameon=False, fontsize=8)
            figure.supxlabel(xlabels[target], fontsize=9, y=0.01)
            figure.supylabel(r"$z(s_{t,n}^i)$", fontsize=9, x=0.005)
            save(figure, f"{model['label'].lower()}_bivariate_{target}")
        figure, axes = plt.subplots(len(models), 6, figsize=(10.1, 1.7 * len(models) + 0.65),
                                    sharex=True, sharey=True, squeeze=False)
        figure.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.10, hspace=0.24, wspace=0.15)
        for row, model in enumerate(models):
            for column, case in enumerate(CASES):
                plot_contour(axes[row, column], model, target, case, 10)
                if row == 0:
                    axes[row, column].set_title(LABELS[column].replace(" + ", " +\n"), fontsize=7, pad=4)
                if column == 0:
                    axes[row, column].set_ylabel(model["label"], fontsize=8, labelpad=5)
        figure.legend(handles=contour_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                      ncol=2, frameon=False, fontsize=8)
        figure.supxlabel(xlabels[target], fontsize=9, y=0.015)
        figure.supylabel(r"$z(s_{t,n}^i)$", fontsize=9, x=0.005)
        save(figure, f"bivariate_overview_{target}_n10")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=ROOT / "outputs_arr_oct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--io-workers", type=int, default=16)
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "temporal_selection_summary.json"
    if args.replot:
        plot(json.loads(result_path.read_text())["models"], args.output / "figures")
        return
    if args.trials < 1 or args.io_workers < 1:
        parser.error("Trials and I/O workers must be positive")
    raw = args.reference.read_bytes()
    reference = json.loads(raw)
    references = sorted(reference["models"], key=lambda model: ("Qwen", "Gemma", "OSS").index(model["label"]))
    raw_path = args.output / "temporal_candidate_rewards.json"
    if raw_path.exists():
        payload = json.loads(raw_path.read_text())
        if payload["reference_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("Cached raw matrix uses a different source summary")
        models = payload["models"]
    else:
        models = []
        for model in references:
            print(f"Loading all 50 training rewards for {model['label']} candidates", flush=True)
            models.append(load_model(model, args.archive.resolve(), args.io_workers))
            print(f"Verified {model['label']}: {models[-1]['loaded_candidate_train_results']} rewards", flush=True)
        raw_path.write_text(json.dumps({"reference_sha256": hashlib.sha256(raw).hexdigest(), "models": models},
                                       separators=(",", ":")) + "\n")
    keys = fixed_cohort(models, 10)
    if not keys:
        raise ValueError("No common checkpoints support n=10 for every selector")
    results, checkpoint_rows = [], []
    for model in models:
        print(f"Computing {model['label']}: {len(keys)} fixed checkpoints, {args.trials} paired draws", flush=True)
        result, rows = calculate(model, keys, args.trials, args.seed, 10)
        results.append(result)
        checkpoint_rows.extend(rows)
    summary = {"reference": str(args.reference.resolve()), "reference_sha256": hashlib.sha256(raw).hexdigest(),
               "sampling_seed": args.seed, "trials": args.trials, "maximum_n": 10, "new_model_calls": 0,
               "checkpoint_origin": "Original Qwen PREMiSE checkpoints and training order. Gemma/OSS retain "
               "the original alignment experiment's Qwen context-transfer condition.",
               "raw_reward_cache": str(raw_path.resolve()), "raw_reward_cache_sha256": hashlib.sha256(
                   raw_path.read_bytes()).hexdigest(), "fixed_checkpoint_keys": keys,
               "sampling": "Uniform without-replacement draws from the entire chronological past/future training "
               "pools, not from the old frozen Train10. Current is the proposal task. Stable random permutations "
               "pair task subsets across candidates, cases, weighting schemes, n, and models.",
               "current_baseline": "Current-only uses one task at every displayed n; it is a one-task baseline, "
               "not an n-task condition. Two-period mixtures start at n=2 and three-period mixtures at n=3.",
               "cohort": "Fixed across all n and all cases: at least 10 past and 10 future tasks. Each figure "
               "uses these same checkpoints. Gain keeps every CP; common rho additionally requires every "
               "active case and both weightings to be nonconstant within that trial and target.",
               "aggregation": {"task_mean": "All selected tasks have equal weight", "period_mean": "Mean "
               "within each present period, then equal weight across present periods (1:1 or 1:1:1)"},
               "correlation": "Mean within-checkpoint Pearson rho over the same jointly defined CPs per trial, "
               "then mean over nonempty trials. Individually defined correlations are retained separately. "
               "Undefined rho is never replaced with zero. CP eligibility can vary by n and draw for rho.",
               "gain": "Monte Carlo average with exact uniform tie-breaking expectation per draw (atol=1e-12). "
               "Full train uses the parent baseline; full validation uses the candidate proposal mean. "
               "No validation parent evaluation exists.",
               "uncertainty": "Mean +/- one population SD across task-subset draws with contexts fixed. "
               "Not confidence intervals over checkpoints or seeds. Monte Carlo standard errors are saved.",
               "bivariate": "Moment-matched Gaussian with mean (0,0), covariance [[1,rho],[rho,1]], using "
               "exactly the common-rho curve. Within-CP standardized pre-selection candidates; no normality "
               "claim. Solid/fill=50% probability mass; dashed=90%; T/P denote task/period weighting.",
               "targets": "Full50 training and Full50 validation. All selectors use training tasks only. "
               "Full-train target overlaps selector tasks; validation task IDs are disjoint.",
               "original_errors": "Existing zero-scored errors retained; no new inference or retries. OSS "
               "has 7456/27500 original errors, so its patterns also reflect execution failures.",
               "models": results}
    result_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "scores.csv", flat_scores(results))
    write_csv(args.output / "checkpoint_scores.csv", checkpoint_rows)
    write_report(summary, args.output)
    plot(results, args.output / "figures")
    print(f"Saved temporal selection analysis to {args.output}", flush=True)


if __name__ == "__main__":
    main()
