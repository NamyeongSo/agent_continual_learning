# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Add validation-selected performance to the existing temporal-selector gain figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_temporal_selection import (  # noqa: E402
    describe,
    plot,
    relocated,
    sample_prefix_sums,
    selected_performance,
    write_csv,
)


def load_validation_rewards(raw_model: dict, keys: list[str], reference: dict, archive: Path, workers: int) -> dict:
    points = [point for point in raw_model["points"] if point["key"] in keys]
    if len(points) != len(keys) or len({point["key"] for point in points}) != len(keys):
        raise ValueError("Validation reference must use the identical fixed checkpoint cohort")
    manifests = {}
    for old, expected_digest in reference["source_manifests"].items():
        source = relocated(old, archive).resolve()
        raw = (source / "launch_manifest.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_digest:
            raise ValueError(f"Original manifest digest differs: {source}")
        manifests[str(source)] = {point["key"]: point for point in json.loads(raw)["checkpoints"]}
    frozen = {}
    for point in points:
        plan = manifests[str(Path(point["source"]).resolve())][point["key"]]
        tasks = plan["validation_full_task_ids"]
        if len(tasks) != 50 or len(set(tasks)) != 50 or set(tasks) & set(point["train_order"]):
            raise ValueError(f"Expected 50 distinct held-out tasks: {point['key']}")
        frozen[point["key"]] = plan

    def read(job):
        point, candidate, index = job
        task = frozen[point["key"]]["validation_full_task_ids"][index]
        path = Path(point["source"]) / "results" / point["key"] / f"candidate_{candidate + 1:02d}" / "val"
        raw = (path / f"{index:03d}.json").read_bytes()
        row = json.loads(raw)
        identity = (row["checkpoint_key"], row["context"], row["partition"], row["task_id"], row["context_sha256"])
        expected = (point["key"], f"candidate_{candidate + 1:02d}", "val", task, point["context_sha256"][candidate])
        score = float(row["score"])
        if identity != expected or row["status"] not in ("completed", "error") or not np.isfinite(score):
            raise ValueError(f"Invalid result identity/status: {path}")
        if not 0 <= score <= 1:
            raise ValueError(f"Invalid reward: {path}")
        return point["key"], candidate, index, score, row["status"] == "error", hashlib.sha256(raw).hexdigest()

    matrices = {point["key"]: np.zeros((5, 50)) for point in points}
    errors = {point["key"]: np.zeros((5, 50), dtype=bool) for point in points}
    jobs = [(point, candidate, index) for point in points for candidate in range(5) for index in range(50)]
    digest = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, candidate, index, score, error, result_digest in pool.map(read, jobs):
            matrices[key][candidate, index] = score
            errors[key][candidate, index] = error
            digest.update(result_digest.encode())
    output = []
    for point in points:
        matrix = matrices[point["key"]]
        tasks = frozen[point["key"]]["validation_full_task_ids"]
        if not np.allclose(matrix.mean(axis=1), point["scores"]["full_validation"], atol=1e-12, rtol=0):
            raise ValueError(f"Full validation rewards differ from the original: {point['key']}")
        indices = [tasks.index(task) for task in frozen[point["key"]]["validation_task_ids"]]
        if not np.allclose(matrix[:, indices].mean(axis=1), point["scores"]["val10"], atol=1e-12, rtol=0):
            raise ValueError(f"Original Val10 rewards differ: {point['key']}")
        output.append({"key": point["key"], "seed": point["seed"], "validation_task_ids": tasks,
                       "candidate_rewards": matrix.tolist(), "error_flags": errors[point["key"]].tolist()})
    return {"label": raw_model["label"], "points": output, "loaded_results": len(jobs),
            "results_digest_in_manifest_order": digest.hexdigest(), "source_sha256": raw_model["source_sha256"]}


def validation_reference(model: dict, keys: list[str], trials: int, seed: int, maximum_n: int) -> tuple[list, list]:
    if trials < 1 or maximum_n < 1:
        raise ValueError("Trials and maximum sample size must be positive")
    points = [point for point in model["points"] if point["key"] in keys]
    if len(points) != len(keys) or len({point["key"] for point in points}) != len(keys):
        raise ValueError("Validation reference checkpoint cohort differs")
    selected_sum = np.zeros((maximum_n, trials))
    proposal_sum = 0.0
    checkpoint_rows = []
    for point in points:
        matrix = np.asarray(point["candidate_rewards"], dtype=float)
        if matrix.shape != (5, 50) or len(set(point["validation_task_ids"])) != 50:
            raise ValueError("Expected a five-candidate, 50-task validation matrix")
        # Canonical task ordering makes draws independent of the manifest's old Val10-first ordering.
        order = np.argsort(point["validation_task_ids"])
        matrix = matrix[:, order]
        targets = matrix.mean(axis=1)
        proposal = float(targets.mean())
        proposal_sum += proposal
        digest = hashlib.sha256(f"{seed}:validation-reference:{point['key']}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:16], "big"))
        sums = sample_prefix_sums(matrix, trials, maximum_n, rng)
        for n in range(1, maximum_n + 1):
            # A single validation subset scores all five candidates; integrate uniform argmax ties.
            performance = selected_performance(sums[:, n] / n, targets)
            selected_sum[n - 1] += performance
            checkpoint_rows.append({"model": model["label"], "key": point["key"], "seed": point["seed"],
                                    "n": n, "candidate_count": 5, "validation_pool_size": 50,
                                    "selected_performance": float(performance.mean()), "proposal_mean": proposal,
                                    "selection_lift": float(performance.mean()) - proposal})
    proposal = proposal_sum / len(points)
    rows = []
    for n in range(1, maximum_n + 1):
        performance = selected_sum[n - 1] / len(points)
        rows.append({"model": model["label"], "n": n, "checkpoints": len(points), "validation_pool_size": 50,
                     "candidate_count": 5, "proposal_mean": proposal,
                     "selected_performance": describe(performance),
                     "selection_lift": describe(performance - proposal)})
    return rows, checkpoint_rows


def update(output: Path, archive: Path, workers: int, skip_plot: bool = False) -> None:
    path = output / "temporal_selection_summary.json"
    summary = json.loads(path.read_text())
    reference_path = Path(summary["reference"])
    reference_raw = reference_path.read_bytes()
    if hashlib.sha256(reference_raw).hexdigest() != summary["reference_sha256"]:
        raise ValueError("Original sample-size reference digest differs")
    reference = json.loads(reference_raw)
    references = {model["label"]: model for model in reference["models"]}
    raw_path = Path(summary["raw_reward_cache"])
    raw = raw_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != summary["raw_reward_cache_sha256"]:
        raise ValueError("Original temporal rewards digest differs")
    raw_models = json.loads(raw)["models"]
    keys = summary["fixed_checkpoint_keys"]
    cache_path = output / "validation_candidate_rewards.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache["reference_sha256"] != summary["reference_sha256"] or cache["fixed_checkpoint_keys"] != keys:
            raise ValueError("Cached validation reference provenance/cohort differs")
        models = cache["models"]
    else:
        models = []
        for model in raw_models:
            print(f"Loading full-validation candidate rewards for {model['label']}", flush=True)
            models.append(load_validation_rewards(model, keys, references[model["label"]], archive, workers))
            print(f"Verified {model['label']}: {models[-1]['loaded_results']} saved validation results", flush=True)
        cache_path.write_text(json.dumps({"reference_sha256": summary["reference_sha256"],
            "fixed_checkpoint_keys": keys, "models": models}, separators=(",", ":")) + "\n")
    by_label = {model["label"]: model for model in models}
    checkpoint_rows, csv_rows = [], []
    for model in summary["models"]:
        rows, cp_rows = validation_reference(by_label[model["label"]], keys, summary["trials"],
                                             summary["sampling_seed"], summary["maximum_n"])
        if not np.isclose(rows[0]["proposal_mean"], model["baselines"]["full_validation"]["proposal_mean"],
                          atol=1e-12, rtol=0):
            raise ValueError("Validation reference uses a different proposal baseline")
        model["validation_reference"] = rows
        checkpoint_rows.extend(cp_rows)
        for row in rows:
            flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
            for metric in ("selected_performance", "selection_lift"):
                flat.update({f"{metric}_{stat}": value for stat, value in row[metric].items()})
            csv_rows.append(flat)
        print(f"{model['label']} n=10 validation-selected lift: "
              f"{100 * rows[-1]['selection_lift']['mean']:+.6f}pp", flush=True)
    summary["validation_selection_reference"] = {
        "method": "Uniform n-of-50 validation task sampling without replacement. The same subset scores "
        "every candidate. One period means task-mean and period-mean weighting are identical. Exact uniform "
        "tie-breaking per draw; Monte Carlo mean over the same fixed checkpoints and trial count "
        "as temporal selectors.",
        "target": "Full validation 50; selection tasks are included in this reference target. This is a "
        "within-validation reference, not performance on a second unseen evaluation set.",
        "pool": "Full 50-task validation pool, not the old fixed Val10 pool",
        "raw_cache": str(cache_path.resolve()),
        "raw_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "new_model_calls": 0, "display": "Rose dashed line with diamond markers, gain panels only; "
        "same curve in task-mean and period-mean panels. Existing temporal correlations/cohorts are unchanged.",
    }
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(output / "validation_reference_scores.csv", csv_rows)
    write_csv(output / "validation_reference_checkpoint_scores.csv", checkpoint_rows)
    notes = ["# Validation-selected reference for temporal selection", "",
        "Validation gain 패널에 Validation 점선과 다이아몬드 marker를 추가했습니다.",
        "전체 validation 50개에서 n=1…10개를 균등 비복원 추출해 같은 후보 5개를 점수화하고 선택합니다.",
        "기존 여섯 training 조건과 같은 36 CP (각 seed 12개), "
        "같은 proposal 평균 기준선, 같은 10,000회 반복입니다.",
        "Validation은 시점 구분이 하나라 Task mean과 Period mean이 동일합니다. 두 패널에 같은 곡선을 표시합니다.",
        "선정용 validation task가 full validation 50개에 포함되는 비교 기준선입니다. "
        "별도의 미관측 평가 집합에서 일반화 성능을 측정한 결과로 해석하지 않습니다.",
        "기존 temporal score, correlation, cohort 및 bivariate 수치는 그대로 유지했습니다. "
        "추가 모델 호출은 없습니다.", "",
        "| Model | Validation-selected lift at n=10 (pp) |", "|---|---:|"]
    for model in summary["models"]:
        lift = model["validation_reference"][-1]["selection_lift"]["mean"] * 100
        notes.append(f"| {model['label']} | {lift:+.4f} |")
    notes.extend(["", "수치: validation_reference_scores.csv, CP별 수치: validation_reference_checkpoint_scores.csv.",
        "원시 행렬 및 출처: validation_candidate_rewards.json, temporal_selection_summary.json.",
        "음영은 task 추출에 따른 평균 ±1 SD로 기존 gain 그림과 같습니다. 신뢰구간이 아닙니다.", "",
        "```bash", ".venv/bin/python scripts/premise/add_bfcl_temporal_validation_reference.py \\",
        "  --output outputs_arr_oct/bfcl_alignment_sample_size_n1_10_20261013/temporal_selection_20261021",
        "```", ""])
    (output / "validation_reference_README.md").write_text("\n".join(notes))
    readme_path = output / "README.md"
    readme = readme_path.read_text()
    if "validation_reference_README.md" not in readme:
        readme += "\n## Validation-selected reference\n\n"
        readme += "Validation gain 패널에 validation n개로 선택한 기준선을 추가했습니다. "
        readme += "방법/수치는 [validation_reference_README.md](validation_reference_README.md)를 참고하세요.\n"
        readme_path.write_text(readme)
    if not skip_plot:
        plot(summary["models"], output / "figures", validation_gain_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=ROOT / "outputs_arr_oct")
    parser.add_argument("--io-workers", type=int, default=16)
    args = parser.parse_args()
    if args.io_workers < 1:
        parser.error("I/O workers must be positive")
    update(args.output.resolve(), args.archive.resolve(), args.io_workers)


if __name__ == "__main__":
    main()
