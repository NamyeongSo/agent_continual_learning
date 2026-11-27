# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Validate percentage encounter imports without aligning scores to another task schedule."""

import csv
import hashlib

import pytest

from scripts.utils.plot_sequential_online_performance import load_skillopt_runs


def write_encounters(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["mode", "seed", "iteration", "benchmark", "task_id", "score", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)


def encounter_rows(mode="interleaved"):
    return [
        {
            "mode": mode,
            "seed": 43,
            "iteration": i,
            "benchmark": benchmark,
            "task_id": task,
            "score": score,
            "source": f"step_{i}/rollout/results.json",
        }
        for i, (benchmark, task, score) in enumerate(
            [("bfcl", "a", "50"), ("appworld", "b", "75"), ("terminalbench2", "c", "100")], 1
        )
    ]


def reference(order=None):
    return [
        {"method": "ace", "seed": 43, "order": order or [("bfcl", "a"), ("appworld", "b"), ("terminalbench2", "c")]}
    ]


def test_csv_percentage_scale_tgc_and_native_schedule_provenance(tmp_path):
    path = tmp_path / "encounters.csv"
    rows = encounter_rows()
    rows.append({**rows[0], "mode": "sequential", "score": "0"})
    rows.append({**rows[0], "seed": 42, "score": "0"})
    write_encounters(path, list(reversed(rows)))
    refs = reference([("terminalbench2", "c"), ("bfcl", "a"), ("appworld", "b")])
    run = load_skillopt_runs(path, refs, "tgc", mode="interleaved")[0]
    assert run["seed"] == 43 and run["cursor"] == 3
    assert run["order"] == [("bfcl", "a"), ("appworld", "b"), ("terminalbench2", "c")]
    assert run["raw_scores"] == {1: 0.5, 2: 0.75, 3: 1}
    assert run["scores"] == {1: 0.5, 2: 0, 3: 1}
    assert not run["task_order_matches_ace"] and run["task_set_matches_ace"]
    assert run["task_position_mismatches"] == run["benchmark_position_mismatches"] == 3
    assert run["encounter_sources"][2] == "step_2/rollout/results.json"
    assert run["csv_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not run["error_status_available"]
    assert load_skillopt_runs(path, refs, "partial", mode="interleaved")[0]["scores"][2] == 0.75


def test_csv_missing_scores_preserve_iteration_and_matching_sequential_boundaries(tmp_path):
    path = tmp_path / "encounters.csv"
    rows = encounter_rows("sequential")
    rows[1]["score"] = ""
    write_encounters(path, rows)
    run = load_skillopt_runs(path, reference(), "tgc", mode="sequential")[0]
    assert run["unscored_iterations"] == [2] and 2 not in run["scores"]
    assert run["task_order_matches_ace"]
    with pytest.raises(ValueError, match="boundaries"):
        load_skillopt_runs(
            path, reference([("appworld", "b"), ("bfcl", "a"), ("terminalbench2", "c")]), "tgc", mode="sequential"
        )


@pytest.mark.parametrize("score", ["-1", "101", "nan", "inf"])
def test_csv_rejects_invalid_percentage_scores(tmp_path, score):
    path = tmp_path / "encounters.csv"
    rows = encounter_rows()
    rows[0]["score"] = score
    write_encounters(path, rows)
    with pytest.raises(ValueError, match="percentage"):
        load_skillopt_runs(path, reference(), "tgc", mode="interleaved")


@pytest.mark.parametrize("iterations", [[1, 1, 3], [1, 3, 4], [2, 3, 4]])
def test_csv_rejects_duplicate_or_missing_iteration_positions(tmp_path, iterations):
    path = tmp_path / "encounters.csv"
    rows = encounter_rows()
    for row, iteration in zip(rows, iterations, strict=True):
        row["iteration"] = iteration
    write_encounters(path, rows)
    with pytest.raises(ValueError, match="contiguous and unique"):
        load_skillopt_runs(path, reference(), "tgc", mode="interleaved")


def test_csv_requires_selected_seed_and_matching_benchmark_counts(tmp_path):
    path = tmp_path / "encounters.csv"
    write_encounters(path, [])
    with pytest.raises(ValueError, match="contiguous and unique"):
        load_skillopt_runs(path, reference(), "tgc", mode="interleaved")
    rows = encounter_rows()
    rows[-1]["benchmark"] = "bfcl"
    write_encounters(path, rows)
    with pytest.raises(ValueError, match="benchmark coverage"):
        load_skillopt_runs(path, reference(), "tgc", mode="interleaved")
