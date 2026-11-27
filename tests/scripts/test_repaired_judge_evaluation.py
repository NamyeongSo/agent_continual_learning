# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Keep repair comparisons label-blind, paired, and invariant to swapped letter labels."""

import copy
import json

import pytest

from exgentic.agents.premise.reward_data import digest, write_json
from scripts.premise.evaluate_repaired_judge import build_queue, score_cases, validation_inputs


def inputs(tmp_path):
    records, visible = [], {"original": {}, "repaired": {}}
    for bench in ("bfcl", "terminalbench2"):
        records.append({"id": bench, "benchmark": bench, "task_id": f"unexposed_task_{bench}", "split": "val",
                        "label": "A", "a_is_original": True, "provenance": {"private_score": 0.999}})
        value = {"task": "Move the requested item", "context": {}, "public_tools": [], "common_prefix": [],
                 "trajectory_a": [{"action": {"arguments": {}}}], "trajectory_b": [{"action": {"arguments": {}}}]}
        visible["original"][bench] = value
        visible["repaired"][bench] = copy.deepcopy(value)
        if bench == "bfcl":
            visible["repaired"][bench]["trajectory_a"][0]["action"]["arguments"] = {"destination": "archive"}
    for variant in visible:
        rows = [{**r, "input_sha256": digest(visible[variant][r["id"]])} for r in records]
        write_json(tmp_path / variant / "manifest.json", {"records": rows})
        for pair_id, value in visible[variant].items():
            write_json(tmp_path / variant / "pairs" / f"{pair_id}.json", value)
    return records, visible


@pytest.mark.parametrize("field,value", [("label", "B"), ("split", "train"), ("provenance", {"private_score": 0})])
def test_evaluation_refuses_repairs_that_change_gold_or_splits(tmp_path, field, value):
    inputs(tmp_path)
    path = tmp_path / "repaired/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["records"][0][field] = value
    write_json(path, manifest)
    with pytest.raises(ValueError, match="changed labels, splits or source score provenance"):
        validation_inputs(tmp_path / "original", tmp_path / "repaired")


def test_queue_deduplication_preserves_every_case_without_exposing_gold(tmp_path):
    records, _ = inputs(tmp_path)
    loaded, visible, _ = validation_inputs(tmp_path / "original", tmp_path / "repaired")
    jobs = build_queue(loaded, visible, {"no_rubric": [], "discovered": [{"description": "Check requested outcomes"}]})
    cases = [case for job in jobs for case in job["cases"]]
    assert len(cases) == len(records) * 2 * 2 * 2
    assert {case["swapped"] for case in cases} == {False, True}
    for job in jobs:
        payload = json.loads(job["messages"][1]["content"])
        assert set(payload) == {"comparison", "criteria"}
        assert "private_score" not in str(payload)
        assert "unexposed_task" not in str(payload)
    terminal = [job for job in jobs if any(c["benchmark"] == "terminalbench2" for c in job["cases"])]
    assert all({c["variant"] for c in job["cases"]} == {"original", "repaired"} for job in terminal)


def test_scoring_maps_swapped_choice_back_to_same_trajectory():
    job = {"request_id": "request", "cases": [{"record_id": "pair", "swapped": True}]}
    row = score_cases(job, {"status": "completed", "choice": "B"}, {"pair": "A"})[0]
    assert row["choice"] == "A"
    assert row["raw_choice"] == "B"
    assert row["reward"] == 1
