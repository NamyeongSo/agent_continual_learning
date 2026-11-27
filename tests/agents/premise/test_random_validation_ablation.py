# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""The validation ablation uses held-out tasks shared across candidates."""

import json
from collections import defaultdict

from exgentic.agents.premise.pipeline import PremisePipeline


def test_random_validation_selects_two_candidates_from_shared_held_out_tasks(tmp_path):
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"bfcl_42": {
        "train": ["train-one"], "val": [f"val-{index}" for index in range(5)], "test": ["test-one"],
    }}))
    pipeline = PremisePipeline(
        output=tmp_path / "run", manifest=manifest, benchmarks=["bfcl"], mode="sequential",
        seed=42, count=None, past_count=3, model="openai/test", base_url="http://localhost/v1",
        candidate_validation="random_val4",
    )
    seen = defaultdict(list)

    def run_agent(**kwargs):
        label = kwargs["label"]
        if "_parent" in label:
            return {"benchmark": "bfcl", "task_id": "train-one", "score": 0.0, "actions": []}
        candidate_id = label.split("_val", 1)[0].split("_candidate_")[1]
        seen[candidate_id].append(kwargs["task_id"])
        assert "replay_source" not in kwargs
        return {"score": {"1": 0.1, "2": 0.9, "3": 0.6, "4": 0.4}[candidate_id]}

    pipeline._run_agent = run_agent
    pipeline._one_reflection = lambda rank, index, trace, playbook: {
        "id": f"reflection_{index}", "parent_rank": rank, "span": {"start": 0}, "entry_assessments": [],
    }
    pipeline._one_candidate = lambda reflection, trace, parent: {
        "id": f"candidate_{reflection['id'].split('_')[-1]}", "parent_rank": reflection["parent_rank"],
        "reflection": reflection, "playbook": parent, "valid": True, "applied": [],
    }
    artifact = pipeline.step()

    assert pipeline.cursor == 1
    sampled = {row["task_id"] for row in artifact["validation_sample"]}
    assert len(sampled) == 4
    assert sampled <= {f"val-{index}" for index in range(5)}
    assert all(set(tasks) == sampled for tasks in seen.values())
    assert artifact["selection"]["selected"] == ["candidate_2", "candidate_3"]
    assert artifact["past_sample"] == []
    assert pipeline.states["global"]["memory"] == []
    assert json.loads((pipeline.output / "state.json").read_text())["config"]["validation_sample_size"] == 4


def test_seen_scope_excludes_future_benchmarks(tmp_path):
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({
        "bfcl_42": {"train": ["b-train"], "val": [f"b-val-{i}" for i in range(6)]},
        "appworld_42": {"train": ["a-train"], "val": [f"a-val-{i}" for i in range(6)]},
    }))
    pipeline = PremisePipeline(
        output=tmp_path / "run", manifest=manifest, benchmarks=["bfcl", "appworld"], mode="sequential",
        seed=42, count=None, past_count=3, model="openai/test", base_url="http://localhost/v1",
        candidate_validation="random_val4", validation_scope="seen",
    )
    assert all(row["benchmark"] == "bfcl" for row in pipeline._validation_sample(1))
    assert {row["benchmark"] for row in pipeline._validation_sample(2)} <= {"bfcl", "appworld"}
