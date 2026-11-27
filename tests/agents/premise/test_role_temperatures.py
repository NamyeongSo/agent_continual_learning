# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Independent analysis temperatures must not change the frozen actor's sampling policy."""

import json
from types import SimpleNamespace

import pytest

from exgentic.agents.premise.llm import PremiseLLM
from exgentic.agents.premise.pipeline import PremisePipeline, write_json
from scripts.utils import run_qwen_isolated_when_free as evaluation


def test_analysis_requests_use_independent_temperatures_and_legacy_defaults_remain(tmp_path):
    requests = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"operations": []}'))],
                               model_dump=lambda **_kwargs: {"choices": []})

    def complete(**kwargs):
        requests.append(kwargs)
        return response

    for options, expected in [({}, [0.0, 0.0]),
                              ({"reflection_temperature": 0.7, "mutation_temperature": 0.9}, [0.7, 0.9])]:
        llm = PremiseLLM("test", "http://localhost/v1", **options)
        llm._client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
        llm.ask("reflection", stage="reflection")
        llm.ask("mutation", stage="mutation")
        assert [request["temperature"] for request in requests[-2:]] == expected


def test_role_temperatures_are_pinned_on_resume_and_final_evaluation(tmp_path, monkeypatch):
    # The temperature-zero comparison snapshot pins this evaluator constant to zero.
    monkeypatch.setattr(evaluation, "TEMPERATURE", 0.0)
    manifest = tmp_path / "split.json"
    write_json(manifest, {"bfcl_42": {"train": ["one"]}})
    options = {"output": tmp_path / "run/train", "manifest": manifest, "benchmarks": ["bfcl"],
               "mode": "isolated", "seed": 42, "count": None, "past_count": 3,
               "model": evaluation.MODEL, "base_url": "http://localhost/v1", "actor_temperature": 0.0,
               "reflection_temperature": 0.7, "mutation_temperature": 0.7}
    pipeline = PremisePipeline(**options)
    state = json.loads((pipeline.output / "state.json").read_text())
    assert state["config"]["role_temperatures"] == {"actor": 0, "reflection": 0.7, "mutation": 0.7}
    assert "temperature_all_roles" not in state["config"]
    with pytest.raises(ValueError, match="configuration differs"):
        PremisePipeline(**{**options, "reflection_temperature": 0.8})
    with pytest.raises(ValueError, match="either all-role"):
        PremisePipeline(**options, temperature=0)
    state["cursor"] = 1
    write_json(pipeline.output / "state.json", state)
    write_json(pipeline.output / "training_complete.json", {"completed_task_count": 1})
    job = {"id": "job", "method": "premise", "benchmark": "bfcl", "seed": 42, "train_count": 1,
           "train_dir": str(pipeline.output), "output": str(pipeline.output.parent), "test_ids": ["test"]}
    tasks = evaluation.frozen_evaluation_jobs(job)
    frozen = json.loads((pipeline.output.parent / "evaluation/frozen_checkpoint.json").read_text())
    assert tasks[0]["task_id"] == "test" and frozen["actor_temperature"] == 0
    assert frozen["training_role_temperatures"]["reflection"] == 0.7
    assert "temperature_all_roles" not in frozen
