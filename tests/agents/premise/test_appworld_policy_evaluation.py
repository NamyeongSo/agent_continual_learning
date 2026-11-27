# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Explicit policy replacement and ordered, frozen checkpoint evaluation."""

import copy
import json
import threading
from pathlib import Path

import pytest

from exgentic.agents.litellm_tool_calling.instance import LiteLLMToolCallingAgentInstance
from exgentic.agents.premise import pipeline as premise_pipeline
from exgentic.agents.premise.instance import PremiseAgentInstance
from exgentic.agents.premise.reward_data import write_json
from scripts.premise import evaluate_appworld_policy as evaluation


def test_policy_override_uses_actual_apps_preserves_empty_arguments_and_other_context(monkeypatch):
    captured = {}

    def start(self, task, context, actions):
        captured.update(task=task, context=context, actions=actions)
        self.messages = []

    monkeypatch.setattr(LiteLLMToolCallingAgentInstance, "start", start)
    agent = object.__new__(PremiseAgentInstance)
    agent.playbook = "unchanged playbook"
    agent.context_policy_override = "Apps: {allowed}. Complete with finish {} and no summary."
    context = {
        "policy": "The applications include: supervisor, gmail, phone_contacts.\nOriginal policy.",
        "datetime": "2026-01-01",
        "supervisor": {"first_name": "Example"},
    }
    before = copy.deepcopy(context)
    agent.start("task", context, [])
    assert context == before
    assert (
        captured["context"]["policy"]
        == "Apps: supervisor, gmail, phone_contacts. Complete with finish {} and no summary."
    )
    assert captured["context"]["supervisor"] == context["supervisor"]
    assert agent.playbook == "unchanged playbook"
    agent.context_policy_override = None
    agent.start("task", context, [])
    assert captured["context"] == before
    agent.context_policy_override = "{allowed}"
    with pytest.raises(ValueError, match="actual allowed applications"):
        agent.start("task", {}, [])


def test_base_prompt_override_is_exact_and_keeps_benchmark_policy(monkeypatch):
    captured = {}

    def start(self, task, context, actions):
        captured["context"] = context
        self.messages = []

    monkeypatch.setattr(LiteLLMToolCallingAgentInstance, "start", start)
    agent = object.__new__(PremiseAgentInstance)
    agent.playbook = "ignored learned playbook"
    agent.context_policy_override = None
    agent.system_prompt_override = "Exact base prompt."
    agent.start("task", {"policy": "built-in AppWorld policy"}, [])
    assert agent.messages[0]["content"] == "Exact base prompt."
    assert captured["context"]["policy"] == "built-in AppWorld policy"


def test_training_delivers_appworld_policy_through_agent_context(tmp_path, monkeypatch):
    captured = {}

    def execute(*, config):
        captured["config"] = config
        raise RuntimeError("configuration captured")

    monkeypatch.setattr(premise_pipeline, "execute", execute)
    pipeline = object.__new__(premise_pipeline.PremisePipeline)
    pipeline.run_namespace = "test"
    pipeline.benchmark_configs = {"appworld": {"subset": "test_challenge"}}
    pipeline.appworld_policy_override = "Apps: {allowed}. Finish with {}."
    pipeline.max_steps = {"appworld": 50}
    pipeline.model = "openai/test"
    pipeline.output = tmp_path
    pipeline.tool_calling_max_tokens = 2048
    pipeline.llm_timeout = 600
    pipeline.llm_retries = 1
    with pytest.raises(RuntimeError, match="configuration captured"):
        pipeline._run_agent(benchmark="appworld", task_id="task", playbook={"playbook": ""}, label="test")
    config = captured["config"]
    assert "policy_override" not in config.benchmark_kwargs
    assert config.agent_kwargs["context_policy_override"] == pipeline.appworld_policy_override


def test_tgc_uses_binary_official_success_and_keeps_errors_visible():
    rows = [
        {"status": "completed", "tgc": 1, "partial_score": 1.0},
        {"status": "completed", "tgc": 0, "partial_score": 0.9},
        {"status": "error", "tgc": 0, "partial_score": 0.0},
    ]
    summary = evaluation.summarize(rows)
    assert summary["tgc"] is None
    assert summary["tgc_with_errors_zero"] == 1 / 3
    assert summary["partial_task_mean"] == pytest.approx(1.9 / 3)
    assert summary["errors"] == 1
    assert summary["complete"]


def test_seeds_finish_in_order_resume_results_and_preserve_source_checkpoints(tmp_path, monkeypatch):
    jobs = []
    for seed in (42, 43, 44):
        checkpoint = tmp_path / f"source_s{seed}.json"
        write_json(checkpoint, {"playbook": "frozen"})
        previous = tmp_path / f"previous_s{seed}.json"
        write_json(previous, {"results": [{"task_id": "one", "score": 1.0, "status": "completed"}]})
        jobs.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": evaluation.sha256(checkpoint),
                "previous_evaluation": str(previous),
                "task_ids": ["one", "two"],
            }
        )
    starts = []
    finished = {42: 0, 43: 0, 44: 0}
    lock = threading.Lock()

    def evaluate_one(job, index, task_id, policy, directory):
        with lock:
            if job["seed"] == 43:
                assert finished[42] == 2
            if job["seed"] == 44:
                assert finished[43] == 2
            starts.append(job["seed"])
            finished[job["seed"]] += 1
        return {"task_id": task_id, "status": "completed", "success": True, "tgc": 1, "partial_score": 1.0}

    monkeypatch.setattr(evaluation, "evaluate_one", evaluate_one)
    before = [Path(job["checkpoint"]).read_bytes() for job in jobs]
    output = tmp_path / "evaluation"
    assert evaluation.run(output, jobs, "policy", workers=2) == 0
    assert starts == [42, 42, 43, 43, 44, 44]
    assert [Path(job["checkpoint"]).read_bytes() for job in jobs] == before
    assert evaluation.run(output, jobs, "policy", workers=2) == 0
    assert len(starts) == 6
    assert json.loads((output / "status.json").read_text())["seed_order"] == [42, 43, 44]
    for seed in (42, 43, 44):
        assert json.loads((output / f"s{seed}_sequential_cp100/evaluation.json").read_text())["tgc"] == 1.0
