# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Protect the AppWorld teammate-policy delivery path from stale benchmark venvs."""

import pytest

from scripts.premise import evaluate_appworld_base_prompts as evaluation


@pytest.mark.parametrize("policy_name", ["existing", "teammate"])
def test_policy_is_delivered_through_agent_context_not_benchmark(tmp_path, monkeypatch, policy_name):
    captured = {}

    def execute(*, config):
        captured["config"] = config
        raise RuntimeError("configuration captured")

    monkeypatch.setattr(evaluation, "execute", execute)
    job = {"seed": 42, "prompt": "gepa", "policy": policy_name}
    row = evaluation.evaluate_one(job, 0, "task_id", tmp_path, "teammate policy")
    assert row["status"] == "error"
    config = captured["config"]
    assert "policy_override" not in config.benchmark_kwargs
    expected = "teammate policy" if policy_name == "teammate" else None
    assert config.agent_kwargs["context_policy_override"] == expected
    assert config.agent_kwargs["system_prompt_override"] == evaluation.GEPA_PROMPT


def test_exported_checkpoint_prompt_uses_existing_policy(tmp_path, monkeypatch):
    captured = {}

    def execute(*, config):
        captured["config"] = config
        raise RuntimeError("configuration captured")

    monkeypatch.setattr(evaluation, "execute", execute)
    job = {"seed": 44, "prompt": "skillopt", "policy": "existing", "system_prompt": "Frozen CP100 skill",
           "run_id_prefix": "team_prompt_cp100"}
    row = evaluation.evaluate_one(job, 0, "task_id", tmp_path, "", max_tokens=8192)

    assert row["status"] == "error"
    config = captured["config"]
    assert config.agent_kwargs["system_prompt_override"] == "Frozen CP100 skill"
    assert config.agent_kwargs["context_policy_override"] is None
    assert "policy_override" not in config.benchmark_kwargs
    assert config.agent_kwargs["model_settings"].max_tokens == 2048
    assert config.run_id == "team_prompt_cp100_s44_skillopt_existing_000"
