# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from exgentic.agents.ace.ace_agent import ACEAgent
from scripts.ace import run_experiment as ace_cli
from scripts.premise import run_experiment as premise_cli


def _arguments(tmp_path, mode):
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"bfcl_42": {"train": ["one"]}}))
    return ["experiment", "--task-split-manifest", str(manifest), "--output-dir", str(tmp_path / "run"),
            "--benchmarks", "bfcl", "--mode", "isolated", "--seed", "42", "--all-train-tasks",
            "--playbook-edit-mode", mode, "--temperature", "0.7"]


@pytest.mark.parametrize("mode", ["add_only", "add_update", "add_update_delete"])
def test_premise_cli_records_edit_policy_and_protects_resume(tmp_path, monkeypatch, mode):
    args = [*_arguments(tmp_path, mode), "--dry-run"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    premise_cli.main()
    path = tmp_path / "run/state.json"
    before = path.read_bytes()
    config = json.loads(before)["config"]
    assert config.get("playbook_edit_mode", "add_only") == mode
    assert config["temperature_all_roles"] == 0.7
    assert config["experiment_limits"]["bfcl"]["output_tokens"] == 2048
    other = "add_update_delete" if mode != "add_update_delete" else "add_only"
    args[args.index("--playbook-edit-mode") + 1] = other
    with pytest.raises(ValueError, match="configuration differs"):
        premise_cli.main()
    assert path.read_bytes() == before


@pytest.mark.parametrize("mode", ["isolated", "interleaved"])
def test_ace_cli_forwards_policy_to_sessions_and_preserves_config_on_mismatched_resume(tmp_path, monkeypatch, mode):
    args = _arguments(tmp_path, "add_update_delete")
    args[args.index("--mode") + 1] = mode
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(ace_cli, "load_benchmark", lambda _slug: lambda **_kwargs: object())
    monkeypatch.setattr(ace_cli, "load_agent", lambda _slug: ACEAgent)
    agents = []

    def evaluate(**kwargs):
        agents.append(kwargs["agent"])
        session = SimpleNamespace(score=1, cost_reports={}, steps=1, action_count=1, agent_cost=0,
                                  execution_time=0.1, status="success", task_id="one")
        return SimpleNamespace(benchmark_score=1, session_results=[session])

    monkeypatch.setattr(ace_cli, "evaluate", evaluate)
    ace_cli.main()
    path = tmp_path / "run/experiment_config.json"
    before = path.read_bytes()
    assert json.loads(before)["playbook_edit_mode"] == agents[0].playbook_edit_mode == "add_update_delete"
    assert agents[0].temperature_override == 0.7
    args[args.index("--playbook-edit-mode") + 1] = "add_only"
    with pytest.raises(ValueError, match="playbook edit mode differs"):
        ace_cli.main()
    assert path.read_bytes() == before
