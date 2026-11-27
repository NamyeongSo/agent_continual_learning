# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Select the same Terminal-Bench release throughout GEPA and SkillOpt runs."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from scripts.baselines import run_experiment


@pytest.mark.parametrize("method", ["gepa", "skillopt"])
def test_release_override_preserves_splits_limits_and_manifest_selection(tmp_path, method):
    manifest = run_experiment.ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
    args = SimpleNamespace(
        method=method, benchmark="terminalbench2", seed=42, manifest=manifest,
        pilot=False, model="openai/test", base_url="http://localhost:8006/v1", workers=2,
        output=tmp_path / method, terminalbench2_version=None,
    )
    original = run_experiment.make_config(args)
    assert original["benchmark_kwargs"]["subset"] == "2.0"
    args.terminalbench2_version = "2.1"
    selected = run_experiment.make_config(args)
    assert selected["benchmark_kwargs"]["subset"] == "2.1"
    assert selected["benchmark_kwargs"]["actor"] == "terminus2"
    for phase, count in (("train", 30), ("validation", 30), ("test", 29)):
        assert selected[phase] == original[phase]
        assert len(selected[phase]) == count
    assert selected["actor_max_turns"] == 40
    assert selected["actor_output_tokens"] == 2048
    assert selected["total_training_limit"] is None

    payload = json.loads(manifest.read_text())
    payload["terminalbench2_42"]["benchmark_kwargs"] = {"subset": "2.1"}
    args.manifest = tmp_path / "manifest21.json"
    args.manifest.write_text(json.dumps(payload))
    args.terminalbench2_version = None
    assert run_experiment.make_config(args)["benchmark_kwargs"]["subset"] == "2.1"


@pytest.mark.parametrize("method", ["gepa", "skillopt"])
def test_launcher_refuses_reusing_21_results_for_a_20_run(tmp_path, monkeypatch, method):
    monkeypatch.setattr(run_experiment, "check_sources", lambda *_: None)
    monkeypatch.setattr(run_experiment, "verify_prompts", lambda *_: {"ready": True})
    output = tmp_path / method
    command = ["run_experiment.py", "--method", method, "--benchmark", "terminalbench2",
               "--terminalbench2-version", "2.1", "--output", str(output), "--dry-run"]
    monkeypatch.setattr(sys, "argv", command)
    run_experiment.main()
    original = (output / "config.json").read_bytes()
    assert json.loads(original)["benchmark_kwargs"]["subset"] == "2.1"
    command[command.index("--terminalbench2-version") + 1] = "2.0"
    with pytest.raises(ValueError, match="immutable experiment configuration"):
        run_experiment.main()
    assert (output / "config.json").read_bytes() == original
