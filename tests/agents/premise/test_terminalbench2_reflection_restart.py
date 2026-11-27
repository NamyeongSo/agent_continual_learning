# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Verify fresh reflection restarts preserve the frozen split and source state."""

import json

import pytest

from exgentic.agents.premise.pipeline import manifest_order, write_json
from scripts.premise.rerun_isolated_terminalbench2_reflection import digest, prepare


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    manifest = tmp_path / "split.json"
    write_json(manifest, {"terminalbench2_42": {
        "train": [f"train-{i}" for i in range(30)], "val": [f"val-{i}" for i in range(30)],
        "test": [f"test-{i}" for i in range(29)],
    }})
    order, configs = manifest_order(manifest, ["terminalbench2"], 42, None, "isolated")
    write_json(root / "s42/run/state.json", {"cursor": 30, "config": {
        "manifest": str(manifest), "manifest_sha256": digest(manifest), "mode": "isolated",
        "benchmarks": ["terminalbench2"], "past_count": 3, "replay_retries": 2,
        "order": order, "benchmark_configs": configs, "model": "openai/test",
    }})
    return root, manifest


def test_restart_changes_only_timeout_policy_in_copied_split_and_preserves_source(source, tmp_path):
    root, manifest = source
    state_path = root / "s42/run/state.json"
    original_manifest, original_state = manifest.read_bytes(), state_path.read_bytes()
    output = tmp_path / "restart"
    jobs, plan = prepare(root, output, [42], False)
    split = json.loads((output / "task_split_manifest.json").read_text())
    baseline = json.loads(original_manifest)
    assert split["terminalbench2_42"].pop("benchmark_kwargs") == {"enforce_session_timeouts": False}
    assert split == baseline
    assert manifest.read_bytes() == original_manifest
    assert state_path.read_bytes() == original_state
    assert not (output / "s42/run/state.json").exists()  # Training initializes a fresh beam, not the old checkpoint.
    assert plan["training_tasks"] == 30 and plan["test_tasks"] == 29
    assert jobs[0]["source_state_sha256"] == digest(state_path)
    assert plan["enable_summarize"] is False


def test_changed_restart_policy_is_rejected_before_overwriting_frozen_manifest(source, tmp_path):
    root, _ = source
    output = tmp_path / "restart"
    prepare(root, output, [42], False)
    split_path = output / "task_split_manifest.json"
    original = split_path.read_bytes()
    with pytest.raises(ValueError, match="task split differs"):
        prepare(root, output, [42], True)
    assert split_path.read_bytes() == original
