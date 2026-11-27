# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import copy
import json

import pytest

from exgentic.agents.premise.llm import PremiseLLM
from exgentic.agents.premise.pipeline import PremisePipeline, apply_operations


def test_premise_edit_changes_child_without_mutating_parent():
    parent = {"playbook": "## OTHERS\n[misc-00001] helpful=2 harmful=1 :: Old rule", "next_global_id": 2}
    original = copy.deepcopy(parent)
    child, applied = apply_operations(parent, [
        {"type": "UPDATE", "entry_id": "misc-00001", "content": "Rewritten rule"},
        {"type": "ADD", "section": "others", "content": "Independent lesson"},
    ], "bfcl", edit_mode="add_update_delete")
    assert parent == original and child["next_global_id"] == 3
    assert "[misc-00001] helpful=2 harmful=1 :: Rewritten rule" in child["playbook"]
    assert "[misc-00002] helpful=0 harmful=0 :: Independent lesson" in child["playbook"]
    assert len(applied) == 2


@pytest.mark.parametrize("mode", ["add_update", "add_update_delete"])
def test_mutation_prompt_and_candidate_application_use_same_edit_mode(mode):
    pipeline = object.__new__(PremisePipeline)
    pipeline.playbook_edit_mode = mode
    pipeline.llm = PremiseLLM("openai/test", "http://localhost/v1", playbook_edit_mode=mode)
    prompts = []
    operations = [{"type": "DELETE", "entry_id": "misc-00001", "rationale": "Superseded"}]

    def ask(prompt, **_kwargs):
        prompts.append(prompt)
        return {"operations": operations, "reasoning": "Remove unsupported guidance"}

    pipeline.llm.ask = ask
    trace = {"benchmark": "bfcl", "task": "Do the task", "context": {},
             "actions": [{"action": {"name": "act"}, "observation": "failed"}], "score": 0, "details": {}}
    reflection = {"id": "reflection_1", "parent_rank": 1, "span": {"start": 0, "end": 1},
                  "analysis": "Wrong rule", "expected_behavior": "Follow the public contract"}
    parent = {"playbook": "## OTHERS\n[misc-00001] helpful=0 harmful=1 :: Wrong rule", "next_global_id": 2}
    child = pipeline._one_candidate(reflection, trace, parent)
    assert child["valid"]
    assert bool(child["applied"]) == (mode == "add_update_delete")
    assert ("misc-00001" in child["playbook"]["playbook"]) == (mode == "add_update")
    assert child["proposed_operations"] == operations
    assert "Only add justified procedural guidance" not in prompts[0]
    assert "UPDATE the existing entry" in prompts[0]
    assert ("DELETE" in prompts[0]) == (mode == "add_update_delete")


def test_checkpoint_rejects_edit_policy_changes_and_legacy_default_remains_resumable(tmp_path):
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"bfcl_42": {"train": ["one"]}}))
    options = {"output": tmp_path / "run", "manifest": manifest, "benchmarks": ["bfcl"], "mode": "isolated", "seed": 42,
               "count": None, "past_count": 3, "model": "openai/test", "base_url": "http://localhost/v1"}
    baseline = PremisePipeline(**options)
    before = (baseline.output / "state.json").read_bytes()
    assert "playbook_edit_mode" not in json.loads(before)["config"]
    assert PremisePipeline(**options).cursor == 0
    with pytest.raises(ValueError, match="configuration differs"):
        PremisePipeline(**options, playbook_edit_mode="add_update")
    assert (baseline.output / "state.json").read_bytes() == before
    options["output"] = tmp_path / "editable"
    editable = PremisePipeline(**options, playbook_edit_mode="add_update_delete")
    config = json.loads((editable.output / "state.json").read_text())["config"]
    assert config["playbook_edit_mode"] == editable.llm.playbook_edit_mode == "add_update_delete"
    with pytest.raises(ValueError, match="configuration differs"):
        PremisePipeline(**options)
