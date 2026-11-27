# SPDX-License-Identifier: Apache-2.0
"""Shared reflection semantics, first-decision validation and ACE consumption."""

import json

import pytest

from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.agents.ace.playbook_store import PlaybookStore
from exgentic.reflection import RULES, SYSTEM, ace_tags, build_prompt, instruction_updates, validate


def reflection():
    return {
        "span": {"start": 1, "end": 2, "type": "failure"},
        "analysis": "action[1] changed the wrong record.",
        "expected_behavior": "Use the identifier obtained from the lookup, then verify the changed record.",
        "entry_assessments": [{"entry_id": "str-00001", "tag": "harmful"}],
    }


def test_shared_template_has_no_benchmark_specific_examples_and_keeps_evidence():
    for name in ("AppWorld", "BFCL", "Terminus", "Terminal-Bench", "run_terminus2"):
        assert name not in SYSTEM + RULES
    trace = {
        "task": "Update the requested record",
        "actions": [
            {"action": {"name": "lookup"}, "observation": "correct ID"},
            {"action": {"name": "update"}, "observation": "wrong ID changed"},
        ],
        "details": {"requirement": "target record", "expected_output": "private answer", "trace": "private trace"},
    }
    prompt = build_prompt(trace, "existing playbook")
    assert "FIRST decision" in prompt
    assert "action[0]" in prompt and "action[1]" in prompt
    assert "wrong ID changed" in prompt
    assert "private answer" not in prompt and "private trace" not in prompt


@pytest.mark.parametrize("bounds", [(-1, 1), (1, 1), (1, 3), (True, 2)])
def test_invalid_first_decision_bounds_cannot_reach_curator(bounds):
    value = reflection()
    value["span"].update(start=bounds[0], end=bounds[1])
    with pytest.raises(ValueError, match="bounds"):
        validate(value, 2, "[str-00001] Existing rule")


def test_unknown_entry_ids_rejected_and_bracketed_known_ids_normalized():
    value = reflection()
    value["entry_assessments"][0]["entry_id"] = "[str-00001]"
    validate(value, 2, "[str-00001] Existing rule")
    assert ace_tags(value) == [{"id": "str-00001", "tag": "harmful"}]
    with pytest.raises(ValueError, match="nonexistent"):
        validate(value, 2, "empty playbook")


def test_ace_pairs_tool_results_by_call_id_and_passes_the_full_json(tmp_path, monkeypatch):
    agent = ACEAgentInstance(session_id="shared", actor_context="shared-empty-v1")
    agent.task = "Update the right record"
    agent.context = {}
    agent._store = PlaybookStore("shared-test", "## STRATEGIES & INSIGHTS\n[str-00001] helpful=0 harmful=0 :: old rule")
    agent._registry = None
    agent.messages = [
        {"role": "assistant", "tool_calls": [{"id": "call_lookup", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_lookup", "content": "right ID"},
        {"role": "user", "content": "Now update the record"},
        {"role": "assistant", "tool_calls": [{"id": "call_update", "function": {"name": "update", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_update", "content": "wrong ID changed"},
    ]
    agent._training_feedback = {"score": 0, "success": False}
    from exgentic.utils.paths import SessionPaths

    agent._paths = SessionPaths(session_id="shared", run_id="test", output_dir=str(tmp_path))
    agent.paths.agent_dir.mkdir(parents=True, exist_ok=True)
    seen = []

    def response(model, prompt, **kwargs):
        seen.append((prompt, kwargs))
        return json.dumps(reflection())

    monkeypatch.setattr(agent, "_llm_call_simple", response)
    result = json.loads(agent._shared_reflection(agent._store.playbook, "crosscheck"))
    assert result["expected_behavior"] == reflection()["expected_behavior"]
    assert result["span"]["start"] == 1
    assert seen[0][1]["system_prompt"] == SYSTEM
    artifact = json.loads((agent.paths.agent_dir / "shared_reflection.json").read_text())
    assert artifact["actions"][0]["observation"] == {
        "tool_results": "right ID",
        "subsequent_user_messages": ["Now update the record"],
    }
    assert instruction_updates(artifact["actions"]) == [
        {"after_action": 0, "applies_from_action": 1, "message": "Now update the record"}
    ]
    assert artifact["actions"][1]["observation"] == "wrong ID changed"
    assert "harmful=1" in agent._store.playbook
