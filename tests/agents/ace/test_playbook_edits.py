# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json

import pytest

from exgentic.agents.ace.ace_agent import ACEAgent
from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.agents.ace.playbook_store import PlaybookStore
from exgentic.agents.ace.playbook_utils import apply_curator_operations, apply_playbook_edits
from exgentic.agents.ace.prompts.curator import CURATOR_PROMPT_NO_GT, get_curator_prompt

PLAYBOOK = (
    "## STRATEGIES & INSIGHTS\n\n"
    "[str-00001] helpful=4 harmful=2 :: Broad old rule\n"
    "  Old continuation and example\n\n"
    "[str-00002] helpful=1 harmful=3 :: Obsolete rule\n\n"
    "## OTHERS\n"
    "[misc-00003] helpful=2 harmful=0 :: Unrelated useful rule"
)


def test_update_rewrites_complete_entry_and_preserves_identity_counters_and_neighbors():
    text, next_id, applied = apply_playbook_edits(PLAYBOOK, [
        {"type": "UPDATE", "entry_id": "str-00001",
         "content": "TYPICALLY: inspect the tool schema.\n  Verify its defaults."},
    ], 4, edit_mode="add_update")
    assert "Broad old rule" not in text and "Old continuation" not in text
    assert "[str-00001] helpful=4 harmful=2 :: TYPICALLY: inspect the tool schema.\n  Verify its defaults." in text
    assert text.index("str-00001") < text.index("str-00002") < text.index("## OTHERS")
    assert "[misc-00003] helpful=2 harmful=0 :: Unrelated useful rule" in text
    assert next_id == 4 and len(applied) == 1


def test_delete_removes_entire_multiline_entry_without_consuming_or_reusing_ids():
    text, next_id, applied = apply_playbook_edits(PLAYBOOK, [
        {"op": "DELETE", "entry_id": "str-00001"},
        {"op": "ADD", "section": "strategies_and_insights", "content": "Replacement rule"},
    ], 1, edit_mode="add_update_delete")
    assert "str-00001" not in text and "Old continuation" not in text
    assert "[str-00004] helpful=0 harmful=0 :: Replacement rule" in text
    assert "Obsolete rule" in text and "## STRATEGIES & INSIGHTS" in text
    assert next_id == 5
    assert [op["type"] for op in applied] == ["DELETE", "ADD"]


@pytest.mark.parametrize("mode,expected", [("add_only", []), ("add_update", ["UPDATE"]),
                                          ("add_update_delete", ["UPDATE", "DELETE"])])
def test_edit_policy_is_enforced_in_application_code(mode, expected):
    text, next_id, applied = apply_playbook_edits(PLAYBOOK, [
        {"type": "UPDATE", "entry_id": "str-00001", "content": "Rewritten guidance"},
        {"type": "DELETE", "entry_id": "str-00002"},
    ], 4, edit_mode=mode)
    assert [op["type"] for op in applied] == expected
    assert next_id == 4
    if mode == "add_only":
        assert text == PLAYBOOK
    assert ("Obsolete rule" in text) == (mode != "add_update_delete")


def test_invalid_unknown_and_ambiguous_targets_never_create_entries():
    ambiguous = PLAYBOOK + "\n[str-00001] helpful=0 harmful=0 :: Duplicate ID"
    text, next_id, applied = apply_playbook_edits(ambiguous, [
        {"type": "UPDATE", "entry_id": "str-00001", "content": "Ambiguous target"},
        {"type": "UPDATE", "entry_id": "missing", "content": "Invented target"},
        {"type": "UPDATE", "entry_id": "str-00002", "content": "  "},
        {"type": "UPDATE", "entry_id": "str-00002", "content": 12},
        {"type": "DELETE", "entry_id": "missing"},
        {"type": "DELETE", "op": "ADD", "entry_id": "str-00002"},
        {"type": "DELETE", "entry_id": "str-00002", "bullet_id": "misc-00003"},
        {"type": "ADD", "section": {}, "content": "Bad section"},
        None,
    ], 4, edit_mode="add_update_delete")
    assert (text, next_id, applied) == (ambiguous, 4, [])


@pytest.mark.parametrize("kind, mode", [("ADD", "add_only"), ("UPDATE", "add_update")])
def test_playbook_edits_preserve_lone_surrogates_as_escapes(kind, mode):
    content = json.loads(r'"예시: \ud800-\udfff, emoji \ud83d\ude00"')
    operation = {"type": kind, "entry_id": "str-00001", "section": "others", "content": content}

    text, _, applied = apply_playbook_edits(PLAYBOOK, [operation], 4, edit_mode=mode)

    assert len(applied) == 1
    assert r"예시: \ud800-\udfff, emoji " + "😀" in text
    assert text.encode("utf-8").decode("utf-8") == text


def test_edits_are_ordered_and_do_not_report_already_deleted_targets_as_applied():
    text, next_id, applied = apply_playbook_edits(PLAYBOOK, [
        {"type": "UPDATE", "bullet_id": "str-00002", "content": "Rewritten"},
        {"type": "DELETE", "entry_id": "str-00002"},
        {"type": "UPDATE", "entry_id": "str-00002", "content": "Must not resurrect"},
    ], 4, edit_mode="add_update_delete")
    assert "str-00002" not in text and "Must not resurrect" not in text
    assert next_id == 4 and len(applied) == 2


def test_empty_and_identical_updates_are_no_change_and_keep_allocator():
    assert apply_curator_operations(PLAYBOOK, [], 4, edit_mode="add_update_delete") == (PLAYBOOK, 4)
    text, next_id, applied = apply_playbook_edits(PLAYBOOK, [
        {"type": "UPDATE", "entry_id": "misc-00003", "content": "Unrelated useful rule"},
    ], 4, edit_mode="add_update_delete")
    assert (text, next_id, applied) == (PLAYBOOK, 4, [])


@pytest.mark.parametrize("mode", ["add_only", "add_update", "add_update_delete"])
def test_prompt_advertises_only_enabled_operations_and_formats_as_json(mode):
    template = get_curator_prompt(mode)
    prompt = template.format(token_budget=80000, current_step=1, total_samples=50, playbook_stats="{}",
                             recent_reflection="Evidence", current_playbook=PLAYBOOK, question_context="Task")
    assert ("UPDATE" in prompt) == (mode != "add_only")
    assert ("DELETE" in prompt) == (mode == "add_update_delete")
    if mode == "add_only":
        assert template == CURATOR_PROMPT_NO_GT
    else:
        assert "0-3 operations" in prompt and "complete replacement content" in prompt
        assert "rationale" in prompt and "correct guidance ignored is not a new lesson" in prompt


@pytest.mark.parametrize("array_response", [False, True])
def test_ace_curator_applies_update_and_delete_from_object_or_user_style_array(monkeypatch, array_response):
    instance = ACEAgentInstance("edit-test", playbook_edit_mode="add_update_delete", training_time=True)
    instance._store = PlaybookStore(store_id="edit-test", initial_playbook=PLAYBOOK)
    instance._store.next_global_id = 4
    instance.task = "Inspect the API and complete the requested action"
    instance.context = {}
    instance._action_log = [{"step": 1, "action": "inspect_api", "arguments": {}}]
    instance.receive_training_feedback({"score": 0, "success": False})
    operations = [
        {"op": "UPDATE", "entry_id": "str-00001", "content": "Read the public contract before acting"},
        {"op": "DELETE", "entry_id": "str-00002"},
    ]
    captured = []

    def ask(_model, prompt, **_kwargs):
        captured.append(prompt)
        return json.dumps(operations if array_response else {"operations": operations})

    monkeypatch.setattr(instance, "_llm_call_simple", ask)
    instance._run_curator("The broad rule caused a contract violation")
    assert "Read the public contract" in instance._store.playbook
    assert "str-00002" not in instance._store.playbook
    assert instance._store.next_global_id == 4
    assert "inspect_api" in captured[0] and "Official post-execution feedback" in captured[0]


def test_factory_forwards_edit_mode_and_rejects_invalid_modes():
    agent = ACEAgent(model="openai/test", playbook_edit_mode="add_update_delete")
    assert agent._get_instance_kwargs("test")["playbook_edit_mode"] == "add_update_delete"
    with pytest.raises(ValueError):
        ACEAgent(model="openai/test", playbook_edit_mode="unknown")
    with pytest.raises(ValueError):
        apply_curator_operations(PLAYBOOK, [], 4, edit_mode="unknown")
