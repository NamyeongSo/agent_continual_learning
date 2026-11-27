# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Proposal retrieval must change evidence only, not replay, official scoring or task history."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import ClassVar

import pytest

from exgentic.agents.premise.pipeline import PremisePipeline, write_json
from exgentic.agents.premise.proposal_retrieval import (
    ProposalRetriever,
    action_bag,
    reflection_text,
    render_examples,
    select_examples,
)
from exgentic.agents.premise.span_policy_shift import strict_f1


class Embeddings:
    identity: ClassVar[dict] = {"model": "sentence-transformers/all-MiniLM-L6-v2", "revision": "test"}

    def __init__(self, *_args):
        self.texts = []

    def encode(self, texts):
        self.texts.extend(texts)
        return [
            {"vector": [1, 0] if "alpha" in text else [0, 1], "input_wordpieces": len(text.split()), "truncated": False}
            for text in texts
        ]


def trace(task, *, success=True):
    return {
        "benchmark": "bfcl",
        "task_id": task,
        "task": f"Task {task}",
        "success": success,
        "actions": [{"action": {"name": "lookup", "arguments": {"key": task}}}],
        "score": float(success),
        "context": {},
        "details": {},
    }


def reflection(identity="reflection_1", rank=1):
    return {
        "id": identity,
        "parent_rank": rank,
        "analysis": "alpha observed target error",
        "expected_behavior": "alpha identify target",
        "span": {"start": 0, "end": 1, "type": "failure"},
        "entry_assessments": [],
    }


def test_full_action_bag_matches_exact_metric_and_preserves_parallel_duplicate_calls():
    original = {
        "actions": [
            {
                "action": {
                    "actions": [
                        {"name": "f", "arguments": {"x": 1}, "id": "call_a", "validation": {"valid": True}},
                        {"name": "g", "arguments": {}},
                        {"name": "f", "arguments": '{"x": 1}'},
                    ]
                }
            }
        ]
    }
    actions = action_bag(original)
    assert actions == [
        {"name": "f", "arguments": {"x": 1}},
        {"name": "g", "arguments": {}},
        {"name": "f", "arguments": {"x": 1}},
    ]
    assert strict_f1(actions[:2], actions) == 0.8
    assert strict_f1(list(reversed(actions)), actions) == 1
    assert action_bag({"actions": [{"action": {"name": "message", "arguments": {"content": "hello"}}}]}) == [
        {"name": "message", "arguments": {"content": "hello"}},
    ]


def test_reflection_embedding_contains_only_analysis_and_expected_behavior():
    first = reflection()
    other = {
        **first,
        "id": "different",
        "parent_rank": 99,
        "entry_assessments": [{"entry_id": "secret"}],
        "span": {"start": 22, "end": 25},
    }
    assert reflection_text(first) == reflection_text(other)
    assert "alpha identify target" in reflection_text(first)


def scored_rows():
    return [
        {"id": name, "task_id": name, "benchmark": "bfcl", "action_f1": f1, "reflection_cosine": cosine}
        for name, f1, cosine in [
            ("a", 0.9, 0.9),
            ("b", 0.8, 0.1),
            ("c", 0.7, 0.4),
            ("d", 0.3, 0.8),
            ("e", 0.2, 0.7),
            ("f", 0.1, 0.2),
        ]
    ]


@pytest.mark.parametrize("mode,expected", [("a", ["b", "c"]), ("b", ["e", "d"]), ("ab", ["b", "e"])])
def test_two_stage_ranking_uses_primary_similarity_and_secondary_contrast(mode, expected):
    rows = scored_rows()
    assert [row["id"] for row in select_examples(rows, mode, random_seed=42)] == expected
    assert [row["id"] for row in select_examples(list(reversed(rows)), mode, random_seed=42)] == expected


@pytest.mark.parametrize("mode", ["a", "b", "ab", "random"])
def test_count_two_distinct_tasks_ties_and_early_missing_history(mode):
    rows = scored_rows()
    rows.append({**rows[0], "id": "alternate_reflection"})
    result = select_examples(rows, mode, random_seed=42)
    assert len(result) == len({row["task_id"] for row in result}) == 2
    assert len(select_examples(rows[:1], mode, random_seed=42)) == 1
    assert select_examples([], mode, random_seed=42) == []
    zeros = [{**row, "action_f1": 0} for row in scored_rows()]
    result = select_examples(zeros, mode, random_seed=42)
    assert len(result) == 2 and all(row["action_f1"] == 0 for row in result)


def test_random_sampling_is_reproducible_without_changing_replay_rng():
    rng = random.Random(42)
    before = rng.getstate()
    first = select_examples(scored_rows(), "random", random_seed="42|iteration|reflection")
    assert first == select_examples(scored_rows(), "random", random_seed="42|iteration|reflection")
    assert rng.getstate() == before


def test_archive_reads_only_committed_training_parents_and_keeps_unselected_reflections(tmp_path):
    one, two = trace("one", success=False), trace("two")
    write_json(
        tmp_path / "iterations/iter_0001.json",
        {
            "parents": [one, two],
            "reflections": [reflection(), reflection("reflection_3", 2)],
            "selection": {"selected": ["candidate_3"]},
        },
    )
    # A future iteration is never eligible, even if its file exists.
    write_json(tmp_path / "iterations/iter_0003.json", {"parents": [trace("future")], "reflections": [reflection()]})
    embed = Embeddings()
    retriever = ProposalRetriever(tmp_path, "ab", embed, 42)
    current = trace("new")
    packet = retriever.prepare(2, [(current, reflection())])["reflection_1"]
    assert packet["audit"]["eligible_tasks"] == 2
    assert {example["task_id"] for example in packet["audit"]["examples"]} == {"one", "two"}
    assert all(example["iteration"] < 2 for example in packet["audit"]["examples"])
    assert "Original official success: False" in packet["prompt"]
    assert "Task future" not in packet["prompt"]
    # Same-task retrieval and repeat archive ingestion are both excluded.
    packet = retriever.prepare(2, [(one, reflection())])["reflection_1"]
    assert packet["audit"]["eligible_tasks"] == 1
    assert len(retriever.archive) == 2


@pytest.mark.parametrize("mode", ["a", "b", "ab", "random"])
def test_three_task_pipeline_keeps_real_replay_past_three_and_official_score_pareto(tmp_path, monkeypatch, mode):
    import exgentic.agents.premise.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "MiniLMClient", Embeddings)
    manifest = tmp_path / "split.json"
    write_json(manifest, {"bfcl_42": {"train": ["one", "two", "three"]}})
    options = {
        "output": tmp_path / "run",
        "manifest": manifest,
        "benchmarks": ["bfcl"],
        "mode": "isolated",
        "seed": 42,
        "count": None,
        "past_count": 3,
        "model": "openai/test",
        "base_url": "http://localhost/v1",
        "alignment_scoring": False,
        "proposal_retrieval": mode,
    }
    pipeline = PremisePipeline(**options)
    calls, prompts = [], []

    def run_agent(**kwargs):
        calls.append(kwargs)
        return trace(kwargs["task_id"])

    def mutate(*_args, proposal_examples):
        prompts.append(proposal_examples)
        return {"operations": [{"type": "ADD", "section": "strategies_and_insights", "content": "Check target."}]}

    pipeline._run_agent = run_agent
    pipeline.llm = SimpleNamespace(
        reflect=lambda *_a, **_kw: reflection(), mutate=mutate, judge=lambda *_a: pytest.fail("No alignment calls")
    )
    artifacts = [pipeline.step() for _ in range(3)]
    assert len(prompts) == 12
    assert [artifact["candidates"][0]["proposal_retrieval"]["selected"] for artifact in artifacts] == [0, 1, 2]
    assert len(artifacts[-1]["past_sample"]) == 2
    assert artifacts[-1]["parallelism"]["replay_jobs"] == 12
    assert any("replay_source" in call for call in calls)
    assert all(
        artifact["selection"]["criterion"] == "official_current_past_without_alignment" for artifact in artifacts
    )
    assert all(
        "current_official_score" in candidate and "current_evaluation_score" not in candidate
        for artifact in artifacts
        for candidate in artifact["candidates"]
    )
    assert all(
        len({row["task_id"] for row in candidate["proposal_retrieval"]["examples"]}) == 2
        for candidate in artifacts[-1]["candidates"]
    )
    restored = PremisePipeline(**options)
    assert restored.cursor == 3
    changed = {**options, "proposal_retrieval": "a" if mode != "a" else "b"}
    with pytest.raises(ValueError, match="configuration differs"):
        PremisePipeline(**changed)


def test_none_mode_does_not_change_existing_mutation_prompt(monkeypatch):
    from exgentic.agents.premise.llm import PremiseLLM

    llm = PremiseLLM("openai/test", "http://localhost/v1", playbook_edit_mode="add_update_delete")
    captured = []
    monkeypatch.setattr(llm, "ask", lambda prompt, **_kw: captured.append(prompt) or {"operations": []})
    original = trace("one")
    parent = {"playbook": "## STRATEGIES & INSIGHTS\n"}
    llm.mutate(original, parent, reflection())
    baseline = captured[-1]
    llm.mutate(original, parent, reflection(), proposal_examples=render_examples([]))
    assert captured[-1].startswith(baseline)
    assert "historical_training_evidence" not in baseline
    assert "No previous distinct training task" in captured[-1]
    assert original == trace("one")
