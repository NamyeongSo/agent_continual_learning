# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Grounding must fail closed on unsupported evidence and preserve actual strict-F1 priority."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import ClassVar

import pytest

from exgentic.agents.premise.grounded_proposals import (
    CHECKS,
    GroundedProposals,
    catalogue,
    materialize_citations,
    ranked_tasks,
    render_card,
    validate_reflection,
)
from exgentic.agents.premise.pipeline import PremisePipeline, write_json


class Embeddings:
    identity: ClassVar[dict] = {"model": "test", "revision": "fixed"}

    def __init__(self, *_args):
        pass

    def encode(self, texts):
        return [{"vector": [1.0, 0.0], "truncated": False, "chunks": 1} for _ in texts]


def trace(task="one"):
    return {"benchmark": "bfcl", "task_id": task, "task": "Use the requested target folder", "context": {},
            "details": {}, "success": False, "score": 0,
            "public_tool_contract": {"available_action_names": ["mkdir", "pwd"]},
            "actions": [{"action": {"name": "mkdir", "arguments": {"name": "target"}},
                         "observation": {"result": "wrong location"}}]}


def reflection():
    return {"id": "reflection_1", "parent_rank": 1, "span": {"start": 0, "end": 1, "type": "failure"},
            "analysis": "A directory was created before resolving its location.",
            "expected_behavior": "Check the location first.", "entry_assessments": [],
            "evidence_card": {"observation": "wrong location", "condition": "requested target folder",
                              "correction": "check working directory", "check": "verify resolved location",
                              "counterevidence": "No repair observed", "uncertainty": "Cause not intervened on",
                              "citations": [{"source": "o:0", "quote": "wrong location"}],
                              "known_before": [{"source": "task", "quote": "requested target folder"}],
                              "corrected_actions": [{"name": "pwd", "arguments": {}}]},
            "grounding": {"verified": True}}


def review():
    return {"verdict": "supported", "checks": dict.fromkeys(CHECKS, "supported"),
            "reason": "Citations support a location check", "evidence_sources": ["task", "o:0"]}


def proposal(support=None):
    return {"comparison": dict.fromkeys(("shared_observation", "relevant_difference", "supported_invariant",
                                         "counterevidence", "current_applicability"), "Cited location evidence"),
            "rule": {"condition": "Before a filesystem mutation", "procedure": "Resolve the intended location",
                     "check": "Inspect working directory", "exception": "Limited to this observed API",
                     "support_ids": support or ["current"], "evidence_scope": "current_only"},
            "operations": [{"type": "ADD", "section": "strategies_and_insights",
                            "content": "For BFCL filesystem mutations, resolve the requested target location first."}]}


def test_action_priority_cannot_be_overridden_by_reflection_distance_or_zero_ties():
    rows = [{"id": identity, "benchmark": "bfcl", "task_id": identity,
             "action_f1": score, "reflection_cosine": cosine}
            for identity, score, cosine in [("near", 0.9, 0.99), ("far", 0.2, -0.8), ("zero", 0, -1)]]
    assert [row["id"] for row in ranked_tasks(rows, "A")] == ["near", "far"]
    assert ranked_tasks([{**row, "action_f1": 0} for row in rows], "A") == []
    assert len(ranked_tasks(rows * 2, "A")) == 2


@pytest.mark.parametrize("violation", ["future", "fake_quote", "unchanged", "invented_tool", "direction"])
def test_bad_reflections_are_rejected_without_trusting_a_confident_model(violation):
    source, value = trace(), reflection()
    if violation == "future":
        value["evidence_card"]["known_before"] = [{"source": "o:0", "quote": "wrong location"}]
    elif violation == "fake_quote":
        value["evidence_card"]["citations"][0]["quote"] = "The task ended before the next turn"
    elif violation == "unchanged":
        value["evidence_card"]["corrected_actions"] = [source["actions"][0]["action"]]
    elif violation == "invented_tool":
        value["evidence_card"]["corrected_actions"][0]["name"] = "invented"
    else:
        value["span"]["type"] = "success"
    with pytest.raises(ValueError):
        validate_reflection(value, source)


def test_evidence_card_keeps_literal_excerpts_instead_of_full_large_observations():
    source = trace()
    source["actions"][0]["observation"]["debug"] = "LARGE_IRRELEVANT_FIELD" * 1000
    value = reflection()
    validate_reflection(value, source)
    card = render_card("current", source, value)
    assert "LARGE_IRRELEVANT_FIELD" not in json.dumps(card)
    assert card["quoted_sources"]["o:0"] == ["wrong location"]


def test_model_selects_source_ids_and_program_attaches_unmodified_original_text():
    source, value = trace(), reflection()
    value["evidence_card"]["citations"] = [{"source": "o:0"}]
    materialize_citations(value, source)
    assert value["evidence_card"]["citations"][0]["quote"] == catalogue(source)["o:0"]
    validate_reflection(value, source)


def test_reflection_semantic_review_may_abstain_and_repairs_are_bounded(tmp_path):
    calls = []

    def ask(_prompt, *, stage, **_kwargs):
        calls.append(stage)
        return reflection() if stage == "grounded_reflection" else {**review(), "verdict": "uncertain"}

    method = GroundedProposals(tmp_path, SimpleNamespace(ask=ask), Embeddings(), 42)
    value = method.reflect(trace(), {"playbook": ""})
    assert not value["grounding"]["verified"]
    assert calls == ["grounded_reflection", "evidence_review"] * 2


def test_archive_excludes_unverified_future_and_current_tasks_and_deduplicates(tmp_path):
    good, bad = reflection(), reflection()
    bad["grounding"] = {"verified": False}
    write_json(tmp_path / "iterations/iter_0001.json", {
        "parents": [trace(), trace("bad")],
        "reflections": [good, {**good, "id": "reflection_2"}, {**bad, "parent_rank": 2}]})
    write_json(tmp_path / "iterations/iter_0003.json", {"parents": [trace("future")], "reflections": [good]})
    method = GroundedProposals(tmp_path, SimpleNamespace(), Embeddings(), 42, mode="none")
    packets = method.prepare(2, [(trace(), good)])
    assert len(method.archive) == 1
    assert packets["reflection_1"]["audit"]["eligible_tasks"] == 0
    assert packets["reflection_1"]["audit"]["fallback"] == "current_only"
    method.prepare(2, [(trace(), good)])
    assert len(method.archive) == 1


def test_missing_a_does_not_silently_become_two_b_examples(tmp_path):
    source, ref = trace("past1"), reflection()
    source["actions"][0]["action"]["arguments"] = {"name": "different"}
    write_json(tmp_path / "iterations/iter_0001.json", {
        "parents": [source, {**source, "task_id": "past2"}],
        "reflections": [ref, {**ref, "id": "reflection_3", "parent_rank": 2}]})

    def ask(prompt, **_kwargs):
        data = json.loads(prompt.split("RECORDED DATA:\n")[1])
        return {"relations": [{"id": card["id"], "compatible": True, "reason": "Different observed paths",
                               "current_evidence_sources": ["task"], "past_evidence_sources": ["task"]}
                              for card in data["past"]]}

    method = GroundedProposals(tmp_path, SimpleNamespace(ask=ask), Embeddings(), 42)
    packet = method.prepare(2, [(trace("current"), ref)])["reflection_1"]
    assert len(packet["audit"]["examples"]) == 1
    assert packet["audit"]["examples"][0]["retrieval_role"] == "B"


def test_proposal_rejects_fake_support_and_an_independent_reviewer_can_block_operations(tmp_path):
    result = proposal(["current", "unsupplied"])

    def ask(_prompt, *, stage, **_kwargs):
        if stage == "grounded_mutation":
            return copy.deepcopy(result)
        return {"verdict": "contradicted", "unsupported_claims": ["Rule anticipates a future request"],
                "evidence_sources": ["current"]}

    method = GroundedProposals(tmp_path, SimpleNamespace(ask=ask), Embeddings(), 42)
    with pytest.raises(ValueError, match="unsupplied"):
        method.mutate(trace(), {"playbook": ""}, reflection(), {}, "add_update_delete")
    result = proposal()
    with pytest.raises(ValueError, match="review rejected"):
        method.mutate(trace(), {"playbook": ""}, reflection(), {}, "add_update_delete")


def test_three_task_pipeline_uses_real_replay_and_never_mutates_unverified_reflections(tmp_path, monkeypatch):
    monkeypatch.setattr("exgentic.agents.premise.grounded_proposals.CompleteMiniLMClient", Embeddings)
    manifest = tmp_path / "split.json"
    write_json(manifest, {"bfcl_42": {"train": ["one", "two", "three"]}})
    pipeline = PremisePipeline(output=tmp_path / "run", manifest=manifest, benchmarks=["bfcl"],
                               mode="isolated", seed=42, count=None, past_count=3, model="openai/test",
                               base_url="http://localhost/v1", alignment_scoring=False, grounded_proposals=True,
                               actor_temperature=0, reflection_temperature=0.7, mutation_temperature=0.7)
    sessions, stages = [], []

    def run_agent(**kwargs):
        sessions.append(kwargs)
        return trace(kwargs["task_id"])

    def ask(prompt, *, stage, **_kwargs):
        stages.append(stage)
        if stage == "grounded_reflection":
            return reflection()
        if stage == "evidence_review":
            return review()
        if stage == "retrieval_review":
            data = json.loads(prompt.split("RECORDED DATA:\n")[1])
            return {"relations": [{"id": card["id"], "compatible": True, "reason": "Related location condition",
                                   "current_evidence_sources": ["task"], "past_evidence_sources": ["task"]}
                                  for card in data["past"]]}
        if stage == "grounded_mutation":
            return proposal()
        assert stage == "proposal_review"
        return {"verdict": "supported", "unsupported_claims": [], "evidence_sources": ["current"]}

    pipeline._run_agent = run_agent
    pipeline.llm.ask = ask
    artifacts = [pipeline.step() for _ in range(3)]
    assert any("replay_source" in session for session in sessions)
    assert all(artifact["selection"]["criterion"] == "official_current_past_without_alignment"
               for artifact in artifacts)
    assert all("grounded_proposal" in candidate for artifact in artifacts for candidate in artifact["candidates"])
    assert "evidence_review" in stages and "proposal_review" in stages
    before = len(stages)
    rejected = {**reflection(), "grounding": {"verified": False}}
    candidate = pipeline._one_candidate(rejected, trace(), pipeline.states["bfcl"]["beam"][0])
    assert candidate["valid"] is False and len(stages) == before
    assert catalogue(trace())["task"] == "Use the requested target folder"
