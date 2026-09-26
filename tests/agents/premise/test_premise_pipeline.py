# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import random
import threading
import time
from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from exgentic.agents.premise.pipeline import PremisePipeline, apply_operations, is_nontransient_task_error, write_json
from exgentic.agents.premise.selection import sample_past, select_beam
from exgentic.core.orchestrator.prefix_replay import PrefixReplaySession, _restore_action
from exgentic.core.types import ActionType, EmptyObservation, SingleAction, SingleObservation


def test_balanced_past_sample_and_capacity_redistribution():
    memory = [{"benchmark": slug, "task_id": f"{slug}-{n}"}
              for slug in ("appworld", "bfcl", "swebench") for n in range(6)]
    sample = sample_past(memory, 3, random.Random(42))
    assert Counter(item["benchmark"] for item in sample) == {
        "appworld": 1, "bfcl": 1, "swebench": 1,
    }
    sample = sample_past(memory, 6, random.Random(42))
    assert set(Counter(item["benchmark"] for item in sample).values()) == {2}
    two = [item for item in memory if item["benchmark"] != "swebench"]
    assert sorted(Counter(item["benchmark"] for item in sample_past(two, 3, random.Random(42))).values()) == [1, 2]
    depleted = memory[:1] + [item for item in memory if item["benchmark"] == "bfcl"]
    assert Counter(item["benchmark"] for item in sample_past(depleted, 6, random.Random(42))) == {
        "appworld": 1, "bfcl": 5,
    }


def test_alignment_then_pct_uses_nested_pareto_layers():
    candidates = [
        {"id": "candidate_1", "valid": True, "current_alignment": 1, "past_alignment": 0.75, "current_pct": 1, "past_pct": 0.95},
        {"id": "candidate_2", "valid": True, "current_alignment": 1, "past_alignment": 1, "current_pct": 1, "past_pct": 0.88925},
        {"id": "candidate_3", "valid": True, "current_alignment": 0, "past_alignment": 1, "current_pct": 1, "past_pct": 0.95},
        {"id": "candidate_4", "valid": True, "current_alignment": 1, "past_alignment": 0.875, "current_pct": 1, "past_pct": 0.95},
        {"id": "candidate_5", "valid": True, "current_alignment": 1, "past_alignment": 1, "current_pct": 1, "past_pct": 0.95},
    ]
    selected, diagnostics = select_beam(candidates)
    assert [item["id"] for item in selected] == ["candidate_5", "candidate_2"]
    assert diagnostics["alignment_fronts"][0] == ["candidate_2", "candidate_5"]


class Arguments(BaseModel):
    value: int


class AddAction(SingleAction):
    arguments: Arguments


class FakeSession:
    task_id = "one"
    context = {"base": "context"}
    actions = [ActionType(name="add", description="add", cls=AddAction)]

    def __init__(self):
        self.value = 0

    def start(self):
        return EmptyObservation()

    def step(self, action):
        self.value += action.arguments.value
        return SingleObservation(result={"value": self.value}, invoking_actions=[action])

    def done(self):
        return False


def test_prefix_replay_executes_real_action_and_returns_clean_start():
    session = FakeSession()
    replay = PrefixReplaySession(session, {
        "task_id": "one",
        "actions": [{"action": {"name": "add", "arguments": {"value": 3}}, "action_class": "SingleAction"}],
    })
    assert replay.start().is_empty()
    assert session.value == 3
    assert replay.context["premise_replay_prefix"][0]["index"] == 0


def test_prefix_replay_preserves_full_observation():
    class LongObservationSession(FakeSession):
        def step(self, action):
            return SingleObservation(result="x" * 700 + "OBSERVATION_END", invoking_actions=[action])

    replay = PrefixReplaySession(LongObservationSession(), {
        "task_id": "one",
        "actions": [{"action": {"name": "add", "arguments": {"value": 3}}}],
    })
    replay.start()
    assert "OBSERVATION_END" in replay.context["premise_replay_prefix"][0]["observation"]


def test_prefix_replay_reexecutes_invalid_and_unknown_actions():
    class ErrorObservationSession(FakeSession):
        def step(self, action):
            return SingleObservation(result=f"Error from {action.name}", invoking_actions=[action])

    session = ErrorObservationSession()
    invalid = _restore_action({"name": "add", "arguments": {}}, session.actions)
    unknown = _restore_action({"name": "missing_tool", "arguments": {}}, session.actions)
    assert not invalid.validation.args_valid
    assert not unknown.validation.name_valid

    replay = PrefixReplaySession(session, {
        "task_id": "one",
        "actions": [
            {"action": {"name": "add", "arguments": {}}},
            {"action": {"name": "missing_tool", "arguments": {}}},
        ],
    })
    assert replay.start().is_empty()
    assert [entry["action"]["name"] for entry in replay.context["premise_replay_prefix"]] == [
        "add", "missing_tool",
    ]


def test_empty_prefix_preserves_initial_observation():
    session = FakeSession()
    replay = PrefixReplaySession(session, {"task_id": "one", "actions": []})
    assert replay.start().is_empty()
    assert session.value == 0


def test_playbook_edit_cannot_change_another_benchmark_entry():
    parent = {"entries": [{"id": "p000001", "benchmark": "bfcl", "content": "old"}], "next_id": 2}
    child, applied = apply_operations(parent, [{"op": "DELETE", "entry_id": "p000001"}], "appworld")
    assert child == parent
    assert applied == []


def test_memory_trace_reads_existing_iteration_artifact(tmp_path):
    pipeline = object.__new__(PremisePipeline)
    pipeline.output = tmp_path
    write_json(tmp_path / "iterations" / "iter_0001.json", {"parents": [{"task_id": "first"}, {"task_id": "second"}]})
    assert pipeline._memory_trace({"trace_artifact": "iterations/iter_0001.json", "parent_index": 1}) == {"task_id": "second"}


def test_unlimited_sessions_and_experiment_names_are_isolated(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"benchmarks": {"appworld": {"train": ["one"]}}})
    options = dict(
        manifest=manifest, benchmarks=["appworld"], mode="sequential",
        seed=42, count=1, past_count=3, model="openai/test", base_url="http://127.0.0.1:8006/v1",
        benchmark_limits={"appworld": None},
    )
    first = PremisePipeline(output=tmp_path / "t3", **options)
    second = PremisePipeline(output=tmp_path / "t6", **options)
    assert isinstance(first.benchmark_slots["appworld"], nullcontext)
    assert first.run_namespace != second.run_namespace


def test_skip_nontransient_task_preserves_beam_and_advances_checkpoint(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"benchmarks": {"bfcl": {"train": ["one"]}}})
    options = dict(output=tmp_path / "run", manifest=manifest, benchmarks=["bfcl"],
                   mode="sequential", seed=42, count=1, past_count=3,
                   model="openai/test", base_url="http://127.0.0.1:8006/v1")
    pipeline = PremisePipeline(**options, skip_task_errors="nontransient")
    before = json.loads((pipeline.output / "state.json").read_text())
    pipeline.step = lambda: (_ for _ in ()).throw(ValueError("maximum context length exceeded"))
    pipeline.run()
    after = json.loads((pipeline.output / "state.json").read_text())
    artifact = json.loads((pipeline.output / "iterations/iter_0001.json").read_text())
    assert after["cursor"] == 1
    assert after["states"] == before["states"]
    assert artifact["status"] == "skipped"
    assert artifact["train_update_applied"] is False
    assert PremisePipeline(**options, skip_task_errors="none").cursor == 1


def test_nontransient_policy_does_not_hide_service_failures(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"benchmarks": {"bfcl": {"train": ["one"]}}})
    pipeline = PremisePipeline(output=tmp_path / "run", manifest=manifest,
                               benchmarks=["bfcl"], mode="sequential", seed=42,
                               count=1, past_count=3, model="openai/test",
                               base_url="http://127.0.0.1:8006/v1",
                               skip_task_errors="nontransient")
    pipeline.step = lambda: (_ for _ in ()).throw(ConnectionError("server disconnected"))
    with pytest.raises(ConnectionError, match="server disconnected"):
        pipeline.run()
    assert json.loads((pipeline.output / "state.json").read_text())["cursor"] == 0
    assert is_nontransient_task_error(RuntimeError("reflection_1 failed: Invalid reflection span [-1, -1)"))


def test_replay_retries_fresh_session_and_releases_capacity():
    pipeline = object.__new__(PremisePipeline)
    pipeline.replay_retries = 2
    pipeline.benchmark_slots = {"bfcl": threading.BoundedSemaphore(1)}
    pipeline._replay_lock = threading.Lock()
    pipeline._active_replays = pipeline._peak_replays = 0
    calls = []

    def run_agent(**kwargs):
        calls.append(kwargs["label"])
        if len(calls) == 1:
            raise RuntimeError("temporary environment error")
        return {"score": 0.5, "actions": [], "details": {"grader": "stored in session"}}

    pipeline._run_agent = run_agent
    pipeline.llm = SimpleNamespace(judge=lambda *_: {"score": 1.0})
    result = pipeline._score_job(
        {"playbook": {}}, {"benchmark": "bfcl", "task_id": "task"},
        {"span": {"start": 0}}, "job",
    )
    assert calls == ["job_try1", "job_try2"]
    assert result["replay_attempts"] == 2
    assert "details" not in result["replay"]
    assert pipeline._active_replays == 0


def test_validation_runs_frozen_playbook_without_memory_update(tmp_path):
    pipeline = object.__new__(PremisePipeline)
    pipeline.output = tmp_path
    pipeline.manifest = tmp_path / "manifest.json"
    write_json(pipeline.manifest, {"benchmarks": {"bfcl": {"val": ["val-one", "val-two"]}}})
    pipeline.benchmarks = ["bfcl"]
    pipeline.mode = "sequential"
    pipeline.states = {"global": {"beam": [{"entries": []}, {"entries": []}], "memory": []}}
    pipeline.benchmark_slots = {"bfcl": threading.BoundedSemaphore(2)}
    pipeline.replay_workers = 2
    seen = []

    def run_agent(**kwargs):
        seen.append(kwargs["task_id"])
        return {"score": 0.25}

    pipeline._run_agent = run_agent
    assert [item["score"] for item in pipeline.evaluate_validation()] == [0.25, 0.25]
    assert sorted(seen) == ["val-one", "val-two"]
    assert pipeline.states["global"]["memory"] == []


@pytest.mark.parametrize("past_count,expected_jobs", [(3, 16), (6, 28)])
def test_candidate_replays_overlap_and_results_keep_candidate_order(past_count, expected_jobs):
    pipeline = object.__new__(PremisePipeline)
    pipeline.replay_workers = 16
    lock = threading.Lock()
    active = 0
    peak = 0

    def score_job(candidate, source, reflection, label):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(active, peak)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"alignment": {"score": 1.0}, "pct": 0.5, "replay": {"actions": []}}

    pipeline._score_job = score_job
    candidates = [{"id": f"candidate_{i}", "parent_rank": 1, "reflection": {"span": {"start": 0}}, "valid": True}
                  for i in range(1, 5)]
    traces = [{"benchmark": "fake"}]
    past = [{"trace": traces[0], "reflection": {"span": {"start": 0}}} for _ in range(past_count)]
    pipeline._score_candidates(1, candidates, traces, past)
    assert peak > 1
    assert peak <= pipeline.replay_workers
    assert len(candidates) == 4
    assert sum(1 + len(candidate["past_results"]) for candidate in candidates) == expected_jobs
