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

from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
from exgentic.agents.premise.pipeline import (
    PremisePipeline,
    TaskTimeLimitError,
    apply_operations,
    is_nontransient_task_error,
    playbook_text,
    write_json,
)
from exgentic.agents.premise.selection import sample_past, select_beam, select_ps_st_pareto_beam
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


def test_alignment_then_official_score_uses_nested_pareto_layers():
    candidates = [
        {"id": "candidate_1", "valid": True, "current_alignment": 1, "past_alignment": 0.75, "current_official_score": 1, "past_official_score": 0.95},
        {"id": "candidate_2", "valid": True, "current_alignment": 1, "past_alignment": 1, "current_official_score": 1, "past_official_score": 0.88925},
        {"id": "candidate_3", "valid": True, "current_alignment": 0, "past_alignment": 1, "current_official_score": 1, "past_official_score": 0.95},
        {"id": "candidate_4", "valid": True, "current_alignment": 1, "past_alignment": 0.875, "current_official_score": 1, "past_official_score": 0.95},
        {"id": "candidate_5", "valid": True, "current_alignment": 1, "past_alignment": 1, "current_official_score": 1, "past_official_score": 0.95},
    ]
    selected, diagnostics = select_beam(candidates)
    assert [item["id"] for item in selected] == ["candidate_5", "candidate_2"]
    assert diagnostics["alignment_fronts"][0] == ["candidate_2", "candidate_5"]


def test_ps_st_pareto_uses_official_frontier_then_equal_weighted_score():
    candidates = [
        {"id": "candidate_1", "valid": True, "current_official_score": 1.0, "past_official_score": 0.2},
        {"id": "candidate_2", "valid": True, "current_official_score": 0.7, "past_official_score": 0.7},
        {"id": "candidate_3", "valid": True, "current_official_score": 0.2, "past_official_score": 1.0},
        {"id": "candidate_4", "valid": True, "current_official_score": 0.65, "past_official_score": 0.65},
    ]
    selected, diagnostics = select_ps_st_pareto_beam(candidates)
    assert diagnostics["performance_fronts"] == [
        ["candidate_1", "candidate_2", "candidate_3"], ["candidate_4"],
    ]
    assert [item["id"] for item in selected] == ["candidate_2", "candidate_1"]
    assert "alignment_fronts" not in diagnostics


def test_ps_st_pareto_fills_second_parent_from_next_frontier_and_handles_empty_memory():
    candidates = [
        {"id": "candidate_1", "valid": True, "current_official_score": 1.0, "past_official_score": 1.0},
        {"id": "candidate_2", "valid": True, "current_official_score": 0.9, "past_official_score": 0.7},
        {"id": "candidate_3", "valid": True, "current_official_score": 0.8, "past_official_score": 0.8},
    ]
    selected, diagnostics = select_ps_st_pareto_beam(candidates)
    assert diagnostics["performance_fronts"][0] == ["candidate_1"]
    assert [item["id"] for item in selected] == ["candidate_1", "candidate_2"]
    for candidate in candidates:
        candidate["past_official_score"] = None
    selected, diagnostics = select_ps_st_pareto_beam(candidates)
    assert [item["id"] for item in selected] == ["candidate_1", "candidate_2"]
    assert diagnostics["objectives"]["candidate_1"]["past"] is None


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


@pytest.mark.parametrize("static", [False, True])
def test_static_prefix_sends_tool_definitions_once_and_preserves_state_and_observations(static):
    class CountingSession(FakeSession):
        reads = 0

        @property
        def actions(self):
            self.reads += 1
            return FakeSession.actions

    session = CountingSession()
    replay = PrefixReplaySession(session, {
        "task_id": "one", "actions": [
            {"action": {"name": "add", "arguments": {"value": value}}} for value in (2, 3, 4)
        ],
    }, static_actions=static)
    replay.start()
    assert replay.actions == FakeSession.actions
    assert session.reads == (1 if static else 4)
    assert session.value == 9
    observations = [json.loads(item["observation"])[0] for item in replay.context["premise_replay_prefix"]]
    assert observations == [{"value": 2}, {"value": 5}, {"value": 9}]


def test_dynamic_prefix_refreshes_tool_definitions_after_each_action():
    class DynamicSession(FakeSession):
        calls = 0

        @property
        def actions(self):
            return [ActionType(name=f"tool_{self.calls}", description="current tool", cls=AddAction)]

        def step(self, action):
            assert action.name == f"tool_{self.calls}"
            self.calls += 1
            return super().step(action)

    session = DynamicSession()
    replay = PrefixReplaySession(session, {
        "task_id": "one", "actions": [
            {"action": {"name": f"tool_{i}", "arguments": {"value": 1}}} for i in range(3)
        ],
    })
    replay.start()
    assert session.calls == 3
    assert session.value == 3


def test_playbook_shares_ace_sections_bullets_and_full_content_across_benchmarks():
    parent = {"playbook": DEFAULT_PLAYBOOK, "next_global_id": 1}
    child, applied = apply_operations(parent, [
        {"type": "ADD", "section": "strategies_and_insights", "content": "Use observed IDs for API calls"},
    ], "bfcl")
    assert applied
    assert "[str-00001] helpful=0 harmful=0 :: Use observed IDs" in child["playbook"]
    assert child["next_global_id"] == 2
    assert parent["playbook"] == DEFAULT_PLAYBOOK
    assert playbook_text(child, "bfcl") == playbook_text(child, "appworld") == playbook_text(child, "terminalbench2")


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


def test_terminalbench_release_is_pinned_across_replay_and_resume(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"benchmarks": {"terminalbench2": {"train": ["task-one"]}}})
    options = dict(
        output=tmp_path / "run", manifest=manifest, benchmarks=["terminalbench2"], mode="isolated",
        seed=42, count=1, past_count=3, model="openai/test", base_url="http://127.0.0.1:8006/v1",
    )
    pipeline = PremisePipeline(**options, terminalbench2_version="2.1")
    assert pipeline.benchmark_configs["terminalbench2"]["subset"] == "2.1"
    assert pipeline.benchmark_configs["terminalbench2"]["actor"] == "terminus2"
    saved = json.loads((pipeline.output / "state.json").read_text())
    assert saved["config"]["benchmark_configs"]["terminalbench2"]["subset"] == "2.1"
    assert PremisePipeline(**options, terminalbench2_version="2.1").cursor == 0
    with pytest.raises(ValueError, match="checkpoint configuration differs"):
        PremisePipeline(**options, terminalbench2_version="2.0")


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


def test_truncated_agent_completion_skips_only_current_task(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"benchmarks": {"appworld": {"train": ["one", "two"]}}})
    pipeline = PremisePipeline(
        output=tmp_path / "run", manifest=manifest, benchmarks=["appworld"],
        mode="sequential", seed=42, count=2, past_count=3,
        model="openai/test", base_url="http://127.0.0.1:8006/v1",
        skip_task_errors="nontransient",
    )
    error = RuntimeError(
        "parent1: error: {'session_metadata': {'error': 'LiteLLM completion truncated "
        "(finish_reason=length). To allow truncated responses, increase max_tokens.'}}"
    )
    assert is_nontransient_task_error(error)
    assert not is_nontransient_task_error(RuntimeError("Docker SSH session open refused by peer"))
    before = json.loads((pipeline.output / "state.json").read_text())
    pipeline.step = lambda: (_ for _ in ()).throw(error)
    pipeline.run()
    after = json.loads((pipeline.output / "state.json").read_text())
    assert after["cursor"] == 2
    assert after["states"] == before["states"]
    for iteration in (1, 2):
        artifact = json.loads((pipeline.output / f"iterations/iter_{iteration:04d}.json").read_text())
        assert artifact["status"] == "skipped"
        assert artifact["train_update_applied"] is False


@pytest.mark.parametrize("benchmark,error_type,error_message,task_error", [
    ("terminalbench2", "TimeoutError", "", True),
    ("terminalbench2", "TimeoutError", "provider connection timed out", False),
    ("terminalbench2", "RuntimeError", "Docker server disconnected", False),
    ("bfcl", "TimeoutError", "", False),
])
def test_terminal_execution_deadline_invalidates_branch_without_hiding_outages(
    tmp_path, monkeypatch, benchmark, error_type, error_message, task_error,
):
    import exgentic.agents.premise.pipeline as module

    manifest = tmp_path / "split.json"
    write_json(manifest, {f"{benchmark}_42": {"train": ["one"]}})
    pipeline = PremisePipeline(output=tmp_path / "run", manifest=manifest, benchmarks=[benchmark],
                               mode="sequential", seed=42, count=1, past_count=3,
                               model="openai/test", base_url="http://localhost/v1")
    metadata = {"error_type": error_type, "error_source": "execution", "error": error_message}
    result = SimpleNamespace(status=SimpleNamespace(value="error"), details={"session_metadata": metadata})
    monkeypatch.setattr(module, "execute", lambda **_kwargs: SimpleNamespace(session_results=[result]))
    error_class = TaskTimeLimitError if task_error else RuntimeError
    with pytest.raises(error_class) as raised:
        pipeline._run_agent(benchmark=benchmark, task_id="one", playbook={}, label="parent1")
    assert is_nontransient_task_error(raised.value) is task_error


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


def test_ps_st_pareto_replay_skips_alignment_judge_and_keeps_official_scores():
    pipeline = object.__new__(PremisePipeline)
    pipeline.method = "ps_st_pareto"
    pipeline.replay_retries = 0
    pipeline.replay_workers = 2
    pipeline.benchmark_slots = {"bfcl": nullcontext()}
    pipeline._replay_lock = threading.Lock()
    pipeline._active_replays = pipeline._peak_replays = 0
    pipeline._run_agent = lambda **kwargs: {
        "score": 0.75 if "current" in kwargs["label"] else 0.25,
        "actions": [], "details": {"grader": "stored in session"},
    }
    pipeline.llm = SimpleNamespace(judge=lambda *_: pytest.fail("alignment judge was called"))
    reflection = {"span": {"start": 0}}
    source = {"benchmark": "bfcl", "task_id": "current"}
    candidates = [{"id": "candidate_1", "parent_rank": 1, "reflection": reflection,
                   "playbook": {}, "valid": True}]
    pipeline._score_candidates(1, candidates, [source], [{"trace": source, "reflection": reflection}])
    candidate = candidates[0]
    assert candidate["current_official_score"] == 0.75
    assert candidate["past_official_score"] == 0.25
    assert "current_alignment" not in candidate
    assert "alignment" not in candidate["current_result"]
    assert "details" not in candidate["current_result"]["replay"]


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
        return {"alignment": {"score": 1.0}, "official_score": 0.5, "replay": {"actions": []}}

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


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("mode", ["sequential", "interleaved"])
def test_nine_key_split_has_exact_same_task_order_as_ace(seed, mode):
    import importlib.util
    import sys
    from pathlib import Path

    from exgentic.agents.premise.pipeline import manifest_order

    root = Path(__file__).resolve().parents[3]
    script = root / "scripts/ace/run_experiment.py"
    sys.path.insert(0, str(script.parent))
    try:
        spec = importlib.util.spec_from_file_location("ace_order_reference", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = root / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
        benchmarks = ["bfcl", "appworld", "terminalbench2"]
        order, configs = manifest_order(manifest, benchmarks, seed, None, mode)
        assert order == module.task_order_from_train_manifest(manifest, benchmarks, seed, None, mode)
        assert Counter(slug for slug, _ in order) == {"bfcl": 50, "appworld": 50, "terminalbench2": 30}
        assert configs["bfcl"]["subset"] == "multi_turn_base"
        assert configs["appworld"]["subset"] == "test_challenge"
        assert configs["terminalbench2"]["actor"] == "terminus2"
    finally:
        sys.path.remove(str(script.parent))


def test_context_replay_error_is_not_retried():
    pipeline = object.__new__(PremisePipeline)
    pipeline.replay_retries = 2
    pipeline.benchmark_slots = {"bfcl": nullcontext()}
    pipeline._replay_lock = threading.Lock()
    pipeline._active_replays = pipeline._peak_replays = 0
    calls = []

    def run_agent(**kwargs):
        calls.append(kwargs["label"])
        raise RuntimeError("context length exceeded")

    pipeline._run_agent = run_agent
    with pytest.raises(RuntimeError, match="context length"):
        pipeline._score_job({"playbook": {}}, {"benchmark": "bfcl", "task_id": "x"},
                            {"span": {"start": 0}}, "job")
    assert calls == ["job_try1"]
    assert pipeline._active_replays == 0


def test_mutator_context_error_invalidates_candidate_instead_of_creating_noop():
    pipeline = object.__new__(PremisePipeline)
    pipeline.llm = SimpleNamespace(mutate=lambda *_: (_ for _ in ()).throw(RuntimeError("context length exceeded")))
    candidate = pipeline._one_candidate({"id": "reflection_1", "parent_rank": 1},
                                       {"benchmark": "bfcl"}, {"playbook": DEFAULT_PLAYBOOK})
    assert candidate["valid"] is False
    assert "context length exceeded" in candidate["errors"][0]


@pytest.mark.parametrize("error", [RuntimeError("maximum context length exceeded"),
                                 TaskTimeLimitError("no completed episodes before task deadline")])
def test_one_parent_task_failure_still_selects_other_parent_candidates(tmp_path, error):
    manifest = tmp_path / "split.json"
    write_json(manifest, {"bfcl_42": {"train": ["one"]}})
    pipeline = PremisePipeline(output=tmp_path / "run", manifest=manifest, benchmarks=["bfcl"],
                               mode="sequential", seed=42, count=None, past_count=3,
                               model="openai/test", base_url="http://localhost/v1")

    def run_agent(**kwargs):
        if "parent1" in kwargs["label"]:
            raise error
        return {"benchmark": "bfcl", "task_id": "one", "actions": [{"action": {}}],
                "score": 0.375, "details": {}}

    pipeline._run_agent = run_agent
    pipeline.llm = SimpleNamespace(
        reflect=lambda *_args, **_kwargs: {"span": {"start": 0, "end": 1}, "entry_assessments": []},
        mutate=lambda *_: {"operations": [], "reasoning": "same guidance"},
        judge=lambda *_: {"score": 1},
    )
    artifact = pipeline.step()
    assert artifact["selection"]["selected"] == ["candidate_3", "candidate_4"]
    assert artifact["parents"][0] is None
    assert pipeline.cursor == 1
    assert all(c["current_official_score"] == 0.375 for c in artifact["candidates"])


def test_ps_st_pareto_step_selects_without_alignment_and_cannot_resume_as_premise(tmp_path):
    manifest = tmp_path / "split.json"
    write_json(manifest, {"bfcl_42": {"train": ["one"]}})
    options = dict(output=tmp_path / "run", manifest=manifest, benchmarks=["bfcl"],
                   mode="sequential", seed=42, count=None, past_count=3,
                   model="openai/test", base_url="http://localhost/v1")
    pipeline = PremisePipeline(**options, method="ps_st_pareto")
    pipeline._run_agent = lambda **_: {
        "benchmark": "bfcl", "task_id": "one", "actions": [{"action": {}}],
        "score": 0.375, "details": {},
    }
    pipeline.llm = SimpleNamespace(
        reflect=lambda *_args, **_kwargs: {"span": {"start": 0, "end": 1}, "entry_assessments": []},
        mutate=lambda *_: {"operations": [], "reasoning": "same guidance"},
    )
    artifact = pipeline.step()
    assert artifact["selection"]["selected"] == ["candidate_1", "candidate_2"]
    assert "alignment" not in artifact["candidates"][0]["current_result"]
    assert json.loads((pipeline.output / "state.json").read_text())["config"]["method"] == "ps_st_pareto"
    with pytest.raises(ValueError, match="configuration differs"):
        PremisePipeline(**options)


def test_bfcl_checkpoint_test_can_run_before_training_finishes(tmp_path):
    manifest = tmp_path / "split.json"
    write_json(manifest, {"bfcl_42": {
        "train": [f"train-{index}" for index in range(51)],
        "val": ["val-one"], "test": ["test-one", "test-two"],
    }})
    pipeline = PremisePipeline(output=tmp_path / "run", manifest=manifest, benchmarks=["bfcl"],
                               mode="sequential", seed=42, count=None, past_count=3,
                               model="openai/test", base_url="http://localhost/v1",
                               method="ps_st_pareto")
    pipeline.cursor = 50
    pipeline._save()
    before = (pipeline.output / "state.json").read_bytes()
    pipeline._run_agent = lambda **_: {"score": 0.5}
    summaries = pipeline.evaluate_test_checkpoints(
        checkpoints=(50,), benchmarks=("bfcl",), allow_incomplete=True,
    )
    assert len(summaries) == 1
    assert summaries[0]["mean_official_score"] == 0.5
    assert (pipeline.output / "state.json").read_bytes() == before
    assert not (pipeline.output / "test_evaluation" / "complete.json").exists()
    resumed = PremisePipeline(output=tmp_path / "run", manifest=manifest, benchmarks=["bfcl"],
                              mode="sequential", seed=42, count=None, past_count=3,
                              model="openai/test", base_url="http://localhost/v1",
                              method="ps_st_pareto")
    assert resumed.cursor == 50
    assert (resumed.output / "state.json").read_bytes() == before


@pytest.mark.parametrize("error", [RuntimeError("context length exceeded"),
                                 TaskTimeLimitError("no completed episodes before task deadline")])
def test_all_parent_task_failures_skip_without_beam_or_memory_update(tmp_path, error):
    manifest = tmp_path / "split.json"
    write_json(manifest, {"bfcl_42": {"train": ["one"]}})
    pipeline = PremisePipeline(output=tmp_path / "run", manifest=manifest, benchmarks=["bfcl"],
                               mode="sequential", seed=42, count=None, past_count=3,
                               model="openai/test", base_url="http://localhost/v1")
    before = json.dumps(pipeline.states)
    pipeline._run_agent = lambda **_: (_ for _ in ()).throw(error)
    artifact = pipeline.step()
    assert artifact["status"] == "skipped"
    assert json.dumps(pipeline.states) == before
    assert pipeline.cursor == 1


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("mode", ["sequential", "interleaved"])
def test_checkpoint_test_protocol_is_frozen_resumable_and_seed_specific(tmp_path, seed, mode):
    from pathlib import Path

    manifest = Path(__file__).resolve().parents[3] / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
    pipeline = PremisePipeline(output=tmp_path / "run", manifest=manifest,
                               benchmarks=["bfcl", "appworld", "terminalbench2"], mode=mode, seed=seed,
                               count=None, past_count=3, model="openai/test", base_url="http://localhost/v1")
    for point in (50, 100, 130):
        pipeline.cursor = point
        pipeline.states["global"]["beam"][0]["playbook"] = f"{DEFAULT_PLAYBOOK}\n[str-00001] helpful=0 harmful=0 :: at {point}"
        pipeline._save()
    frozen50 = (pipeline.output / "checkpoints/checkpoint_050.json").read_bytes()
    pipeline.states["global"]["beam"][0]["playbook"] += " later mutation"
    pipeline.cursor = 50
    pipeline._save()
    assert (pipeline.output / "checkpoints/checkpoint_050.json").read_bytes() == frozen50
    pipeline.cursor = 130
    pipeline._save()
    calls = []

    def run_agent(**kwargs):
        calls.append(kwargs)
        point = int(kwargs["label"].split("_")[1][2:])
        assert f"at {point}" in kwargs["playbook"]["playbook"]
        assert "later mutation" not in kwargs["playbook"]["playbook"]
        return {"score": 0.375}

    pipeline._run_agent = run_agent
    before = (pipeline.output / "state.json").read_bytes()
    summaries = pipeline.evaluate_test_checkpoints()
    assert len(summaries) == 9
    assert len(calls) == 3 * (50 + 50 + 29)
    assert (pipeline.output / "state.json").read_bytes() == before
    source = json.loads(manifest.read_text())
    for summary in summaries:
        slug = summary["identity"]["benchmark"]
        assert summary["identity"]["task_ids"] == source[f"{slug}_{seed}"]["test"]
        assert summary["mean_official_score"] == 0.375
        assert summary["identity"]["learning"] is False
    calls.clear()
    assert len(pipeline.evaluate_test_checkpoints()) == 9
    assert calls == []


@pytest.mark.parametrize("method", ["premise", "ps_st_pareto"])
def test_isolated_terminalbench_final_checkpoint_uses_only_its_test_split(tmp_path, method):
    from pathlib import Path

    manifest = Path(__file__).resolve().parents[3] / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
    pipeline = PremisePipeline(output=tmp_path / "run", manifest=manifest,
                               benchmarks=["terminalbench2"], mode="isolated", seed=42,
                               count=None, past_count=3, model="openai/test", base_url="http://localhost/v1",
                               method=method)
    assert len(pipeline.order) == 30
    pipeline.cursor = 30
    pipeline.states["terminalbench2"]["beam"][0]["playbook"] = "trained terminal playbook"
    pipeline._save()
    assert (pipeline.output / "checkpoints/checkpoint_030.json").is_file()
    calls = []
    pipeline._run_agent = lambda **kwargs: calls.append(kwargs) or {"score": 0.5}
    summaries = pipeline.evaluate_test_checkpoints(checkpoints=(30,), benchmarks=("terminalbench2",))
    assert len(summaries) == 1
    assert len(calls) == 29
    assert all(call["playbook"]["playbook"] == "trained terminal playbook" for call in calls)
    assert summaries[0]["mean_official_score"] == 0.5
    assert json.loads((pipeline.output / "test_evaluation/complete.json").read_text())["evaluation_groups"] == 1


@pytest.mark.parametrize("trace_source", ["observation", "metadata", "timeout", "legacy_timeout"])
def test_terminus_internal_episodes_and_official_zero_are_read_from_session(tmp_path, trace_source):
    from exgentic.agents.premise.pipeline import trajectory_from_session

    session = tmp_path / "run" / "sessions" / "session"
    agent_dir = session / "benchmark" / "harbor" / "agent"
    agent_dir.mkdir(parents=True)
    write_json(session / "session.json", {"task_id": "fix-git", "task": "Fix Git", "context": {}})
    full_output = "FRESH_TERMINAL_OUTPUT_" + "x" * 12000
    write_json(agent_dir / "trajectory.json", {"steps": [
        {"source": "user", "message": "instruction"},
        *[{"source": "agent", "message": f"episode {i}",
           "tool_calls": [{"function_name": "bash_command", "arguments": {"keystrokes": "pwd\\n"}}],
           "observation": {"results": [{"content": full_output}]}}
          for i in range(3)],
    ]})
    timed_out = trace_source in ("timeout", "legacy_timeout")
    write_json(agent_dir / "premise_episodes.json", {
        "prefix_length": 1,
        "episodes": [{"response": f"response {i}"} for i in range(3 + int(timed_out))],
    })
    metadata = {"agent_timed_out": timed_out}
    if trace_source != "legacy_timeout":
        metadata["trace_path"] = str(agent_dir / "trajectory.json")
    else:
        (session / "benchmark/session.log").write_text("Action run_terminus2 failed: TimeoutError")
        metadata = {}
    result = metadata if trace_source in ("observation", "timeout") else "Action run_terminus2 failed: "
    events = [
        {"event": "action", "step": 1, "action": {"name": "run_terminus2", "arguments": {}}},
        {"event": "observation", "step": 1,
         "observation": {"result": result}},
    ]
    (session / "trajectory.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    trace = trajectory_from_session(tmp_path, "run", SimpleNamespace(
        session_id="session", score=0.0, success=True,
        details={"session_metadata": metadata, "harbor_rewards": {"reward": 0.0}},
    ), "terminalbench2")
    assert trace["score"] == 0.0
    assert trace["score_source"] == "official_benchmark_session_score"
    assert trace["prefix_length"] == 1
    episode_file = "premise_completed_episodes.json" if timed_out else "premise_episodes.json"
    assert trace["terminus_episode_path"] == str(agent_dir / episode_file)
    assert trace["agent_timed_out"] == timed_out
    if timed_out:
        completed = json.loads((agent_dir / episode_file).read_text())
        assert completed["interrupted_episodes"] == [{"response": "response 3"}]
        assert len(json.loads((agent_dir / "premise_episodes.json").read_text())["episodes"]) == 4
    assert len(trace["actions"]) == 2
    assert trace["actions"][0]["episode"] == 1
    assert trace["actions"][0]["action"]["arguments"]["response"] == "response 1"
    assert full_output in trace["actions"][0]["observation"]["results"][0]["content"]


def test_terminus_deadline_without_a_recording_invalidates_only_that_branch(tmp_path):
    from exgentic.agents.premise.pipeline import trajectory_from_session

    session = tmp_path / "run" / "sessions" / "session"
    session.mkdir(parents=True)
    write_json(session / "session.json", {"task_id": "fix-git", "task": "Fix Git"})
    (session / "trajectory.jsonl").write_text("")
    result = SimpleNamespace(session_id="session", score=0.0,
                             details={"session_metadata": {"agent_timed_out": True}})
    with pytest.raises(TaskTimeLimitError, match="before recording a trajectory") as raised:
        trajectory_from_session(tmp_path, "run", result, "terminalbench2")
    assert is_nontransient_task_error(raised.value)


@pytest.mark.parametrize("message,task_error", [("context length exceeded", True), ("server disconnected", False)])
def test_terminus_original_error_survives_the_action_handler_wrapper(tmp_path, message, task_error):
    from exgentic.agents.premise.pipeline import trajectory_from_session

    session = tmp_path / "run" / "sessions" / "session"
    session.mkdir(parents=True)
    write_json(session / "session.json", {"task_id": "fix-git", "task": "Fix Git"})
    (session / "trajectory.jsonl").write_text("")
    result = SimpleNamespace(session_id="session", score=0.0,
                             details={"session_metadata": {"agent_error": message}})
    with pytest.raises(RuntimeError, match=message) as raised:
        trajectory_from_session(tmp_path, "run", result, "terminalbench2")
    assert is_nontransient_task_error(raised.value) is task_error


def test_infrastructure_error_in_mutator_is_not_silently_accepted():
    pipeline = object.__new__(PremisePipeline)
    pipeline.llm = SimpleNamespace(mutate=lambda *_: (_ for _ in ()).throw(ConnectionError("server down")))
    with pytest.raises(ConnectionError, match="server down"):
        pipeline._one_candidate({"id": "reflection_1", "parent_rank": 1},
                                {"benchmark": "bfcl"}, {"playbook": DEFAULT_PLAYBOOK})


def test_terminal_reflector_context_overflow_skips_task_after_both_parents(tmp_path):
    manifest = tmp_path / "split.json"
    write_json(manifest, {"terminalbench2_42": {"train": ["fix-git"], "val": [], "test": []}})
    pipeline = PremisePipeline(output=tmp_path / "run", manifest=manifest,
                               benchmarks=["terminalbench2"], mode="sequential", seed=42,
                               count=None, past_count=3, model="openai/test",
                               base_url="http://localhost/v1")
    before = json.dumps(pipeline.states)
    calls = []

    def run_parent(**kwargs):
        calls.append(kwargs["label"])
        return {"benchmark": "terminalbench2", "task_id": "fix-git",
                "actions": [{"action": {"name": "terminus2_episode"}}],
                "score": 0.0, "details": {}}

    pipeline._run_agent = run_parent
    message = "This model's maximum context length is 131072 tokens"
    pipeline.llm = SimpleNamespace(reflect=lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError(message)
    ))
    artifact = pipeline.step()
    assert sorted(calls) == ["i0001_parent1", "i0001_parent2"]
    assert artifact["status"] == "skipped"
    assert len(artifact["errors"]) == 4
    assert all("maximum context length" in error for error in artifact["errors"])
    assert json.dumps(pipeline.states) == before
    assert pipeline.cursor == 1


def test_exhausted_replay_restore_invalidates_only_affected_candidate():
    pipeline = object.__new__(PremisePipeline)
    pipeline.replay_retries = 2
    pipeline.replay_workers = 2
    pipeline.benchmark_slots = {"terminalbench2": threading.BoundedSemaphore(2)}
    pipeline._replay_lock = threading.Lock()
    pipeline._active_replays = pipeline._peak_replays = 0
    pipeline.llm = SimpleNamespace(judge=lambda *_: {"score": 1.0})
    calls = []

    def run_agent(**kwargs):
        calls.append(kwargs["label"])
        if kwargs["playbook"]["broken"]:
            raise RuntimeError("execution failed: ValueError: Terminus-2 ended before the replay prefix was restored")
        return {"score": 0.5}

    pipeline._run_agent = run_agent
    reflection = {"span": {"start": 1}}
    candidates = [{"id": name, "parent_rank": 1, "reflection": reflection,
                   "playbook": {"broken": broken}, "valid": True}
                  for name, broken in [("bad", True), ("good", False)]]
    pipeline._score_candidates(1, candidates, [{"benchmark": "terminalbench2", "task_id": "task"}], [])
    assert not candidates[0]["valid"]
    assert "official_score" not in candidates[0]
    assert candidates[1]["valid"] and candidates[1]["current_official_score"] == 0.5
    assert sum("bad" in label for label in calls) == 3
    assert pipeline._active_replays == 0
