# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy
import json
from types import SimpleNamespace

import pytest

from exgentic.agents.litellm_tool_calling.instance import CompletionRetriesExhaustedError
from exgentic.agents.premise.pipeline import PremisePipeline, skip_task_after_llm_error, write_json


@pytest.mark.parametrize("stage", ["parent", "reflection", "mutation", "current_replay", "past_replay", "judge"])
@pytest.mark.parametrize("error", [
    CompletionRetriesExhaustedError("request timed out after recovery"),
    RuntimeError("session error: CompletionRetriesExhaustedError: 2 attempts; APITimeoutError"),
    RuntimeError("LiteLLM completion truncated (finish_reason=length)"),
])
def test_failed_llm_task_preserves_state_and_continues_to_next_task(tmp_path, stage, error):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"appworld_43": {"train": ["one", "two"]}})
    pipeline = PremisePipeline(
        output=tmp_path / "run", manifest=manifest, benchmarks=["appworld"], mode="sequential",
        seed=43, count=None, past_count=3, model="openai/test", base_url="http://localhost/v1",
    )
    trace = {"benchmark": "appworld", "task_id": "old", "actions": [], "score": 1, "details": {}}
    reflection = {"span": {"start": 0, "end": 1, "type": "success"}, "entry_assessments": []}
    pipeline.states["global"]["memory"].append({
        "memory_id": "old", "benchmark": "appworld", "task_id": "old", "trace": trace,
        "reflection": reflection,
    })
    before = copy.deepcopy(pipeline.states)
    attempts = []
    completed_next = []

    def fail_here(location):
        if pipeline.cursor == 0 and stage == location:
            raise error

    def run_agent(**kwargs):
        location = "parent" if kwargs.get("replay_source") is None else (
            "past_replay" if kwargs["task_id"] == "old" else "current_replay"
        )
        attempts.append(kwargs["label"])
        fail_here(location)
        return {**trace, "task_id": kwargs["task_id"]}

    def reflect(*_args, **_kwargs):
        fail_here("reflection")
        return copy.deepcopy(reflection)

    def mutate(*_args):
        fail_here("mutation")
        return {"operations": [{"type": "ADD", "section": "strategies_and_insights", "content": "Check results"}]}

    def judge(*_args):
        fail_here("judge")
        return {"score": 1}

    pipeline._run_agent = run_agent
    pipeline.llm = SimpleNamespace(reflect=reflect, mutate=mutate, judge=judge)
    original_step = pipeline.step

    def step():
        if pipeline.cursor == 1:
            # The first task failed despite other valid parents/candidates.
            assert pipeline.states == before
            artifact = json.loads((pipeline.output / "iterations/iter_0001.json").read_text())
            assert artifact["status"] == "skipped"
            assert not artifact["train_update_applied"]
            completed_next.append(pipeline.order[1])
        return original_step()

    pipeline.step = step
    pipeline.run()
    assert pipeline.cursor == 2
    assert completed_next == [pipeline.order[1]]
    assert (pipeline.output / "training_complete.json").exists()
    assert all("_try2" not in label and "_try3" not in label for label in attempts)
    assert pipeline.states != before  # The next task did train normally.


def test_native_provider_error_is_skipped_but_environment_errors_are_not_misclassified():
    assert skip_task_after_llm_error(RuntimeError("execution: litellm.Timeout: request timed out"))
    assert skip_task_after_llm_error(RuntimeError("execution: APIConnectionError"))
    assert not skip_task_after_llm_error(RuntimeError("Docker SSH session open refused by peer"))
    assert not skip_task_after_llm_error(RuntimeError("Terminal-Bench agent time limit reached"))


def test_none_policy_stops_on_truncated_completion_without_advancing(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"appworld_43": {"train": ["one"]}})
    pipeline = PremisePipeline(
        output=tmp_path / "run", manifest=manifest, benchmarks=["appworld"], mode="sequential",
        seed=43, count=None, past_count=3, model="openai/test", base_url="http://localhost/v1",
        skip_task_errors="none",
    )
    failure = RuntimeError("LiteLLM completion truncated (finish_reason=length)")
    pipeline.step = lambda: (_ for _ in ()).throw(failure)
    with pytest.raises(RuntimeError, match="completion truncated"):
        pipeline.run()
    assert pipeline.cursor == 0
    assert not (pipeline.output / "training_complete.json").exists()


def test_execution_tag_changes_session_ids_without_changing_checkpoint_configuration(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"bfcl_43": {"train": ["one"]}})
    options = {
        "output": tmp_path / "run", "manifest": manifest, "benchmarks": ["bfcl"], "mode": "sequential",
        "seed": 43, "count": None, "past_count": 3, "model": "openai/test", "base_url": "http://localhost/v1",
    }
    old = PremisePipeline(**options)
    saved = (old.output / "state.json").read_bytes()
    new = PremisePipeline(**options, execution_tag="skip_llm_errors_v1")
    assert new.run_namespace != old.run_namespace
    assert (new.output / "state.json").read_bytes() == saved
    assert new.states == old.states


@pytest.mark.parametrize("policy", ["none", "nontransient", "all"])
def test_tool_parser_configuration_error_stops_without_advancing_task(tmp_path, policy):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"bfcl_43": {"train": ["one", "two"]}})
    pipeline = PremisePipeline(
        output=tmp_path / "run", manifest=manifest, benchmarks=["bfcl"], mode="sequential",
        seed=43, count=None, past_count=3, model="openai/test", base_url="http://localhost/v1",
        skip_task_errors=policy,
    )
    before = (pipeline.output / "state.json").read_bytes()
    failure = RuntimeError(
        'CompletionRetriesExhaustedError: "auto" tool choice requires '
        "--enable-auto-tool-choice and --tool-call-parser to be set"
    )
    pipeline.step = lambda: (_ for _ in ()).throw(failure)
    assert not skip_task_after_llm_error(failure)
    with pytest.raises(RuntimeError, match="tool-call-parser"):
        pipeline.run()
    assert pipeline.cursor == 0
    assert (pipeline.output / "state.json").read_bytes() == before
    assert not (pipeline.output / "iterations/iter_0001.json").exists()
    assert not (pipeline.output / "training_complete.json").exists()
