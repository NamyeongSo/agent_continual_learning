# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json
import logging
from types import SimpleNamespace

import pytest

from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.agents.ace.playbook_store import PlaybookStore
from exgentic.core.orchestrator.termination import BenchmarkError
from exgentic.core.types import ModelSettings
from exgentic.utils.context_errors import is_context_length_error


@pytest.mark.parametrize("error", [
    ValueError("maximum context length is 131072 tokens"),
    BenchmarkError(ValueError("context_length_exceeded")),
    RuntimeError("ContextLengthExceededError: input too large"),
])
def test_detects_context_errors_through_wrappers(error):
    assert is_context_length_error(error)


@pytest.mark.parametrize("message", ["Argument list too long", "rate limit exceeded", "JSON parse failed"])
def test_does_not_treat_other_failures_as_context_errors(message):
    assert not is_context_length_error(ValueError(message))


@pytest.mark.parametrize("stage", ["generator", "reflector", "curator"])
def test_context_error_skips_task_and_restores_playbook(tmp_path, stage):
    instance = ACEAgentInstance("test-session", benchmark_id="terminalbench2", training_time=True)
    instance._paths = SimpleNamespace(agent_dir=tmp_path)
    instance._logger = logging.getLogger(__name__)
    instance.context = {"task_id": "task"}
    instance._store = PlaybookStore("ace_sequential_global", "original playbook")
    instance._store.increment_session()
    instance._observation_log = [{"content": "terminal output"}]
    instance._log_bullet_usage = lambda _ids: None
    calls = []

    def reflect():
        calls.append("reflector")
        instance._store.playbook = "changed tags"
        if stage == "reflector":
            raise ValueError("maximum context length exceeded")
        return "reflection"

    def curate(_reflection):
        calls.append("curator")
        instance._store.playbook = "partial curator changes"
        instance._store.next_global_id = 99
        raise ValueError("context_length_exceeded")

    instance._run_post_session_reflection = reflect
    instance._run_curator = curate
    if stage == "generator":
        instance.receive_training_feedback({"session_metadata": {
            "skip_reason": "context_length_exceeded", "error": "generator context overflow",
        }})
    instance.close()
    assert calls == {"generator": [], "reflector": ["reflector"], "curator": ["reflector", "curator"]}[stage]
    assert instance._store.playbook == "original playbook"
    assert instance._store.next_global_id == 1
    skip = json.loads((tmp_path / "task_skip.json").read_text())
    assert skip["stage"] == stage
    assert skip["playbook_updated"] is False
    checkpoint = json.loads((tmp_path / "playbook_checkpoint.json").read_text())
    assert checkpoint["session_count"] == 1
    assert checkpoint["benchmark_counts"] == {"terminalbench2": 1}


@pytest.mark.parametrize("method", ["generator", "simple"])
def test_context_errors_are_not_retried(monkeypatch, method):
    calls = []

    def fail(**_kwargs):
        calls.append(1)
        raise ValueError("maximum context length exceeded")

    monkeypatch.setattr("exgentic.agents.ace.ace_instance.litellm.completion", fail)
    instance = ACEAgentInstance("test-session")
    with pytest.raises(ValueError, match="context"):
        if method == "generator":
            instance._completion(model="openai/test", messages=[])
        else:
            instance._llm_call_simple("openai/test", "entire trace")
    assert len(calls) == 1


@pytest.mark.parametrize("json_mode", [False, True])
def test_reflector_and_curator_preserve_configured_reasoning_effort(monkeypatch, json_mode):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

    monkeypatch.setattr("exgentic.agents.ace.ace_instance.litellm.completion", complete)
    instance = ACEAgentInstance("test-session", enable_thinking=False,
                                model_settings=ModelSettings(max_tokens=8192, reasoning_effort="low"))
    assert instance._llm_call_simple("openai/test", "reflection or curation", json_mode=json_mode) == "{}"
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["max_tokens"] == 8192
    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
