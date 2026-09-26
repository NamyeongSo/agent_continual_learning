# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from exgentic.agents.premise.llm import PremiseLLM
from exgentic.agents.premise.pipeline import PremisePipeline


def _trace():
    return {
        "benchmark": "bfcl", "task": "Do the task", "context": {},
        "actions": [{"action": {"name": "act"}, "observation": "ok"}],
        "score": 0.0, "details": {},
    }


def test_reflection_prompt_specifies_real_bounds_json_and_no_thinking():
    llm = object.__new__(PremiseLLM)
    captured = {}

    def ask(prompt, *, temperature=0.0):
        captured.update(prompt=prompt, temperature=temperature)
        return {
            "span": {"start": 0, "end": 1, "type": "failure"},
            "analysis": "The action failed", "expected_behavior": "Try a valid tool",
            "entry_assessments": [],
        }

    llm.ask = ask
    result = llm.reflect(_trace(), {"entries": []}, correction="Invalid reflection span [-1, -1)")
    assert result["span"] == {"start": 0, "end": 1, "type": "failure"}
    assert "0 <= start < end <= 1" in captured["prompt"]
    assert "Few-shot example A" in captured["prompt"]
    assert "Few-shot example B" in captured["prompt"]
    assert "actions [0, start)" in captured["prompt"]
    assert "Observations and dialogue turns do not get separate indices" in captured["prompt"]
    assert "Previous output was rejected" in captured["prompt"]
    assert "no Markdown fences" in captured["prompt"]
    assert captured["temperature"] == 0.0


def test_reflection_rejects_empty_action_trace_without_model_call():
    llm = object.__new__(PremiseLLM)
    with pytest.raises(ValueError, match="empty action trace"):
        llm.reflect({**_trace(), "actions": []}, {"entries": []})


def test_reflection_rejects_out_of_bounds_and_invalid_type():
    llm = object.__new__(PremiseLLM)
    base = {
        "analysis": "Visible failure", "expected_behavior": "Try again",
        "entry_assessments": [],
    }
    llm.ask = lambda *_args, **_kwargs: {
        **base, "span": {"start": -1, "end": 0, "type": "failure"}
    }
    with pytest.raises(ValueError, match="Invalid reflection span"):
        llm.reflect(_trace(), {"entries": []})
    llm.ask = lambda *_args, **_kwargs: {
        **base, "span": {"start": 0, "end": 1, "type": "unknown"}
    }
    with pytest.raises(ValueError, match="span type"):
        llm.reflect(_trace(), {"entries": []})


def test_reflection_retry_includes_previous_validation_error():
    pipeline = object.__new__(PremisePipeline)
    seen = []

    def reflect(_trace_arg, _playbook, *, correction=None):
        seen.append(correction)
        if len(seen) == 1:
            raise ValueError("Invalid reflection span [-1, -1)")
        return {"span": {"start": 0, "end": 1}}

    pipeline.llm = SimpleNamespace(reflect=reflect)
    result = pipeline._one_reflection(1, 1, _trace(), {"entries": []})
    assert seen == [None, "Invalid reflection span [-1, -1)"]
    assert result["id"] == "reflection_1"


def test_analysis_prompts_preserve_full_observation_context_and_feedback():
    llm = object.__new__(PremiseLLM)
    prompts = []
    responses = iter([
        {
            "span": {"start": 0, "end": 1, "type": "failure"},
            "analysis": "Visible failure", "expected_behavior": "Try again",
            "entry_assessments": [],
        },
        {"operations": [], "mutation_summary": "No justified change"},
        {"score": 0, "label": "not_aligned", "rationale": "No change"},
    ])

    def ask(prompt, *, temperature=0.0):
        prompts.append(prompt)
        return next(responses)

    llm.ask = ask
    trace = {
        **_trace(),
        "context": {"note": "c" * 4000 + "CONTEXT_END"},
        "actions": [{"action": {"name": "act"}, "observation": "o" * 700 + "OBSERVATION_END"}],
        "details": {"note": "f" * 6000 + "FEEDBACK_END"},
    }
    reflection = llm.reflect(trace, {"entries": []})
    llm.mutate(trace, {"entries": []}, reflection)
    llm.judge(trace, reflection, {"actions": trace["actions"]})

    assert "CONTEXT_END" in prompts[0]
    assert "FEEDBACK_END" in prompts[0]
    assert "FEEDBACK_END" in prompts[1]
    assert all("OBSERVATION_END" in prompt for prompt in prompts)
    assert all("[TRUNCATED]" not in prompt for prompt in prompts)
