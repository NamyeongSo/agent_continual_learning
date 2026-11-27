from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from exgentic.agents.premise.llm import (
    TERMINAL_REFLECTION_SYSTEM_PROMPT,
    PremiseLLM,
    public_tool_contract,
)
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

    def ask(prompt, *, temperature=0.0, stage=None):
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


@pytest.mark.parametrize("bad", [[], {"span": [{"start": 0, "end": 1}]},
                                  {"span": {"start": 0, "end": 1, "type": ["failure"]}}])
def test_shared_reflector_retries_malformed_nested_json_with_schema_feedback(monkeypatch, bad):
    import exgentic.inference_profile as inference

    monkeypatch.setattr(inference, "profile", lambda: {"reflection_variant": "instruction_aware"})
    llm = object.__new__(PremiseLLM)
    valid = {"span": {"start": 0, "end": 1, "type": "failure"}, "analysis": "Visible failure",
             "expected_behavior": "Inspect the observed failure", "entry_assessments": []}
    responses = iter([bad, valid])
    prompts = []

    def ask(prompt, **kwargs):
        prompts.append(prompt)
        return next(responses)

    llm.ask = ask
    pipeline = object.__new__(PremisePipeline)
    pipeline.llm = llm
    value = pipeline._one_reflection(1, 1, _trace(), {"playbook": ""})
    assert value["span"] == valid["span"] and len(prompts) == 2
    assert "Previous output rejected" in prompts[1]


def test_shared_reflector_malformed_shape_respects_three_attempt_limit(monkeypatch):
    import exgentic.inference_profile as inference

    monkeypatch.setattr(inference, "profile", lambda: {"reflection_variant": "instruction_aware"})
    llm = object.__new__(PremiseLLM)
    calls = []

    def ask(prompt, **kwargs):
        calls.append(prompt)
        return {"span": [0, 1]}

    llm.ask = ask
    pipeline = object.__new__(PremisePipeline)
    pipeline.llm = llm
    with pytest.raises(RuntimeError, match="reflection_1 failed"):
        pipeline._one_reflection(1, 1, _trace(), {"playbook": ""})
    assert len(calls) == 3


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

    def ask(prompt, *, temperature=0.0, stage=None):
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


def test_terminalbench_prompts_omit_duplicate_agent_trace_and_bound_long_actions():
    llm = object.__new__(PremiseLLM)
    prompts = []

    def ask(prompt, **_kwargs):
        prompts.append(prompt)
        return ({"span": {"start": 0, "end": 1, "type": "failure"},
                 "analysis": "A visible command failed", "expected_behavior": "Check the command output",
                 "entry_assessments": []} if len(prompts) == 1 else {"operations": []})

    llm.ask = ask
    trace = {
        **_trace(), "benchmark": "terminalbench2",
        "actions": [
            {"action": {"name": "terminus2_episode", "arguments": {
                "response": "COMMAND_START" + "a" * 90000 + "COMMAND_END",
            }},
             "observation": "OBS_START" + "b" * 90000 + "OBS_END"},
            {"action": {"name": "terminus2_episode", "arguments": {"response": "second command"}},
             "observation": "second result"},
        ],
        "details": {"session_metadata": {"agent_trace": "DUPLICATE_TRACE" * 10000,
                                         "harbor_rewards": {"reward": 0.0}}},
    }
    reflection = llm.reflect(trace, {"playbook": "Check results"})
    llm.mutate(trace, {"playbook": "Check results"}, reflection)
    for prompt in prompts:
        assert len(prompt) < 260000
        assert "DUPLICATE_TRACE" not in prompt
        assert "COMMAND_START" in prompt and "COMMAND_END" in prompt
        assert "OBS_START" in prompt and "OBS_END" in prompt
        assert "action[1]" in prompt and "second result" in prompt
        assert "recorded field omitted" in prompt
        assert '"reward": 0.0' in prompt
    assert "agent_trace" in trace["details"]["session_metadata"]


def test_terminalbench_reflection_keeps_one_copy_of_recorded_evidence():
    llm = object.__new__(PremiseLLM)
    captured = {}
    evidence = "VERIFIED_SOLUTION: all requested tests passed"
    trace = {
        **_trace(),
        "benchmark": "terminalbench2",
        "score": 1.0,
        "actions": [{
            "action": {"name": "terminus2_episode", "arguments": {"response": "run tests"}},
            "observation": {"results": [{"content": evidence}]},
        }],
        "details": {
            "score": 1.0,
            "session_metadata": {
                "agent_trace": json.dumps({"observation": evidence}),
                "harbor_rewards": {"reward": 1.0},
                "phase_timings": {"verifier": {"status": "ok"}},
            },
        },
    }
    original = json.dumps(trace, sort_keys=True)

    def ask(prompt, **_kwargs):
        captured["prompt"] = prompt
        return {
            "span": {"start": 0, "end": 1, "type": "success"},
            "analysis": "The requested tests passed",
            "expected_behavior": "Verify the requested solution",
            "entry_assessments": [],
        }

    llm.ask = ask
    llm.reflect(trace, {"playbook": "Check the test results"})

    assert captured["prompt"].count(evidence) == 1
    feedback = json.loads(captured["prompt"].split("Official feedback: ", 1)[1].split("\nCurrent playbook:", 1)[0])
    assert feedback == {
        "score": 1.0,
        "session_metadata": {
            "harbor_rewards": {"reward": 1.0},
            "phase_timings": {"verifier": {"status": "ok"}},
        },
    }
    assert json.dumps(trace, sort_keys=True) == original


def test_terminal_reflection_fits_token_budget_without_changing_episode_indices():
    llm = object.__new__(PremiseLLM)
    captured, measured = {}, []
    trace = {**_trace(), "benchmark": "terminalbench2", "actions": [
        {"action": {"name": "terminus2_episode", "arguments": {
            "response": f"COMMAND_{i}_START " + "x" * 25000 + f" COMMAND_{i}_END",
        }}, "observation": f"RESULT_{i}_START " + "y" * 25000 + f" RESULT_{i}_END"}
        for i in range(4)
    ]}
    original = json.dumps(trace, sort_keys=True)

    def tokens(prompt):
        measured.append(len(prompt))
        return len(prompt), 32768

    def ask(prompt, **kwargs):
        captured.update(prompt=prompt, **kwargs)
        return {"span": {"start": 2, "end": 3, "type": "failure"}, "analysis": "Action[2] failed",
                "expected_behavior": "Inspect the failing command", "entry_assessments": []}

    llm._terminal_reflection_tokens = tokens
    llm.ask = ask
    llm.reflect(trace, {"playbook": "Check observations"})
    assert len(measured) > 1
    assert measured[-1] <= 32768 - 8192 - 1024
    for i in range(4):
        assert f"### action[{i}]" in captured["prompt"]
        assert f"COMMAND_{i}_START" in captured["prompt"] and f"COMMAND_{i}_END" in captured["prompt"]
        assert f"RESULT_{i}_START" in captured["prompt"] and f"RESULT_{i}_END" in captured["prompt"]
    assert "recorded field omitted" in captured["prompt"]
    assert captured["system_prompt"] == TERMINAL_REFLECTION_SYSTEM_PROMPT
    assert "AppWorld" not in captured["prompt"]
    assert captured["response_schema"]["properties"]["span"]["properties"]["start"]["maximum"] == 3
    assert captured["response_schema"]["properties"]["span"]["properties"]["end"]["maximum"] == 4
    assert json.dumps(trace, sort_keys=True) == original


def test_terminal_reflection_preserves_large_full_fields_when_tokens_fit():
    llm = object.__new__(PremiseLLM)
    captured = {}
    evidence = "HEAD " + "x" * 90000 + " RELEVANT_MIDDLE " + "y" * 90000 + " TAIL"
    trace = {**_trace(), "benchmark": "terminalbench2", "actions": [
        {"action": {"name": "terminus2_episode", "arguments": {"response": evidence}}, "observation": "passed"},
    ]}
    llm._terminal_reflection_tokens = lambda _prompt: (60000, 131072)

    def ask(prompt, **_kwargs):
        captured["prompt"] = prompt
        return {"span": {"start": 0, "end": 1, "type": "success"}, "analysis": "The check passed",
                "expected_behavior": "Verify the change", "entry_assessments": []}

    llm.ask = ask
    llm.reflect(trace, {})
    assert evidence in captured["prompt"]
    assert "recorded field omitted" not in captured["prompt"]


@pytest.mark.parametrize("start,end", [(0.5, 1), (True, 1), (0, "1")])
def test_reflection_rejects_noninteger_span_instead_of_silently_changing_replay(start, end):
    llm = object.__new__(PremiseLLM)
    llm.ask = lambda *_args, **_kwargs: {
        "span": {"start": start, "end": end, "type": "failure"},
        "analysis": "Failed", "expected_behavior": "Check", "entry_assessments": [],
    }
    with pytest.raises(ValueError, match="indices must be integers"):
        llm.reflect(_trace(), {})


@pytest.mark.parametrize("legacy", [False, True])
def test_completion_contract_reaches_both_stages_despite_misleading_reflection(tmp_path, legacy):
    """Optional answer docs and later recovery survive span selection and old memories."""
    definitions = [
        {"name": "finish", "is_finish": True, "description": "End the AppWorld task", "arguments_schema": {
            "type": "object", "properties": {"answer": {
                "default": None, "description": "Pass iff the task requests an answer; otherwise keep None.",
            }},
        }},
        {"name": "spotify__create_playlist", "arguments_schema": {"properties": {"title": {"type": "string"}}}},
        {"name": "private_checker", "is_hidden": True, "description": "HIDDEN_CHECKER_SENTINEL"},
    ]
    trace = {
        **_trace(), "benchmark": "appworld", "task": "Create a playlist",
        "actions": [
            {"action": {"name": "supervisor__complete_task", "arguments": {"output": "done"}},
             "observation": "Unexpected output; allowed parameters answer,status"},
            {"action": {"name": "supervisor__complete_task", "arguments": {"answer": "done"}},
             "observation": "Marked the active task complete. LATER_RECOVERY_SENTINEL"},
        ], "details": {"session_metadata": {"test_tracker": {"failures": [{
            "requirement": "assert answers match", "trace": "PRIVATE_EXPECTED_ANSWER_SENTINEL",
        }]}}},
    }
    contract = public_tool_contract(definitions, trace["actions"])
    assert contract["available_action_names"] == ["finish", "spotify__create_playlist"]
    assert contract["definitions"] == [definitions[0]]
    if legacy:
        (tmp_path / "session.json").write_text(json.dumps({
            "actions": definitions, "private_evaluator": "HIDDEN_EVALUATOR_SENTINEL",
        }))
        trace["session_dir"] = str(tmp_path)
    else:
        trace["public_tool_contract"] = contract
    prompts = []
    wrong_reflection = {
        "span": {"start": 0, "end": 1, "type": "failure"},
        "analysis": "Wrong output argument", "expected_behavior": "Always pass answer via supervisor__finish_task",
        "entry_assessments": [],
    }
    llm = object.__new__(PremiseLLM)

    def ask(prompt, **_kwargs):
        prompts.append(prompt)
        return wrong_reflection if len(prompts) == 1 else {"operations": []}

    llm.ask = ask
    reflection = llm.reflect(trace, {"playbook": "Always submit a summary"})
    llm.mutate(trace, {"playbook": "Always submit a summary"}, reflection)
    for prompt in prompts:
        assert json.dumps(contract) in prompt
        assert "LATER_RECOVERY_SENTINEL" in prompt
        assert "HIDDEN_CHECKER_SENTINEL" not in prompt
        assert "HIDDEN_EVALUATOR_SENTINEL" not in prompt
        assert "PRIVATE_EXPECTED_ANSWER_SENTINEL" not in prompt
        assert "assert answers match" in prompt
    assert trace["details"]["session_metadata"]["test_tracker"]["failures"][0]["trace"] == (
        "PRIVATE_EXPECTED_ANSWER_SENTINEL"
    )
    assert "Independently verify" in prompts[1]
    assert "otherwise keep None" in prompts[1]
