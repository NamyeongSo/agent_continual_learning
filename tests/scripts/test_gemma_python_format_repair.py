# SPDX-License-Identifier: Apache-2.0
"""Verify format repair preserves code, native actions, raw logs and reasoning."""

import importlib.util
import json
from types import SimpleNamespace

import pytest
from litellm import ModelResponse

from scripts.ace import gemma_python_format_repair as repair
from scripts.ace import rerun_gemma_format_errors as rerun


def response(content, **fields):
    return ModelResponse(choices=[{"finish_reason": "stop", "message": {
        "role": "assistant", "content": content, **fields,
    }}], usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


@pytest.mark.parametrize("suffix", ["", "<tool_call|>", "<|tool_call|>"])
def test_extracts_exact_code_without_changing_whitespace(suffix):
    code = 'print("hello")\r\n\r\n'
    assert repair.extract_python(f"Explanation\r\n```python\r\n{code}```{suffix}\r\n") == (code, "repaired")


@pytest.mark.parametrize("text", [
    "I finished the task.", "```python\nprint(1)", "```csv\na,b\n```",
    "```python\nprint(1)\n```\n```python\nprint(2)\n```",
    "```python\n# Done\n```", "```python\nif (\n```", "```python\n\n```",
    "```python\nprint(1)\n```}<tool_call|>",
])
def test_does_not_invent_actions_for_ambiguous_noncode_or_incomplete_replies(text):
    assert repair.extract_python(text)[0] is None


def test_raw_response_untouched_and_repaired_history_keeps_tool_turn_reasoning():
    raw = response("```python\napis.supervisor.complete_task()\n```", reasoning_content="ORIGINAL_THOUGHT")
    before = raw.model_dump(mode="json")
    fixed, event = repair.repair_response(raw, step=2, tool_names=["execute_python"])
    assert raw.model_dump(mode="json") == before
    assert fixed.choices[0].finish_reason == "tool_calls"
    assert fixed.choices[0].message.reasoning_content == "ORIGINAL_THOUGHT"
    tool = fixed.choices[0].message.tool_calls[0]
    assert tool.function.name == "execute_python"
    assert json.loads(tool.function.arguments) == {"code": "apis.supervisor.complete_task()\n"}
    assert event["decision"] == "repaired"


def test_native_calls_take_precedence_even_when_content_contains_a_python_fence():
    raw = response("```python\nprint('DO NOT RUN')\n```", tool_calls=[{
        "id": "call_native", "type": "function", "function": {"name": "execute_python", "arguments": '{}'},
    }])
    fixed, event = repair.repair_response(raw, step=1, tool_names=["execute_python"])
    assert fixed is raw and event is None


def test_never_extracts_reasoning_or_repairs_a_truncated_output():
    raw = response("Done.", reasoning_content="```python\nprint('THOUGHT ONLY')\n```")
    fixed, event = repair.repair_response(raw, step=1, tool_names=["execute_python"])
    assert fixed is raw and event["decision"] != "repaired"
    raw = response("```python\nprint(1)\n```")
    raw.choices[0].finish_reason = "length"
    assert repair.repair_response(raw, step=1, tool_names=["execute_python"]) == (raw, None)
    assert repair.repair_response(raw, step=1, tool_names=["other_tool"]) == (raw, None)


def test_scoped_install_writes_a_separate_audit_log_after_raw_completion(tmp_path):
    raw = response("```py\nprint(1)\n```")

    class Actor:
        _step_count = 1
        paths = SimpleNamespace(agent_dir=tmp_path)

        def _completion(self, **kwargs):
            (tmp_path / "raw.json").write_text(json.dumps(raw.model_dump(mode="json")))
            return raw

    original = repair.install(Actor)
    try:
        fixed = Actor()._completion(tools=[{"function": {"name": "execute_python"}}])
        assert not json.loads((tmp_path / "raw.json").read_text())["choices"][0]["message"].get("tool_calls")
        assert fixed.choices[0].message.tool_calls
        assert json.loads((tmp_path / "format_repair.jsonl").read_text())["decision"] == "repaired"
    finally:
        Actor._completion = original


def test_real_frozen_actor_dispatches_repaired_code_and_keeps_raw_response(tmp_path):
    path = (rerun.ORIGINAL_31 / "thinking/accumulation/source_snapshot"
            "/src/exgentic/agents/ace/ace_instance.py")
    if not path.exists():
        pytest.skip("Requires the local frozen Gemma31 actor")
    spec = importlib.util.spec_from_file_location("exgentic.agents.ace.regex_dispatch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actor = module.ACEAgentInstance.__new__(module.ACEAgentInstance)
    actor._step_count, actor._all_actions, actor._action_log, actor.messages = 0, [], [], []
    actor.model, actor._task_context_error = "openai/google/gemma-4-31B-it", None
    actor._observe = actor._log_observation = lambda *_: None
    actor._assistant_tools = lambda: [{"type": "function", "function": {"name": "execute_python"}}]
    actor._add_message = actor.messages.append
    actor._registry = SimpleNamespace(tool_calls_to_action=lambda calls: calls)
    actor._logger = SimpleNamespace(info=lambda *args: None)
    actor._paths = SimpleNamespace(agent_dir=tmp_path)
    actor._cost = SimpleNamespace(update_cost_from_tokens=lambda *args: None)
    raw = response("```python\nprint(1)\n```<tool_call|>", reasoning_content="TOOL_TURN_THOUGHT")
    original_method = module.ACEAgentInstance._completion
    module.ACEAgentInstance._completion = lambda self, **kwargs: raw
    repair.install(module.ACEAgentInstance)
    try:
        calls = actor.react(None)
        assert calls[0]["name"] == "execute_python"
        assert json.loads(calls[0]["arguments"])["code"] == "print(1)\n"
        assert actor._action_log[-1]["action"] == "execute_python"
        assert actor.messages[-1]["reasoning_content"] == "TOOL_TURN_THOUGHT"
        assert not raw.choices[0].message.tool_calls
    finally:
        module.ACEAgentInstance._completion = original_method


def test_server_switch_waits_for_complete_31b_results_before_signalling_anything(tmp_path, monkeypatch):
    (tmp_path / "31b").mkdir()
    (tmp_path / "31b/status.json").write_text('{"state": "running"}')
    monkeypatch.setattr(rerun, "endpoint", lambda _: {"data": [{"id": "google/gemma-4-31B-it"}]})
    monkeypatch.setattr(rerun, "serving_process", lambda: {
        "pid": 1234, "args": ["vllm", "serve", "google/gemma-4-31B-it"],
    })
    signalled = []
    monkeypatch.setattr(rerun.os, "kill", lambda *args: signalled.append(args))
    with pytest.raises(ValueError, match="completion barrier"):
        rerun.ensure_server(tmp_path, "26b", {"model": "openai/google/gemma-4-26B-A4B-it"})
    assert not signalled
