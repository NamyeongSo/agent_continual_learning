# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy
from types import SimpleNamespace

from scripts.utils import serve_vllm_tool_parser_bridge as bridge


def test_bridge_applies_selected_native_parser_and_preserves_truncation(monkeypatch):
    seen = []

    class FakeParser:
        def extract_tool_calls(self, content, request):
            seen.append((content, request))
            return SimpleNamespace(tools_called=True, content=None, tool_calls=[SimpleNamespace(
                model_dump=lambda: {"type": "function", "function": {"name": "ping", "arguments": '{"x":1}'}}
            )])

    def factory(name, tools):
        assert name == "gemma4" and tools == ["frozen-tool"]
        return FakeParser()

    monkeypatch.setattr(bridge, "tool_parser", factory)
    request = SimpleNamespace(tools=["frozen-tool"])
    response = {"choices": [{"finish_reason": "stop", "message": {"content": "native Gemma tool tokens"}},
                            {"finish_reason": "length", "message": {"content": "truncated"}}]}
    result = bridge.parse_response(copy.deepcopy(response), request, "gemma4")
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ping"
    assert result["choices"][1] == response["choices"][1]
    assert len(seen) == 1


def test_text_requests_pass_through_and_both_vocabularies_preserve_sentinels():
    response = {"choices": [{"finish_reason": "stop", "message": {"content": "plain text"}}]}
    assert bridge.parse_response(response, None) is response
    vocab = bridge.ParsingTokenizer().get_vocab()
    assert all(key in vocab for key in ("<tool_call>", "</tool_call>", "<|tool_call>", "<tool_call|>"))


def test_harmony_bridge_recovers_hidden_calls_from_tokens_without_changing_reasoning_or_length(monkeypatch):
    class FakeHarmony:
        def parse(self, text, request, **kwargs):
            assert text == "" and kwargs["model_output_token_ids"] == [1, 2, 3]
            assert kwargs["enable_auto_tools"] is True
            return None, None, [SimpleNamespace(model_dump=lambda: {"name": "ping", "arguments": '{"x":1}'})]

    monkeypatch.setattr(bridge, "harmony_parser", lambda _tools: FakeHarmony())
    request = SimpleNamespace(tools=["ping"], return_token_ids=False)
    response = {"choices": [{"finish_reason": "stop", "token_ids": [1, 2, 3],
                             "message": {"content": None, "reasoning": "preserved", "tool_calls": []}},
                            {"finish_reason": "length", "message": {"content": "truncated"}}]}
    parsed = bridge.parse_response(copy.deepcopy(response), request, "gpt_oss")
    first = parsed["choices"][0]
    assert first["finish_reason"] == "tool_calls"
    assert first["message"]["tool_calls"][0]["function"] == {"name": "ping", "arguments": '{"x":1}'}
    assert first["message"]["reasoning"] == "preserved"
    assert "token_ids" not in first
    assert parsed["choices"][1] == response["choices"][1]
