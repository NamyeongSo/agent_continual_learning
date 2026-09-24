# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import asyncio
import json

import pytest

from exgentic.core.context import Context, Role
from exgentic.integrations.litellm.trace_logger import (
    FILE_ENV,
    TRACE_FORMAT_ENV,
    TraceLogger,
    iter_expanded_trace_rows,
)
from exgentic.integrations.litellm.trace_cost import load_trace_cost


def _sample_payload() -> tuple[dict, dict]:
    kwargs = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_cost": 0.0,
    }
    response = {
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    }
    return kwargs, response


def test_trace_logger_writes(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(log_path))

    logger = TraceLogger()
    kwargs, response = _sample_payload()
    logger.log_success_event(kwargs, response, None, None)

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "success"
    assert record["model"] == "openai/gpt-4o-mini"


def test_trace_logger_writes_without_toggle(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(log_path))

    logger = TraceLogger()
    kwargs, response = _sample_payload()
    logger.log_success_event(kwargs, response, None, None)

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1


def test_compact_trace_reconstructs_history_without_repeating_it(tmp_path, monkeypatch) -> None:
    full_path = tmp_path / "full.jsonl"
    compact_path = tmp_path / "compact.jsonl"
    full_logger = TraceLogger(str(full_path))
    compact_logger = TraceLogger(str(compact_path))
    history = [{"role": "system", "content": "instructions " * 400}]
    tools = [{"type": "function", "function": {"name": "lookup", "description": "find " * 400}}]
    requests = []
    response = {
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }

    for turn in range(12):
        history.append({"role": "user", "content": f"observation {turn}"})
        kwargs = {
            "model": "openai/gpt-4o-mini",
            "messages": list(history),
            "tools": tools,
            "response_cost": 0.1,
        }
        requests.append({"messages": list(history), "tools": tools})
        full_logger.log_success_event(kwargs, response, None, None)
        monkeypatch.setenv(TRACE_FORMAT_ENV, "dedup_v1")
        compact_logger.log_success_event(kwargs, response, None, None)
        monkeypatch.delenv(TRACE_FORMAT_ENV)
        history.append({"role": "assistant", "content": f"answer {turn}"})

    expanded = list(iter_expanded_trace_rows(compact_path))
    assert len(expanded) == 12
    for row, expected in zip(expanded, requests):
        assert row["request"]["messages"] == expected["messages"]
        assert row["request"]["tools"] == expected["tools"]
    assert compact_path.stat().st_size < full_path.stat().st_size / 3
    assert load_trace_cost(compact_path, "openai/gpt-4o-mini") == pytest.approx(1.2)


def test_trace_logger_reads_context_from_metadata(tmp_path, monkeypatch) -> None:
    env_log_path = tmp_path / "env_trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(env_log_path))

    logger = TraceLogger()
    kwargs, response = _sample_payload()
    kwargs["litellm_metadata"] = {
        "context": Context(
            run_id="run_meta",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path / "cache"),
            session_id="sess_meta",
            role=Role.AGENT,
        )
    }
    logger.log_success_event(kwargs, response, None, None)

    expected = tmp_path / "run_meta" / "sessions" / "sess_meta" / "agent" / "litellm" / "trace.jsonl"
    assert expected.exists()
    assert not env_log_path.exists()


def test_trace_logger_reads_context_from_nested_metadata(tmp_path, monkeypatch) -> None:
    env_log_path = tmp_path / "env_trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(env_log_path))

    logger = TraceLogger()
    kwargs, response = _sample_payload()
    kwargs["litellm_params"] = {
        "litellm_metadata": {
            "context": Context(
                run_id="run_nested",
                output_dir=str(tmp_path),
                cache_dir=str(tmp_path / "cache"),
                session_id="sess_nested",
                role=Role.AGENT,
            )
        }
    }
    logger.log_success_event(kwargs, response, None, None)

    expected = tmp_path / "run_nested" / "sessions" / "sess_nested" / "agent" / "litellm" / "trace.jsonl"
    assert expected.exists()
    assert not env_log_path.exists()


def test_trace_logger_async_writes_with_kwargs_context(tmp_path, monkeypatch) -> None:
    env_log_path = tmp_path / "env_trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(env_log_path))

    logger = TraceLogger()
    kwargs, response = _sample_payload()
    kwargs["litellm_metadata"] = {
        "context": Context(
            run_id="run_async",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path / "cache"),
            session_id="sess_async",
            role=Role.AGENT,
        )
    }

    asyncio.run(logger.async_log_success_event(kwargs, response, None, None))

    expected = tmp_path / "run_async" / "sessions" / "sess_async" / "agent" / "litellm" / "trace.jsonl"
    assert expected.exists()
    lines = expected.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "success"
    assert not env_log_path.exists()


def test_trace_logger_uses_kwargs_context_for_log_path(tmp_path, monkeypatch) -> None:
    env_log_path = tmp_path / "env_trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(env_log_path))

    logger = TraceLogger()
    kwargs, response = _sample_payload()
    kwargs["context"] = Context(
        run_id="run_abc",
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        session_id="sess_123",
        role=Role.AGENT,
    )
    logger.log_success_event(kwargs, response, None, None)

    expected = tmp_path / "run_abc" / "sessions" / "sess_123" / "agent" / "litellm" / "trace.jsonl"
    assert expected.exists()
    assert not env_log_path.exists()


def test_trace_logger_get_context_falls_back_to_try_get_context(monkeypatch) -> None:
    fallback = Context(
        run_id="run_fallback",
        output_dir="/tmp/out",
        cache_dir="/tmp/cache",
        session_id="sess_fallback",
        role=Role.AGENT,
    )
    monkeypatch.setattr("exgentic.core.context.try_get_context", lambda: fallback)

    logger = TraceLogger()
    assert logger.get_context({}) == fallback
