# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy
import sys

import pytest

from scripts.utils import evaluate_isolated_checkpoint_queue as runner
from scripts.utils.evaluate_isolated_checkpoint_queue import (
    OutputLengthRetriesExhaustedError,
    completion_with_length_retries,
    queue_jobs,
)


@pytest.mark.parametrize("limit", [2048, 8192])
def test_output_length_failure_has_exactly_three_identical_calls_and_blocks_later_calls(limit):
    calls = []

    def complete(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return {"choices": [{"finish_reason": "length"}]}

    audit = {"max_tokens": limit}
    kwargs = {"messages": [{"role": "user", "content": "original request"}], "max_tokens": 2048}
    with pytest.raises(OutputLengthRetriesExhaustedError, match=f"3 attempts at max_tokens={limit}"):
        completion_with_length_retries(complete, kwargs, audit)
    with pytest.raises(OutputLengthRetriesExhaustedError, match="already exhausted"):
        completion_with_length_retries(complete, kwargs, audit)
    assert len(calls) == 3
    assert calls[0] == calls[1] == calls[2]
    assert calls[0]["max_tokens"] == limit
    assert calls[0]["max_retries"] == calls[0]["num_retries"] == 0
    assert audit["output_length_retries"] == 2
    assert audit["length_exhausted"]
    assert {event["max_tokens"] for event in audit["length_events"]} == {limit}


def test_output_length_can_recover_on_last_allowed_attempt():
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"finish_reason": "length" if len(calls) < 3 else "tool_calls"}]}

    audit = {}
    result = completion_with_length_retries(complete, {}, audit)
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert len(calls) == 3
    assert audit["output_length_retries"] == 2
    assert not audit.get("length_exhausted")


def test_other_errors_do_not_consume_output_length_retry_budget():
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("network timeout")

    with pytest.raises(TimeoutError):
        completion_with_length_retries(complete, {}, {})
    assert len(calls) == 1


@pytest.mark.parametrize("recover", [True, False])
def test_five_request_retries_are_independent_of_length_budget(recover):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 5 or not recover:
            raise TimeoutError("temporary request failure")
        return {"choices": [{"finish_reason": "stop"}]}

    audit = {"max_tokens": 8192, "transport_additional_retries": 5, "transport_retry_after": 0}
    if recover:
        assert completion_with_length_retries(complete, {}, audit)["choices"][0]["finish_reason"] == "stop"
    else:
        with pytest.raises(TimeoutError):
            completion_with_length_retries(complete, {}, audit)
    assert len(calls) == 6
    assert all(call == calls[0] and call["max_tokens"] == 8192 for call in calls)
    assert audit["transport_retries"] == 5
    assert not audit.get("length_exhausted") and not audit.get("output_length_retries")


def test_request_retry_budget_does_not_allow_extra_length_retries():
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("temporary request failure")
        return {"choices": [{"finish_reason": "length"}]}

    audit = {"max_tokens": 8192, "transport_additional_retries": 5, "transport_retry_after": 0}
    with pytest.raises(OutputLengthRetriesExhaustedError):
        completion_with_length_retries(complete, {}, audit)
    assert len(calls) == 4
    assert audit["transport_retries"] == 1
    assert audit["output_length_retries"] == 2 and audit["length_exhausted"]


def test_queue_mixes_groups_and_preserves_split_task_mapping_on_resume():
    groups = [{"group": "ace/train", "task_ids": [f"train-{i}" for i in range(50)]},
              {"group": "premise/test", "task_ids": [f"test-{i}" for i in range(50)]}]
    records = {group["group"]: [None] * 50 for group in groups}
    records["ace/train"][0] = {"status": "completed"}
    jobs = list(queue_jobs(groups, records))
    assert len(jobs) == 99
    assert [(job["group"], job["task_id"]) for job in jobs[:3]] == [
        ("premise/test", "test-0"), ("ace/train", "train-1"), ("premise/test", "test-1"),
    ]


def test_explicit_hold_prevents_cli_from_launching_workers(tmp_path, monkeypatch):
    runner.write_json(tmp_path / "automatic_resume_disabled.json", {"reason": "user requested hold"})
    runner.write_json(tmp_path / "status.json", {"phase": "paused", "processed_tasks": 1960})

    def unexpected_run(*args):
        raise AssertionError("Held experiment must not evaluate tasks")

    monkeypatch.setattr(runner, "run", unexpected_run)
    monkeypatch.setattr(sys, "argv", ["evaluation", "--output", str(tmp_path)])
    assert runner.main() == 0
    state = runner.read_json(tmp_path / "status.json")
    assert state["phase"] == "paused"
    assert state["processed_tasks"] == 1960
    assert state["automatic_resume_disabled"]
