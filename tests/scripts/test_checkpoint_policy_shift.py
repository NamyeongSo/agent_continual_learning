# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Check agreement semantics, history preservation, the completion gate and actor budgets."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest

from scripts.utils import run_checkpoint_policy_shift as runner
from scripts.utils.policy_shift_metrics import (
    canonical,
    decode_record,
    digest,
    load_json,
    normalize_terminal,
    score_actions,
    splice_request,
    write_json,
)


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def test_accuracy_preserves_order_but_f1_preserves_duplicate_counts():
    reference = [call("a", value=1), call("b"), call("a", value=1)]
    predicted = [call("b"), call("a", value=1)]
    score = score_actions(predicted, reference, "bfcl")
    assert score["strict_accuracy"] == 0
    assert score["strict_f1"] == pytest.approx(0.8)
    score = score_actions([reference[1], reference[0], reference[2]], reference, "appworld")
    assert score["strict_f1"] == 1
    assert score["strict_accuracy"] == 0
    assert score_actions([call("a", value=2)], [call("a", value=1)], "bfcl") == {
        "loose_accuracy": 1,
        "loose_f1": 1,
        "strict_accuracy": 0,
        "strict_f1": 0,
    }


def test_strict_arguments_ignore_object_key_order_and_empty_action_is_valid():
    assert score_actions([call("f", a=1, b=2)], [call("f", b=2, a=1)], "appworld")["strict_accuracy"] == 1
    assert score_actions([], [call("f")], "bfcl")["strict_accuracy"] == 0
    assert score_actions([], [], "bfcl")["strict_f1"] == 1


def test_unchanged_root_reuses_evaluation_repeat_without_self_comparison(tmp_path):
    db = runner.own_db(tmp_path)
    reference, repeat = runner.add_comparison_jobs(db, "history", "root", "root", "settings")
    alias_reference, alias_prediction = runner.add_comparison_jobs(db, "history", "root", "root", "settings")
    _, changed_prediction = runner.add_comparison_jobs(db, "history", "new", "root", "settings")
    assert alias_prediction == repeat
    assert alias_reference == reference
    assert alias_prediction != alias_reference
    assert changed_prediction not in {reference, repeat}
    assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 3
    # A stochastic same-policy repeat can disagree; policy aliases must retain it.
    responses = {reference: [call("a")], repeat: [call("b")]}
    assert score_actions(responses[alias_prediction], responses[alias_reference], "bfcl")["strict_accuracy"] == 0


def test_existing_alias_repair_preserves_jobs_and_other_policy_comparisons(tmp_path):
    db = runner.own_db(tmp_path)
    for point, prompt in [(0, "root"), (5, "root"), (10, "new")]:
        db.execute("INSERT INTO policies VALUES('s',?,'bfcl','gepa',42,?)", (point, prompt))
    for key, prompt, sample in [("ref", "root", "primary"), ("repeat", "root", "repeat"), ("new", "new", "primary")]:
        db.execute("INSERT INTO jobs(id,anchor,prompt,sample) VALUES(?,?,?,?)", (key, "a", prompt, sample))
    for point, job in [(0, "repeat"), (5, "ref"), (10, "new")]:
        db.execute("INSERT INTO bindings VALUES('s',?,'a',?,'ref')", (point, job))
    db.commit()
    before = [tuple(r) for r in db.execute("SELECT * FROM jobs ORDER BY id")]
    result = runner.repair_root_alias_bindings(db)
    assert result["changed_bindings"] == 1
    assert dict(db.execute("SELECT point,job FROM bindings")) == {0: "repeat", 5: "repeat", 10: "new"}
    assert [tuple(r) for r in db.execute("SELECT * FROM jobs ORDER BY id")] == before
    assert runner.repair_root_alias_bindings(db)["changed_bindings"] == 0


def test_terminal_accuracy_compares_keystrokes_durations_and_completion_only():
    parser = SimpleNamespace(
        parse_response=lambda _: SimpleNamespace(
            error="", commands=[SimpleNamespace(keystrokes="ls\n", duration=0.5)], is_task_complete=False
        )
    )
    action = normalize_terminal({"content": "analysis A"}, parser)
    assert score_actions(action, normalize_terminal({"content": "analysis B"}, parser), "terminalbench2") == {
        "accuracy": 1
    }
    for modified in [{**action, "task_complete": True}, {"commands": [], "task_complete": False}]:
        assert score_actions(action, modified, "terminalbench2") == {"accuracy": 0}
    parser.parse_response = lambda _: SimpleNamespace(error="Invalid JSON")
    with pytest.raises(ValueError, match="Invalid JSON"):
        normalize_terminal({"content": "bad"}, parser)


def test_legacy_response_decoder_rejects_arbitrary_execution():
    assert decode_record("Message(content='ok', role='assistant')") == {"content": "ok", "role": "assistant"}
    with pytest.raises(ValueError):
        decode_record("__import__('os').system('true')")


@pytest.mark.parametrize("multimodal", [False, True])
def test_only_exact_policy_span_changes_even_for_terminal_user_messages(multimodal):
    content = "Original task\nOLD POLICY\nTerminal output"
    initial = [{"type": "text", "text": content}, {"type": "image_url", "image_url": {"url": "data:image/x"}}]
    anchor = {
        "messages": [
            {"role": "user", "content": initial if multimodal else content},
            {"role": "assistant", "content": "historical OLD POLICY"},
        ],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "policy_span": {
            "message": 0,
            "part": 0 if multimodal else None,
            "start": 14,
            "end": 24,
            "sha256": digest("OLD POLICY"),
        },
    }
    body = splice_request(anchor, "NEW POLICY", "openai/Qwen/test")
    changed = body["messages"][0]["content"]
    assert (changed[0]["text"] if multimodal else changed) == content.replace("OLD POLICY", "NEW POLICY")
    assert body["messages"][1] == anchor["messages"][1]
    assert body["tools"] == anchor["tools"]
    assert "OLD POLICY" in (anchor["messages"][0]["content"][0]["text"] if multimodal else content)


@pytest.fixture
def gate(tmp_path):
    db = sqlite3.connect(tmp_path / "queue.sqlite3")
    db.executescript("""
        CREATE TABLE checkpoints(id TEXT,stream TEXT,point INTEGER);
        CREATE TABLE tasks(cp TEXT,status TEXT);
        CREATE TABLE stream_invalidations(stream TEXT);
        INSERT INTO checkpoints VALUES('cp','s',5);
        INSERT INTO tasks VALUES('cp','completed');
    """)
    db.commit()
    write_json(tmp_path / "checkpoint_queue_manifest.json", {"selected_stream_ids": ["s"]})
    write_json(tmp_path / "dashboard.json", {"phase": "completed", "issues": []})
    write_json(tmp_path / "heartbeat.json", {"active": 0, "selected_stream_ids": ["s"]})
    return db, {"queue_output": str(tmp_path), "selected_stream_ids": ["s"], "minimum_checkpoint_points": {"s": [5]}}


def test_gate_accepts_only_successful_drained_scope(gate):
    db, plan = gate
    assert runner.checkpoint_gate(plan)["ready"]
    db.execute("INSERT INTO stream_invalidations VALUES('s')")
    db.commit()
    assert not runner.checkpoint_gate(plan)["ready"]


@pytest.mark.parametrize("status", ["pending", "running", "error"])
def test_gate_rejects_unfinished_and_exhausted_errors(gate, status):
    db, plan = gate
    db.execute("UPDATE tasks SET status=?", (status,))
    db.commit()
    assert not runner.checkpoint_gate(plan)["ready"]


def test_gate_rejects_paused_empty_queue_or_missing_checkpoints(gate):
    _, plan = gate
    write_json(runner.Path(plan["queue_output"]) / "dashboard.json", {"phase": "paused"})
    assert not runner.checkpoint_gate(plan)["ready"]
    write_json(runner.Path(plan["queue_output"]) / "dashboard.json", {"phase": "completed"})
    plan["minimum_checkpoint_points"]["s"].append(10)
    assert not runner.checkpoint_gate(plan)["ready"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,jobs", [(None, 5), ("input_budget", 1), ("truncated", 1), ("outage", 1)])
async def test_probe_enforces_budgets_without_resampling_measured_output(tmp_path, monkeypatch, failure, jobs):
    db = runner.own_db(tmp_path)
    anchor = {
        "benchmark": "bfcl",
        "messages": [{"role": "system", "content": "old"}],
        "tools": [],
        "policy_span": {"message": 0, "start": 0, "end": 3, "sha256": digest("old")},
    }
    write_json(tmp_path / "anchor.json", anchor)
    anchor_id, prompt_id = digest(canonical(anchor)), digest("new")
    db.execute("INSERT INTO anchors VALUES(?,'bfcl',42,'task',?)", (anchor_id, str(tmp_path / "anchor.json")))
    db.execute("INSERT INTO prompts VALUES(?,'new')", (prompt_id,))
    for i in range(jobs):
        db.execute("INSERT INTO jobs(id,anchor,prompt,sample) VALUES(?,?,?,'primary')", (f"j{i}", anchor_id, prompt_id))
    db.commit()
    monkeypatch.setattr(runner, "terminal_parser", lambda _: None)
    completed = []
    active = maximum_active = 0
    outage_observed = False

    async def handler(request):
        nonlocal active, maximum_active, outage_observed
        if request.method == "GET":
            if failure == "outage" and len(completed) == 3 and not outage_observed:
                outage_observed = True
                return httpx.Response(503)
            return httpx.Response(200, json={"data": [{"id": "Qwen/test"}]})
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            check = body["messages"][0]["content"] == "protocol check"
            return httpx.Response(200, json={"count": 100001 if failure == "input_budget" and not check else 10})
        completed.append(body)
        if failure == "outage" and len(completed) <= 3:
            raise httpx.ConnectError("model service unavailable", request=request)
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.005)
        active -= 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length" if failure == "truncated" else "stop",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    plan = {
        "workers": 2,
        "terminal_parser_path": "unused",
        "base_url": "http://test/v1",
        "model": "openai/Qwen/test",
        "temperature": 0.7,
        "input_tokens": 100000,
        "output_tokens": 2048,
    }
    await runner.probe(tmp_path, plan)
    row = db.execute("SELECT * FROM jobs").fetchone()
    successful = failure in (None, "outage")
    assert row["status"] == ("completed" if successful else "error")
    assert row["attempts"] == (4 if failure == "outage" else 1)
    assert len(completed) == (0 if failure == "input_budget" else 4 if failure == "outage" else jobs)
    if failure is None:
        assert maximum_active == 2
        assert db.execute("SELECT count(*) FROM jobs WHERE status='completed'").fetchone()[0] == jobs
    if completed:
        assert completed[0]["max_tokens"] == 2048
        assert completed[0]["temperature"] == 0.7
    response = load_json(tmp_path / "responses/j0.json")
    assert (response["error"] is None) if successful else response["error"] == row["error"]


def test_analysis_weights_tasks_equally_and_keeps_errors_out_of_agreement(tmp_path, monkeypatch):
    db = runner.own_db(tmp_path)
    db.execute("INSERT INTO policies VALUES('s',5,'bfcl','ace',42,'p')")
    for i in range(5):
        task = "short" if i == 0 else "long" if i < 4 else "missing"
        db.execute("INSERT INTO anchors VALUES(?,'bfcl',42,?,'unused')", (f"a{i}", task))
        db.execute("INSERT INTO jobs(id,status,action) VALUES(?,'completed',?)", (f"r{i}", json.dumps([call("f")])))
        db.execute(
            "INSERT INTO jobs(id,status,action) VALUES(?,?,?)",
            (
                f"j{i}",
                "error" if i == 4 else "completed",
                json.dumps([call("f" if i == 0 else "g")]),
            ),
        )
        db.execute("INSERT INTO bindings VALUES('s',5,?,?,?)", (f"a{i}", f"j{i}", f"r{i}"))
    db.commit()
    monkeypatch.setattr(runner, "plot", lambda *_: None)
    result = runner.analyze(tmp_path, {"workers": 128})
    strict = next(r for r in result["scores"] if r["metric"] == "strict_accuracy")
    assert strict["agreement"] == 0.5  # Mean(1 for short task, 0 for long task).
    assert result["common_histories"] == 4
    assert result["status"] == "completed_with_errors"
