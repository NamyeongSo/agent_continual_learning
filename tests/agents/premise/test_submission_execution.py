# SPDX-License-Identifier: Apache-2.0
"""Check the shared queue and frozen reporting without live models or environments."""

import hashlib
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from scripts.premise import run_submission_ablations as runner
from scripts.premise.ablation_proxy import make_handler, route
from scripts.premise.run_submission_ablations import cost_summary
from scripts.premise.submission_panels import digest, evaluate_contexts


def test_proxy_forwards_exact_bytes_and_bounds_all_jobs_in_one_queue(tmp_path):
    received = []
    concurrent = {"active": 0, "peak": 0}
    lock = threading.Lock()

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                received.append(body)
                concurrent["active"] += 1
                concurrent["peak"] = max(concurrent["peak"], concurrent["active"])
            time.sleep(0.025)
            data = json.dumps({"id": "reply", "usage": {"prompt_tokens": 7, "completion_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            with lock:
                concurrent["active"] -= 1

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(
        f"http://127.0.0.1:{upstream.server_port}", tmp_path, limit=2,
    ))
    for server in [upstream, proxy]:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    body = b'{ "model": "test", "temperature": 0.0, "messages": [{"role":"user","content":"test"}] }'

    def call(index):
        request = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_port}/jobs/job_{index}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(call, range(8))) == [200] * 8
        assert received == [body] * 8
        assert concurrent["peak"] == 2
        rows = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
        assert len(rows) == 8
        assert {row["job"] for row in rows} == {f"job_{i}" for i in range(8)}
        assert all(row["request_body_sha256"] == hashlib.sha256(body).hexdigest() for row in rows)
        assert all(row["temperature"] == 0 and row["usage"]["completion_tokens"] == 3 for row in rows)
    finally:
        for server in [proxy, upstream]:
            server.shutdown()
            server.server_close()


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/jobs/../v1/chat/completions", "/jobs/a.b/v1"])
def test_proxy_requires_an_explicit_safe_study_job(path):
    with pytest.raises(ValueError):
        route(path)


def test_frozen_eval_keeps_infrastructure_missing_and_uses_official_appworld_tgc(tmp_path):
    calls = []

    def actor(**kwargs):
        calls.append(kwargs["task_id"])
        if kwargs["task_id"] == "missing":
            raise ConnectionError("server unavailable")
        return {"score": 0.99999, "partial_score": 0.99999, "success": False, "session_dir": "saved"}

    result = evaluate_contexts(
        SimpleNamespace(_run_agent=actor), {"arm": {"playbook": "frozen"}},
        {"appworld": ["measured", "missing"]}, tmp_path,
    )
    assert not result["complete"]
    assert result["groups"][0]["measured"] == 1
    assert result["groups"][0]["score"] == 0
    assert result["groups"][0]["infrastructure_errors"] == 1
    assert calls.count("measured") == 1 and calls.count("missing") == 3
    assert any(json.loads(path.read_text())["score"] is None for path in (tmp_path / "records").glob("*.json"))


def test_frozen_eval_resumes_third_attempt_and_does_not_regrade_a_changed_policy(tmp_path):
    key = digest(json.dumps(["arm", "bfcl", "task"]).encode())[:24]
    record = {"arm": "arm", "benchmark": "bfcl", "task_id": "task", "status": "error", "attempt": 2,
              "policy_sha256": digest(b"frozen"), "score": None}
    (tmp_path / "records").mkdir()
    (tmp_path / "records" / f"{key}.json").write_text(json.dumps(record))
    calls = []

    def actor(**kwargs):
        calls.append(kwargs)
        return {"score": 1, "session_dir": "saved"}

    pipeline = SimpleNamespace(_run_agent=actor)
    result = evaluate_contexts(pipeline, {"arm": {"playbook": "frozen"}}, {"bfcl": ["task"]}, tmp_path)
    assert result["complete"]
    assert len(calls) == 1 and calls[0]["label"].endswith("_a3")
    with pytest.raises(ValueError, match="Frozen policy changed"):
        evaluate_contexts(pipeline, {"arm": {"playbook": "changed"}}, {"bfcl": ["task"]}, tmp_path)


def test_costs_separate_test_from_training_and_do_not_treat_unknown_usage_as_zero(tmp_path):
    calls = tmp_path / "run/train/llm_calls"
    calls.mkdir(parents=True)
    (calls / "judge.json").write_text(json.dumps({"stage": "judge", "response": {"id": "analysis"}}))
    base = {"path": "/v1/chat/completions", "health_probe": False, "request_seconds": 1.0}
    rows = [
        {**base, "job": "run__train", "response_id": "analysis",
         "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        {**base, "job": "run__test", "response_id": "actor",
         "usage": {"prompt_tokens": 200, "completion_tokens": 20}},
        {**base, "job": "run__test", "response_id": None, "usage": None},
    ]
    (tmp_path / "serving").mkdir()
    (tmp_path / "serving/usage.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    plan = {"output": str(tmp_path), "training_jobs": [{"id": "run", "output": str(tmp_path / "run"),
            "kind": "training"}], "panel_jobs": [], "cost_rates_usd_per_million": {"input": 0.05, "output": 0.70}}
    cost_summary(plan)
    result = json.loads((tmp_path / "cost_summary.json").read_text())
    assert result["unknown_usage_is_not_assumed_zero"] is True
    summary = {(row["phase"], row["category"]): row for row in result["rows"]}
    assert summary["train", "judge"]["input_tokens"] == 100
    assert summary["test", "actor"]["input_tokens"] == 200
    assert summary["test", "actor"]["unknown_usage_calls"] == 1


@pytest.mark.parametrize("contaminated", [False, True])
def test_bfcl_scope_stops_a_legacy_stream_before_appworld(tmp_path, monkeypatch, contaminated):
    root = tmp_path / "job"
    (root / "train").mkdir(parents=True)
    (root / "train/state.json").write_text("{}")
    order = [("bfcl", str(i)) for i in range(50)] + [("appworld", "later")]
    if contaminated:
        order[49] = ("appworld", "disallowed")
    pipeline = SimpleNamespace(cursor=12, order=order)
    calls = []

    def run(*, stop_after):
        calls.append(stop_after)
        pipeline.cursor = stop_after

    pipeline.run = run
    monkeypatch.setattr(runner, "pipeline_for", lambda *_: pipeline)
    job = {"id": "job", "output": str(root), "kind": "training"}
    plan = {"training_jobs": [job], "panel_jobs": [], "benchmarks": ["bfcl"], "train_stop_after": 50}
    if contaminated:
        with pytest.raises(ValueError, match="BFCL-only"):
            runner.work(plan, "job", "train")
        assert calls == []
    else:
        runner.work(plan, "job", "train")
        assert calls == [50]
        assert json.loads((root / "training_complete.json").read_text())["cursor"] == 50
