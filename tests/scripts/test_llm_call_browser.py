# SPDX-License-Identifier: Apache-2.0
"""Stored reasoning, SDK repr decoding, and bounded read-only source access."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from scripts.experiment_atlas.llm_calls import CallBrowser, normalise, sdk_value
from scripts.experiment_atlas.server import Atlas, make_handler


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def browser(tmp_path):
    root = tmp_path / "cohort"
    jobs = [{"id": f"{method}_bfcl_s42", "method": method, "benchmark": "bfcl", "seed": 42,
             "output": str(root / method)} for method in ("ace", "premise", "gepa")]
    write(root / "launch_manifest.json", {"jobs": jobs, "inference_profile_sha256": "pinned-profile"})
    choice = (
        "Choices(finish_reason='tool_calls', index=0, message=Message(content=None, role='assistant', "
        "reasoning_content='Inspect names before changing files.\\nThen verify the result.', "
        "tool_calls=[ChatCompletionMessageToolCall(function=Function(name='find', "
        "arguments='{\"name\":\"test\"}'), id='tool-1', type='function')]))"
    )
    record = {"timestamp": "2026-11-12T00:00:00Z", "status": "success", "prompt_tokens": 11,
              "completion_tokens": 7, "request": {"temperature": 0.0, "api_key": "private-value",
              "messages": [{"role": "user", "content": "Find files"}],
              "tools": [{"properties": {"password": {"type": "string"}}}]},
              "response": {"id": "response-1", "choices": [choice]}}
    trace = root / "ace/train/run/sessions/task/agent/litellm/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps(record) + "\n")
    write(trace.parents[2] / "config.json", {"task_id": "task-1", "run_id": "first-run"})
    bridge = {"id": "response-1", "role": "actor", "temperature": 0.0, "reasoning_effort": "medium",
              "enable_thinking": True, "visible_limit": 2048, "reasoning_limit": 16384,
              "profile_sha256": "pinned-profile", "usage": {"prompt_tokens": 11, "completion_tokens": 7},
              "choices": [{"reasoning_tokens": 5, "visible_tokens": 2, "finish_reason": "stop"}]}
    (root / "serving").mkdir()
    (root / "serving/requests.jsonl").write_text(json.dumps(bridge) + "\n")
    return CallBrowser(root), root, trace, record


def test_sdk_reasoning_and_tool_calls_keep_content_and_observed_token_counts(browser):
    calls, _, _, _ = browser
    source = calls.catalog({"job": ["ace_bfcl_s42"]})["items"][0]
    listing = calls.records("ace_bfcl_s42", source["id"])
    assert listing["items"][0]["reasoning_tokens"] == 5
    value = calls.detail("ace_bfcl_s42", source["id"], 0)
    assert value["messages"][0]["reasoning"] == "Inspect names before changing files.\nThen verify the result."
    assert value["messages"][0]["tool_calls"][0]["function"]["name"] == "find"
    assert value["messages"][0]["finish_reason"] == "tool_calls"
    assert value["observed"]["choices"][0]["finish_reason"] == "stop"
    assert value["observed"]["reasoning_effort"] == "medium" and value["visible_tokens"] == 2
    assert "api_key" not in value["request"]
    assert value["request"]["tools"][0]["properties"]["password"] == {"type": "string"}


def test_sdk_repr_parser_never_executes_calls_or_attribute_access(tmp_path):
    marker = tmp_path / "must-not-exist"
    payload = f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')"
    assert sdk_value(payload) == payload
    assert not marker.exists()
    assert sdk_value("Message(**{'content':'bad'})") == "Message(**{'content':'bad'})"


def test_source_allowlist_excludes_private_files_and_escaping_symlinks(browser, tmp_path):
    calls, root, _, _ = browser
    write(root / "ace/private_environment.json", {"api_key": "secret"})
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"secret":"outside"}\n')
    link = root / "ace/train/escape/litellm/trace.jsonl"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    assert calls.catalog({"job": ["ace_bfcl_s42"]})["total"] == 1
    with pytest.raises(KeyError):
        calls.detail("ace_bfcl_s42", "../../private_environment.json", 0)
    with pytest.raises(KeyError):
        calls.catalog({"job": ["../../outside"]})


def test_partial_appended_records_become_visible_without_losing_complete_calls(browser):
    calls, _, trace, record = browser
    second = {**record, "response": {**record["response"], "id": "response-2"}}
    encoded = json.dumps(second)
    with trace.open("a") as stream:
        stream.write(encoded[:50])
    source = calls.catalog({"job": ["ace_bfcl_s42"]})["items"][0]["id"]
    assert len(calls.records("ace_bfcl_s42", source)["items"]) == 1
    with trace.open("a") as stream:
        stream.write(encoded[50:] + "\n")
    assert len(calls.records("ace_bfcl_s42", source)["items"]) == 2
    assert calls.detail("ace_bfcl_s42", source, 1)["request_id"] == "response-2"


def test_optimizer_effective_request_and_peace_json_are_distinct_modules(browser):
    calls, root, _, _ = browser
    folder = root / "gepa/train/analysis/proposal"
    write(folder / "request.json", {"stage": "instruction_proposal", "enable_thinking": False})
    write(folder / "effective_request.json", {"temperature": 0.7, "reasoning_effort": "medium",
                                             "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}})
    response = {"id": "proposal", "choices": [{"finish_reason": "stop", "message": {
        "reasoning_content": "Compare shared failure conditions.", "content": "Reusable instruction."}}],
        "usage": {"completion_tokens": 12, "completion_tokens_details": {"reasoning_tokens": 9}}}
    write(folder / "response.json", response)
    source = calls.catalog({"job": ["gepa_bfcl_s42"], "kind": ["analysis"]})["items"][0]
    value = calls.detail("gepa_bfcl_s42", source["id"], 0)
    assert value["module"] == "instruction_proposal"
    assert value["request"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert value["reasoning_tokens"] == 9 and value["visible_tokens"] == 3
    write(root / "premise/train/llm_calls/reflection.json", {"stage": "reflection", "status": "ok",
                                                          "request": {}, "response": response})
    source = calls.catalog({"job": ["premise_bfcl_s42"], "kind": ["analysis"]})["items"][0]
    assert calls.detail("premise_bfcl_s42", source["id"], 0)["module"] == "reflection"


def test_missing_reasoning_is_not_reported_as_zero_or_fabricated():
    value = normalise({"request": {}, "response": {"choices": [{"message": {"content": "Only answer"}}]}})
    enriched = CallBrowser.with_observation(value, {})
    assert enriched["messages"][0]["reasoning"] == ""
    assert enriched["reasoning_tokens"] is None


def test_native_terminal_actor_uses_the_session_task_identity(browser):
    calls, root, _, record = browser
    native = root / "ace/train/native/run/litellm/trace.jsonl"
    native.parent.mkdir(parents=True)
    native.write_text(json.dumps(record) + "\n")
    write(root / "ace/train/native/sessions/task/config.json", {"task_id": "build-pov-ray"})
    rows = calls.catalog({"job": ["ace_bfcl_s42"], "q": ["build-pov-ray"]})["items"]
    assert len(rows) == 1 and rows[0]["task_id"] == "build-pov-ray"


def test_http_browser_is_read_only_and_rejects_unregistered_sources(browser, tmp_path):
    calls, _, _, _ = browser
    write(tmp_path / "settings.json", {"sources": [], "runs": [], "generated_at": "now"})
    write(tmp_path / "tasks.json", [])
    atlas = Atlas(tmp_path)
    atlas.calls = calls
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(atlas))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/llm/jobs") as response:
            assert len(json.load(response)["jobs"]) == 3
        with urllib.request.urlopen(base + "/api/llm/sources?job=ace_bfcl_s42") as response:
            source = json.load(response)["items"][0]["id"]
        with urllib.request.urlopen(base + f"/api/llm/call?job=ace_bfcl_s42&source={source}&download=1") as response:
            assert response.headers["Content-Disposition"] == 'attachment; filename="llm-call.json"'
            assert json.load(response)["messages"][0]["reasoning"].startswith("Inspect")
        for path, code in [("/api/llm/call?job=ace_bfcl_s42&source=../../private_environment.json", 404),
                           ("/api/llm/sources?job=ace_bfcl_s42&page=invalid", 400)]:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path)
            assert error.value.code == code
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(base + "/api/llm/jobs", data=b"{}", method="POST"))
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
