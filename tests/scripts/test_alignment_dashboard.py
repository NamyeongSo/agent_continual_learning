# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Check audit evidence provenance and durable binary annotations through HTTP."""

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts.premise.serve_alignment_dashboard import (
    AXES,
    Application,
    Dataset,
    ReviewStore,
    digest,
    make_handler,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def trajectory(task_id):
    return {
        "benchmark": "bfcl", "task_id": task_id, "task": f"Request {task_id}", "score": 0,
        "actions": [{"action": {"name": f"original_{task_id}", "arguments": {"target": task_id}},
                     "observation": f"observed_{task_id}"}],
    }


def reflection(task_id):
    return {"span": {"start": 0, "end": 1, "type": "failure"},
            "analysis": f"analysis_{task_id}", "expected_behavior": f"expected_{task_id}"}


def result(task_id):
    return {"replay": trajectory(task_id), "official_score": 0,
            "alignment": {"score": 0.5, "label": "partial", "rationale": "partial evidence"}}


def artifact(iteration, task_id, past=None):
    return {
        "iteration": iteration, "benchmark": "bfcl", "task_id": task_id,
        "parents": [trajectory(task_id)],
        "candidates": [{
            "id": "candidate_1", "parent_rank": 1, "reflection": reflection(task_id), "valid": True,
            "applied": [{"type": "ADD"}], "current_result": result(task_id),
            "past_results": [result(past)] if past else [],
        }],
        "past_sample": [{"benchmark": "bfcl", "task_id": past}] if past else [],
        "selection": {"selected": ["candidate_1"]},
    }


@pytest.fixture
def app(tmp_path):
    outputs = tmp_path / "outputs"
    run = outputs / "example" / "s42"
    write_json(run / "iterations/iter_0001.json", artifact(1, "first"))
    write_json(run / "iterations/iter_0002.json", artifact(2, "second", "first"))
    return Application(outputs, tmp_path / "reviews.json")


def binary_review(value=1, checked=True):
    return {"axes": {axis["id"]: value for axis in AXES}, "checked": checked, "notes": "human evidence"}


def test_past_evidence_uses_accepted_original_reflection(app):
    dataset = app.dataset(app.catalog()["default_run"])
    row = next(row for row in dataset.rows if row["scope"] == "past")
    case = dataset.case(row["id"])
    assert case["task_id"] == "first"
    assert case["source"]["actions"][0]["action"]["name"] == "original_first"
    assert case["reflection"]["expected_behavior"] == "expected_first"
    assert case["source_artifact"] == "iterations/iter_0001.json"
    assert case["warnings"] == []


def test_missing_past_evidence_is_visible_and_never_uses_current_reflection(app):
    run = next(iter(app.runs.values()))
    write_json(run["path"] / "iterations/iter_0002.json", artifact(2, "second", "missing"))
    dataset = Dataset(run)
    row = next(row for row in dataset.rows if row["scope"] == "past")
    case = dataset.case(row["id"])
    assert row["source_available"] is False
    assert case["source"] is None
    assert case["reflection"] is None
    assert case["warnings"]


def test_zero_scores_and_partial_reviews_survive_store_restart(app):
    dataset = app.dataset(app.catalog()["default_run"])
    row = dataset.rows[0]
    review = binary_review(0, checked=False)
    review["axes"][AXES[0]["id"]] = None
    saved = app.store.save(row["id"], dataset.run, row, review)
    restored = ReviewStore(app.store.path).snapshot()["reviews"][row["id"]]
    assert restored == saved
    assert restored["axes"][AXES[1]["id"]] == 0
    assert restored["axes"][AXES[0]["id"]] is None
    assert restored["checked"] is False


@pytest.mark.parametrize("invalid", [0.5, 2, "1", True, False])
def test_axes_reject_nonbinary_values_without_writing(app, invalid):
    dataset = app.dataset(app.catalog()["default_run"])
    row = dataset.rows[0]
    review = binary_review()
    review["axes"][AXES[0]["id"]] = invalid
    with pytest.raises(ValueError, match="0, 1"):
        app.store.save(row["id"], dataset.run, row, review)
    assert not app.store.path.exists()


def test_cannot_check_incomplete_review(app):
    dataset = app.dataset(app.catalog()["default_run"])
    row = dataset.rows[0]
    review = binary_review()
    review["axes"][AXES[0]["id"]] = None
    with pytest.raises(ValueError, match="다섯 축"):
        app.store.save(row["id"], dataset.run, row, review)


def test_modified_evidence_does_not_inherit_old_review_ids(app):
    dataset = app.dataset(app.catalog()["default_run"])
    row = dataset.rows[0]
    path = dataset.run["path"] / row["artifact"]
    changed = artifact(1, "first")
    changed["candidates"][0]["current_result"]["alignment"]["rationale"] = "new evidence"
    write_json(path, changed)
    with pytest.raises(ValueError, match="변경"):
        dataset.case(row["id"])
    new = Dataset(dataset.run)
    assert row["id"] not in new.locations


def test_modified_past_source_invalidates_past_review_identity(app):
    dataset = app.dataset(app.catalog()["default_run"])
    row = next(row for row in dataset.rows if row["scope"] == "past")
    changed = artifact(1, "first")
    changed["parents"][0]["actions"][0]["observation"] = "repaired observation"
    write_json(dataset.run["path"] / "iterations/iter_0001.json", changed)
    with pytest.raises(ValueError, match="변경"):
        dataset.case(row["id"])
    assert row["id"] not in Dataset(dataset.run).locations


def test_corrupt_review_file_is_preserved(tmp_path):
    path = tmp_path / "reviews.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        ReviewStore(path)
    assert path.read_text() == "{broken"


def test_concurrent_saves_preserve_distinct_cases(app):
    dataset = app.dataset(app.catalog()["default_run"])
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(app.store.save, row["id"], dataset.run, row, binary_review(i % 2))
                   for i, row in enumerate(dataset.rows)]
        for future in futures:
            future.result()
    assert len(ReviewStore(app.store.path).snapshot()["reviews"]) == 3
    assert list(app.store.path.parent.glob(".reviews-*.tmp")) == []


@pytest.fixture
def live_server(app):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_http_save_export_reload_and_local_assets(app, live_server):
    run_id = app.catalog()["default_run"]
    dataset = app.dataset(run_id)
    row = dataset.rows[0]
    body = {"run": run_id, "id": row["id"], "review": binary_review(0)}
    request = Request(f"{live_server}/api/review", json.dumps(body).encode(),
                      {"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=5) as response:
        assert json.load(response)["review"]["checked"] is True
    with urlopen(f"{live_server}/api/run?run={run_id}", timeout=5) as response:
        assert json.load(response)["reviews"][row["id"]]["axes"][AXES[0]["id"]] == 0
    with urlopen(f"{live_server}/api/export", timeout=5) as response:
        assert row["id"] in json.load(response)["reviews"]
    for path in ("/", "/alignment_dashboard.js", "/alignment_dashboard.css"):
        with urlopen(live_server + path, timeout=5) as response:
            assert response.status == 200
            assert len(response.read()) > 100


def test_http_rejects_cross_origin_writes(app, live_server):
    run_id = app.catalog()["default_run"]
    row = app.dataset(run_id).rows[0]
    request = Request(f"{live_server}/api/review",
                      json.dumps({"run": run_id, "id": row["id"], "review": binary_review()}).encode(),
                      {"Content-Type": "application/json", "Origin": "https://unrelated.invalid"}, method="POST")
    with pytest.raises(HTTPError) as caught:
        urlopen(request, timeout=5)
    assert caught.value.code == 403
    assert not app.store.path.exists()


def test_discovery_ignores_non_alignment_runs_and_session_trees(tmp_path):
    outputs = tmp_path / "outputs"
    no_alignment = copy.deepcopy(artifact(1, "no_judge"))
    no_alignment["candidates"][0]["current_result"].pop("alignment")
    write_json(outputs / "ps_st/iterations/iter_0001.json", no_alignment)
    write_json(outputs / "hidden/sessions/iterations/iter_0001.json", artifact(1, "hidden"))
    app = Application(outputs, tmp_path / "reviews.json")
    assert app.catalog()["runs"] == []


def test_imported_state_memory_restores_missing_iteration_prefix(tmp_path):
    outputs = tmp_path / "outputs"
    root = outputs / "restart"
    write_json(root / "iterations/iter_0002.json", artifact(2, "second", "first"))
    write_json(root / "state.json", {"states": {"global": {"memory": [{
        "memory_id": "1|bfcl|first", "benchmark": "bfcl", "task_id": "first",
        "trace": trajectory("first"), "reflection": reflection("first"),
    }]}}})
    dataset = Application(outputs, tmp_path / "reviews.json").dataset(digest("restart")[:16])
    row = next(row for row in dataset.rows if row["scope"] == "past")
    assert dataset.case(row["id"])["reflection"] == reflection("first")
