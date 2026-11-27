# SPDX-License-Identifier: Apache-2.0
"""Public catalog filtering and the read-only evidence API."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from scripts.experiment_atlas.build import clean
from scripts.experiment_atlas.server import Atlas, make_handler, runtime_snapshot


@pytest.fixture
def atlas(tmp_path):
    settings = {
        "generated_at": "2026-11-11T00:00:00Z",
        "cohort": str(tmp_path),
        "runs": [],
        "sources": [{"id": "allowed", "text": "Frozen public source", "sha256": "recorded-sha"}],
    }
    tasks = [
        {
            "benchmark": "bfcl",
            "id": "train-1",
            "split": "train",
            "instruction": "First user instruction",
            "meta": {"variant": 1},
            "turns": [{"text": "First user instruction"}, {"text": "Unique future instruction"}],
        },
        {
            "benchmark": "bfcl",
            "id": "unused-1",
            "split": "unused",
            "instruction": "Unselected instruction",
            "meta": {},
            "turns": [],
        },
        {
            "benchmark": "appworld",
            "id": "test-1",
            "split": "test",
            "instruction": "Calendar request",
            "meta": {},
            "turns": [],
        },
    ]
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    (tmp_path / "tasks.json").write_text(json.dumps(tasks))
    return Atlas(tmp_path)


def test_search_includes_later_user_instructions_and_intersects_split_filters(atlas):
    assert atlas.catalog({"q": ["UNIQUE FUTURE"]})["items"][0]["id"] == "train-1"
    assert atlas.catalog({"q": ["future"], "split": ["test"]})["total"] == 0
    selected = atlas.catalog({"benchmark": ["bfcl"], "selected": ["selected"]})
    assert [t["id"] for t in selected["items"]] == ["train-1"]
    assert atlas.catalog({"page": ["malformed"]})["page"] == 1


def test_catalog_export_retains_all_user_turns(atlas):
    export = atlas.task_csv().decode("utf-8-sig")
    assert "First user instruction\n\nUnique future instruction" in export
    assert "Unselected instruction" in export


def test_private_runtime_fields_are_removed_without_changing_valid_zero_values(tmp_path):
    value = clean({"environment": {"DOCKER_HOST": "private-host"}, "api_key": "private-key", "score": 0})
    assert "environment" not in value
    assert value["api_key"] == "[비공개]"
    assert value["score"] == 0
    path = tmp_path / "status.json"
    path.write_text('{"phase":')
    assert runtime_snapshot(path)["read_error"] == "JSONDecodeError"


def test_http_routes_only_expose_registered_sources_and_do_not_mutate(atlas):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(atlas))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/source/allowed") as response:
            assert json.load(response)["text"] == "Frozen public source"
        with urllib.request.urlopen(base + "/api/task/bfcl/train-1") as response:
            assert len(json.load(response)["turns"]) == 2
        for path in ("/api/source/unknown", "/%2e%2e/settings.json", "/api/task/bfcl/missing"):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path)
            assert error.value.code == 404
        request = urllib.request.Request(base + "/api/settings", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
