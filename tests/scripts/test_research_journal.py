# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import base64
import csv
import io
import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from scripts.research_journal.reports import (
    backup_archive,
    markdown_archive,
    metric_matrix,
    report_csv,
    report_html,
    report_latex,
)
from scripts.research_journal.server import make_handler
from scripts.research_journal.store import JournalError, JournalStore


@pytest.fixture
def store(tmp_path):
    sources = tmp_path / "runs"
    sources.mkdir()
    return JournalStore(tmp_path / "journal", [sources])


@pytest.fixture
def experiment(store):
    return store.create(
        {
            "title": "UPDATE ablation",
            "purpose": "Compare matched ADD and UPDATE runs.",
            "seeds": "42,43,44",
            "metrics": [{"name": "score", "value": 0, "split": "test"}],
        }
    )


def test_notes_and_progress_survive_reopening_the_database(store, experiment):
    updated = store.update(
        experiment["id"], {"revision": experiment["revision"], "notes": "첫 관찰\n다음 질문 <script>alert(1)</script>"}
    )
    final = store.add_note(experiment["id"], {"text": "모든 seed에서 같은 조건을 확인했다."})
    reopened = JournalStore(store.data_dir, store.allowed_roots)
    assert reopened.get(experiment["id"]) == final
    assert final["notes"] == updated["notes"]
    assert final["metrics"][0]["value"] == 0
    assert final["log"][0]["text"] == "모든 seed에서 같은 조건을 확인했다."
    assert final["revision"] == 3


def test_stale_editor_cannot_overwrite_a_note_saved_in_another_tab(store, experiment):
    saved = store.update(experiment["id"], {"revision": 1, "notes": "Already saved"})
    with pytest.raises(JournalError) as error:
        store.update(experiment["id"], {"revision": 1, "notes": "Stale content"})
    assert error.value.status == 409
    assert store.get(experiment["id"]) == saved


def test_freeform_notes_preserve_indentation_and_trailing_newlines(store, experiment):
    note = "    indented code\n\n관찰한 내용\n"
    saved = store.update(experiment["id"], {"revision": 1, "notes": note})
    assert saved["notes"] == note
    assert JournalStore(store.data_dir).get(experiment["id"])["notes"] == note


def test_concurrent_progress_entries_are_not_lost(store, experiment):
    with ThreadPoolExecutor(max_workers=6) as workers:
        list(workers.map(lambda index: store.add_note(experiment["id"], {"text": f"observation {index}"}), range(15)))
    doc = store.get(experiment["id"])
    assert len(doc["log"]) == 15
    assert len({entry["text"] for entry in doc["log"]}) == 15
    assert doc["revision"] == 16


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "", "purpose": "required"},
        {"title": "valid", "purpose": ""},
        {"title": "valid", "purpose": "ok", "status": []},
        {"title": "valid", "purpose": "ok", "tags": "string"},
        {"title": "valid", "purpose": "ok", "metrics": [{"name": "score", "value": float("nan")}]},
        {"title": "valid", "purpose": "ok", "metrics": [{"name": "score", "value": True}]},
        {"title": "valid", "purpose": "ok", "metrics": [{"name": "score", "value": ""}]},
        {"title": "valid", "purpose": "ok", "metrics": [{"name": "score", "value": 0}, {"name": "score", "value": 1}]},
    ],
)
def test_invalid_records_are_rejected_without_writing(store, payload):
    with pytest.raises(JournalError):
        store.create(payload)
    assert store.list() == []


def test_import_preserves_a_hashed_snapshot_when_the_original_changes(store, experiment):
    source = store.allowed_roots[0] / "scores.csv"
    source.write_text("seed,score\n42,0\n", encoding="utf-8")
    doc = store.import_artifact(experiment["id"], {"path": str(source), "caption": "Frozen results"})
    artifact = doc["artifacts"][0]
    source.write_text("changed after attaching", encoding="utf-8")
    _, copy = store.artifact(experiment["id"], artifact["id"])
    assert copy.read_text() == "seed,score\n42,0\n"
    assert artifact["source_path"] == str(source)
    assert len(artifact["sha256"]) == 64
    assert artifact["caption"] == "Frozen results"
    store.change_artifact(experiment["id"], artifact["id"])
    assert store.get(experiment["id"])["artifacts"] == []
    assert source.read_text() == "changed after attaching"


def test_duplicate_upload_does_not_create_a_second_attachment(store, experiment):
    first = store.add_artifact(experiment["id"], "result.txt", b"one")
    second = store.add_artifact(experiment["id"], "result.txt", b"one")
    assert first == second
    assert len(list(store.blobs.iterdir())) == 1


def test_import_and_scan_reject_paths_outside_allowed_roots_and_hidden_files(store, experiment, tmp_path):
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    hidden = store.allowed_roots[0] / ".env"
    hidden.write_text("private")
    symlink = store.allowed_roots[0] / "escape.txt"
    symlink.symlink_to(outside)
    for path in (outside, hidden, symlink):
        with pytest.raises(JournalError) as error:
            store.import_artifact(experiment["id"], {"path": str(path)})
        assert error.value.status == 403
    (store.allowed_roots[0] / "figure.svg").write_text("<svg/>")
    store.update(experiment["id"], {"revision": 1, "run_path": str(store.allowed_roots[0])})
    assert [item["relative"] for item in store.scan_files(experiment["id"])["files"]] == ["figure.svg"]


def test_offline_html_includes_notes_images_csv_and_downloadable_attachments(store, experiment):
    doc = store.update(
        experiment["id"], {"revision": 1, "notes": "노트 <script>bad()</script>", "summary": "Score remains zero."}
    )
    image = b"<svg xmlns='http://www.w3.org/2000/svg'><text>figure</text></svg>"
    store.add_artifact(doc["id"], "figure.svg", image, "Graph caption")
    store.add_artifact(doc["id"], "scores.csv", b"seed,score\n42,0\n")
    doc = store.add_artifact(doc["id"], "appendix.pdf", b"%PDF-test")
    report = report_html(store, [doc]).decode()
    assert "노트 &lt;script&gt;bad()&lt;/script&gt;" in report
    assert "<script>bad()" not in report
    assert "data:image/svg+xml;base64," + base64.b64encode(image).decode() in report
    assert "<td>42</td><td>0</td>" in report
    assert 'download="appendix.pdf"' in report
    assert "http://127.0.0.1" not in report
    assert "Graph caption" in report


def test_exports_match_metrics_by_name_split_and_unit_without_inventing_missing_values(store, experiment):
    other = store.create(
        {
            "title": "Other",
            "purpose": "Different split",
            "metrics": [
                {"name": "score", "value": 0.5, "split": "val"},
                {"name": "score", "value": 50, "split": "test", "unit": "%"},
            ],
        }
    )
    _, rows = metric_matrix([experiment, other])
    assert rows == [["score", "test", "", 0, "—"], ["score", "val", "", "—", 0.5], ["score", "test", "%", "—", 50]]
    latex = report_latex([experiment, other]).decode()
    assert r"\%" in latex
    assert "0.5" in latex


def test_csv_preserves_zero_and_prevents_spreadsheet_formula_execution(store, experiment):
    doc = store.update(experiment["id"], {"revision": 1, "title": '=HYPERLINK("unsafe")'})
    rows = list(csv.DictReader(io.StringIO(report_csv([doc]).decode("utf-8-sig"))))
    assert rows[0]["title"].startswith("'=")
    assert rows[0]["metric_value"] == "0"


def test_markdown_archive_contains_relative_attachments(store, experiment):
    doc = store.add_artifact(experiment["id"], "한글 그림.svg", b"<svg/>")
    with closing(markdown_archive(store, [doc])) as stream, zipfile.ZipFile(stream) as archive:
        report = archive.read("report.md").decode()
        artifact_path = next(name for name in archive.namelist() if name.startswith("artifacts/"))
        assert artifact_path in report
        assert archive.read(artifact_path) == b"<svg/>"


def test_full_backup_restores_records_notes_archives_and_files(store, experiment, tmp_path):
    doc = store.add_artifact(experiment["id"], "figure.txt", b"stable artifact")
    doc = store.update(doc["id"], {"revision": doc["revision"], "notes": "노트", "status": "archived"})
    restored_dir = tmp_path / "restored"
    with closing(backup_archive(store)) as stream, zipfile.ZipFile(stream) as archive:
        assert json.loads(archive.read("experiments.json")) == [doc]
        archive.extractall(restored_dir)
    restored = JournalStore(restored_dir)
    assert restored.get(doc["id"]) == doc
    _, artifact_path = restored.artifact(doc["id"], doc["artifacts"][0]["id"])
    assert artifact_path.read_bytes() == b"stable artifact"


@pytest.fixture
def server(store):
    with ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, token="test-token")) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield httpd.server_address[1]
        httpd.shutdown()
        thread.join(timeout=5)


def request(port, path, method="GET", payload=None, headers=None):
    body = json.dumps(payload).encode() if payload is not None else None
    with closing(HTTPConnection("127.0.0.1", port, timeout=5)) as connection:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()


def test_http_record_note_export_and_csrf_protection(server):
    status, _, _ = request(server, "/api/experiments", "POST", {"title": "x", "purpose": "y"})
    assert status == 403
    token = {"X-Journal-Token": "test-token"}
    status, _, body = request(
        server, "/api/experiments", "POST", {"title": "HTTP test", "purpose": "Persist notes"}, token
    )
    assert status == 201
    doc = json.loads(body)
    status, _, body = request(
        server, f"/api/experiments/{doc['id']}", "PUT", {"revision": 1, "notes": "브라우저 노트"}, token
    )
    assert status == 200
    assert json.loads(body)["notes"] == "브라우저 노트"
    status, headers, body = request(server, f"/api/export?format=html&ids={doc['id']}")
    assert status == 200
    assert "attachment" in headers["Content-Disposition"]
    assert "브라우저 노트" in body.decode()
    assert request(server, "/api/experiments", headers={"Host": "attacker.example"})[0] == 403
    assert (
        request(
            server,
            f"/api/experiments/{doc['id']}",
            "PUT",
            {"revision": 2, "notes": "bad"},
            {**token, "Origin": "https://attacker.example"},
        )[0]
        == 403
    )


def test_http_errors_and_static_routes(server):
    status, _, body = request(server, "/api/live-progress")
    assert status == 200 and json.loads(body)["summary"]["runs"] == 0
    assert request(server, "/static/live.js")[0] == 200
    assert request(server, "/static/checkpoint_charts.js")[0] == 200
    assert request(server, "/static/live.css")[0] == 200
    assert request(server, "/")[0] == 200
    assert request(server, "/static/app.js")[1]["Content-Type"].startswith("text/javascript")
    assert request(server, "/static/../../.env")[0] == 404
    assert request(server, "/api/experiments/missing")[0] == 404
    assert request(server, "/api/export?format=invalid")[0] == 400
    assert request(server, "/api/experiments", "POST", [], {"X-Journal-Token": "test-token"})[0] == 400


def test_external_viewing_serves_the_monitor_without_tokens_or_write_access(store, experiment):
    handler = make_handler(store, token="test-token", allowed_hosts=("research.example",))

    class RemoteHandler(handler):
        def setup(self):
            super().setup()
            self.client_address = ("203.0.113.10", self.client_address[1])

    with ThreadingHTTPServer(("127.0.0.1", 0), RemoteHandler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        port = httpd.server_address[1]
        headers = {"Host": f"research.example:{port}"}
        try:
            assert request(port, "/", headers=headers)[0] == 200
            status, _, body = request(port, "/api/info", headers=headers)
            info = json.loads(body)
            assert status == 200 and info["read_only"] and info["csrf_token"] is None
            assert info["allowed_roots"] == [] and not info["data_dir"]
            assert request(port, "/api/live-progress", headers=headers)[0] == 200
            assert request(port, "/api/experiments", headers=headers)[0] == 200
            assert request(port, "/api/backup", headers=headers)[0] == 403
            assert request(port, f"/api/experiments/{experiment['id']}/files", headers=headers)[0] == 403
            # Spoofed loopback headers and a known token cannot change the real peer's viewing-only role.
            spoofed = {"Host": f"localhost:{port}", "X-Forwarded-For": "127.0.0.1",
                       "X-Journal-Token": "test-token"}
            assert request(port, "/api/info", headers=spoofed)[0] == 200
            assert request(port, "/api/experiments", "POST", {"title": "bad", "purpose": "bad"}, spoofed)[0] == 403
            assert len(store.list()) == 1
            assert request(port, "/", headers={"Host": f"other.example:{port}"})[0] == 403
        finally:
            httpd.shutdown()
            thread.join(timeout=5)


def test_allowed_hostnames_require_explicit_addresses_and_reject_url_wildcards():
    import argparse

    from scripts.research_journal.server import allowed_hostname

    assert allowed_hostname("Research.Example") == "research.example"
    assert allowed_hostname("192.0.2.5") == "192.0.2.5"
    for value in ("*", "http://example.com", "example.com:8878", "example.com/path", "user@example.com", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            allowed_hostname(value)


def test_public_binding_does_not_enumerate_interfaces_for_reverse_dns(store, monkeypatch):
    from scripts.research_journal.server import JournalHTTPServer

    monkeypatch.setattr("socket.getfqdn", lambda *_: pytest.fail("Public binding must not perform reverse DNS"))
    with JournalHTTPServer(("0.0.0.0", 0), make_handler(store)) as httpd:
        assert httpd.server_name == "0.0.0.0" and httpd.server_port > 0


def test_live_latex_export_is_current_read_only_and_rejects_unknown_tables(server, monkeypatch):
    from scripts.research_journal.result_tables import build_table

    config = {"id": "live", "title": "Live", "model": "Qwen", "seeds": [42, 43, 44],
              "split_manifest_sha256": "split", "sources": []}
    table = build_table(config, [], [])
    monkeypatch.setattr("scripts.research_journal.server.LiveProgress.snapshot",
                        lambda self: {"result_tables": [table]})
    status, headers, body = request(server, "/api/live-results.tex?id=live")
    assert status == 200 and "attachment" in headers["Content-Disposition"]
    assert "Offline-val" in body.decode() and "Isolated" in body.decode()
    assert request(server, "/api/live-results.tex?id=unknown")[0] == 404


def test_html_uploads_are_downloaded_with_an_inert_content_type(server, store, experiment):
    doc = store.add_artifact(experiment["id"], "untrusted.html", b"<script>alert(1)</script>")
    artifact_id = doc["artifacts"][0]["id"]
    status, headers, body = request(server, f"/api/experiments/{doc['id']}/artifacts/{artifact_id}/file")
    assert status == 200
    assert headers["Content-Type"] == "application/octet-stream"
    assert headers["Content-Security-Policy"].startswith("sandbox;")
    assert "attachment" in headers["Content-Disposition"]
    assert body == b"<script>alert(1)</script>"
