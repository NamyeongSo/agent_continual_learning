# SPDX-License-Identifier: Apache-2.0
"""Backups preserve original bytes and advance only after complete Notion receipts."""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.utils import notion_experiment_backup as backup


def receipt(job):
    return {"snapshot_id": job["snapshot_id"], "parent_page_id": job["parent_page_id"], "page_id": "saved-page",
            "files": [{**row, "attached": True} for row in job["files"]]}


def extract_payload(job):
    chunks = []
    for item in job["files"]:
        if item["path"].endswith(".zip"):
            with zipfile.ZipFile(item["path"]) as archive:
                chunks.extend(archive.read(n) for n in sorted(archive.namelist())
                              if n.startswith("payload.tar.gz.part"))
    return tarfile.open(fileobj=io.BytesIO(b"".join(chunks)), mode="r:gz")


def test_full_incremental_and_tombstones_preserve_bytes(tmp_path):
    source, state = tmp_path / "experiment", tmp_path / "state"
    source.mkdir()
    (source / "result.json").write_bytes(b'{"pct":0.5}\n')
    (source / "context.txt").write_text("context v1")
    first = backup.snapshot(state, source, "notion-parent")
    assert first["kind"] == "full" and first["changed_count"] == 2
    with extract_payload(first) as archive:
        assert archive.extractfile("result.json").read() == b'{"pct":0.5}\n'
    backup.acknowledge(first, receipt(first))
    assert backup.snapshot(state, source, "notion-parent") is None
    (source / "context.txt").write_text("context v2")
    (source / "result.json").unlink()
    second = backup.snapshot(state, source, "notion-parent")
    assert second["kind"] == "incremental" and second["changed_count"] == second["deleted_count"] == 1
    manifest = backup.read(Path(second["directory"]) / "manifest.json")
    assert manifest["previous_snapshot_id"] == first["snapshot_id"]
    assert manifest["deleted_files"] == ["result.json"]
    with extract_payload(second) as archive:
        assert archive.getnames() == ["context.txt"]
        assert archive.extractfile("context.txt").read() == b"context v2"


def test_partial_or_wrong_parent_receipt_never_advances_watermark(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "results.csv").write_text("pct\n1\n")
    job = backup.snapshot(tmp_path / "state", source, "parent")
    incomplete = receipt(job)
    incomplete["files"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        backup.acknowledge(job, incomplete)
    assert not Path(job["watermark_path"]).exists()
    wrong = receipt(job)
    wrong["parent_page_id"] = "another-parent"
    with pytest.raises(ValueError, match="Incomplete"):
        backup.acknowledge(job, wrong)


def test_excludes_dependencies_credentials_and_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in [".env", "auth.json", "worker.lock", "id_rsa"]:
        (source / name).write_text("private")
    (source / "benchmark_dependencies").mkdir()
    (source / "benchmark_dependencies/installed.py").write_text("dependency")
    (source / "outside_link").symlink_to(tmp_path)
    (source / "result.json").write_text("{}")
    job = backup.snapshot(tmp_path / "state", source, "parent")
    with extract_payload(job) as archive:
        assert archive.getnames() == ["result.json"]


def test_large_payload_uses_individually_valid_zip_parts(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.bin").write_bytes(bytes(range(256)) * 10)
    monkeypatch.setattr(backup, "PART_BYTES", 100)
    job = backup.snapshot(tmp_path / "state", source, "parent")
    assert len(job["files"]) > 2
    with extract_payload(job) as archive:
        assert archive.extractfile("file.bin").read() == (source / "file.bin").read_bytes()


def test_failure_in_one_folder_does_not_skip_other_folder(tmp_path, monkeypatch):
    backup.write(tmp_path / "config.json", {"sources": [
        {"path": "experiment_a", "page_id": "a"}, {"path": "experiment_b", "page_id": "b"},
    ]})
    seen = []

    def fail_first(state, source, page):
        seen.append(source)
        if source == "experiment_a":
            raise RuntimeError("Notion unavailable for A")

    monkeypatch.setattr(backup, "snapshot", fail_first)
    with pytest.raises(RuntimeError, match="unavailable"):
        backup.cycle(tmp_path)
    assert seen == ["experiment_a", "experiment_b"]


def test_reused_initial_history_scope_excludes_obsolete_updates(tmp_path):
    source = tmp_path / "old_source"
    source.mkdir()
    backup.write(source / "stage1/task/history.json", {"actions": [1]})
    backup.write(source / "stage2/obsolete.json", {"reflection": "obsolete grading"})
    backup.write(source / "protocol.json", {"policy": "pinned"})
    job = backup.snapshot(tmp_path / "state", source, "parent", include_paths=["stage1", "protocol.json"])
    with extract_payload(job) as archive:
        assert set(archive.getnames()) == {"stage1/task/history.json", "protocol.json"}
