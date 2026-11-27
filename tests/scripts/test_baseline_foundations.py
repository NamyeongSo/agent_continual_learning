# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import hashlib
import subprocess
import tomllib

import pytest

from scripts.baselines import prepare


def git(folder, *args):
    return subprocess.check_output(["git", "-C", str(folder), *args], text=True, stderr=subprocess.PIPE).strip()


@pytest.fixture
def origin(tmp_path, monkeypatch):
    folder = tmp_path / "origin"
    folder.mkdir()
    git(folder, "init")
    (folder / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="1"\n')
    git(folder, "add", "pyproject.toml")
    git(folder, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "initial")
    first = git(folder, "rev-parse", "HEAD")
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    return {"url": str(folder), "commit": first, "path": "sources/fixture"}


def test_materializes_a_no_checkout_clone_at_the_pinned_commit(origin, tmp_path):
    prepare.ensure_source(origin)
    status = prepare.source_status(origin)
    assert status["ready"]
    assert status["commit"] == origin["commit"]
    assert (tmp_path / origin["path"] / "pyproject.toml").is_file()


def test_dirty_upstream_is_preserved_even_when_pyproject_was_deleted(origin, tmp_path):
    prepare.ensure_source(origin)
    path = tmp_path / origin["path"]
    (path / "pyproject.toml").unlink()
    (path / "local-work.txt").write_text("Keep this local change")
    with pytest.raises(ValueError, match="local changes"):
        prepare.ensure_source(origin)
    assert (path / "local-work.txt").read_text() == "Keep this local change"
    assert not (path / "pyproject.toml").exists()


def test_source_with_an_unexpected_origin_is_not_repurposed(origin, tmp_path):
    prepare.ensure_source(origin)
    path = tmp_path / origin["path"]
    git(path, "remote", "set-url", "origin", "https://example.invalid/other.git")
    with pytest.raises(ValueError, match="origin differs"):
        prepare.ensure_source(origin)
    assert git(path, "remote", "get-url", "origin") == "https://example.invalid/other.git"
    assert git(path, "rev-parse", "HEAD") == origin["commit"]


def test_clean_checkout_can_return_to_the_exact_pin(origin, tmp_path):
    folder = tmp_path / "origin"
    (folder / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="2"\n')
    git(folder, "add", "pyproject.toml")
    git(folder, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "next")
    prepare.ensure_source(origin)
    checkout = tmp_path / origin["path"]
    assert 'version="1"' in (checkout / "pyproject.toml").read_text()
    assert prepare.source_status(origin)["ready"]


def test_prompt_changes_are_detected_without_rewriting_them(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "HERE", tmp_path)
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    path = prompts / "base.txt"
    path.write_bytes(b"original prompt\n")
    manifest = {"prompt_archive": {"files": {"base.txt": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}}
    assert prepare.verify_prompts(manifest)["ready"]
    path.write_bytes(b"intentional local edit\n")
    result = prepare.verify_prompts(manifest)
    assert not result["ready"]
    assert result["changed_or_missing"] == ["base.txt"]
    assert path.read_bytes() == b"intentional local edit\n"


def test_all_supplied_prompt_bytes_match_their_manifest():
    manifest = tomllib.loads(prepare.LOCK.read_text())
    assert prepare.verify_prompts(manifest)["ready"]
    for name, entry in manifest["prompt_archive"]["files"].items():
        path = prepare.HERE / "prompts" / name
        assert len(path.read_bytes()) == entry["bytes"]
        assert not any(part.startswith(".") or part == "__MACOSX" for part in path.relative_to(prepare.HERE).parts)
