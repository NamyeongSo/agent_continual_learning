# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Keep Terminal-Bench releases, task paths and result provenance consistent."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from exgentic.benchmarks.terminalbench2.terminalbench2_benchmark import (
    TERMINALBENCH_COMMITS,
    TerminalBench2Benchmark,
    default_task_root,
)
from exgentic.benchmarks.terminalbench2.terminalbench2_eval import (
    TerminalBench2Evaluator,
    TerminalBench2Session,
    resolve_task_root,
)
from exgentic.core.types import SessionIndex
from exgentic.interfaces.registry import apply_subset_kwargs, get_benchmark_subsets


@pytest.fixture
def release_checkout(tmp_path, monkeypatch):
    root = tmp_path / "terminal-bench-2-1"
    tasks = root / "tasks"
    for index in range(89):
        directory = tasks / f"task-{index:02d}"
        directory.mkdir(parents=True)
        (directory / "task.toml").write_text("[agent]\ntimeout_sec = 3600\n")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init", "--quiet")
    git("add", "tasks")
    git("-c", "user.name=Release Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    monkeypatch.setitem(TERMINALBENCH_COMMITS, "2.1", git("rev-parse", "HEAD"))
    return root, tasks


def test_release_selection_and_defaults_are_explicit(monkeypatch):
    monkeypatch.setenv("EXGENTIC_TERMINALBENCH2_TASK_ROOT", "/old-release")
    monkeypatch.delenv("EXGENTIC_TERMINALBENCH21_TASK_ROOT", raising=False)
    assert TerminalBench2Benchmark().subset == "2.0"
    assert default_task_root() == Path("/old-release")
    assert default_task_root("2.1") == Path.home() / ".exgentic/benchmarks/terminalbench2/terminal-bench-2-1/tasks"
    monkeypatch.setenv("EXGENTIC_TERMINALBENCH21_TASK_ROOT", "/new-release")
    assert default_task_root("2.1") == Path("/new-release")
    assert get_benchmark_subsets("terminalbench2") == ["2.0", "2.1"]
    kwargs = apply_subset_kwargs("terminalbench2", "2.1", {})
    assert TerminalBench2Benchmark(**kwargs)._get_evaluator_kwargs()["subset"] == "2.1"
    with pytest.raises(ValidationError):
        TerminalBench2Benchmark(subset="latest")


def test_release_paths_and_deadlines_use_nested_tasks(release_checkout):
    root, tasks = release_checkout
    assert resolve_task_root(str(root), "2.1") == tasks
    assert resolve_task_root(str(tasks), "2.1") == tasks
    benchmark = TerminalBench2Benchmark(subset="2.1", task_root=str(root))
    assert benchmark.runner_kwargs()["transport_timeout"] == 4200
    evaluator = TerminalBench2Evaluator(**benchmark._get_evaluator_kwargs())
    assert len(evaluator.list_tasks()) == 89
    kwargs = evaluator.get_session_kwargs(SessionIndex(task_id="task-00", session_id="release-test"))
    assert kwargs["subset"] == "2.1"
    assert kwargs["task_dir"] == str(tasks / "task-00")
    assert kwargs["enforce_session_timeouts"] is True


def test_wrong_release_and_modified_tests_are_rejected(release_checkout):
    root, tasks = release_checkout
    with pytest.raises(ValueError, match=r"expected pinned 2\.0 commit"):
        resolve_task_root(str(root), "2.0")
    path = tasks / "task-00/task.toml"
    original = path.read_text()
    path.write_text(original + "# local change\n")
    with pytest.raises(ValueError, match="local changes"):
        resolve_task_root(str(tasks), "2.1")
    path.write_text(original)
    (tasks / "task-00/untracked-test.py").write_text("# additional verifier\n")
    with pytest.raises(ValueError, match="local changes"):
        resolve_task_root(str(tasks), "2.1")


def test_unverified_archive_cannot_claim_the_pinned_21_release(tmp_path):
    with pytest.raises(ValueError, match="pinned Git checkout"):
        resolve_task_root(str(tmp_path), "2.1")


def test_aggregate_records_release_and_rejects_old_results(release_checkout, tmp_path, monkeypatch):
    root, _ = release_checkout
    evaluator = TerminalBench2Evaluator(task_root=str(root), subset="2.1")
    result = tmp_path / "benchmark_results.json"
    result.write_text(json.dumps({"score": 1, "session_metadata": {"dataset_version": "2.1"}}))
    monkeypatch.setattr(evaluator, "get_sessions_paths", lambda _sessions: [SimpleNamespace(benchmark_results=result)])
    aggregate = evaluator.aggregate_sessions([])
    assert aggregate.score == 1
    assert aggregate.metrics["dataset_version"] == "2.1"
    assert aggregate.metrics["dataset_commit"] == TERMINALBENCH_COMMITS["2.1"]
    # Historical results without version metadata belong to 2.0.
    result.write_text(json.dumps({"score": 1}))
    with pytest.raises(ValueError, match="Cannot aggregate"):
        evaluator.aggregate_sessions([])


def test_budget_failures_retain_the_release_for_aggregation(monkeypatch):
    session = object.__new__(TerminalBench2Session)
    session._subset = "2.1"
    saved = []
    monkeypatch.setattr(session, "save_results", saved.append)
    score = session.fail_experiment_budget({"reason": "turn_limit"}, {"max_turns": 40})
    assert score.score == 0 and score.success is False
    assert score.session_metadata["dataset_version"] == "2.1"
    assert saved[0]["session_metadata"]["dataset_version"] == "2.1"
