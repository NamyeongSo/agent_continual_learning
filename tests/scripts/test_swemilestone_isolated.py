# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Sequential experiment execution and unbiased three-repeat score summaries."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import ClassVar

import pytest


@pytest.fixture
def experiment():
    script = Path(__file__).resolve().parents[2] / "scripts/ace/run_swemilestone_isolated.py"
    spec = importlib.util.spec_from_file_location("swemilestone_isolated_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def catalog():
    return [
        {"repo_alias": repo, "task_id": f"{repo}/{milestone}", "graded": graded}
        for repo in ("ripgrep", "navidrome")
        for milestone, graded in (("seed", False), ("implementation", True))
    ]


def result(output, job, score, *, valid=True, missing=0):
    path = output / job["name"] / "experiment_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "score": score, "valid_for_statistics": valid, "missing_graded_milestones": missing,
        "infrastructure_errors": [] if valid else [{"error": "Docker disconnected"}],
    }))


def test_repeats_keep_dependency_order_and_include_ungraded_milestones(experiment):
    plan = experiment.build_plan(catalog(), ["ripgrep", "navidrome"], 3, {})
    assert [(job["repeat"], job["repo"]) for job in plan["jobs"]] == [
        (repeat, repo) for repeat in (1, 2, 3) for repo in ("ripgrep", "navidrome")
    ]
    assert len({job["name"] for job in plan["jobs"]}) == 6
    assert plan["total_sessions"] == 12
    for job in plan["jobs"]:
        assert job["task_ids"] == [f"{job['repo']}/seed", f"{job['repo']}/implementation"]
        assert job["graded_milestones"] == 1
        assert job["ungraded_milestones"] == 1


def test_sample_std_and_overall_use_independent_repeat_scores(experiment, tmp_path):
    plan = experiment.build_plan(catalog(), ["ripgrep", "navidrome"], 3, {})
    for job in plan["jobs"]:
        score = job["repeat"] / 10 if job["repo"] == "ripgrep" else job["repeat"] / 10 + 0.2
        result(tmp_path, job, score)
    summary = experiment.summarize(plan, tmp_path)
    assert summary["complete"]
    assert summary["repositories"]["ripgrep"]["mean"] == pytest.approx(0.2)
    assert summary["repositories"]["ripgrep"]["std"] == pytest.approx(0.1)
    assert summary["overall"]["mean"] == pytest.approx(0.3)
    assert summary["overall"]["std"] == pytest.approx(0.1)
    experiment.write_summary(plan, tmp_path)
    assert "20.000" in (tmp_path / "summary.md").read_text()


def test_valid_zero_is_counted_but_missing_and_infrastructure_errors_are_not(experiment, tmp_path):
    plan = experiment.build_plan(catalog(), ["ripgrep", "navidrome"], 3, {})
    result(tmp_path, plan["jobs"][0], 0.0, missing=1)
    result(tmp_path, plan["jobs"][1], 0.2)
    result(tmp_path, plan["jobs"][2], 0.8, valid=False)
    result(tmp_path, plan["jobs"][3], 0.4)
    summary = experiment.summarize(plan, tmp_path)
    assert not summary["complete"]
    assert summary["repositories"]["ripgrep"]["n"] == 1
    assert summary["repositories"]["ripgrep"]["mean"] == 0
    assert summary["repositories"]["ripgrep"]["std"] is None
    assert summary["overall"]["n"] == 1
    assert summary["overall"]["mean"] == pytest.approx(0.1)
    assert summary["excluded_jobs"][0]["job"] == "repeat2_ripgrep"


@pytest.mark.parametrize("synthetic_turns", [0, 1, 3])
def test_legacy_failed_context_recovery_is_excluded_without_rewriting_results(
    experiment, tmp_path, synthetic_turns
):
    plan = experiment.build_plan(catalog(), ["ripgrep"], 1, {})
    job = plan["jobs"][0]
    result(tmp_path, job, 0.0)
    original = tmp_path / job["name"] / "experiment_result.json"
    before = original.read_bytes()
    session = original.parent / "sessions" / "legacy"
    trajectory = session / "benchmark/terminus/trajectory.json"
    trajectory.parent.mkdir(parents=True)
    session.joinpath("session.json").write_text(json.dumps({"task_id": "ripgrep/implementation"}))
    trajectory.write_text(json.dumps({"steps": [
        {"source": "agent", "message": "Technical difficulties. Please continue with the task.",
         "metrics": {"prompt_tokens": 0, "completion_tokens": 0}}
        for _ in range(synthetic_turns)
    ]}))
    selected = experiment.read_job_result(plan, tmp_path, job)
    assert selected["valid_for_statistics"] is (synthetic_turns < 3)
    assert experiment.retryable_repository_result(selected) is (synthetic_turns >= 3)
    summary = experiment.summarize(plan, tmp_path)
    assert summary["repositories"]["ripgrep"]["n"] == int(synthetic_turns < 3)
    assert original.read_bytes() == before


def test_queue_runs_only_one_repository_at_a_time_across_three_repeats(experiment, tmp_path, monkeypatch):
    plan = experiment.build_plan(catalog(), ["ripgrep", "navidrome"], 3, {})
    real_sleep = time.sleep
    monkeypatch.setattr(experiment.time, "sleep", lambda _seconds: real_sleep(0.01))
    monkeypatch.setattr(experiment.signal, "signal", lambda *_args: None)

    class FakeProcess:
        active = 0
        peak = 0
        started: ClassVar[list[str]] = []

        def __init__(self, command, **kwargs):
            self.name = command[command.index("--worker-job") + 1]
            self.job = next(job for job in plan["jobs"] if job["name"] == self.name)
            self.started.append(self.name)
            self.deadline = time.monotonic() + 0.015
            self.returncode = None
            self.pid = 1000 + len(self.started)
            FakeProcess.active += 1
            FakeProcess.peak = max(FakeProcess.peak, FakeProcess.active)
            assert kwargs["start_new_session"]

        def poll(self):
            if self.returncode is None and time.monotonic() >= self.deadline:
                result(tmp_path, self.job, 0.1)
                self.returncode = 0
                FakeProcess.active -= 1
            return self.returncode

    monkeypatch.setattr(experiment.subprocess, "Popen", FakeProcess)
    assert experiment.run_queue(plan, tmp_path) == 0
    assert FakeProcess.peak == 1
    assert FakeProcess.started == [job["name"] for job in plan["jobs"]]
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "done"
    assert [job["status"] for job in status["jobs"]] == ["done"] * 6
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["complete"]
    assert summary["overall"]["n"] == 3


def test_repository_recovery_preserves_original_and_replaces_only_infrastructure_results(
    experiment, tmp_path, monkeypatch
):
    plan = experiment.build_plan(catalog(), ["ripgrep", "navidrome"], 1, {"model": "frozen-model"})
    experiment.write_json(tmp_path / "launch_manifest.json", plan)
    for job in plan["jobs"]:
        result(tmp_path, job, 0.0)
    failed = plan["jobs"][1]
    original = tmp_path / failed["name"] / "experiment_result.json"
    original.write_text(json.dumps({
        "score": 0.0, "valid_for_statistics": False, "missing_graded_milestones": 1,
        "infrastructure_errors": [{"error": (
            "SWE-Milestone backend failed (255): kex_exchange_identification: Connection reset by peer"
        )}],
    }))
    before = original.read_bytes()
    queues = []

    def queue(repair_plan, output):
        queues.append(output)
        assert output != tmp_path
        assert repair_plan["jobs"] == [failed]
        assert repair_plan["settings"] == plan["settings"]
        result(output, failed, 0.5)
        return 0

    monkeypatch.setattr(experiment, "run_queue", queue)
    monkeypatch.setattr(experiment, "original_worker_running", lambda _output: False)
    assert experiment.recover_infrastructure(tmp_path) == 0
    assert len(queues) == 1
    assert original.read_bytes() == before
    assert experiment.read_job_result(plan, tmp_path, failed)["score"] == 0.5
    summary = experiment.summarize(plan, tmp_path)
    assert summary["complete"]
    assert summary["overall"]["mean"] == pytest.approx(0.25)
    assert not summary["excluded_jobs"]
    original.write_text("{}")
    with pytest.raises(ValueError, match="original result changed"):
        experiment.summarize(plan, tmp_path)


def test_only_known_ssh_failures_reopen_a_repository_stream(experiment):
    assert not experiment.retryable_repository_result({"valid_for_statistics": True, "score": 0})
    assert not experiment.retryable_repository_result({
        "valid_for_statistics": False, "infrastructure_errors": [{"error": "unknown failure"}],
    })
    assert not experiment.retryable_repository_result({
        "valid_for_statistics": False, "infrastructure_errors": [{"error": "is blocked; submit prerequisites first"}],
    })
    assert experiment.retryable_repository_result({
        "valid_for_statistics": False, "infrastructure_errors": [{"error": (
            "Terminus-2 failed: SWE-Milestone backend failed (255): kex_exchange_identification: Connection reset"
        )}],
    })
    assert experiment.retryable_repository_result({
        "valid_for_statistics": False, "infrastructure_errors": [{"error": (
            "Official milestone evaluation failed: residue prune: capture_filter does not match pinned repo config"
        )}],
    })
