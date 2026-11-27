# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from collections import Counter

import pytest

from scripts.baselines import run_queue
from scripts.baselines.runtime import write


@pytest.mark.parametrize("joined", [False, True])
@pytest.mark.parametrize("version", [None, "2.1"])
def test_frozen_queue_has_all_18_unique_runs_and_preserves_dependency(tmp_path, monkeypatch, joined, version):
    root = tmp_path / "repository"
    for name in ("src", "scripts"):
        (root / name).mkdir(parents=True)
        (root / name / "fixture.py").write_text("# frozen source\n")
    for name in ("pyproject.toml", "uv.lock"):
        (root / name).write_text("# fixture\n")
    monkeypatch.setattr(run_queue, "ROOT", root)
    monkeypatch.setattr(run_queue, "source_files", lambda root: [
        path.relative_to(root) for name in ("src", "scripts") for path in (root / name).glob("*.py")])
    output = tmp_path / "queue"
    predecessor = tmp_path / "existing_curator_queue"
    plan = run_queue.freeze(output, predecessor, max_running_per_method=3 if joined else 6,
                            join_output=tmp_path / "active" if joined else None, terminalbench2_version=version)
    assert len(plan["jobs"]) == len({job["id"] for job in plan["jobs"]}) == 18
    assert Counter(job["method"] for job in plan["jobs"]) == {"gepa": 9, "skillopt": 9}
    assert Counter(job["benchmark"] for job in plan["jobs"]) == {"bfcl": 6, "appworld": 6, "terminalbench2": 6}
    assert plan["max_running_per_method"] == (3 if joined else 6) and plan["internal_workers"] == 50
    assert plan["max_running"] == (6 if joined else 12)
    assert bool(plan["join_output"]) == joined
    assert plan["after_output"] == str(predecessor)
    assert all(job["status"] == "pending" for job in plan["jobs"])
    assert (output / "code_snapshot/src/fixture.py").read_text() == "# frozen source\n"
    assert not list(output.rglob("summary.json"))
    for job in plan["jobs"]:
        if job["benchmark"] == "terminalbench2" and version is not None:
            assert job["terminalbench2_version"] == version
            index = job["command"].index("--terminalbench2-version")
            assert job["command"][index + 1] == version
        else:
            assert "--terminalbench2-version" not in job["command"]


def test_queue_completion_requires_all_online_encounters_and_test_results(tmp_path):
    job = {"output": str(tmp_path)}
    write(tmp_path / "config.json", {"train": [{"task_id": "a"}, {"task_id": "b"}],
                                   "test": [{"task_id": "test1"}, {"task_id": "test2"}]})
    write(tmp_path / "summary.json", {"complete": True, "tasks": 2})
    write(tmp_path / "test_results.json", [{"id": "test1", "hard": 0}, {"id": "test2", "hard": 1}])
    assert not run_queue.complete(job)
    write(tmp_path / "online/a.json", {"task_id": "a", "hard": 0})
    assert not run_queue.complete(job)
    write(tmp_path / "online/b.json", {"task_id": "b", "hard": 0})
    assert run_queue.complete(job)
    job["terminalbench2_version"] = "2.1"
    assert not run_queue.complete(job)
    config = run_queue.read(tmp_path / "config.json")
    config["benchmark_kwargs"] = {"subset": "2.1"}
    write(tmp_path / "config.json", config)
    assert run_queue.complete(job)


def test_resume_persists_finished_orphans_without_launching_another_actor(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.utils import run_qwen_isolated_when_free as runner

    output = tmp_path / "queue"
    folder = output / "finished"
    write(folder / "config.json", {"train": [{"task_id": "a"}], "test": [{"task_id": "b"}]})
    write(folder / "summary.json", {"complete": True, "tasks": 1})
    write(folder / "online/a.json", {"task_id": "a", "hard": 0})
    write(folder / "test_results.json", [{"id": "b", "hard": 0}])
    write(output / "launch_manifest.json", {"join_output": str(tmp_path / "service"),
                                           "after_output": str(tmp_path / "service"),
                                           "max_running_per_method": 3, "max_running": 6})
    write(output / "progress.json", {"jobs": [{"id": "last", "method": "skillopt", "status": "running",
                                              "pid": 123, "output": str(folder), "command": ["original"]}]})
    monkeypatch.setattr(run_queue, "reuse_services", lambda *_: (SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(runner, "preflight_model", lambda *_: None)
    monkeypatch.setattr(run_queue, "journal", lambda *_: None)
    monkeypatch.setattr(run_queue.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Already finished"))
    run_queue.run(output, poll_seconds=0)
    assert run_queue.read(output / "progress.json")["jobs"][0]["status"] == "completed"
    assert run_queue.read(output / "status.json")["phase"] == "completed"
