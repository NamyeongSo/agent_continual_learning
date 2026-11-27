# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.utils import run_qwen_followup as followup
from scripts.utils import run_qwen_isolated_when_free as runner


def _finished_predecessor(path):
    job = {"id": "one", "status": "completed", "output": str(path / "job"), "test_ids": ["test"]}
    runner.write(path / "launch_manifest.json", {"jobs": [job]})
    runner.write(path / "status.json", {"phase": "completed"})
    runner.write(path / "progress.json", {"jobs": [job], "running_training": 0, "pending_evaluations": 0})
    runner.write(path / "job/evaluation/summary.json", {"complete": True, "tasks": 1})
    runner.write(path / "job/evaluation/records/000.json", {"status": "completed", "score": 0})
    return job


@pytest.mark.parametrize("pending", ["training", "evaluation", "unstarted", "missing_record"])
def test_predecessor_completion_marker_does_not_skip_unfinished_work(tmp_path, pending):
    _finished_predecessor(tmp_path)
    assert followup.predecessor_finished(tmp_path)
    progress = runner.read(tmp_path / "progress.json")
    if pending == "training":
        progress["running_training"] = 1
    elif pending == "evaluation":
        progress["pending_evaluations"] = 1
    elif pending == "unstarted":
        progress["jobs"][0]["status"] = "pending"
    else:
        (tmp_path / "job/evaluation/records/000.json").unlink()
    runner.write(tmp_path / "progress.json", progress)
    assert not followup.predecessor_finished(tmp_path)


def test_followup_waits_before_acquiring_gpu_or_using_services_and_preserves_predecessor(tmp_path, monkeypatch):
    predecessor = tmp_path / "before"
    output = tmp_path / "after"
    _finished_predecessor(predecessor)
    before = {str(p): p.read_bytes() for p in predecessor.rglob("*") if p.is_file()}
    job = {"id": "new", "status": "completed"}
    runner.write(output / "launch_manifest.json", {"after_output": str(predecessor), "jobs": [job],
                                                  "train_workers": 12, "playbook_edit_mode": "add_update_delete"})
    ready = [False]
    events = []
    monkeypatch.setattr(followup, "predecessor_finished", lambda _path: ready[0])

    def finish(_seconds):
        assert runner.read(output / "status.json")["phase"] == "waiting_predecessor"
        events.append("wait")
        ready[0] = True

    def lease(*_args):
        assert ready[0]
        events.append("lease")
        return SimpleNamespace(release=lambda: events.append("release"))

    def reuse(*_args):
        assert ready[0]
        events.append("reuse")
        return SimpleNamespace(pid=123), SimpleNamespace(pid=456)

    monkeypatch.setattr(followup.time, "sleep", finish)
    monkeypatch.setattr(followup, "acquire_gpu_lease", lease)
    monkeypatch.setattr(followup, "reuse_services", reuse)
    monkeypatch.setattr(runner, "preflight_model", lambda _path: events.append("preflight"))
    monkeypatch.setattr(runner, "journal", lambda _plan, _path, phase, _text: events.append(phase))
    monkeypatch.setattr(runner, "run_training_and_evaluation", lambda *_args: events.append("train") or [job])
    followup.run_followup(output, predecessor, 1)
    assert events == ["wait", "lease", "reuse", "preflight", "running", "train", "completed", "release"]
    assert all(Path(p).read_bytes() == data for p, data in before.items())


def test_reused_services_keep_original_log_paths_without_restarting(tmp_path, monkeypatch):
    predecessor = tmp_path / "before"
    output = tmp_path / "after"
    runner.write(predecessor / "server_process.json", {"pid": 123})
    runner.write(predecessor / "bridge_process.json", {"pid": 456})
    expected = SimpleNamespace(pid=123), SimpleNamespace(pid=456)
    monkeypatch.setattr(runner, "existing_services", lambda path: expected if path == predecessor else None)
    assert followup.reuse_services(predecessor, output) == expected
    assert runner.read(output / "server_process.json")["pid"] == 123
    assert runner.read(output / "services_reused.json")["vllm_log"] == str(
        predecessor / "output_nohup_vllm_server_qwen_gpu.log",
    )
