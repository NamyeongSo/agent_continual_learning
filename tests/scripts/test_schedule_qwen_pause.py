# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.utils import schedule_qwen_pause as scheduler


def test_resumption_keeps_completed_results_and_does_not_charge_pause_as_failure():
    progress = {"jobs": [
        {"id": "done", "status": "completed", "attempts": 1},
        {"id": "train", "status": "paused", "pause_previous_status": "training", "pid": 999, "attempts": 2},
        {"id": "gepa", "status": "paused", "pause_previous_status": "running", "launches": 1},
        {"id": "test", "status": "paused", "pause_previous_status": "evaluating", "attempts": 1},
        {"id": "waiting", "status": "pending"},
    ]}
    jobs = scheduler.resumed_jobs(progress)["jobs"]
    assert jobs[0] == {"id": "done", "status": "completed", "attempts": 1}
    assert jobs[1] == {"id": "train", "status": "pending", "attempts": 1}
    assert jobs[2] == {"id": "gepa", "status": "pending", "launches": 0}
    assert jobs[3] == {"id": "test", "status": "evaluating", "attempts": 1}
    assert jobs[4]["status"] == "pending"


def test_pause_cleanup_preserves_ace_checkpoints_and_archives_unfinished_cancellation(tmp_path):
    folder = tmp_path / "run/train"
    unfinished = folder / "namespace/sessions/unfinished/results.json"
    finished = folder / "namespace/sessions/finished/results.json"
    scheduler.write(unfinished, {"status": "cancelled"})
    scheduler.write(finished, {"status": "success", "score": 1})
    scheduler.write(finished.parent / "agent/playbook_checkpoint.json", {"session_count": 1})
    job = {"id": "ace1", "method": "ace", "output": str(tmp_path / "run"), "train_dir": str(folder)}
    scheduler.preserve_interrupted(job, tmp_path / "control", cutoff=0)
    assert not unfinished.exists()
    assert finished.exists()
    preserved = scheduler.read(tmp_path / "control/interrupted_artifacts/ace1/unfinished/results.json")
    assert preserved["status"] == "cancelled"


def test_completed_actor_response_is_recovered_after_parent_interruption(tmp_path):
    folder = tmp_path / "run"
    scheduler.write(folder / "rollouts/a/attempt_0.json", {"id": "t", "hard": 0, "trace": {}})
    scheduler.preserve_interrupted({"id": "g1", "method": "gepa", "output": str(folder)}, tmp_path / "control", 0)
    assert scheduler.read(folder / "rollouts/a/result.json")["hard"] == 0


@pytest.mark.parametrize("suffix", [b'{"unfinished":', b'{"complete": 2}'])
def test_only_incomplete_jsonl_tail_is_repaired_and_completed_records_survive(tmp_path, suffix):
    path = tmp_path / "data.jsonl"
    prefix = b'{"complete": 1}\n'
    path.write_bytes(prefix + suffix)
    scheduler.trim_partial_jsonl(path, tmp_path / "control", "task")
    if suffix.endswith(b"}"):
        assert path.read_bytes() == prefix + suffix + b"\n"
    else:
        assert path.read_bytes() == prefix
        assert (tmp_path / "control/interrupted_artifacts/task/data.jsonl.partial").read_bytes() == suffix


def test_process_control_never_signals_a_reused_pid_or_an_unrelated_process():
    one = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    two = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        identity = scheduler.identify(one.pid)
        scheduler.send({**identity, "start_ticks": str(int(identity["start_ticks"]) + 1)}, signal.SIGTERM)
        assert one.poll() is None
        scheduler.stop_processes([identity], grace=1)
        one.wait(timeout=3)
        assert two.poll() is None
    finally:
        for process in (one, two):
            if process.poll() is None:
                process.kill()
            process.wait()


def test_wall_clock_runner_executes_pause_then_resume_and_keeps_idempotence(tmp_path, monkeypatch):
    plan = {"output": str(tmp_path), "pause_at": "2000-01-01T18:00:00+09:00",
            "resume_at": "2000-01-02T10:00:00+09:00"}
    calls = []

    def pause(_plan):
        calls.append("pause")
        scheduler.write(tmp_path / "state.json", {"phase": "paused"})

    def resume(_plan):
        calls.append("resume")
        scheduler.write(tmp_path / "state.json", {"phase": "resumed"})

    monkeypatch.setattr(scheduler, "pause", pause)
    monkeypatch.setattr(scheduler, "resume", resume)
    scheduler.run(plan)
    scheduler.run(plan)
    assert calls == ["pause", "resume"]


def test_resume_uses_original_frozen_commands_and_preserves_queue_dependency(tmp_path, monkeypatch):
    base, curator, optimized = [tmp_path / name for name in ("base", "curator", "optimized")]
    roots = [base, curator, optimized]
    control = tmp_path / "schedule"
    plan = {"output": str(control), "base_output": str(base), "roots": list(map(str, roots))}
    for root in roots:
        scheduler.write(root / "launch_manifest.json", {"join_output": str(base) if root == optimized else None})
        scheduler.write(root / "progress.json", {"jobs": [
            {"id": "complete", "status": "completed", "score": 0},
            {"id": "working", "status": "paused", "pause_previous_status": "running",
             "pid": 999, "launches": 1},
            {"id": "queued", "status": "pending"},
        ]})
    scheduler.write(base / "server_process.json", {"pid": 111})
    scheduler.write(base / "bridge_process.json", {"pid": 222})
    services = (None, SimpleNamespace(pid=111), SimpleNamespace(pid=222))
    monkeypatch.setattr(scheduler, "start_services", lambda _: services)
    monkeypatch.setattr(scheduler, "journal_log", lambda *_: None)
    commands = []

    def launch(command, **kwargs):
        commands.append((command, kwargs["cwd"]))
        return SimpleNamespace(pid=300 + len(commands))

    monkeypatch.setattr(scheduler.subprocess, "Popen", launch)
    scheduler.resume(plan)
    assert len(commands) == 3
    assert commands[0][0][-1] == "--resume-existing"
    assert commands[1][0][-2:] == ["--resume-followup", str(curator)]
    assert commands[2][0][1] == str(optimized / "code_snapshot/scripts/baselines/run_queue.py")
    assert [cwd for _, cwd in commands] == [root / "code_snapshot" for root in roots]
    for root in roots:
        jobs = scheduler.read(root / "progress.json")["jobs"]
        assert jobs[0] == {"id": "complete", "status": "completed", "score": 0}
        assert jobs[1] == {"id": "working", "status": "pending", "launches": 0}
        assert jobs[2] == {"id": "queued", "status": "pending"}
        assert scheduler.read(root / "server_process.json") == {"pid": 111}
        assert scheduler.read(root / "control_schedule.json")["phase"] == "resumed"
    assert scheduler.read(control / "state.json")["phase"] == "resumed"
    assert Path(commands[1][0][1]).name == "schedule_qwen_pause.py"


def test_scheduled_ownership_survives_wall_clock_adjustment_and_rejects_other_boot(monkeypatch):
    import os
    import psutil

    record = scheduler.identify(os.getpid())
    actual = psutil.Process.create_time
    monkeypatch.setattr(psutil.Process, "create_time", lambda self: actual(self) + 3)
    assert scheduler.owned(record).pid == os.getpid()
    assert scheduler.owned({**record, "boot_id": "other-boot"}) is None
    assert scheduler.owned({**record, "start_ticks": "0"}) is None
