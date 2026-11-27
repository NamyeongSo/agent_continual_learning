# SPDX-License-Identifier: Apache-2.0
"""Preserve existing trajectories when increasing standalone evaluation concurrency."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.litellm import evaluate_terminus2_after_queue as queue


def test_adoption_rejects_pid_reuse_and_changed_command():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        root = Path("/proc") / str(proc.pid)
        start = (root / "stat").read_text().rsplit(") ", 1)[1].split()[19]
        for _ in range(100):
            command = (root / "cmdline").read_bytes().decode().split("\0")
            if command != [""]:
                break
            time.sleep(.01)
        identity = {"pid": proc.pid, "start_ticks": start, "command": command}
        assert queue.adopted_worker_alive(identity)
        assert not queue.adopted_worker_alive({**identity, "start_ticks": str(int(start) - 1)})
        assert not queue.adopted_worker_alive({**identity, "command": ["another worker"]})
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_worker_exit_race_keeps_the_official_outcome(tmp_path, monkeypatch):
    task = {"task_id": "active"}
    outcome = {**task, "status": "completed", "success": True, "score": 1}
    record = tmp_path / "records/active.json"

    def exit_after_commit(_identity):
        queue.write(record, outcome)
        return False

    monkeypatch.setattr(queue, "adopted_worker_alive", exit_after_commit)
    assert queue.wait_adopted_worker({"output": str(tmp_path)}, task, {"pid": 1}) == outcome
    assert json.loads(record.read_text()) == outcome


def test_dead_adopted_worker_is_infrastructure_error_not_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "adopted_worker_alive", lambda _identity: False)
    row = queue.wait_adopted_worker({"output": str(tmp_path)}, {"task_id": "lost"}, {"pid": 1})
    assert row["status"] == "infrastructure_error"
    assert "score" not in row and "success" not in row


@pytest.mark.parametrize("identities", [
    [{"task_id": "a"}, {"task_id": "a"}],
    [{"task_id": "unknown"}],
])
def test_invalid_adoption_selection_is_rejected(tmp_path, identities):
    path = tmp_path / "adopt.json"
    path.write_text(json.dumps({"workers": identities}))
    with pytest.raises(ValueError):
        queue.adoption_manifest({"adopt_workers_file": str(path), "tasks": [{"task_id": "a"}], "workers": 10})
