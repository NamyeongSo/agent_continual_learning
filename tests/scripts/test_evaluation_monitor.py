# SPDX-License-Identifier: Apache-2.0
import json

from scripts.litellm.evaluate_terminus2_after_queue import predecessor_finished
from scripts.research_journal.evaluation_progress import evaluations
from scripts.research_journal.live_progress import FileCache


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_monitor_counts_official_zeros_and_excludes_infrastructure_errors(tmp_path):
    root = tmp_path / "outputs/current"
    write(root / "evaluation_monitor.json", {"nested_records": True, "order": 2})
    tasks = [{"task_id": str(i), "benchmark": "appworld", "split": "test"} for i in range(4)]
    write(root / "config.json", {"tasks": tasks, "workers": 100})
    write(root / "controller_process.json", {"pid": 123})
    write(root / "status.json", {"phase": "evaluating"})
    for i, outcome in enumerate([
        {"status": "completed", "success": True, "attempt": 2},
        {"status": "completed", "success": False},
        {"status": "infrastructure_error"},
    ]):
        write(root / "records/appworld" / f"{i}.json", outcome)
    cache = FileCache()
    def get():
        return evaluations(tmp_path, cache, lambda p: True, lambda pid, path: pid == 123)

    row = get()[0]["rows"][0]
    assert row["splits"]["all"] == {
        "total": 4, "measured": 2, "successes": 1, "failed": 1, "errors": 1,
        "processed": 3, "remaining": 1, "score_percent": 50, "complete": False,
    }
    # A partially written record keeps the preceding complete snapshot.
    (root / "records/appworld/0.json").write_text('{"status":')
    assert get()[0]["rows"][0]["splits"]["all"]["measured"] == 2


def test_queue_gate_requires_all_300_terminal_records(tmp_path):
    tasks = [{"benchmark": "bfcl", "task_id": str(i)} for i in range(300)]
    write(tmp_path / "config.json", {"tasks": tasks})
    write(tmp_path / "status.json", {"phase": "completed"})
    assert not predecessor_finished(tmp_path)
    for task in tasks:
        write(tmp_path / "records/bfcl" / f"{task['task_id']}.json", {"status": "completed"})
    assert predecessor_finished(tmp_path)
    write(tmp_path / "status.json", {"phase": "evaluating"})
    assert not predecessor_finished(tmp_path)


def test_retry_monitor_links_source_journal_and_shows_only_its_workers(tmp_path, monkeypatch):
    root = tmp_path / "outputs/retry"
    source = tmp_path / "outputs/original"
    write(source / "journal.json", {"id": "original-journal"})
    write(root / "evaluation_monitor.json", {"journal_ids": ["retry-journal"]})
    write(root / "config.json", {"tasks": [], "agent": "terminus2", "retry_timeout_source": str(source)})
    monkeypatch.setattr("scripts.research_journal.evaluation_progress.native_workers", lambda: {
        str(root): ["task-b", "task-a"], str(source): ["other-task"],
    })
    row = evaluations(tmp_path, FileCache(), lambda p: True, lambda pid, path: False)[0]
    assert row["source_journal_ids"] == ["original-journal"]
    assert row["journal_ids"] == ["retry-journal"]
    assert row["active_tasks"] == ["task-a", "task-b"]
