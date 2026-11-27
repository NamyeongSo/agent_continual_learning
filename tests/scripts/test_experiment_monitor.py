# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def monitor():
    path = Path(__file__).resolve().parents[2] / "scripts/utils/monitor_experiments.py"
    spec = importlib.util.spec_from_file_location("experiment_monitor_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_zero_is_valid_but_infrastructure_zero_is_flagged(tmp_path, monitor):
    path = tmp_path / "result.json"
    identity = {"task_ids": ["official-zero", "infrastructure-zero"], "learning": False}
    path.write_text(json.dumps({"identity": identity, "complete": True, "results": [
        {"task_id": "official-zero", "score": 0, "status": "completed"},
        {"task_id": "infrastructure-zero", "score": 0, "status": "error",
         "error": "all predefined address pools have been fully subnetted"},
    ]}))
    result = monitor.group_progress([path], identity["task_ids"], identity=identity, checkpoint_hash=None)
    assert result["complete"]
    assert not result["clean_complete"]
    assert result["processed"] == 2
    assert result["valid_results"] == 1
    assert result["error_kinds"] == {"docker_network_pool": 1}


def test_published_copy_does_not_double_count_and_reopened_result_supersedes_it(tmp_path, monitor):
    native = tmp_path / "native.json"
    repair = tmp_path / "repair.json"
    identity = {"task_ids": ["one", "two"]}
    original = {"identity": identity, "complete": True, "results": [
        {"task_id": task, "status": "error"} for task in identity["task_ids"]
    ]}
    native.write_text(json.dumps(original))
    repair.write_text(json.dumps({**original, "complete": False, "results": [None, None]}))
    result = monitor.group_progress([native, repair], identity["task_ids"], identity=identity, checkpoint_hash=None)
    assert result["processed"] == 0
    assert not result["complete"]
    assert result["total"] == 2


def test_partial_ace_results_are_read_without_a_summary(tmp_path, monitor):
    folder = tmp_path / "benchmark/run/sessions/one"
    folder.mkdir(parents=True)
    (folder / "results.json").write_text(json.dumps({"task_id": "one", "status": "success", "score": 0}))
    result = monitor.group_progress([tmp_path / "benchmark/summary.json"], ["one", "two"],
                                    identity={}, checkpoint_hash=None)
    assert result["processed"] == result["valid_results"] == 1
    assert not result["complete"]


def test_completed_ace_summary_retains_original_rpc_error_classification(tmp_path, monitor):
    session = tmp_path / "run/sessions/id"
    session.mkdir(parents=True)
    row = {"task_id": "one", "session_id": "id", "status": "error", "score": 0}
    (session / "results.json").write_text(json.dumps({**row, "details": {"session_metadata": {
        "error": "timed out", "error_source": "benchmark",
    }}}))
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"sessions": [row]}))
    result = monitor.group_progress([path], ["one"], identity={}, checkpoint_hash=None)
    assert result["complete"]
    assert result["valid_results"] == 0
    assert result["error_kinds"] == {"runner_rpc_timeout": 1}
    assert json.loads(path.read_text())["sessions"] == [row]


def test_wrong_frozen_checkpoint_is_a_protocol_error(tmp_path, monitor):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"identity": {"task_ids": ["one"], "checkpoint_sha256": "changed"},
                                "complete": True, "results": [{"task_id": "one", "status": "completed"}]}))
    result = monitor.group_progress([path], ["one"], identity={"task_ids": ["one"]}, checkpoint_hash="frozen")
    assert not result["complete"]
    assert "checkpoint hash differs" in result["protocol_errors"]


def test_wait_uses_sleep_process_and_does_not_audit_during_wait(tmp_path, monkeypatch, monitor):
    config = tmp_path / "config.json"
    config.write_text("{}")
    events = []
    handlers = {}
    monkeypatch.setattr(monitor.signal, "signal", lambda signum, handler: handlers.update({signum: handler}))
    monkeypatch.setattr(monitor, "write_report", lambda *_args: events.append("audit"))

    class Timer:
        pid = 123

        def __init__(self, command):
            assert command == ["sleep", "1200"]
            events.append("timer_start")

        def poll(self):
            return 0

        def wait(self):
            events.append("wait")
            handlers[monitor.signal.SIGTERM](monitor.signal.SIGTERM, None)

    monkeypatch.setattr(monitor.subprocess, "Popen", Timer)
    monkeypatch.setattr(sys, "argv", ["monitor", "--config", str(config), "--output", str(tmp_path / "out"),
                                      "--watch"])
    monitor.main()
    assert events == ["audit", "timer_start", "wait"]


def test_log_reports_redact_passwords_and_keys(tmp_path, monitor):
    path = tmp_path / "run.log"
    path.write_text("RuntimeError: password=private api_key=private failed\n")
    assert "private" not in "".join(monitor.log_errors(path))


def test_milestone_running_status_is_checked_against_live_worker(tmp_path, monkeypatch, monitor):
    (tmp_path / "status.json").write_text(json.dumps({"jobs": [
        {"name": "stale-worker", "status": "running", "pid": 123},
        {"name": "completed", "status": "done"},
    ]}))
    monkeypatch.setattr(monitor, "processes", lambda: [])
    snapshot = monitor.collect({"learning_roots": [], "milestone_evaluation_roots": [str(tmp_path)]})
    assert snapshot["alerts"] == [{"severity": "error", "kind": "milestone_worker_stopped", "job": "stale-worker"}]
    assert not snapshot["milestone_evaluations"][0]["worker_alive"]


def test_rpc_timeout_is_separate_from_an_official_task_deadline(monitor):
    assert monitor.error_kind({"status": "error", "details": {"session_metadata": {
        "error": "timed out", "error_source": "benchmark",
    }}}) == "runner_rpc_timeout"
    assert monitor.error_kind({"status": "error", "details": {"session_metadata": {
        "error": "", "error_source": "execution", "error_type": "TimeoutError",
    }}}) == "task_deadline"
    assert monitor.error_kind({"status": "error", "error": (
        "RuntimeError: {'session_metadata': {'error': '', 'error_type': 'TimeoutError', "
        "'error_source': 'execution'}}"
    )}) == "task_deadline"


def test_model_probe_timeout_is_retryable_but_authentication_failure_is_not(monitor):
    assert monitor.error_kind({"status": "error", "error": (
        "RuntimeError: Model openai/test is not accessible: TimeoutError()"
    )}) == "provider_timeout"
    assert monitor.error_kind({"status": "error", "error": (
        "RuntimeError: Model openai/test is not accessible: AuthenticationError: invalid key"
    )}) == "other_execution_error"


def test_internal_terminal_progress_is_visible_without_advancing_training_cursor(tmp_path, monitor):
    session = tmp_path / "sessions/premise_test_i0008_candidate_1_current_try1_terminalbench2/sessions/id"
    actor = session / "benchmark/harbor/agent"
    actor.mkdir(parents=True)
    (session / "session.json").write_text(json.dumps({"task_id": "task"}))
    episodes = actor / "premise_episodes.json"
    episodes.write_text(json.dumps({"prefix_length": 1, "episodes": [{"response": "prefix"}]}))
    first = monitor.iteration_activity(tmp_path, 7)
    episodes.write_text(json.dumps({"prefix_length": 1, "episodes": [{"response": "prefix"}, {"response": "new"}]}))
    second = monitor.iteration_activity(tmp_path, 7, first)
    assert second["iteration"] == 8
    assert second["phase"] == "replay_execution"
    assert second["replays_finished"] == 0
    assert second["changed_jobs"] == ["candidate_1_current_terminalbench2"]
    assert second["jobs"]["candidate_1_current_terminalbench2"]["episodes"] == 2


def test_failed_attempt_is_replaced_by_live_retry_for_progress_counts(tmp_path, monitor):
    for attempt in (1, 2):
        session = tmp_path / f"sessions/premise_test_i0008_candidate_1_current_try{attempt}_terminalbench2/sessions/id"
        session.mkdir(parents=True)
        if attempt == 1:
            (session / "results.json").write_text(json.dumps({"task_id": "task", "status": "error"}))
    result = monitor.iteration_activity(tmp_path, 7)
    assert result["replays_total"] == 1
    assert result["replays_finished"] == 0
    assert result["jobs"]["candidate_1_current_terminalbench2"]["status"] == "running"


def test_resumed_progress_excludes_preserved_sessions_from_other_executions(tmp_path, monitor):
    namespace = hashlib.sha256((str(tmp_path) + "\nresumed").encode()).hexdigest()[:10]
    labels = (
        "premise_old_i0008_candidate_1_current_try2_terminalbench2",
        "premise_old_i0008_candidate_1_past0_try1_appworld",
        f"premise_{namespace}_i0008_candidate_1_current_try1_terminalbench2",
    )
    for label in labels:
        session = tmp_path / "sessions" / label / "sessions/id"
        session.mkdir(parents=True)
        if label.startswith("premise_old_"):
            (session / "results.json").write_text(json.dumps({"task_id": "old", "status": "success"}))
    result = monitor.iteration_activity(tmp_path, 7, execution_tag="resumed")
    assert result["replays_total"] == 1
    assert result["replays_finished"] == 0
    assert list(result["jobs"]) == ["candidate_1_current_terminalbench2"]


def test_new_execution_waiting_for_sessions_does_not_inherit_old_progress(tmp_path, monitor):
    session = tmp_path / "sessions/premise_old_i0008_parent1_terminalbench2/sessions/id"
    session.mkdir(parents=True)
    (session / "results.json").write_text(json.dumps({"task_id": "old", "status": "success"}))
    result = monitor.iteration_activity(tmp_path, 7, execution_tag="resumed")
    assert result["phase"] == "starting"
    assert result["jobs"] == {}


def test_worker_restart_is_detected_even_if_pid_is_reused_and_cursor_does_not_advance(tmp_path, monkeypatch, monitor):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({f"{slug}_42": {"test": ["test"]} for slug in monitor.BENCHMARKS}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "state.json").write_text('{"cursor": 1}')
    job = {"name": "s42_sequential_t3", "mode": "sequential", "seed": 42,
           "output_dir": str(run), "task_order": [["bfcl", "first"], ["bfcl", "second"]]}
    (tmp_path / "launch_manifest.json").write_text(json.dumps({
        "method": "premise", "source_manifest": str(split),
        "source_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(), "jobs": [job],
    }))
    monkeypatch.setattr(monitor, "processes", lambda: [{
        "pid": 123, "start_ticks": "20", "script": "run_experiment.py", "flags": {"--output-dir": str(run)},
    }])
    previous = {"learning": [{"key": "premise:s42_sequential_t3", "cursor": 1,
                              "pids": [123], "worker_instances": [{"pid": 123, "start_ticks": "10"}]}]}
    snapshot = monitor.collect({"learning_roots": [str(tmp_path)]}, previous)
    assert snapshot["learning"][0]["restarted_since_previous"]
    assert snapshot["learning"][0]["delta_tasks"] == 0
    assert [alert["kind"] for alert in snapshot["alerts"]] == ["learning_worker_restarted"]


def test_resumed_run_tracks_only_planned_evaluations_and_shared_controller(tmp_path, monkeypatch, monitor):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({f"{slug}_43": {"test": ["test"]} for slug in monitor.BENCHMARKS}))
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "state.json").write_text('{"cursor": 1}')
    (run / "checkpoints/checkpoint_001.json").write_text("{}")
    controller_root = tmp_path / "resumed_experiment"
    job = {
        "name": "s43_sequential_t3", "mode": "sequential", "seed": 43,
        "output_dir": str(run), "task_order": [["bfcl", "first"]],
        "evaluation_checkpoints": [1], "evaluation_controller_root": str(controller_root),
    }
    (tmp_path / "launch_manifest.json").write_text(json.dumps({
        "method": "premise", "source_manifest": str(split),
        "source_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(), "jobs": [job],
    }))
    monkeypatch.setattr(monitor, "processes", lambda: [
        {"pid": 123, "script": "restart_sequential.py", "flags": {"--output-root": str(controller_root)}},
        {"pid": 456, "script": "restart_sequential.py", "flags": {"--output-root": str(tmp_path / "other")}},
    ])

    snapshot = monitor.collect({"learning_roots": [str(tmp_path)]})

    assert len(snapshot["checkpoint_evaluations"]) == 3
    assert {group["checkpoint"] for group in snapshot["checkpoint_evaluations"]} == {1}
    assert all(group["owner_pids"] == [123] for group in snapshot["checkpoint_evaluations"])
    assert not any(alert["kind"] == "evaluation_unowned" for alert in snapshot["alerts"])


def test_output_launched_controller_owns_running_checkpoint(tmp_path, monkeypatch, monitor):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({f"{slug}_44": {"test": ["test"]} for slug in monitor.BENCHMARKS}))
    run = tmp_path / "s44_sequential_t3"
    (run / "checkpoints").mkdir(parents=True)
    (run / "state.json").write_text('{"cursor": 1}')
    (run / "checkpoints/checkpoint_001.json").write_text("{}")
    controller_root = tmp_path / "restart"
    controller_root.mkdir()
    (controller_root / "restart_status.json").write_text(json.dumps({
        "controller_pid": 123,
        "queue": {"running": 1, "queued": 0},
        "experiments": [{"output": str(run), "evaluations": {"1": "running"}}],
    }))
    (tmp_path / "launch_manifest.json").write_text(json.dumps({
        "method": "premise", "source_manifest": str(split),
        "source_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "jobs": [{"name": run.name, "mode": "sequential", "seed": 44, "output_dir": str(run),
                  "task_order": [["bfcl", "first"]], "evaluation_checkpoints": [1],
                  "evaluation_controller_root": str(controller_root)}],
    }))
    proc_root = tmp_path / "proc"
    (proc_root / "123").mkdir(parents=True)
    (proc_root / "123/stat").write_text("123 (python) S 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")
    (proc_root / "123/cmdline").write_bytes(b"python\0/work/outputs/resume_controller.py\0")
    real_owner = monitor.active_shared_controller
    monkeypatch.setattr(monitor, "active_shared_controller",
                        lambda root, run, point: real_owner(root, run, point, proc_root=proc_root))
    monkeypatch.setattr(monitor, "processes", lambda: [])

    snapshot = monitor.collect({"learning_roots": [str(tmp_path)]})

    assert all(group["owner_pids"] == [123] for group in snapshot["checkpoint_evaluations"])
    assert not any(alert["kind"] == "evaluation_unowned" for alert in snapshot["alerts"])
