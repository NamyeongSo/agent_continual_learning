# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import hashlib
import io
import json
import os
import threading

import pytest

from scripts.research_journal.live_progress import FileCache, LiveProgress
from scripts.research_journal.store import JournalStore


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = JournalStore(tmp_path / "journal", [tmp_path])
    root = tmp_path / "outputs/batch"
    monkeypatch.setattr("scripts.research_journal.live_progress.live_pid", lambda pid, target: pid == 123)
    return store, root, LiveProgress(store, base_dir=tmp_path, ttl=0)


def queue(root, job, status="training"):
    job = {"id": "s42", "method": "premise", "seed": 42, "benchmark": "bfcl", "train_count": 2,
           "test_ids": ["t1", "t2"], "output": str(root / "run"), "pid": 123, **job}
    write(root / "launch_manifest.json", {"created_at": "2026-10-21", "model": "Qwen", "jobs": [job]})
    write(root / "progress.json", {"jobs": [{**job, "status": status}]})
    return root / "run"


def first(collector):
    return collector.get()["batches"][0]["runs"][0]


def test_checkpoint_validation_keeps_variations_separate_and_uses_official_val(setup):
    _, root, collector = setup
    key = (root.name, "gepa", "appworld", 42)
    collector.latest_checkpoints = {
        (*key, "gepa_val"): {"point": 50, "splits": {"val": {"score": 0.2, "done": 50, "total": 50}}},
        (*key, "gepa_train_as_val"): {
            "point": 50, "splits": {"val": {"score": 0.4, "done": 50, "total": 50}}},
    }
    row = {"method": "gepa", "benchmark": "appworld", "seed": 42, "variant": "gepa_train_as_val",
           "state": "completed", "validation_source": "train", "scores": {}}
    collector.checkpoint_validation(row, root)
    assert row["scores"]["val"]["value"] == 0.4
    assert row["scores"]["val"]["label"] == "Offline-val · TGC · CP50"


@pytest.mark.parametrize("method", ["ace", "premise"])
def test_matrix_reports_actual_curator_mode_and_training_actor_limits(setup, method):
    _, root, collector = setup
    train = root / "run/train"
    queue(root, {"method": method, "train_dir": str(train), "playbook_edit_mode": "add_update_delete"}, "pending")
    limits = {"version": "actor-turn-v1", "input_tokens": 100000, "output_tokens": 2048,
              "max_turns": 30, "token_scope": "environment_turn"}
    cfg = {"experiment_limits": {"bfcl": limits}, "task_order": [["bfcl", "a"], ["bfcl", "b"]]}
    if method == "ace":
        write(train / "experiment_config.json", cfg)
    else:
        write(train / "state.json", {"config": {**cfg, "order": cfg["task_order"]}, "cursor": 0})
    batch = collector.get()["batches"][0]
    assert batch["playbook_edit_mode"] == "add_update_delete"
    assert batch["runs"][0]["actor_limits"] == limits


def test_checkpoint_resume_reservation_is_visible_without_claiming_evaluation_is_running(setup):
    _, root, collector = setup
    schedule = root.parent / "resume"
    write(root / "checkpoint_queue_manifest.json", {"journal_id": "record"})
    write(root / "dashboard.json", {"kind": "five_task_checkpoint_queue", "controller_pid": 999,
                                   "phase": "paused", "task_counts": {"completed": 10, "pending": 5},
                                   "scope_label": "original policy · 36 streams", "rows": []})
    write(root / "resume_schedule.json", {"output": str(schedule)})
    write(schedule / "scheduler_process.json", {"pid": 123})
    write(schedule / "state.json", {"phase": "waiting_prerequisites", "contexts": [
        {"seed": 42, "training_complete": True, "test_complete": False}], "internal_debug": "DO_NOT_EXPOSE"})
    data = collector.checkpoint_queues()[0]
    assert data["phase"] == "waiting_prerequisites" and data["task_counts"]["pending"] == 5
    assert data["resume_schedule"]["contexts"][0]["training_complete"]
    assert data["scope_label"] == "original policy · 36 streams"
    assert "DO_NOT_EXPOSE" not in json.dumps(data)


def test_hidden_checkpoint_queue_keeps_history_available_for_tables_but_leaves_live_payload(setup, monkeypatch):
    _, root, collector = setup
    for name, visible in [("old_cp5", False), ("current_cp10", True)]:
        folder = root.parent / name
        write(folder / "checkpoint_queue_manifest.json", {"journal_id": name})
        write(folder / "dashboard.json", {"kind": "five_task_checkpoint_queue", "controller_pid": 123,
                                           "phase": "running", "task_counts": {"completed": 7}, "rows": []})
        write(folder / "live_monitor_visibility.json", {"visible": visible})
    assert [q["id"] for q in collector.checkpoint_queues()] == ["current_cp10"]
    assert {q["id"] for q in collector.checkpoint_queues(include_hidden=True)} == {"old_cp5", "current_cp10"}
    seen = []

    def tables(batches, queues):
        seen.extend(q["id"] for q in queues)
        return []

    monkeypatch.setattr(collector, "result_tables", tables)
    snapshot = collector.get()
    assert [q["id"] for q in snapshot["checkpoint_queues"]] == ["current_cp10"]
    assert set(seen) == {"old_cp5", "current_cp10"}
    assert json.loads((root.parent / "old_cp5/dashboard.json").read_text())["task_counts"]["completed"] == 7


def test_policy_shift_followup_projects_progress_and_detects_stopped_runner(setup):
    _, root, collector = setup
    target = root.parent / "shift"
    write(root / "policy_shift_followup.json", {"output": str(target), "journal_id": "shift-record"})
    write(target / "prepared.json", {"histories": 100, "effective_requests": 50,
                                     "source_hashes": "DO_NOT_EXPOSE"})
    write(target / "plan.json", {"temperature": 0, "workers": 128, "messages": "DO_NOT_EXPOSE"})
    write(target / "status.json", {"phase": "probing", "counts": {
        "completed": 10, "error": 2, "running": 3, "pending": 35}, "private": "DO_NOT_EXPOSE"})
    write(target / "controller_process.json", {"pid": 123})
    value = collector.policy_shift_followup(root)
    assert value["phase"] == "probing" and value["workers"] == 128 and value["temperature"] == 0
    assert value["progress"]["done"] == 12 and value["progress"]["total"] == 50
    assert value["progress"]["errors"] == 2 and value["counts"]["running"] == 3
    assert "DO_NOT_EXPOSE" not in json.dumps(value)
    write(target / "controller_process.json", {"pid": 9999})
    assert collector.policy_shift_followup(root)["phase"] == "stopped"
    write(root / "policy_shift_followup.json", {"output": "/etc", "journal_id": "outside"})
    assert collector.policy_shift_followup(root) is None


def test_policy_shift_has_standalone_progress_without_training_runs(setup):
    _, root, collector = setup
    target = root.parent / "shift"
    write(root / "checkpoint_queue_manifest.json", {"journal_id": "cp-record"})
    write(root / "dashboard.json", {"kind": "five_task_checkpoint_queue", "phase": "completed", "rows": []})
    write(root / "policy_shift_followup.json", {"output": str(target), "journal_id": "shift-record"})
    write(target / "controller_process.json", {"pid": 123})
    write(target / "plan.json", {"temperature": 0, "workers": 256, "per_gpu_query_limit": 128})
    write(target / "prepared.json", {"effective_requests": 100})
    write(target / "status.json", {"phase": "probing", "counts": {"completed": 20, "pending": 80}})
    snapshot = collector.get()
    assert snapshot["batches"] == []
    shift = snapshot["policy_shifts"][0]
    assert shift["journal_id"] == "shift-record" and shift["source_journal_ids"] == ["cp-record"]
    assert shift["per_gpu_query_limit"] == 128 and shift["workers"] == 256
    assert shift["progress"]["done"] == 20 and shift["updated_at"]


def test_policy_shift_rate_includes_idle_time_and_resets_after_restart(setup, monkeypatch):
    _, root, collector = setup
    clock = [0.0]
    monkeypatch.setattr("scripts.research_journal.live_progress.time.monotonic", lambda: clock[0])
    owner = {"pid": 123, "start_ticks": "1"}
    assert collector.policy_shift_rate(root, owner, 10, 90, "probing")["per_minute"] is None
    clock[0] = 30
    value = collector.policy_shift_rate(root, owner, 25, 75, "probing")
    assert value["per_minute"] == 30 and value["eta_seconds"] == 150
    clock[0] = 60
    assert collector.policy_shift_rate(root, owner, 25, 75, "probing")["per_minute"] == 15
    assert collector.policy_shift_rate(root, {**owner, "start_ticks": "2"}, 25, 75, "probing")[
        "per_minute"] is None
    clock[0] = 90
    assert collector.policy_shift_rate(root, {**owner, "start_ticks": "2"}, 1, 99, "probing")[
        "per_minute"] is None
    assert collector.policy_shift_rate(root, owner, 25, 75, "stopped")["eta_seconds"] is None
    assert root not in collector.policy_shift_samples


def test_policy_shift_router_reports_only_live_query_counts(setup, monkeypatch):
    _, _, collector = setup
    monkeypatch.setattr("scripts.research_journal.live_progress.urllib.request.urlopen", lambda *args, **kwargs:
                        io.StringIO(json.dumps({"per_gpu_query_limit": 128, "active": {"0": 120, "1": 122},
                                                "endpoints": "DO_NOT_EXPOSE"})))
    result = collector.policy_shift_gpu_queries("http://127.0.0.1:18008/health")
    assert result == {"per_gpu_query_limit": 128, "active": {"0": 120, "1": 122}}
    assert collector.policy_shift_gpu_queries("http://external.example/health") == {}


def test_http_snapshot_remains_available_while_one_background_scan_is_blocked(setup, monkeypatch):
    _, _, collector = setup
    started, release = threading.Event(), threading.Event()
    calls = []

    def blocked_get():
        calls.append(True)
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(collector, "get", blocked_get)
    try:
        initial = collector.snapshot()
        assert initial["initializing"] and initial["result_tables"] == []
        assert started.wait(timeout=1)
        collector.cached = {"generated_at": "previous", "result_tables": [{"id": "ready"}]}
        for _ in range(3):
            snapshot = collector.snapshot()
            assert snapshot["generated_at"] == "previous" and snapshot["refreshing"]
            assert not snapshot["initializing"]
        assert len(calls) == 1
    finally:
        release.set()
        collector.refresh_thread.join(timeout=5)


def test_checkpoint_progress_updates_even_when_historical_scan_is_blocked(setup, monkeypatch):
    _, root, collector = setup
    write(root / "checkpoint_queue_manifest.json", {
        "journal_id": "checkpoint-record", "source_journal_ids": ["training-record"],
        "live_overview": True, "expected_checkpoints": 156, "initial_reused_outcomes": 3454,
    })
    dashboard = {"kind": "five_task_checkpoint_queue", "controller_pid": 123, "phase": "running",
                 "workers": 208, "checkpoints": 153, "rows": [], "task_counts": {"completed": 4000},
                 "gpu_queries": {"active": {"8007": 128, "8006": 80},
                                 "per_endpoint_query_limits": {"8007": 128, "8006": 80}}}
    path = write(root / "dashboard.json", dashboard)
    started, release = threading.Event(), threading.Event()

    def blocked_get():
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(collector, "get", blocked_get)
    try:
        first = collector.snapshot()
        assert started.wait(timeout=1) and first["initializing"]
        queue = first["checkpoint_queues"][0]
        assert queue["live_overview"] and queue["expected_checkpoints"] == 156
        assert queue["initial_reused_outcomes"] == 3454 and queue["journal_id"] == "checkpoint-record"
        assert queue["gpu_queries"]["per_endpoint_query_limits"] == {"8007": 128, "8006": 80}
        write(path, {**dashboard, "updated_at": "newer", "task_counts": {"completed": 4010}})
        collector.checkpoint_cached_at = 0
        updated = collector.snapshot()
        assert updated["checkpoint_queues"][0]["task_counts"]["completed"] == 4010
        assert updated["refreshing"] and updated["initializing"]
    finally:
        release.set()
        collector.refresh_thread.join(timeout=5)


def test_checkpoint_queue_projection_exposes_progress_and_aliases_without_prompts_or_secrets(setup):
    _, root, collector = setup
    write(root / "checkpoint_queue_manifest.json", {"journal_id": "record", "source_journal_ids": ["source"]})
    write(root / "dashboard.json", {"kind": "five_task_checkpoint_queue", "controller_pid": 123,
                                   "phase": "running", "workers": 100, "skipped": 1,
                                   "task_counts": {"running": 100, "completed": 5}, "private": "DO_NOT_EXPOSE",
                                   "rows": [{"id": "cp", "method": "gepa", "point": 10, "alias": "cp5",
                                             "content": "DO_NOT_EXPOSE", "splits": {"val": {
                                                 "done": 5, "total": 50, "score": 0, "trace": "DO_NOT_EXPOSE"}}}]})
    data = collector.get()
    queue = data["checkpoint_queues"][0]
    assert queue["task_counts"]["running"] == queue["workers"] == 100
    assert queue["rows"][0]["alias"] == "cp5" and queue["rows"][0]["splits"]["val"]["score"] == 0
    assert queue["source_journal_ids"] == ["source"] and queue["phase"] == "running"
    assert "DO_NOT_EXPOSE" not in json.dumps(data)


def optimized(root, method="gepa"):
    output = queue(root, {"method": method}, "running")
    write(output / "config.json", {"train": [{"task_id": "a"}, {"task_id": "b"}],
                                   "validation": [{"task_id": "v1"}, {"task_id": "v2"}],
                                   "test": [{"task_id": "t1"}, {"task_id": "t2"}]})
    return output


def rollout(output, name, split, task, *, completed=True, label="validation", prompt="p1"):
    folder = output / "rollouts" / name
    path = write(folder / "request.json", {"label": label, "item": {"split": split, "task_id": task},
                                          "prompt_sha256": prompt, "private": "DO_NOT_EXPOSE"})
    if completed:
        write(folder / "result.json", {"id": task, "hard": 0, "trace": "DO_NOT_EXPOSE"})
    return path


def test_premise_cursor_and_valid_zero_test_results_have_separate_progress(setup):
    _, root, collector = setup
    output = queue(root, {})
    write(output / "train/state.json", {"cursor": 1, "config": {"order": [["bfcl", "a"], ["bfcl", "b"]]},
                                       "states": "DO_NOT_EXPOSE"})
    write(output / "evaluation/records/t1.json", {"task_id": "t1", "status": "completed", "score": 0})
    write(output / "evaluation/records/unplanned.json", {"task_id": "unplanned", "status": "completed", "score": 1})
    row = first(collector)
    assert row["train"]["done"] == 1 and row["train"]["ratio"] == 0.5
    assert row["test"]["done"] == 1 and row["test"]["errors"] == 0
    assert row["state"] == "running" and row["phase"] == "training"
    assert "DO_NOT_EXPOSE" not in json.dumps(collector.get())


def test_premise_scores_use_first_parent_once_per_committed_task_and_exclude_test_errors(setup):
    _, root, collector = setup
    output = queue(root, {})
    write(output / "train/state.json", {"cursor": 2})
    write(output / "train/iterations/iter_0001.json", {
        "task_id": "a", "parents": [{"score": 0, "trace": "DO_NOT_EXPOSE"}, {"score": 1}],
        "candidates": [{"current_official_score": 1}],
    })
    write(output / "train/iterations/iter_0002.json", {"task_id": "b", "parents": [{"score": 1}]})
    write(output / "train/iterations/iter_0003.json", {"task_id": "c", "parents": [{"score": 1}]})
    write(output / "evaluation/records/t1.json", {"task_id": "t1", "status": "completed", "score": 0})
    write(output / "evaluation/records/t2.json", {"task_id": "t2", "status": "error"})
    scores = first(collector)["scores"]
    assert scores["train"]["value"] == 0.5 and scores["train"]["count"] == 2
    assert scores["test"]["value"] == 0 and scores["test"]["count"] == 1
    assert scores["test"]["errors"] == 1 and not scores["test"]["complete"]
    assert "DO_NOT_EXPOSE" not in repr(collector.files.entries)


def test_appworld_online_and_offline_use_tgc_and_count_only_measured_outcomes(setup):
    _, root, collector = setup
    output = queue(root, {"benchmark": "appworld", "train_count": 3})
    write(output / "train/state.json", {"cursor": 3})
    for index, parents in enumerate((
        [{"score": 1.0, "details": {"success": False}}, {"score": 1.0, "details": {"success": True}}],
        [{"score": 1.0, "details": {"success": True}}],
        [{"score": 1.0}],  # Missing success evidence must not be guessed from a rounded score.
    ), 1):
        write(output / f"train/iterations/iter_{index:04d}.json", {"task_id": str(index), "parents": parents})
    write(output / "evaluation/records/t1.json", {
        "task_id": "t1", "status": "completed", "score": 1.0, "success": False,
    })
    write(output / "evaluation/records/t2.json", {
        "task_id": "t2", "status": "completed", "score": 0.9, "session_status": "unsuccessful",
    })
    scores = first(collector)["scores"]
    assert scores["train"]["value"] == 0.5
    assert scores["train"]["count"] == 2 and scores["train"]["successes"] == 1
    assert scores["test"]["value"] == 0 and scores["test"]["successes"] == 0
    assert scores["test"]["complete"]
    assert all(score["metric"] == "appworld_tgc_v1" and "TGC" in score["label"] for score in scores.values())


def native_final_record(task="t1", index=0, success=False):
    return {"task_id": task, "index": index, "success": success, "score": int(success),
            "official_reward": float(success), "actor_prompt_sha256": "a" * 64,
            "session_id": "session", "run_id": "run", "attempt": 1, "cost_reports": "DO_NOT_EXPOSE"}


@pytest.mark.parametrize("method", ["ace", "premise"])
@pytest.mark.parametrize("benchmark", ["bfcl", "appworld", "terminalbench2"])
def test_shared_reflector_native_test_records_without_status_show_completed_test_and_score(setup, method, benchmark):
    _, root, collector = setup
    output = queue(root, {"method": method, "benchmark": benchmark}, status="completed")
    if method == "ace":
        write(output / "train/experiment_config.json", {"task_order": [[benchmark, "a"], [benchmark, "b"]]})
        (output / "train/online_metrics.jsonl").write_text("\n".join(
            json.dumps({"benchmark_slug": benchmark, "task_id": task, "score": 1, "success": True})
            for task in ["a", "b"]) + "\n")
    else:
        write(output / "train/state.json", {"cursor": 2})
    first_record = native_final_record()
    if benchmark == "appworld":
        first_record["official_reward"] = .99  # A partial outcome still has TGC zero.
    first_path = write(output / "evaluation/records/000.json", first_record)
    write(output / "evaluation/records/001.json", native_final_record("t2", 1, True))
    row = first(collector)
    assert row["state"] == row["phase"] == "completed"
    assert row["test"]["complete"] and row["test"]["done"] == row["test"]["total"] == 2
    assert row["scores"]["test"]["value"] == .5 and row["scores"]["test"]["count"] == 2
    if benchmark == "appworld":
        assert row["scores"]["test"]["metric"] == "appworld_tgc_v1"
        assert row["scores"]["test"]["successes"] == 1
    assert json.loads(first_path.read_text()) == first_record  # Projection never edits raw evidence.
    assert "DO_NOT_EXPOSE" not in json.dumps(collector.get())
    assert "DO_NOT_EXPOSE" not in repr(collector.files.entries)


@pytest.mark.parametrize("field,value", [
    ("index", None), ("index", -1), ("index", True), ("index", 1),
    ("success", None), ("score", True), ("score", 1),
    ("official_reward", None), ("official_reward", float("nan")), ("official_reward", 1.1),
    ("session_id", ""), ("run_id", None), ("actor_prompt_sha256", "unverified"),
    ("infrastructure_error", "connection failed"), ("status", "running"), ("task_id", "foreign"),
])
def test_statusless_partial_wrong_task_or_infrastructure_records_do_not_become_measured_test(setup, field, value):
    _, root, collector = setup
    output = queue(root, {})
    write(output / "train/state.json", {"cursor": 2})
    record = {**native_final_record(), field: value}
    write(output / "evaluation/records/000.json", record)
    row = first(collector)
    assert row["test"]["done"] == row["scores"]["test"]["count"] == 0
    assert row["scores"]["test"]["value"] is None and not row["test"]["complete"]


def test_shared_reflector_profile_projects_actor_temperature_for_cohort_matching(setup):
    _, root, collector = setup
    queue(root, {})
    path = root / "launch_manifest.json"
    plan = json.loads(path.read_text())
    plan.update(profile={"actor_temperature": 0, "analysis_temperature": .7}, inference_profile_sha256="profile")
    write(path, plan)
    batch = collector.get()["batches"][0]
    assert batch["temperature"] == 0 and batch["inference_profile_sha256"] == "profile"


def test_ace_appworld_legacy_metrics_use_official_status_instead_of_partial_score(setup):
    _, root, collector = setup
    output = queue(root, {"method": "ace", "benchmark": "appworld"})
    write(output / "train/experiment_config.json", {"task_order": [["appworld", "a"], ["appworld", "b"]]})
    (output / "train/online_metrics.jsonl").write_text("\n".join(json.dumps(row) for row in (
        {"benchmark_slug": "appworld", "task_id": "a", "score": 0.9, "status": "unsuccessful"},
        {"benchmark_slug": "appworld", "task_id": "b", "score": 1.0, "status": "success"},
    )))
    score = first(collector)["scores"]["train"]
    assert score["value"] == 0.5 and score["successes"] == 1
    assert score["count"] == score["total"] == 2 and score["complete"]


def test_optimizer_scores_keep_candidate_validation_separate_and_preserve_zero(setup):
    _, root, collector = setup
    output = optimized(root)
    write(output / "online/a.json", {"task_id": "a", "hard": 0})
    rollout(output, "old1", "val", "v1")
    rollout(output, "old2", "val", "v2")
    new = rollout(output, "new1", "val", "v1", prompt="p2")
    write(new.parent / "result.json", {"id": "v1", "hard": 1})
    os.utime(new, (new.stat().st_mtime + 10, new.stat().st_mtime + 10))
    scores = first(collector)["scores"]
    assert scores["train"]["value"] == 0 and scores["train"]["count"] == 1
    assert scores["val"]["value"] == 1 and scores["val"]["count"] == 1
    assert not scores["val"]["complete"] and scores["test"]["value"] is None


def test_completed_optimizer_uses_committed_summary_only_for_matching_frozen_policy(setup):
    _, root, collector = setup
    output = optimized(root)
    for task in ("a", "b"):
        write(output / f"online/{task}.json", {"task_id": task, "hard": 0})
    write(output / "training_result.json", {})
    write(output / "frozen_instruction.json", {"content": "final policy"})
    summary = {"complete": True, "tasks": 2, "score": 0,
               "prompt_sha256": hashlib.sha256(b"final policy").hexdigest()}
    write(output / "summary.json", summary)
    # Corrupt unrelated history must not be reopened after the final commit.
    path = output / "rollouts/unrelated/request.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken")
    row = first(collector)
    assert row["train"]["complete"] and row["test"]["complete"]
    assert row["scores"]["test"]["value"] == 0 and collector.files.issues == 0
    summary["prompt_sha256"] = "a different policy"
    write(output / "summary.json", summary)
    assert not first(collector)["test"]["complete"]


def test_ace_score_deduplicates_task_ids_and_ignores_infrastructure_errors(setup):
    _, root, collector = setup
    output = queue(root, {"method": "ace"})
    write(output / "train/experiment_config.json", {"task_order": [["bfcl", "a"], ["bfcl", "b"]]})
    (output / "train/online_metrics.jsonl").write_text(
        '\n'.join(json.dumps(row) for row in [
            {"benchmark_slug": "bfcl", "task_id": "a", "score": 0, "status": "unsuccessful"},
            {"benchmark_slug": "bfcl", "task_id": "a", "score": 0, "status": "unsuccessful"},
            {"benchmark_slug": "bfcl", "task_id": "b", "score": 1, "status": "error"},
        ]) + '\n')
    scores = first(collector)["scores"]
    assert scores["train"]["value"] == 0 and scores["train"]["count"] == 1


def test_ace_scoring_before_curator_does_not_advance_training(setup):
    _, root, collector = setup
    output = queue(root, {"method": "ace"})
    write(output / "train/experiment_config.json", {"task_order": [["bfcl", "a"], ["bfcl", "b"]]})
    session = output / "train/run1/sessions/session1"
    write(session / "config.json", {"benchmark": "bfcl", "task_id": "a"})
    write(session / "results.json", {"status": "unsuccessful", "score": 0})
    assert first(collector)["train"]["done"] == 0
    write(session / "agent/playbook_checkpoint.json", {})
    assert first(collector)["train"]["done"] == 1


def test_separate_final_queue_reports_test_activity_and_completion_without_restarting_training(setup):
    _, root, collector = setup
    output = queue(root, {"method": "ace"}, status="test_pending")
    write(output / "train/experiment_config.json", {"task_order": [["bfcl", "a"], ["bfcl", "b"]]})
    (output / "train/online_metrics.jsonl").write_text('\n'.join(json.dumps({
        "benchmark_slug": "bfcl", "task_id": task, "score": 0, "status": "unsuccessful",
    }) for task in ("a", "b")) + '\n')
    write(output / "evaluation/frozen_checkpoint.json", {"playbook": "final"})
    marker = {"phase": "running", "controller_pid": 123, "queue_output": str(root / "final_queue"),
              "policy_sha256": hashlib.sha256(b"final").hexdigest()}
    write(output / "evaluation/live_evaluation.json", marker)
    assert first(collector)["phase"] == "test"
    write(output / "evaluation/records/000.json", {"task_id": "t1", "status": "completed", "score": 0})
    write(output / "evaluation/records/001.json", {"task_id": "t2", "status": "completed", "score": 1})
    row = first(collector)
    assert row["phase"] == "completed" and row["scores"]["test"]["value"] == 0.5
    marker["policy_sha256"] = "other-context"
    write(output / "evaluation/live_evaluation.json", marker)
    assert first(collector)["phase"] == "test_pending"


def test_gepa_initial_validation_does_not_look_like_completed_training(setup):
    _, root, collector = setup
    output = optimized(root)
    rollout(output, "v1", "val", "v1")
    rollout(output, "v2", "val", "v2", completed=False)
    row = first(collector)
    assert row["phase"] == "validation"
    assert row["train"]["done"] == row["test"]["done"] == 0
    assert row["current"]["done"] == 1 and row["current"]["total"] == 2


def test_skillopt_validation_rounds_are_isolated_by_candidate_hash(setup):
    _, root, collector = setup
    output = optimized(root, "skillopt")
    a = rollout(output, "old1", "val", "v1")
    rollout(output, "old2", "val", "v2")
    write(output / "online/a.json", {"task_id": "a"})
    b = rollout(output, "new1", "val", "v1", prompt="p2", completed=False)
    os.utime(b, (a.stat().st_mtime + 10, a.stat().st_mtime + 10))
    row = first(collector)
    assert row["current"]["label"] == "후보 Offline-val"
    assert row["current"]["done"] == 0 and row["current"]["total"] == 2


def test_partial_batch_launch_uses_planned_size_not_request_file_count(setup):
    _, root, collector = setup
    output = optimized(root)
    rollout(output, "a", "train", "a", label="train_evaluation_0002")
    row = first(collector)
    assert row["current"]["done"] == 1 and row["current"]["total"] == 2
    assert row["phase"] == "training"


def test_last_train_exposure_waits_for_optimizer_and_test_tracks_each_result(setup):
    _, root, collector = setup
    output = optimized(root)
    for task in ("a", "b"):
        write(output / f"online/{task}.json", {"task_id": task})
    assert not first(collector)["train"]["complete"]
    write(output / "training_result.json", {})
    write(output / "frozen_instruction.json", {})
    rollout(output, "t1", "test", "t1", label="final_test")
    rollout(output, "t2", "test", "t2", label="final_test", completed=False)
    row = first(collector)
    assert row["phase"] == "test" and row["train"]["complete"]
    assert row["test"]["done"] == 1 and row["test"]["ratio"] == 0.5


def test_error_results_count_as_processed_but_never_clean_completion(setup):
    _, root, collector = setup
    output = queue(root, {}, "completed")
    write(output / "train/state.json", {"cursor": 2})
    for task in ("t1", "t2"):
        write(output / f"evaluation/records/{task}.json", {"task_id": task, "status": "error"})
    row = first(collector)
    assert row["test"]["done"] == 2 and row["test"]["errors"] == 2
    assert not row["test"]["complete"] and row["state"] == "error"


def test_dead_worker_is_stopped_even_if_saved_status_says_running(setup):
    _, root, collector = setup
    queue(root, {"pid": 999})
    assert first(collector)["state"] == "stopped"


@pytest.mark.parametrize("method", ["ace", "premise", "gepa", "skillopt"])
def test_scheduled_pause_preserves_progress_and_overrides_cancellation_status(setup, method):
    _, root, collector = setup
    output = queue(root, {"method": method, "pid": 999}, "paused")
    write(output / "status.json", {"phase": "error"})
    if method == "ace":
        write(output / "train/experiment_config.json", {"task_order": [["bfcl", "a"], ["bfcl", "b"]]})
        (output / "train/online_metrics.jsonl").write_text('{"benchmark_slug":"bfcl","task_id":"a"}\n')
    elif method == "premise":
        write(output / "train/state.json", {"cursor": 1})
    else:
        write(output / "config.json", {"train": [{"task_id": "a"}, {"task_id": "b"}], "test": []})
        write(output / "online/a.json", {"task_id": "a"})
    write(root / "control_schedule.json", {"phase": "paused", "resume_at": "2026-10-22T10:00:00+09:00",
                                          "private": "DO_NOT_EXPOSE"})
    result = collector.get()
    batch = result["batches"][0]
    row = batch["runs"][0]
    assert row["state"] == row["phase"] == "paused"
    assert row["pid"] is None and row["current"] is None and row["train"]["done"] == 1
    assert result["summary"]["paused"] == 1 and result["summary"]["attention"] == 0
    assert batch["schedule"] == {"phase": "paused", "resume_at": "2026-10-22T10:00:00+09:00"}
    assert "DO_NOT_EXPOSE" not in json.dumps(result)


def test_waiting_runs_have_planned_totals_and_superseded_queues_are_excluded(setup):
    _, root, collector = setup
    output = queue(root, {"method": "gepa"}, "pending")
    snapshot = root / "snapshot"
    plan = json.loads((root / "launch_manifest.json").read_text())
    plan["snapshot"] = str(snapshot)
    write(root / "launch_manifest.json", plan)
    write(snapshot / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json", {
        "bfcl_42": {"train": ["a", "b"], "val": ["v"], "test": ["t1", "t2"]}})
    row = first(collector)
    assert row["state"] == "pending" and row["train"]["total"] == row["test"]["total"] == 2
    assert not output.exists()
    write(root / "superseded_by.json", {"output": "replacement"})
    assert collector.get()["summary"]["runs"] == 0


def test_malformed_rewrite_keeps_last_good_counter_in_cache(tmp_path):
    path = write(tmp_path / "state.json", {"cursor": 1, "secret": "DO_NOT_EXPOSE"})
    reader = FileCache()
    assert reader.read(path, ("cursor",)) == {"cursor": 1}
    path.write_text('{"cursor":')
    assert reader.read(path, ("cursor",)) == {"cursor": 1}
    assert reader.issues == 1


def test_outside_paths_cannot_be_read_or_disclosed(setup, tmp_path):
    _, root, collector = setup
    queue(root, {"output": str(tmp_path.parent / "private")})
    assert collector.get()["batches"] == []


@pytest.mark.parametrize("benchmark,train_size,test_size", [
    ("bfcl", 50, 50), ("appworld", 50, 50), ("terminalbench2", 30, 29)])
def test_matrix_pending_jobs_use_fixed_planned_splits_without_creating_run_directories(
    setup, benchmark, train_size, test_size,
):
    _, root, collector = setup
    output = root / "runs/gepa/train"
    job = {"id": "matrix", "method": "gepa", "variant": "gepa_val", "benchmark": benchmark, "seed": 42,
           "output": str(output.parent), "train_dir": str(output), "train_ids": [f"a{i}" for i in range(train_size)],
           "val_ids": ["v"], "test_ids": [f"t{i}" for i in range(test_size)], "validation_source": "val"}
    write(root / "launch_manifest.json", {"jobs": [job], "max_parallel": 6})
    write(root / "progress.json", {"jobs": {"matrix": {"phase": "pending", "launches": 0}}})
    row = first(collector)
    assert row["state"] == "pending" and row["train"]["total"] == train_size
    assert row["test"]["total"] == test_size and row["variant_label"] == "GEPA · 기본 val"
    assert not output.exists()


def test_matrix_reads_nested_training_results_dictionary_progress_and_live_test_phase(setup, monkeypatch):
    _, root, collector = setup
    output = root / "runs/gepa/bfcl/s42"
    train = output / "train"
    job = {"id": "matrix", "method": "gepa", "variant": "gepa_train_as_val", "benchmark": "bfcl", "seed": 42,
           "output": str(output), "train_dir": str(train), "train_ids": ["a", "b"], "val_ids": ["heldout"],
           "test_ids": ["t1", "t2"], "validation_source": "train",
           "command": ["worker", str(root / "launch_manifest.json")]}
    write(root / "launch_manifest.json", {"jobs": [job], "max_parallel": 6, "server_max_num_seqs": 256})
    write(root / "progress.json", {"jobs": {"matrix": {"phase": "running", "pid": 123}}})
    write(root / "controller_process.json", {"identity": {"pid": 123}})
    monkeypatch.setattr("scripts.research_journal.live_progress.live_pid",
                        lambda pid, target: pid == 123 and target == root)
    write(train / "config.json", {"train": [{"task_id": "a"}, {"task_id": "b"}],
                                 "validation": [{"task_id": "a"}, {"task_id": "b"}],
                                 "test": [{"task_id": "t1"}, {"task_id": "t2"}], "split_manifest_sha256": "fixed"})
    for task, score in [("a", 1), ("b", 0)]:
        write(train / f"online/{task}.json", {"task_id": task, "hard": score})
        path = rollout(train, "val_" + task, "val", task)
        write(path.parent / "result.json", {"id": task, "hard": score})
    write(train / "training_result.json", {})
    write(train / "frozen_instruction.json", {"content": "DO_NOT_EXPOSE"})
    write(output / "job_status.json", {"phase": "test", "completed": 1, "total": 2})
    rollout(train, "test", "test", "t1", label="final_test")
    row = first(collector)
    assert row["state"] == "running" and row["phase"] == "test" and row["pid"] == 123
    assert row["train"]["complete"] and row["test"]["done"] == 1 and row["test"]["total"] == 2
    assert row["scores"]["train"]["value"] == 0.5 and row["scores"]["test"]["value"] == 0
    assert row["scores"]["val"]["label"] == "선택 평가 · Train"
    assert row["split_manifest_sha256"] == "fixed" and row["validation_source"] == "train"
    assert collector.get()["summary"]["running"] == 1
    assert "DO_NOT_EXPOSE" not in json.dumps(collector.get())


def test_matrix_legacy_test_marker_overrides_training_activity_and_completed_queue_phase(setup):
    _, root, collector = setup
    output = root / "runs/ace/bfcl/s42"
    job = {"id": "matrix", "method": "ace", "variant": "ace_aud", "benchmark": "bfcl", "seed": 42,
           "output": str(output), "train_dir": str(output / "train"), "train_count": 2, "test_ids": ["t1", "t2"]}
    write(root / "launch_manifest.json", {"jobs": [job]})
    write(root / "progress.json", {"jobs": {"matrix": {"phase": "running", "pid": 123}}})
    write(output / "train/experiment_config.json", {"task_order": [["bfcl", "a"], ["bfcl", "b"]]})
    (output / "train/online_metrics.jsonl").write_text("\n".join(
        json.dumps({"benchmark_slug": "bfcl", "task_id": task, "score": 1}) for task in ["a", "b"]) + "\n")
    write(output / "job_status.json", {"phase": "test"})
    write(output / "evaluation/records/0.json", {"task_id": "t1", "status": "completed", "score": 0})
    row = first(collector)
    assert row["phase"] == "test" and row["train"]["done"] == 2 and row["test"]["done"] == 1
    write(output / "evaluation/records/1.json", {"task_id": "t2", "status": "completed", "score": 1})
    write(root / "progress.json", {"jobs": {"matrix": {"phase": "completed"}}})
    write(output / "job_status.json", {"phase": "completed"})
    row = first(collector)
    assert row["state"] == "completed" and row["scores"]["test"]["value"] == 0.5


def test_ace_repeated_epoch_counts_exposures_without_premature_completion(setup):
    _, root, collector = setup
    output = queue(root, {"method": "ace", "epoch1_count": 2})
    write(output / "train/experiment_config.json", {"task_order": [
        ["bfcl", "a"], ["bfcl", "b"], ["bfcl", "b"], ["bfcl", "a"]]})
    metrics = output / "train/online_metrics.jsonl"
    rows = [{"benchmark_slug": "bfcl", "task_id": task, "score": score, "status": "successful"}
            for task, score in [("a", 0), ("b", 1), ("b", 0)]]
    metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    row = first(collector)
    assert row["train"]["done"] == 3 and row["train"]["total"] == 4
    assert not row["train"]["complete"]
    assert row["scores"]["train"]["count"] == 3
    assert row["scores"]["train"]["value"] == pytest.approx(1 / 3)


def test_optimizer_repeated_epoch_progress_uses_epoch_and_unique_final_test_label(setup):
    _, root, collector = setup
    output = optimized(root)
    write(output / "config.json", {"train": [
        {"task_id": "a"}, {"task_id": "b"}, {"task_id": "b", "epoch": 2}, {"task_id": "a", "epoch": 2}],
        "validation": [], "test": [{"task_id": "t1"}, {"task_id": "t2"}]})
    for key, task, epoch, score in [("a", "a", 1, 0), ("b", "b", 1, 1), ("epoch2_b", "b", 2, 0)]:
        write(output / f"online/{key}.json", {"task_id": task, "epoch": epoch, "hard": score})
    rollout(output, "test1", "test", "t1", label="epoch2_final_test")
    row = first(collector)
    assert row["train"]["done"] == 3 and row["train"]["total"] == 4
    assert row["scores"]["train"]["value"] == pytest.approx(1 / 3)
    assert row["test"]["done"] == 1 and row["test"]["total"] == 2
