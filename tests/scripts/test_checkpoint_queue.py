# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import io
import pickle
from concurrent.futures import Future
from pathlib import Path

import pytest

from scripts.utils import checkpoint_sources as sources
from scripts.utils.checkpoint_eval_proxy import wait_for_result
from scripts.utils.checkpoint_queue import Queue, collect, discover_roots, dispatch, worker_limit
from scripts.utils.run_qwen_isolated_when_free import write


def stream(tmp_path, method="ace", count=15):
    return {"id": "stream", "method": method, "benchmark": "bfcl", "seed": 42, "variant": "baseline",
            "model": "openai/Qwen/Qwen3.6-35B-A3B", "temperature": 0.7,
            "output": str(tmp_path), "train_dir": str(tmp_path), "train_total": count,
            "splits": {"train": [f"t{i}" for i in range(count)], "val": ["v1", "v2"], "test": ["z1", "z2"]}}


def cp(point, content="policy"):
    return {"point": point, "content": content, "selection": "fixture", "source": "fixture"}


def test_checkpoint_registration_is_immutable_idempotent_and_skips_only_unchanged_neighbors(tmp_path):
    queue, spec = Queue(tmp_path / "queue"), stream(tmp_path)
    assert queue.register(spec, cp(5))
    assert not queue.register(spec, {**cp(5), "source": "later recovery evidence"})
    assert queue.register(spec, cp(10))
    dashboard = queue.dashboard()
    assert dashboard["checkpoints"] == 2 and dashboard["skipped"] == 1
    assert dashboard["task_counts"]["pending"] == 19
    assert dashboard["rows"][1]["alias"] == "stream_cp005"
    assert queue.register(spec, cp(15, "different policy"))
    assert queue.dashboard()["task_counts"]["pending"] == 38
    with pytest.raises(ValueError, match="immutable"):
        queue.register(spec, cp(5, "changed policy"))


def test_invalidated_learning_stream_never_dispatches_or_contributes_to_scores(tmp_path):
    queue = Queue(tmp_path / "queue")
    queue.register(stream(tmp_path), cp(5))
    task = queue.claim(queue.pending(1)[0])
    queue.finish(task, {"status": "completed", "score": 1})
    with queue.db:
        queue.db.execute("INSERT INTO stream_invalidations VALUES(?,?)", ("stream", "Incorrect training reward"))
    assert not queue.pending(100)
    dashboard = queue.dashboard()
    assert dashboard["rows"] == [] and dashboard["task_counts"] == {}
    assert dashboard["invalidated_checkpoints"] == 1
    assert queue.db.execute("SELECT count(*) FROM tasks").fetchone()[0] == 19


def test_resume_scope_preserves_excluded_work_and_completed_zero_scores(tmp_path):
    queue = Queue(tmp_path / "queue")
    for name in ("original_aud", "add_only", "teammate"):
        queue.register({**stream(tmp_path), "id": name}, cp(5))
        row = queue.db.execute("SELECT * FROM tasks WHERE cp=? LIMIT 1", (f"{name}_cp005",)).fetchone()
        queue.claim(row)
    selected = ["original_aud"]
    excluded_before = [tuple(r) for r in queue.db.execute("SELECT * FROM tasks WHERE cp!='original_aud_cp005'")]
    assert queue.recover_interrupted(selected) == 1
    row = queue.pending(1, selected)[0]
    queue.finish(queue.claim(row), {"status": "completed", "score": 0})
    assert all(row["cp"] == "original_aud_cp005" for row in queue.pending(100, selected))
    assert queue.pending(100, []) == []
    dashboard = queue.dashboard(stream_ids=selected)
    assert dashboard["task_counts"] == {"completed": 1, "pending": 18}
    assert len(dashboard["rows"]) == 3  # Retain the historical charts and scores.
    assert all(r["state"] == "paused" for r in dashboard["rows"] if not r["selected"])
    assert excluded_before == [tuple(r) for r in queue.db.execute("SELECT * FROM tasks WHERE cp!='original_aud_cp005'")]


def test_explicit_roots_disable_new_cohort_discovery(tmp_path):
    write(tmp_path / "new_teammate/launch_manifest.json", {
        "created_at": "2026-10-31", "model": "Qwen/Qwen3.6-35B-A3B", "jobs": [{}]})
    plan = {"roots": ["original"], "watch_directory": str(tmp_path), "earliest_source": "2026"}
    assert len(discover_roots(plan)) == 2
    assert discover_roots({**plan, "auto_discover": False}) == ["original"]


def test_zero_is_completed_without_retry_and_infrastructure_retries_are_finite(tmp_path):
    queue = Queue(tmp_path / "queue")
    queue.register(stream(tmp_path), cp(5))
    task = queue.claim(queue.pending(1)[0])
    queue.finish(task, {"status": "completed", "score": 0, "budget_failure": {"reason": "output_truncated"}})
    stored = queue.db.execute("SELECT * FROM tasks WHERE id=?", (task["id"],)).fetchone()
    assert stored["status"] == "completed" and stored["attempts"] == 1
    task = queue.claim(queue.pending(1)[0])
    identity = task["id"]
    for attempt in range(1, 4):
        queue.finish(task, {"status": "error", "score": None, "error": "temporary transport failure"})
        row = queue.db.execute("SELECT * FROM tasks WHERE id=?", (identity,)).fetchone()
        assert row["attempts"] == attempt
        if attempt < 3:
            assert row["status"] == "pending"
            task = queue.claim(row)
    assert row["status"] == "error"
    assert queue.dashboard()["rows"][0]["state"] == "error"


def test_proxy_reuses_alias_results_and_rejects_a_different_policy(tmp_path):
    queue, spec = Queue(tmp_path / "queue"), stream(tmp_path)
    queue.register(spec, cp(5))
    queue.register(spec, cp(10))
    row = queue.db.execute("SELECT * FROM tasks WHERE split='test' AND task_id='z1'").fetchone()
    queue.finish(queue.claim(row), {"status": "completed", "score": 0, "partial_score": 0.25})
    marker = {"output": str(queue.output), "stream": "stream", "point": 10}
    result = wait_for_result(marker, "z1", sources.digest("policy"))
    assert result["score"] == 0 and result["partial_score"] == 0.25
    with pytest.raises(ValueError, match="differs"):
        wait_for_result(marker, "z1", sources.digest("future policy"))


def test_unchanged_final_checkpoint_promotes_existing_test_tasks_without_duplicates(tmp_path):
    queue, spec = Queue(tmp_path / "queue"), stream(tmp_path, count=10)
    queue.register(spec, cp(5))
    assert queue.db.execute("SELECT min(priority) FROM tasks WHERE split='test'").fetchone()[0] == 1
    queue.register(spec, cp(10))
    assert queue.db.execute("SELECT count(*) FROM tasks").fetchone()[0] == 14
    assert queue.db.execute("SELECT max(priority) FROM tasks WHERE split='test'").fetchone()[0] == 0


def test_ace_counts_completed_task_ids_and_waits_for_curator_excluding_infrastructure_errors(tmp_path):
    spec = stream(tmp_path, count=10)
    write(tmp_path / "experiment_config.json", {"task_order": [["bfcl", f"t{i}"] for i in range(10)]})
    for index in range(10):
        session = tmp_path / "run/sessions" / str(index)
        write(session / "config.json", {"benchmark": "bfcl", "task_id": f"t{index}"})
        write(session / "results.json", {"status": "unsuccessful" if index < 6 else "error", "score": 0})
        if index != 5:
            write(session / "agent/playbook_checkpoint.json", {"playbook": f"policy{index}", "session_count": 99})
    snapshots = sources.ace(spec, sources.Reader())
    assert [(c["point"], c["content"]) for c in snapshots] == [(5, "policy4")]


def test_ace_recovers_later_boundaries_after_a_logged_task_without_curator_snapshot(tmp_path):
    spec = stream(tmp_path, count=10)
    order = [["bfcl", f"t{i}"] for i in range(10)]
    write(tmp_path / "experiment_config.json", {"task_order": order})
    metrics = []
    for index in range(10):
        session = tmp_path / "run/sessions" / str(index)
        write(session / "config.json", {"benchmark": "bfcl", "task_id": f"t{index}",
                                        "agent_kwargs": {"initial_playbook": f"policy{index - 1}"}})
        write(session / "results.json", {"status": "unsuccessful", "score": 0})
        if index not in {4, 6}:
            write(session / "agent/playbook_checkpoint.json", {"playbook": f"policy{index}"})
        metrics.append({"session_index": index, "benchmark_slug": "bfcl", "task_id": f"t{index}",
                        "status": "unsuccessful"})
    import json

    (tmp_path / "online_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    snapshots = sources.ace(spec, sources.Reader())
    assert [(c["point"], c["content"]) for c in snapshots] == [(5, "policy4"), (10, "policy9")]


def test_gepa_selects_only_candidates_born_by_each_boundary_and_uses_validation(tmp_path):
    spec = stream(tmp_path, "gepa", count=10)
    write(tmp_path / "config.json", {"train": [{"task_id": f"t{i}"} for i in range(10)],
                                    "validation": [{"task_id": "v1"}, {"task_id": "v2"}]})
    path = tmp_path / "upstream/replay_0000/gepa_state.bin"
    path.parent.mkdir(parents=True)
    state = {"program_candidates": [{"shared": "seed"}, {"shared": "future best"}, {"shared": "equal later"}],
             "iteration_ids_by_candidate_idx": ["seed", "i1", "i1"],
             "prog_candidate_val_subscores": [{0: 0.5, 1: 0.5}, {0: 1.0, 1: 1.0}, {0: 1.0, 1: 1.0}],
             "full_program_trace": [{"i": 0, "iteration_id": "i0", "subsample_ids": [f"t{i}" for i in range(5)]},
                                    {"i": 1, "iteration_id": "i1", "subsample_ids": [f"t{i}" for i in range(5, 10)]}]}
    path.write_bytes(pickle.dumps(state))
    snapshots = sources.gepa(spec, sources.Reader())
    assert [(s["point"], s["content"], s["selected_val_score"]) for s in snapshots] == [
        (5, "seed", 0.5), (10, "future best", 1.0)]
    assert snapshots[1]["candidate_index"] == 1


def test_optimizer_pickle_cannot_import_or_instantiate_objects():
    with pytest.raises(ValueError, match="plain data"):
        sources.PlainState(io.BytesIO(pickle.dumps(Path("untrusted")))).load()


def test_skillopt_uses_historical_gate_decisions_instead_of_current_best_file(tmp_path):
    spec = stream(tmp_path, "skillopt", count=10)
    folder = tmp_path / "upstream/replay_0000"
    write(folder / "runtime_state.json", {"last_completed_step": 2})
    write(folder / "history.json", [{"step": 1, "best_step": 0, "best_score": 0.2},
                                   {"step": 2, "best_step": 2, "best_score": 0.8}])
    (folder / "skills").mkdir()
    (folder / "skills/skill_v0000.md").write_text("seed")
    (folder / "skills/skill_v0002.md").write_text("updated")
    (folder / "best_skill.md").write_text("future instruction that must not leak backwards")
    assert [(s["point"], s["content"]) for s in sources.skillopt(spec, sources.Reader())] == [
        (5, "seed"), (10, "updated")]


def test_premise_reconstruction_checks_terminal_beam_and_carries_skipped_updates(tmp_path):
    spec = stream(tmp_path, "premise", count=5)
    for index in range(1, 6):
        if index == 5:
            artifact = {"status": "skipped", "train_update_applied": False}
        else:
            artifact = {"selection": {"selected": ["c", "c"]}, "candidates": [
                {"id": "c", "playbook": {"playbook": f"policy{index}"}, "reflection": {"entry_assessments": []}}]}
        write(tmp_path / f"iterations/iter_{index:04d}.json", {
            **artifact, "iteration": index, "benchmark": "bfcl", "task_id": f"t{index - 1}"})
    state = {"cursor": 5, "config": {"mode": "isolated", "order": [["bfcl", f"t{i}"] for i in range(5)]},
             "states": {"bfcl": {"beam": [{"playbook": "policy4"}, {"playbook": "policy4"}]}}}
    write(tmp_path / "state.json", state)
    assert sources.premise(spec, sources.Reader())[0]["content"] == "policy4"
    state["states"]["bfcl"]["beam"][0]["playbook"] = "unverified policy"
    write(tmp_path / "state.json", state)
    with pytest.raises(ValueError, match="terminal beam"):
        sources.premise(spec, sources.Reader())


def test_completed_final_test_is_reused_only_for_matching_policy(tmp_path):
    queue, spec = Queue(tmp_path / "queue"), stream(tmp_path, count=5)
    queue.register(spec, cp(5))
    frozen = {"playbook": "policy", "model": spec["model"], "temperature_all_roles": 0.7}
    write(tmp_path / "evaluation/frozen_checkpoint.json", frozen)
    write(tmp_path / "evaluation/records/one.json", {"task_id": "z1", "status": "completed", "score": 0})
    queue.import_final(spec)
    assert queue.dashboard()["task_counts"]["completed"] == 1
    write(tmp_path / "evaluation/frozen_checkpoint.json", {**frozen, "temperature_all_roles": 0.0})
    queue.import_final(spec)
    assert queue.dashboard()["task_counts"]["completed"] == 1
    write(tmp_path / "evaluation/frozen_checkpoint.json", {"playbook": "other"})
    write(tmp_path / "evaluation/records/two.json", {"task_id": "z2", "status": "completed", "score": 1})
    queue.import_final(spec)
    assert queue.dashboard()["task_counts"]["completed"] == 1


@pytest.mark.parametrize("workers", [100, 128])
def test_all_methods_and_splits_share_the_configured_slots(tmp_path, workers):
    queue = Queue(tmp_path / "queue")
    for method in ("ace", "premise", "gepa", "skillopt"):
        spec = {**stream(tmp_path, method, count=100), "id": method}
        queue.register(spec, cp(5))

    class Pool:
        submitted = 0

        def submit(self, fn, task):
            self.submitted += 1
            return Future()

    pool, active = Pool(), {}
    dispatch(queue, pool, active, workers=workers)
    dispatch(queue, pool, active, workers=workers)
    assert len(active) == pool.submitted == workers
    dashboard = queue.dashboard(workers=workers)
    assert dashboard["task_counts"]["running"] == dashboard["workers"] == workers
    for future, task in list(active.items())[:5]:
        queue.finish(task, {"status": "completed", "score": 0})
        del active[future]
    dispatch(queue, pool, active, workers=workers)
    assert len(active) == workers and pool.submitted == workers + 5


def test_checkpoint_worker_limit_defaults_and_rejects_invalid_manifest_values():
    assert worker_limit({}) == 100
    assert worker_limit({"workers": 128}) == 128
    for value in (0, -1, 128.5, "128", True):
        with pytest.raises(ValueError, match="positive integer"):
            worker_limit({"workers": value})


def test_watcher_backfills_and_automatically_registers_new_five_task_boundaries(tmp_path):
    root = tmp_path / "experiments/source"
    output = root / "run"
    manifest = tmp_path / "split.json"
    write(manifest, {"bfcl_42": {"train": [f"t{i}" for i in range(10)], "val": ["v"], "test": ["z"]}})
    write(root / "launch_manifest.json", {"model": "Qwen/Qwen3.6-35B-A3B", "jobs": [{
        "id": "ace42", "method": "ace", "benchmark": "bfcl", "seed": 42, "output": str(output),
        "train_dir": str(output / "train"), "command": ["python", "--task-split-manifest", str(manifest)]}]})
    write(output / "train/experiment_config.json", {"task_order": [["bfcl", f"t{i}"] for i in range(10)]})
    queue, reader = Queue(tmp_path / "queue"), sources.Reader()
    plan = {"roots": [str(root)], "watch_directory": str(tmp_path / "experiments"), "earliest_source": "2026"}
    for index in range(10):
        folder = output / f"train/run/sessions/task{index}"
        write(folder / "config.json", {"benchmark": "bfcl", "task_id": f"t{index}"})
        write(folder / "results.json", {"status": "unsuccessful", "score": 0})
        write(folder / "agent/playbook_checkpoint.json", {"playbook": f"policy{index}"})
        if index == 4:
            assert collect(queue, plan, reader) == []
            assert queue.dashboard()["checkpoints"] == 1
    assert collect(queue, plan, reader) == []
    assert [row["point"] for row in queue.dashboard()["rows"]] == [5, 10]
    assert queue.db.execute("SELECT count(*) FROM tasks").fetchone()[0] == 24
    assert (output / "checkpoint_queue.json").is_file()
