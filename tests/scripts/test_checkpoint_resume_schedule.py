# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import sqlite3

import pytest

from scripts.utils import schedule_checkpoint_resume as schedule
from scripts.utils.checkpoint_queue import Queue
from scripts.utils.checkpoint_sources import digest
from scripts.utils.run_qwen_isolated_when_free import write


@pytest.fixture
def finished(tmp_path):
    output = tmp_path / "final"
    queue = Queue(output)
    contexts, jobs = [], []
    policy = "pinned teammate environment policy"
    for method in ("ace", "premise"):
        for mode in ("add_only", "add_update_delete"):
            for seed in (42, 43, 44):
                name = f"{method}_{mode}_{seed}"
                folder = tmp_path / name
                ids = [f"test{i}" for i in range(50)]
                spec = {"id": name, "method": method, "benchmark": "appworld", "seed": seed,
                        "variant": mode, "train_total": 50, "splits": {"test": ids}}
                queue.register(spec, {"point": 50, "content": name, "selection": "final"})
                frozen = folder / "evaluation/frozen_checkpoint.json"
                write(frozen, {"playbook": name, "appworld_policy_override": policy,
                               "appworld_policy_sha256": digest(policy), "temperature_all_roles": 0.7})
                contexts.append({"method": method, "variant": mode, "seed": seed, "stream": name,
                                 "source_output": str(folder), "frozen": str(frozen),
                                 "frozen_sha256": schedule.file_sha(frozen)})
                write(folder / "train/training_complete.json", {"completed_task_count": 50, "expected_task_count": 50})
                write(folder / "train/state.json", {"cursor": 50})
                write(folder / "evaluation/summary.json", {"complete": True, "tasks": 50, "execution_errors": 0})
                if method == "premise":
                    jobs.append({"output": str(folder), "train_dir": str(folder / "train"), "test_ids": ids,
                                 "variant": mode, "seed": seed})
    with queue.db:
        queue.db.execute("UPDATE tasks SET status='completed',result=?", (json.dumps({
            "status": "completed", "score": 0, "score_metric": "appworld_tgc_v1"}),))
    queue.db.close()
    write(output / "final_test_queue_manifest.json", {"contexts": contexts})
    return {"prerequisites": jobs, "teammate_policy_sha256": digest(policy), "final_queue_output": str(output)}


def test_all_six_committed_contexts_and_600_results_release_even_with_zero_tgc(finished):
    evidence = schedule.prerequisites(finished)
    assert evidence["ready"] and evidence["final_queue_drained"]
    assert all(r["test_done"] == 50 and not r["test_errors"] for r in evidence["contexts"])


@pytest.mark.parametrize("condition", ["no_commit", "partial_train", "partial_test", "infra_error", "missing_tgc",
                                       "unpublished_summary", "ace_not_drained"])
def test_partial_or_invalid_outcomes_never_release_the_schedule(finished, condition):
    job = finished["prerequisites"][-1]
    folder = schedule.Path(job["output"])
    if condition == "no_commit":
        (folder / "train/training_complete.json").unlink()
    elif condition == "partial_train":
        write(folder / "train/state.json", {"cursor": 49})
    elif condition == "unpublished_summary":
        write(folder / "evaluation/summary.json", {"complete": False, "tasks": 49, "execution_errors": 0})
    else:
        with schedule.readonly_db(finished["final_queue_output"]) as db:
            pattern = "ace%" if condition == "ace_not_drained" else "premise%"
            task_id = db.execute("""SELECT t.id FROM tasks t JOIN checkpoints c ON c.id=t.cp
                WHERE c.stream LIKE ? LIMIT 1""", (pattern,)).fetchone()[0]
        with sqlite3.connect(folder.parent / "final/queue.sqlite3") as db:
            status = "error" if condition == "infra_error" else "pending"
            if condition == "missing_tgc":
                db.execute("UPDATE tasks SET result=? WHERE id=?", (json.dumps({"status": "completed", "score": 0.8}),
                                                                   task_id))
            else:
                db.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
    assert not schedule.prerequisites(finished)["ready"]


def test_changed_environment_policy_is_rejected(finished):
    manifest = schedule.read(schedule.Path(finished["final_queue_output"]) / "final_test_queue_manifest.json")
    context = manifest["contexts"][-1]
    write(context["frozen"], {**schedule.read(context["frozen"]), "appworld_policy_override": "original"})
    with pytest.raises(ValueError, match="policy/configuration changed"):
        schedule.prerequisites(finished)


def test_queue_pause_owner_changes_are_preserved_without_launching(tmp_path, monkeypatch):
    queue = tmp_path / "queue"
    marker = {"owner": "new_user_pause"}
    write(queue / "pause_dispatch.json", marker)
    monkeypatch.setattr(schedule, "validate_sources", lambda plan: None)
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda *a, **kw: pytest.fail("Premature launch"))
    with pytest.raises(ValueError, match="pause owner changed"):
        schedule.start_queue(tmp_path, {"queue_output": str(queue), "expected_pause_marker": {"owner": "old"}},
                             {"ready": True})
    assert schedule.read(queue / "pause_dispatch.json") == marker


def test_release_waits_for_scoped_controller_heartbeat_and_rechecks_completion(tmp_path, monkeypatch):
    queue, output = tmp_path / "queue", tmp_path / "schedule"
    marker = {"owner": "old_user_pause"}
    write(queue / "pause_dispatch.json", marker)
    write(queue / "checkpoint_queue_manifest.json", {"old": True})
    write(tmp_path / "private.json", {})
    plan = {"queue_output": str(queue), "expected_pause_marker": marker,
            "previous_queue_manifest_sha256": schedule.file_sha(queue / "checkpoint_queue_manifest.json"),
            "queue_plan": {"selected_stream_ids": ["original_aud"], "base_url": "http://127.0.0.1:19007/v1"},
            "private_environment": str(tmp_path / "private.json"), "runtime": str(tmp_path), "command": ["fixture"]}
    monkeypatch.setattr(schedule, "validate_sources", lambda plan: None)
    monkeypatch.setattr(schedule, "alive", lambda record: True)
    monkeypatch.setattr(schedule, "stable_identity", lambda pid: {"boot_id": "boot", "start_ticks": "ticks"})
    monkeypatch.setattr(schedule, "prerequisites", lambda plan: {"ready": True})
    monkeypatch.setattr(schedule.subprocess, "run", lambda *a, **kw: type("Result", (), {"returncode": 0})())

    def launch(*args, **kwargs):
        assert (queue / "pause_dispatch.json").exists()
        assert schedule.read(queue / "checkpoint_queue_manifest.json") == plan["queue_plan"]
        write(queue / "controller_process.json", {"pid": 123})
        write(queue / "heartbeat.json", {"selected_stream_ids": ["original_aud"]})
        return type("Child", (), {"pid": 123})()

    monkeypatch.setattr(schedule.subprocess, "Popen", launch)
    record = schedule.start_queue(output, plan, {"ready": True})
    assert record["pid"] == 123 and record["workers"] == 100
    assert not (queue / "pause_dispatch.json").exists()
    assert schedule.read(output / "started.json") == record


def test_source_scope_rejects_add_only_teammate_and_replaced_outputs(finished, monkeypatch):
    specs = [{"id": f"{method}_{bench}_{seed}", "method": method, "benchmark": bench, "seed": seed,
              "playbook_edit_mode": "add_update_delete"} for method in ("ace", "premise", "gepa", "skillopt")
             for bench in ("bfcl", "appworld", "terminalbench2") for seed in (42, 43, 44)]
    plan = {"queue_output": finished["final_queue_output"],
            "queue_plan": {"roots": ["original"], "selected_stream_ids": [s["id"] for s in specs]},
            "source_specs_sha256": digest(json.dumps(specs, sort_keys=True))}
    monkeypatch.setattr(schedule.sources, "streams", lambda *args: specs)
    assert len(schedule.validate_sources(plan)) == 36
    specs[0]["playbook_edit_mode"] = "add_only"
    with pytest.raises(ValueError, match="Only AUD"):
        schedule.validate_sources(plan)
    specs[0]["playbook_edit_mode"] = "add_update_delete"
    specs[0]["appworld_policy_sha256"] = "teammate"
    with pytest.raises(ValueError, match="teammate-policy"):
        schedule.validate_sources(plan)
    specs[0].pop("appworld_policy_sha256")
    specs[0]["id"] = "replacement"
    with pytest.raises(ValueError, match="output paths changed"):
        schedule.validate_sources(plan)
