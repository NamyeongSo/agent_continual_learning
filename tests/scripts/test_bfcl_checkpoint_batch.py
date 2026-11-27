# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from pathlib import Path

import pytest

from scripts.ace.run_experiment import task_order_from_train_manifest
from scripts.utils import evaluate_bfcl_checkpoints as batch
from tests.scripts.test_ace_checkpoint_evaluation import create_session

MANIFEST = Path(__file__).resolve().parents[2] / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"


def create_launch(tmp_path, method):
    root = tmp_path / method
    root.mkdir()
    rows = []
    modes = ("sequential", "interleaved") if method == "ace" else ("sequential",)
    for mode in modes:
        for seed in (42, 43, 44):
            order = task_order_from_train_manifest(MANIFEST, list(batch.BENCHMARKS), seed, None, mode)
            name = f"s{seed}_{mode}_t3"
            output = root / (f"{mode}_s{seed}" if method == "ace" else name)
            row = {"name": name, "mode": mode, "seed": seed, "past": 3,
                   "output_dir": str(output), "task_order": order}
            rows.append(row)
            if mode != "sequential":
                continue
            if method == "ace":
                for index, (slug, task) in enumerate(order[:50], 1):
                    create_session(output, mode, slug, task, index)
            else:
                batch.ace.save_json(output / "checkpoints/checkpoint_050.json", {
                    "cursor": 50, "config": {
                        "order": order, "mode": mode, "seed": seed, "past_count": 3,
                        "manifest_sha256": batch.ace.digest(MANIFEST), "method": method,
                    }, "states": {"global": {"beam": [{"playbook": "BFCL only"}]}}
                })
                batch.ace.save_json(output / "state.json", {"cursor": 80, "future_learning": "preserve"})
    path = root / "launch_manifest.json"
    batch.ace.save_json(path, {
        "agent": method, "method": method, "jobs": rows, "source_manifest": str(MANIFEST),
        "source_manifest_sha256": batch.ace.digest(MANIFEST), "model": "openai/test",
        "max_tokens": 8192, "terminalbench2_max_turns": 100,
    })
    return path


def test_batch_freezes_nine_bfcl_checkpoints_without_changing_live_training(tmp_path):
    manifests = [create_launch(tmp_path, method) for method in ("ace", "premise", "ps_st_pareto")]
    output = tmp_path / "evaluation"
    plan = batch.prepare_plan(manifests, output)
    assert len(plan["jobs"]) == 9
    assert len({job["port_block_start"] for job in plan["jobs"]}) == 9
    assert plan["planned_evaluations"] == 27
    assert sum(len(ids) for job in plan["jobs"] for ids in job["task_ids"].values()) == 1161
    splits = batch.read_json(MANIFEST)
    for job in plan["jobs"]:
        for slug in batch.BENCHMARKS:
            assert job["task_ids"][slug] == splits[f"{slug}_{job['seed']}"]["test"]
        batch.assert_checkpoint(job)
        if job["method"] != "ace":
            assert batch.read_json(Path(job["source_run"]) / "state.json")["cursor"] == 80
            assert batch.read_json(Path(job["evaluation_output"]) / "state.json")["cursor"] == 50


def test_complete_results_are_reused_and_live_partial_results_are_preserved(tmp_path):
    manifests = [create_launch(tmp_path, method) for method in ("ace", "premise", "ps_st_pareto")]
    output = tmp_path / "evaluation"
    plan = batch.prepare_plan(manifests, output)
    job = next(job for job in plan["jobs"] if job["method"] == "ps_st_pareto")
    ids = job["task_ids"]["bfcl"]
    summary = {"identity": batch.premise_identity(job, "bfcl"), "complete": True,
               "results": [{"task_id": task, "score": 1.0, "status": "completed"} for task in ids],
               "mean_official_score": 1.0, "execution_errors": 0}
    assert batch.publish_result(job, "bfcl", summary) == "published"
    assert batch.publish_result(job, "bfcl", summary) == "native_complete"
    batch.prepare_plan(manifests, output)
    assert batch.progress(job, "bfcl")["complete"]
    assert batch.progress(job, "bfcl")["score"] == 1.0
    appworld_ids = job["task_ids"]["appworld"]
    partial = {"identity": batch.premise_identity(job, "appworld"), "complete": False,
               "results": [None] * len(appworld_ids)}
    native = batch.result_path(job, "appworld", native=True)
    batch.ace.save_json(native, partial)
    raw = native.read_bytes()
    complete = {**partial, "complete": True,
                "results": [{"task_id": task, "score": 0.0, "status": "completed"} for task in appworld_ids]}
    assert batch.publish_result(job, "appworld", complete) == "native_evaluation_in_progress"
    assert native.read_bytes() == raw
    summary["identity"]["checkpoint_sha256"] = "wrong-checkpoint"
    with pytest.raises(ValueError, match="Wrong frozen"):
        batch.publish_result(job, "bfcl", summary)


def test_infrastructure_recovery_archives_and_reopens_only_invalid_results(tmp_path):
    manifests = [create_launch(tmp_path, method) for method in ("ace", "premise", "ps_st_pareto")]
    output = tmp_path / "evaluation"
    job = next(row for row in batch.prepare_plan(manifests, output)["jobs"] if row["method"] == "premise")
    slug = "terminalbench2"
    records = [{"task_id": task, "score": 0, "status": "completed"} for task in job["task_ids"][slug]]
    records[0].update(status="error", error="all predefined address pools have been fully subnetted")
    records[1].update(status="error", error="maximum context length", context_error=True)
    summary = {"identity": batch.premise_identity(job, slug), "results": records, "complete": True,
               "mean_official_score": 0, "execution_errors": 2}
    batch.ace.save_json(batch.result_path(job, slug), summary)
    batch.publish_result(job, slug, summary)
    state_before = (Path(job["source_run"]) / "state.json").read_bytes()
    checkpoint_before = Path(job["checkpoint_path"]).read_bytes()
    reopened = batch.prepare_infrastructure_retry(job, output)
    assert reopened == {slug: [records[0]["task_id"]]}
    repaired = batch.read_json(batch.result_path(job, slug))
    assert repaired["results"][0] is None
    assert repaired["results"][1:] == records[1:]
    assert repaired["complete"] is False
    assert not batch.result_path(job, slug, native=True).exists()
    assert len(list((output / "repairs").glob("*/*/*.json"))) == 2
    assert (Path(job["source_run"]) / "state.json").read_bytes() == state_before
    assert Path(job["checkpoint_path"]).read_bytes() == checkpoint_before


def test_infrastructure_recovery_does_not_reopen_context_or_official_zero():
    assert not batch.retryable_infrastructure_record({"status": "completed", "score": 0})
    assert not batch.retryable_infrastructure_record({"status": "error", "error": "APITimeoutError",
                                                     "context_error": True})
    assert not batch.retryable_infrastructure_record({"status": "error", "error": "unknown task failure"})
    assert batch.retryable_infrastructure_record({"status": "error", "error": "litellm.Timeout: request timed out"})
    assert batch.retryable_infrastructure_record({"status": "error", "error": (
        "RuntimeError: {'session_metadata': {'error': 'timed out', 'error_source': 'benchmark'}}"
    )})
    assert not batch.retryable_infrastructure_record({"status": "error", "error": (
        "RuntimeError: {'session_metadata': {'error': 'timed out', 'error_source': 'agent'}}"
    )})


def test_completed_result_cannot_hide_missing_tasks(tmp_path):
    job = {"name": "premise_sequential_s42_t3", "method": "premise", "seed": 42,
           "task_ids": {"bfcl": ["a", "b"]}, "checkpoint_sha256": "checkpoint", "manifest_sha256": "manifest"}
    summary = {"identity": batch.premise_identity(job, "bfcl"), "complete": True,
               "results": [{"task_id": "a", "score": 1.0, "status": "completed"}, None]}
    with pytest.raises(ValueError, match="incomplete records"):
        batch.validate_result(job, "bfcl", summary)
