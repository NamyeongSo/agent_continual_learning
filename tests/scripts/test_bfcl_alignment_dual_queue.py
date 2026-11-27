# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from collections import deque

import pytest

from scripts.premise.run_bfcl_alignment_dual_queue import (
    original_running,
    pop_job,
    publish,
    save_runtime_plan,
    validate_record,
)
from scripts.utils.evaluate_isolated_checkpoint_queue import read_json


def test_tail_pop_reserves_running_front_prefix_and_endpoints_never_claim_the_same_job():
    queued = deque((index, {"key": f"job-{index}"}) for index in range(10))
    records = {"job-0": {"status": "completed"}}
    tail = [pop_job(queued, records, True, reserve_prefix=5)[0] for _ in range(5)]
    assert tail == [9, 8, 7, 6, 5]
    assert pop_job(queued, records, True, reserve_prefix=5) is None
    front = [pop_job(queued, records, False)[0] for _ in range(4)]
    assert front == [1, 2, 3, 4]
    assert len(set(tail + front)) == 9
    assert pop_job(queued, records, False) is None


def test_completed_results_discovered_after_dispatch_freeze_are_skipped_from_both_ends():
    queued = deque((index, {"key": f"job-{index}"}) for index in range(4))
    records = {"job-0": {}, "job-3": {}}
    assert pop_job(queued, records, True)[0] == 2
    assert pop_job(queued, records, False)[0] == 1
    assert pop_job(queued, records, True) is None


def test_runtime_server_reallocation_preserves_original_manifest_and_rejects_changed_queue(tmp_path):
    original = {"endpoints": [{"base_url": "http://localhost:18008/v1", "workers": 500}],
                "queue_sha256": "frozen-queue", "settings": {"max_steps": 20}, "model": "gemma",
                "single_coordinator_ownership": True, "source_inputs_preserved": True}
    save_runtime_plan(tmp_path, original)
    original_bytes = (tmp_path / "dual_endpoint_plan.json").read_bytes()
    updated = {**original, "endpoints": [{"base_url": "http://localhost:18009/v1", "workers": 300}]}
    save_runtime_plan(tmp_path, updated)
    assert (tmp_path / "dual_endpoint_plan.json").read_bytes() == original_bytes
    assert read_json(tmp_path / "dual_endpoint_runtime_plan.json") == updated
    assert len(list((tmp_path / "runtime_allocations").glob("*.json"))) == 2
    with pytest.raises(ValueError, match="queue_sha256"):
        save_runtime_plan(tmp_path, {**updated, "queue_sha256": "wrong-queue"})
    assert read_json(tmp_path / "dual_endpoint_runtime_plan.json") == updated


def test_publishing_records_endpoint_provenance_and_never_overwrites_an_existing_result(tmp_path):
    job = {"key": "cp/candidate/train/000", "model": "gemma", "context_sha256": "frozen", "max_steps": 20}
    endpoint = {"base_url": "http://localhost:18016/v1", "upstream": "http://localhost:8006/v1",
                "output": "contributor", "direction": "LIFO"}
    row = {**job, "status": "completed", "score": 1}
    publish(tmp_path, job, row, endpoint, 99)
    path = tmp_path / "results/cp/candidate/train/000.json"
    original = path.read_bytes()
    assert read_json(path)["runtime_upstream"] == "http://localhost:8006/v1"
    with pytest.raises(RuntimeError, match="Duplicate"):
        publish(tmp_path, job, {**job, "status": "error", "score": 0}, endpoint, 99)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="frozen job"):
        validate_record({**job, "status": "completed", "model": "wrong-model"}, job)


def test_front_handover_reads_the_preserved_contributor_worker_directory(tmp_path, monkeypatch):
    from scripts.premise import run_bfcl_alignment_dual_queue as coordinator
    from scripts.utils.evaluate_isolated_checkpoint_queue import write_json

    output = tmp_path / "contributor"
    child = {"pid": 123}
    write_json(output / "active_workers/123.json", {"phase": "running", "key": "still-evaluating"})
    write_json(tmp_path / "active_workers/123.json", {"phase": "idle"})
    monkeypatch.setattr(coordinator, "same_process", lambda _child: True)
    assert original_running(tmp_path, {"children": [child], "worker_output": str(output)}) == ["still-evaluating"]


def test_persisted_endpoint_stop_never_creates_a_tail_pool_on_restart(tmp_path, monkeypatch):
    from scripts.premise import run_bfcl_alignment_dual_queue as coordinator
    from scripts.utils.evaluate_isolated_checkpoint_queue import write_json

    job = {"key": "cp/candidate/train/000", "model": "openai/google/gemma-4-26B-A4B-it", "max_steps": 20}
    write_json(tmp_path / "evaluation_queue.json", {"jobs": [job]})
    write_json(tmp_path / "results/cp/candidate/train/000.json", {**job, "status": "completed", "score": 1})
    write_json(tmp_path / "launch_manifest.json", {
        "workers": 150, "base_url": "http://localhost:18008/v1", "max_tokens": 8192,
        "temperature": 0, "request_timeout": 600, "transport_additional_retries": 5, "max_steps": 20,
    })
    write_json(tmp_path / "dual_endpoint_handover.json", {
        "stage": "drained", "original_status": {"started_at": "earlier"}, "original_pipeline": {},
        "active_at_handover": [],
    })
    write_json(tmp_path / "endpoint_8006_stop.json", {"reason": "GPU released for teammate"})

    def no_pool(*_args):
        pytest.fail("A completed queue with a stopped tail must not create a pool")

    monkeypatch.setattr(coordinator, "make_pool", no_pool)
    monkeypatch.setattr(coordinator.alignment, "analyze", lambda *_args: {"execution_errors": 0})
    monkeypatch.setattr(coordinator.subprocess, "run", lambda *_args, **_kwargs: None)
    coordinator.run(tmp_path, "http://localhost:18016/v1", "http://localhost:8006/v1", 500)
    status = read_json(tmp_path / "status.json")
    assert status["workers"] == 150
    tail = next(endpoint for endpoint in status["endpoints"] if endpoint["name"] == "tail_8006")
    assert tail["enabled"] is False and tail["workers"] == tail["active"] == 0
