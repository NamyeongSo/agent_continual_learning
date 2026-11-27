# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import hashlib
import json
import threading
import time
from collections import Counter
from types import SimpleNamespace

import pytest

from exgentic.agents.premise.evaluation_queue import CheckpointEvaluationQueue
from exgentic.agents.premise.pipeline import write_json


def make_pipeline(tmp_path, seed=43, task_count=2):
    manifest = tmp_path / "manifest.json"
    slugs = ["appworld", "terminalbench2", "bfcl"]
    write_json(manifest, {f"{slug}_{seed}": {
        "train": ["train"], "val": ["val"], "test": [f"test{i}" for i in range(task_count)],
    } for slug in slugs})
    output = tmp_path / "run"
    config = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    snapshot = {"config": config, "cursor": 100, "states": {"global": {"beam": [{"playbook": "cp100"}]}}}
    write_json(output / "state.json", snapshot)
    write_json(output / "checkpoints/checkpoint_100.json", snapshot)
    write_json(output / "checkpoints/checkpoint_130.json", {
        **snapshot, "cursor": 130, "states": {"global": {"beam": [{"playbook": "cp130"}]}},
    })
    return SimpleNamespace(output=output, manifest=manifest, benchmarks=slugs, seed=seed, mode="sequential")


def drain(queue, timeout=5):
    deadline = time.monotonic() + timeout
    while queue.futures and time.monotonic() < deadline:
        queue.poll()
        time.sleep(0.005)
    assert not queue.futures


def test_slow_task_does_not_block_other_benchmarks_checkpoints_or_seeds(tmp_path):
    pipelines = [make_pipeline(tmp_path / str(seed), seed) for seed in (43, 44)]
    release = threading.Event()
    reached_later_seed = threading.Event()
    calls = []
    lock = threading.Lock()
    before = [(pipeline.output / "state.json").read_bytes() for pipeline in pipelines]

    def run(seed, **kwargs):
        label, slug = kwargs["label"], kwargs["benchmark"]
        point = 100 if "cp100" in label else 130
        assert kwargs["playbook"]["playbook"] == f"cp{point}"
        kwargs["playbook"]["playbook"] = "mutated locally"
        if seed == 43 and slug == "terminalbench2" and label.endswith("0000") and point == 100:
            assert release.wait(5)
        with lock:
            calls.append((seed, point, slug, kwargs["task_id"]))
        if seed == 44 and point == 130:
            reached_later_seed.set()
        return {"score": 0.5}

    for pipeline in pipelines:
        pipeline._run_agent = lambda seed=pipeline.seed, **kwargs: run(seed, **kwargs)
    queue = CheckpointEvaluationQueue(2)
    try:
        groups = [queue.enqueue(pipeline, point) for pipeline in pipelines for point in (100, 130)]
        assert reached_later_seed.wait(5)
        assert not release.is_set()
        release.set()
        drain(queue)
        assert len(calls) == 24
        assert len(set(calls)) == 24
        assert all(queue.complete(keys) for keys in groups)
        assert before == [(pipeline.output / "state.json").read_bytes() for pipeline in pipelines]
    finally:
        release.set()
        queue.close()
    resumed = CheckpointEvaluationQueue(2)
    try:
        for pipeline in pipelines:
            for point in (100, 130):
                assert resumed.complete(resumed.enqueue(pipeline, point))
        assert resumed.submitted == 0
    finally:
        resumed.close()


def test_fifty_workers_share_one_queue_with_no_benchmark_caps(tmp_path):
    pipeline = make_pipeline(tmp_path, task_count=30)
    release = threading.Event()
    fifty_running = threading.Event()
    lock = threading.Lock()
    counts = Counter()
    active = peak = 0

    def run(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            counts[kwargs["benchmark"]] += 1
            if active == 50:
                fifty_running.set()
        assert release.wait(5)
        with lock:
            active -= 1
        return {"score": 1}

    pipeline._run_agent = run
    queue = CheckpointEvaluationQueue(50)
    try:
        queue.enqueue(pipeline, 100)
        assert fifty_running.wait(5)
        assert peak == 50
        assert counts["terminalbench2"] > 1
        assert counts["appworld"] > 4
        assert counts["bfcl"] > 4
        assert queue.status()["queued"] == 40
        release.set()
        drain(queue)
        assert queue.finished == 90
    finally:
        release.set()
        queue.close()


def test_evaluation_error_is_persisted_and_modified_checkpoint_is_rejected(tmp_path):
    pipeline = make_pipeline(tmp_path, task_count=1)

    def fail(**_kwargs):
        raise TimeoutError("upstream failed")

    pipeline._run_agent = fail
    queue = CheckpointEvaluationQueue(2)
    try:
        queue.enqueue(pipeline, 100)
        drain(queue)
    finally:
        queue.close()

    summary = json.loads((pipeline.output / "test_evaluation/cp100_appworld.json").read_text())
    assert summary["complete"]
    assert summary["execution_errors"] == 1
    assert summary["results"][0]["status"] == "error"
    path = pipeline.output / "checkpoints/checkpoint_100.json"
    snapshot = json.loads(path.read_text())
    snapshot["states"]["global"]["beam"][0]["playbook"] = "changed"
    write_json(path, snapshot)
    queue = CheckpointEvaluationQueue(2)
    try:
        with pytest.raises(ValueError, match="Existing evaluation differs"):
            queue.enqueue(pipeline, 100)
    finally:
        queue.close()


def test_temperature_override_applies_only_to_selected_checkpoint_task_and_benchmark(tmp_path):
    pipeline = make_pipeline(tmp_path)
    pipeline.evaluation_temperature_overrides = {(100, "appworld", "test0"): 0.1}
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {"score": 1.0}

    pipeline._run_agent = run
    queue = CheckpointEvaluationQueue(50)
    try:
        queue.enqueue(pipeline, 100)
        queue.enqueue(pipeline, 130)
        drain(queue)
    finally:
        queue.close()
    selected = [call for call in calls if "temperature" in call]
    assert len(selected) == 1
    assert selected[0]["temperature"] == 0.1
    assert selected[0]["benchmark"] == "appworld" and selected[0]["task_id"] == "test0"
    assert "cp100" in selected[0]["label"]
    summary = json.loads((pipeline.output / "test_evaluation/cp100_appworld.json").read_text())
    assert summary["results"][0]["temperature"] == 0.1
    assert "temperature" not in summary["results"][1]
