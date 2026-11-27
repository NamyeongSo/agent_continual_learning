# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.utils import checkpoint_eval_worker as worker
from scripts.utils.checkpoint_queue import Queue
from scripts.utils.checkpoint_sources import digest
from scripts.utils.prepare_fixed_checkpoint_matrix import initial_fingerprint
from scripts.utils.serve_checkpoint_query_router import QuerySlots


def spec(name="s42", method="gepa"):
    return {"id": name, "method": method, "benchmark": "bfcl", "seed": 42, "variant": "cohort",
            "run_variant": "val", "model": "Qwen", "temperature": 0.7, "train_total": 20,
            "splits": {"train": ["t1", "t2"], "val": ["v1"], "test": ["z1"]}}


def initial(stream):
    return {"point": 0, "content": "initial", "selection": "initial",
            "initial_group": initial_fingerprint(stream, "initial")}


def test_cp0_reuses_seeds_and_variations_but_respects_baseline_policy_and_membership(tmp_path):
    queue = Queue(tmp_path, interval=10, share_initial=True)
    first = spec()
    queue.register(first, initial(first))
    second = {**first, "id": "s43_train", "seed": 43, "run_variant": "train-as-val",
              "splits": {**first["splits"], "train": ["t2", "t1"]}}
    queue.register(second, initial(second))
    assert queue.dashboard()["skipped"] == 1
    assert len(queue.pending(100)) == 4
    for change in [{"method": "skillopt"}, {"appworld_policy_sha256": "different"},
                   {"splits": {**first["splits"], "val": ["other-val"]}}]:
        other = {**first, **change}
        assert initial_fingerprint(first, "initial") != initial_fingerprint(other, "initial")
    queue.register(first, {"point": 10, "content": "initial", "selection": "same"})
    queue.register(first, {"point": 20, "content": "updated", "selection": "updated"})
    assert queue.dashboard()["task_counts"]["pending"] == 8
    with pytest.raises(ValueError):
        queue.register(first, {"point": 5, "content": "not-requested"})
    with pytest.raises(ValueError):
        Queue(tmp_path / "legacy").register(first, initial(first))


def test_cp0_and_existing_stream_never_mix_actor_context_versions(tmp_path):
    from exgentic.actor_context import actor_context_metadata

    legacy = spec()
    shared = {**legacy, "actor_context": "shared-empty-v1",
              "actor_context_metadata": actor_context_metadata("shared-empty-v1", ["bfcl"])}
    assert initial_fingerprint(legacy, "initial") != initial_fingerprint(shared, "initial")
    queue = Queue(tmp_path, interval=10, share_initial=True)
    queue.register(legacy, initial(legacy))
    with pytest.raises(ValueError, match="actor_context"):
        queue.register(shared, initial(shared))
    queue.register({**shared, "id": "new-cohort"}, initial(shared))
    assert queue.dashboard()["skipped"] == 0


def test_shared_queue_result_rejects_a_matching_policy_with_different_context(tmp_path):
    from exgentic.actor_context import actor_context_metadata
    from scripts.utils.checkpoint_eval_proxy import wait_for_result

    profile = {"actor_context": "shared-empty-v1",
               "actor_context_metadata": actor_context_metadata("shared-empty-v1", ["bfcl"])}
    shared = {**spec(), **profile}
    queue = Queue(tmp_path, interval=10, share_initial=True)
    queue.register(shared, initial(shared))
    pending = next(row for row in queue.pending(100) if row["split"] == "test")
    queue.finish(queue.claim(pending), {"status": "completed", "score": 1})
    marker = {"output": str(tmp_path), "stream": shared["id"], "point": 0}
    assert wait_for_result(marker, "z1", digest("initial"), profile)["score"] == 1
    with pytest.raises(ValueError, match="actor context"):
        wait_for_result(marker, "z1", digest("initial"), {**profile, "actor_context_metadata": {}})


def test_each_gpu_has_a_real_query_limit_and_releases_slots_after_failure():
    slots = QuerySlots({"0": "gpu0", "1": "gpu1"}, limit=2)

    def run(index):
        try:
            with slots.claim() as (gpu, endpoint):
                assert endpoint == f"gpu{gpu}"
                time.sleep(0.01)
                if index == 0:
                    raise RuntimeError("transport failed")
                return gpu
        except RuntimeError:
            return "error"

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(run, range(32)))
    assert set(results) == {"0", "1", "error"}
    assert slots.snapshot()["active"] == {"0": 0, "1": 0}
    assert slots.snapshot()["maximum"] == {"0": 2, "1": 2}


def test_legacy_actor_receives_router_endpoint_and_explicit_code_generation(tmp_path, monkeypatch):
    from scripts.utils import run_qwen_isolated_when_free as legacy

    stream = {**spec(method="ace"), "benchmark": "appworld",
              "benchmark_kwargs": {"interaction_mode": "code_generation"},
              "appworld_policy_override": "original", "appworld_policy_sha256": digest("original")}
    frozen = tmp_path / "cp.json"
    frozen.write_text(json.dumps({"stream": stream, "content": "initial"}))
    (tmp_path / "workers").mkdir()
    import os

    (tmp_path / "workers" / f"{os.getpid()}.json").write_text('{}')
    monkeypatch.setattr(worker, "_OUTPUT", tmp_path)
    monkeypatch.setattr(worker, "_BASE_URL", "http://query-router/v1")
    calls = []

    def fake(job):
        calls.append(job)
        return {"status": "completed", "score": 0, "tgc": 0}

    monkeypatch.setattr(legacy, "evaluate_one", fake)
    result = worker.evaluate({"id": 1, "attempt": 1, "checkpoint_file": str(frozen),
                              "content_sha256": digest("initial"), "task_id": "test", "split": "test"})
    assert result["status"] == "completed"
    assert calls[0]["base_url"] == "http://query-router/v1"
    assert calls[0]["benchmark_kwargs"]["interaction_mode"] == "code_generation"
    assert calls[0]["appworld_policy_sha256"] == digest("original")


@pytest.mark.parametrize("actor_context", ["legacy", "shared-empty-v1"])
def test_optimizer_actor_uses_openai_provider_for_reported_model_name(tmp_path, monkeypatch, actor_context):
    from exgentic.actor_context import actor_context_metadata, shared_actor_prompt
    from scripts.baselines import actor

    stream = {**spec(), "snapshot": str(tmp_path), "benchmark_kwargs": {}}
    if actor_context != "legacy":
        stream.update(actor_context=actor_context,
                      actor_context_metadata=actor_context_metadata(actor_context, ["bfcl"]))
    base = tmp_path / "scripts/baselines/prompts/gepa/base"
    base.mkdir(parents=True)
    (base / "bfcl.txt").write_text("Frozen wrapper")
    frozen = tmp_path / "cp.json"
    frozen.write_text(json.dumps({"stream": stream, "content": "initial"}))
    (tmp_path / "workers").mkdir()
    import os

    (tmp_path / "workers" / f"{os.getpid()}.json").write_text('{}')
    monkeypatch.setattr(worker, "_OUTPUT", tmp_path)
    calls = []

    def fake(request):
        calls.append(request)
        return {"hard": 0, "soft": 0, "run_id": "test"}

    monkeypatch.setattr(actor, "run", fake)
    result = worker.evaluate({"id": 1, "attempt": 1, "checkpoint_file": str(frozen),
                              "content_sha256": digest("initial"), "task_id": "test", "split": "test"})
    assert result["status"] == "completed"
    assert calls[0]["model"] == "openai/Qwen"
    if actor_context != "legacy":
        assert calls[0]["actor_context"] == actor_context
        assert calls[0]["learned_policy"] == "initial"
        assert calls[0]["system_prompt"] == shared_actor_prompt("bfcl", "initial")
