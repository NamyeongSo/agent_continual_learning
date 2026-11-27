# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from exgentic.actor_context import actor_context_metadata, shared_actor_prompt
from scripts.utils import run_shared_reflection_checkpoints as runner
from scripts.utils.checkpoint_queue import Queue
from scripts.utils.checkpoint_sources import digest
from scripts.utils.run_shared_reflection_checkpoints import reuse_rollouts
from scripts.utils.serve_checkpoint_query_router import QuerySlots, is_health_probe


def spec(tmp_path):
    return {"id": "stream", "method": "gepa", "benchmark": "bfcl", "seed": 42,
            "variant": "cohort", "model": "openai/Qwen", "temperature": 0.0, "train_total": 30,
            "train_dir": str(tmp_path / "source"), "playbook_edit_mode": "aud",
            "actor_context": "shared-empty-v1", "inference_profile_sha256": "pinned",
            "actor_context_metadata": actor_context_metadata("shared-empty-v1", ["bfcl"]),
            "splits": {"train": ["a"], "val": ["v"], "test": ["z"]}}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_nonadjacent_identical_checkpoint_reuses_same_policy_without_mixing_inference_profiles(tmp_path):
    queue = Queue(tmp_path / "queue", interval=10)
    stream = spec(tmp_path)
    for point, content in [(10, "a"), (20, "b"), (30, "a")]:
        queue.register(stream, {"point": point, "content": content, "selection": "fixed"})
    assert queue.dashboard()["checkpoints"] == 3 and queue.dashboard()["skipped"] == 1
    assert len(queue.pending(100)) == 6
    with pytest.raises(ValueError, match="inference_profile"):
        queue.register({**stream, "inference_profile_sha256": "different"}, {
            "point": 30, "content": "a", "selection": "fixed"})


@pytest.mark.parametrize("field,value", [(None, None), ("actor_temperature", .7),
                                       ("reasoning_effort", "high"), ("system_prompt", "other"),
                                       ("actor_context", "legacy")])
def test_training_cache_reuse_requires_frozen_prompt_and_actor_settings(tmp_path, field, value):
    queue = Queue(tmp_path / "queue", interval=10)
    stream = spec(tmp_path)
    queue.register(stream, {"point": 10, "content": "policy", "selection": "fixed"})
    prompt = shared_actor_prompt("bfcl", "policy")
    request = {**stream, "actor_temperature": 0.0, "reasoning_effort": "medium", "benchmark_kwargs": {},
               "system_prompt": prompt, "learned_policy": "policy", "prompt_sha256": digest("policy"),
               "item": {"split": "val", "task_id": "v"}}
    if field:
        request[field] = value
    folder = tmp_path / "source/rollouts/one"
    write(tmp_path / "source/config.json", {"benchmark_kwargs": {}})
    write(folder / "request.json", request)
    write(folder / "result.json", {"id": "v", "hard": 0, "soft": 0,
                                  "prompt_sha256": digest(prompt), "run_id": "official"})
    reuse_rollouts(queue, stream)
    counts = queue.dashboard()["task_counts"]
    assert counts.get("completed", 0) == (0 if field else 1)
    if not field:
        result = json.loads(queue.db.execute("SELECT result FROM tasks WHERE status='completed'").fetchone()[0])
        assert result["score"] == 0 and result["reused_from"].endswith("result.json")


def test_single_queue_fills_asymmetric_endpoint_slots_and_releases_them_on_error():
    slots = QuerySlots({"8007": "gpu-a", "8006": "gpu-b"}, {"8007": 3, "8006": 2})
    filled = threading.Barrier(6)
    release = threading.Event()

    def call(index):
        try:
            with slots.claim() as (port, _):
                if index < 5:
                    filled.wait(timeout=5)
                    release.wait(timeout=5)
                if index == 0:
                    raise RuntimeError("failed response")
                return port
        except RuntimeError:
            return "error"

    with ThreadPoolExecutor(max_workers=5) as pool:
        first = [pool.submit(call, i) for i in range(5)]
        filled.wait(timeout=5)
        snapshot = slots.snapshot()
        assert snapshot["active"] == {"8007": 3, "8006": 2}
        assert snapshot["queue"] == "single_priority"
        release.set()
        assert "error" in [future.result() for future in first]
        list(pool.map(call, range(5, 30)))
    assert slots.snapshot()["active"] == {"8007": 0, "8006": 0}
    assert slots.snapshot()["maximum"] == {"8007": 3, "8006": 2}


def test_startup_probe_passes_queued_actors_in_one_queue_without_exceeding_limit():
    slots = QuerySlots({"8006": "gpu1"}, {"8006": 1})
    entered = []

    def call(name, priority):
        with slots.claim(priority=priority):
            entered.append(name)

    with ThreadPoolExecutor(max_workers=3) as pool:
        with slots.claim():
            first = pool.submit(call, "actor1", 1)
            with slots.condition:
                assert slots.condition.wait_for(lambda: len(slots.waiting) == 1, timeout=5)
            health = pool.submit(call, "health", 0)
            last = pool.submit(call, "actor2", 1)
            with slots.condition:
                assert slots.condition.wait_for(lambda: len(slots.waiting) == 3, timeout=5)
        for future in (first, health, last):
            future.result(timeout=5)
    assert entered == ["health", "actor1", "actor2"]
    assert slots.snapshot()["maximum"] == {"8006": 1}
    probe = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    assert is_health_probe(probe)
    assert not is_health_probe({**probe, "exgentic_reasoning_budget": 16384})
    assert not is_health_probe({**probe, "max_tokens": 2048})


def test_worker_inherits_training_docker_path_and_preserves_pinned_inference(tmp_path, monkeypatch):
    binary = tmp_path / ".venv/bin/docker"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setenv("PATH", "/missing")
    monkeypatch.setenv("EXGENTIC_INFERENCE_PROFILE_SHA256", "pinned")
    runner.initialize(str(tmp_path / "output"))
    import os
    import shutil

    assert shutil.which("docker") == str(binary)
    assert os.environ["EXGENTIC_INFERENCE_PROFILE_SHA256"] == "pinned"
