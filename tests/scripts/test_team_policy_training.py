# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Regression checks for stopping learning at CP100 and freezing cross-method evaluation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exgentic.agents.premise.pipeline import PremisePipeline, write_json

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("team_policy_training", ROOT / "scripts/premise/run_team_policy.py")
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def test_stop_after_keeps_full_order_and_stops_without_terminal_training(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"benchmarks": {"bfcl": {"train": ["a"]}, "appworld": {"train": ["b"]},
                                       "terminalbench2": {"train": ["c"]}}})
    pipeline = PremisePipeline(output=tmp_path / "run", manifest=manifest,
                               benchmarks=["bfcl", "appworld", "terminalbench2"], mode="sequential",
                               seed=42, count=None, past_count=3, model="openai/test", base_url="http://unused")
    visited = []

    def step():
        visited.append(pipeline.order[pipeline.cursor][0])
        pipeline.cursor += 1
        pipeline._save()
        return {"status": "skipped", "iteration": pipeline.cursor,
                "benchmark": visited[-1], "task_id": pipeline.order[pipeline.cursor - 1][1]}

    pipeline.step = step
    pipeline.run(stop_after=2)
    assert visited == ["bfcl", "appworld"]
    assert len(pipeline.order) == 3
    complete = json.loads((pipeline.output / "training_complete.json").read_text())
    assert complete == {"completed_task_count": 2, "expected_task_count": 2}
    with pytest.raises(ValueError):
        pipeline.run(stop_after=1)


def test_cp100_requires_completed_ace_learning_and_freezes_latest_valid_playbook(tmp_path, monkeypatch):
    job = {"method": "ace", "seed": 42, "output_dir": str(tmp_path), "task_order": [["bfcl", "a"]] * 100}
    prefix = [{"checkpoint": None}] * 99
    monkeypatch.setattr(controller, "completed_prefix", lambda *_: prefix)
    assert controller.checkpoint_at_100(job) is None
    source = tmp_path / "source.json"
    write_json(source, {"store_id": "ace_sequential_global", "session_count": 100, "playbook": "frozen"})
    prefix.append({"checkpoint": json.loads(source.read_text()), "checkpoint_path": str(source)})
    path = controller.checkpoint_at_100(job)
    frozen = path.read_bytes()
    assert json.loads(frozen)["training_task_ordinal"] == 100
    assert json.loads(frozen)["playbook"] == "frozen"
    source.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Frozen ACE checkpoint changed"):
        controller.checkpoint_at_100(job)
    assert path.read_bytes() == frozen


@pytest.mark.parametrize("method", ["ace", "premise"])
@pytest.mark.parametrize("benchmark", ["appworld", "bfcl", "terminalbench2"])
def test_frozen_eval_policy_scope_and_native_terminal_sampling(tmp_path, monkeypatch, method, benchmark):
    import exgentic.interfaces.lib.api as api

    checkpoint = tmp_path / "checkpoint.json"
    playbook = {"playbook": "frozen", "next_global_id": 1}
    snapshot = {"playbook": "frozen", "states": {"global": {"beam": [playbook]}}}
    write_json(checkpoint, snapshot)
    policy = tmp_path / "policy.txt"
    policy.write_text("Apps: {allowed}. finish {}")
    captured = []

    def execute(*, config):
        captured.append(config)
        session = SimpleNamespace(session_id="one", status=SimpleNamespace(value="completed"), score=1, details={})
        return SimpleNamespace(session_results=[session])

    monkeypatch.setattr(api, "execute", execute)
    spec = {"checkpoint": str(checkpoint), "checkpoint_sha256": controller.digest(checkpoint),
            "policy": str(policy), "policy_sha256": controller.digest(policy), "benchmark": benchmark,
            "benchmark_configs": {benchmark: {}}, "max_steps": {benchmark: 50}, "model": "openai/test",
            "method": method, "seed": 42, "id": f"{method}_42", "index": 0, "task_id": "task",
            "evaluation_dir": str(tmp_path / "eval")}
    controller.evaluate_task(spec)
    config = captured[0]
    assert ("policy_override" in config.benchmark_kwargs) == (benchmark == "appworld")
    assert config.agent_kwargs["model_settings"].max_tokens == (None if benchmark == "terminalbench2" and method == "ace" else 2048)
    assert ("temperature" in config.agent_kwargs["model_settings"].model_fields_set) == (benchmark != "terminalbench2")
    if method == "ace":
        assert config.agent_kwargs["evaluation_mode"] is True
        assert config.agent_kwargs["training_time"] is False
        assert config.agent_kwargs["initial_playbook"] == "frozen"


def test_all_methods_share_one_fifty_worker_queue_at_checkpoint_100(tmp_path, monkeypatch):
    from concurrent.futures import Future

    policy = tmp_path / "policy.txt"
    policy.write_text("team policy")
    jobs = []
    for method in ("premise", "ace"):
        for seed in (42, 43, 44):
            directory = tmp_path / method / str(seed)
            write_json(directory / "state.json", {"cursor": 100})
            jobs.append({"id": f"{method}_{seed}", "method": method, "seed": seed,
                         "output_dir": str(directory), "port_block_start": 42000,
                         "command": ["fake-training"], "log": str(tmp_path / f"{method}_{seed}.log"),
                         "task_order": [["bfcl", "task"]] * 100,
                         "test_tasks": {b: ["a", "b"] for b in controller.BENCHMARKS}})
    manifest = {"policy": str(policy), "policy_sha256": controller.digest(policy), "jobs": jobs,
                "base_url": "http://fake", "evaluation_workers": 50, "evaluation_tasks": 36}
    write_json(tmp_path / "launch_manifest.json", manifest)
    monkeypatch.setattr(controller.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(controller, "completed_prefix", lambda *_: [None] * 100)
    monkeypatch.setattr(controller, "checkpoint_at_100", lambda *_: policy)
    monkeypatch.setattr(controller.signal, "signal", lambda *_: None)
    submitted = []
    pools = []

    class Pool:
        def __init__(self, *, max_workers):
            pools.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def submit(self, function, spec):
            submitted.append(spec)
            future = Future()
            future.set_result({"task_id": spec["task_id"], "status": "completed", "score": 1, "tgc": 1})
            return future

    monkeypatch.setattr(controller, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(controller.subprocess, "Popen",
                        lambda *_args, **_kwargs: SimpleNamespace(pid=100, poll=lambda: 0))
    assert controller.controller(tmp_path) == 0
    assert pools == [50]
    assert len(submitted) == 36
    assert {(s["method"], s["seed"], s["benchmark"]) for s in submitted} == {
        (m, seed, b) for m in ("ace", "premise") for seed in (42, 43, 44) for b in controller.BENCHMARKS
    }
    assert all(s["checkpoint_sha256"] == controller.digest(policy) for s in submitted)
    assert json.loads((tmp_path / "status.json").read_text())["phase"] == "complete"
