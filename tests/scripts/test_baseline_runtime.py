# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from scripts.baselines import actor
from scripts.baselines.gepa_runtime import OrderedSampler, StreamLoader
from scripts.baselines.runtime import (
    ROOT,
    InfrastructureError,
    Runtime,
    actor_context_metadata,
    initial_instruction,
    read,
    write,
)


def config(tmp_path, method="gepa"):
    return {
        "method": method, "benchmark": "bfcl", "seed": 42, "workers": 2, "output": str(tmp_path),
        "model": "openai/Qwen/Qwen3.6-35B-A3B", "base_url": "http://127.0.0.1:18006/v1",
        "train": [{"task_id": f"train_{i}", "split": "train"} for i in range(7)],
        "validation": [{"task_id": f"val_{i}", "split": "val"} for i in range(2)],
        "test": [{"task_id": "test_0", "split": "test"}],
    }


def test_only_current_batch_is_visible_and_final_batch_has_no_padding(tmp_path):
    cfg = config(tmp_path)
    loader = StreamLoader(cfg["train"])
    assert len(loader) == 5
    assert loader.all_ids() == [f"train_{i}" for i in range(5)]
    with pytest.raises(KeyError):
        loader.fetch(["train_5"])
    ids = OrderedSampler().next_minibatch_ids(loader, SimpleNamespace(i=1))
    assert ids == ["train_5", "train_6"]
    assert len(loader) == 2
    with pytest.raises(KeyError):
        loader.fetch(["train_0"])


@pytest.mark.parametrize("method", ["gepa", "skillopt"])
@pytest.mark.parametrize("actor_context", ["legacy", "shared-empty-v1"])
@pytest.mark.parametrize("benchmark", ["bfcl", "terminalbench2"])
def test_official_loops_cover_stream_choose_validation_best_and_recover(tmp_path, method, actor_context, benchmark):
    python = ROOT / f".exgentic/baselines/venvs/{method}/bin/python"
    if not python.exists():
        pytest.skip(f"Pinned {method} environment not installed; run scripts/baselines/prepare.py install")
    cfg = config(tmp_path, method)
    cfg["benchmark"] = benchmark
    if benchmark == "terminalbench2":
        cfg["benchmark_kwargs"] = {"subset": "2.1", "actor": "terminus2", "max_interactions": 40}
    if actor_context != "legacy":
        cfg.update(actor_context=actor_context,
                   actor_context_metadata=actor_context_metadata(actor_context, [benchmark]))
    write(tmp_path / "config.json", cfg)
    command = [str(python), str(ROOT / "scripts/baselines/run_experiment.py"),
               "--worker-config", str(tmp_path / "config.json"), "--scripted"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert read(tmp_path / "summary.json")["score"] == 1.0
    assert read(tmp_path / "training_result.json")["unique_train_encounters"] == 7
    online = [read(p) for p in (tmp_path / "online").glob("*.json")]
    assert {row["task_id"] for row in online} == {f"train_{i}" for i in range(7)}
    assert sum(row["hard"] for row in online) == 2  # pre-update first batch, then improved second batch
    assert "scripted improved" in read(tmp_path / "frozen_instruction.json")["content"]
    if benchmark == "terminalbench2":
        requests = [read(path) for path in (tmp_path / "rollouts").glob("*/request.json")]
        assert {request["item"]["split"] for request in requests} == {"train", "val", "test"}
        assert all(request["benchmark_kwargs"]["subset"] == "2.1" for request in requests)
    attempts_before = (tmp_path / "attempts.jsonl").read_text()
    frozen_before = read(tmp_path / "frozen_instruction.json")
    # Simulate a crash just before the completion marker; rebuild full optimizer state
    # from the initial state and cached decisions without more environment attempts.
    (tmp_path / "summary.json").unlink()
    recovered = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert (tmp_path / "attempts.jsonl").read_text() == attempts_before
    assert read(tmp_path / "frozen_instruction.json") == frozen_before
    assert len(list((tmp_path / "upstream").glob("replay_*"))) == 2
    if method == "skillopt":
        history = read(tmp_path / "upstream/replay_0001/history.json")
        assert history[0]["action"] == "accept_new_best"
        assert history[1]["action"] == "reject"
    else:
        assert read(tmp_path / "training_result.json")["total_training_limit"] is None


def test_infrastructure_failure_does_not_become_a_score_or_partial_gate(tmp_path, monkeypatch):
    runtime = Runtime(config(tmp_path), scripted=True)
    original = runtime.task

    def task(item, learned, label):
        if item["task_id"] == "val_1":
            raise InfrastructureError("Backend missing")
        return original(item, learned, label)

    monkeypatch.setattr(runtime, "task", task)
    with pytest.raises(InfrastructureError):
        runtime.batch(runtime.config["validation"], initial_instruction("gepa"), "validation")
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("source,error,official,accepted", [
    ("agent", "LiteLLM completion missing assistant content.", {"score": 0.0, "success": False}, True),
    ("agent", "LiteLLM completion missing assistant content.", None, False),
    ("agent", "LiteLLM completion missing assistant content.", {"score": 1.0, "success": True}, False),
    ("agent", "APITimeoutError", {"score": 0.0, "success": False}, False),
    ("benchmark", "LiteLLM completion missing assistant content.", {"score": 0.0, "success": False}, False),
])
def test_protocol_failure_requires_recorded_official_bfcl_outcome(tmp_path, source, error, official, accepted):
    session = SimpleNamespace(status=SimpleNamespace(value="error"), score=0.0, success=False,
                              details={"session_metadata": {"error_source": source, "error": error}})
    if official is not None:
        write(tmp_path / "benchmark/bfcl_score.json", official)
    assert bool(actor.official_protocol_failure(session, tmp_path, "bfcl")) is accepted
    assert actor.official_protocol_failure(session, tmp_path, "appworld") is None


def test_frozen_actor_keeps_official_protocol_failure_and_partial_trajectory(tmp_path, monkeypatch):
    import exgentic.agents.premise.pipeline as pipeline
    import exgentic.interfaces.lib.api as api

    session = SimpleNamespace(status=SimpleNamespace(value="error"), score=0.0, success=False, session_id="session",
                              details={"session_metadata": {"error_source": "agent",
                                      "error": "LiteLLM completion missing assistant content."}})

    def execute(*, config):
        path = tmp_path / config.run_id / "sessions/session/benchmark/bfcl_score.json"
        write(path, {"score": 0.0, "success": False})
        return SimpleNamespace(session_results=[session])

    monkeypatch.setattr(api, "execute", execute)
    monkeypatch.setattr(pipeline, "trajectory_from_session", lambda *args: {"task": "t", "actions": ["partial"]})
    result = actor.run({"benchmark": "bfcl", "method": "gepa", "benchmark_kwargs": {},
                        "item": {"task_id": "t"}, "session_output": str(tmp_path),
                        "model": "openai/Qwen/Qwen3.6-35B-A3B", "system_prompt": "policy"})
    assert result["hard"] == result["soft"] == 0
    assert result["trace"]["actions"] == ["partial"]
    assert result["trace"]["agent_protocol_error"] == session.details["session_metadata"]["error"]
    assert read(result["trace"]["official_outcome_source"])["success"] is False


@pytest.mark.parametrize("score,success", [(0.125, False), (0.999, False), (1.0, True)])
def test_appworld_protocol_recovery_uses_official_tgc_and_preserves_partial_score(tmp_path, score, success):
    session = SimpleNamespace(status=SimpleNamespace(value="error"), score=0.0, success=False,
                              session_id="session", task_id="task",
                              details={"session_metadata": {"error_source": "agent",
                                      "error": "LiteLLM completion missing assistant content."}})
    path = tmp_path / "benchmark/results.json"
    official = {"score": score, "success": success, "session_id": "session", "task_id": "task"}
    write(path, official)
    proof = actor.official_protocol_failure(session, tmp_path, "appworld")
    restored = actor.restore_protocol_outcome(session, proof)
    assert restored.success is success and restored.score == score
    assert session.success is False and session.score == 0.0
    assert restored.details["session_metadata"]["official_outcome_source"] == str(path)
    for invalid in [{**official, "task_id": "different"}, {**official, "success": None},
                    {**official, "score": 1.001}]:
        write(path, invalid)
        assert actor.official_protocol_failure(session, tmp_path, "appworld") is None
    path.unlink()
    assert actor.official_protocol_failure(session, tmp_path, "appworld") is None


@pytest.mark.parametrize("benchmark,steps", [("bfcl", 30), ("appworld", 40), ("terminalbench2", 40)])
@pytest.mark.parametrize("reward,execution_success", [(0.0, True), (0.8, False), (1.0, True)])
@pytest.mark.parametrize("method", ["gepa", "skillopt"])
def test_frozen_actor_preserves_limits_and_uses_official_binary_success(
    tmp_path, monkeypatch, benchmark, steps, reward, execution_success, method,
):
    import exgentic.agents.premise.pipeline as pipeline
    import exgentic.interfaces.lib.api as api

    captured = []
    session = SimpleNamespace(status=SimpleNamespace(value="unsuccessful"), score=reward,
                              success=execution_success, details={})

    def execute(*, config):
        captured.append(config)
        return SimpleNamespace(session_results=[session])

    monkeypatch.setattr(api, "execute", execute)
    monkeypatch.setattr(pipeline, "trajectory_from_session", lambda *args: {"task": "t", "actions": []})
    benchmark_kwargs = {"subset": "2.1"} if benchmark == "terminalbench2" else {}
    request = {"benchmark": benchmark, "method": method, "benchmark_kwargs": benchmark_kwargs,
               "item": {"task_id": "t"}, "session_output": str(tmp_path),
               "model": "openai/Qwen/Qwen3.6-35B-A3B", "system_prompt": "fixed base + learned skill"}
    result = actor.run(request)
    expected = int(reward == 1.0) if benchmark == "terminalbench2" else int(execution_success)
    assert result["hard"] == expected and result["soft"] == reward
    cfg = captured[0]
    assert cfg.max_steps == steps
    assert cfg.agent_kwargs["model_settings"].max_tokens == 2048
    assert cfg.agent_kwargs["temperature_override"] == 0.7
    assert cfg.agent_kwargs["playbook"] == cfg.agent_kwargs["system_prompt_override"] == request["system_prompt"]
    assert cfg.agent_kwargs["enable_thinking"] is False
    if benchmark == "terminalbench2":
        assert cfg.benchmark_kwargs["subset"] == "2.1"
        assert cfg.benchmark_kwargs["actor"] == "terminus2"
        assert cfg.benchmark_kwargs["max_interactions"] == 40
