# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script():
    path = Path(__file__).resolve().parents[2] / "scripts/premise/restart_sequential.py"
    spec = importlib.util.spec_from_file_location("premise_restart_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fork_copies_checkpoint_rng_beam_and_memory_without_touching_source(tmp_path):
    module = load_script()
    source, destination = tmp_path / "original", tmp_path / "restart"
    (source / "checkpoints").mkdir(parents=True)
    (source / "iterations").mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    config = {
        "seed": 43, "mode": "sequential", "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "order": [[slug, str(index)] for slug, count in (("bfcl", 50), ("appworld", 50), ("terminalbench2", 30))
                  for index in range(count)],
    }
    snapshot = {
        "cursor": 50, "config": config, "rng_state": "untouched rng",
        "states": {"global": {"beam": [{"playbook": "learned"}], "memory": [
            {"benchmark": "bfcl", "trace_artifact": "iterations/iter_0001.json", "parent_index": 0},
        ]}},
    }
    checkpoint = source / "checkpoints/checkpoint_050.json"
    checkpoint.write_text(json.dumps(snapshot))
    (source / "state.json").write_text('{"cursor": 130}')
    for index in range(1, 51):
        (source / "iterations" / f"iter_{index:04d}.json").write_text(json.dumps({"iteration": index}))
    before = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    module.fork_bfcl_checkpoint(source, destination, 43)
    assert (destination / "state.json").read_bytes() == checkpoint.read_bytes()
    assert (destination / "checkpoints/checkpoint_050.json").read_bytes() == checkpoint.read_bytes()
    assert (destination / "iterations/iter_0001.json").stat().st_ino != (
        source / "iterations/iter_0001.json"
    ).stat().st_ino
    (destination / "state.json").write_text('{"cursor": 51}')
    after = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert before == after
    provenance = json.loads((destination / "restart_provenance.json").read_text())
    assert len(provenance["copied_artifacts"]) == 50
    with pytest.raises(FileExistsError):
        module.fork_bfcl_checkpoint(source, destination, 43)


@pytest.mark.parametrize("interleaved", [False, True])
def test_training_reaches_terminal_checkpoint_while_first_evaluation_is_pending(tmp_path, monkeypatch, interleaved):
    module = load_script()
    output = tmp_path / "run"
    run = output / "s43_sequential_t3"
    (run / "checkpoints").mkdir(parents=True)
    (output / "logs").mkdir()
    state_path = run / "state.json"
    config = {"seed": 43}
    module.write_json(state_path, {"cursor": 50, "config": config})
    paths = [run]
    if interleaved:
        paths.append(output / "s42_interleaved_t3")
        module.write_json(paths[1] / "state.json", {"cursor": 0, "config": {"seed": 42}})
        (paths[1] / "checkpoints").mkdir()
    monkeypatch.setattr(module, "pipeline_for", lambda path, *_args: SimpleNamespace(output=path))
    monkeypatch.setattr(module, "training_command", lambda *_args: ["fake-training"])
    turns = []
    enqueued = []

    class FakeProcess:
        pid = 12345

        def __init__(self, cmd, **_kwargs):
            assert cmd == ["fake-training"]

        def poll(self):
            return 0 if len(turns) >= 2 else None

    class FakeQueue:
        def __init__(self, workers):
            assert workers == 50
            self.futures = {}
            self.groups = {}

        def enqueue(self, pipeline, checkpoint):
            key = (str(pipeline.output), checkpoint)
            if checkpoint == 130:
                assert (str(pipeline.output), 100) in self.futures  # Earlier evaluation still running.
            enqueued.append(key)
            self.futures[key] = object()
            self.groups[key] = {"results": [1]}
            return [key]

        def poll(self):
            if len(turns) >= 3:
                self.futures.clear()

        def complete(self, keys):
            return all(key not in self.futures for key in keys)

        def status(self):
            return {"workers": 50}

        def close(self):
            pass

    def advance_training(_seconds):
        turns.append(1)
        for path in paths:
            cursor = (50 if path != run else 100) if len(turns) == 1 else 130
            module.write_json(path / "state.json", {"cursor": cursor, "config": config})
            module.write_json(path / "checkpoints" / f"checkpoint_{cursor:03d}.json", {})
            if cursor == 130:
                module.write_json(path / "checkpoints/checkpoint_100.json", {})

    monkeypatch.setattr(module, "CheckpointEvaluationQueue", FakeQueue)
    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(module.time, "sleep", advance_training)
    assert module.run_controller(
        output, [43], "http://localhost/v1", 50, 30000, interleaved_seeds=[42] if interleaved else [],
    ) == 0
    assert [point for path, point in enqueued if path == str(run)] == [100, 130]
    if interleaved:
        assert [point for path, point in enqueued if path == str(paths[1])] == [50, 100, 130]
        assert json.loads((paths[1] / "test_evaluation/complete.json").read_text())["evaluation_groups"] == 9
    assert json.loads((output / "restart_complete.json").read_text())["success"]


def test_interleaved_restart_reuses_order_but_discards_all_learned_state(tmp_path):
    module = load_script()
    source, destination = tmp_path / "original", tmp_path / "fresh"
    manifest = tmp_path / "manifest.json"
    slugs = ["bfcl", "appworld", "terminalbench2"]
    module.write_json(manifest, {f"{slug}_42": {
        "train": [f"{slug}_train_{i}" for i in range(3)], "val": ["val"], "test": ["test"],
    } for slug in slugs})
    module.PremisePipeline(
        output=source, manifest=manifest, benchmarks=slugs, mode="interleaved", seed=42,
        count=None, past_count=3, model="model", base_url="http://localhost/v1",
    )
    state_path = source / "state.json"
    initial = json.loads(state_path.read_text())
    module.write_json(state_path, {
        **initial, "cursor": 130, "rng_state": "not the initial RNG",
        "states": {"global": {"beam": [{"playbook": "WRONG_COMPLETION_RULE"}], "memory": ["old trace"]}},
    })
    before = state_path.read_bytes()
    module.fork_interleaved_initial(source, destination, 42, "http://localhost/v1")
    fresh = json.loads((destination / "state.json").read_text())
    assert fresh == initial
    assert state_path.read_bytes() == before
    assert not (destination / "checkpoints/checkpoint_050.json").exists()
    provenance = json.loads((destination / "restart_provenance.json").read_text())
    assert provenance["resume_cursor"] == 0


def test_restart_passes_workspace_source_to_isolated_benchmark_subprocesses(monkeypatch):
    module = load_script()
    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    module.configure_workspace_source()
    assert os.environ["PYTHONPATH"].split(os.pathsep) == [str(module.ROOT / "src"), "/existing/path"]
    module.configure_workspace_source()
    assert os.environ["PYTHONPATH"].split(os.pathsep).count(str(module.ROOT / "src")) == 1


def test_adopted_worker_checks_output_identity_pid_reuse_and_completion(tmp_path):
    module = load_script()
    run = tmp_path / "run"
    run.mkdir()
    proc = tmp_path / "proc/123"
    proc.mkdir(parents=True)
    fields = ["S", "1", *["0"] * 17, "100"]
    (proc / "stat").write_text("123 (worker with spaces) " + " ".join(fields))
    argv = ["python", str(module.ROOT / "scripts/premise/run_experiment.py"), "--output-dir", str(run)]
    (proc / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv))

    worker = module.AdoptedTrainingProcess(123, run, tmp_path / "proc")
    assert worker.poll() is None
    with pytest.raises(ValueError, match="not a live training worker"):
        module.AdoptedTrainingProcess(123, tmp_path / "other", tmp_path / "proc")
    fields[19] = "200"  # A live PID reused by another process must not keep training marked as active.
    (proc / "stat").write_text("123 (another worker) " + " ".join(fields))
    assert worker.poll() == 1
    (run / "training_complete.json").write_text('{"completed_task_count":130,"expected_task_count":130}')
    assert worker.poll() == 0


def test_adopted_worker_refresh_waits_for_completed_task_and_preserves_queue(tmp_path, monkeypatch):
    module = load_script()
    run = tmp_path / "s43_sequential_t3"
    (run / "checkpoints").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    state_path = run / "state.json"
    module.write_json(state_path, {"cursor": 100, "config": {"seed": 43}})
    module.write_json(run / "checkpoints/checkpoint_100.json", {})
    turns, stopped, launched = [], [], []

    class Adopted:
        pid = 10

        def __init__(self, pid, output):
            assert pid == 10 and output == run
            self.output = output

        def poll(self):
            return None

    class Fresh:
        pid = 20

        def __init__(self, command, **kwargs):
            assert command == ["new-runtime"]
            assert kwargs["env"]["EXGENTIC_PORT_BLOCK_START"] == "30000"
            assert json.loads(state_path.read_text())["cursor"] == 101
            launched.append(self.pid)

        def poll(self):
            return 0 if len(turns) >= 2 else None

    class Queue:
        def __init__(self, workers):
            assert workers == 50
            self.futures = {}
            self.groups = {}

        def poll(self):
            pass

        def enqueue(self, pipeline, point):
            key = (str(pipeline.output), point)
            self.groups[key] = {"results": [1]}
            return [key]

        def complete(self, keys):
            return all(key in self.groups for key in keys)

        def status(self):
            return {"workers": 50}

        def close(self):
            assert len(self.groups) == 2

    def advance(_seconds):
        if not turns:
            assert not stopped and not launched  # The active task must finish before refresh.
        turns.append(1)
        cursor = 101 if len(turns) == 1 else 130
        module.write_json(state_path, {"cursor": cursor, "config": {"seed": 43}})
        module.write_json(run / "checkpoints" / f"checkpoint_{cursor:03d}.json", {})

    def stop(worker, archive):
        assert worker.pid == 10 and json.loads(state_path.read_text())["cursor"] == 101
        stopped.append(archive)

    monkeypatch.setattr(module, "AdoptedTrainingProcess", Adopted)
    monkeypatch.setattr(module, "stop_training_worker_at_boundary", stop)
    monkeypatch.setattr(module, "pipeline_for", lambda path, *_args: SimpleNamespace(output=path))
    monkeypatch.setattr(module, "training_command", lambda *_args: ["new-runtime"])
    monkeypatch.setattr(module, "CheckpointEvaluationQueue", Queue)
    monkeypatch.setattr(module.subprocess, "Popen", Fresh)
    monkeypatch.setattr(module.time, "sleep", advance)

    assert module.run_controller(tmp_path, [43], "http://localhost/v1", 50, 30000, "refresh",
                                 adopt_training_pids={43: 10}, refresh_adopted_at_boundary=True) == 0
    assert len(stopped) == 1 and launched == [20]
