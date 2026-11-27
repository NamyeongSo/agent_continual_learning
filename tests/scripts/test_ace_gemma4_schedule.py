# SPDX-License-Identifier: Apache-2.0
"""Prevent Gemma GPU handoff before complete Train coverage and downloaded weights."""

import subprocess
import sys

import pytest

from scripts.ace import schedule_gemma4_train_only as scheduler


def test_completed_status_alone_does_not_release_the_handoff(tmp_path):
    scheduler.base.write(tmp_path / "status.json", {"state": "completed", "phase": "train_only_complete"})
    assert not scheduler.predecessor_complete(tmp_path)


def test_wrong_receipt_or_partial_evaluation_never_releases_the_handoff(tmp_path):
    scheduler.base.write(tmp_path / "status.json", {"state": "completed", "phase": "train_only_complete"})
    scheduler.base.write(tmp_path / "completed.json", {"contexts": 29, "train_evaluations": 1500})
    with pytest.raises(ValueError, match="completion receipt"):
        scheduler.predecessor_complete(tmp_path)
    scheduler.base.write(tmp_path / "completed.json", {"contexts": 30, "train_evaluations": 1500})
    scheduler.base.write(tmp_path / "control/evaluation_queue.json", [])
    with pytest.raises(ValueError, match="coverage"):
        scheduler.predecessor_complete(tmp_path)


def test_complete_train_only_coverage_releases_the_handoff(tmp_path, monkeypatch):
    scheduler.base.write(tmp_path / "status.json", {"state": "completed", "phase": "train_only_complete"})
    scheduler.base.write(tmp_path / "completed.json", {"contexts": 30, "train_evaluations": 1500})
    scheduler.base.write(tmp_path / "control/evaluation_queue.json", [
        {"job": {"phase": "train_final", "id": str(i)}} for i in range(1500)
    ])
    scheduler.base.write(tmp_path / "train_final_by_seed.json", [{"tasks": 50} for _ in range(30)])
    calls = []

    def validate(item):
        calls.append(item["job"]["id"])
        return {"success": False}

    monkeypatch.setattr(scheduler.study.training, "validate_result", validate)
    assert scheduler.predecessor_complete(tmp_path)
    assert len(calls) == 1500


def test_weights_are_not_ready_until_every_shard_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler.Path, "home", lambda: tmp_path)
    root = tmp_path / ".cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/revision"
    scheduler.base.write(root / "model.safetensors.index.json", {"weight_map": {"a": "shard1", "b": "shard2"}})
    for name in ("config.json", "tokenizer.json", "chat_template.jinja", "shard1"):
        (root / name).write_text("not-empty")
    assert not scheduler.checkpoint_ready("google/gemma-4-31B-it", "revision")
    (root / "shard2").write_text("not-empty")
    assert scheduler.checkpoint_ready("google/gemma-4-31B-it", "revision")


def test_scheduler_script_is_executable_from_outside_the_repository(tmp_path):
    result = subprocess.run([sys.executable, scheduler.__file__, "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--predecessor" in result.stdout


def test_periodic_health_checks_use_two_hours_not_one_hour(tmp_path, monkeypatch):
    scheduler.base.write(tmp_path / "thinking/reflection/protocol.json", {"model": "test"})
    completed = iter([False, False, False, False, True])
    times = iter([100000.0, 103600.0, 107199.0, 107200.0])
    clock = {"now": 0.0}
    checks = []

    def monotonic():
        clock["now"] = next(times)
        return clock["now"]

    def endpoint(_):
        checks.append(clock["now"])
        return {"server": {"pid": 42}}

    monkeypatch.setattr(scheduler, "predecessor_complete", lambda _: next(completed))
    monkeypatch.setattr(scheduler.time, "monotonic", monotonic)
    monkeypatch.setattr(scheduler.time, "sleep", lambda _: None)
    monkeypatch.setattr(scheduler.base, "check_endpoint", endpoint)
    monkeypatch.setattr(scheduler, "emit", lambda *args, **kwargs: None)
    scheduler.monitor(tmp_path, tmp_path)
    assert scheduler.CHECK_INTERVAL_SECONDS == 7200
    assert checks == [100000.0, 107200.0]


@pytest.mark.parametrize("marker", ["STOP", "control/STOP", "chain_schedule/STOP"])
def test_resume_respects_every_user_stop_marker(tmp_path, marker):
    stop = tmp_path / marker
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.touch()
    with pytest.raises(InterruptedError, match="STOP marker"):
        scheduler.resume_current(tmp_path)


def test_resume_checks_pins_then_launches_only_the_current_cohort(tmp_path, monkeypatch):
    scheduler.base.write(tmp_path / "input_hashes.json", {})
    scheduler.base.write(tmp_path / "thinking/reflection/protocol.json", {"gpu_physical_id": 0})
    events = []
    monkeypatch.setattr(scheduler.base, "check_endpoint", lambda p: {"server": {"pid": 12}})
    monkeypatch.setattr(scheduler, "launch_controller", lambda p: events.append(("launch", p)) or 34)
    monkeypatch.setattr(scheduler, "monitor", lambda p, s: events.append(("monitor", p, s)))
    scheduler.resume_current(tmp_path)
    assert events == [("launch", tmp_path), ("monitor", tmp_path, tmp_path)]


def test_controller_launcher_pins_explicit_gpu0(tmp_path, monkeypatch):
    from types import SimpleNamespace

    (tmp_path / "control").mkdir()
    scheduler.base.write(tmp_path / "thinking/reflection/protocol.json", {"gpu_physical_id": 0})
    monkeypatch.setattr(scheduler, "controller_command", lambda p: ["controller"])
    captured = {}

    def popen(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)
    assert scheduler.launch_controller(tmp_path) == 123
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert scheduler.base.read(tmp_path / "control/process.json")["pid"] == 123


def test_monitor_resume_attaches_to_a_live_controller_without_duplicate_calls(tmp_path, monkeypatch):
    scheduler.base.write(tmp_path / "input_hashes.json", {})
    scheduler.base.write(tmp_path / "thinking/reflection/protocol.json", {"gpu_physical_id": 0})
    scheduler.base.write(tmp_path / "control/process.json", {"pid": 34})
    events = []
    monkeypatch.setattr(scheduler.base, "check_endpoint", lambda p: {"server": {"pid": 12}})
    monkeypatch.setattr(scheduler, "pid_alive", lambda p: p == 34)
    monkeypatch.setattr(scheduler, "launch_controller", lambda p: pytest.fail("Duplicate controller launch"))
    monkeypatch.setattr(scheduler, "monitor", lambda p, s: events.append((p, s)))
    scheduler.resume_current(tmp_path)
    assert events == [(tmp_path, tmp_path)]
