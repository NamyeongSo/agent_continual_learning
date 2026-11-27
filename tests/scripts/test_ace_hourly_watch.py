# SPDX-License-Identifier: Apache-2.0
"""Exercise hourly recovery decisions without touching a model or live processes."""

from datetime import UTC, datetime, timedelta

import pytest

from scripts.ace import watch_training_first as watcher


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    control = tmp_path / "control"
    watch = control / "hourly_watch"
    watch.mkdir(parents=True)
    watcher.write(control / "status.json", {
        "state": "running", "phase": "all_context_training", "completed_steps": 900, "total_steps": 2892,
    })
    watcher.write(control / "launcher.json", {"pid": 1234, "command": []})
    watcher.write(control / "input_hashes.json", {})
    monkeypatch.setattr(watcher, "endpoint", lambda _: {"healthy": True, "running_requests": 54})
    monkeypatch.setattr(watcher, "process", lambda pid: {"pid": pid})
    events = []
    monkeypatch.setattr(watcher, "stop_owned_controller", lambda *args: events.append("stop") or [1234])
    monkeypatch.setattr(watcher, "start_controller", lambda *args: events.append("resume") or 5678)
    return tmp_path, control, watch, events


def test_healthy_training_is_left_running(cohort):
    output, control, watch, events = cohort
    report = watcher.audit(output, control, watch)
    assert report["state"] == "healthy"
    assert events == []
    assert datetime.fromisoformat(report["next_check_at"]) - datetime.fromisoformat(report["checked_at"]) == (
        timedelta(hours=1)
    )


def test_dead_controller_is_resumed_once_with_saved_configuration(cohort, monkeypatch):
    output, control, watch, events = cohort
    monkeypatch.setattr(watcher, "process", lambda _: None)
    report = watcher.audit(output, control, watch)
    assert report["state"] == "recovered"
    assert report["controller_restarts"] == 1
    assert events == ["stop", "resume"]


def test_user_stop_prevents_all_repairs_and_endpoint_requests(cohort, monkeypatch):
    output, control, watch, events = cohort
    (control / "STOP").touch()
    monkeypatch.setattr(watcher, "endpoint", lambda _: pytest.fail("Must respect STOP before probing"))
    assert watcher.audit(output, control, watch)["state"] == "stopped_by_request"
    assert not events


@pytest.mark.parametrize("message,should_resume", [("litellm.APITimeoutError", True),
                                                 ("ValueError: reflection hash changed", False)])
def test_only_known_infrastructure_errors_are_automatically_resumed(cohort, monkeypatch, message, should_resume):
    output, control, watch, events = cohort
    watcher.write(control / "status.json", {"state": "error", "phase": "error", "error": "Training failed"})
    monkeypatch.setattr(watcher, "infrastructure_evidence", lambda _: [message])
    report = watcher.audit(output, control, watch)
    assert bool(events) == should_resume
    assert report["state"] == ("recovered" if should_resume else "needs_debugging")


def test_changed_inputs_block_restart(cohort, monkeypatch):
    output, control, watch, events = cohort
    pin = control / "protocol.json"
    watcher.write(pin, {"unexpected": "change"})
    watcher.write(control / "input_hashes.json", {str(pin): "old-hash"})
    monkeypatch.setattr(watcher, "process", lambda _: None)
    assert watcher.audit(output, control, watch)["state"] == "needs_debugging"
    assert not events


def test_slow_active_inference_is_reported_without_killing_the_worker(cohort):
    output, control, watch, events = cohort
    status = watcher.read(control / "status.json")
    watcher.write(watch / "latest.json", {
        "checked_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(), "experiment": status,
    })
    report = watcher.audit(output, control, watch)
    assert report["alerts"]
    assert events == []
