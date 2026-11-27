from __future__ import annotations

import sys
from types import SimpleNamespace

from exgentic.benchmarks.swebench.swebench_evaluation import _docker_ssh_transport
from exgentic.benchmarks.swebench.swebench_benchmark import SessionScore
from exgentic.benchmarks.swebench.swebench_logs import _apply_harness_data


def test_harness_uses_ssh_cli_transport_and_restores_docker_sdk(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "ssh://benchmark-host:12345")
    original = lambda **kwargs: kwargs
    docker = SimpleNamespace(from_env=original)
    monkeypatch.setitem(sys.modules, "docker", docker)

    with _docker_ssh_transport():
        assert docker.from_env() == {"use_ssh_client": True}
        assert docker.from_env(use_ssh_client=False) == {"use_ssh_client": False}

    assert docker.from_env is original


def test_harness_transport_error_marks_benchmark_session_as_error():
    score = SessionScore(score=0.0, success=False, instance_id="example")
    harness = SimpleNamespace(harness_report={"submitted_ids": ["example"]}, error="SSH broken pipe")
    _apply_harness_data(score, harness, "example")
    assert score.session_metadata["error_source"] == "benchmark"
    assert score.session_metadata["error"] == "SSH broken pipe"
