# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Verify the interactive adapter's Harbor verifier boundary without Docker."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import nullcontext
from ipaddress import IPv4Network
from pathlib import Path
from types import SimpleNamespace

import pytest

from exgentic.benchmarks.terminalbench2.terminalbench2_benchmark import (
    TERMINALBENCH_COMMITS,
    BashAction,
    BashArgs,
    FinishActionForTerminalBench,
    FinishArgs,
    RunTerminus2Action,
    RunTerminus2Args,
    TerminalBench2Benchmark,
)
from exgentic.benchmarks.terminalbench2.terminalbench2_eval import (
    TerminalBench2Session,
    copy_verifier_artifacts,
    list_task_ids,
    quota_thread_environment,
)


@pytest.mark.parametrize("failures", [0, 1, 3])
@pytest.mark.parametrize("reward", [0, 1])
def test_verifier_copy_recovers_official_reward_without_accepting_partial_transfers(
    tmp_path, monkeypatch, failures, reward,
):
    target = tmp_path / "verifier"
    diagnostics = tmp_path / "transfer.json"
    calls = []

    async def sleep(_delay):
        pass

    async def download_dir(source, destination):
        calls.append("directory")
        assert source == "/logs/verifier"
        if len(calls) <= failures:
            (destination / "reward.json").write_text('{"reward": 1}')
            try:
                raise OSError("SSH transfer interrupted")
            except OSError as exc:
                raise RuntimeError("directory copy failed") from exc
        (destination / "reward.json").unlink(missing_ok=True)
        (destination / "reward.txt").write_text(str(reward))

    async def download_file(source, destination):
        calls.append(source)
        if source == "/logs/verifier/reward.txt":
            destination.write_text(str(reward))
        elif source == "/logs/verifier/test-stdout.txt":
            destination.write_text("Official verifier output")
        else:
            raise FileNotFoundError(source)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    asyncio.run(copy_verifier_artifacts(download_dir, download_file, "/logs/verifier", target, diagnostics))
    assert (target / "reward.txt").read_text() == str(reward)
    assert not (target / "reward.json").exists()
    report = json.loads(diagnostics.read_text())
    assert report["status"] == ("recovered" if failures == 3 else "completed")
    assert calls.count("directory") == min(failures + 1, 3)
    if failures:
        assert "SSH transfer interrupted" in report["attempts"][0]["traceback"]
    if failures == 3:
        assert (target / "test-stdout.txt").is_file()


def test_failed_verifier_transfer_never_manufactures_a_reward(tmp_path, monkeypatch):
    async def fail(*_args):
        raise OSError("transport unavailable")

    async def sleep(_delay):
        pass

    monkeypatch.setattr(asyncio, "sleep", sleep)
    with pytest.raises(RuntimeError, match="no official reward"):
        asyncio.run(copy_verifier_artifacts(
            fail, fail, "/logs/verifier", tmp_path / "verifier", tmp_path / "copy.json",
        ))
    assert not list((tmp_path / "verifier").glob("reward.*"))
    assert json.loads((tmp_path / "copy.json").read_text())["status"] == "error"


def test_actor_transport_failure_cannot_be_scored_as_an_official_zero():
    session = object.__new__(TerminalBench2Session)
    session._score = None
    session._terminus_error = "RuntimeError: Docker SSH connection failed before command establishment"
    with pytest.raises(RuntimeError, match=r"execution failed before submission.*Docker SSH"):
        session.score()


def test_successful_empty_directory_transfer_retries_and_recovers_only_official_reward(tmp_path, monkeypatch):
    target = tmp_path / "verifier"
    target.mkdir()
    (target / "reward.json").write_text('{"reward": 1}')
    attempts = []

    async def directory(_source, _destination):
        attempts.append("directory")

    async def file(source, destination):
        if source.endswith("reward.txt"):
            destination.write_text("0")
        else:
            raise FileNotFoundError(source)

    async def sleep(_delay):
        pass

    monkeypatch.setattr(asyncio, "sleep", sleep)
    asyncio.run(copy_verifier_artifacts(directory, file, "/logs/verifier", target, tmp_path / "copy.json"))
    assert len(attempts) == 3
    assert (target / "reward.txt").read_text() == "0" and not (target / "reward.json").exists()
    assert json.loads((tmp_path / "copy.json").read_text())["status"] == "recovered"


def test_task_listing_only_uses_task_definitions(tmp_path: Path) -> None:
    (tmp_path / "valid").mkdir()
    (tmp_path / "valid" / "task.toml").write_text("", encoding="utf-8")
    (tmp_path / "unrelated").mkdir()
    assert list_task_ids(tmp_path) == ["valid"]


def test_runner_timeout_allows_full_official_agent_and_verifier_deadlines(tmp_path):
    task = tmp_path / "long-task"
    task.mkdir()
    (task / "task.toml").write_text("[agent]\ntimeout_sec = 3600\n[verifier]\ntimeout_sec = 12000\n")
    benchmark = TerminalBench2Benchmark(task_root=str(tmp_path))
    assert benchmark.runner_kwargs()["transport_timeout"] == 12600
    assert benchmark.model_copy(update={"runner": "direct"}).runner_kwargs() == {}
    unlimited = benchmark.model_copy(update={"enforce_session_timeouts": False})
    assert unlimited.runner_kwargs()["transport_timeout"] is None
    assert unlimited._get_evaluator_kwargs()["enforce_session_timeouts"] is False


def test_live_old_runner_does_not_forward_new_timeout_to_evaluator(tmp_path, monkeypatch):
    from exgentic.adapters.runners import venv

    class ExistingRunner:
        def __init__(self, target_cls, **kwargs):
            pass

    monkeypatch.setattr(venv, "VenvRunner", ExistingRunner)
    assert "transport_timeout" not in TerminalBench2Benchmark(task_root=str(tmp_path)).runner_kwargs()


@pytest.mark.parametrize("reward", [0, 1])
@pytest.mark.parametrize("network_collision", [False, True])
@pytest.mark.parametrize("enforce_session_timeouts", [False, True])
@pytest.mark.parametrize("capture_timeout", [False, True])
@pytest.mark.parametrize("subset", ["2.0", "2.1"])
def test_terminal_session_grades_after_agent_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reward: int,
    network_collision: bool,
    enforce_session_timeouts: bool,
    capture_timeout: bool,
    subset: str,
) -> None:
    pytest.importorskip("harbor")
    from harbor.environments.capabilities import EnvironmentCapabilities
    from harbor.models.task.config import TaskOS

    task_dir = tmp_path / "demo"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Create /answer.txt", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        'schema_version = "1.1"\n'
        '[task]\nname = "terminal-bench/demo"\n'
        "[agent]\ntimeout_sec = 60\n"
        "[verifier]\ntimeout_sec = 900\n"
        '[environment]\ndocker_image = "ubuntu:24.04"\ncpus = 1\n',
        encoding="utf-8",
    )

    class FakeDocker:
        latest = None

        def __init__(self, **_kwargs):
            self.uploaded_tests = False
            self.commands = []
            self.stopped = False
            self.docker_session_id = _kwargs["session_id"]
            self.session_id = self.docker_session_id
            self.environment_dir = _kwargs["environment_dir"]
            self.network_configs = []
            self.task_env_config = SimpleNamespace(workdir="/workspace")
            FakeDocker.latest = self

        def _resolve_user(self, user):
            return user

        def _merge_env(self, env):
            return env or {}

        @property
        def capabilities(self):
            return EnvironmentCapabilities(mounted=True)

        @property
        def os(self):
            return TaskOS.LINUX

        def with_default_user(self, _user):
            return nullcontext()

        @staticmethod
        def runtime():
            return SimpleNamespace(compose=["fake-compose"], supports_compose_project_directory=False)

        def _compose_env_vars(self, **_kwargs):
            return {}

        async def _collect_buffered_output(self, _process, **_kwargs):
            # Longer than the scaled infrastructure cap, shorter than the
            # official verifier deadline. The real Compose override is used.
            await asyncio.sleep(0.03)
            return SimpleNamespace(stdout="", stderr="", return_code=0)

        @property
        def _docker_compose_paths(self):
            return []

        async def start(self, force_build):
            assert force_build is False
            config = json.loads(self._docker_compose_paths[-1].read_text())
            self.network_configs.append(config)
            assert config["services"]["main"]["environment"]["OMP_NUM_THREADS"] == "1"
            assert config["services"]["main"]["environment"]["MKL_NUM_THREADS"] == "1"
            assert config["services"]["main"]["environment"]["OPAMJOBS"] == "1"
            subnet = IPv4Network(config["networks"]["default"]["ipam"]["config"][0]["subnet"])
            assert subnet.prefixlen == 28
            assert subnet.subnet_of(IPv4Network("10.240.0.0/16"))
            if network_collision and len(self.network_configs) == 1:
                raise RuntimeError("Pool overlaps with other one on this address space")

        async def run_healthcheck(self):
            pass

        async def upload_dir(self, source_dir, target_dir):
            assert Path(source_dir).name == "tests"
            assert target_dir == "/tests"
            self.uploaded_tests = True

        async def exec(self, command, **_kwargs):
            self.commands.append(command)
            if command == "touch /changed" or (capture_timeout and command.startswith("tmux capture-pane")):
                raise TimeoutError("Compose observation stream stalled")
            if "/tests/test.sh" in command and not command.startswith("chmod"):
                return await self._run_docker_compose_command(["exec", "main", command])
            return SimpleNamespace(stdout="/workspace\n", stderr="", return_code=0)

        async def download_dir(self, source_dir, target_dir):
            assert source_dir == "/logs/verifier"
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            (Path(target_dir) / "reward.txt").write_text(f"{reward}\n", encoding="utf-8")

        async def download_file(self, source_path, target_path):
            raise AssertionError("A successful directory copy must not use the file fallback")

        async def stop(self, delete):
            assert delete is True
            self.stopped = True

    import harbor.environments.docker.docker as docker_module

    from exgentic.benchmarks.terminalbench2 import docker_capture

    recoveries = []

    def capture(project, command, **kwargs):
        recoveries.append(command)
        assert project.startswith("exgentic-demo-session-")
        return {"stdout": "fresh recovered screen", "stderr": "", "return_code": 0}

    monkeypatch.setattr(docker_capture, "capture_pane", capture)
    real_wait_for = asyncio.wait_for

    async def wait_for(coro, timeout):
        # Reproduce the 180-second truncation without spending three minutes.
        scaled = {180: 0.01, 900: 1.0}.get(timeout, timeout)
        return await real_wait_for(coro, scaled)

    async def create_subprocess_exec(*_args, **_kwargs):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(docker_module, "DockerEnvironment", FakeDocker)
    monkeypatch.chdir(tmp_path)
    session = TerminalBench2Session(
        str(task_dir), "demo", session_id="demo-session", enforce_session_timeouts=enforce_session_timeouts,
        subset=subset,
    )
    try:
        session.start()
        assert (session._agent_deadline is not None) == enforce_session_timeouts
        assert FakeDocker.latest is not None
        first_docker_session_id = FakeDocker.latest.docker_session_id
        assert len(FakeDocker.latest.network_configs) == (2 if network_collision else 1)
        network_path = FakeDocker.latest._docker_compose_paths[-1]
        assert first_docker_session_id.startswith("exgentic-demo-session-")
        assert FakeDocker.latest.uploaded_tests is False
        assert "test.sh" not in session.task
        observation = session.step(BashAction(arguments=BashArgs(command="pwd")))
        assert "exit_code: 0" in observation.result
        assert FakeDocker.latest.uploaded_tests is False
        observed = session._loop_runner.run(FakeDocker.latest.exec("tmux capture-pane -p -S - -t agent"))
        assert observed.stdout == ("fresh recovered screen" if capture_timeout else "/workspace\n")
        assert len(recoveries) == int(capture_timeout)
        with pytest.raises(TimeoutError, match="observation stream"):
            session._loop_runner.run(FakeDocker.latest.exec("touch /changed"))
        assert len(recoveries) == int(capture_timeout)
        session.step(FinishActionForTerminalBench(arguments=FinishArgs(summary="done")))
        score = session.score()
        assert score.score == float(reward)
        assert score.success is True
        assert score.session_metadata["dataset_version"] == subset
        assert score.session_metadata["dataset_commit"] == TERMINALBENCH_COMMITS[subset]
        assert score.session_metadata["enforce_session_timeouts"] is enforce_session_timeouts
        assert score.session_metadata["phase_timings"]["verifier"]["status"] == "ok"
        assert score.session_metadata["phase_timings"]["verifier"]["elapsed_seconds"] >= 0
        assert FakeDocker.latest.uploaded_tests is True
        assert FakeDocker.latest._exgentic_verifier_active is False
        assert session.paths.benchmark_results.is_file()
        resource = json.loads((session.paths.benchmark_dir / "harbor/container_resources.json").read_text())
        assert resource["version"] == "task-cpu-runtime-v2" and resource["task_cpus"] == 1
    finally:
        session.close()
    assert FakeDocker.latest.stopped is True
    assert not network_path.exists()
    # Two runs of an identical task/config have the same ExGentic session ID.
    # Harbor must still give each execution a separate Compose project.
    repeated = TerminalBench2Session(str(task_dir), "demo", session_id="demo-session")
    try:
        repeated.start()
        assert FakeDocker.latest.docker_session_id != first_docker_session_id
    finally:
        repeated.close()


@pytest.mark.parametrize("enforce_session_timeouts", [False, True])
@pytest.mark.parametrize("model", [
    "openai/test", "openai/Qwen/Qwen3.6-35B-A3B", "openai/google/gemma-4-26B-A4B-it",
    "openai/Qwen/Qwen3.8-27B-FP8",
])
@pytest.mark.parametrize("temperature", [None, 0.7])
@pytest.mark.parametrize("actor_context", ["legacy", "shared-empty-v1"])
def test_terminus2_receives_playbook_and_preserves_trajectory(
    tmp_path, monkeypatch, enforce_session_timeouts, model, temperature, actor_context,
):
    pytest.importorskip("harbor")
    import exgentic.benchmarks.terminalbench2.terminus_limits as terminus_module

    calls = {}
    model_settings = {"max_tokens": None, "reasoning_effort": "xhigh", "top_p": 0.95,
                      "extra_body": {"top_k": 20, "chat_template_kwargs": {"preserve_thinking": True}}} \
        if "Qwen3.8" in model else {}
    terminal_output = "BEGIN " + "terminal output\n" * 5000 + " END"

    class FakeTerminus2:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self._trajectory_steps = [SimpleNamespace(
                source="agent",
                model_dump=lambda **_kwargs: {
                    "message": "solved",
                    "tool_calls": [{"command": "pwd"}],
                    "observation": {"stdout": terminal_output},
                },
            )]

        async def setup(self, environment):
            calls["setup"] = environment

        async def run(self, instruction, environment, context):
            calls["instruction"] = instruction
            context.n_input_tokens = 15
            context.n_output_tokens = 7
            context.metadata = {"n_episodes": 1, "api_request_times_msec": [1234.5]}

    monkeypatch.setattr(terminus_module, "BudgetedTerminus2", FakeTerminus2)
    session = object.__new__(TerminalBench2Session)
    session._loop_runner = asyncio.Runner()
    session._environment = SimpleNamespace(with_default_user=lambda _user: nullcontext())
    session._trial_paths = SimpleNamespace(agent_dir=tmp_path / "agent")
    session._task_obj = SimpleNamespace(
        instruction="Solve task", config=SimpleNamespace(agent=SimpleNamespace(user="root"))
    )
    session._agent_deadline = time.monotonic() + 60 if enforce_session_timeouts else None
    session._max_interactions = 10
    session._action_count = 0
    session._done = False
    try:
        result = session._handle_terminus2(RunTerminus2Action(arguments=RunTerminus2Args(
            model=model, playbook="[str-00001] inspect files", temperature=temperature, actor_context=actor_context,
            model_settings=model_settings,
        )))
    finally:
        session._loop_runner.close()
    assert calls["init"]["max_turns"] == 10
    assert calls["init"]["enable_summarize"] is False
    assert "temperature" not in calls["init"]
    assert calls["init"]["llm_call_kwargs"]["max_tokens"] == (None if model_settings else 2048)
    assert calls["init"]["interleaved_thinking"] == bool(model_settings)
    assert calls["init"]["llm_call_kwargs"].get("temperature") == temperature
    if model_settings:
        for key, value in model_settings.items():
            assert calls["init"]["llm_call_kwargs"][key] == value
    elif model != "openai/test":
        assert calls["init"]["llm_call_kwargs"]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    else:
        expected = {"max_tokens": 2048, **({"temperature": temperature} if temperature is not None else {})}
        assert calls["init"]["llm_call_kwargs"] == expected
    assert "[str-00001] inspect files" in calls["instruction"]
    if actor_context != "legacy":
        from exgentic.actor_context import shared_actor_prompt

        assert calls["instruction"] == "Solve task\n\n" + shared_actor_prompt(
            "terminalbench2", "[str-00001] inspect files",
        )
    assert "solved" in session._terminus_trace
    assert len(session._terminus_trace) > 40000
    assert json.loads(session._terminus_trace)["observation"]["stdout"] == terminal_output
    assert session._terminus_usage == {"input_tokens": 15, "output_tokens": 7}
    assert session._action_count == 1
    assert session._done is True
    assert result["input_tokens"] == 15
    assert session._terminus_metadata["api_request_times_msec"] == [1234.5]
    assert json.loads((tmp_path / "agent" / "execution_metrics.json").read_text()) == session._terminus_metadata
    phases = json.loads((tmp_path / "phase_timings.json").read_text())
    assert phases["agent_run"]["status"] == "ok"
    assert phases["agent_run"]["elapsed_seconds"] >= 0


@pytest.mark.parametrize("task_deadline_expired", [True, False, None])
def test_terminus2_exports_partial_trajectory_only_for_task_deadline(tmp_path, monkeypatch, task_deadline_expired):
    pytest.importorskip("harbor")
    import exgentic.benchmarks.terminalbench2.terminus_limits as terminus_module

    class FakeTerminus2:
        def __init__(self, **kwargs):
            self.logs_dir = kwargs["logs_dir"]
            self._trajectory_steps = [SimpleNamespace(
                source="agent", model_dump=lambda **_kwargs: {"message": "partial progress"},
            )]

        async def setup(self, _environment):
            pass

        async def run(self, _instruction, _environment, context):
            context.n_input_tokens = 17
            context.n_output_tokens = 9
            context.metadata = {"n_episodes": 1}
            (self.logs_dir / "trajectory.json").write_text('{"steps": []}')
            if task_deadline_expired:
                session._agent_deadline = time.monotonic() - 1
            raise TimeoutError("upstream call interrupted")

    monkeypatch.setattr(terminus_module, "BudgetedTerminus2", FakeTerminus2)
    session = object.__new__(TerminalBench2Session)
    session._logger = logging.getLogger("test_terminus_deadline")
    session._loop_runner = asyncio.Runner()
    session._environment = SimpleNamespace(with_default_user=lambda _user: nullcontext())
    session._trial_paths = SimpleNamespace(agent_dir=tmp_path / "agent")
    session._task_obj = SimpleNamespace(
        instruction="Solve task", config=SimpleNamespace(agent=SimpleNamespace(user="root")),
    )
    session._agent_deadline = time.monotonic() + 60 if task_deadline_expired is not None else None
    session._max_interactions = 100
    session._action_count = 0
    session._done = False
    action = RunTerminus2Action(arguments=RunTerminus2Args(model="openai/test"))
    try:
        if not task_deadline_expired:
            with pytest.raises(TimeoutError, match="upstream call interrupted"):
                session._handle_terminus2(action)
            assert session._done is False
            assert session._terminus_error == "TimeoutError: upstream call interrupted"
            return
        result = session._handle_terminus2(action)
        assert result["agent_timed_out"] is True
        assert Path(result["trace_path"]).is_file()
        assert session._terminus_usage == {"input_tokens": 17, "output_tokens": 9}
        assert "partial progress" in session._terminus_trace
        assert session._action_count == 1
        assert session._done is True
        phases = json.loads((tmp_path / "phase_timings.json").read_text())
        assert phases["agent_run"]["error_type"] == "TimeoutError"
        assert json.loads((tmp_path / "agent" / "execution_metrics.json").read_text())["n_episodes"] == 1
    finally:
        session._loop_runner.close()


@pytest.mark.parametrize("enforce_session_timeouts", [False, True])
@pytest.mark.parametrize("capture_logs", [False, True])
def test_verifier_timeout_keeps_timing_and_official_deadline(
    tmp_path, monkeypatch, enforce_session_timeouts, capture_logs,
):
    pytest.importorskip("harbor")
    import harbor.verifier.verifier as verifier_module

    called = []

    class FakeVerifier:
        def __init__(self, **_kwargs):
            pass

        async def verify(self):
            raise TimeoutError("verifier interrupted")

    real_wait_for = asyncio.wait_for

    async def wait_for(coro, timeout):
        called.append(timeout)
        return await real_wait_for(coro, timeout)

    monkeypatch.setattr(verifier_module, "Verifier", FakeVerifier)
    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    session = object.__new__(TerminalBench2Session)
    session._logger = logging.getLogger(__name__)
    session._score = None
    session._enforce_session_timeouts = enforce_session_timeouts
    session._loop_runner = asyncio.Runner()
    session._environment = SimpleNamespace(with_default_user=lambda _user: nullcontext())
    session._trial_paths = SimpleNamespace(agent_dir=tmp_path / "agent", verifier_dir=tmp_path / "verifier")
    if capture_logs:
        async def download_file(source, target):
            assert not source.endswith(("reward.txt", "reward.json"))
            if source.endswith("test-stdout.txt"):
                target.write_text("apt-get was still running")
            else:
                raise FileNotFoundError(source)

        session._environment.download_file = download_file
    session._task_obj = SimpleNamespace(config=SimpleNamespace(
        verifier=SimpleNamespace(user="root", timeout_sec=7200),
    ))
    try:
        with pytest.raises(TimeoutError, match="verifier interrupted"):
            session.score()
    finally:
        session._loop_runner.close()
    assert called == [7200 if enforce_session_timeouts else None] + ([15] if capture_logs else [])
    assert session._score is None
    assert session._environment._exgentic_verifier_active is False
    if capture_logs:
        assert (tmp_path / "verifier/test-stdout.txt").is_file()
    phases = json.loads((tmp_path / "phase_timings.json").read_text())
    assert phases["verifier"]["status"] == "error"
    assert phases["verifier"]["error_type"] == "TimeoutError"
    assert "verifier interrupted" in phases["verifier"]["traceback"]
    assert phases["verifier"]["elapsed_seconds"] >= 0


def test_thread_budget_follows_declared_cpu_quota_and_preserves_explicit_task_settings():
    defaults = quota_thread_environment(2, {"OMP_NUM_THREADS": "7", "OTHER": "value"})
    assert defaults["OPAMJOBS"] == "2"
    assert "OPAMJOBS" not in quota_thread_environment(2, {"OPAMJOBS": "4"})
    assert "OMP_NUM_THREADS" not in defaults
    assert defaults["MKL_NUM_THREADS"] == "2"
    assert defaults["OPENBLAS_NUM_THREADS"] == "2"
    assert quota_thread_environment(None, {}) == {}
