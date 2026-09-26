# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Verify the interactive adapter's Harbor verifier boundary without Docker."""

from __future__ import annotations

import asyncio
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from exgentic.benchmarks.terminalbench2.terminalbench2_benchmark import (
    BashAction,
    BashArgs,
    FinishActionForTerminalBench,
    FinishArgs,
    RunTerminus2Action,
    RunTerminus2Args,
)
from exgentic.benchmarks.terminalbench2.terminalbench2_eval import TerminalBench2Session, list_task_ids


def test_task_listing_only_uses_task_definitions(tmp_path: Path) -> None:
    (tmp_path / "valid").mkdir()
    (tmp_path / "valid" / "task.toml").write_text("", encoding="utf-8")
    (tmp_path / "unrelated").mkdir()
    assert list_task_ids(tmp_path) == ["valid"]


@pytest.mark.parametrize("reward", [0, 1])
def test_terminal_session_grades_after_agent_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reward: int,
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
        "[verifier]\ntimeout_sec = 60\n"
        '[environment]\ndocker_image = "ubuntu:24.04"\n',
        encoding="utf-8",
    )

    class FakeDocker:
        latest = None

        def __init__(self, **_kwargs):
            self.uploaded_tests = False
            self.commands = []
            self.stopped = False
            FakeDocker.latest = self

        @property
        def capabilities(self):
            return EnvironmentCapabilities(mounted=True)

        @property
        def os(self):
            return TaskOS.LINUX

        def with_default_user(self, _user):
            return nullcontext()

        async def start(self, force_build):
            assert force_build is False

        async def run_healthcheck(self):
            pass

        async def upload_dir(self, source_dir, target_dir):
            assert Path(source_dir).name == "tests"
            assert target_dir == "/tests"
            self.uploaded_tests = True

        async def exec(self, command, **_kwargs):
            self.commands.append(command)
            return SimpleNamespace(stdout="/workspace\n", stderr="", return_code=0)

        async def download_dir(self, source_dir, target_dir):
            assert source_dir == "/logs/verifier"
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            (Path(target_dir) / "reward.txt").write_text(f"{reward}\n", encoding="utf-8")

        async def stop(self, delete):
            assert delete is True
            self.stopped = True

    import harbor.environments.docker.docker as docker_module

    monkeypatch.setattr(docker_module, "DockerEnvironment", FakeDocker)
    monkeypatch.chdir(tmp_path)
    session = TerminalBench2Session(str(task_dir), "demo", session_id="demo-session")
    try:
        session.start()
        assert FakeDocker.latest is not None
        assert FakeDocker.latest.uploaded_tests is False
        assert "test.sh" not in session.task
        observation = session.step(BashAction(arguments=BashArgs(command="pwd")))
        assert "exit_code: 0" in observation.result
        assert FakeDocker.latest.uploaded_tests is False
        session.step(FinishActionForTerminalBench(arguments=FinishArgs(summary="done")))
        score = session.score()
        assert score.score == float(reward)
        assert score.success is True
        assert FakeDocker.latest.uploaded_tests is True
        assert session.paths.benchmark_results.is_file()
    finally:
        session.close()
    assert FakeDocker.latest.stopped is True


def test_terminus2_receives_playbook_and_preserves_trajectory(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    import harbor.agents.terminus_2 as terminus_module

    calls = {}

    class FakeTerminus2:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self._trajectory_steps = [SimpleNamespace(
                source="agent",
                model_dump=lambda **_kwargs: {"message": "solved", "tool_calls": [{"command": "pwd"}]},
            )]

        async def setup(self, environment):
            calls["setup"] = environment

        async def run(self, instruction, environment, context):
            calls["instruction"] = instruction
            context.n_input_tokens = 15
            context.n_output_tokens = 7
            context.metadata = {"n_episodes": 1}

    monkeypatch.setattr(terminus_module, "Terminus2", FakeTerminus2)
    session = object.__new__(TerminalBench2Session)
    session._loop_runner = asyncio.Runner()
    session._environment = SimpleNamespace(with_default_user=lambda _user: nullcontext())
    session._trial_paths = SimpleNamespace(agent_dir=tmp_path / "agent")
    session._task_obj = SimpleNamespace(
        instruction="Solve task", config=SimpleNamespace(agent=SimpleNamespace(user="root"))
    )
    session._agent_deadline = time.monotonic() + 60
    session._max_interactions = 10
    session._action_count = 0
    session._done = False
    try:
        result = session._handle_terminus2(RunTerminus2Action(arguments=RunTerminus2Args(
            model="openai/test", playbook="[str-00001] inspect files",
        )))
    finally:
        session._loop_runner.close()
    assert calls["init"]["max_turns"] == 10
    assert "[str-00001] inspect files" in calls["instruction"]
    assert "solved" in session._terminus_trace
    assert session._terminus_usage == {"input_tokens": 15, "output_tokens": 7}
    assert session._action_count == 1
    assert session._done is True
    assert result["input_tokens"] == 15
