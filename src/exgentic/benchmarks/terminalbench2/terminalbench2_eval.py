# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Interactive Terminal-Bench 2.0 sessions using Harbor's Docker environment and verifier.

The agent sees only instruction.md and the live container. Harbor uploads tests
after the agent has stopped acting, then runs the task's official test.sh.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ...core.actions import ActionsHandler
from ...core.evaluator import Evaluator
from ...core.session import Session
from ...core.types import Action, ActionType, BenchmarkResults, SessionIndex, SessionScore, SingleObservation
from .terminalbench2_benchmark import (
    TERMINALBENCH2_COMMIT,
    BashAction,
    FinishActionForTerminalBench,
    RunTerminus2Action,
    default_task_root,
)


def resolve_task_root(task_root: str | None) -> Path:
    root = Path(task_root).expanduser() if task_root else default_task_root()
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Terminal-Bench 2.0 tasks missing at {root}. Run 'exgentic install --benchmark terminalbench2' "
            "or set EXGENTIC_TERMINALBENCH2_TASK_ROOT."
        )
    if (root / ".git").exists():
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if head != TERMINALBENCH2_COMMIT:
            raise ValueError(
                f"Terminal-Bench task checkout is {head}; expected pinned 2.0 commit {TERMINALBENCH2_COMMIT}"
            )
    return root


def list_task_ids(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "task.toml").is_file())


def truncate_output(output: str, limit: int) -> str:
    if len(output) <= limit:
        return output
    head = limit // 2
    tail = limit - head
    return output[:head] + f"\n... {len(output) - limit} characters omitted ...\n" + output[-tail:]


class TerminalBench2Session(Session):
    """One task container, operated by Exgentic and graded by Harbor."""

    def __init__(
        self,
        task_dir: str,
        task_id: str,
        max_interactions: int | None = 100,
        command_timeout_sec: int = 900,
        observation_size_limit: int = 10000,
        actor: str = "interactive",
        session_id: str | None = None,
    ) -> None:
        from harbor.models.task.config import TaskOS, VerifierEnvironmentMode
        from harbor.models.task.task import Task
        from harbor.models.task.verifier_mode import resolve_task_verifier_mode

        self._task_obj = Task(task_dir)
        if self._task_obj.has_steps or self._task_obj.config.environment.os != TaskOS.LINUX:
            raise ValueError("Terminal-Bench 2.0 adapter requires a single-step Linux task")
        if resolve_task_verifier_mode(self._task_obj.config) != VerifierEnvironmentMode.SHARED:
            raise ValueError("Terminal-Bench 2.0 adapter requires a shared Harbor verifier environment")
        if self._task_obj.short_name != task_id:
            raise ValueError(f"Task ID {task_id!r} does not match Harbor task {self._task_obj.name!r}")

        self._task_id = task_id
        if session_id is not None:
            self._session_id = session_id
        self._max_interactions = max_interactions
        self._command_timeout_sec = command_timeout_sec
        self._observation_size_limit = observation_size_limit
        if actor not in {"interactive", "terminus2"}:
            raise ValueError(f"Unknown Terminal-Bench actor: {actor}")
        self._actor = actor
        self._action_count = 0
        self._done = False
        self._score: SessionScore | None = None
        self._loop_runner: asyncio.Runner | None = None
        self._environment = None
        self._trial_paths = None
        self._agent_deadline: float | None = None
        self._terminus_trace: str | None = None
        self._terminus_usage: dict[str, int] | None = None
        self._registry = ActionsHandler(
            logger=self.logger,
            warn_on_validation_error=False,
            warn_on_unknown_action=True,
            handle_validation_error=None,
            handle_unknown_action=None,
        )
        super().__init__()

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def task(self) -> str:
        if self._actor == "terminus2":
            return self._task_obj.instruction
        return (
            self._task_obj.instruction
            + "\n\nUse the bash action to work in the task container. Each command runs in a new shell; "
            "filesystem changes persist. Call finish when done. The official Terminal-Bench verifier runs afterward."
        )

    @property
    def context(self) -> dict[str, Any]:
        return {"benchmark": "terminalbench2", "task_id": self._task_id}

    @property
    def actions(self) -> list[ActionType]:
        if not self._registry.actions:
            if self._actor == "terminus2":
                self._registry.add_action(
                    name="run_terminus2",
                    description="Run Harbor Terminus-2 in the task container with the ACE playbook",
                    action_cls=RunTerminus2Action,
                    handler=self._handle_terminus2,
                    is_finish=True,
                )
                return self._registry.actions
            self._registry.add_action(
                name="bash",
                description="Execute a shell command inside the Terminal-Bench task container",
                action_cls=BashAction,
                handler=self._handle_bash,
            )
            self._registry.add_action(
                name="finish",
                description="Finish work and submit the current container state for official verification",
                action_cls=FinishActionForTerminalBench,
                handler=self._handle_finish,
                is_finish=True,
            )
        return self._registry.actions

    def start(self) -> SingleObservation:
        from harbor.environments.docker.docker import DockerEnvironment
        from harbor.models.trial.paths import TrialPaths

        class CopyingDockerEnvironment(DockerEnvironment):
            """Copy verifier artifacts via Docker CLI, including with remote DOCKER_HOST."""

            @property
            def capabilities(self):
                return super().capabilities.model_copy(update={"mounted": False})

        if self._loop_runner is not None:
            raise RuntimeError("Terminal-Bench session already started")
        self._loop_runner = asyncio.Runner()
        task = self._task_obj
        trial_paths = TrialPaths(self.paths.benchmark_dir / "harbor")
        trial_paths.mkdir()
        self._trial_paths = trial_paths
        self._environment = CopyingDockerEnvironment(
            environment_dir=task.paths.environment_dir,
            environment_name=task.short_name,
            session_id=f"exgentic-{self.session_id}",
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
            logger=self.logger,
            mounts=[],
            network_policy=task.config.environment.resolve_baseline(),
        )
        try:
            self._loop_runner.run(self._environment.start(force_build=False))
            self._loop_runner.run(
                self._environment.exec("mkdir -p /logs/agent /logs/verifier /logs/artifacts", user="root")
            )
            self._loop_runner.run(self._environment.run_healthcheck())
            self._agent_deadline = time.monotonic() + self._task_obj.config.agent.timeout_sec
        except BaseException:
            self.close()
            raise
        return SingleObservation(invoking_actions=[], result=None)

    def _handle_bash(self, action: BashAction) -> str:
        if self._done:
            return "Terminal-Bench task is already finished."
        if self._loop_runner is None or self._environment is None:
            raise RuntimeError("Terminal-Bench session has not started")
        remaining = (self._agent_deadline or time.monotonic()) - time.monotonic()
        if remaining <= 0:
            self._done = True
            return "Terminal-Bench agent time limit reached."
        with self._environment.with_default_user(self._task_obj.config.agent.user):
            try:
                result = self._loop_runner.run(
                    self._environment.exec(
                        action.arguments.command,
                        timeout_sec=max(1, min(self._command_timeout_sec, int(remaining))),
                    )
                )
            except RuntimeError as exc:
                if "timed out" not in str(exc).lower():
                    raise
                return f"Command timed out: {exc}"
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        rendered = f"exit_code: {result.return_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        return truncate_output(rendered, self._observation_size_limit)

    def _handle_finish(self, action: FinishActionForTerminalBench) -> None:
        self._done = True

    def _handle_terminus2(self, action: RunTerminus2Action) -> dict[str, Any]:
        if self._loop_runner is None or self._environment is None or self._trial_paths is None:
            raise RuntimeError("Terminal-Bench session has not started")
        from harbor.agents.terminus_2 import Terminus2
        from harbor.models.agent.context import AgentContext

        logs_dir = self._trial_paths.agent_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        agent = Terminus2(
            logs_dir=logs_dir,
            model_name=action.arguments.model,
            api_base=os.environ.get("OPENAI_API_BASE"),
            max_turns=self._max_interactions,
            record_terminal_session=False,
            llm_call_kwargs={
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
            } if "Qwen3.6" in action.arguments.model else None,
        )
        context = AgentContext()
        instruction = self._task_obj.instruction
        if action.arguments.playbook.strip():
            instruction += (
                "\n\nACE playbook from previous tasks (use relevant strategies; "
                "mention a bullet ID when it influences an action):\n"
                + action.arguments.playbook
            )
        remaining = (self._agent_deadline or time.monotonic()) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Terminal-Bench agent time limit reached before Terminus-2 started")
        with self._environment.with_default_user(self._task_obj.config.agent.user):
            self._loop_runner.run(asyncio.wait_for(agent.setup(self._environment), timeout=min(360, remaining)))
            remaining = (self._agent_deadline or time.monotonic()) - time.monotonic()
            self._loop_runner.run(
                asyncio.wait_for(agent.run(instruction, self._environment, context), timeout=max(1, remaining))
            )
        self._done = True
        self._action_count = int((context.metadata or {}).get("n_episodes", 0))
        # Keep the full ATIF trajectory on disk; send a bounded trace to ACE's reflector.
        trace_parts = []
        for step in agent._trajectory_steps:
            if step.source != "agent":
                continue
            payload = step.model_dump(exclude_none=True, exclude={"metrics", "timestamp", "model_name"})
            trace_parts.append(truncate_output(json.dumps(payload, ensure_ascii=False, default=str), 3000))
        self._terminus_trace = truncate_output("\n".join(trace_parts), 40000)
        self._terminus_usage = {
            "input_tokens": context.n_input_tokens or 0,
            "output_tokens": context.n_output_tokens or 0,
        }
        return {
            "trace_path": str(logs_dir / "trajectory.json"),
            **self._terminus_usage,
        }

    def step(self, action: Action) -> SingleObservation | None:
        if self._done:
            return None
        if self._agent_deadline is not None and time.monotonic() >= self._agent_deadline:
            self._done = True
            return None
        count = len(action.to_action_list())
        if self._max_interactions is not None and self._action_count + count > self._max_interactions:
            self.logger.warning("Terminal-Bench action limit reached: %s", self._max_interactions)
            self._done = True
            return None
        observation = self._registry.execute(action)
        if self._actor != "terminus2":
            self._action_count += count
        return observation

    def done(self) -> bool:
        return self._done

    def score(self) -> SessionScore:
        if self._score is not None:
            return self._score
        if self._loop_runner is None or self._environment is None or self._trial_paths is None:
            raise RuntimeError("Cannot score Terminal-Bench session before its environment starts")
        from harbor.verifier.verifier import Verifier

        verifier = Verifier(
            task=self._task_obj,
            trial_paths=self._trial_paths,
            environment=self._environment,
            logger=self.logger,
        )
        with self._environment.with_default_user(self._task_obj.config.verifier.user):
            result = self._loop_runner.run(
                asyncio.wait_for(verifier.verify(), timeout=self._task_obj.config.verifier.timeout_sec)
            )
        rewards = result.rewards or {}
        if "reward" not in rewards:
            raise RuntimeError(f"Terminal-Bench verifier produced no reward for {self._task_id}")
        reward = float(rewards["reward"])
        if not 0 <= reward <= 1:
            raise ValueError(f"Terminal-Bench reward for {self._task_id} is outside [0,1]: {reward}")
        self._score = SessionScore(
            score=reward,
            success=True,
            is_finished=self._done,
            session_metadata={
                "task_id": self._task_id,
                "harbor_rewards": rewards,
                "harbor_verifier_dir": str(self._trial_paths.verifier_dir),
                "actions": self._action_count,
                "actor": self._actor,
                **({"agent_trace": self._terminus_trace} if self._terminus_trace is not None else {}),
                **({"agent_usage": self._terminus_usage} if self._terminus_usage is not None else {}),
            },
        )
        self.save_results(self._score.model_dump())
        return self._score

    def close(self) -> None:
        if self._loop_runner is not None:
            try:
                if self._environment is not None:
                    self._loop_runner.run(self._environment.stop(delete=True))
            finally:
                self._loop_runner.close()
                self._loop_runner = None
                self._environment = None


class TerminalBench2Evaluator(Evaluator):
    def __init__(
        self,
        task_root: str | None = None,
        max_interactions: int | None = 100,
        command_timeout_sec: int = 900,
        observation_size_limit: int = 10000,
        actor: str = "interactive",
    ) -> None:
        self._root = resolve_task_root(task_root)
        self._ids = list_task_ids(self._root)
        if len(self._ids) != 89:
            raise RuntimeError(f"Expected 89 Terminal-Bench 2.0 tasks at {self._root}, found {len(self._ids)}")
        self._max_interactions = max_interactions
        self._command_timeout_sec = command_timeout_sec
        self._observation_size_limit = observation_size_limit
        self._actor = actor

    def list_tasks(self) -> list[str]:
        return list(self._ids)

    def get_session_kwargs(self, index: SessionIndex) -> dict[str, Any]:
        task_id = str(index.task_id)
        if task_id not in self._ids:
            raise KeyError(f"Unknown Terminal-Bench 2.0 task: {task_id}")
        return {
            "task_dir": str(self._root / task_id),
            "task_id": task_id,
            "max_interactions": self._max_interactions,
            "command_timeout_sec": self._command_timeout_sec,
            "observation_size_limit": self._observation_size_limit,
            "actor": self._actor,
            "session_id": index.session_id,
        }

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        scores = []
        for paths in self.get_sessions_paths(sessions):
            if not paths.benchmark_results.is_file():
                raise FileNotFoundError(f"Missing Terminal-Bench result: {paths.benchmark_results}")
            payload = json.loads(paths.benchmark_results.read_text(encoding="utf-8"))
            scores.append(float(payload["score"]))
        return BenchmarkResults(
            benchmark_name="terminalbench2",
            total_tasks=len(scores),
            score=sum(scores) / len(scores) if scores else 0.0,
            metrics={"resolved": sum(score == 1.0 for score in scores)},
        )
