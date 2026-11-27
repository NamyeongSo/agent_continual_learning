# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Pinned Terminal-Bench 2.0/2.1 configuration; Harbor loads in the benchmark runner."""

from __future__ import annotations

import inspect
import tomllib
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from ...actor_context import ActorContextMode
from ...core import Benchmark
from ...core.experiment_limits import BENCHMARK_MAX_TURNS
from ...core.types import FinishAction, SingleAction
from ...utils.settings import RunnerName

TERMINALBENCH2_COMMIT = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"  # pragma: allowlist secret
TERMINALBENCH21_COMMIT = "12b7565679ce5006230b820666353673a133db93"  # pragma: allowlist secret
TERMINALBENCH_COMMITS = {"2.0": TERMINALBENCH2_COMMIT, "2.1": TERMINALBENCH21_COMMIT}


class BashArgs(BaseModel):
    command: str = Field(description="Shell command to execute in the task container")


class BashAction(SingleAction):
    name: str = "bash"
    arguments: BashArgs


class FinishArgs(BaseModel):
    summary: str = Field(default="", description="Brief summary of completed work")


class FinishActionForTerminalBench(FinishAction):
    name: str = "finish"
    arguments: FinishArgs


class RunTerminus2Args(BaseModel):
    model: str = Field(description="LiteLLM model name used by Terminus-2")
    playbook: str = Field(default="", description="ACE strategies learned from earlier tasks")
    actor_context: ActorContextMode = "legacy"
    record_episodes: bool = False
    record_policy_shift: bool = False
    replay_episodes_path: str | None = None
    replay_episodes_count: int = Field(default=0, ge=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    model_settings: dict[str, Any] = Field(default_factory=dict)
    model_info: dict[str, Any] | None = None


class RunTerminus2Action(SingleAction):
    name: str = "run_terminus2"
    arguments: RunTerminus2Args


class TerminalBench2Benchmark(Benchmark):
    display_name: ClassVar[str] = "Terminal-Bench 2.0 / 2.1"
    slug_name: ClassVar[str] = "terminalbench2"

    subset: Literal["2.0", "2.1"] = "2.0"
    task_root: str | None = None
    runner: RunnerName | None = "venv"
    docker_socket: bool = True
    max_interactions: int | None = BENCHMARK_MAX_TURNS["terminalbench2"]
    command_timeout_sec: int = Field(default=900, ge=1)
    observation_size_limit: int = Field(default=10000, ge=1)
    actor: str = Field(default="interactive", pattern="^(interactive|terminus2)$")
    enforce_session_timeouts: bool = True
    exclude_model_call_time: bool = False
    native_terminus2: bool = False
    native_context_recovery: bool = False

    @classmethod
    def _get_evaluator_class(cls):
        return "exgentic.benchmarks.terminalbench2.terminalbench2_eval:TerminalBench2Evaluator"

    @classmethod
    def _get_session_class(cls):
        return "exgentic.benchmarks.terminalbench2.terminalbench2_eval:TerminalBench2Session"

    def _get_evaluator_kwargs(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "task_root": self.task_root,
            "max_interactions": self.max_interactions,
            "command_timeout_sec": self.command_timeout_sec,
            "observation_size_limit": self.observation_size_limit,
            "actor": self.actor,
            "enforce_session_timeouts": self.enforce_session_timeouts,
            "exclude_model_call_time": self.exclude_model_call_time,
            "native_terminus2": self.native_terminus2,
            "native_context_recovery": self.native_context_recovery,
        }

    def runner_kwargs(self) -> dict[str, Any]:
        kwargs = super().runner_kwargs()
        if self.resolve_runner() != "venv":
            return kwargs
        from ...adapters.runners.venv import VenvRunner

        # A live experiment may import this benchmark for the first time after
        # updating code while retaining an older runner in memory. Avoid passing
        # its unsupported option into the remote evaluator constructor.
        if "transport_timeout" not in inspect.signature(VenvRunner).parameters:
            return kwargs
        if not self.enforce_session_timeouts or self.exclude_model_call_time:
            kwargs["transport_timeout"] = None
            return kwargs
        # run_terminus2() is one RPC covering the complete agent run. The generic
        # 30-minute RPC timeout must not interrupt a longer official task deadline.
        root = Path(self.task_root).expanduser() if self.task_root else default_task_root(self.subset)
        if (root / "tasks").is_dir():
            root = root / "tasks"
        deadlines = [1200.0]
        for path in root.glob("*/task.toml"):
            config = tomllib.loads(path.read_text(encoding="utf-8"))
            for section, field in (("agent", "timeout_sec"), ("verifier", "timeout_sec"),
                                   ("environment", "build_timeout_sec")):
                deadlines.append(float(config.get(section, {}).get(field, 600)))
        kwargs["transport_timeout"] = max(deadlines) + 600
        return kwargs


def default_task_root(subset: str = "2.0") -> Path:
    import os

    if subset not in TERMINALBENCH_COMMITS:
        raise ValueError(f"Unsupported Terminal-Bench release: {subset}")
    variable = "EXGENTIC_TERMINALBENCH21_TASK_ROOT" if subset == "2.1" else "EXGENTIC_TERMINALBENCH2_TASK_ROOT"
    if override := os.environ.get(variable):
        return Path(override).expanduser()
    base = Path.home() / ".exgentic/benchmarks/terminalbench2"
    return base / "terminal-bench-2-1/tasks" if subset == "2.1" else base / "terminal-bench-2"
