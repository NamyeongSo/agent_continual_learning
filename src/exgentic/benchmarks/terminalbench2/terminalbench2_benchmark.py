# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Lightweight Terminal-Bench 2.0 configuration; Harbor loads in the benchmark runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from ...core import Benchmark
from ...core.types import FinishAction, SingleAction
from ...utils.settings import RunnerName

TERMINALBENCH2_COMMIT = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"  # pragma: allowlist secret


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


class RunTerminus2Action(SingleAction):
    name: str = "run_terminus2"
    arguments: RunTerminus2Args


class TerminalBench2Benchmark(Benchmark):
    display_name: ClassVar[str] = "Terminal-Bench 2.0"
    slug_name: ClassVar[str] = "terminalbench2"

    subset: str = "2.0"
    task_root: str | None = None
    runner: RunnerName | None = "venv"
    docker_socket: bool = True
    max_interactions: int | None = 100
    command_timeout_sec: int = Field(default=900, ge=1)
    observation_size_limit: int = Field(default=10000, ge=1)
    actor: str = Field(default="interactive", pattern="^(interactive|terminus2)$")

    @classmethod
    def _get_evaluator_class(cls):
        return "exgentic.benchmarks.terminalbench2.terminalbench2_eval:TerminalBench2Evaluator"

    @classmethod
    def _get_session_class(cls):
        return "exgentic.benchmarks.terminalbench2.terminalbench2_eval:TerminalBench2Session"

    def _get_evaluator_kwargs(self) -> dict[str, Any]:
        if self.subset != "2.0":
            raise ValueError("TerminalBench2Benchmark supports only the pinned 2.0 task release")
        return {
            "task_root": self.task_root,
            "max_interactions": self.max_interactions,
            "command_timeout_sec": self.command_timeout_sec,
            "observation_size_limit": self.observation_size_limit,
            "actor": self.actor,
        }


def default_task_root() -> Path:
    import os

    override = os.environ.get("EXGENTIC_TERMINALBENCH2_TASK_ROOT")
    return (
        Path(override).expanduser()
        if override
        else Path.home() / ".exgentic/benchmarks/terminalbench2/terminal-bench-2"
    )
