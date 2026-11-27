# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Lightweight configuration; the official harness runs beside its Docker engine."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from ...core import Benchmark
from ...core.types import FinishAction, SingleAction
from ...utils.settings import RunnerName

HARNESS_COMMIT = "17a8f1593e172e26b36cea15e2b30fb9536c93f5"  # pragma: allowlist secret
DATA_COMMIT = "3500478e740d42583277c2658aa7729eb36e10f4"  # pragma: allowlist secret
BENCHMARK_VERSION = "v1.0.2"
REPOSITORIES = {
    "ripgrep": "BurntSushi_ripgrep_14.1.1_15.0.0",
    "navidrome": "navidrome_navidrome_v0.57.0_v0.58.0",
    "dubbo": "apache_dubbo_dubbo-3.3.3_dubbo-3.3.6",
    "go-zero": "zeromicro_go-zero_v1.6.0_v1.9.3",
    "nushell": "nushell_nushell_0.106.0_0.108.0",
    "element-web": "element-hq_element-web_v1.11.95_v1.11.97",
    "scikit-learn": "scikit-learn_scikit-learn_1.5.2_1.6.0",
}


class BashArgs(BaseModel):
    command: str = Field(description="Bash command to execute in the persistent development container")


class BashAction(SingleAction):
    name: str = "bash"
    arguments: BashArgs


class SubmitArgs(BaseModel):
    summary: str = Field(default="", description="Brief summary of this milestone's implementation")


class SubmitMilestoneAction(FinishAction):
    name: str = "finish"
    arguments: SubmitArgs


class RunTerminus2Args(BaseModel):
    model: str = Field(description="LiteLLM model used by Terminus-2")
    playbook: str = Field(default="", description="Strategies learned from earlier milestones")


class RunTerminus2Action(SingleAction):
    name: str = "run_terminus2"
    arguments: RunTerminus2Args


class SWEMilestoneBenchmark(Benchmark):
    display_name: ClassVar[str] = "SWE-Milestone"
    slug_name: ClassVar[str] = "swemilestone"
    requires_sequential_sessions: ClassVar[bool] = True

    subset: str = Field(
        default="ripgrep", pattern="^(all|ripgrep|navidrome|dubbo|go-zero|nushell|element-web|scikit-learn)$"
    )
    # Upstream dependencies are isolated on the Docker host; the bridge uses
    # Exgentic core/stdlib and can wait for hour-long official test suites.
    runner: RunnerName | None = "direct"
    max_interactions: int | None = 200
    docker_host: str | None = None
    backend_root: str | None = None
    stream_id: str | None = None
    start_snapshot: dict[str, Any] | None = None
    command_timeout_sec: int = Field(default=900, ge=1)
    evaluation_timeout_sec: int = Field(default=10800, ge=1)
    observation_size_limit: int = Field(default=10000, ge=1)
    pull_images: bool = True
    actor: str = Field(default="interactive", pattern="^(interactive|terminus2)$")
    terminus_python: str | None = None
    agent_timeout_sec: int = Field(default=10800, ge=1)
    model_info: dict[str, Any] | None = None

    @classmethod
    def _get_evaluator_class(cls):
        return "exgentic.benchmarks.swemilestone.swemilestone_eval:SWEMilestoneEvaluator"

    @classmethod
    def _get_session_class(cls):
        return "exgentic.benchmarks.swemilestone.swemilestone_eval:SWEMilestoneSession"

    def _get_evaluator_kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"runner", "docker_socket", "seed", "use_cache"})
