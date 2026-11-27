# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""AppWorld benchmark adapter -- light benchmark class only.

Evaluator and session classes live in ``appworld_eval.py`` and are loaded
inside the runner subprocess via ``_get_evaluator_class()`` and
``_get_session_class()``.  This file must remain importable without the
``appworld`` package installed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...core.benchmark import Benchmark
from ...core.types import SessionIndex, SingleObservation
from ...utils.paths import get_run_id


class AppWorldObservation(SingleObservation):
    pass


def make_session_kwargs(
    index: SessionIndex,
    *,
    env_kwargs: dict[str, Any],
    max_interactions: int,
    tool_name_separator: str,
    use_cache: bool,
    experiment_name: str,
    policy_override: str | None = None,
    interaction_mode: Literal["code_generation", "function_calling"] = "code_generation",
) -> dict[str, Any]:
    """Build AppWorld metadata without importing AppWorld or loading a task."""
    kwargs = {
        "session_id": index.session_id,
        "task_spec": {"task_id": index.task_id},
        "env_kwargs": {
            **env_kwargs,
            "max_interactions": max_interactions,
            "experiment_name": f"{experiment_name}__{index.session_id}",
        },
        "use_cache": use_cache,
        "tool_name_separator": tool_name_separator,
        "max_interactions": max_interactions,
        "interaction_mode": interaction_mode,
    }

    if policy_override is not None:
        kwargs["policy_override"] = policy_override
    return kwargs


class AppWorldBenchmark(Benchmark, BaseModel):
    display_name: ClassVar[str] = "AppWorld"
    slug_name: ClassVar[str] = "appworld"
    available_subsets: ClassVar[list[str]] = ["train", "dev", "test_normal", "test_challenge"]
    static_action_space: ClassVar[bool] = True
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def _get_evaluator_class(cls):
        return "exgentic.benchmarks.appworld.appworld_eval:AppWorldEvaluator"

    @classmethod
    def _get_session_class(cls):
        return "exgentic.benchmarks.appworld.appworld_eval:AppWorldSession"

    # Inputs
    subset: Literal["train", "dev", "test_normal", "test_challenge"] = "test_challenge"
    env_kwargs: dict[str, Any] = Field(default_factory=dict)
    policy_override: str | None = None
    interaction_mode: Literal["code_generation", "function_calling"] = "code_generation"
    max_interactions: int = 200
    tool_name_separator: Literal[".", "__"] = "__"
    SCORES_FILE_NAME: ClassVar[str] = "scores.json"

    def runner_kwargs(self) -> dict[str, Any]:
        kwargs = super().runner_kwargs()
        if self.resolve_runner() == "venv":
            kwargs["health_timeout"] = 120.0
        return kwargs

    def list_subsets(self) -> list[str]:  # type: ignore[override]
        return list(self.available_subsets)

    def _get_evaluator_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "subset": self.subset,
            "env_kwargs": self.env_kwargs,
            "max_interactions": self.max_interactions,
            "tool_name_separator": self.tool_name_separator,
            "use_cache": self.use_cache,
            "interaction_mode": self.interaction_mode,
        }

        if self.policy_override is not None:
            kwargs["policy_override"] = self.policy_override
        return kwargs

    def get_session_kwargs(self, index: SessionIndex) -> dict[str, Any]:
        return make_session_kwargs(
            index,
            env_kwargs=self.env_kwargs,
            max_interactions=self.max_interactions,
            tool_name_separator=self.tool_name_separator,
            use_cache=self.use_cache,
            experiment_name=get_run_id(),
            policy_override=self.policy_override,
            interaction_mode=self.interaction_mode,
        )
