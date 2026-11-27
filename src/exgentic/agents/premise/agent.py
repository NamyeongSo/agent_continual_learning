# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from typing import Any, ClassVar

from ...actor_context import ActorContextMode
from ...core.types import ModelSettings
from ..litellm_tool_calling.litellm_tool_calling_agent import LiteLLMToolCallingAgent


class PremiseAgent(LiteLLMToolCallingAgent):
    """An acting agent with a frozen playbook for one session."""

    display_name: ClassVar[str] = "Premise Pareto Agent"
    slug_name: ClassVar[str] = "premise"
    playbook: str = ""
    actor_context: ActorContextMode = "legacy"
    actor_benchmark: str = ""
    system_prompt_override: str | None = None
    context_policy_override: str | None = None
    enable_thinking: bool = False
    allow_shared_thinking: bool = False
    native_model_settings: dict[str, Any] | None = None
    terminus_replay_path: str | None = None
    terminus_replay_count: int = 0
    model_settings: ModelSettings | None = None
    temperature_override: float | None = None
    record_policy_shift: bool = False

    @classmethod
    def _get_instance_class(cls):
        from .instance import PremiseAgentInstance
        return PremiseAgentInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "exgentic.agents.premise.instance:PremiseAgentInstance"

    def _get_instance_kwargs(self, session_id: str) -> dict[str, Any]:
        return {
            **super()._get_instance_kwargs(session_id),
            "playbook": self.playbook,
            "actor_context": self.actor_context,
            "actor_benchmark": self.actor_benchmark,
            "system_prompt_override": self.system_prompt_override,
            "context_policy_override": self.context_policy_override,
            "enable_thinking": self.enable_thinking,
            "allow_shared_thinking": self.allow_shared_thinking,
            "native_model_settings": self.native_model_settings,
            "terminus_replay_path": self.terminus_replay_path,
            "terminus_replay_count": self.terminus_replay_count,
            "temperature_override": self.temperature_override,
            "record_policy_shift": self.record_policy_shift,
        }
