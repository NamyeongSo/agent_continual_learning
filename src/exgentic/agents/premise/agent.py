# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from typing import Any, ClassVar

from ...core.types import ModelSettings
from ..litellm_tool_calling.litellm_tool_calling_agent import LiteLLMToolCallingAgent


class PremiseAgent(LiteLLMToolCallingAgent):
    """An acting agent with a frozen playbook for one session."""

    display_name: ClassVar[str] = "Premise Pareto Agent"
    slug_name: ClassVar[str] = "premise"
    playbook: str = ""
    enable_thinking: bool = False
    model_settings: ModelSettings | None = None

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
            "enable_thinking": self.enable_thinking,
        }
