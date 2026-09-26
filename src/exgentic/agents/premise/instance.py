# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
from typing import Any

from litellm import ChatCompletionSystemMessage, ChatCompletionUserMessage

from ..litellm_tool_calling.instance import LiteLLMToolCallingAgentInstance
from .llm import shared_llm_slots


class PremiseAgentInstance(LiteLLMToolCallingAgentInstance):
    def __init__(self, *args: Any, playbook: str = "", enable_thinking: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.playbook = playbook
        self.enable_thinking = enable_thinking

    def start(self, task, context, actions):
        clean_context = dict(context or {})
        prefix = clean_context.pop("premise_replay_prefix", None)
        super().start(task, clean_context, actions)
        self.messages.insert(0, ChatCompletionSystemMessage(
            role="system",
            content=(
                "Complete the user's task using the available tools. Use relevant "
                "playbook entries as procedural guidance; the task and live tool "
                "outputs remain authoritative. Never seek private grading data.\n\n"
                f"PLAYBOOK:\n{self.playbook or '(empty)'}"
            ),
        ))
        if prefix:
            self._add_message(ChatCompletionUserMessage(
                role="user",
                content=(
                    "The environment has been restored by executing the following "
                    "recorded actions in a fresh session. Continue from this state. "
                    "Do not repeat completed setup unless a check is needed.\n"
                    + json.dumps(prefix, ensure_ascii=False, default=str)
                ),
            ))

    def _completion(self, **kwargs):
        body = dict(kwargs.pop("extra_body", {}) or {})
        body["chat_template_kwargs"] = {
            **(body.get("chat_template_kwargs") or {}),
            "enable_thinking": self.enable_thinking,
        }
        with shared_llm_slots():
            return super()._completion(extra_body=body, **kwargs)
