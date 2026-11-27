# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any

from litellm import ChatCompletionSystemMessage, ChatCompletionUserMessage

from ...actor_context import ActorContextMode, shared_actor_prompt, validate_actor_context
from ...core.experiment_limits import ExperimentBudgetExceededError, actor_completion
from ...core.types import RetryStrategy
from ...utils.context_errors import is_context_length_error
from ..litellm_tool_calling.instance import (
    CompletionRetriesExhaustedError,
    LiteLLMToolCallingAgentInstance,
    NonRetryableCompletionError,
    OutputTruncatedCompletionError,
)
from .llm import shared_llm_slots


class PremiseAgentInstance(LiteLLMToolCallingAgentInstance):
    actor_context: ActorContextMode = "legacy"
    actor_benchmark: str = ""
    output_truncation_retries = 5

    def __init__(self, *args: Any, playbook: str = "", enable_thinking: bool = False,
                 terminus_replay_path: str | None = None, terminus_replay_count: int = 0,
                 context_policy_override: str | None = None,
                 system_prompt_override: str | None = None,
                 actor_context: ActorContextMode = "legacy", actor_benchmark: str = "",
                 record_policy_shift: bool = False,
                 allow_shared_thinking: bool = False,
                 native_model_settings: dict[str, Any] | None = None,
                 temperature_override: float | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        validate_actor_context(actor_context)
        if actor_context != "legacy" and enable_thinking and not allow_shared_thinking:
            raise ValueError("Shared actor context requires enable_thinking=false")
        if actor_context != "legacy" and system_prompt_override is not None:
            raise ValueError("Shared actor context cannot be combined with system_prompt_override")
        self.actor_context, self.actor_benchmark = actor_context, actor_benchmark
        if actor_context != "legacy":
            self.enable_tool_shortlisting = False
        self.playbook = playbook
        self.temperature_override = temperature_override
        self.context_policy_override = context_policy_override
        self.system_prompt_override = system_prompt_override
        self.record_policy_shift = record_policy_shift
        self.enable_thinking = enable_thinking
        self.native_model_settings = native_model_settings
        self.terminus_replay_path = terminus_replay_path
        self.terminus_replay_count = terminus_replay_count
        self._terminus_started = False

    def _render_history_for_shortlist(self) -> str:
        # Match ACE's tool-selection window; the acting model still receives
        # the complete conversation through self.messages.
        return super()._render_history_for_shortlist()[-8000:]

    def start(self, task, context, actions):
        from ..ace.prompts.generator import build_generator_system_prompt

        clean_context = dict(context or {})
        prefix = clean_context.pop("premise_replay_prefix", None)
        if self.context_policy_override is not None:
            policy = self.context_policy_override
            if "{allowed}" in policy:
                match = re.search(r"The applications include:\s*([^\n]+?)\.(?:\s|$)", clean_context.get("policy", ""))
                if match is None:
                    raise ValueError("Policy override requires the task's actual allowed applications")
                policy = policy.replace("{allowed}", match.group(1))
            clean_context["policy"] = policy
        super().start(task, clean_context, actions)
        self.messages.insert(0, ChatCompletionSystemMessage(
            role="system",
            content=(
                shared_actor_prompt(self.actor_benchmark, self.playbook) if self.actor_context != "legacy" else
                getattr(self, "system_prompt_override", None) or build_generator_system_prompt(self.playbook)
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

    def react(self, observation):
        terminus = next((action for action in self.actions if action.name == "run_terminus2"), None)
        if terminus is not None:
            if self._terminus_started:
                return None
            self._terminus_started = True
            return terminus.build_action({
                "model": self.model, "playbook": self.playbook,
                **({"actor_context": self.actor_context} if self.actor_context != "legacy" else {}),
                "record_episodes": True, "replay_episodes_path": self.terminus_replay_path,
                "replay_episodes_count": self.terminus_replay_count,
                **({"record_policy_shift": True} if getattr(self, "record_policy_shift", False) else {}),
                **({"temperature": self.temperature_override}
                   if getattr(self, "temperature_override", None) is not None else {}),
                **({"model_settings": self.native_model_settings}
                   if getattr(self, "native_model_settings", None) is not None else {}),
            })
        return super().react(observation)

    def _completion(self, **kwargs):
        # Only the acting call uses this exact conversation object. Tool-selection
        # calls use their own messages and must not become span action anchors.
        record = getattr(self, "record_policy_shift", False) and kwargs.get("messages") is self.messages
        request = (
            copy.deepcopy({k: kwargs[k] for k in ("messages", "tools", "tool_choice") if k in kwargs})
            if record else None
        )
        if getattr(self, "temperature_override", None) is not None:
            kwargs["temperature"] = self.temperature_override
        if self.model_settings.reasoning_effort is not None:
            # Self-hosted model IDs may not be in LiteLLM's reasoning-model map.
            # Without this allowlist, drop_params=True silently removes the effort.
            allowed = list(kwargs.pop("allowed_openai_params", None) or [])
            if "reasoning_effort" not in allowed:
                allowed.append("reasoning_effort")
            kwargs["allowed_openai_params"] = allowed
        body = dict(kwargs.pop("extra_body", {}) or {})
        body["chat_template_kwargs"] = {
            **(body.get("chat_template_kwargs") or {}),
            "enable_thinking": self.enable_thinking,
        }
        with shared_llm_slots():
            response = super()._completion(extra_body=body, **kwargs)
        if record:
            from .span_policy_shift import HISTORY_FILE

            path = self.paths.agent_dir / HISTORY_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "action_index": self._step_count - 1, "request": request,
                    "message": response.model_dump(mode="json")["choices"][0]["message"],
                    "generation_settings": getattr(self, "_recorded_generation_settings", {}),
                }, ensure_ascii=False) + "\n")
        return response

    def _completion_with_retries(self, call_kwargs):
        def complete(**kwargs):
            import litellm

            from .span_policy_shift import generation_settings

            self._recorded_generation_settings = generation_settings(kwargs)
            return litellm.completion(**kwargs)

        retries = self.model_settings.num_retries or 0
        request_failures = truncations = 0
        for attempt in range(retries + self.output_truncation_retries + 1):
            started = time.monotonic()
            try:
                response = actor_completion(
                    self, {**call_kwargs, "max_retries": 0},
                    completion_fn=complete if getattr(self, "record_policy_shift", False) else None,
                )
                self._raise_if_invalid_completion(response)
                return response
            except OutputTruncatedCompletionError:
                if truncations >= self.output_truncation_retries:
                    raise
                truncations += 1
                self.logger.warning(
                    "PREMiSE output token limit reached; resending unchanged request (retry %d/%d)",
                    truncations, self.output_truncation_retries,
                )
            except (NonRetryableCompletionError, ExperimentBudgetExceededError):
                raise
            except Exception as exc:
                if is_context_length_error(exc):
                    raise
                self.logger.warning(
                    "PREMiSE LLM request failed after %.3fs (attempt %d/%d): %s: %s",
                    time.monotonic() - started, request_failures + 1, retries + 1, type(exc).__name__, exc,
                )
                if request_failures >= retries:
                    if isinstance(exc, ValueError):
                        raise
                    # The session runner serializes exceptions. Keep this marker
                    # in the message so replay orchestration can honor the budget.
                    raise CompletionRetriesExhaustedError(
                        f"CompletionRetriesExhaustedError: {attempt + 1} attempts; {type(exc).__name__}: {exc}"
                    ) from exc
                delay = self.model_settings.retry_after
                if self.model_settings.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                    delay *= 2**request_failures
                request_failures += 1
                if delay > 0:
                    time.sleep(delay)
        raise RuntimeError("Completion retry loop exhausted")
