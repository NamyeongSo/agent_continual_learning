# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Apply the shared actor policy to Harbor without its truncation recovery loop."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from harbor.llms.base import ContextLengthExceededError, OutputLengthExceededError

from ...core.experiment_limits import ActorTurnBudget, experiment_limits
from .terminus_replay import ReplayTerminus2


class BudgetedTerminus2(ReplayTerminus2):
    def __init__(self, *args: Any, record_policy_shift: bool = False, task_clock=None, **kwargs: Any):
        self.record_policy_shift = record_policy_shift
        self._task_clock = task_clock
        limits = experiment_limits("terminalbench2")
        kwargs["max_turns"] = min(kwargs.get("max_turns") or limits["max_turns"], limits["max_turns"])
        prefix = list(kwargs.pop("replay_episodes", None) or [])
        super().__init__(*args, replay_episodes=[], **kwargs)
        # A historical prefix may already fill the new budget. Restore at most
        # that budget, then record an ordinary task failure with no fresh calls.
        self.replay_episodes = prefix[:self._max_episodes]
        self._prefix_exhausts_budget = len(prefix) >= self._max_episodes
        self._actor_budget = ActorTurnBudget({**limits, "max_turns": self._max_episodes})
        self._confirmed_complete = False
        from ...inference_profile import profile, reasoning_tokens

        if profile():
            original_usage = self._llm._extract_usage_info

            def extract_usage(response):
                self._last_reasoning_tokens = reasoning_tokens(response.usage)
                return original_usage(response)

            self._llm._extract_usage_info = extract_usage

    async def _query_or_replay(self, chat, prompt, original_instruction, session):
        if self._prefix_cursor < len(self.replay_episodes):
            return await super()._query_or_replay(chat, prompt, original_instruction, session)
        budget = self._actor_budget
        budget.begin(len(self.episodes) + 1)
        from ...inference_profile import apply_request, profile

        kwargs = apply_request(budget.request_kwargs(self._llm_call_kwargs), "actor")
        if profile() or kwargs.get("reasoning_effort") is not None:
            kwargs["allowed_openai_params"] = ["reasoning_effort"]
        messages = [*copy.deepcopy(chat.messages), {"role": "user", "content": prompt}] \
            if self.record_policy_shift else None
        started = time.monotonic()
        try:
            # Calling Chat directly bypasses Harbor's recursive output-length
            # recovery. The provider response is checked before any commands run.
            with self._task_clock.model_call() if self._task_clock is not None else nullcontext():
                response = await chat.chat(prompt, **kwargs)
        except OutputLengthExceededError:
            budget.fail("output_truncated", budget.limits["output_tokens"] or 0, 0)
        except ContextLengthExceededError:
            budget.fail("input_context_exceeded", budget.limits["input_tokens"], 0)
        self._api_request_times.append((time.monotonic() - started) * 1000)
        if profile():
            usage = response.usage.model_dump()
            usage["completion_tokens_details"] = {"reasoning_tokens": self._last_reasoning_tokens}
            budget.record(usage)
        else:
            budget.record(response.usage)
        if self.record_policy_shift:
            from ...agents.premise.span_policy_shift import HISTORY_FILE, generation_settings

            parser_path = inspect.getfile(type(self._parser))
            with (self.logs_dir / HISTORY_FILE).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "action_index": len(self.episodes), "request": {"messages": messages},
                    "message": {"role": "assistant", "content": response.content},
                    "generation_settings": generation_settings(kwargs),
                    "terminal_parser": {
                        "path": parser_path,
                        "sha256": hashlib.sha256(Path(parser_path).read_bytes()).hexdigest(),
                    },
                }, ensure_ascii=False) + "\n")
        return response

    async def _handle_llm_interaction(self, *args, **kwargs):
        result = await super()._handle_llm_interaction(*args, **kwargs)
        if result[1] and self._pending_completion and not (result[2] and "ERROR:" in result[2]):
            self._confirmed_complete = True
        return result

    async def run(self, instruction, environment, context):
        self._confirmed_complete = False
        # Retain a valid empty recording when the first request fails its budget.
        (self.logs_dir / "premise_episodes.json").write_text(json.dumps({
            "prefix_length": len(self.replay_episodes), "episodes": [],
        }), encoding="utf-8")
        await super().run(instruction, environment, context)
        if self._prefix_exhausts_budget or (self._n_episodes >= self._max_episodes and not self._confirmed_complete):
            self._actor_budget.begin(self._max_episodes + 1)
