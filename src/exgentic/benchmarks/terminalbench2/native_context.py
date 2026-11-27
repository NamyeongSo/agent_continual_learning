# SPDX-License-Identifier: Apache-2.0
"""Native Terminus-2 with serving-token accounting and bounded context recovery."""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.request

from harbor.agents.terminus_2 import Terminus2
from harbor.llms.base import ContextLengthExceededError

SYNTHETIC_FAILURE = "Technical difficulties. Please continue with the task."
RECOVERY_VERSION = "terminus-native-context-v1"


class ContextRecoveryError(RuntimeError):
    """Context management failed without producing an executable model response."""


class ServingTokenCounter:
    """Use the serving model's actual chat template, including preserved reasoning."""

    def __init__(self, api_base, model, template_kwargs, api_key=None):
        self.url = api_base.removesuffix("/").removesuffix("/v1") + "/tokenize"
        self.model = model.removeprefix("openai/")
        self.template_kwargs = dict(template_kwargs)
        self.api_key = api_key
        self.cache = {}

    def __call__(self, messages):
        if not messages:
            return 0
        # ChatCompletionRequest normalizes the legacy field before rendering;
        # TokenizeChatRequest does not. Mirror that normalization on a copy so
        # /tokenize sees the same retained reasoning as the generation endpoint.
        normalized = []
        for message in messages:
            message = dict(message)
            if message.get("reasoning") is None and message.get("reasoning_content") is not None:
                message["reasoning"] = message["reasoning_content"]
            normalized.append(message)
        payload = {"model": self.model, "messages": normalized, "add_generation_prompt": True,
                   "chat_template_kwargs": self.template_kwargs}
        body = json.dumps(payload, ensure_ascii=False).encode()
        key = hashlib.sha256(body).digest()
        if key in self.cache:
            return self.cache[key]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            count = result["count"]
            if not isinstance(count, int) or count < 0:
                raise ValueError("Invalid token count")
        except Exception as exc:
            raise ContextRecoveryError(f"Serving tokenizer failed: {type(exc).__name__}") from exc
        if len(self.cache) >= 8:
            self.cache.clear()
        self.cache[key] = count
        return count


class RecoveringTerminus2(Terminus2):
    """Retain the upstream prompt/parser/loop and repair only context management."""

    def __init__(self, *args, serving_token_counter, **kwargs):
        super().__init__(*args, **kwargs)
        self._serving_token_counter = serving_token_counter
        self._query_depth = 0

    def _event(self, event, **fields):
        with (self.logs_dir / "context_recovery.jsonl").open("a") as stream:
            stream.write(json.dumps({"version": RECOVERY_VERSION, "event": event,
                                     "timestamp": time.time(), **fields}) + "\n")

    def _count_total_tokens(self, chat):
        return self._serving_token_counter(chat.messages)

    def _shrink(self, chat, *, reserve, force=False):
        limit = self._llm.get_model_context_limit()
        target = max(1, limit - reserve)
        for _round in range(12):
            count = self._count_total_tokens(chat)
            if count <= target and not force:
                return
            # Keep the original task and the newest complete exchange. Remove
            # whole old exchanges; never edit the retained reasoning blocks.
            available = 2 * ((len(chat.messages) - 3) // 2)
            if available < 2:
                raise ContextRecoveryError("Task and latest exchange cannot fit in the serving context")
            fraction = max(.25 if force else 0, (count - target) / max(1, count))
            remove = min(available, max(2, 2 * math.ceil((len(chat.messages) - 1) * fraction / 2)))
            chat._messages = [chat.messages[0], *chat.messages[1 + remove:]]
            chat.reset_response_chain()
            self._event("history_reduced", old_tokens=count, removed_messages=remove,
                        retained_messages=len(chat.messages))
            force = False
        raise ContextRecoveryError("Context shrinking did not converge")

    def _unwind_messages_to_free_tokens(self, chat, target_free_tokens=4000):
        self._shrink(chat, reserve=max(8000, target_free_tokens))

    async def _summarize(self, chat, original_instruction, session):
        last_error = None
        for attempt in range(3):
            self._shrink(chat, reserve=12000, force=attempt > 0)
            try:
                result = await super()._summarize(chat, original_instruction, session)
                self._event("summary_completed", attempt=attempt + 1,
                            retained_tokens=self._count_total_tokens(chat))
                return result
            except ContextLengthExceededError as exc:
                last_error = exc
                self._event("summary_overflow", attempt=attempt + 1)
            except Exception as exc:
                raise ContextRecoveryError(f"Native summarization failed: {type(exc).__name__}") from exc
        raise ContextRecoveryError("Native summary rejected after three shrinking attempts") from last_error

    async def _check_proactive_summarization(self, chat, original_instruction, session):
        if not self.options.enable_summarize:
            return None
        free = self._llm.get_model_context_limit() - self._count_total_tokens(chat)
        if free < self.options.proactive_summarization_threshold:
            # Unlike upstream, do not swallow a failed summary and immediately
            # send the same overlong history again.
            return await self._summarize(chat, original_instruction, session)
        return None

    async def _query_llm(self, chat, prompt, original_instruction="", session=None):
        self._query_depth += 1
        try:
            if self._query_depth > 3:
                raise ContextRecoveryError("Repeated output/context recovery exceeded three levels")
            count = self._serving_token_counter([*chat.messages, {"role": "user", "content": prompt}])
            if count >= self._llm.get_model_context_limit() - 8000:
                if not self.options.enable_summarize or session is None:
                    raise ContextRecoveryError("Request exceeds context and summary is unavailable")
                handoff, refs = await self._summarize(chat, original_instruction, session)
                self._pending_subagent_refs = refs
                self._pending_handoff_prompt = handoff
                prompt = handoff + "\n\nLatest terminal feedback:\n" + prompt
                count = self._serving_token_counter([*chat.messages, {"role": "user", "content": prompt}])
                if count >= self._llm.get_model_context_limit() - 4000:
                    raise ContextRecoveryError("Recovered request still exceeds serving context")
            response = await super()._query_llm(chat, prompt, original_instruction, session)
            if response.usage is None and response.content == SYNTHETIC_FAILURE:
                self._event("synthetic_failure_blocked")
                raise ContextRecoveryError("Native recovery failed; refusing a synthetic agent response")
            return response
        finally:
            self._query_depth -= 1
