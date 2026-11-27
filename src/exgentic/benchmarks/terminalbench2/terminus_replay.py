# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Record and restore Terminus-2 episodes using its original parser and tmux loop.

Imported only inside the Harbor environment. Prefix responses bypass the model,
but their commands execute normally in a fresh persistent terminal. Chat history
therefore contains fresh observations, and completion/parser state is restored by
the same agent loop used for training. An action boundary is one command batch.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2 import Terminus2
from harbor.llms.base import LLMResponse


def load_recorded_prefix(path: str | None, count: int) -> list[dict[str, Any]]:
    if path is None:
        if count:
            raise ValueError("Replay count requires an episode file")
        return []
    if count < 0:
        raise ValueError("Replay count must be nonnegative")
    recorded = json.loads(Path(path).read_text(encoding="utf-8"))
    if recorded.get("prefix_length") != 0:
        raise ValueError("Replay source must be an original Terminus-2 trajectory")
    episodes = recorded["episodes"]
    if count > len(episodes):
        raise ValueError("Replay prefix exceeds recorded Terminus-2 episodes")
    return episodes[:count]


class ReplayTerminus2(Terminus2):
    def __init__(self, *args: Any, replay_episodes: list[dict[str, Any]] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.replay_episodes = list(replay_episodes or [])
        if len(self.replay_episodes) >= self._max_episodes:
            raise ValueError("Replay prefix consumes the Terminus-2 turn budget")
        self.episodes: list[dict[str, Any]] = []
        self._prefix_cursor = 0
        self._query_depth = 0

    async def _query_llm(self, chat, prompt, original_instruction="", session=None):
        depth = self._query_depth
        self._query_depth += 1
        call_id = uuid.uuid4().hex
        replay = self._prefix_cursor < len(self.replay_episodes)
        started = time.monotonic()

        def record_event(event: str, **fields):
            with (self.logs_dir / "query_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "call_id": call_id, "event": event, "depth": depth, "replay": replay,
                    "timestamp": datetime.now(UTC).isoformat(), **fields,
                }) + "\n")

        try:
            record_event("start")
            response = await self._query_or_replay(chat, prompt, original_instruction, session)
        except BaseException as exc:
            record_event("error", elapsed_seconds=time.monotonic() - started, error_type=type(exc).__name__)
            raise
        else:
            record_event("end", elapsed_seconds=time.monotonic() - started)
        finally:
            self._query_depth -= 1

        # Harbor can recursively call self._query_llm to recover a truncated
        # response. Only the outermost return becomes an executed command batch.
        if depth == 0:
            self.episodes.append({"response": response.content})
            (self.logs_dir / "premise_episodes.json").write_text(
                json.dumps({"prefix_length": len(self.replay_episodes), "episodes": self.episodes},
                           ensure_ascii=False, indent=2), encoding="utf-8",
            )
        return response

    async def _query_or_replay(self, chat, prompt, original_instruction, session):
        if self._prefix_cursor < len(self.replay_episodes):
            record = self.replay_episodes[self._prefix_cursor]
            self._prefix_cursor += 1
            response = LLMResponse(content=record["response"], model_name=self._model_name)
            # Use the candidate instruction and the current terminal observation.
            chat.messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.content},
            ])
            chat.reset_response_chain()
        else:
            response = await super()._query_llm(chat, prompt, original_instruction, session)
        return response

    async def run(self, instruction, environment, context):
        await super().run(instruction, environment, context)
        if self._prefix_cursor != len(self.replay_episodes):
            raise ValueError("Terminus-2 ended before the replay prefix was restored")
