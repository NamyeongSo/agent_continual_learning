# SPDX-License-Identifier: Apache-2.0
"""Restore a fresh benchmark session by executing recorded prefix actions.

Unlike agents.replay.ReplaySession, this wrapper executes actions against the
real benchmark.  The agent sees observations from the restored environment.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..actions import build_unknown_action
from ..types import EmptyObservation, ParallelAction, SequentialAction


def _restore_action(payload: dict[str, Any], action_types: list, kind: str | None = None):
    available = {action.name: action for action in action_types}

    def one(item: dict[str, Any]):
        name = item.get("name")
        if name not in available:
            return build_unknown_action(name, item.get("arguments", {}), action_id=item.get("id"))
        # The original agent may have issued an invalid call. The benchmark
        # handled that call and returned an observation, so replay it with the
        # same validation outcome rather than rejecting the entire prefix.
        return available[name].build_action(item.get("arguments", {}), action_id=item.get("id"))

    if "actions" not in payload:
        return one(payload)
    actions = [one(item) for item in payload["actions"]]
    if kind == "SequentialAction":
        return SequentialAction(actions=actions)
    if kind == "ParallelAction":
        return ParallelAction(actions=actions)
    raise ValueError("Recorded multi-action prefix needs an action_class")


class PrefixReplaySession:
    """Delegate a Session after replaying the prefix in a fresh environment."""

    def __init__(self, session: Any, spec: dict[str, Any], *, static_actions: bool = False,
                 max_turns: int | None = None):
        self._session = session
        self._spec = spec
        self._resume_context: list[dict[str, Any]] = []
        self._static_actions = static_actions
        self._max_turns = max_turns
        self._action_types = None
        self._timings: dict[str, Any] = {
            "static_actions": static_actions, "prefix_actions": 0, "prefix_apply_actions_seconds": 0.0,
        }
        paths = getattr(session, "paths", None)
        self._timing_path = paths.benchmark_dir / "prefix_replay_timings.json" if paths is not None else None
        if str(session.task_id) != str(spec["task_id"]):
            raise ValueError("Replay task ID differs from the recorded task")

    def __getattr__(self, name: str):
        return getattr(self._session, name)

    @property
    def context(self) -> dict[str, Any]:
        return {**self._session.context, "premise_replay_prefix": self._resume_context}

    @property
    def actions(self):
        if not self._static_actions:
            return self._session.actions
        if self._action_types is None:
            started = time.monotonic()
            self._action_types = self._session.actions
            self._timings["action_types_transfer_seconds"] = time.monotonic() - started
        return self._action_types

    def start(self):
        started = time.monotonic()
        try:
            observation = self._session.start()
            if not self._spec.get("actions"):
                return observation
            for index, entry in enumerate(self._spec.get("actions", [])):
                if self._max_turns is not None and index >= self._max_turns:
                    break
                if self._session.done():
                    raise RuntimeError(f"Replay prefix ended the task before action {index}")
                action = _restore_action(
                    entry["action"], self.actions, entry.get("action_class")
                )
                action_started = time.monotonic()
                observation = self._session.step(action)
                self._timings["prefix_apply_actions_seconds"] += time.monotonic() - action_started
                if observation is None:
                    raise RuntimeError(f"Replay prefix action {index} returned no observation")
                payload = [item.result for item in observation.to_observation_list()]
                rendered = json.dumps(payload, ensure_ascii=False, default=str)
                self._resume_context.append({
                    "index": index,
                    "action": entry["action"],
                    "observation": rendered,
                })
                self._timings["prefix_actions"] += 1
            if self._session.done():
                raise RuntimeError("Replay prefix completed the task before continuation")
        except Exception:
            try:
                self._session.close()
            except Exception:
                pass
            raise
        finally:
            self._timings["start_and_prefix_seconds"] = time.monotonic() - started
            if self._timing_path is not None:
                path = self._timing_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self._timings, indent=2), encoding="utf-8")
        # The continuation agent did not issue the replayed action, so sending
        # its tool observation through react() would reference an unknown tool
        # call ID.  The actual prefix outputs are already in context above.
        return EmptyObservation()
