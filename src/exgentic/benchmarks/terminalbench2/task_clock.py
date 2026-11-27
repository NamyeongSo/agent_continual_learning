# SPDX-License-Identifier: Apache-2.0
"""Account for task execution independently of provider request latency."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def initialize_model_call_ledger(path):
    save(path, {"enabled": True, "completed_seconds": 0, "active_started": None, "calls": 0})


def read_ledger(path):
    return json.loads(Path(path).read_text())


@contextmanager
def record_model_call(path):
    """Share the interactive actor's request interval with its isolated benchmark."""
    data = read_ledger(path)
    started = time.monotonic()
    data.update(active_started=started, calls=data["calls"] + 1)
    save(path, data)
    try:
        yield
    finally:
        data.update(completed_seconds=data["completed_seconds"] + time.monotonic() - started, active_started=None)
        save(path, data)


class AgentTimeLimitExceededError(TimeoutError):
    """The non-model portion of the agent run consumed its time budget."""


class AgentTimeBudget:
    def __init__(self, limit_seconds, *, report_path=None, ledger_path=None, now=None):
        self.limit = limit_seconds
        self.report_path = report_path
        self.ledger_path = ledger_path
        self._now = now or time.monotonic
        self.started = self._now()
        self.stopped = None
        self._excluded = 0.0
        self._paused_since = None
        self._depth = 0
        self._calls = 0
        self._timeout = None
        self._loop = None

    def snapshot(self):
        now = self.stopped if self.stopped is not None else self._now()
        excluded = self._excluded
        calls = self._calls
        active = self._paused_since is not None
        if active:
            excluded += now - self._paused_since
        if self.ledger_path:
            ledger = read_ledger(self.ledger_path)
            excluded += ledger["completed_seconds"]
            calls += ledger["calls"]
            if ledger["active_started"] is not None:
                excluded += now - ledger["active_started"]
                active = True
        wall = max(0.0, now - self.started)
        charged = max(0.0, wall - excluded)
        return {"mode": "agent-active-time-v1", "limit_seconds": self.limit,
                "wall_seconds": wall, "model_call_seconds": excluded, "charged_seconds": charged,
                "remaining_seconds": max(0.0, self.limit - charged) if self.limit is not None else None,
                "model_call_count": calls, "model_call_active": active}

    def remaining(self):
        return self.snapshot()["remaining_seconds"]

    def _publish(self):
        if self.report_path:
            save(self.report_path, self.snapshot())

    def _reschedule(self):
        if self._timeout is not None and not self._timeout.expired():
            remaining = self.remaining()
            self._timeout.reschedule(
                None if self._depth or remaining is None else self._loop.time() + remaining,
            )
        self._publish()

    @contextmanager
    def model_call(self):
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            raise AgentTimeLimitExceededError("Terminal-Bench non-model agent time limit reached")
        if not self._depth:
            self._paused_since = self._now()
            self._calls += 1
        self._depth += 1
        self._reschedule()
        try:
            yield
        finally:
            self._depth -= 1
            if not self._depth:
                self._excluded += self._now() - self._paused_since
                self._paused_since = None
            self._reschedule()

    async def run(self, awaitable):
        """Cancel on charged time only, including while an environment command runs."""
        timeout = asyncio.timeout(None)
        try:
            async with timeout:
                self._timeout = timeout
                self._loop = asyncio.get_running_loop()
                self._reschedule()
                return await awaitable
        except TimeoutError as exc:
            if timeout.expired():
                raise AgentTimeLimitExceededError("Terminal-Bench non-model agent time limit reached") from exc
            raise
        finally:
            self._timeout = self._loop = None

    def stop(self):
        if self.stopped is None:
            self.stopped = self._now()
        self._publish()
        return self.snapshot()
