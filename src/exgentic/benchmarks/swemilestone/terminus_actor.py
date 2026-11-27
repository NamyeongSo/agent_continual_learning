# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Isolated Terminus-2 runtime attached to the existing SWE-Milestone container.

Only the isolated benchmark Python imports this module. Container creation,
network policy, submission capture, and grading remain with the official harness.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import os
import shlex
import sys
from pathlib import Path

from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import ExecResult
from harbor.llms.base import ContextLengthExceededError
from harbor.models.agent.context import AgentContext

from .transport import MilestoneBackend

HARBOR_VERSION = "0.23.0"


class MilestoneEnvironment:
    """Harbor's command interface, with ownership enforced by the remote worker."""

    default_user = "fakeroot"

    def __init__(self, request: dict):
        self.backend = MilestoneBackend(request.get("docker_host"), request.get("backend_root"))
        self.session_id = request["session_id"]
        self.command_timeout = request["command_timeout_sec"]
        self.scope = {key: request[key] for key in ("task_id", "namespace", "session_id", "pull_images")}
        self.scope["actor"] = "terminus2"

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        timeout = timeout_sec or self.command_timeout
        result = await asyncio.to_thread(
            self.backend.call,
            "execute",
            timeout=timeout + 120,
            **self.scope,
            command=command,
            command_timeout_sec=timeout,
            cwd=cwd or "/testbed",
            env=env or {},
            user=user or self.default_user,
        )
        return ExecResult(stdout=result["stdout"], stderr=result["stderr"], return_code=result["exit_code"])

    async def is_dir(self, path, user=None) -> bool:
        result = await self.exec(f"test -d {shlex.quote(str(path))}", user=user)
        return result.return_code == 0


class MilestoneTerminus2(Terminus2):
    """Use upstream overflow recovery while disabling every proactive check."""

    async def _check_proactive_summarization(self, chat, original_instruction, session):
        # Harbor 0.23.0 compares free_tokens < threshold without a zero guard.
        # Skip even when estimated free tokens are negative: wait for the error.
        return None

    async def _summarize(self, chat, original_instruction, session):
        # The estimator can undercount this model's tokens. A rejected summary
        # must make room based on the rejection, rather than repeat the estimate.
        for _attempt in range(2):
            try:
                return await super()._summarize(chat, original_instruction, session)
            except ContextLengthExceededError:
                if len(chat.messages) <= 1:
                    raise
                pairs = (len(chat.messages) - 1) // 2
                chat._messages = chat.messages[:1 + 2 * (pairs // 2)]
                chat.reset_response_chain()
                self.logger.warning("Context summary rejected; retrying with fewer history messages")
        return await super()._summarize(chat, original_instruction, session)

    async def _query_llm(self, *args, **kwargs):
        response = await super()._query_llm(*args, **kwargs)
        if response.usage is None and response.content == "Technical difficulties. Please continue with the task.":
            # Harbor 0.23.0 turns failed overflow recovery into a synthetic reply.
            # Counting it as an agent turn can consume the entire turn budget.
            raise RuntimeError("Terminus-2 context recovery failed; no model response was produced")
        return response

    async def setup(self, environment) -> None:
        suffix = hashlib.sha256(environment.session_id.encode()).hexdigest()[:16]
        self.terminal_name = f"exgentic-{suffix}"
        log_dir = f"/tmp/{self.terminal_name}"
        result = await environment.exec(f"mkdir -p {shlex.quote(log_dir)}")
        if result.return_code:
            raise RuntimeError(f"Cannot prepare Terminus terminal logs: {result.stderr}")
        self._session = TmuxSession(
            session_name=self.terminal_name,
            environment=environment,
            logging_path=Path(log_dir) / "pane.log",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
            pane_width=self.options.tmux_pane_width,
            pane_height=self.options.tmux_pane_height,
            extra_env=self._extra_env,
            user=environment.default_user,
        )
        await self._session.start()

    async def _run_agent_loop(self, *args, **kwargs):
        await super()._run_agent_loop(*args, **kwargs)
        agent_steps = [step for step in self._trajectory_steps if step.source == "agent"]
        self.completed = len(agent_steps) >= 2 and all(
            any(call.function_name == "mark_task_complete" for call in step.tool_calls or [])
            for step in agent_steps[-2:]
        )


async def run_actor(request: dict) -> dict:
    version = importlib.metadata.version("harbor")
    if version != HARBOR_VERSION:
        raise RuntimeError(f"SWE-Milestone requires harbor=={HARBOR_VERSION}; found {version}")
    environment = MilestoneEnvironment(request)
    instruction = request["instruction"]
    if request.get("playbook", "").strip():
        instruction += "\n\nACE playbook from previous milestones:\n" + request["playbook"]
    agent = MilestoneTerminus2(
        logs_dir=Path(request["logs_dir"]),
        model_name=request["model"],
        api_base=os.environ.get("OPENAI_API_BASE"),
        max_turns=request["max_turns"],
        record_terminal_session=False,
        enable_summarize=True,
        proactive_summarization_threshold=0,
        model_info=request.get("model_info"),
        llm_call_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        if "Qwen3.6" in request["model"]
        else None,
    )
    agent.completed = False
    context = AgentContext()
    try:
        async with asyncio.timeout(request["agent_timeout_sec"]):
            await agent.setup(environment)
            await agent.run(instruction, environment, context)
    finally:
        if getattr(agent, "terminal_name", None):
            cleanup = await environment.exec(f"tmux kill-session -t {shlex.quote(agent.terminal_name)}")
            if cleanup.return_code not in (0, 1):
                raise RuntimeError(f"Cannot stop Terminus terminal: {cleanup.stderr}")
    trace = "\n".join(
        json.dumps(step.model_dump(exclude_none=True, exclude={"metrics", "timestamp", "model_name"}))
        for step in agent._trajectory_steps
        if step.source == "agent"
    )
    return {
        "completed": agent.completed,
        "n_episodes": agent._n_episodes,
        "trace_path": str(Path(request["logs_dir"]) / "trajectory.json"),
        "agent_trace": trace,
        "input_tokens": context.n_input_tokens or 0,
        "output_tokens": context.n_output_tokens or 0,
        "cache_tokens": context.n_cache_tokens or 0,
        "cost_usd": context.cost_usd,
        "summarization_count": (context.metadata or {}).get("summarization_count", 0),
        "enable_summarize": True,
        "proactive_summarization_threshold": 0,
        "harbor_version": version,
    }


def main() -> None:
    request = json.load(sys.stdin)
    with os.fdopen(os.dup(1), "w") as protocol:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            try:
                response = {"result": asyncio.run(run_actor(request))}
            except Exception as exc:
                response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), file=protocol, flush=True)
        if "error" in response:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
