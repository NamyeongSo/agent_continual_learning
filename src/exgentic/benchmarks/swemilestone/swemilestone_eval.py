# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""One development stream per repository, with one Exgentic session per milestone."""

from __future__ import annotations

import json
from typing import Any

from ...core.actions import ActionsHandler
from ...core.evaluator import Evaluator
from ...core.session import Session
from ...core.types import Action, ActionType, BenchmarkResults, SessionIndex, SessionScore, SingleObservation
from ...utils.paths import get_run_paths
from .swemilestone_benchmark import (
    BENCHMARK_VERSION,
    DATA_COMMIT,
    HARNESS_COMMIT,
    BashAction,
    RunTerminus2Action,
    SubmitMilestoneAction,
)
from .terminus import run_terminus
from .transport import MilestoneBackend


def truncate_output(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = limit // 2
    tail = limit - head
    return value[:head] + f"\n... {len(value) - limit} characters omitted ...\n" + value[-tail:]


class SWEMilestoneSession(Session):
    """A fresh agent session operating on the repository's continuing container."""

    def __init__(
        self,
        task_id: str,
        srs: str,
        graded: bool,
        prerequisites: list[str],
        namespace: str,
        session_id: str,
        docker_host: str | None = None,
        backend_root: str | None = None,
        start_snapshot: dict[str, Any] | None = None,
        max_interactions: int | None = 200,
        command_timeout_sec: int = 900,
        evaluation_timeout_sec: int = 10800,
        observation_size_limit: int = 10000,
        pull_images: bool = True,
        actor: str = "interactive",
        terminus_python: str | None = None,
        agent_timeout_sec: int = 10800,
        model_info: dict[str, Any] | None = None,
    ) -> None:
        if actor not in ("interactive", "terminus2"):
            raise ValueError(f"Unknown SWE-Milestone actor: {actor}")
        self._task_id = task_id
        self._srs = srs
        self._graded = graded
        self._prerequisites = prerequisites
        self._session_id = session_id
        self._namespace = namespace
        self._backend = MilestoneBackend(docker_host, backend_root)
        self._snapshot = start_snapshot
        self._max_interactions = max_interactions
        self._command_timeout = command_timeout_sec
        self._evaluation_timeout = evaluation_timeout_sec
        self._output_limit = observation_size_limit
        self._pull_images = pull_images
        self._actor = actor
        self._terminus_python = terminus_python
        self._agent_timeout = agent_timeout_sec
        self._model_info = model_info
        self._terminus_result: dict[str, Any] | None = None
        self._started = False
        self._done = False
        self._action_count = 0
        self._score: SessionScore | None = None
        self._handler = ActionsHandler()
        if actor == "terminus2":
            self._handler.add_action(
                "run_terminus2",
                "Develop this milestone with Terminus-2, recovering context only after an overflow error. "
                "Call finish after this action to submit a confirmed completion.",
                RunTerminus2Action,
                self._run_terminus2,
            )
        else:
            self._handler.add_action(
                "bash", "Execute a Bash command in the persistent development repository.", BashAction, self._bash
            )
        self._handler.add_action(
            "finish",
            "Commit and submit this milestone. Its source snapshot becomes immutable and the session ends.",
            SubmitMilestoneAction,
            self._finish,
            is_finish=True,
        )
        super().__init__()

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def task(self) -> str:
        if self._actor == "terminus2":
            return (
                "Implement this SWE-Milestone specification using run_terminus2. "
                "Pass your model and current playbook, then call finish. "
                "Only Terminus-2's confirmed completion can be submitted; a turn limit does not submit partial work. "
                "Code persists from earlier milestones; the conversation starts fresh.\n\n" + self._srs
            )
        return (
            "Implement the following SWE-Milestone specification in the current repository. "
            "Code and dependency caches persist from earlier milestones; your conversation starts fresh. "
            "Use bash to inspect, edit, and test the code. Each command starts in the repository root. "
            "Call finish when ready; it commits your changes and creates the required submission tag. "
            "Official hidden tests run after submission.\n\n" + self._srs
        )

    @property
    def context(self) -> dict[str, Any]:
        repo, milestone = self._task_id.split("/", 1)
        return {
            "task_id": self._task_id,
            "repository": repo,
            "milestone_id": milestone,
            "graded": self._graded,
            "prerequisites": self._prerequisites,
            "persistent_repository": True,
            "actor": self._actor,
        }

    @property
    def actions(self) -> list[ActionType]:
        return self._handler.actions

    def get_config(self) -> dict[str, Any]:
        return {
            "task_id": self._task_id,
            "namespace": self._namespace,
            "graded": self._graded,
            "benchmark_version": BENCHMARK_VERSION,
            "harness_commit": HARNESS_COMMIT,
            "data_commit": DATA_COMMIT,
            "actor": self._actor,
        }

    def _call(self, operation: str, *, timeout: int = 3600, **kwargs):
        return self._backend.call(
            operation,
            timeout=timeout,
            task_id=self._task_id,
            namespace=self._namespace,
            session_id=self.session_id,
            pull_images=self._pull_images,
            actor=self._actor,
            **kwargs,
        )

    def start(self) -> SingleObservation | None:
        result = self._call("start", timeout=10800, start_snapshot=self._snapshot)
        self._started = True
        self._done = result["submitted"]
        self._snapshot = result["start_snapshot"]
        # Local artifact is sufficient to branch a PREMiSE/Pareto candidate on the same Docker host.
        path = self.paths.benchmark_dir / "start_snapshot.json"
        path.write_text(json.dumps(self._snapshot, indent=2), encoding="utf-8")
        if self._done:
            return None
        return SingleObservation(invoking_actions=[], result="Development repository ready for this milestone.")

    def _bash(self, action: BashAction) -> str:
        if self._done:
            return "This milestone session is finished."
        if not self._started:
            raise RuntimeError("Milestone session has not started")
        result = self._call(
            "execute",
            timeout=self._command_timeout + 120,
            command=action.arguments.command,
            command_timeout_sec=self._command_timeout,
        )
        rendered = f"exit_code: {result['exit_code']}\nstdout:\n{result['stdout']}\nstderr:\n{result['stderr']}"
        return truncate_output(rendered, self._output_limit)

    def _run_terminus2(self, action: RunTerminus2Action) -> dict:
        if not self._started:
            raise RuntimeError("Milestone session has not started")
        if self._terminus_result is not None:
            raise RuntimeError("Terminus-2 already ran for this milestone; call finish")
        instruction = (
            "Implement the following SWE-Milestone specification in /testbed. "
            "Code and dependency caches persist from earlier milestones. "
            "Inspect, edit, and test using the terminal. Set task_complete only when this milestone is ready. "
            "The external benchmark commits and submits after you confirm completion; do not create submission tags. "
            "Official hidden tests run after submission.\n\n" + self._srs
        )
        self._terminus_result = run_terminus(
            {
                "model": action.arguments.model,
                "playbook": action.arguments.playbook,
                "instruction": instruction,
                "logs_dir": str(self.paths.benchmark_dir / "terminus"),
                "docker_host": self._backend.docker_host,
                "backend_root": self._backend.backend_root,
                "task_id": self._task_id,
                "namespace": self._namespace,
                "session_id": self.session_id,
                "pull_images": self._pull_images,
                "command_timeout_sec": self._command_timeout,
                "agent_timeout_sec": self._agent_timeout,
                "max_turns": self._max_interactions or 1000000,
                "model_info": self._model_info,
            },
            python=self._terminus_python,
        )
        self._action_count = self._terminus_result["n_episodes"]
        return self._terminus_result

    def _finish(self, action: SubmitMilestoneAction) -> None:
        if not self._done:
            if self._actor == "terminus2":
                if self._terminus_result is None:
                    raise RuntimeError("Run Terminus-2 before finishing this milestone")
                if not self._terminus_result["completed"]:
                    self._done = True
                    return
            self._call("submit", timeout=10800)
            self._done = True

    def step(self, action: Action):
        if self._done:
            return None
        if self._actor == "terminus2":
            actions = action.to_action_list() if action is not None else []
            if len(actions) != 1:
                raise ValueError("Terminus-2 requires one run_terminus2 or finish action")
            single = actions[0]
            # Infrastructure errors must reach the orchestrator, rather than
            # becoming ordinary tool observations through ActionsHandler.
            if isinstance(single, RunTerminus2Action):
                result = self._run_terminus2(single)
                return SingleObservation(invoking_actions=[single], result=result)
            if isinstance(single, SubmitMilestoneAction):
                self._finish(single)
                return None
            raise ValueError("Terminus-2 exposes run_terminus2 and finish")
        observation = self._handler.execute(action)
        self._action_count += len(action.to_action_list())
        if self._max_interactions is not None and self._action_count >= self._max_interactions:
            # A limit does not submit partial work; close() rolls back this milestone.
            self._done = True
        return None if self._done else observation

    def done(self) -> bool:
        return self._done

    def score(self) -> SessionScore:
        if self._score is not None:
            return self._score
        if not self._started:
            raise RuntimeError("Cannot score a milestone before starting its development stream")
        result = self._call("evaluate", timeout=self._evaluation_timeout)
        metadata = {
            **self.get_config(),
            "submitted": result["submitted"],
            "resolved": result["resolved"],
            "start_snapshot": self._snapshot,
            "actions": self._action_count,
            "remote_evaluation_dir": result.get("remote_evaluation_dir"),
            "training_feedback_available": self._graded and result["submitted"],
            "evaluation_error": result.get("evaluation_error"),
        }
        if self._terminus_result is not None:
            metadata.update(
                agent_trace_path=self._terminus_result["trace_path"],
                agent_usage={
                    key: self._terminus_result.get(key)
                    for key in ("input_tokens", "output_tokens", "cache_tokens", "cost_usd")
                },
                terminus_completed=self._terminus_result["completed"],
                summarization_count=self._terminus_result["summarization_count"],
                context_recovery="on_overflow",
                proactive_summarization_threshold=0,
                harbor_version=self._terminus_result["harbor_version"],
            )
        self._score = SessionScore(
            score=float(result["score"]),
            success=True,
            is_finished=result["submitted"],
            session_metrics={
                "resolved": result["resolved"],
                "graded": self._graded,
                "precision": result.get("precision", 0.0),
                "recall": result.get("recall", 0.0),
                "build_failure": result.get("build_failure", False),
            },
            session_metadata=metadata,
        )
        if "evaluation" in result:
            (self.paths.benchmark_dir / "evaluation_result.json").write_text(
                json.dumps(result["evaluation"], indent=2), encoding="utf-8"
            )
        self.save_results(self._score.model_dump())
        return self._score

    def close(self) -> None:
        if self._started:
            self._call("close", timeout=120)
            self._started = False


class SWEMilestoneEvaluator(Evaluator):
    def __init__(
        self,
        subset: str = "ripgrep",
        docker_host: str | None = None,
        backend_root: str | None = None,
        stream_id: str | None = None,
        start_snapshot: dict[str, Any] | None = None,
        **session_options,
    ) -> None:
        self._backend = MilestoneBackend(docker_host, backend_root)
        self._rows = self._backend.call("catalog", subset=subset, timeout=120)
        self._by_id = {row["task_id"]: row for row in self._rows}
        self._stream_id = stream_id
        self._options = {
            "docker_host": docker_host,
            "backend_root": backend_root,
            "start_snapshot": start_snapshot,
            **session_options,
        }

    def list_tasks(self) -> list[str]:
        return list(self._by_id)

    def get_session_kwargs(self, index: SessionIndex) -> dict[str, Any]:
        row = self._by_id[index.task_id]
        return {
            **self._options,
            "task_id": index.task_id,
            "srs": row["srs"],
            "graded": row["graded"],
            "prerequisites": row["prerequisites"],
            "session_id": index.session_id,
            "namespace": self._backend.namespace(str(get_run_paths().root.resolve()), self._stream_id),
        }

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        results = {}
        for index, path in zip(sessions, self.get_sessions_paths(sessions), strict=True):
            if index.task_id not in self._by_id:
                raise KeyError(f"Unknown SWE-Milestone task: {index.task_id}")
            if index.task_id in results:
                raise ValueError(
                    f"Duplicate milestone result: {index.task_id}; aggregate candidates in separate streams"
                )
            if path.benchmark_results.exists():
                results[index.task_id] = json.loads(path.benchmark_results.read_text(encoding="utf-8"))
        repositories = {}
        for alias in dict.fromkeys(row["repo_alias"] for row in self._rows):
            graded = [row for row in self._rows if row["repo_alias"] == alias and row["graded"]]
            payloads = [results.get(row["task_id"], {}) for row in graded]
            repositories[alias] = {
                "graded_milestones": len(graded),
                "evaluated_milestones": sum(bool(p.get("session_metadata", {}).get("submitted")) for p in payloads),
                "resolved": sum(bool(p.get("session_metrics", {}).get("resolved")) for p in payloads),
                "score": sum(float(p.get("score", 0.0)) for p in payloads) / len(graded),
            }
        graded_count = sum(repo["graded_milestones"] for repo in repositories.values())
        evaluated_count = sum(repo["evaluated_milestones"] for repo in repositories.values())
        return BenchmarkResults(
            benchmark_name="swemilestone",
            total_tasks=len(self._rows),
            score=sum(repo["score"] for repo in repositories.values()) / len(repositories) if repositories else 0.0,
            metrics={
                "repositories": repositories,
                "graded_milestones": graded_count,
                "ungraded_milestones": len(self._rows) - graded_count,
                "evaluated_milestones": evaluated_count,
                "missing_graded_milestones": graded_count - evaluated_count,
                "partial": evaluated_count < graded_count,
                "benchmark_version": BENCHMARK_VERSION,
            },
        )
