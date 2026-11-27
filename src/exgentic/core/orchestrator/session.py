# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from contextlib import ExitStack

from ...utils.context_errors import is_context_length_error
from ..context import agent_scope, benchmark_scope, session_scope
from ..experiment_limits import (
    LIMITS_CONTEXT_KEY,
    ExperimentBudgetExceededError,
    budget_error,
    budget_failure_score,
    experiment_limits,
)
from ..types import SessionConfig
from .controller import Controller
from .observer import Observer
from .termination import (
    AgentError,
    AgentTerminationError,
    BenchmarkError,
    BenchmarkTerminationError,
    RunCancelError,
    SessionCancelError,
    SessionLimitReachedError,
)
from .tracker import Tracker


def _send_training_feedback(agent, agent_instance, score) -> None:
    if getattr(agent, "training_time", False) and score.session_metadata.get("training_feedback_available", True):
        with agent_scope():
            agent_instance.receive_training_feedback(score.model_dump(mode="json"))


def _send_context_error(agent, agent_instance, error) -> None:
    if getattr(agent, "slug_name", None) == "ace" and is_context_length_error(error):
        with agent_scope():
            agent_instance.receive_training_feedback({"session_metadata": {
                "skip_reason": "context_length_exceeded", "error": str(getattr(error, "error", None) or error),
            }})


def _budget_failure(session, details, limits):
    with benchmark_scope():
        persist = getattr(session, "fail_experiment_budget", None)
        if persist is not None:
            return persist(details, limits)
    return budget_failure_score(details, experiment_limits=limits)


def run_session(
    session_config: SessionConfig,
    session,
    agent,
    observers: list[Observer] | None = None,
    controllers: list[Controller] | None = None,
    *,
    tracker: Tracker | None = None,
) -> None:
    """Process a single session.

    The *agent_instance* is created via ``agent.get_instance()`` for
    runner isolation.
    """
    if tracker is None:
        tracker = Tracker(observers=observers, controllers=controllers)
    limits = experiment_limits(session_config.benchmark) if session_config is not None else None
    prefix_turns = len((session_config.replay_prefix or {}).get("actions", [])) if session_config else 0
    turns = prefix_turns

    with session_scope(session.session_id, task_id=session.task_id), ExitStack() as cleanup:
        agent_instance = agent.get_instance(session_id=session.session_id)
        cleanup.callback(agent_instance.close)

        with benchmark_scope():
            observation = session.start()
        cleanup.callback(session.close)

        with agent_scope():
            context = dict(session.context or {})
            if limits:
                context[LIMITS_CONTEXT_KEY] = {**limits, "prefix_turns": prefix_turns}
            agent_instance.start(
                task=session.task,
                context=context,
                actions=session.actions,
            )

        try:
            tracker.on_session_start(session, agent_instance, observation)
            while not session.done() and observation is not None:
                if limits and turns >= limits["max_turns"]:
                    raise ExperimentBudgetExceededError(
                        "max_turns", limit=limits["max_turns"], actual=turns + 1, turn=turns + 1,
                    )
                try:
                    with agent_scope():
                        action = agent_instance.react(observation)
                except Exception as exc:
                    if failure := budget_error(exc):
                        raise failure from exc
                    tracker.on_react_error(session, exc)

                tracker.on_react_success(
                    session,
                    action,
                )

                try:
                    with benchmark_scope():
                        observation = session.step(action)
                except Exception as exc:
                    if failure := budget_error(exc):
                        raise failure from exc
                    tracker.on_step_error(
                        session,
                        exc,
                    )

                turns += 1
                tracker.on_step_success(
                    session,
                    observation,
                )

            if session.done():
                raise BenchmarkTerminationError()
            raise AgentTerminationError()

        except KeyboardInterrupt:
            tracker.on_session_error(session, RunCancelError())
            raise
        except ExperimentBudgetExceededError as exc:
            score = _budget_failure(session, exc.details, limits)
            _send_training_feedback(agent, agent_instance, score)
            tracker.on_session_success(session, score, agent_instance)
        except AgentError as exc:
            _send_context_error(agent, agent_instance, exc)
            tracker.on_session_error(session, exc)
        except SessionLimitReachedError as exc:
            if limits:
                failure = ExperimentBudgetExceededError(
                    "max_turns" if exc.reason == "max_steps" else exc.reason,
                    limit=exc.max_steps + prefix_turns if exc.reason == "max_steps" else exc.max_actions,
                    actual=exc.steps + prefix_turns + 1 if exc.reason == "max_steps" else exc.actions + 1,
                    turn=turns + 1,
                )
                score = _budget_failure(session, failure.details, limits)
                _send_training_feedback(agent, agent_instance, score)
                tracker.on_session_success(session, score, agent_instance)
                return
            with benchmark_scope():
                tracker.on_session_scoring(session)
            with benchmark_scope():
                score = session.score()
            score.is_finished = False
            score.session_metadata = {
                **(score.session_metadata or {}),
                "limit_reached": True,
                "limit_reason": exc.reason,
                "max_steps": exc.max_steps,
                "max_actions": exc.max_actions,
                "steps": exc.steps,
                "actions": exc.actions,
            }
            _send_training_feedback(agent, agent_instance, score)
            tracker.on_session_success(session, score, agent_instance)
        except (AgentTerminationError, BenchmarkTerminationError):
            with benchmark_scope():
                tracker.on_session_scoring(session)
            with benchmark_scope():
                score = session.score()
            if score.is_finished is None:
                score.is_finished = True
            _send_training_feedback(agent, agent_instance, score)
            tracker.on_session_success(session, score, agent_instance)
        except BenchmarkError as exc:
            _send_context_error(agent, agent_instance, exc)
            tracker.on_session_error(session, exc)
        except SessionCancelError as exc:
            tracker.on_session_error(session, exc)
        except RunCancelError as exc:
            tracker.on_session_error(session, exc)
            raise
