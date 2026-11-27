# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Evaluate frozen action prefixes without constructing or stepping an environment."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ...actor_context import shared_actor_prompt
from ...core.experiment_limits import (
    ACTOR_OUTPUT_TOKENS,
    ActorTurnBudget,
    ExperimentBudgetExceededError,
    actor_completion,
    experiment_limits,
)
from ...utils.context_errors import is_context_length_error
from ..ace.prompts.generator import build_generator_system_prompt
from ..litellm_tool_calling.instance import CompletionRetriesExhaustedError
from .llm import shared_llm_slots

VERSION = "span-policy-shift-strict-f1-v1"
SCHEDULING_VERSION = "shared_action_queue_v1"
HISTORY_FILE = "policy_shift_actions.jsonl"
GENERATION_KEYS = (
    "model", "temperature", "max_tokens", "top_p", "top_k", "min_p", "presence_penalty",
    "frequency_penalty", "repetition_penalty", "reasoning_effort", "seed", "extra_body",
    "parallel_tool_calls", "stop", "response_format", "allowed_openai_params",
)


def generation_settings(kwargs: dict) -> dict:
    """Record only generation parameters; never provider credentials or headers."""
    return {key: copy.deepcopy(kwargs[key]) for key in GENERATION_KEYS if key in kwargs}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def strict_f1(predicted: list[dict], reference: list[dict]) -> float:
    """Match CP policy-shift strict F1: exact action multisets, including duplicates."""
    pred, ref = Counter(map(canonical, predicted)), Counter(map(canonical, reference))
    total = sum(pred.values()) + sum(ref.values())
    return 2 * sum((pred & ref).values()) / total if total else 1.0


@lru_cache(maxsize=8)
def terminal_parser(path: str, sha256: str):
    """Load only the recorded native parser, without importing Harbor or its environment."""
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
        raise ValueError("Recorded Terminus-2 parser changed")
    name = f"_premise_shift_parser_{sha256}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Recorded Terminus-2 parser is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.TerminusJSONPlainParser()


def normalize_action(message: dict, benchmark: str, parser=None) -> list[dict]:
    if benchmark == "terminalbench2":
        parsed = parser.parse_response(message.get("content") or "")
        if parsed.error:
            raise ValueError(f"Invalid native action: {parsed.error}")
        actions = [
            {
                "name": "terminal_command",
                "arguments": {
                    "keystrokes": command.keystrokes,
                    "duration": float(command.duration),
                },
            }
            for command in parsed.commands
        ]
        if parsed.is_task_complete:
            actions.append({"name": "task_complete", "arguments": {}})
        return actions
    actions = []
    for call in message.get("tool_calls") or []:
        function = call["function"]
        arguments = function["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict) or not isinstance(function["name"], str) or not function["name"]:
            raise ValueError("Invalid function-call action")
        actions.append({"name": function["name"], "arguments": arguments})
    if not actions:
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError("Empty actor response")
        actions.append({"name": "message", "arguments": {"content": content}})
    return actions


def replace_policy(request: dict, original: str, replacement: str) -> dict:
    """Change only the original actor instruction; retain every saved history/tool byte."""
    result = copy.deepcopy(request)
    content = result["messages"][0]["content"]
    parts = (
        [(None, content)]
        if isinstance(content, str)
        else [(index, part["text"]) for index, part in enumerate(content) if part.get("type") == "text"]
    )
    matches = [(index, text) for index, text in parts if original and text.count(original) == 1]
    if len(matches) != 1:
        raise ValueError("Original actor policy is not uniquely located in the saved initial instruction")
    index, text = matches[0]
    changed = text.replace(original, replacement, 1)
    if index is None:
        result["messages"][0]["content"] = changed
    else:
        content[index]["text"] = changed
    return result


@dataclass
class _SpanEvaluation:
    key: Any
    benchmark: str
    summary: dict
    path: Path
    original: str
    replacement: str
    rows: list[dict]
    actions: dict[int, dict] = field(default_factory=dict)
    errors: list[tuple[int, Exception]] = field(default_factory=list)
    budget_failures: list[tuple[int, ExperimentBudgetExceededError]] = field(default_factory=list)


class SpanPolicyShiftEvaluator:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        output: Path,
        temperature: float = 0.0,
        max_tokens: int | None = ACTOR_OUTPUT_TOKENS,
        timeout: float = 600,
        num_retries: int = 1,
        match_source_settings: bool = False,
    ):
        self.model, self.base_url, self.output = model, base_url, output
        self.temperature = temperature
        self.max_tokens = min(max_tokens, ACTOR_OUTPUT_TOKENS) if max_tokens is not None else None
        self.timeout, self.num_retries = timeout, num_retries
        self.match_source_settings = match_source_settings
        self._log_lock = threading.Lock()
        self.last_parallelism = {}

    def _complete(self, request: dict, *, benchmark: str, action_index: int, log_path: Path):
        limits = experiment_limits(benchmark)
        if limits is None:
            raise ValueError(f"Span policy shift has no actor budget policy for {benchmark}")
        if self.max_tokens is None:
            limits = {**limits, "version": "actor-turn-v3-output-unlimited", "output_tokens": None}
        actor = SimpleNamespace(_actor_turn_budget=ActorTurnBudget(limits), _step_count=action_index + 1)
        kwargs = {k: copy.deepcopy(request[k]) for k in ("messages", "tools", "tool_choice") if k in request}
        kwargs.update(
            model=self.model,
            api_base=self.base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            caching=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if self.match_source_settings:
            settings = request.get("generation_settings")
            if not isinstance(settings, dict) or "temperature" not in settings:
                raise ValueError("Matched probes require recorded actor generation settings")
            if settings["temperature"] != self.temperature or settings.get("max_tokens") != self.max_tokens:
                raise ValueError("Rollout and probe temperature/output limits must match")
            source_model = settings.get("model", self.model)
            if source_model.removeprefix("openai/") != self.model.removeprefix("openai/"):
                raise ValueError("Rollout and probe model must match")
            kwargs.update(generation_settings(settings))
        usage_records = []
        attempts = 0

        def complete(**call_kwargs):
            import litellm

            nonlocal attempts
            attempts += 1
            record = {
                "action_index": action_index,
                "attempt": attempts,
                # Explicit allowlist: provider credentials/headers never enter artifacts.
                "request": {
                    k: call_kwargs[k]
                    for k in ("messages", "tools", "tool_choice", *GENERATION_KEYS)
                    if k in call_kwargs
                },
            }
            try:
                response = litellm.completion(**call_kwargs)
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                raise
            else:
                payload = response.model_dump(mode="json")
                cost = getattr(response, "_hidden_params", {}).get("response_cost")
                usage = {**(payload.get("usage") or {}), "cost_usd": cost}
                usage_records.append(usage)
                record.update(response=payload, cost_usd=cost)
            finally:
                with self._log_lock, log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            return response

        for attempt in range(self.num_retries + 1):
            try:
                with shared_llm_slots():
                    response = actor_completion(actor, kwargs, completion_fn=complete)
                return response.model_dump(mode="json")["choices"][0]["message"], usage_records
            except ExperimentBudgetExceededError as exc:
                exc.usage_records = usage_records
                raise
            except Exception as exc:
                if is_context_length_error(exc):
                    raise
                if isinstance(exc, (ValueError, KeyError, TypeError)):
                    raise
                if attempt >= self.num_retries:
                    raise CompletionRetriesExhaustedError(
                        f"Span policy shift exhausted {attempt + 1} transport attempts: {type(exc).__name__}: {exc}"
                    ) from exc
        raise AssertionError("Unreachable request loop")

    def _prepare(
        self, key, candidate: dict, source: dict, reflection: dict, label: str, record_cache: dict
    ) -> _SpanEvaluation:
        benchmark = source["benchmark"]
        success = source.get("success")
        if type(success) is not bool:
            raise ValueError("Span policy shift requires the original official trajectory success flag")
        start, end = reflection["span"]["start"], reflection["span"]["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source["actions"]):
            raise ValueError("Invalid policy-shift action span")
        if source.get("prefix_length", 0):
            raise ValueError("Policy-shift references must be original parent trajectories")
        history_path = source["policy_shift_history"]
        if history_path not in record_cache:
            records = {}
            with Path(history_path).open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    index = row["action_index"]
                    if index in records:
                        raise ValueError("Duplicate saved action prefix")
                    records[index] = row
            record_cache[history_path] = records
        rows = [record_cache[history_path][index] for index in range(start, end)]
        original = source["actor_policy"]
        replacement = candidate["playbook"]["playbook"]
        if source.get("actor_context", "legacy") != "legacy":
            replacement = shared_actor_prompt(benchmark, replacement)
        elif benchmark != "terminalbench2":
            replacement = build_generator_system_prompt(replacement)
        path = self.output / "policy_shift_calls" / f"{label}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "metric": "strict_f1",
            "version": VERSION,
            "reference": "recorded_training_action",
            "source_success": success,
            "source_task_id": source["task_id"],
            "span": {"start": start, "end": end},
            "environment_execution": False,
            "scheduling": SCHEDULING_VERSION,
        }
        return _SpanEvaluation(key, benchmark, summary, path, original, replacement, rows)

    def _evaluate_action(self, span: _SpanEvaluation, row: dict) -> dict:
        index = row["action_index"]
        parser = terminal_parser(**row["terminal_parser"]) if span.benchmark == "terminalbench2" else None
        reference = normalize_action(row["message"], span.benchmark, parser)
        request = replace_policy(row["request"], span.original, span.replacement)
        if self.match_source_settings:
            request["generation_settings"] = row.get("generation_settings", request.get("generation_settings"))
        message, usage = self._complete(request, benchmark=span.benchmark, action_index=index, log_path=span.path)
        predicted = normalize_action(message, span.benchmark, parser)
        return {
            "action_index": index,
            "reference": reference,
            "predicted": predicted,
            "strict_f1": strict_f1(predicted, reference),
            "usage": usage,
            "history_sha256": hashlib.sha256(canonical(row["request"]).encode()).hexdigest(),
        }

    @staticmethod
    def _summarize(span: _SpanEvaluation) -> dict | Exception:
        results = [span.actions[index] for index in sorted(span.actions)]
        summary = {**span.summary, "actions": results}
        if span.budget_failures:
            failures = sorted(span.budget_failures, key=lambda item: item[0])
            usage = [record for _, failure in failures for record in failure.usage_records]
            # Pending work is cancelled; already running calls are drained and charged.
            return {
                **summary,
                "evaluation_score": 0.0,
                "success": False,
                "experiment_budget_failure": failures[0][1].details,
                "failed_action_usage": usage,
                "action_generations": len(results) + len(usage),
                "status": "budget_exceeded",
            }
        if span.errors:
            return min(span.errors, key=lambda item: item[0])[1]
        agreement = sum(result["strict_f1"] for result in results) / len(results)
        return {
            **summary,
            "strict_f1": agreement,
            "policy_shift": 1 - agreement,
            "evaluation_score": agreement if summary["source_success"] else 1 - agreement,
            "action_generations": len(results),
            "status": "completed",
        }

    def evaluate_batch(self, evaluations: list[tuple], *, max_workers: int) -> dict:
        """Submit all candidates' current/past action positions to one bounded queue."""
        if max_workers < 1:
            raise ValueError("Policy-shift action workers must be positive")
        spans, outcomes, record_cache = {}, {}, {}
        for key, candidate, source, reflection, label in evaluations:
            if key in spans or key in outcomes:
                raise ValueError("Duplicate policy-shift evaluation key")
            try:
                spans[key] = self._prepare(key, candidate, source, reflection, label, record_cache)
            except Exception as exc:
                outcomes[key] = exc
        jobs = [(span, row) for span in spans.values() for row in span.rows]
        stats = {"queued_action_jobs": len(jobs), "peak_action_workers": 0, "cancelled_action_jobs": 0}
        lock = threading.Lock()
        active = 0

        def run_action(span, row):
            nonlocal active
            with lock:
                active += 1
                stats["peak_action_workers"] = max(stats["peak_action_workers"], active)
            try:
                return self._evaluate_action(span, row)
            finally:
                with lock:
                    active -= 1

        with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs)) or 1) as pool:
            futures = {pool.submit(run_action, span, row): (span, row["action_index"]) for span, row in jobs}

            def cancel_pending(key=None):
                for future, (span, _) in futures.items():
                    if (key is None or span.key == key) and not future.done() and future.cancel():
                        stats["cancelled_action_jobs"] += 1

            try:
                for future in as_completed(futures):
                    span, index = futures[future]
                    try:
                        span.actions[index] = future.result()
                    except CancelledError:
                        continue
                    except ExperimentBudgetExceededError as exc:
                        span.budget_failures.append((index, exc))
                        cancel_pending(span.key)
                    except CompletionRetriesExhaustedError:
                        cancel_pending()
                        raise
                    except Exception as exc:
                        span.errors.append((index, exc))
                        cancel_pending(span.key)
            finally:
                self.last_parallelism = stats
        outcomes.update({key: self._summarize(span) for key, span in spans.items()})
        return outcomes

    def evaluate(self, candidate: dict, source: dict, reflection: dict, label: str, *, max_workers: int = 28) -> dict:
        """Evaluate one span with the same independent-action queue used by the pipeline."""
        result = self.evaluate_batch([(0, candidate, source, reflection, label)], max_workers=max_workers)[0]
        if isinstance(result, Exception):
            raise result
        return result
