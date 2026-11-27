# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""PREMiSE-style shared-playbook learning for one-step trajectory preferences."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .reward_data import training_order, write_json
from .reward_prompts import analysis_messages, judge_messages
from .selection import select_ps_st_pareto_beam


class RewardRequestError(RuntimeError):
    """An exhausted request or invalid output abandons this single bandit decision."""


class RewardLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        traces: Path,
        slots: threading.BoundedSemaphore,
        timeout: float = 600,
        request_retries: int = 1,
        output_retries: int = 5,
    ):
        self.base_url, self.model, self.traces, self.slots = base_url, model.removeprefix("openai/"), traces, slots
        self.timeout, self.request_retries, self.output_retries = timeout, request_retries, output_retries
        self.local = threading.local()

    def ask(self, messages: list[dict], stage: str, *, temperature: float = 0.0) -> dict:
        if not hasattr(self.local, "client"):
            self.local.client = OpenAI(
                base_url=self.base_url,
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                timeout=self.timeout,
                max_retries=0,
            )
        trace = {
            "stage": stage,
            "started_at": datetime.now(UTC).isoformat(),
            "status": "queued",
            "request": {
                "messages": messages,
                "model": self.model,
                "temperature": temperature,
                "max_tokens": 2048,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "attempts": [],
        }
        path = self.traces / f"{uuid.uuid4().hex}.json"
        write_json(path, trace)
        errors = truncations = 0
        started = time.monotonic()
        try:
            with self.slots:
                trace.update(status="requesting", queue_seconds=time.monotonic() - started)
                write_json(path, trace)
                while True:
                    attempt_started = time.monotonic()
                    try:
                        response = self.local.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=2048,
                            response_format={"type": "json_object"},
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        )
                        choice = response.choices[0]
                        trace["attempts"].append(
                            {
                                "finish_reason": choice.finish_reason,
                                "usage": response.usage.model_dump() if response.usage else None,
                                "elapsed_seconds": time.monotonic() - attempt_started,
                                "content": getattr(choice.message, "content", None)
                                if hasattr(choice, "message")
                                else None,
                            }
                        )
                        if choice.finish_reason == "length":
                            if truncations >= self.output_retries:
                                raise RewardRequestError("Output limit exhausted after six identical requests")
                            truncations += 1
                            write_json(path, trace)
                            continue
                        value = json.loads(choice.message.content or "")
                        if not isinstance(value, dict):
                            raise RewardRequestError("Expected one JSON object")
                        trace.update(status="completed", response=value)
                        return value
                    except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
                        retryable = (
                            not isinstance(exc, APIStatusError)
                            or exc.status_code in (408, 409, 429)
                            or exc.status_code >= 500
                        )
                        trace["attempts"].append(
                            {
                                "error_type": type(exc).__name__,
                                "status_code": getattr(exc, "status_code", None),
                                "elapsed_seconds": time.monotonic() - attempt_started,
                            }
                        )
                        if not retryable or errors >= self.request_retries:
                            raise RewardRequestError(f"Request recovery exhausted: {type(exc).__name__}") from exc
                        errors += 1
                        time.sleep(0.5)
        except Exception as exc:
            trace.update(status="error", error_type=type(exc).__name__, error_message=str(exc))
            raise
        finally:
            trace.update(
                elapsed_seconds=time.monotonic() - started, output_limit_retries=truncations, request_retries=errors
            )
            write_json(path, trace)


def mutate_rules(parent: list[dict], next_id: int, response: dict, forbidden: list[str]) -> tuple[list[dict], int]:
    rules = copy.deepcopy(parent)
    operations = response.get("operations")
    if not isinstance(operations, list) or len(operations) > 3:
        raise ValueError("Expected at most three mutation operations")
    for operation in operations:
        kind, content = operation.get("type"), operation.get("content", "")
        if kind not in ("ADD", "UPDATE"):
            raise ValueError("Only ADD and UPDATE are supported; rule deletion is disabled")
        if not isinstance(content, str) or not content.strip() or len(content) > 1000:
            raise ValueError("Rule content must be concise nonempty text")
        if not content.startswith("Prefer "):
            raise ValueError("Rule must express a judge comparison, starting with Prefer")
        if re.search(r"\b(physics|solver|simulation|passport|nationality|drafts?|playlists?)\b", content, re.I):
            raise ValueError("Rule describes a particular task domain or artifact")
        if re.search(r"\bbfcl\b|appworld|terminal[ _-]?bench|multi_turn|/home/|/workspace/|/tmp/", content, re.I):
            raise ValueError("Rule contains benchmark or concrete path metadata")
        if re.search(r"\be\.g\.|\bfor example\b|\b[A-Za-z]\w*_\w+\b|['\"`]finish['\"`]", content, re.I):
            raise ValueError("Rule contains a concrete example or function identifier")
        if any(term and term.lower() in content.lower() for term in forbidden):
            raise ValueError("Rule memorizes a concrete task/tool identifier")
        if kind == "ADD":
            if not any(rule["content"].casefold() == content.casefold() for rule in rules):
                rules.append({"id": f"rule-{next_id:05d}", "content": content.strip()})
                next_id += 1
        else:
            matches = [i for i, rule in enumerate(rules) if rule["id"] == operation.get("id")]
            if not matches:
                raise ValueError("Mutation references an unknown rule")
            rules[matches[0]]["content"] = content.strip()
    return rules, next_id


class RewardPipeline:
    def __init__(
        self,
        dataset: Path,
        output: Path,
        mode: str,
        seed: int,
        llm: RewardLLM,
        past_count: int = 10,
        *,
        revise_prompts: bool = False,
        parent_count: int = 3,
        children_per_parent: int = 3,
    ):
        if parent_count < 1 or children_per_parent < 1 or past_count < 0:
            raise ValueError("Parent/child counts must be positive and past count nonnegative")
        self.dataset, self.output, self.mode, self.seed, self.llm = dataset, output, mode, seed, llm
        self.manifest_bytes = (dataset / "manifest.json").read_bytes()
        self.manifest = json.loads(self.manifest_bytes)
        self.records = {record["id"]: record for record in self.manifest["records"]}
        self.order = training_order(self.manifest["records"], mode, seed)
        self.past_count = past_count
        self.parent_count, self.children_per_parent = parent_count, children_per_parent
        self.rng = random.Random(seed)
        from . import reward_prompts

        self.config = {
            "dataset": str(dataset.resolve()),
            "manifest_sha256": hashlib.sha256(self.manifest_bytes).hexdigest(),
            "mode": mode,
            "seed": seed,
            "past_count": past_count,
            "order": self.order,
            "model": llm.model,
            "selection": "Pareto current/past binary judge accuracy",
            "replay": "Full single-decision replay; no environment, no span or alignment judge",
            "parent_count": parent_count,
            "children_per_parent": children_per_parent,
            "max_rules": None,
            "max_rule_chars": None,
            "mutation_operations": ["ADD", "UPDATE"],
            "reward": "1 iff judge choice matches preference derived from official environment scores; else 0",
            "alignment_scoring": False,
            "judge_temperature": 0.0,
            "analysis_temperature": 0.7,
            "timeout": llm.timeout,
            "request_retries": llm.request_retries,
            "output_retries": llm.output_retries,
            "max_tokens": 2048,
            "prompts_sha256": hashlib.sha256(
                json.dumps(
                    [reward_prompts.JUDGE_SYSTEM, reward_prompts.REFLECTOR_SYSTEM, reward_prompts.MUTATOR_SYSTEM]
                ).encode()
            ).hexdigest(),
        }
        state_path = output / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["config"] != self.config:
                old_settings = {k: v for k, v in state["config"].items() if k != "prompts_sha256"}
                new_settings = {k: v for k, v in self.config.items() if k != "prompts_sha256"}
                if not revise_prompts or old_settings != new_settings:
                    raise ValueError("Existing reward training configuration differs")
                write_json(
                    output / "prompt_revisions" / f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}.json",
                    {
                        "previous_state": state,
                        "new_config": self.config,
                        "reason": "Explicit prompt revision; preserve cursor, beam, memory, split, and RNG",
                    },
                )
            self.cursor, self.beam, self.memory = state["cursor"], state["beam"], state["memory"]
            import ast

            self.rng.setstate(ast.literal_eval(state["rng_state"]))
            if state["config"] != self.config:
                self.save()
        else:
            self.cursor, self.beam, self.memory = 0, [{"rules": [], "next_id": 1} for _ in range(parent_count)], []
            self.save()
        if not (output / "checkpoints/checkpoint_00000.json").exists():
            self.checkpoint()

    @lru_cache(maxsize=96)  # noqa: B019 - two long-lived experiment pipelines own this bounded cache.
    def visible(self, pair_id: str) -> dict:
        value = json.loads((self.dataset / "pairs" / f"{pair_id}.json").read_text())
        if set(value) != {"task", "context", "public_tools", "common_prefix", "trajectory_a", "trajectory_b"}:
            raise ValueError("Unexpected judge input metadata")
        return value

    def judge(self, pair_id: str, rules: list[dict]) -> dict:
        value = self.llm.ask(judge_messages(self.visible(pair_id), rules), "judge")
        if value.get("choice") not in ("A", "B") or not isinstance(value.get("reason"), str):
            raise ValueError("Invalid judge decision")
        return value

    def scored_judge(self, pair_id: str, rules: list[dict]) -> dict:
        decision = self.judge(pair_id, rules)
        return {**decision, "reward": int(decision["choice"] == self.records[pair_id]["label"]), "pair_id": pair_id}

    def save(self) -> None:
        write_json(
            self.output / "state.json",
            {
                "cursor": self.cursor,
                "config": self.config,
                "beam": self.beam,
                "memory": self.memory,
                "rng_state": repr(self.rng.getstate()),
            },
        )

    def checkpoint(self) -> Path:
        path = self.output / "checkpoints" / f"checkpoint_{self.cursor:05d}.json"
        if not path.exists():
            write_json(path, json.loads((self.output / "state.json").read_text()))
        return path

    def candidate(self, rank: int, index: int, pair_id: str, decision: dict, parent: dict) -> dict:
        visible = self.visible(pair_id)
        label = self.records[pair_id]["label"]
        reflection = self.llm.ask(
            analysis_messages("reflection", visible, parent["rules"], decision, label), "reflection", temperature=0.7
        )
        if not all(
            isinstance(reflection.get(key), str)
            for key in ("diagnosis", "evidence", "criterion", "limits", "uncertainty")
        ):
            raise ValueError("Invalid reflection")
        mutation = self.llm.ask(
            analysis_messages("mutation", visible, parent["rules"], decision, label, reflection),
            "mutation",
            temperature=0.7,
        )
        forbidden = [self.records[pair_id]["task_id"]]
        forbidden += [
            name
            for tool in visible["public_tools"]
            if "_" in tool["name"]
            for name in (tool["name"], tool["name"].rsplit("__", 1)[-1])
        ]
        rules, next_id = mutate_rules(parent["rules"], parent["next_id"], mutation, forbidden)
        return {
            "id": f"candidate_{index}",
            "parent_rank": rank + 1,
            "rules": rules,
            "next_id": next_id,
            "reflection": reflection,
            "mutation": mutation,
            "valid": True,
        }

    def step(self) -> dict:
        from .selection import sample_past

        started = time.monotonic()
        pair_id = self.order[self.cursor]
        if self.records[pair_id]["split"] != "train":
            raise RuntimeError("Validation example reached training")
        iteration = self.cursor + 1
        decisions = [None] * self.parent_count
        errors, candidates = [], []
        with ThreadPoolExecutor(max_workers=50) as pool:
            jobs = {
                pool.submit(self.scored_judge, pair_id, parent["rules"]): rank for rank, parent in enumerate(self.beam)
            }
            for future in as_completed(jobs):
                try:
                    decisions[jobs[future]] = future.result()
                except Exception as exc:
                    errors.append(
                        {"stage": "parent", "rank": jobs[future], "error_type": type(exc).__name__, "message": str(exc)}
                    )
            jobs = {
                pool.submit(
                    self.candidate,
                    rank,
                    rank * self.children_per_parent + child + 1,
                    pair_id,
                    decisions[rank],
                    parent,
                ): rank * self.children_per_parent + child + 1
                for rank, parent in enumerate(self.beam)
                if decisions[rank] is not None
                for child in range(self.children_per_parent)
            }
            for future in as_completed(jobs):
                try:
                    candidates.append(future.result())
                except Exception as exc:
                    errors.append(
                        {
                            "stage": "mutation",
                            "candidate": jobs[future],
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
            past = sample_past(
                [{"benchmark": self.records[item]["benchmark"], "id": item} for item in self.memory],
                self.past_count,
                self.rng,
            )
            past_ids = [item["id"] for item in past]
            outcomes = {}
            jobs = {
                pool.submit(self.scored_judge, target, child["rules"]): (child["id"], target)
                for child in candidates
                for target in [pair_id, *past_ids]
            }
            for future in as_completed(jobs):
                key = jobs[future]
                try:
                    outcomes[key] = future.result()
                except Exception as exc:
                    outcomes[key] = {"error_type": type(exc).__name__, "message": str(exc)}
            for child in candidates:
                current = outcomes[child["id"], pair_id]
                previous = [outcomes[child["id"], item] for item in past_ids]
                child.update(current_result=current, past_results=previous)
                if "reward" not in current or any("reward" not in value for value in previous):
                    child["valid"] = False
                    continue
                child["current_official_score"] = current["reward"]
                child["past_official_score"] = (
                    sum(value["reward"] for value in previous) / len(previous) if previous else None
                )
        candidates.sort(key=lambda c: c["id"])
        selected, selection = (None, None)
        if sum(child["valid"] for child in candidates) >= self.parent_count:
            selected, selection = select_ps_st_pareto_beam(candidates, beam_size=self.parent_count)
            self.beam = [{"rules": child["rules"], "next_id": child["next_id"]} for child in selected]
            self.memory.append(pair_id)
        result = {
            "iteration": iteration,
            "pair_id": pair_id,
            "benchmark": self.records[pair_id]["benchmark"],
            "status": "updated" if selected else "skipped",
            "parents": decisions,
            "candidates": candidates,
            "past_ids": past_ids,
            "selection": selection,
            "errors": errors,
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(self.output / "iterations" / f"iter_{iteration:05d}.json", result)
        self.cursor = iteration
        self.save()
        print(
            f"[{self.mode} {self.cursor}/{len(self.order)}] {result['benchmark']} {result['status']} "
            f"rules={len(self.beam[0]['rules'])} seconds={result['elapsed_seconds']:.1f}",
            flush=True,
        )
        return result
