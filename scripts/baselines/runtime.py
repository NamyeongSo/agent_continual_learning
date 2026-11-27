# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Shared isolated-baseline transport; optimizer dependencies stay outside Exgentic."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import runpy
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = Path(__file__).parent / "prompts"
# Optimizer venvs intentionally do not install Exgentic's core dependencies.
# Load the same dependency-free renderer that the core agents import normally.
_CONTEXT = runpy.run_path(str(ROOT / "src/exgentic/actor_context.py"))
ACTOR_CONTEXT_MODES = _CONTEXT["ACTOR_CONTEXT_MODES"]
DEFAULT_ACTOR_CONTEXT = _CONTEXT["DEFAULT_ACTOR_CONTEXT"]
actor_context_metadata = _CONTEXT["actor_context_metadata"]
require_matching_actor_context = _CONTEXT["require_matching_actor_context"]
shared_actor_prompt = _CONTEXT["shared_actor_prompt"]
_INFERENCE = runpy.run_path(str(ROOT / "src/exgentic/inference_profile.py"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prompt(method, name):
    return (PROMPTS / method / name).read_text(encoding="utf-8")


def initial_instruction(method, actor_context="legacy"):
    _CONTEXT["validate_actor_context"](actor_context)
    if actor_context != "legacy":
        return ""
    return prompt(method, "base/shared.txt" if method == "gepa" else "base/seed_skill.md")


def actor_prompt(method, benchmark, learned, actor_context="legacy"):
    _CONTEXT["validate_actor_context"](actor_context)
    if actor_context != "legacy":
        return shared_actor_prompt(benchmark, learned)
    suffix = "txt" if method == "gepa" else "md"
    return prompt(method, f"base/{benchmark}.{suffix}") + "\n\n" + learned


def recover_actor_response(folder, request):
    """Recover an atomic official response interrupted before the parent committed its cache."""
    folder = Path(folder)
    try:
        if read(folder / "request.json") != request:
            return None
    except (OSError, ValueError):
        return None
    for path in sorted(folder.glob("attempt_*.json")):
        try:
            result = read(path)
        except (OSError, ValueError):
            continue
        score = result.get("soft")
        if ("error" not in result and result.get("id") == request["item"]["task_id"]
                and result.get("prompt_sha256") == digest(request["system_prompt"])
                and result.get("hard") in (0, 1) and isinstance(score, (int, float))
                and math.isfinite(score) and 0 <= score <= 1 and isinstance(result.get("trace"), dict)
                and result.get("run_id")):
            return result
    return None


class InfrastructureError(RuntimeError):
    """No official task outcome exists; never use this as an optimizer score."""


class Runtime:
    def __init__(self, config, *, scripted=False):
        self.config = config
        self.actor_context = config.get("actor_context", "legacy")
        require_matching_actor_context(config, self.actor_context, [config["benchmark"]])
        self.output = Path(config["output"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.workers = config["workers"]
        self.scripted = scripted
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(self.workers)

    def append(self, filename, record):
        with self.lock:
            with (self.output / filename).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def token_summary(self):
        """Measured teacher usage, including responses later rejected as truncated."""
        summary = {}
        path = self.output / "analysis_calls.jsonl"
        with self.lock:
            records = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        for raw in records:
            record = json.loads(raw)
            stage = record["stage"]
            counts = summary.setdefault(stage, {"calls": 0, "prompt_tokens": 0,
                                                "completion_tokens": 0, "total_tokens": 0})
            counts["calls"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                counts[key] += (record.get("usage") or {}).get(key, 0) or 0
        summary["_total"] = {key: sum(row[key] for row in summary.values())
                             for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")}
        return summary

    def task(self, item, learned, label):
        identity = {"item": item, "prompt_sha256": digest(learned), "label": label}
        if self.actor_context != "legacy":
            identity["actor_context_sha256"] = digest(json.dumps(self.config["actor_context_metadata"], sort_keys=True))
        key = digest(json.dumps(identity, sort_keys=True))
        folder = self.output / "rollouts" / key
        result_path = folder / "result.json"
        if result_path.exists():
            return read(result_path)
        folder.mkdir(parents=True, exist_ok=True)
        request = {
            **self.config, **identity, "session_output": str(folder / "sessions"),
            "system_prompt": actor_prompt(self.config["method"], self.config["benchmark"], learned, self.actor_context),
            **({"learned_policy": learned} if self.actor_context != "legacy" else {}),
        }
        recovered = recover_actor_response(folder, request)
        if recovered is not None:
            write(result_path, recovered)
            self.append("actor_response_recovery.jsonl", {**identity, "run_id": recovered["run_id"],
                                                         "hard": recovered["hard"]})
            return recovered
        write(folder / "request.json", request)
        if self.scripted:
            # Deliberately start at zero, allowing the official update/gate paths to run offline.
            hard = int("scripted improved" in learned)
            result = {"id": item["task_id"], "hard": hard, "soft": float(hard), "trace": {
                "task": "Scripted offline task", "actions": [{"action": "inspect", "observation": "ok"}],
                "details": {"success": bool(hard)}, "public_tool_contract": {},
            }, "prompt_sha256": digest(learned)}
            write(result_path, result)
            self.append("attempts.jsonl", {**identity, "scripted": True, "hard": hard})
            return result
        last_error = None
        for attempt in range(3):
            response = folder / f"attempt_{attempt}.json"
            started = time.time()
            with self.slots, (folder / f"attempt_{attempt}.log").open("a", encoding="utf-8") as log:
                process = subprocess.run([
                    self.config["core_python"], str(ROOT / "scripts/baselines/actor.py"),
                    "--request", str(folder / "request.json"), "--response", str(response),
                ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            record = {**identity, "attempt": attempt, "seconds": time.time() - started,
                      "returncode": process.returncode, "response": str(response)}
            self.append("attempts.jsonl", record)
            if process.returncode == 0 and response.exists():
                result = read(response)
                write(result_path, result)
                return result
            error = read(response) if response.exists() else {"error": "Actor worker exited without response"}
            last_error = error.get("error")
            if error.get("retryable") is False:
                break
        raise InfrastructureError(f"{item['task_id']}: {last_error}; see {folder}")

    def batch(self, items, learned, label):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = list(pool.map(lambda item: self.task(item, learned, label), items))
        if len(results) != len(items) or any(row.get("hard") not in (0, 1) for row in results):
            raise InfrastructureError("Incomplete batch: refusing to score or gate a partial evaluation")
        return results

    def online(self, items, results, learned, batch):
        for item, result in zip(items, results, strict=True):
            # Immutable per-task record also deduplicates an interrupted iteration replay.
            path = self.output / "online" / f"{digest(item['task_id'])}.json"
            record = {**item, "batch": batch, "hard": result["hard"], "prompt_sha256": digest(learned)}
            if path.exists() and read(path) != record:
                raise RuntimeError("Recovery diverged from an immutable first-encounter record")
            write(path, record)

    def chat(self, system, user, stage):
        if self.scripted:
            if self.config["method"] == "gepa":
                return "```\nscripted improved\n```", {}
            patch = {"edits": [{"op": "append", "content": "scripted improved"}]}
            return json.dumps({"patch": patch} if stage == "analyst" else patch), {}
        cache_path = self.output / "analysis_cache" / f"{digest(json.dumps([system, user, stage]))}.json"
        if cache_path.exists():
            cached = read(cache_path)
            if "error" in cached:
                raise RuntimeError(cached["error"])
            return cached["text"], cached["usage"]
        from openai import OpenAI

        client = OpenAI(base_url=self.config["base_url"], api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        timeout=600, max_retries=0)
        last_error = None
        for attempt in range(3):
            request_id = uuid.uuid4().hex
            folder = self.output / "analysis" / request_id
            write(folder / "request.json", {"system": system, "user": user, "stage": stage,
                                           "temperature": 0.7, "max_tokens": 8192, "enable_thinking": False})
            try:
                with self.slots:
                    request = dict(
                        model=self.config["model"].removeprefix("openai/"),
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        temperature=0.7, max_tokens=8192,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    request = _INFERENCE["apply_request"](request, "analysis")
                    write(folder / "effective_request.json", request)
                    result = client.chat.completions.create(**request)
                payload = result.model_dump(mode="json")
                write(folder / "response.json", payload)
                self.append("analysis_calls.jsonl", {"request_id": request_id, "stage": stage,
                            "attempt": attempt, "usage": payload.get("usage"), "status": "completed"})
                choice = result.choices[0]
                if choice.finish_reason == "length" or not choice.message.content:
                    raise ValueError("Analysis response empty or truncated")
                usage = payload.get("usage") or {}
                write(cache_path, {"text": choice.message.content, "usage": usage})
                return choice.message.content, usage
            except Exception as exc:
                last_error = exc
                self.append("analysis_errors.jsonl", {"request_id": request_id, "stage": stage,
                            "attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
                # Context overflow cannot be fixed by replaying the identical analysis prompt.
                if "context" in str(exc).lower() and "length" in str(exc).lower():
                    break
        message = f"Analysis failed: {last_error}"
        write(cache_path, {"error": message})
        raise RuntimeError(message)


def unfence(text):
    match = re.search(r"```(?:[a-zA-Z]+)?\s*\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def final_test(runtime, learned):
    write(runtime.output / "frozen_instruction.json", {"content": learned, "sha256": digest(learned),
                                                     "selection": "validation_only",
         **({"actor_context": runtime.actor_context,
             "actor_context_metadata": runtime.config["actor_context_metadata"]}
            if runtime.actor_context != "legacy" else {})})
    results = runtime.batch(runtime.config["test"], learned, "final_test")
    summary = {"complete": True, "tasks": len(results), "successes": sum(row["hard"] for row in results),
               "score": sum(row["hard"] for row in results) / len(results), "prompt_sha256": digest(learned)}
    write(runtime.output / "test_results.json", results)
    write(runtime.output / "summary.json", summary)
    return summary
