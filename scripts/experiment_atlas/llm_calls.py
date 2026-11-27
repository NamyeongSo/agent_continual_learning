# SPDX-License-Identifier: Apache-2.0
"""Browse recorded model returns without calling a model or loading entire runs."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

from .shared_reflection import COHORT

MAX_LINE = 16 * 1024 * 1024
PAGE_SIZE = 30
REQUEST_FIELDS = {
    "model", "messages", "tools", "tool_choice", "response_format", "temperature", "top_p", "top_k",
    "max_tokens", "max_completion_tokens", "reasoning_effort", "chat_template_kwargs", "thinking_token_budget",
    "exgentic_reasoning_budget", "exgentic_role", "exgentic_profile_sha256", "extra_body",
}
SECRET_FIELDS = {"api_key", "authorization", "headers", "extra_headers", "environment"}


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def sdk_value(value):
    """Decode logged SDK object reprs as data. Never evaluate constructors."""
    if not isinstance(value, str):
        return value
    try:
        node = ast.parse(value, mode="eval")
    except (SyntaxError, ValueError):
        return value
    if sum(1 for _ in ast.walk(node)) > 20000:
        return value

    def decode(item):
        if isinstance(item, ast.Constant):
            return item.value
        if isinstance(item, (ast.List, ast.Tuple)):
            return [decode(v) for v in item.elts]
        if isinstance(item, ast.Dict):
            return {decode(k): decode(v) for k, v in zip(item.keys, item.values, strict=True)}
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, ast.USub) and isinstance(item.operand, ast.Constant):
            return -item.operand.value
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id in {
            "Choices", "Message", "ChatCompletionMessageToolCall", "Function", "CompletionUsage",
            "ChatCompletionMessage", "Choice", "CompletionTokensDetails", "PromptTokensDetails",
        } and not item.args and all(k.arg is not None for k in item.keywords):
            return {k.arg: decode(k.value) for k in item.keywords}
        raise ValueError("Unsupported logged representation")

    try:
        return decode(node.body)
    except (ValueError, TypeError, RecursionError):
        return value


def safe_request(request):
    result = {k: v for k, v in request.items() if k in REQUEST_FIELDS}
    if isinstance(result.get("extra_body"), dict):
        result["extra_body"] = {k: v for k, v in result["extra_body"].items() if k not in SECRET_FIELDS}
    return result


def module_name(record, request):
    if record.get("stage"):
        return record["stage"]
    if request.get("temperature") == 0.0:
        return "actor"
    messages = request.get("messages") or []
    text = "\n".join(str(m.get("content", "")) for m in messages[:2] if isinstance(m, dict)).lower()
    if "analyze recorded decisions using visible evidence" in text or "first decision that caused" in text:
        return "reflection"
    if "master curator of knowledge" in text:
        return "mutation"
    return "analysis" if request.get("temperature") == 0.7 else "actor"


def normalise(record):
    request = record.get("request") or {}
    response = record.get("response") or {}
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise ValueError("Malformed recorded model call")
    choices = response.get("choices") or []
    parsed = [sdk_value(c) for c in choices] if isinstance(choices, list) else []
    messages = []
    for choice in parsed:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            continue
        specific = message.get("provider_specific_fields") or {}
        messages.append({
            "index": choice.get("index", 0), "finish_reason": choice.get("finish_reason"),
            "reasoning": message.get("reasoning_content") or message.get("reasoning")
            or specific.get("reasoning_content") or specific.get("reasoning") or "",
            "content": message.get("content"), "tool_calls": message.get("tool_calls") or [],
            "function_call": message.get("function_call"),
        })
    usage = sdk_value(response.get("usage")) or {}
    if not isinstance(usage, dict):
        usage = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key not in usage and record.get(key) is not None:
            usage[key] = record[key]
    timestamp = record.get("started_at") or record.get("timestamp")
    if timestamp is None and isinstance(response.get("created"), (int, float)):
        timestamp = datetime.fromtimestamp(response["created"], UTC).isoformat()
    return {
        "request_id": response.get("id"), "timestamp": timestamp,
        "module": module_name(record, request), "status": record.get("status", "recorded"),
        "elapsed_seconds": record.get("elapsed_seconds", record.get("request_seconds")),
        "request": safe_request(request), "response": {k: v for k, v in response.items() if k not in SECRET_FIELDS},
        "messages": messages, "usage": usage, "error": record.get("error"),
    }


class CallBrowser:
    def __init__(self, cohort=COHORT):
        self.cohort = Path(cohort).resolve()
        self.lock = threading.RLock()
        self.source_cache = {}
        self.record_cache = OrderedDict()
        self.bridge_cache = OrderedDict()

    def jobs(self):
        plan = read(self.cohort / "launch_manifest.json", {})
        progress = read(self.cohort / "progress.json", {}).get("jobs", {})
        return {
            "cohort": self.cohort.name,
            "profile_sha256": plan.get("inference_profile_sha256"),
            "jobs": [{k: job[k] for k in ("id", "method", "benchmark", "seed")}
                     | {"status": progress.get(job["id"], {}).get("status", "pending")}
                     for job in plan.get("jobs", [])],
        }

    def job(self, identity):
        plan = read(self.cohort / "launch_manifest.json", {})
        job = next((j for j in plan.get("jobs", []) if j["id"] == identity), None)
        if job is None:
            raise KeyError("Unknown experiment")
        root = Path(job["output"]).resolve()
        if not root.is_relative_to(self.cohort):
            raise ValueError("Experiment is outside the recorded cohort")
        return job, root

    def sources(self, identity):
        with self.lock:
            cached = self.source_cache.get(identity)
            if cached and time.monotonic() - cached[0] < 20:
                return cached[1]
        job, root = self.job(identity)
        result = {}
        stages = {}
        for raw in (root / "train/analysis_calls.jsonl").read_text().splitlines() if (
            root / "train/analysis_calls.jsonl"
        ).exists() else []:
            try:
                value = json.loads(raw)
                stages[value["request_id"]] = value.get("stage", "analysis")
            except (ValueError, KeyError):
                continue
        paths = list(root.glob("**/litellm/trace.jsonl"))
        paths += list((root / "train/llm_calls").glob("*.json"))
        paths += list((root / "train/analysis").glob("*/response.json"))
        context_cache = {}
        for path in paths:
            try:
                resolved = path.resolve()
                if not resolved.is_relative_to(root) or not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                split, task, label = ("test" if relative.startswith("evaluation/") else "train"), "", ""
                if "/rollouts/" in "/" + relative:
                    folder = root / "train/rollouts" / relative.split("rollouts/", 1)[1].split("/", 1)[0]
                    if folder not in context_cache:
                        context_cache[folder] = read(folder / "request.json", {})
                    context = context_cache[folder]
                    task = context.get("item", {}).get("task_id", "")
                    split = context.get("item", {}).get("split", split)
                    label = context.get("label", "")
                elif path.name == "trace.jsonl":
                    for parent in path.parents:
                        if parent == root:
                            break
                        if parent.parent.name == "sessions":
                            if parent not in context_cache:
                                context_cache[parent] = read(parent / "config.json", {})
                            task = context_cache[parent].get("task_id", "")
                            label = context_cache[parent].get("run_id", "")
                            break
                    if not task and path.parent.parent.name == "run":
                        configs = list((path.parents[2] / "sessions").glob("*/config.json"))
                        if len(configs) == 1:
                            if configs[0] not in context_cache:
                                context_cache[configs[0]] = read(configs[0], {})
                            task = context_cache[configs[0]].get("task_id", "")
                            label = context_cache[configs[0]].get("run_id", "")
                stage = ""
                if "/llm_calls/" in "/" + relative:
                    with path.open() as stream:
                        prefix = stream.read(512)
                    match = re.search(r'"stage"\s*:\s*"([^"]+)"', prefix)
                    stage = match.group(1) if match else "analysis"
                    kind, label = "analysis", stage
                elif "/analysis/" in "/" + relative:
                    stage = stages.get(path.parent.name, "analysis")
                    kind, label = "analysis", stage
                else:
                    kind = "analysis" if "/framework/" in "/" + relative else "actor"
                # Tiny health-only traces have no experiment response to inspect.
                if path.name == "trace.jsonl" and path.stat().st_size < 4096:
                    tiny = [v for _, _, v in self.raw_records(path)]
                    if not tiny or all(self.health(v) for v in tiny):
                        continue
                key = hashlib.sha256(relative.encode()).hexdigest()[:24]
                result[key] = {
                    "id": key, "path": relative, "kind": kind, "split": split, "task_id": task,
                    "label": label, "stage": stage, "modified_at": path.stat().st_mtime,
                    "bytes": path.stat().st_size, "method": job["method"],
                }
            except (OSError, ValueError):
                continue
        with self.lock:
            self.source_cache[identity] = (time.monotonic(), result)
        return result

    @staticmethod
    def health(value):
        request = value.get("request") or {}
        return request.get("messages") == [{"role": "user", "content": "hi"}]

    @staticmethod
    def raw_records(path):
        if path.suffix == ".jsonl":
            with path.open("rb") as stream:
                line = 0
                while True:
                    offset = stream.tell()
                    raw = stream.readline(MAX_LINE + 1)
                    if not raw:
                        break
                    if len(raw) > MAX_LINE:
                        raise ValueError("One recorded call exceeds the viewer size limit")
                    try:
                        value = json.loads(raw)
                        if isinstance(value, dict):
                            yield line, offset, value
                    except ValueError:
                        pass  # A concurrent writer may not have finished the last line.
                    line += 1
        else:
            if path.stat().st_size > MAX_LINE:
                raise ValueError("Recorded call exceeds the viewer size limit")
            if path.name == "response.json":
                metadata = read(path.parent / "request.json", {})
                request = read(path.parent / "effective_request.json", {})
                if not request:
                    request = {"messages": [{"role": "system", "content": metadata.get("system", "")},
                                            {"role": "user", "content": metadata.get("user", "")}],
                               **{k: v for k, v in metadata.items() if k in REQUEST_FIELDS}}
                yield 0, 0, {"request": request, "response": read(path, {}), "stage": metadata.get("stage"),
                             "status": "recorded"}
            else:
                value = read(path)
                if isinstance(value, dict):
                    yield 0, 0, value

    def source(self, identity, source_id):
        _, root = self.job(identity)
        source = self.sources(identity).get(source_id)
        if source is None:
            raise KeyError("Unknown recorded source")
        path = (root / source["path"]).resolve()
        if not path.is_relative_to(root):
            raise KeyError("Source is outside the recorded experiment")
        return source, path

    def bridge(self, identities):
        wanted = set(identities) - {None, ""}
        with self.lock:
            missing = wanted - self.bridge_cache.keys()
        if missing:
            path = self.cohort / "serving/requests.jsonl"
            try:
                with path.open() as stream:
                    for line in stream:
                        value = json.loads(line)
                        if value.get("id") in missing:
                            with self.lock:
                                self.bridge_cache[value["id"]] = value
                            missing.remove(value["id"])
                            if not missing:
                                break
            except (OSError, ValueError):
                pass
        with self.lock:
            result = {key: self.bridge_cache[key] for key in wanted if key in self.bridge_cache}
            while len(self.bridge_cache) > 4096:
                self.bridge_cache.popitem(last=False)
            return result

    @staticmethod
    def with_observation(value, observation):
        usage = dict(observation.get("usage") or value["usage"])
        choices = observation.get("choices") or []
        details = usage.get("completion_tokens_details") or {}
        reason = sum(c["reasoning_tokens"] for c in choices) if choices else details.get("reasoning_tokens")
        visible = sum(c["visible_tokens"] for c in choices) if choices else (
            usage["completion_tokens"] - reason if reason is not None and usage.get("completion_tokens") is not None
            else None
        )
        return {**value, "usage": usage, "reasoning_tokens": reason, "visible_tokens": visible,
                "observed": {k: observation[k] for k in (
                    "temperature", "reasoning_effort", "enable_thinking", "visible_limit", "reasoning_limit",
                    "role", "profile_sha256", "seconds", "choices",
                ) if k in observation}}

    def records(self, identity, source_id):
        source, path = self.source(identity, source_id)
        signature = (path.stat().st_size, path.stat().st_mtime_ns)
        key = (identity, source_id)
        with self.lock:
            cached = self.record_cache.get(key)
            if cached and cached[0] == signature:
                self.record_cache.move_to_end(key)
                return cached[1]
        items = []
        for line, offset, raw in self.raw_records(path):
            if self.health(raw):
                continue
            value = normalise(raw)
            reasoning = "\n\n".join(str(m["reasoning"]) for m in value["messages"])
            items.append({k: value[k] for k in (
                "request_id", "timestamp", "module", "status", "usage", "elapsed_seconds", "error",
            )} | {"line": line, "offset": offset, "preview": reasoning[:170],
                  "has_reasoning": bool(reasoning), "temperature": value["request"].get("temperature"),
                  "finish_reason": value["messages"][0]["finish_reason"] if value["messages"] else None})
        observations = self.bridge(v["request_id"] for v in items)
        items = [self.with_observation(v, observations.get(v["request_id"], {})) for v in items]
        result = {"source": source, "items": items}
        with self.lock:
            self.record_cache[key] = (signature, result)
            while len(self.record_cache) > 64:
                self.record_cache.popitem(last=False)
        return result

    def detail(self, identity, source_id, line):
        source, path = self.source(identity, source_id)
        metadata = next((v for v in self.records(identity, source_id)["items"] if v["line"] == line), None)
        if metadata is None:
            raise KeyError("Recorded call not found")
        if path.suffix == ".jsonl":
            with path.open("rb") as stream:
                stream.seek(metadata["offset"])
                raw = json.loads(stream.readline(MAX_LINE + 1))
        else:
            raw = next(self.raw_records(path))[2]
        value = normalise(raw)
        observation = self.bridge([value["request_id"]]).get(value["request_id"], {})
        return {"source": source, "line": line, **self.with_observation(value, observation),
                "provenance": {"relative_path": source["path"], "byte_offset": metadata["offset"],
                               "cohort": self.cohort.name},
                "missing_fields": "필드가 없으면 미기록으로 표시합니다. Reasoning 본문은 저장된 모델 응답입니다."}

    def catalog(self, query):
        identity = query.get("job", [""])[0]
        kind, split = query.get("kind", [""])[0], query.get("split", [""])[0]
        text = query.get("q", [""])[0].casefold().strip()
        page = max(1, min(10000, int(query.get("page", ["1"])[0])))
        rows = [v for v in self.sources(identity).values()
                if (not kind or v["kind"] == kind) and (not split or v["split"] == split)
                and (not text or text in " ".join(str(v[k]) for k in (
                    "task_id", "label", "stage", "path",
                )).casefold())]
        rows.sort(key=lambda v: v["modified_at"], reverse=True)
        return {"total": len(rows), "page": page, "page_size": PAGE_SIZE,
                "items": rows[(page - 1) * PAGE_SIZE:page * PAGE_SIZE]}
