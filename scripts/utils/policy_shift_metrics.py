# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Data-only response decoding and conditional action-agreement metrics."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    data = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(data).hexdigest()


def decode_record(value):
    """Accept saved LiteLLM reprs without evaluating arbitrary Python calls."""
    if not isinstance(value, str):
        return value

    def visit(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [visit(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {visit(k): visit(v) for k, v in zip(node.keys, node.values, strict=True)}
        allowed = {"Choices", "Message", "Function", "ChatCompletionMessageToolCall"}
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in allowed
            and not node.args
            and all(k.arg is not None for k in node.keywords)
        ):
            return {k.arg: visit(k.value) for k in node.keywords}
        raise ValueError("Unsupported trace representation")

    return visit(ast.parse(value, mode="eval").body)


def response_message(response):
    choice = decode_record(response["choices"][0])
    if choice.get("finish_reason") == "length":
        raise ValueError("Actor output truncated at its token limit")
    message = decode_record(choice["message"])
    if message.get("role") != "assistant":
        raise ValueError("Missing assistant message")
    if not message.get("tool_calls") and not message.get("content"):
        raise ValueError("Empty assistant response")
    return message


def normalize_calls(message):
    calls = []
    for item in message.get("tool_calls") or []:
        function = decode_record(decode_record(item)["function"])
        args = function["arguments"]
        args = json.loads(args) if isinstance(args, str) else args
        if not isinstance(args, dict) or not isinstance(function["name"], str) or not function["name"]:
            raise ValueError("Invalid function-call action")
        calls.append({"name": function["name"], "arguments": args})
    return calls


def terminal_parser(path):
    """Load the frozen official Harbor parser without importing an environment."""
    name = "_policy_shift_frozen_terminus_parser"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.TerminusJSONPlainParser()


def normalize_terminal(message, parser):
    result = parser.parse_response(message.get("content") or "")
    if result.error:
        raise ValueError(result.error)
    return {
        "commands": [{"keystrokes": c.keystrokes, "duration": float(c.duration)} for c in result.commands],
        "task_complete": bool(result.is_task_complete),
    }


def score_actions(predicted, reference, benchmark):
    if benchmark == "terminalbench2":
        return {"accuracy": float(predicted == reference)}
    scores = {}
    for mode in ("loose", "strict"):
        pred = [canonical(c) if mode == "strict" else c["name"] for c in predicted]
        ref = [canonical(c) if mode == "strict" else c["name"] for c in reference]
        matched = sum((Counter(pred) & Counter(ref)).values())
        precision = matched / len(pred) if pred else float(not ref)
        recall = matched / len(ref) if ref else float(not pred)
        scores[mode + "_accuracy"] = float(pred == ref)
        scores[mode + "_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return scores


def splice_request(anchor, prompt, model, temperature=0.7, max_tokens=2048):
    messages = copy.deepcopy(anchor["messages"])
    span = anchor["policy_span"]
    content = messages[span["message"]]["content"]
    if span.get("part") is not None:
        content = content[span["part"]]["text"]
    if digest(content[span["start"] : span["end"]]) != span["sha256"]:
        raise ValueError("Frozen history policy span changed")
    replacement = content[: span["start"]] + prompt + content[span["end"] :]
    if span.get("part") is not None:
        messages[span["message"]]["content"][span["part"]]["text"] = replacement
    else:
        messages[span["message"]]["content"] = replacement
    body = {
        "model": model.removeprefix("openai/"),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if anchor.get("tools"):
        body.update(tools=copy.deepcopy(anchor["tools"]), tool_choice="auto")
    return body


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)
