# SPDX-License-Identifier: Apache-2.0
"""Cohort-scoped fallback for one unambiguous Python fence in an ordinary reply.

This module never makes requests, changes a prompt, evaluates code, or invents a
completion call. Install it explicitly around a diagnostic worker, not globally.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

VERSION = "gemma-single-python-fence-v1"
FENCES = re.compile(r"^[ \t]*```(?P<label>[^\r\n]*)(?:\r?\n|\Z)", re.MULTILINE)
CLOSING = re.compile(r"[ \t]*(?:<tool_call\|>|<\|tool_call\|>)?[ \t]*\Z")


def extract_python(content):
    """Return unchanged code and a decision; reject ambiguous or incomplete text."""
    if not isinstance(content, str):
        return None, "no_text_content"
    fences = list(FENCES.finditer(content))
    if len(fences) != 2:
        return None, "no_single_complete_fence"
    opening, closing = fences
    if opening["label"].strip().lower() not in {"", "py", "python"}:
        return None, "non_python_fence"
    if not CLOSING.fullmatch(closing["label"]):
        return None, "invalid_closing_fence"
    code = content[opening.end():closing.start()]
    if not code.strip():
        return None, "empty_code"
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None, "invalid_python_syntax"
    if not tree.body:
        return None, "no_python_statements"
    return code, "repaired"


def repair_response(response, *, step, tool_names):
    """Copy a response for dispatch; the logged raw response stays byte-equivalent."""
    if response is None or not response.choices or set(tool_names) != {"execute_python"}:
        return response, None
    choice = response.choices[0]
    message = choice.message
    if message.tool_calls:
        return response, None
    if choice.finish_reason not in {None, "stop", "tool_calls"}:
        return response, None
    content = message.content
    code, decision = extract_python(content)
    event = {
        "version": VERSION, "at": datetime.now(UTC).isoformat(), "step": step,
        "original_finish_reason": choice.finish_reason, "decision": decision,
        "content_sha256": hashlib.sha256((content or "").encode()).hexdigest()
        if isinstance(content, (str, type(None))) else None,
    }
    if code is None:
        return response, event
    tool_call = {
        "id": f"call_regex_{step}_{event['content_sha256'][:12]}", "type": "function",
        "function": {"name": "execute_python", "arguments": json.dumps({"code": code}, ensure_ascii=False)},
    }
    repaired = response.model_copy(deep=True)
    fields = message.model_dump()
    fields["tool_calls"] = [tool_call]
    repaired.choices[0].message = type(message)(**fields)
    repaired.choices[0].finish_reason = "tool_calls"
    event.update(code=code, tool_call=tool_call)
    return repaired, event


def install(actor_class):
    """Wrap dispatch after the frozen worker has persisted the raw API response."""
    original = actor_class._completion

    def completion(instance, **kwargs):
        response = original(instance, **kwargs)
        names = [tool.get("function", {}).get("name") for tool in kwargs.get("tools") or []]
        repaired, event = repair_response(response, step=instance._step_count, tool_names=names)
        if event is not None:
            path = Path(instance.paths.agent_dir) / "format_repair.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        return repaired

    actor_class._completion = completion
    return original
