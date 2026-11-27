# SPDX-License-Identifier: Apache-2.0
"""Opt-in, hash-pinned inference settings shared by core and optimizer venvs."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=16)
def _read(path: str, expected: str):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected and digest != expected:
        raise ValueError("Inference profile bytes changed")
    value = json.loads(data)
    if value["reasoning_effort"] != "medium" or value["actor_temperature"] != 0.0:
        raise ValueError("Unexpected actor inference profile")
    if value["analysis_temperature"] != 0.7 or value["reasoning_tokens"] < 1:
        raise ValueError("Unexpected analysis inference profile")
    return {**value, "sha256": digest, "source": path}


def profile() -> dict | None:
    path = os.environ.get("EXGENTIC_INFERENCE_PROFILE")
    return _read(path, os.environ.get("EXGENTIC_INFERENCE_PROFILE_SHA256", "")) if path else None


def apply_request(kwargs: dict, role: str) -> dict:
    active = profile()
    if active is None:
        return kwargs
    result = dict(kwargs)
    result["temperature"] = active["actor_temperature" if role == "actor" else "analysis_temperature"]
    result["reasoning_effort"] = active["reasoning_effort"]
    visible = 2048 if role == "actor" else 8192
    result["max_tokens"] = min(
        result.pop("max_completion_tokens", None) or result.get("max_tokens") or visible, visible
    )
    body = dict(result.get("extra_body") or {})
    body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), "enable_thinking": True}
    body["exgentic_reasoning_budget"] = active["reasoning_tokens"]
    body["exgentic_role"] = role
    body["exgentic_profile_sha256"] = active["sha256"]
    result["extra_body"] = body
    return result


def reasoning_tokens(usage) -> int:
    details = (
        usage.get("completion_tokens_details")
        if isinstance(usage, dict)
        else getattr(usage, "completion_tokens_details", None)
    )
    value = details.get("reasoning_tokens") if isinstance(details, dict) else getattr(details, "reasoning_tokens", None)
    if value is None:
        raise ValueError("Missing measured reasoning token usage")
    return int(value)
