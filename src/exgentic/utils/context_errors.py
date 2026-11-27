# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Recognize context capacity errors across local, provider and RPC wrappers."""

from __future__ import annotations


def is_context_length_error(error: BaseException | str) -> bool:
    pending = [error]
    seen = set()
    markers = (
        "contextwindowexceedederror", "contextlengthexceedederror", "context_length_exceeded",
        "maximum context length", "context length exceeded", "context window exceeded",
        "exceeds the context window", "exceed context window", "prompt is too long",
        "input tokens exceeds", "input length exceeds", "context size exceeded",
    )
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        message = f"{type(current).__name__}: {current}".lower()
        if any(marker in message for marker in markers):
            return True
        if isinstance(current, BaseException):
            pending.extend(
                nested for name in ("error", "__cause__", "__context__")
                if isinstance(nested := getattr(current, name, None), BaseException)
            )
    return False
