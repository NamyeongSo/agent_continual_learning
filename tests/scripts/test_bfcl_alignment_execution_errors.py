# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Failure classes must distinguish actor/protocol errors from scored benchmark failures."""

import pytest

from scripts.premise.audit_bfcl_alignment_execution_errors import compact_error, error_category, pearson


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("LiteLLM completion missing assistant content.", "missing_assistant_content"),
        ("InternalServerError: unexpected tokens remaining in message header", "harmony_header_protocol"),
        ("InternalServerError: Unknown role: assistant<|channel|>commentary", "harmony_header_protocol"),
        ("Venv service failed to start on port 1200", "isolated_environment_startup"),
        ("Output length failure after 3 attempts", "actor_output_length"),
        ("litellm.Timeout: APITimeoutError - Request timed out.", "request_timeout"),
        ("BadGatewayError: Connection refused", "server_transport"),
    ],
)
def test_error_categories(message, expected):
    assert error_category({"status": "error", "error": message}) == expected


def test_scored_failure_is_not_execution_error():
    assert error_category({"status": "completed", "session_status": "limit_reached", "score": 0}) == "none"


def test_nested_error_without_execution():
    assert compact_error({"error": "{'session_metadata': {'error': 'sample'}}"}) == "sample"
    assert compact_error({"error": "__import__('os').system('echo unsafe')"}).startswith("__import__")


def test_undefined_correlation_is_null():
    assert pearson([1, 1, 1], [1, 2, 3]) is None
    assert pearson([1, 2, float("nan")], [1, 2, 3]) is None
    assert pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1)
