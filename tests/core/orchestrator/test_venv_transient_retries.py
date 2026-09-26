# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Transport and startup failures should be retried as whole sessions."""

import httpx

from exgentic.core.orchestrator.execution import _is_transient_error


def test_venv_startup_and_disconnects_are_transient():
    assert _is_transient_error(RuntimeError("Venv service failed to start on port 40000"))
    assert _is_transient_error(TimeoutError("Venv service did not become healthy"))
    assert _is_transient_error(httpx.RemoteProtocolError("Server disconnected"))
    assert _is_transient_error(httpx.ReadTimeout("timed out"))


def test_unrelated_error_is_not_retried():
    assert not _is_transient_error(ValueError("invalid benchmark configuration"))
