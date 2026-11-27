# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Background progress rendering should not run in redirected experiment logs."""

from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from exgentic.core.orchestrator.tracker import ConsoleLoggerObserver


def test_redirected_log_does_not_start_a_refresh_thread():
    output = StringIO()
    observer = ConsoleLoggerObserver(console=Console(file=output, force_terminal=False))
    observer._start_progress(SimpleNamespace(task_ids=["task"], num_tasks=1))
    try:
        assert observer._progress.live._refresh_thread is None
        observer._advance_run_progress()
    finally:
        observer._stop_progress()
    assert "Run" in output.getvalue()
