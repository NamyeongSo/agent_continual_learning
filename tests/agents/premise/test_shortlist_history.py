# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""PREMiSE tool selection uses ACE's bounded recent history."""

from __future__ import annotations

from exgentic.agents.premise.instance import PremiseAgentInstance


def test_shortlist_uses_recent_history_without_changing_agent_messages() -> None:
    agent = object.__new__(PremiseAgentInstance)
    agent.messages = [
        {"role": "user", "content": "old-marker " + "x" * 9000},
        {"role": "tool", "content": "recent-result"},
    ]

    history = agent._render_history_for_shortlist()

    assert len(history) == 8000
    assert "old-marker" not in history
    assert history.endswith("tool: recent-result")
    assert agent.messages[0]["content"].startswith("old-marker")
