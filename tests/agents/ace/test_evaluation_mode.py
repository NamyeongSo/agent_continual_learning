# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from exgentic.agents.ace.ace_agent import ACEAgent
from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.agents.ace.playbook_store import PlaybookStore


def test_evaluation_mode_is_forwarded_to_instance():
    agent = ACEAgent(model="openai/test", evaluation_mode=True,
                     initial_playbook="## frozen")
    kwargs = agent._get_instance_kwargs("val-session")
    assert kwargs["evaluation_mode"] is True
    assert kwargs["initial_playbook"] == "## frozen"


def test_evaluation_close_does_not_update_frozen_playbook():
    store = PlaybookStore("ace_sequential_global", "## frozen")
    instance = object.__new__(ACEAgentInstance)
    instance._store = store
    instance.evaluation_mode = True
    instance.messages = []
    instance._log_bullet_usage = lambda _ids: None
    instance._run_post_session_reflection = lambda: (_ for _ in ()).throw(AssertionError("reflection ran"))
    instance._run_curator = lambda _text: (_ for _ in ()).throw(AssertionError("curator ran"))
    instance.close()
    assert store.playbook == "## frozen"
    assert store.session_count == 0
