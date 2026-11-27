# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""AppWorld request arguments should omit absent optional parameters."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from exgentic.benchmarks.appworld.appworld_eval import AppWorldSession
from exgentic.core.actions import ActionsHandler
from exgentic.core.types import SingleAction


class SearchArgs(BaseModel):
    query: str
    starred: bool | None = None
    from_email: str | None = None


def test_appworld_omits_null_optional_arguments_before_request() -> None:
    received = []

    def request(app: str, api: str, **kwargs: object) -> dict[str, bool]:
        received.append((app, api, kwargs))
        return {"ok": True}

    session = object.__new__(AppWorldSession)
    session._world = SimpleNamespace(requester=SimpleNamespace(request=request))
    session._tool_name_separator = "."
    session._step_count = 0
    session._logger = logging.getLogger(__name__)

    action = SingleAction(
        name="gmail.show_inbox_threads",
        arguments=SearchArgs(query="receipt", starred=None, from_email=None),
    )
    observation = session.apply_action(action)

    assert received == [("gmail", "show_inbox_threads", {"query": "receipt"})]
    assert observation.result == {"ok": True}
    assert session._step_count == 1


@pytest.mark.parametrize("host_failure", [False, True])
def test_code_generation_executes_python_and_propagates_host_errors(host_failure) -> None:
    executed = []

    def execute(code: str) -> str:
        executed.append(code)
        if host_failure:
            raise RuntimeError("shell host failed")
        return "shell output"

    session = object.__new__(AppWorldSession)
    session._world = SimpleNamespace(execute=execute)
    session._interaction_mode = "code_generation"
    session._registry = ActionsHandler(
        logger=logging.getLogger(__name__),
        warn_on_validation_error=False,
        warn_on_unknown_action=False,
    )
    session._step_count = 0
    session._action_count = 0
    session._max_interactions = 10
    session._done = False
    session._logger = logging.getLogger(__name__)

    actions = session.actions
    assert session.requires_main_thread
    assert [action.name for action in actions] == ["execute_python"]
    assert "code" in actions[0].arguments.model_fields

    code = "result = apis.gmail.search_emails(query='receipt')\nprint(result)"
    action = actions[0].build_action({"code": code})
    if host_failure:
        with pytest.raises(RuntimeError, match="shell host failed"):
            session.step(action)
        assert executed == [code]
        return
    observation = session.step(action)

    assert executed == [code]
    assert observation.result == "shell output"
    assert session._step_count == 1
    assert session._action_count == 1


def test_function_calling_mode_still_exposes_individual_app_api_tools() -> None:
    api_docs = SimpleNamespace(
        function_calling=lambda: [
            {
                "type": "function",
                "function": {
                    "name": "gmail__search_emails",
                    "description": "Search email messages.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
    )
    session = object.__new__(AppWorldSession)
    session._world = SimpleNamespace(task=SimpleNamespace(api_docs=api_docs))
    session._interaction_mode = "function_calling"
    assert not session.requires_main_thread
    session._registry = ActionsHandler(
        logger=logging.getLogger(__name__),
        warn_on_validation_error=False,
        warn_on_unknown_action=False,
    )
    session._tool_name_separator = "."
    session._startup_timings = {}
    session._logger = logging.getLogger(__name__)

    actions = session.actions

    assert [action.name for action in actions] == ["gmail.search_emails"]
