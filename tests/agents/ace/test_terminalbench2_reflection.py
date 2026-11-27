# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Terminal-Bench reflection preserves the trace and uses the common ACE prompt."""

import json
import logging

import pytest

from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.agents.ace.playbook_store import PlaybookStore
from exgentic.agents.ace.prompts.reflector import REFLECTOR_PROMPT_NO_GT, REFLECTOR_PROMPT_TRAINING_TIME


@pytest.mark.parametrize("training_time", [False, True])
def test_terminalbench_reflection_receives_full_trace_and_common_prompt(training_time):
    trace = "BEGIN " + "long terminal observation\n" * 5000 + " END"
    instance = object.__new__(ACEAgentInstance)
    instance._store = PlaybookStore("terminal-reflection-test", "## playbook")
    instance._action_log = []
    instance._observation_log = []
    instance._training_feedback = {"score": 0.0, "session_metadata": {"agent_trace": trace}}
    instance.training_time = training_time
    instance.benchmark_id = "terminalbench2"
    instance.model = "openai/test"
    instance.task = "Solve the terminal task"
    instance.use_json_mode = True
    instance._logger = logging.getLogger(__name__)
    captured = []

    def fake_llm(_model, prompt, **_kwargs):
        captured.append(prompt)
        return '{"reasoning": "Inspect the complete command output", "bullet_tags": []}'

    instance._llm_call_simple = fake_llm
    assert instance._run_post_session_reflection() == "Inspect the complete command output"
    prompt_kwargs = {
        "question": instance.task,
        "reasoning_trace": "Terminus-2 trajectory:\n" + trace,
        "predicted_answer": "(see session trace above)",
        "bullets_used": "## playbook",
    }
    if training_time:
        expected = REFLECTOR_PROMPT_TRAINING_TIME.format(
            **prompt_kwargs, grading_feedback=json.dumps({"score": 0.0, "session_metadata": {}}),
        )
    else:
        expected = REFLECTOR_PROMPT_NO_GT.format(**prompt_kwargs)
    assert captured == [expected]
    assert trace in captured[0]
