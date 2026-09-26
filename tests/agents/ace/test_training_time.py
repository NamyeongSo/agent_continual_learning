# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.agents.ace.playbook_store import PlaybookStore
from exgentic.benchmarks.terminalbench2.terminalbench2_benchmark import RunTerminus2Action
from exgentic.core.types import ActionType, SingleObservation


def test_training_time_reflector_receives_official_grading(monkeypatch):
    instance = ACEAgentInstance(session_id="session-1", training_time=True)
    instance._store = PlaybookStore(store_id="test", initial_playbook="## STRATEGIES")
    instance.task = "Fix the bug"
    instance._action_log.append({"step": 1, "action": "finish"})
    instance.receive_training_feedback({
        "score": 0.0,
        "success": True,
        "session_metadata": {"checker_result": {"valid": False}},
    })

    prompts = []

    def fake_llm_call(model, prompt, json_mode):
        prompts.append(prompt)
        return '{"reasoning": "The checker rejected it", "bullet_tags": []}'

    monkeypatch.setattr(instance, "_llm_call_simple", fake_llm_call)
    assert instance._run_post_session_reflection() == "The checker rejected it"
    assert "Official Post-Session Grading Feedback" in prompts[0]
    assert '"score": 0.0' in prompts[0]
    assert '"valid": false' in prompts[0]
    assert "A successful submission is not necessarily a full score" in prompts[0]


def test_default_reflector_does_not_receive_grading(monkeypatch):
    instance = ACEAgentInstance(session_id="session-2")
    instance._store = PlaybookStore(store_id="test", initial_playbook="## STRATEGIES")
    instance.task = "Fix the bug"
    instance.receive_training_feedback({"score": 1.0, "success": True})

    prompts = []
    monkeypatch.setattr(
        instance,
        "_llm_call_simple",
        lambda model, prompt, json_mode: prompts.append(prompt) or '{"reasoning": "ok"}',
    )
    instance._run_post_session_reflection()
    assert "Official Post-Session Grading Feedback" not in prompts[0]
    assert instance._training_feedback is None


def test_terminus2_runs_with_playbook_and_reflects_on_trajectory(monkeypatch):
    PlaybookStore.reset_all()
    instance = ACEAgentInstance(
        session_id="terminal-session", model="openai/test-model",
        benchmark_id="terminalbench2", training_time=True,
    )
    instance.start(
        task="Solve the terminal task",
        context={"task_id": "demo"},
        actions=[ActionType(name="run_terminus2", description="Run Terminus-2", cls=RunTerminus2Action)],
    )
    action = instance.react(SingleObservation(invoking_actions=[], result=None))
    assert action.name == "run_terminus2"
    assert action.arguments.model == "openai/test-model"
    assert action.arguments.playbook == instance._store.playbook
    assert instance._step_count == 1

    instance.receive_training_feedback({
        "score": 1.0,
        "success": True,
        "session_metadata": {
            "agent_trace": "terminal command: touch /answer.txt",
            "agent_usage": {"input_tokens": 10, "output_tokens": 5},
        },
    })
    prompts = []
    monkeypatch.setattr(
        instance, "_llm_call_simple",
        lambda model, prompt, json_mode: prompts.append(prompt) or '{"reasoning": "ok", "bullet_tags": []}',
    )
    instance._run_post_session_reflection()
    assert "terminal command: touch /answer.txt" in prompts[0]
    assert prompts[0].count("terminal command: touch /answer.txt") == 1
    assert '"score": 1.0' in prompts[0]
    assert "keep each JSON string field under 100 words" in prompts[0]
    monkeypatch.setattr(
        instance, "_llm_call_simple",
        lambda model, prompt, json_mode: prompts.append(prompt) or '{"reasoning": "ok", "operations": []}',
    )
    instance._run_curator("A transferable lesson")
    assert "return at most three concise ADD operations" in prompts[1]
    PlaybookStore.reset_all()
