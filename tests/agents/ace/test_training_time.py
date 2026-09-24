from __future__ import annotations

from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.agents.ace.playbook_store import PlaybookStore


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
