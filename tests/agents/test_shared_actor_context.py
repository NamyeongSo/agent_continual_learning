# SPDX-License-Identifier: Apache-2.0
"""Compare the actual first actor messages/tools across all four methods."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from exgentic.actor_context import (
    DEFAULT_ACTOR_CONTEXT,
    EMPTY_PLAYBOOK,
    actor_context_metadata,
    require_matching_actor_context,
    shared_actor_prompt,
    terminal_actor_instruction,
)
from exgentic.agents.ace.ace_agent import ACEAgent
from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK, PlaybookStore
from exgentic.agents.premise.agent import PremiseAgent
from exgentic.agents.premise.instance import PremiseAgentInstance
from exgentic.benchmarks.terminalbench2.terminalbench2_benchmark import RunTerminus2Action
from exgentic.core.types import ActionType, EmptyObservation, SingleAction, SingleObservation
from scripts.baselines import actor
from scripts.baselines.runtime import actor_prompt, initial_instruction

MODE = DEFAULT_ACTOR_CONTEXT
METHODS = ("ace", "gepa", "skillopt", "premise")


class InspectArgs(BaseModel):
    name: str


class InspectAction(SingleAction):
    name: str = "inspect"
    arguments: InspectArgs


class RequestCapturedError(Exception):
    pass


@pytest.fixture(autouse=True)
def no_model_calls(monkeypatch):
    monkeypatch.setattr(
        "exgentic.agents.litellm_tool_calling.instance.check_model_accessible_sync", lambda *args, **kwargs: None
    )
    PlaybookStore.reset_all()
    yield
    PlaybookStore.reset_all()


def instance(method, benchmark, tmp_path, monkeypatch, learned=None):
    if method == "ace":
        agent = ACEAgent(
            model="openai/test",
            actor_context=MODE,
            benchmark_id=benchmark,
            initial_playbook=learned,
            enable_tool_shortlisting=True,
        )
        return ACEAgentInstance(**agent._get_instance_kwargs("shared-test"))
    if method == "premise":
        agent = PremiseAgent(
            model="openai/test",
            actor_context=MODE,
            actor_benchmark=benchmark,
            playbook=DEFAULT_PLAYBOOK if learned is None else learned,
            enable_tool_shortlisting=True,
        )
    else:
        captured = []

        def execute(*, config):
            captured.append(config)
            return SimpleNamespace(
                session_results=[
                    SimpleNamespace(
                        status=SimpleNamespace(value="successful"),
                        score=1.0,
                        success=True,
                        details={},
                    )
                ]
            )

        monkeypatch.setattr("exgentic.interfaces.lib.api.execute", execute)
        monkeypatch.setattr("exgentic.agents.premise.pipeline.trajectory_from_session", lambda *args: {})
        policy = initial_instruction(method, MODE) if learned is None else learned
        actor.run(
            {
                "method": method,
                "benchmark": benchmark,
                "model": "openai/test",
                "benchmark_kwargs": {},
                "item": {"task_id": "same-task"},
                "session_output": str(tmp_path),
                "actor_context": MODE,
                "learned_policy": policy,
                "system_prompt": actor_prompt(method, benchmark, policy, MODE),
            }
        )
        agent = PremiseAgent(**captured[0].agent_kwargs)
    return PremiseAgentInstance(**agent._get_instance_kwargs("shared-test"))


def test_opted_in_live_settings_survive_agent_construction_and_native_dispatch():
    settings = {"temperature": 1, "max_tokens": None, "reasoning_effort": "xhigh", "top_p": .95,
                "extra_body": {"top_k": 20, "chat_template_kwargs": {
                    "enable_thinking": True, "preserve_thinking": True,
                }}}
    agent = PremiseAgent(model="openai/test", actor_context=MODE, actor_benchmark="terminalbench2",
                         enable_thinking=True, allow_shared_thinking=True, native_model_settings=settings)
    actor_instance = PremiseAgentInstance(**agent._get_instance_kwargs("live-native-test"))
    actor_instance.start("Complete task", {}, [ActionType(
        name="run_terminus2", description="Run native terminal agent", cls=RunTerminus2Action,
    )])
    action = actor_instance.react(EmptyObservation())
    assert actor_instance.enable_thinking is True
    assert action.arguments.model_settings == settings


@pytest.mark.parametrize("benchmark", ["bfcl", "appworld"])
@pytest.mark.parametrize(
    "observation",
    [
        EmptyObservation(),
        SingleObservation(result={"status": "ready"}, invoking_actions=[]),
        SingleObservation(
            result={"screenshot": {"type": "image_url", "data": "data:image/png;base64,AA=="}}, invoking_actions=[]
        ),
    ],
)
def test_first_messages_tools_and_tool_choice_match(benchmark, observation, tmp_path, monkeypatch):
    requests = []
    actions = [ActionType(name="inspect", description="Inspect current state", cls=InspectAction)]
    context = {"policy": "Use the provided execution interface.", "datetime": "2026-01-01"}
    for method in METHODS:
        agent = instance(method, benchmark, tmp_path, monkeypatch)
        agent.start("Complete the same task", deepcopy(context), actions)

        def capture(**kwargs):
            requests.append(deepcopy({k: kwargs.get(k) for k in ("messages", "tools", "tool_choice")}))
            raise RequestCapturedError

        monkeypatch.setattr(agent, "_completion", capture)
        with pytest.raises(RequestCapturedError):
            agent.react(deepcopy(observation))
        assert agent.enable_tool_shortlisting is False
    assert len(requests) == 4
    assert all(request == requests[0] for request in requests)
    assert requests[0]["messages"][0] == {"role": "system", "content": shared_actor_prompt(benchmark)}
    assert requests[0]["tools"]


def test_native_terminal_instruction_matches_through_real_actor_actions(tmp_path, monkeypatch):
    instructions = []
    for method in METHODS:
        agent = instance(method, "terminalbench2", tmp_path, monkeypatch)
        agent.start(
            "Solve the terminal task",
            {},
            [
                ActionType(name="run_terminus2", description="Native terminal", cls=RunTerminus2Action),
            ],
        )
        action = agent.react(EmptyObservation())
        assert action.arguments.actor_context == MODE
        instructions.append(
            terminal_actor_instruction(
                "Solve the terminal task",
                action.arguments.playbook,
                action.arguments.actor_context,
            )
        )
    assert len(set(instructions)) == 1
    assert instructions[0] == "Solve the terminal task\n\n" + shared_actor_prompt("terminalbench2")


@pytest.mark.parametrize("method", METHODS)
def test_nonempty_learned_policy_is_visible_without_ace_bullet_format(method, tmp_path, monkeypatch):
    learned = "A newly learned strategy with no bullet ID."
    agent = instance(method, "bfcl", tmp_path, monkeypatch, learned)
    agent.start("Same task", {}, [ActionType(name="inspect", description="Inspect", cls=InspectAction)])
    assert agent.messages[0]["content"] == shared_actor_prompt("bfcl", learned)
    assert agent.messages[0]["content"].count(learned) == 1


def test_empty_initial_policy_and_resume_guards_preserve_historical_protocol():
    assert DEFAULT_PLAYBOOK == EMPTY_PLAYBOOK
    assert initial_instruction("gepa", MODE) == initial_instruction("skillopt", MODE) == ""
    assert initial_instruction("gepa") != initial_instruction("skillopt")
    assert shared_actor_prompt("bfcl", DEFAULT_PLAYBOOK) == shared_actor_prompt("bfcl", "")
    assert "does not provide a Python REPL" not in shared_actor_prompt("appworld")
    with pytest.raises(ValueError, match="fresh output"):
        require_matching_actor_context({}, MODE, ["bfcl"])
    metadata = actor_context_metadata(MODE, ["bfcl"])
    saved = {"actor_context": MODE, "actor_context_metadata": metadata}
    require_matching_actor_context(saved, MODE, ["bfcl"])
    with pytest.raises(ValueError, match="hashes differ"):
        require_matching_actor_context(saved, MODE, ["appworld"])
    require_matching_actor_context({}, "legacy", ["bfcl"])


def test_shared_context_rejects_an_additional_method_specific_system_override():
    with pytest.raises(ValueError, match="system_prompt_override"):
        PremiseAgentInstance(
            session_id="bad", model="openai/test", actor_context=MODE, system_prompt_override="Extra instructions"
        )


@pytest.mark.parametrize("benchmark", ["bfcl", "appworld", "terminalbench2"])
def test_policy_shift_keeps_shared_context_and_saved_history(benchmark, tmp_path):
    from exgentic.agents.premise.span_policy_shift import SpanPolicyShiftEvaluator, replace_policy

    original = shared_actor_prompt(benchmark)
    content = "Task instruction\n\n" + original if benchmark == "terminalbench2" else original
    row = {
        "action_index": 0,
        "request": {
            "messages": [
                {"role": "user" if benchmark == "terminalbench2" else "system", "content": content},
                {"role": "user", "content": "Preserve this saved history exactly."},
            ]
        },
    }
    history = tmp_path / "history.jsonl"
    history.write_text(json.dumps(row) + "\n")
    source = {
        "benchmark": benchmark,
        "success": True,
        "actions": [{}],
        "task_id": "same-task",
        "policy_shift_history": str(history),
        "actor_policy": original,
        "actor_context": MODE,
    }
    evaluator = SpanPolicyShiftEvaluator(model="openai/test", base_url="http://unused/v1", output=tmp_path)
    candidate = {"playbook": {"playbook": "A newly learned rule."}}
    span = evaluator._prepare("candidate", candidate, source, {"span": {"start": 0, "end": 1}}, "test", {})
    request = replace_policy(row["request"], span.original, span.replacement)
    assert "A newly learned rule." in request["messages"][0]["content"]
    assert "No learned guidance yet" not in request["messages"][0]["content"]
    assert request["messages"][1:] == row["request"]["messages"][1:]
    assert row["request"]["messages"][0]["content"] == content
