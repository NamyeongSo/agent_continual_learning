# SPDX-License-Identifier: Apache-2.0
"""ACE must preserve the pinned sampler for acting and both learning roles."""

import json
from types import SimpleNamespace

import pytest

from exgentic.actor_context import DEFAULT_ACTOR_CONTEXT
from exgentic.agents.ace.ace_agent import ACEAgent
from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.core.types import ModelSettings


def test_shared_thinking_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="Shared actor context"):
        ACEAgentInstance(session_id="test", actor_context=DEFAULT_ACTOR_CONTEXT, enable_thinking=True)


def test_pinned_sampler_reaches_actor_reflector_and_curator(monkeypatch):
    monkeypatch.delenv("EXGENTIC_INFERENCE_PROFILE", raising=False)
    settings = ModelSettings(
        temperature=1, top_p=.95, presence_penalty=0, max_tokens=None,
        reasoning_effort="xhigh", timeout=600,
        extra_body={"top_k": 20, "min_p": 0, "repetition_penalty": 1,
                    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}},
    )
    agent = ACEAgent(
        model="openai/test", actor_context=DEFAULT_ACTOR_CONTEXT, enable_thinking=True,
        allow_shared_thinking=True, model_settings=settings, temperature_override=1,
        analysis_temperature_override=1,
    )
    instance = ACEAgentInstance(**agent._get_instance_kwargs("sampler-test"))
    requests = []
    response = SimpleNamespace(usage=None, choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content="{}"),
    )])
    monkeypatch.setattr("litellm.completion", lambda **kwargs: requests.append(kwargs) or response)
    instance._completion(model="openai/test", messages=[])
    for stage in ("reflector", "curator"):
        instance._active_llm_stage = stage
        instance._llm_call_simple("openai/test", "Learn from the task", json_mode=True)
    assert len(requests) == 3
    for request in requests:
        for key in ("temperature", "top_p", "presence_penalty", "reasoning_effort", "timeout", "extra_body"):
            assert request[key] == getattr(settings, key)
        assert request.get("max_tokens") is None


def test_cli_pins_live_settings_and_refuses_sampler_changes_on_resume(tmp_path, monkeypatch):
    from scripts.ace import run_experiment as runner

    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"benchmarks": {"bfcl": {"train": ["task-1"], "val": [], "test": []}}}))
    settings = tmp_path / "model.json"
    settings.write_text(json.dumps({"temperature": .5, "top_p": .95, "reasoning_effort": "xhigh"}))
    argv = ["ace", "--mode", "isolated", "--seed", "42", "--benchmarks", "bfcl",
            "--output-dir", str(tmp_path / "run"), "--task-split-manifest", str(manifest),
            "--all-train-tasks", "--dry-run", "--enable-thinking", "--model-settings-file", str(settings),
            "--temperature", "1"]
    monkeypatch.setattr("sys.argv", argv)
    runner.main()
    config = json.loads((tmp_path / "run" / "experiment_config.json").read_text())
    assert config["effective_enable_thinking"] is True
    assert config["model_settings"]["temperature"] == 1
    assert config["model_settings"]["top_p"] == .95
    assert config["model_settings_file_sha256"]
    settings.write_text(json.dumps({"temperature": .5, "top_p": .8, "reasoning_effort": "xhigh"}))
    with pytest.raises(ValueError, match="Existing model settings differ"):
        runner.main()
