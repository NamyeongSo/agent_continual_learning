# SPDX-License-Identifier: Apache-2.0
"""Prevent premature execution, thinking-data reuse, and hidden T1 overrides."""

from types import SimpleNamespace

import pytest

from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.core.types import ModelSettings
from scripts.ace import run_context_proposal_bottleneck as base
from scripts.ace import schedule_nonthinking_replication as study


def settings():
    return study.nonthinking_settings({"temperature": 1.0, "reasoning_effort": "xhigh", "max_tokens": None,
                                      "timeout": 600, "num_retries": 1})


def assert_request(request):
    assert request["temperature"] == 0.7
    assert request["top_p"] == 0.8
    assert request["presence_penalty"] == 1.5
    extra = request["extra_body"]
    assert {k: extra[k] for k in ("top_k", "min_p", "repetition_penalty")} == {
        "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
    }
    assert extra["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": False}
    assert "reasoning_effort" not in request


def test_actor_and_analysis_requests_use_nonthinking_sampling(monkeypatch):
    from exgentic.agents.ace import ace_instance

    monkeypatch.delenv("EXGENTIC_INFERENCE_PROFILE", raising=False)
    monkeypatch.delenv("EXGENTIC_INFERENCE_PROFILE_SHA256", raising=False)
    protocol = {"model_settings": settings()}
    instance = ACEAgentInstance("sampling", model="mock-model", model_settings=ModelSettings(**settings()),
                                **base.role_call_overrides(protocol))
    requests = []
    response = SimpleNamespace(usage=None, choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content="{}"))])
    monkeypatch.setattr(ace_instance, "actor_completion", lambda _, request: requests.append(request) or response)
    monkeypatch.setattr(ace_instance.litellm, "completion", lambda **request: requests.append(request) or response)
    instance._completion(model="mock-model", messages=[{"role": "user", "content": "task"}], temperature=1.0)
    assert instance._llm_call_simple("mock-model", "reflection/curator") == "{}"
    assert len(requests) == 2
    for request in requests:
        assert_request(request)


def test_actor_job_passes_nonthinking_overrides_to_actual_agent_factory(tmp_path, monkeypatch):
    from exgentic.interfaces.lib import api

    class CapturedError(Exception):
        pass

    seen = []

    def execute(*, config):
        seen.append(config.agent_kwargs)
        raise CapturedError

    monkeypatch.setattr(api, "execute", execute)
    protocol = {"model_settings": settings(), "initial_playbook": "", "model": "mock-model",
                "actor_context": "shared-empty-v1", "benchmark_kwargs": {},
                "playbook_edit_mode": "add_update_delete"}
    with pytest.raises(CapturedError):
        base.actor_job(tmp_path, protocol, {"stage": 1, "task_id": "task"}, tmp_path)
    assert seen[0]["enable_thinking"] is False
    assert seen[0]["playbook_edit_mode"] == "add_update_delete"
    assert seen[0]["temperature_override"] == seen[0]["analysis_temperature_override"] == 0.7


def test_legacy_settings_still_pin_thinking_and_temperature_one():
    assert base.role_call_overrides({"model_settings": {}}) == {
        "enable_thinking": True, "temperature_override": 1.0, "analysis_temperature_override": 1.0,
    }


def test_fresh_reflection_copies_only_inputs_and_frozen_core(tmp_path):
    source, output = tmp_path / "thinking", tmp_path / "nonthinking"
    policy = "teammate policy"
    base.write(source / "protocol.json", {
        "stage3_repeats_per_context": 3, "task_ids": [str(i) for i in range(50)],
        "turn_limit_scoring": "official_pct_at_limit", "appworld_policy_sha256": base.digest(policy),
        "model_settings": {"temperature": 1.0, "reasoning_effort": "xhigh"},
        "source_initial_bank": "old thinking bank", "completed_proposals_reused": 187,
        "stages": [{"total_jobs": n} for n in (250, 500, 1500)],
    })
    for name in ("references.json", "task_splits.json", "benchmark_dependency_hashes.json"):
        base.write(source / name, {})
    for name, content in (("appworld_policy.txt", policy), ("initial_playbook.txt", "empty"),
                          ("source_snapshot/src/frozen.py", "unchanged baseline core"),
                          ("source_snapshot/scripts/ace/run_context_proposal_bottleneck.py", "old runner"),
                          ("benchmark_dependencies/overlay.txt", "pinned dependencies")):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    base.write(source / "source_hashes.json", {"src/frozen.py": base.digest("unchanged baseline core")})
    base.write(source / "stage1/task_r0/result.json", {"pct": 1})
    base.write(source / "stage2/with_gt_task_r0/result.json", {"reflection": "thinking reflection"})
    study.freeze_reflection(source, output)
    protocol = base.read(output / "protocol.json")
    assert "source_initial_bank" not in protocol and "completed_proposals_reused" not in protocol
    assert protocol["initial_trajectories_reused"] == protocol["proposals_reused"] == 0
    assert not (output / "stage1").exists() and not (output / "stage2").exists()
    assert (output / "source_snapshot/src/frozen.py").read_bytes() == (
        source / "source_snapshot/src/frozen.py").read_bytes()
    assert [len(base.read(output / f"stage{s}_queue.json")) for s in (1, 2, 3)] == [250, 500, 1500]
    assert len({j["source_id"] for j in base.read(output / "stage2_queue.json")}) == 250
    study.verify_sources(output)
    with pytest.raises(FileExistsError):
        study.freeze_reflection(source, output)


def test_completion_barrier_requires_every_seed_and_all_2700_distinct_results(tmp_path):
    base.write(tmp_path / "status.json", {"state": "running"})
    assert not study.predecessor_ready(tmp_path)
    base.write(tmp_path / "protocol.json", {"task_orders": {
        str(seed): {"train": list(range(50)), "val": list(range(50))} for seed in (42, 43, 44)
    }})
    base.write(tmp_path / "status.json", {"state": "completed"})
    with pytest.raises(ValueError, match="seed42"):
        study.predecessor_ready(tmp_path)
    for seed in (42, 43, 44):
        base.write(tmp_path / f"seed{seed}/completed.json", {"evaluations": 900})
        rows = [{"condition_id": c["id"], "task_id": t}
                for c in study.accumulation.conditions(seed) for t in range(50)]
        for phase in ("train_final", "bwt_checkpoint", "val"):
            base.write(tmp_path / f"seed{seed}/{phase}_results.json", rows)
    assert study.predecessor_ready(tmp_path)
    base.write(tmp_path / "seed44/val_results.json", [rows[0]] * 300)
    with pytest.raises(ValueError, match="seed44/val"):
        study.predecessor_ready(tmp_path)


def test_scheduler_waiting_makes_no_model_calls_or_child_launches(tmp_path, monkeypatch):
    predecessor = tmp_path / "predecessor"
    base.write(predecessor / "status.json", {"state": "running"})
    output = tmp_path / "scheduled"
    base.write(output / "plan.json", {"predecessor": str(predecessor), "scheduler_source_hashes": {}})
    monkeypatch.setattr(study.time, "sleep", lambda _: (output / "STOP").touch())
    monkeypatch.setattr(study.base, "check_endpoint", lambda _: pytest.fail("Premature endpoint access"))
    monkeypatch.setattr(study.subprocess, "Popen", lambda *a, **k: pytest.fail("Premature child launch"))
    with pytest.raises(InterruptedError):
        study.run(output)
    assert base.read(output / "status.json")["phase"] == "waiting_for_thinking_completion"
