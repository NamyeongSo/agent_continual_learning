# SPDX-License-Identifier: Apache-2.0
"""AUD changes accumulation only and cannot start before both ADD-only runs finish."""

import json
from types import SimpleNamespace

import pytest

from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
from scripts.ace import run_reflection_accumulation as accumulation
from scripts.ace import schedule_aud_accumulation as study


@pytest.mark.parametrize("thinking,temperature", [(True, 1.0), (False, 0.7)])
def test_accumulation_curator_actually_applies_update_and_delete(tmp_path, monkeypatch, thinking, temperature):
    condition = accumulation.conditions(42)[0]
    tasks = ["a", "b", "c"]
    accumulation.base.write(tmp_path / "protocol.json", {
        "playbook_edit_mode": "add_update_delete", "initial_playbook": DEFAULT_PLAYBOOK,
        "task_orders": {"42": {"train": tasks}}, "model": "mock-model",
        "model_settings": {"temperature": temperature, "extra_body": {
            "chat_template_kwargs": {"enable_thinking": thinking}}},
    })
    directory = tmp_path / "conditions" / condition["id"]
    choices = [{"candidate": {"task_id": t, "source_id": t + "_r0", "task": t, "context": {},
                              "reflection": t, "reflection_sha256": accumulation.base.digest(t)},
                "tie_indices": [0, 1], "selection_rng_seed": "42"} for t in tasks]
    accumulation.base.write(directory / "selections.json", choices)
    monkeypatch.setattr(accumulation.base, "configure_environment", lambda _: None)

    def forbidden(*args, **kwargs):
        pytest.fail("New actor or reflector call during AUD accumulation")

    monkeypatch.setattr(ACEAgentInstance, "start", forbidden)
    monkeypatch.setattr(ACEAgentInstance, "_run_post_session_reflection", forbidden)
    seen = []

    def response(instance, model, prompt, **kwargs):
        assert instance.enable_thinking is thinking
        assert instance.analysis_temperature_override == temperature
        assert "UPDATE" in prompt and "DELETE" in prompt
        seen.append(instance.task)
        ops = {
            "a": [{"type": "ADD", "section": "strategies_and_insights", "content": "Old rule"},
                  {"type": "ADD", "section": "strategies_and_insights", "content": "Obsolete rule"}],
            "b": [{"type": "UPDATE", "entry_id": "str-00001", "content": "Corrected rule"},
                  {"type": "DELETE", "entry_id": "str-00002"}],
            "c": [{"type": "ADD", "section": "strategies_and_insights", "content": "New rule"}],
        }[instance.task]
        return json.dumps({"operations": ops})

    monkeypatch.setattr(ACEAgentInstance, "_llm_call_simple", response)
    assert accumulation.train_condition(tmp_path, condition)["ok"]
    cp = accumulation.base.read(directory / "checkpoints/003.json")
    assert seen == tasks and cp["playbook_edit_mode"] == "add_update_delete"
    assert "Corrected rule" in cp["playbook"] and "New rule" in cp["playbook"]
    assert "Old rule" not in cp["playbook"] and "Obsolete rule" not in cp["playbook"]
    assert "str-00002" not in cp["playbook"] and "str-00003" in cp["playbook"]
    assert cp["next_global_id"] == 4


def bank_fixture(tmp_path, monkeypatch):
    source, target = tmp_path / "A", tmp_path / "AUD"
    common = dict.fromkeys(study.MATCHED_KEYS, "matched")
    common["task_orders"] = {str(s): {"train": ["task"]} for s in (42, 43, 44)}
    study.base.write(source / "protocol.json", {**common, "playbook_edit_mode": "add_only"})
    study.base.write(target / "protocol.json", {**common, "playbook_edit_mode": "add_update_delete"})
    study.base.write(source / "reflection_bank.json", {"frozen": True})
    study.base.write(source / "bank_ready.json", {"contexts": 500, "evaluations": 1500})
    evidence = tmp_path / "evidence.json"
    study.base.write(evidence, {"pct": 0.5})
    study.base.write(source / "bank_source_hashes.json", {str(evidence): study.base.digest(evidence.read_bytes())})
    for seed in (42, 43, 44):
        for condition in accumulation.conditions(seed):
            study.base.write(source / f"conditions/{condition['id']}/selections.json", [{
                "candidate": {"task_id": "task", "reflection": "same Reflection",
                              "reflection_sha256": study.base.digest("same Reflection")},
                "tie_indices": [0, 2, 4], "selection_rng_seed": "same exact tie decision",
            }])
    study.base.write(source / "conditions/s42_without_gt_random/checkpoints/050.json", {"playbook": "A learned"})
    monkeypatch.setattr(study, "predecessor_ready", lambda _: True)
    return source, target


def test_import_reuses_exact_selections_and_never_transfers_learned_contexts(tmp_path, monkeypatch):
    source, target = bank_fixture(tmp_path, monkeypatch)
    manifest = study.reuse_selected_bank(source, target)
    assert len(manifest["files"]) == 21
    assert manifest["selection_regeneration"] is manifest["learned_context_transfer"] is False
    assert not list((target / "conditions").glob("*/checkpoints/*.json"))
    for name in manifest["files"]:
        assert (source / name).read_bytes() == (target / name).read_bytes()
    (target / "bank_ready.json").unlink()
    assert study.reuse_selected_bank(source, target) == manifest
    (target / "conditions/s42_without_gt_random/selections.json").write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        study.reuse_selected_bank(source, target)


def test_import_rejects_sampling_changes_and_incomplete_source(tmp_path, monkeypatch):
    source, target = bank_fixture(tmp_path, monkeypatch)
    protocol = study.base.read(target / "protocol.json")
    protocol["model_settings"] = {"temperature": 0.1}
    study.base.write(target / "protocol.json", protocol)
    with pytest.raises(ValueError, match="differ"):
        study.reuse_selected_bank(source, target)
    monkeypatch.setattr(study, "predecessor_ready", lambda _: False)
    with pytest.raises(ValueError, match="must be complete"):
        study.reuse_selected_bank(source, target)


def test_aud_waits_for_entire_nonthinking_pipeline_without_inference(tmp_path, monkeypatch):
    pipeline, out = tmp_path / "nonthinking", tmp_path / "aud"
    study.base.write(pipeline / "status.json", {"state": "running", "phase": "accumulation"})
    study.base.write(out / "plan.json", {"nonthinking_pipeline": str(pipeline), "scheduler_source_hashes": {}})
    monkeypatch.setattr(study.time, "sleep", lambda _: (out / "STOP").touch())
    monkeypatch.setattr(study.base, "check_endpoint", lambda _: pytest.fail("Premature model access"))
    monkeypatch.setattr(study.subprocess, "Popen", lambda *a, **k: pytest.fail("Premature AUD launch"))
    with pytest.raises(InterruptedError):
        study.run(out)
    assert study.base.read(out / "status.json")["phase"] == "waiting_for_nonthinking_completion"


@pytest.mark.parametrize("fail", [False, True])
def test_aud_runs_thinking_then_nonthinking_and_stops_on_failure(tmp_path, monkeypatch, fail):
    pipeline, out = tmp_path / "nonthinking", tmp_path / "aud"
    study.base.write(pipeline / "status.json", {"state": "completed"})
    runs = {}
    for mode in ("thinking", "nonthinking"):
        source, target = tmp_path / (mode + "A"), out / mode
        study.base.write(source / "protocol.json", {})
        study.base.write(target / "protocol.json", {})
        study.base.write(target / "status.json", {"state": "scheduled"})
        runs[mode] = {"source": str(source), "target": str(target),
                      "source_protocol_sha256": study.base.digest((source / "protocol.json").read_bytes())}
    study.base.write(out / "plan.json", {"nonthinking_pipeline": str(pipeline),
                                        "scheduler_source_hashes": {}, "runs": runs})
    monkeypatch.setattr(study, "verify_sources", lambda _: None)
    monkeypatch.setattr(study, "reuse_selected_bank", lambda *a: None)
    monkeypatch.setattr(study.base, "check_endpoint", lambda _: {})
    monkeypatch.setattr(study, "predecessor_ready", lambda p: (
        study.base.read(p / "status.json")["state"] == "completed"))
    seen = []

    def launch(command, **kwargs):
        target = study.Path(command[-1])
        seen.append(target.name)
        if fail:
            return SimpleNamespace(pid=1234, wait=lambda: 1)
        study.base.write(target / "status.json", {"state": "completed"})
        return SimpleNamespace(pid=1234, wait=lambda: 0)

    monkeypatch.setattr(study.subprocess, "Popen", launch)
    if fail:
        with pytest.raises(RuntimeError, match="next experiment will not start"):
            study.run(out)
        assert seen == ["thinking"]
        assert study.base.read(out / "nonthinking/status.json")["state"] == "scheduled"
    else:
        study.run(out)
        assert seen == ["thinking", "nonthinking"]
        assert study.base.read(out / "status.json")["state"] == "completed"
