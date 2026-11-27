# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy
import json
from pathlib import Path

import pytest

from scripts.ace import run_appworld_policy_shift as study
from scripts.premise.run_bfcl_policy_shift import probe_request, score_calls


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    splits, runs, entries = {}, [], []
    for seed in study.SEEDS:
        source = tmp_path / f"source{seed}"
        task_ids = [f"train{seed}_{i}" for i in range(50)]
        splits[f"appworld_{seed}"] = {"train": task_ids, "val": [f"val{seed}"], "test": [f"test{seed}"]}
        config = {"seed": seed, "model": study.MODEL, "agent": "ace", "mode": "isolated",
                  "benchmarks": ["appworld"], "disable_thinking": True,
                  "task_order": [["appworld", task] for task in task_ids],
                  "benchmark_max_turns": {"appworld": 50}}
        study.write_json(source / "experiment_config.json", config)
        study.write_json(root / "source_configs" / f"ace_s{seed}.json", config)
        study.write_json(source / "training_complete.json", {"completed_task_count": 50})
        (source / "online_metrics.jsonl").write_text("\n".join(
            json.dumps({"task_id": task, "status": "success"}) for task in task_ids) + "\n")
        for point in study.POINTS[1:]:
            snapshot = {"store_id": "ace_isolated_appworld", "playbook": f"context seed{seed} cp{point}"}
            path = root / "ace" / f"s{seed}" / f"checkpoint_{point:03d}.json"
            original = source / f"cp{point}.json"
            study.write_json(path, snapshot)
            study.write_json(original, snapshot)
            entries.append({"method": "ace", "seed": seed, "checkpoint": point,
                            "path": str(path.relative_to(root)), "sha256": study.digest(path.read_bytes()),
                            "source": str(original), "source_sha256": study.digest(original.read_bytes())})
        study.write_json(source / "playbook_ace_isolated_appworld.json", snapshot)
        runs.append({"method": "ace", "seed": seed, "source_run": str(source)})
    study.write_json(root / "task_split_manifest.json", splits)
    split_hash = study.digest((root / "task_split_manifest.json").read_bytes())
    for seed, run in zip(study.SEEDS, runs, strict=True):
        path = Path(run["source_run"]) / "experiment_config.json"
        config = study.read_json(path)
        config["task_split_manifest_sha256"] = split_hash
        study.write_json(path, config)
        study.write_json(root / "source_configs" / f"ace_s{seed}.json", config)
    study.write_json(root / "manifest.json", {"benchmark": "appworld", "mode": "isolated",
                                             "manifest_sha256": split_hash, "runs": runs,
                                             "checkpoints": entries})
    return root


def test_prepare_uses_own_seed_contexts_without_changing_source_learning(tmp_path, bundle):
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK

    original = {path: path.read_bytes() for path in tmp_path.glob("source*/**/*") if path.is_file()}
    output = tmp_path / "study"
    prepared = study.prepare(output, bundle, "http://localhost:18006/v1", 150)
    contexts = study.read_json(output / "contexts.json")["contexts"]
    assert len(contexts) == 33
    assert prepared["learning"] is False
    assert all(row["new_training_tasks"] == 0 for row in prepared["training_audit"])
    for context in contexts:
        text = study.read_json(Path(context["context_path"]))["playbook"]["playbook"]
        assert text == (f"context seed{context['seed']} cp{context['checkpoint']}"
                        if context["checkpoint"] else DEFAULT_PLAYBOOK)
    assert sum(len(run["task_ids"]) for run in prepared["runs"]) == 150
    assert prepared["environment_execution"] is False
    assert all(path.read_bytes() == raw for path, raw in original.items())
    assert study.prepare(output, bundle, "http://localhost:18006/v1", 150) == prepared


@pytest.mark.parametrize("corruption", ["checkpoint", "model", "task_order", "incomplete"])
def test_prepare_refuses_wrong_or_changed_learning_sources(tmp_path, bundle, corruption):
    if corruption == "checkpoint":
        study.write_json(bundle / "ace/s42/checkpoint_005.json", {"playbook": "tampered"})
    else:
        source = tmp_path / "source42"
        name = "training_complete.json" if corruption == "incomplete" else "experiment_config.json"
        path = source / name
        config = study.read_json(path)
        if corruption == "model":
            config["model"] = "openai/another-model"
        elif corruption == "task_order":
            config["task_order"][0][1] = "held-out-task"
        else:
            config["completed_task_count"] = 49
        study.write_json(path, config)
    with pytest.raises(ValueError):
        study.prepare(tmp_path / "study", bundle, "http://localhost:18006/v1", 150)


def test_oss_preparation_requires_own_learning_and_matched_reasoning(tmp_path, bundle):
    model = "openai/openai/gpt-oss-20b"
    with pytest.raises(ValueError, match="requested model"):
        study.prepare(tmp_path / "study", bundle, "http://localhost:18010/v1", 150, model, "low")
    for seed in study.SEEDS:
        source = tmp_path / f"source{seed}/experiment_config.json"
        config = study.read_json(source)
        config.update(model=model, reasoning_effort="low")
        study.write_json(source, config)
        study.write_json(bundle / "source_configs" / f"ace_s{seed}.json", config)
    prepared = study.prepare(tmp_path / "study", bundle, "http://localhost:18010/v1", 150, model, "low")
    assert prepared["model"]["model"] == model
    assert prepared["model"]["reasoning_effort"] == "low"
    assert prepared["model"]["key"] == "gpt_oss_20b_ace"
    with pytest.raises(ValueError, match="reasoning effort differ"):
        study.prepare(tmp_path / "other", bundle, "http://localhost:18010/v1", 150, model, "high")


def test_own_model_training_plan_matches_existing_ace_order_and_twenty_step_budget(tmp_path, bundle):
    from scripts.ace import run_appworld_own_model_policy_shift as own
    from scripts.ace.run_experiment import task_order_from_train_manifest

    manifest = bundle / "task_split_manifest.json"
    plan = own.plan_study(tmp_path / "own", manifest, "http://localhost:18010/v1", 150, own.OSS_MODEL, "low", 20)
    assert len(plan["jobs"]) == 3
    for job in plan["jobs"]:
        expected = task_order_from_train_manifest(manifest, ["appworld"], job["seed"], None, "isolated")
        assert job["task_order"] == [list(item) for item in expected]
        command = job["command"]
        assert command[command.index("--model") + 1] == own.OSS_MODEL
        assert command[command.index("--reasoning-effort") + 1] == "low"
        assert command[command.index("--appworld-max-turns") + 1] == "20"
    assert plan["evaluation_environment_execution"] is False


def test_own_learning_restart_adopts_worker_even_if_initial_fork_cmdline_was_empty(tmp_path, monkeypatch):
    from scripts.ace import run_appworld_own_model_policy_shift as own

    command = ["python", str(own.ROOT / "scripts/ace/run_experiment.py"), "--output-dir", str(tmp_path / "train")]
    job = {"seed": 42, "run": str(tmp_path / "train"), "command": command}
    saved = {"pid": 123, "start_ticks": "456", "argv": []}
    current = {**saved, "argv": command}
    study.write_json(tmp_path / "training_workers/s42.json", {"identity": saved})
    monkeypatch.setattr(own, "identity", lambda _pid: current)
    worker = own.find_training(tmp_path, job)
    assert worker.pid == 123
    assert worker.poll() is None
    current = {**current, "start_ticks": "999"}
    assert own.find_training(tmp_path, job) is None


def test_source_model_alias_preserves_oss_model_namespace():
    assert study.source_model_matches("openai/gpt-oss-20b", "openai/openai/gpt-oss-20b")
    assert study.source_model_matches("openai/openai/gpt-oss-20b", "openai/openai/gpt-oss-20b")
    assert not study.source_model_matches("openai/gpt-oss-120b", "openai/openai/gpt-oss-20b")
    assert not study.source_model_matches(study.MODEL, "openai/openai/gpt-oss-20b")


def test_connection_failures_cannot_be_counted_as_completed_learning():
    from scripts.ace import run_appworld_own_model_policy_shift as own

    valid = {"result": {"status": "limit_reached", "details": {}}}
    own.validate_training_infrastructure([valid], 42)
    failed = {"result": {"status": "error", "details": {"session_metadata": {
        "error": "litellm.InternalServerError: OpenAIException - Connection error."}}}}
    with pytest.raises(RuntimeError, match="seed42 task2"):
        own.validate_training_infrastructure([valid, failed], 42)


def test_endpoint_migration_preserves_frozen_own_model_experiment(tmp_path):
    from scripts.ace import run_appworld_own_model_policy_shift as own

    original = {"base_url": "http://localhost:18010/v1", "model": own.OSS_MODEL, "max_steps": 20, "jobs": []}
    own.save_runtime_plan(tmp_path, original)
    raw = (tmp_path / "launch_manifest.json").read_bytes()
    updated = {**original, "base_url": "http://localhost:8006/v1"}
    assert own.save_runtime_plan(tmp_path, updated) == original
    assert (tmp_path / "launch_manifest.json").read_bytes() == raw
    assert study.read_json(tmp_path / "runtime_route.json")["base_url"] == updated["base_url"]
    with pytest.raises(ValueError, match="beyond endpoint routing"):
        own.save_runtime_plan(tmp_path, {**updated, "max_steps": 50})


@pytest.mark.parametrize("model_name", [study.MODEL, "openai/openai/gpt-oss-20b"])
def test_restore_saved_requests_without_environment_execution(tmp_path, bundle, monkeypatch, model_name):
    from exgentic.agents.ace.prompts.generator import build_generator_system_prompt
    from exgentic.interfaces.lib import api

    monkeypatch.setattr(api, "execute", lambda **_kwargs: pytest.fail("AppWorld must not execute"))
    output = tmp_path / "study"
    effort = None if model_name == study.MODEL else "low"
    if effort:
        for seed in study.SEEDS:
            source = tmp_path / f"source{seed}/experiment_config.json"
            config = study.read_json(source)
            config.update(model=model_name, reasoning_effort=effort)
            study.write_json(source, config)
            study.write_json(bundle / "source_configs" / f"ace_s{seed}.json", config)
    prepared = study.prepare(output, bundle, "http://localhost:18006/v1", 150, model_name, effort)
    folder = "ace_isolated_s42_appworld" + ("_resume001" if effort else "")
    session = tmp_path / "source42" / folder / "sessions/session"
    trace = session / "agent/litellm/trace.jsonl"
    trace.parent.mkdir(parents=True)
    study.write_json(session / "session.json", {"task_id": "train42_0"})
    request = {"model": model_name.removeprefix("openai/"), "messages": [
        {"role": "system", "content": build_generator_system_prompt("[str-00001] helpful=1 harmful=0 :: learned")},
        {"role": "user", "content": "saved environment observation"}],
        "tools": [{"type": "function", "function": {"name": "gmail__search_emails",
                                                    "parameters": {"type": "object"}}}]}
    response = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [
        {"id": "id", "type": "function", "function": {"name": "gmail__search_emails",
                                                         "arguments": '{"query":"hello"}'}}]}}]}
    rows = [{"request": {"messages": []}, "response": {}}]
    rows += [{"request": request, "response": response}] * 25
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    for run in prepared["runs"]:
        (Path(run["source"]) / f"ace_isolated_s{run['seed']}_appworld/sessions").mkdir(parents=True, exist_ok=True)
    anchors = study.restore_history_bank(output, prepared)
    assert len(anchors) == 20
    history = study.read_json(Path(anchors[0]["path"]))
    assert history["messages"] == request["messages"]
    assert history["tools"] == request["tools"]
    original = copy.deepcopy(history)
    probed = probe_request(history, "a different ACE playbook", prepared["model"])
    assert probed["messages"][1:] == request["messages"][1:]
    assert probed["tools"] == request["tools"]
    assert history == original
    assert probed.get("reasoning_effort") == effort
    assert history["reference_calls"] == [{"name": "gmail__search_emails", "arguments": {"query": "hello"}}]
    assert study.restore_history_bank(output, prepared) == anchors
    trace.write_text(trace.read_text() + "\n")
    with pytest.raises(ValueError, match="Original training history changed"):
        study.restore_history_bank(output, prepared)


def test_restore_missing_definitions_from_saved_inputs_and_report_unrecoverable_trace(tmp_path):
    source, output = tmp_path / "source", tmp_path / "study"
    prepared = {"model": {"model": study.MODEL}, "runs": [
        {"seed": 42, "source": str(source), "task_ids": ["task", "donor", "unrecoverable"]}]}
    sessions = source / "ace_isolated_s42_appworld/sessions"
    system = {"role": "system", "content": "You are an expert agent that completes tasks using available tools."}
    user = {"role": "user", "content": "saved task\n\n<datetime>\nrecorded date\n</datetime>"}
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}

    def definition(kind, value):
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return {"record_type": "definition", "id": kind + ":" + study.digest(encoded.encode()), "value": value}

    system_def, user_def, tool_def = (definition("messages", system), definition("messages", user),
                                      definition("tools", tool))
    row = {"trace_format": "dedup_v1", "request": {
        "model": study.MODEL, "messages_refs": [system_def["id"], user_def["id"]],
        "tools_refs": [tool_def["id"]]}, "response": {"choices": [{"message": {
            "role": "assistant", "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}]}}]}}
    task_session = sessions / "task"
    study.write_json(task_session / "session.json", {
        "task_id": "task", "task": "saved task", "context": {"datetime": "recorded date"}})
    trace = task_session / "agent/litellm/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps(row) + "\n")
    donor = sessions / "donor"
    study.write_json(donor / "session.json", {"task_id": "donor"})
    donor_trace = donor / "agent/litellm/trace.jsonl"
    donor_trace.parent.mkdir(parents=True)
    donor_trace.write_text("\n".join(json.dumps(v) for v in (system_def, tool_def)) + "\n")
    missing = sessions / "unrecoverable"
    study.write_json(missing / "session.json", {"task_id": "unrecoverable"})
    missing_trace = missing / "agent/litellm/trace.jsonl"
    missing_trace.parent.mkdir(parents=True)
    broken = copy.deepcopy(row)
    broken["request"]["messages_refs"] = ["messages:" + "f" * 64]
    missing_trace.write_text(json.dumps(broken) + "\n")
    originals = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    anchors = study.restore_history_bank(output, prepared)
    assert len(anchors) == 1
    history = study.read_json(Path(anchors[0]["path"]))
    assert history["messages"] == [system, user]
    assert history["tools"] == [tool]
    audit = study.read_json(output / "history_restore_audit.json")
    assert any(r["status"] == "recovered_training_definitions" for r in audit["excluded"])
    assert any(r["status"] == "missing_training_definition" for r in audit["excluded"])
    assert str(donor_trace) in audit["source_checks"]
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    assert study.restore_history_bank(output, prepared) == anchors
    tampered = copy.deepcopy(system_def)
    tampered["value"]["content"] = "modified"
    donor_trace.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(ValueError, match="content hash"):
        study.recorded_trace_definitions(prepared, {})


def test_own_model_export_uses_context_after_processed_tasks_and_carries_skipped_updates(tmp_path, bundle, monkeypatch):
    from scripts.ace import run_appworld_own_model_policy_shift as own

    plan = {"manifest": str(bundle / "task_split_manifest.json"), "manifest_sha256":
            study.digest((bundle / "task_split_manifest.json").read_bytes()),
            "model": study.MODEL, "reasoning_effort": None, "max_steps": 50, "jobs": []}
    for seed in study.SEEDS:
        source = tmp_path / f"source{seed}"
        config = study.read_json(source / "experiment_config.json")
        config.update(training_time=True, max_tokens=8192)
        study.write_json(source / "experiment_config.json", config)
        plan["jobs"].append({"seed": seed, "run": str(source), "task_order": config["task_order"]})

    def prefix(source, order):
        return [{"checkpoint_path": str(source / f"cp{i}.json") if i % 5 == 0 and i != 20 else None,
                 "result": {"status": "limit_reached", "details": {}}}
                for i in range(1, len(order) + 1)]

    monkeypatch.setattr(own, "completed_prefix", prefix)
    original = {p: p.read_bytes() for p in tmp_path.glob("source*/**/*") if p.is_file()}
    exported = own.export_bundle(tmp_path / "own", plan)
    prepared = study.prepare(tmp_path / "own", exported, "http://localhost:18006/v1", 150)
    assert len(prepared["runs"]) == 3
    for seed in study.SEEDS:
        cp20 = study.read_json(exported / f"ace/s{seed}/checkpoint_020.json")
        assert cp20["playbook"] == f"context seed{seed} cp15"
        cp50 = study.read_json(exported / f"ace/s{seed}/checkpoint_050.json")
        assert cp50["playbook"] == f"context seed{seed} cp50"
    assert all(p.read_bytes() == raw for p, raw in original.items())
    assert own.export_bundle(tmp_path / "own", plan) == exported


def test_appworld_analysis_uses_ace_audit_without_requiring_premise_iterations(tmp_path):
    call = {"name": "supervisor__show_profile", "arguments": {}}
    anchors = [{"key": "a", "task_id": "task", "seed": 42}]
    for point in study.POINTS:
        study.write_json(tmp_path / "probe_results" / f"a_cp{point:03d}.json", {
            "anchor_key": "a", "checkpoint": point, "status": "completed", "predicted_calls": [call],
            "metrics": score_calls([call], [call]),
        })
    study.write_json(tmp_path / "prepared.json", {"model": "ACE", "learning": False})
    audit = [{"seed": 42, "attempted_tasks": 50, "new_training_tasks": 0}]
    summary = study.analyze(tmp_path, anchors, {"key": "qwen_ace"}, benchmark="AppWorld", training_audit=audit,
                            initial_reference=False)
    assert summary["benchmark"] == "AppWorld"
    assert summary["training_audit"] == audit
    assert "recorded histories" in summary["reference"]
    assert "not a repeatability control" in summary["cp0"]
    assert summary["coverage"]["common_anchors_all_checkpoints"] == 1
    assert summary["coverage"]["successful_probes"] == 11
    assert (tmp_path / "analysis/checkpoint_agreement.pdf").is_file()
