# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy
import json

import httpx
import pytest

from scripts.premise import run_bfcl_ancestry_policy_shift as common
from scripts.premise import run_bfcl_root_final_policy_shift as study
from scripts.premise.run_bfcl_policy_shift import score_calls


def calls(value):
    return [{"name": "lookup", "arguments": {"value": value}}]


def test_exact_full_prompt_replacement_preserves_every_other_history_field():
    history = {"messages": [{"role": "system", "content": "old prompt"},
                            {"role": "user", "content": "frozen task"},
                            {"role": "assistant", "tool_calls": [{"id": "original_call_id"}]},
                            {"role": "tool", "tool_call_id": "original_call_id", "content": "observation"}],
               "tools": [{"type": "function", "function": {"name": "lookup"}}]}
    original = copy.deepcopy(history)
    prompt = "a literal full GEPA/SkillOpt system prompt"
    body = common.request_body(history, prompt)
    assert body["messages"][0]["content"] == prompt
    assert body["messages"][1:] == original["messages"][1:]
    assert body["tools"] == original["tools"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0 and body["max_tokens"] == 8192
    assert history == original


@pytest.mark.parametrize("predicted,reference", [
    ([], []), (calls(1), []), ([], calls(1)), (calls(1), calls(2)),
    (calls(1) + calls(1), calls(1)), (calls(1) + calls(2), calls(2) + calls(1)),
])
def test_compiled_action_metrics_match_existing_duplicate_and_order_semantics(predicted, reference):
    assert common.compiled_score(common.compiled_calls(predicted), common.compiled_calls(reference)) == pytest.approx(
        score_calls(predicted, reference))


def test_shared_prompts_are_queried_once_and_resume_checks_provenance(tmp_path, monkeypatch):
    import asyncio

    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}]}}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(handler)))
    anchors = []
    for index in range(2):
        path = tmp_path / "histories" / f"{index}.json"
        common.write_json(path, {"messages": [{"role": "system", "content": "old"},
                                              {"role": "user", "content": str(index)}], "tools": []})
        anchors.append({"key": str(index), "seed": 42, "task_id": str(index), "path": str(path),
                        "sha256": common.digest(path.read_bytes())})
    prompt = "identical root and final"
    sha = common.digest(prompt.encode())
    cohort = {"key": "ace_s42", "seed": 42, "nodes": [
        {"id": kind, "prompt": prompt, "prompt_sha256": sha} for kind in ("root", "final")]}
    prepared = {"base_url": "http://unit.test/v1", "workers": 3, "histories": anchors, "cohorts": [cohort]}
    asyncio.run(common.probe_cohort(tmp_path, prepared, cohort, []))
    assert len(requests) == 2
    asyncio.run(common.probe_cohort(tmp_path, prepared, cohort, []))
    assert len(requests) == 2
    path = common.probe_path(tmp_path, anchors[0], sha)
    record = common.read_json(path)
    record["anchor_sha256"] = "changed"
    common.write_json(path, record)
    with pytest.raises(ValueError, match="provenance"):
        asyncio.run(common.probe_cohort(tmp_path, prepared, cohort, []))


def test_methods_share_valid_inputs_and_average_tasks_equally(tmp_path):
    anchors = [{"key": key, "seed": 42, "task_id": task, "sha256": key}
               for key, task in [("a0", "A"), ("a1", "A"), ("b", "B")]]
    cohorts = []
    for method in ("ace", "gepa"):
        nodes = [{"id": kind, "prompt_sha256": method + kind} for kind in ("root", "final")]
        cohort = {"key": method + "_s42", "method": method, "seed": 42, "nodes": nodes}
        cohorts.append(cohort)
        for anchor in anchors:
            for node in nodes:
                failed = method == "gepa" and anchor["key"] == "a0" and node["id"] == "final"
                value = 2 if method == "ace" and anchor["key"] == "a1" and node["id"] == "final" else 1
                common.write_json(common.probe_path(tmp_path, anchor, node["prompt_sha256"]), {
                    "status": "error" if failed else "completed", "anchor_sha256": anchor["sha256"],
                    "prompt_sha256": node["prompt_sha256"], "model": common.MODEL, "predicted_calls": calls(value)})
    prepared = {"cohorts": cohorts, "histories": anchors}
    study.analyze(tmp_path, prepared, [c["key"] for c in cohorts])
    summary = common.read_json(tmp_path / "analysis/summary.json")
    assert summary["common_histories_by_seed"]["42"] == 2
    by_method = {r["method"]: r for r in summary["seed_metrics"]}
    assert by_method["ace"]["strict_accuracy"] == 0.5
    assert by_method["gepa"]["strict_accuracy"] == 1
    assert by_method["ace"]["common_tasks"] == by_method["gepa"]["common_tasks"] == 2
    assert summary["environment_execution"] is False
    assert (tmp_path / "analysis/root_final_policy_shift.pdf").is_file()


@pytest.mark.parametrize("method", ["gepa", "skillopt"])
def test_cp130_preserves_original_root_when_final_stage_selects_its_inherited_initial_policy(tmp_path, monkeypatch,
                                                                                          method):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    seed = 42
    expected = {f"bfcl_{i}" for i in range(50)}
    initial_tasks = [{"benchmark": "bfcl", "task_id": f"bfcl_{i}"} for i in range(50)]
    last_tasks = [{"benchmark": "terminalbench2", "task_id": f"tb_{i}"} for i in range(30)]
    run_name = common.GEPA_RUNS[seed] if method == "gepa" else common.SKILL_RUNS[seed]
    run = tmp_path / "team_prompt/training_history_20261009/runs" / run_name
    exported = tmp_path / "team_prompt" / f"{method}_sequential_seed{seed}/cp130_terminalbench2"
    if method == "gepa":
        for stage, tasks in [("bfcl", initial_tasks), ("terminalbench2", last_tasks)]:
            common.write_json(run / stage / "raw/experiment_config.json", {
                "task_model": common.MODEL, "mode": "sequential", "seed": seed, "train_task_ids": tasks})
        common.write_json(run / "bfcl/raw/gepa_result.json", {
            "candidates": [{"system_prompt": "original root"}], "parents": [[None]]})
        final = {"system_prompt": "learned through AppWorld; inherited Terminal-Bench root"}
        common.write_json(run / "terminalbench2/raw/gepa_result.json", {
            "candidates": [final], "parents": [[None]], "best_idx": 0, "val_aggregate_scores": [0.5]})
        common.write_json(exported / "candidate.json", {"best_candidate": final})
        expected_final = final["system_prompt"]
    else:
        for stage, tasks in [("bfcl", initial_tasks), ("terminalbench2", last_tasks)]:
            common.write_json(run / stage / "raw/task_ids.json", {
                "model": common.MODEL, "mode": "sequential", "seed": seed, "tasks": {"train": tasks}})
            common.write_json(run / stage / "raw/summary.json", {"best_step": 0})
        initial = run / "bfcl/raw/skills/skill_v0000.md"
        initial.parent.mkdir(parents=True)
        initial.write_text("original root")
        expected_final = "learned through AppWorld; inherited Terminal-Bench root"
        for path in [run / "terminalbench2/raw/skills/skill_v0000.md", run / "terminalbench2/raw/best_skill.md",
                     exported / "skill.md"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected_final)
    root, final, _ = study.selected_prompts(method, seed, expected, common.Sources(), 130)
    assert root == "original root"
    assert final == expected_final and final != root


def test_reused_probes_require_the_exact_full_request_and_leave_source_unchanged(tmp_path):
    source, output = tmp_path / "old", tmp_path / "new"
    history_path = tmp_path / "history.json"
    common.write_json(history_path, {"messages": [{"role": "system", "content": "old"}], "tools": []})
    anchor = {"key": "anchor", "seed": 42, "sha256": common.digest(history_path.read_bytes()),
              "path": str(history_path)}
    node = {"prompt": "original root", "prompt_sha256": common.digest(b"original root")}
    common.write_json(source / "prepared.json", {"model": common.MODEL})
    common.write_json(source / "status.json", {"phase": "complete"})
    body = common.request_body(common.read_json(history_path), node["prompt"])
    record = {"status": "completed", "anchor_sha256": anchor["sha256"], "prompt_sha256": node["prompt_sha256"],
              "model": common.MODEL, "request_sha256": common.digest(
                  json.dumps(body, ensure_ascii=False, sort_keys=True).encode()), "predicted_calls": calls(1)}
    original = common.probe_path(source, anchor, node["prompt_sha256"])
    common.write_json(original, record)
    before = original.read_bytes()
    prepared = {"reuse_from": str(source), "histories": [anchor], "unique_probes": 1,
                "cohorts": [{"seed": 42, "nodes": [node]}]}
    result = study.reuse_cached_probes(output, prepared)
    assert result["imported"] == 1 and result["new_inference_requests"] == 0
    saved = common.read_json(common.probe_path(output, anchor, node["prompt_sha256"]))
    assert saved["predicted_calls"] == record["predicted_calls"]
    assert saved["reused_from"] == str(original)
    assert original.read_bytes() == before
    assert study.reuse_cached_probes(output, prepared) == result
    record["request_sha256"] = "different tools or prompt"
    common.write_json(original, record)
    with pytest.raises(ValueError, match="request"):
        study.reuse_cached_probes(tmp_path / "other", prepared)
