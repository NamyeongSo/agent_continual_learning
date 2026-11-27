# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Ensure full-prefix reconstruction never reuses incompatible model requests."""

import copy
import hashlib
import json

import pytest

from scripts.premise.audit_group_relative_trajectory import (
    actor_request,
    all_jobs,
    copy_additional,
    copy_reusable,
    label,
    prepare_temperature_ablation,
)


def bank(tmp_path):
    path = tmp_path / "history.json"
    anchor = {"messages": [{"role": "system", "content": "before OLD after"},
                           {"role": "user", "content": "task"}], "tools": [],
              "policy_span": {"message": 0, "part": None, "start": 7, "end": 10,
                              "sha256": hashlib.sha256(b"OLD").hexdigest()}}
    path.write_text(json.dumps(anchor))
    case = {"id": "one", "benchmark": "bfcl", "task": "task", "action_index": 0,
            "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    prepared = {"cases": [case], "repeats": 2, "checkpoints": {"bfcl": [0, 10]},
                "policies": {"bfcl": {"0": {"content": "OLD"}, "10": {"content": "NEW"}}},
                "generation_settings": {"model": "test", "temperature": 1, "max_tokens": None}}
    return prepared, case


def test_all_acting_prefixes_have_independent_reference_and_repeat_banks(tmp_path):
    prepared, case = bank(tmp_path)
    assert [(p, k) for _, p, k in all_jobs(prepared)] == [(0, -1), (0, 0), (0, 1), (10, 0), (10, 1)]
    a, b = actor_request(prepared, case, 0), actor_request(prepared, case, 10)
    assert a["messages"][0]["content"] == "before OLD after"
    assert b["messages"][0]["content"] == "before NEW after"
    assert a["messages"][1:] == b["messages"][1:]
    assert a["generation_settings"] == b["generation_settings"]


def test_interrupted_results_require_exact_request_and_identity_match(tmp_path):
    prepared, case = bank(tmp_path)
    source, target = tmp_path / "source", tmp_path / "target"
    (source / "draws").mkdir(parents=True)
    target.mkdir()
    row = {"case": "one", "context": "cp010", "repeat": 0, "status": "completed",
           "actions": [{"name": "act", "arguments": {}}]}
    original = source / "draws" / f"{label(case, 10, 0)}.json"
    original.write_text(json.dumps(row))
    (source / "prepared.json").write_text(json.dumps(prepared))
    assert copy_additional(prepared, target, [source]) == 1
    reused = json.loads((target / "draws" / original.name).read_text())
    assert reused["actions"] == row["actions"] and reused["reused_from"] == str(original)
    assert copy_additional(prepared, target, [source]) == 0
    (target / "draws" / original.name).unlink()
    changed = copy.deepcopy(prepared)
    changed["generation_settings"]["temperature"] = 0
    with pytest.raises(ValueError, match="request differs"):
        copy_additional(changed, target, [source])
    original.write_text(json.dumps({**row, "repeat": 1}))
    with pytest.raises(ValueError, match="identity differs"):
        copy_additional(prepared, target, [source])


def test_temperature_ablation_changes_only_both_probe_temperatures_and_rejects_t1_cache(tmp_path):
    source, case = bank(tmp_path)
    profile = {"temperature": 1, "max_tokens": None, "top_p": .95, "reasoning_effort": "xhigh"}
    config = tmp_path / "live_config.json"
    config.write_text(json.dumps({"model": "test", "model_settings": profile}))
    grid = tmp_path / "grid"
    grid.mkdir()
    (grid / "prepared.json").write_text(json.dumps({"temperature": 1}))
    source.update(protocol="full_trajectory_regenerated_contexts_cp10_t1_v1", model="test", temperature=1,
                  model_call_settings=profile, model_call_source={"path": str(config)},
                  baseline={"path": str(grid)})
    paired = tmp_path / "paired"
    paired.mkdir()
    frozen = paired / "prepared.json"
    frozen.write_text(json.dumps(source))
    before = frozen.read_bytes()
    changed = prepare_temperature_ablation(paired, 0)
    assert changed["temperature"] == changed["generation_settings"]["temperature"] == 0
    assert changed["model_call_settings"]["temperature"] == 0
    assert changed["cases"] == source["cases"] and changed["policies"] == source["policies"]
    assert frozen.read_bytes() == before and source["temperature"] == 1
    assert actor_request(changed, case, 0)["messages"] == actor_request(source, case, 0)["messages"]
    assert copy_reusable(changed, tmp_path / "unused") == 0
    for k, v in source["model_call_settings"].items():
        if k != "temperature":
            assert changed["model_call_settings"][k] == v
    profile["top_p"] = .8
    config.write_text(json.dumps({"model": "test", "model_settings": profile}))
    with pytest.raises(ValueError, match="other than"):
        prepare_temperature_ablation(paired, 0)
