# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy
import json
import random
from pathlib import Path

import pytest

from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
from exgentic.agents.premise.selection import sample_past
from scripts.premise.collect_bfcl_prefix_checkpoints import recover
from scripts.utils.evaluate_isolated_checkpoint_queue import read_json, write_json


def recorded_run(root):
    run = root / "run"
    playbook = {"playbook": DEFAULT_PLAYBOOK, "next_global_id": 1}
    reflection = {"entry_assessments": []}
    config = {"mode": "sequential", "seed": 42, "manifest_sha256": "split-hash", "past_count": 1,
              "model": "openai/gemma", "order": [["bfcl", f"task-{i}"] for i in range(13)]}
    memory = [{"memory_id": "1|bfcl|task-0", "benchmark": "bfcl", "task_id": "task-0",
               "trace_artifact": "iterations/iter_0001.json", "parent_index": 0,
               "reflection": reflection, "accepted_candidate_id": "candidate_1", "selected_official_score": 1,
               "latest_selected_replay_official_score": 1}]
    rng = random.Random(42)
    for index in range(1, 13):
        session = run / "parent_sessions" / str(index)
        write_json(session / "config.json", {"agent_kwargs": {"playbook": DEFAULT_PLAYBOOK}})
        sample_past([] if index == 1 else memory, 1, rng)
        artifact = {"iteration": index, "benchmark": "bfcl", "task_id": f"task-{index - 1}",
                    "parents": [{"session_dir": str(session)}, {"session_dir": str(session)}]}
        if index == 11:
            artifact.update(status="skipped", reason="insufficient_valid_candidates", train_update_applied=False)
        else:
            candidates = [{"id": f"candidate_{rank}", "parent_rank": rank, "playbook": copy.deepcopy(playbook),
                           "reflection": reflection, "applied": index == 1, "current_official_score": 1,
                           "past_results": [] if index == 1 else [{"official_score": 1}]} for rank in (1, 2)]
            artifact.update(candidates=candidates, selection={"selected": ["candidate_1", "candidate_2"]},
                            past_sample=[] if index == 1 else [{"benchmark": "bfcl", "task_id": "task-0"}])
        write_json(run / "iterations" / f"iter_{index:04d}.json", artifact)
    memory[0].update(latest_selected_replay_candidate_id="candidate_1", latest_selected_replay_iteration=12)
    write_json(run / "state.json", {"config": config, "cursor": 12,
                                   "states": {"global": {"beam": [playbook, playbook], "memory": memory}},
                                   "rng_state": repr(rng.getstate())})
    return run


def test_recovery_keeps_skipped_beam_but_advances_rng_and_verifies_full_terminal_state(tmp_path):
    run = recorded_run(tmp_path)
    source_bytes = (run / "state.json").read_bytes()
    groups = recover(run, tmp_path / "frozen", (10, 11), 42, "split-hash")
    ten, eleven = [read_json(tmp_path / "frozen" / f"checkpoint_{point:03d}.json") for point in (10, 11)]
    assert ten["states"] == eleven["states"]
    assert ten["rng_state"] != eleven["rng_state"]
    assert ten["states"]["global"]["memory"][0]["latest_selected_replay_iteration"] == 10
    assert all(group["entire_terminal_state_verified"] for group in groups)
    assert read_json(tmp_path / "frozen/recovery_manifest.json")["parent_configs_verified"] == 24
    assert (run / "state.json").read_bytes() == source_bytes


@pytest.mark.parametrize("field", ["beam", "memory", "rng_state"])
def test_recovery_rejects_corrupted_terminal_state_before_freezing_any_checkpoint(tmp_path, field):
    run = recorded_run(tmp_path)
    saved = read_json(run / "state.json")
    if field == "beam":
        saved["states"]["global"]["beam"][0]["playbook"] = "corrupted"
    elif field == "memory":
        saved["states"]["global"]["memory"][0]["latest_selected_replay_official_score"] = 0
    else:
        saved["rng_state"] = repr(random.Random(7).getstate())
    write_json(run / "state.json", saved)
    with pytest.raises(ValueError, match="entire saved state"):
        recover(run, tmp_path / "frozen", (10,), 42, "split-hash")
    assert not (tmp_path / "frozen").exists()


def test_recovery_rejects_a_recorded_parent_that_does_not_match_its_reconstructed_context(tmp_path):
    run = recorded_run(tmp_path)
    path = run / "parent_sessions/10/config.json"
    path.write_text(json.dumps({"agent_kwargs": {"playbook": "wrong-parent"}}))
    with pytest.raises(ValueError, match="parent differs at iteration 10"):
        recover(run, tmp_path / "frozen", (10,), 42, "split-hash")


def test_recovery_supports_bfcl_isolated_training_and_preserves_the_original_mode(tmp_path):
    run = recorded_run(tmp_path)
    saved = read_json(run / "state.json")
    saved["config"]["mode"] = "isolated"
    saved["config"]["reasoning_effort"] = "low"
    saved["states"] = {"bfcl": saved["states"]["global"]}
    write_json(run / "state.json", saved)
    groups = recover(run, tmp_path / "frozen", (10,), 42, "split-hash")
    checkpoint = read_json(tmp_path / "frozen/checkpoint_010.json")
    assert checkpoint["config"]["mode"] == groups[0]["source_mode"] == "isolated"
    assert checkpoint["config"]["reasoning_effort"] == "low"
    assert set(checkpoint["states"]) == {"bfcl"}


def test_skipped_replay_with_explicit_rng_state_is_recovered_without_guessing_sampling(tmp_path):
    run = recorded_run(tmp_path)
    original = recover(run, tmp_path / "original", (11,), 42, "split-hash")
    skipped_rng = read_json(Path(original[0]["checkpoint_path"]))["rng_state"]
    write_json(run / "iterations/iter_0011.json", {
        "iteration": 11, "benchmark": "bfcl", "task_id": "task-10",
        "status": "skipped", "train_update_applied": False, "reason": "nontransient_task_error",
        "rng_state_after": skipped_rng,
    })
    recovered = recover(run, tmp_path / "recovered", (11,), 42, "split-hash")
    assert read_json(Path(recovered[0]["checkpoint_path"])) == read_json(Path(original[0]["checkpoint_path"]))


def test_policy_probe_can_recover_early_and_terminal_checkpoints_without_next_iteration(tmp_path):
    run = recorded_run(tmp_path)
    with pytest.raises(ValueError):
        recover(run, tmp_path / "default", (5, 12), 42, "split-hash")
    groups = recover(run, tmp_path / "policy", (5, 10, 12), 42, "split-hash",
                     minimum_point=1, allow_terminal=True)
    assert [group["checkpoint"] for group in groups] == [5, 10, 12]
    assert read_json(tmp_path / "policy/checkpoint_012.json") == read_json(run / "state.json")
