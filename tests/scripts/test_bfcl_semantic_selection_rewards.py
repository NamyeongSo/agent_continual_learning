# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy
import hashlib
import json

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_semantic_selection_rewards import (
    MODELS,
    NOVEL,
    REWARDS,
    TEST_KEYS,
    calculate,
    load_parents,
    reward_metadata,
    reward_scores,
    task_draws,
)
from scripts.premise.analyze_bfcl_training_selection_hypotheses import selection_probabilities


def test_repair_aversion_charges_twice_for_regression_and_preserves_current_replay_weights():
    parent = np.zeros((1, 10))
    parent[0, 1] = 1
    candidates = np.repeat(parent[:, None, :], 4, axis=1)
    candidates[0, 0, 0] = 1  # Repairing the current task earns one half.
    candidates[0, 1, 2] = 1  # Repairing a past task earns one eighteenth.
    candidates[0, 2, 1] = 0  # Losing a past success costs two eighteenths.
    scores, diagnostics = reward_scores(candidates, parent)
    assert scores["repair_aversion"][0] == pytest.approx([0.5, 1 / 18, -2 / 18, 0])
    assert diagnostics["selected_repaired_parent_failures"][0].tolist() == [1, 1, 0, 0]
    assert diagnostics["selected_broken_parent_successes"][0].tolist() == [0, 0, 1, 0]
    # Parent subtraction alone assigns only a one-unit cost to the same regression.
    assert scores["paired_linear"][0] == pytest.approx([0.1, 0.1, -0.1, 0])


def test_preservation_barrier_prioritizes_past_safety_over_a_large_raw_reward_advantage():
    parent = np.zeros((1, 10))
    parent[0, 1] = 1
    candidates = np.zeros((1, 2, 10))
    candidates[0, 0, 1] = 1
    candidates[0, 1] = 1
    candidates[0, 1, 1] = 0
    scores, diagnostics = reward_scores(candidates, parent)
    assert scores["period_mean"][0, 1] > scores["period_mean"][0, 0]
    assert selection_probabilities(scores["preserve_barrier"])[0].tolist() == [1, 0]
    assert diagnostics["safe_selected_probability"][0].tolist() == [1, 0]


def test_preservation_barrier_all_unsafe_falls_back_to_period_choice_with_uniform_ties():
    parent = np.ones((1, 10))
    candidates = np.zeros((1, 3, 10))
    candidates[0, :2, :5] = 1
    scores, diagnostics = reward_scores(candidates, parent)
    assert not diagnostics["safe_selected_probability"].any()
    assert selection_probabilities(scores["preserve_barrier"]) == pytest.approx(np.array([[0.5, 0.5, 0]]))
    assert selection_probabilities(scores["preserve_barrier"]) == pytest.approx(
        selection_probabilities(scores["period_mean"])
    )


def test_past_safety_does_not_turn_a_current_regression_into_a_preservation_violation():
    parent = np.zeros((1, 10))
    parent[0, 0] = 1
    candidate = np.zeros((1, 1, 10))
    scores, diagnostics = reward_scores(candidate, parent)
    assert diagnostics["selected_broken_parent_successes"][0, 0] == 1
    assert diagnostics["selected_past_regressions"][0, 0] == 0
    assert diagnostics["safe_selected_probability"][0, 0] == 1
    assert scores["preserve_barrier"] == pytest.approx(scores["period_mean"])


def test_joint_competence_scores_prefer_transfer_balance_at_the_same_task_mean():
    # Both candidates solve six tasks; only the second solves tasks in both replay groups.
    candidates = np.array([[[1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 1, 1, 0, 0]]])
    scores, _ = reward_scores(candidates, np.zeros((1, 10)))
    assert scores["task_mean"][0].tolist() == [0.6, 0.6]
    for key in ("geometric_balance", "soft_worst", "conflict_aversion"):
        assert scores[key][0, 1] > scores[key][0, 0]
    assert np.isfinite(scores["geometric_balance"]).all()
    assert np.isfinite(scores["soft_worst"]).all()


def test_paired_risk_penalizes_cancelling_repairs_and_regressions_despite_equal_advantage():
    parent = np.zeros((1, 10))
    parent[0, :5] = 1
    candidates = np.repeat(parent[:, None, :], 2, axis=1)
    candidates[0, 1, 0] = 0
    candidates[0, 1, 5] = 1
    scores, _ = reward_scores(candidates, parent)
    assert scores["paired_linear"][0].tolist() == [0, 0]
    assert scores["paired_risk"][0, 0] == 0
    assert scores["paired_risk"][0, 1] == pytest.approx(-np.sqrt(2 / 9) / np.sqrt(10))


@pytest.mark.parametrize("parent_value", [0, 1])
def test_harmonic_utility_stays_finite_when_a_parent_partition_is_empty(parent_value):
    parent = np.full((1, 10), parent_value)
    candidates = np.array([[[0] * 10, [1] * 10]])
    scores, _ = reward_scores(candidates, parent)
    harmonic = scores["repair_retention_harmonic"]
    assert np.isfinite(harmonic).all()
    assert harmonic[0, 1] > harmonic[0, 0]
    empty_partition_rate = 0.5
    observed_partition_rate = 10.5 / 11
    assert harmonic[0, 1] == pytest.approx(
        2 * empty_partition_rate * observed_partition_rate / (empty_partition_rate + observed_partition_rate)
    )


def test_transfer_breadth_rewards_cross_group_repairs_without_changing_total_advantage():
    candidates = np.zeros((1, 2, 10))
    candidates[0, 0, 1:3] = 1
    candidates[0, 1, [1, 6]] = 1
    scores, _ = reward_scores(candidates, np.zeros((1, 10)))
    assert scores["paired_linear"][0].tolist() == [0.2, 0.2]
    assert scores["transfer_breadth"][0] == pytest.approx([0.3, 0.4])


def test_parent_linear_and_log_controls_preserve_exact_choice_probabilities_including_ties():
    rng = np.random.default_rng(19)
    candidates = rng.integers(2, size=(200, 5, 10))
    candidates[:, 1] = candidates[:, 0]
    scores, _ = reward_scores(candidates, rng.integers(2, size=(200, 10)))
    expected = selection_probabilities(scores["task_mean"])
    assert (expected[:, 0] == expected[:, 1]).all()
    for key in ("paired_linear", "log_mean"):
        assert selection_probabilities(scores[key]) == pytest.approx(expected)
    # A fully flat candidate set must remain uniform under every implemented formula.
    flat, _ = reward_scores(np.zeros((1, 5, 10)), np.ones((1, 10)))
    for value in flat.values():
        assert selection_probabilities(value) == pytest.approx(np.full((1, 5), 0.2))


def _synthetic_checkpoint():
    rng = np.random.default_rng(7)
    order = [f"task{i}" for i in range(50)]
    train = rng.integers(2, size=(5, 50))
    point = {
        "key": "cp10", "seed": 42, "checkpoint": 10, "train_order": order,
        "train_rewards": train.tolist(),
        "scores": {"full_train": train.mean(axis=1).tolist(), "full_validation": [0.2, 0.4, 0.6, 0.8, 1]},
        "parent_scores": {"full_train": 0.5},
    }
    parents = {("Qwen", "cp10"): {"parent_rewards": rng.integers(2, size=50).tolist()}}
    # Only public information through the current task exists in this fixture.
    metadata = {
        task: {"first_request": f"Schedule document action {i}", "tool_classes": [f"Tools{i % 3}"]}
        for i, task in enumerate(order[:11])
    }
    return {"label": "Qwen", "points": [point]}, parents, metadata


def test_task_draws_use_current_once_and_disjoint_past_groups_with_reproducible_sampling():
    model, _, metadata = _synthetic_checkpoint()
    point = model["points"][0]
    indices, digest = task_draws(point, metadata, 40, 7)
    assert indices.shape == (40, 10)
    assert (indices[:, 0] == 10).all()
    assert (indices[:, 1:] < 10).all()
    assert all(len(set(row)) == 10 for row in indices)
    again, digest_again = task_draws(point, metadata, 40, 7)
    assert np.array_equal(indices, again)
    assert digest == digest_again


def test_candidate_choices_cannot_read_future_rewards_parent_outcomes_or_validation_scores():
    model, parents, metadata = _synthetic_checkpoint()
    _, before, samples_before = calculate(model, parents, metadata, {"cp10"}, 12, 19)
    changed_model, changed_parents = copy.deepcopy(model), copy.deepcopy(parents)
    changed_point = changed_model["points"][0]
    changed_train = np.asarray(changed_point["train_rewards"])
    changed_train[:, 11:] = 1 - changed_train[:, 11:]
    changed_point["train_rewards"] = changed_train.tolist()
    changed_point["scores"] = {"full_train": [1, 0.8, 0.6, 0.4, 0.2], "full_validation": [1, 0, 1, 0, 0]}
    changed_parents["Qwen", "cp10"]["parent_rewards"][11:] = [1] * 39
    _, after, samples_after = calculate(changed_model, changed_parents, metadata, {"cp10"}, 12, 19)
    assert samples_before == samples_after
    for left, right in zip(before, after, strict=True):
        assert left["reward"] == right["reward"]
        for index in range(1, 6):
            assert left[f"candidate_{index}_probability"] == right[f"candidate_{index}_probability"]


def test_zero_safe_feasibility_is_recorded_without_an_abstain_or_missing_selected_candidate():
    model, parents, metadata = _synthetic_checkpoint()
    model["points"][0]["train_rewards"] = np.zeros((5, 50), dtype=int).tolist()
    parents["Qwen", "cp10"]["parent_rewards"] = np.ones(50, dtype=int).tolist()
    _, rows, samples = calculate(model, parents, metadata, {"cp10"}, 5, 7)
    assert samples[0]["safe_feasible_fraction"] == 0
    protected = next(row for row in rows if row["reward"] == "preserve_barrier")
    assert protected["safe_selected_probability"] == 0
    assert [protected[f"candidate_{i}_probability"] for i in range(1, 6)] == pytest.approx([0.2] * 5)


def test_oss_is_rejected_and_the_declared_test_family_excludes_equivalent_controls():
    with pytest.raises(ValueError, match="Only Qwen/Gemma"):
        calculate({"label": "OSS", "points": []}, {}, {}, set(), 1, 7)
    assert MODELS == ("Qwen", "Gemma")
    assert len(NOVEL) == 8 and len(TEST_KEYS) == 10 and len(REWARDS) == 13
    assert not {"paired_linear", "log_mean"}.intersection(TEST_KEYS)
    assert {row["reward"] for row in reward_metadata()} == set(REWARDS)


@pytest.mark.parametrize("bad_value", [0.5, np.nan, np.inf])
def test_binary_reward_contract_rejects_nonbinary_or_nonfinite_inputs(bad_value):
    candidate = np.zeros((1, 5, 10))
    candidate[0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="binary"):
        reward_scores(candidate, np.zeros((1, 10)))


def _parent_results(tmp_path):
    folder = tmp_path / "source"
    context = folder / "frozen" / "cp10" / "parent.json"
    context.parent.mkdir(parents=True)
    context.write_text('{"context": "frozen playbook"}\n')
    digest = hashlib.sha256(context.read_bytes()).hexdigest()
    results = folder / "results" / "cp10" / "parent" / "train"
    results.mkdir(parents=True)
    for index in range(50):
        row = {"checkpoint_key": "cp10", "context": "parent", "partition": "train",
               "context_sha256": digest, "task_id": f"task{index}", "score": index % 2,
               "status": "error" if index == 0 else "completed"}
        (results / f"{index:02}.json").write_text(json.dumps(row))
    point = {"key": "cp10", "source": str(folder), "train_order": [f"task{i}" for i in reversed(range(50))],
             "parent_scores": {"full_train": 0.5}}
    raw = {"models": [{"label": "Qwen", "points": [point]},
                      {"label": "OSS", "points": [{"source": "missing-must-not-read"}]}]}
    return raw, results


def test_parent_loader_matches_task_identity_instead_of_file_order_and_excludes_oss(tmp_path):
    raw, _ = _parent_results(tmp_path)
    output = tmp_path / "cache.json"
    loaded = load_parents(raw, output)
    assert set(loaded) == {("Qwen", "cp10")}
    point = loaded["Qwen", "cp10"]
    assert point["parent_rewards"] == [index % 2 for index in reversed(range(50))]
    assert point["parent_error_flags"] == [False] * 49 + [True]
    assert json.loads(output.read_text())["new_model_calls"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [({"context_sha256": "different-context"}, "identity mismatch"),
     ({"task_id": "task1"}, "Duplicate parent task"),
     ({"status": "error", "score": 1}, "terminal binary"),
     ({"status": "running"}, "incomplete result")],
)
def test_parent_loader_rejects_context_mismatch_duplicate_tasks_and_invalid_terminal_results(
    tmp_path, mutation, message
):
    raw, folder = _parent_results(tmp_path)
    path = folder / "00.json"
    row = {**json.loads(path.read_text()), **mutation}
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match=message):
        load_parents(raw, tmp_path / "cache.json")
