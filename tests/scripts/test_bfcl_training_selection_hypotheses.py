# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_training_selection_hypotheses import (
    BASELINE,
    calculate,
    leave_one_seed_out,
    metadata_indices,
    policies,
    public_metadata,
    replay_score,
    selection_probabilities,
    task_subsets,
)


def test_public_metadata_excludes_solution_paths_and_future_conversation_turns(tmp_path):
    path = tmp_path / "tasks.jsonl"
    record = {"id": "train", "question": [[{"role": "user", "content": "First request"}],
                                          [{"role": "user", "content": "Hidden later request"}]],
              "involved_classes": ["PublicTools"], "path": ["ground_truth.answer"], "initial_config": {"answer": 42}}
    path.write_text(json.dumps(record) + "\n")
    assert public_metadata(path, {"train"}) == {
        "train": {"first_request": "First request", "tool_classes": ["PublicTools"]},
    }
    with pytest.raises(ValueError, match="missing"):
        public_metadata(path, {"train", "missing"})


def test_public_similarity_samplers_choose_distinct_past_tasks_and_complement_current_task():
    similarity = np.array([[1, 0.8, 0, 0.9], [0.8, 1, 0, 0.8], [0, 0, 1, 0], [0.9, 0.8, 0, 1]])
    assert metadata_indices(similarity, 2, "similar").tolist() == [0, 1]
    assert metadata_indices(similarity, 2, "diverse").tolist() == [2, 1]
    assert metadata_indices(similarity, 2, "representative").tolist() == [2, 1]
    for strategy in ("similar", "diverse", "representative"):
        indices = metadata_indices(similarity, 3, strategy)
        assert sorted(indices.tolist()) == [0, 1, 2]


def test_every_replay_sampler_keeps_the_nine_task_budget_and_excludes_current_and_future():
    point = {"key": "test", "checkpoint": 14}
    draws = task_subsets(point, 100, 7, np.eye(15))
    for key, values in draws.items():
        count = 10 if key == "past10" else 9
        assert values.shape[1] == count
        assert all(len(set(row)) == count and max(row) < 14 for row in values)
    assert draws["recent"].tolist() == [list(range(5, 14))]
    assert draws["oldest"].tolist() == [list(range(9))]
    assert draws["uniform"] == pytest.approx(task_subsets(point, 100, 7, np.eye(15))["uniform"])


def test_information_weights_only_use_selected_tasks_and_handle_completely_uninformative_scores():
    matrix = np.array([[1, 0, 99], [0, 1, 99], [0, 1, 99], [0, 0, 99], [0, 0, 99]])
    selected = np.array([[0, 1]])
    variance = np.array([0.16, 0.24])
    expected = (matrix[:, :2] * variance).sum(axis=1) / variance.sum()
    assert replay_score(matrix, selected, information=True)[0] == pytest.approx(expected)
    assert replay_score(np.ones((5, 20)), selected, information=True) == pytest.approx(np.ones((1, 5)))


def test_tied_candidates_get_uniform_probability_and_flat_task_does_not_change_replay_ranking():
    replay = np.array([[0.1, 0.6, 0.6, 0.2, 0]])
    expected = np.array([[0, 0.5, 0.5, 0, 0]])
    assert selection_probabilities(replay) == pytest.approx(expected)
    for alpha in (0.05, 0.1, 0.25, 0.5, 0.75):
        assert selection_probabilities(alpha * np.ones((1, 5)) + (1 - alpha) * replay) == pytest.approx(expected)


def test_held_out_seed_values_cannot_affect_the_policy_chosen_for_that_seed():
    rows = []
    for seed in (42, 43, 44):
        for policy in policies():
            rows.append({"model": "Synthetic", "seed": seed, "policy": policy["key"], "matched36": True,
                         "validation_lift": 0.1 if policy["key"] == BASELINE else 0})
    before = leave_one_seed_out(rows)
    changed = [{**row, "validation_lift": 100 if row["policy"] == "past10" else -100}
               if row["seed"] == 42 else row for row in rows]
    after = leave_one_seed_out(changed)
    for cohort in ("cp36", "cp50"):
        original = next(row for row in before if row["held_out_seed"] == 42 and row["cohort"] == cohort)
        updated = next(row for row in after if row["held_out_seed"] == 42 and row["cohort"] == cohort)
        assert original["chosen_policy"] == updated["chosen_policy"] == BASELINE
        assert original["development_lift"] == updated["development_lift"]


def test_policy_family_has_fixed_cost_limits_and_exactly_one_reference():
    family = policies()
    assert len(family) == 17 and len({row["key"] for row in family}) == 17
    assert sum(row["key"] == BASELINE for row in family) == 1
    assert all(row["tasks_used"] <= 10 for row in family)


def test_candidate_choices_do_not_read_validation_rewards_or_future_task_metadata():
    rng = np.random.default_rng(3)
    rewards = rng.integers(2, size=(5, 50))
    order = [f"task{i}" for i in range(50)]
    point = {"key": "cp10", "seed": 42, "checkpoint": 10, "train_order": order,
             "train_rewards": rewards.tolist(),
             "scores": {"full_train": rewards.mean(axis=1).tolist(), "full_validation": [0, 0.1, 0.2, 0.3, 0.4]},
             "parent_scores": {"full_train": 0.5}}
    # Future metadata is deliberately absent: rules must only read the first eleven task descriptions.
    metadata = {task: {"first_request": f"Action {i} for saved document", "tool_classes": [f"Tools{i % 3}"]}
                for i, task in enumerate(order[:11])}
    model = {"label": "Synthetic", "points": [point]}
    before = calculate(model, metadata, {"cp10"}, 20, 7)[1]
    point["scores"]["full_validation"] = [1, 0.8, 0.4, 0, 0.2]
    after = calculate(model, metadata, {"cp10"}, 20, 7)[1]
    for left, right in zip(before, after, strict=True):
        assert left["policy"] == right["policy"]
        for index in range(1, 6):
            assert left[f"candidate_{index}_probability"] == right[f"candidate_{index}_probability"]
