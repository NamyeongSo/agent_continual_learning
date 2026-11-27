# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Guard diagnostic label isolation, orientation mapping, and rejection denominators."""

import pytest

from scripts.premise.analyze_judge_contracts import restore_public_schemas
from scripts.premise.analyze_judge_specificity import (
    RELAXED_MUTATOR,
    RELAXED_REFLECTOR,
    decision_metrics,
    generation_scores,
    paired_difference,
    relaxed_mutation,
    replay_records,
    strict_mutation,
)


def decision(pair, task, swapped, reward, choice="A", arm="parent", subset="other_benchmark"):
    return {
        "record_id": pair,
        "task_id": task,
        "benchmark": "appworld",
        "swapped": swapped,
        "status": "completed",
        "reward": reward,
        "choice": choice,
        "raw_choice": ("B" if choice == "A" else "A") if swapped else choice,
        "arm": arm,
        "subset": subset,
        "case_id": "case",
    }


def test_relaxing_specificity_preserves_structure_and_metadata_guards():
    parent = [{"id": "rule-00001", "content": "Prefer supported outcomes."}]
    content = (
        "Prefer evidence that next_page was exhausted when all records were requested; first-page tasks are exempt."
    )
    response = {"operations": [{"type": "ADD", "content": content}]}
    record, visible = {"task_id": "hidden_case"}, {"public_tools": []}
    with pytest.raises(ValueError, match="example or function identifier"):
        strict_mutation(parent, response, record, visible)
    assert relaxed_mutation(parent, response, "hidden_case")[-1]["content"] == content
    for operation in (
        {"type": "DELETE", "id": "rule-00001"},
        {"type": "UPDATE", "id": "rule-99999", "content": content},
        {"type": "ADD", "content": "Prefer trajectory A."},
        {"type": "ADD", "content": "Prefer hidden_case's expected answer."},
        {"type": "ADD", "content": "Prefer AppWorld trajectories."},
    ):
        with pytest.raises(ValueError):
            relaxed_mutation(parent, {"operations": [operation]}, "hidden_case")
    assert parent == [{"id": "rule-00001", "content": "Prefer supported outcomes."}]


def test_relaxed_prompt_does_not_retain_conflicting_specificity_bans():
    assert "Omit specific artifacts, applications, API names and examples" not in RELAXED_REFLECTOR
    assert "Abstract all domain nouns" not in RELAXED_MUTATOR
    assert "remove ALL case illustrations" not in RELAXED_MUTATOR
    assert "Technical terms, tool/API names and short examples are allowed" in RELAXED_MUTATOR
    assert "Maximum three operations" in RELAXED_MUTATOR
    assert "Rule deletion is disabled" in RELAXED_MUTATOR


def test_transfer_sampling_excludes_validation_and_other_pairs_of_source_task():
    current = {"id": "current", "task_id": "source", "benchmark": "appworld", "split": "train"}
    records = {"current": current}
    records["same_task"] = {**current, "id": "same_task"}
    records["validation"] = {**current, "id": "validation", "task_id": "val", "split": "val"}
    for bench in ("bfcl", "appworld", "terminalbench2"):
        for task in range(6):
            for pair in range(2):
                identity = f"{bench}-{task}-{pair}"
                records[identity] = {"id": identity, "benchmark": bench, "task_id": str(task), "split": "train"}
    chosen = replay_records(current, records)
    assert chosen == replay_records(current, dict(reversed(list(records.items()))))
    assert chosen[0] == current
    assert len(chosen) == 11
    assert len({(r["benchmark"], r["task_id"]) for r in chosen}) == 11
    assert all(r["split"] == "train" for r in chosen)
    assert "same_task" not in {r["id"] for r in chosen}
    assert sorted(sum(r["benchmark"] == b for r in chosen[1:]) for b in ("bfcl", "appworld", "terminalbench2")) == [
        3,
        3,
        4,
    ]


def test_metrics_keep_both_orientations_together_and_cluster_by_task():
    rows = [decision("x", "many_pairs", False, 1), decision("x", "many_pairs", True, 0, choice="B")]
    rows.extend(decision("y", "many_pairs", swapped, 1) for swapped in (False, True))
    rows.extend(decision("z", "one_pair", swapped, 0) for swapped in (False, True))
    metrics = decision_metrics(rows)
    assert metrics["accuracy"] == 0.5
    assert metrics["task_macro_accuracy"] == 0.375
    assert metrics["flip_rate"] == pytest.approx(1 / 3)
    assert metrics["raw_B_fraction"] == pytest.approx(1 / 3)
    assert metrics["task_groups"] == 2
    improved = [{**r, "reward": 1} for r in rows]
    contrast = paired_difference(rows, improved)
    assert contrast["accuracy_delta"] == 0.5
    assert contrast["task_macro_delta"] == 0.625
    assert contrast["task_groups"] == 2
    assert contrast["task_macro_ci95"] == [0.25, 1.0]
    failed = [{**rows[0], "status": "error"}, *rows[1:]]
    assert decision_metrics(failed)["accuracy"] is None
    assert paired_difference(rows, failed)["status"] == "incomplete"


def test_rejected_proposals_are_not_removed_from_improvement_denominator():
    rows = []
    for subset in ("current", "same_benchmark_other_task", "other_benchmark"):
        for swapped in (False, True):
            rows.append(decision("x", "task", swapped, 0, arm="parent", subset=subset))
            rows.append(decision("x", "task", swapped, 1, arm="strict_0", subset=subset))
    proposals = [
        {"case_id": "case", "condition": "strict", "replicate": 0, "status": "valid", "changed": True},
        {"case_id": "case", "condition": "strict", "replicate": 1, "status": "rejected", "error_message": "identifier"},
    ]
    result = generation_scores(rows, proposals)["summary"]["strict"]["by_subset"]["other_benchmark"]
    assert result["valid_only_delta"] == 1
    assert result["intention_to_treat_delta"] == 0.5
    assert result["independent_source_tasks"] == 1


def test_contract_restoration_changes_only_public_tool_schemas():
    visible = {
        "task": "public request",
        "trajectory_a": ["a"],
        "trajectory_b": ["b"],
        "common_prefix": [],
        "context": {},
        "public_tools": [{"name": "finish", "is_finish": True}],
    }
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    session = {
        "actions": [
            {"name": "finish", "arguments_schema": schema},
            {"name": "finish", "is_hidden": True, "arguments_schema": {"secret": "hidden"}},
        ]
    }
    restored, changed = restore_public_schemas(visible, session)
    assert changed == 1
    assert restored["public_tools"] == [{"name": "finish", "is_finish": True, "arguments_schema": schema}]
    assert {k: v for k, v in restored.items() if k != "public_tools"} == {
        k: v for k, v in visible.items() if k != "public_tools"
    }
    assert "arguments_schema" not in visible["public_tools"][0]
    assert restore_public_schemas(restored, session) == (restored, 0)
    with pytest.raises(ValueError, match="schemas disagree"):
        restore_public_schemas(restored, {"actions": [{"name": "finish", "arguments_schema": {}}]})
