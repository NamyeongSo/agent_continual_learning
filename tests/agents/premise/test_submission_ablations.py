# SPDX-License-Identifier: Apache-2.0
"""Selection changes must preserve topology, provenance, and genuine cost savings."""

import copy
from types import SimpleNamespace

import pytest

from exgentic.agents.premise.ablation_selection import select_ablation_beam
from exgentic.agents.premise.pipeline import PremisePipeline, write_json
from exgentic.agents.premise.selection import select_beam


def pool():
    return [
        {"id": f"candidate_{i}", "valid": True, "current_alignment": a, "past_alignment": a,
         "current_official_score": p, "past_official_score": p}
        for i, a, p in [(1, 1.0, 0.0), (2, 1.0, 0.25), (3, 0.0, 1.0), (4, 0.0, 0.75)]
    ]


def test_pfs_first_reverses_the_two_priorities_without_mutating_the_saved_pool():
    candidates = pool()
    before = copy.deepcopy(candidates)
    assert select_beam(candidates)[1]["selected"] == ["candidate_2", "candidate_1"]
    selected, evidence = select_ablation_beam(candidates, "pfs_first")
    assert [x["id"] for x in selected] == ["candidate_3", "candidate_4"]
    assert selected[0] is candidates[2]
    assert evidence["objectives"]["candidate_3"] == {"ifs": (0.0, 0.0), "pfs": (1.0, 1.0)}
    assert candidates == before


def test_joint_pareto_uses_all_four_dimensions():
    selected, evidence = select_ablation_beam(pool(), "joint_pareto")
    assert set(evidence["joint_fronts"][0]) == {"candidate_2", "candidate_3"}
    assert [x["id"] for x in selected] == ["candidate_3", "candidate_2"]
    assert evidence["joint_axes"] == ["current_ifs", "past_ifs", "current_pfs", "past_pfs"]


def test_cold_start_omits_unavailable_past_axes():
    candidates = pool()
    for item in candidates:
        item["past_alignment"] = item["past_official_score"] = None
    _, evidence = select_ablation_beam(candidates, "joint_pareto")
    assert evidence["joint_axes"] == ["current_ifs", "current_pfs"]


def test_random_ignores_scores_and_is_reproducible_without_replacement():
    candidates = [{"id": f"candidate_{i}", "valid": True} for i in range(1, 5)]
    outcomes = set()
    for seed in range(100):
        selected, evidence = select_ablation_beam(candidates, "random", seed=seed, iteration=9)
        repeated = select_ablation_beam(candidates, "random", seed=seed, iteration=9)[0]
        assert selected == repeated and len({x["id"] for x in selected}) == 2
        assert evidence["candidate_scoring"] is False
        outcomes.add(frozenset(x["id"] for x in selected))
    assert len(outcomes) == 6


@pytest.mark.parametrize("rule", ["pfs_first", "joint_pareto"])
def test_scored_variants_reject_missing_or_nonfinite_measurements(rule):
    candidates = pool()
    candidates[0]["current_alignment"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        select_ablation_beam(candidates, rule)


def test_random_training_preserves_four_to_two_and_skips_all_candidate_rollouts(tmp_path):
    manifest = tmp_path / "split.json"
    write_json(manifest, {"bfcl_42": {"train": ["one", "two"]}})
    options = {
        "output": tmp_path / "run", "manifest": manifest, "benchmarks": ["bfcl"],
        "mode": "isolated", "seed": 42, "count": None, "past_count": 3,
        "model": "openai/test", "base_url": "http://localhost/v1", "candidate_selection": "random",
    }
    pipeline = PremisePipeline(**options)
    calls = []

    def actor(**kwargs):
        calls.append(kwargs)
        assert kwargs.get("replay_source") is None
        return {"benchmark": "bfcl", "task_id": kwargs["task_id"], "task": "task", "context": {},
                "actions": [{"action": {}}], "score": 1, "details": {}}

    pipeline._run_agent = actor
    pipeline._score_candidates = lambda *_: pytest.fail("Random selection must incur no replay cost")
    pipeline.llm = SimpleNamespace(
        reflect=lambda *_a, **_kw: {
            "span": {"start": 0, "end": 1}, "analysis": "grounded", "expected_behavior": "inspect",
            "entry_assessments": [],
        },
        mutate=lambda *_a, **_kw: {
            "operations": [{"type": "ADD", "section": "strategies_and_insights", "content": "Inspect inputs."}],
            "reasoning": "grounded",
        },
        judge=lambda *_: pytest.fail("Random selection must incur no judge cost"),
    )
    for _ in range(2):
        artifact = pipeline.step()
        assert len(artifact["candidates"]) == 4
        assert len(artifact["selection"]["selected"]) == 2
        assert artifact["parallelism"]["replay_jobs"] == 0
        assert artifact["parallelism"]["peak_replay_sessions"] == 0
        assert all("current_result" not in c for c in artifact["candidates"])
    assert len(calls) == 4
    assert artifact["past_sample"]
    assert all("selected_official_score" not in m for m in pipeline.states["bfcl"]["memory"])
    assert PremisePipeline(**options).cursor == 2
    with pytest.raises(ValueError, match="configuration differs"):
        PremisePipeline(**{**options, "candidate_selection": "pfs_first"})
