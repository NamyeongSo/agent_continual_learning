# SPDX-License-Identifier: Apache-2.0
"""Product rewards and the single-parent search must survive checkpoint resume."""

import json
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from exgentic.agents.premise.pipeline import PremisePipeline, write_json
from exgentic.agents.premise.selection import select_alignment_product, select_beam


def child(index, alignment, score, past=()):
    return {"id": f"candidate_{index}", "valid": True,
            "current_result": {"alignment": {"score": alignment}, "official_score": score},
            "past_results": [{"alignment": {"score": a}, "official_score": b} for a, b in past]}


def test_product_can_choose_lower_alignment_and_records_both_means():
    a = child(1, 1, 0.1)
    b = child(2, 0.5, 1, [(1, 0.5), (0, 1)])
    selected, detail = select_alignment_product([a, b])
    assert selected == [b]
    objective = detail["objectives"]["candidate_2"]
    assert objective["alignment"] == 0.5
    assert objective["benchmark_score"] == pytest.approx(2.5 / 3)
    assert objective["product"] == pytest.approx(1.25 / 3)
    assert objective["evaluation_count"] == 3
    assert detail["parent_in_selection_pool"] is False


def test_zero_products_tie_deterministically_and_invalid_children_are_excluded():
    a, b = child(1, 1, 0), child(2, 0, 1)
    invalid = child(3, 1, 1)
    invalid["valid"] = False
    assert select_alignment_product([invalid, b, a])[0] == [a]
    with pytest.raises(RuntimeError, match="valid child"):
        select_alignment_product([invalid])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_product_scores_fail_loudly(value):
    with pytest.raises(ValueError, match="finite"):
        select_alignment_product([child(1, value, 1)])


def test_step_uses_one_parent_four_independent_children_and_one_survivor(tmp_path):
    manifest = tmp_path / "split.json"
    write_json(manifest, {"bfcl_42": {"train": ["one", "two"]}})
    options = {"output": tmp_path / "run", "manifest": manifest, "benchmarks": ["bfcl"],
               "mode": "isolated", "seed": 42, "count": None, "past_count": 3,
               "model": "openai/test", "base_url": "http://localhost/v1", "candidate_selection": "alignment_product"}
    pipeline = PremisePipeline(**options)
    calls = []

    def run_agent(**kwargs):
        calls.append(kwargs)
        return {"benchmark": "bfcl", "task_id": "one", "actions": [{"action": {}}], "score": 1, "details": {}}

    pipeline._run_agent = run_agent
    pipeline.llm = SimpleNamespace(
        reflect=lambda *_args, **_kwargs: {"span": {"start": 0, "end": 1}, "entry_assessments": []},
        mutate=lambda *_: {"operations": [], "reasoning": "already covered"},
        judge=lambda *_: {"score": 1},
    )
    artifact = pipeline.step()
    assert len(artifact["parents"]) == 1
    assert len([call for call in calls if "parent" in call["label"]]) == 1
    assert len(artifact["reflections"]) == len(artifact["candidates"]) == 4
    assert {candidate["parent_rank"] for candidate in artifact["candidates"]} == {1}
    assert artifact["selection"]["selected"] == ["candidate_1"]
    assert len(pipeline.states["bfcl"]["beam"]) == 1
    state = json.loads((pipeline.output / "state.json").read_text())
    assert state["config"]["search_topology"]["children_per_parent"] == 4
    resumed = PremisePipeline(**options)
    assert resumed.cursor == 1 and len(resumed.states["bfcl"]["beam"]) == 1
    with pytest.raises(ValueError, match="configuration differs"):
        PremisePipeline(**{**options, "candidate_selection": "pareto"})


def test_product_requires_alignment_and_environment_replay(tmp_path):
    with pytest.raises(ValueError, match="requires PEACE"):
        PremisePipeline(output=tmp_path, manifest=tmp_path / "none", benchmarks=["bfcl"],
                        mode="isolated", seed=42, count=None, past_count=3,
                        model="test", base_url="test", candidate_selection="alignment_product", alignment_scoring=False)


def pct_pipeline():
    pipeline = object.__new__(PremisePipeline)
    pipeline.appworld_candidate_metric = "pct"
    pipeline.alignment_scoring = True
    pipeline.replay_retries = 0
    pipeline.replay_workers = 4
    pipeline.benchmark_slots = {"appworld": nullcontext(), "bfcl": nullcontext()}
    pipeline._replay_lock = threading.Lock()
    pipeline._active_replays = pipeline._peak_replays = 0
    pipeline.llm = SimpleNamespace(judge=lambda *_: {"score": 1})
    return pipeline


def test_pct_selects_higher_partial_success_while_all_tgc_scores_remain_zero():
    pipeline = pct_pipeline()

    def run_agent(**kwargs):
        pct = kwargs["playbook"]["pct"] - (0.1 if "past" in kwargs["label"] else 0)
        return {"score": 0.0, "partial_score": pct, "success": False, "details": {}, "actions": []}

    pipeline._run_agent = run_agent
    reflection = {"span": {"start": 0}}
    source = {"benchmark": "appworld", "task_id": "current"}
    candidates = [{"id": f"candidate_{i}", "parent_rank": 1, "reflection": reflection,
                   "playbook": {"pct": pct}, "valid": True} for i, pct in [(1, 0.2), (2, 0.8)]]
    pipeline._score_candidates(1, candidates, [source], [{"trace": source, "reflection": reflection}])
    selected, detail = select_alignment_product(candidates, score_key="candidate_score")
    assert selected[0]["id"] == "candidate_2"
    assert detail["objectives"]["candidate_2"]["benchmark_score"] == pytest.approx(0.75)
    assert detail["objectives"]["candidate_2"]["benchmark_score_metrics"] == ["appworld_pct_v1"]
    for candidate in candidates:
        assert candidate["current_official_score"] == candidate["past_official_score"] == 0
        for replay in [candidate["current_result"], *candidate["past_results"]]:
            assert replay["official_score"] == replay["replay"]["score"] == 0
            assert replay["replay"]["success"] is False
            assert replay["candidate_score"] == replay["replay"]["partial_score"]


@pytest.mark.parametrize("pct", [None, True, float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_pct_cannot_fall_back_to_tgc(pct):
    pipeline = pct_pipeline()
    pipeline._run_agent = lambda **_: {"score": 0, "partial_score": pct, "details": {}}
    with pytest.raises(ValueError, match="Candidate score"):
        pipeline._score_job({"playbook": {}}, {"benchmark": "appworld", "task_id": "one"},
                            {"span": {"start": 0}}, "current")


def test_missing_pct_cannot_fall_back_to_tgc():
    pipeline = pct_pipeline()
    pipeline._run_agent = lambda **_: {"score": 0, "details": {}}
    with pytest.raises(KeyError, match="partial_score"):
        pipeline._score_job({"playbook": {}}, {"benchmark": "appworld", "task_id": "one"},
                            {"span": {"start": 0}}, "current")


@pytest.mark.parametrize("selection,parent_count", [("pareto", 2), ("alignment_product", 1)])
def test_appworld_pct_step_pins_metric_and_preserves_preupdate_tgc(tmp_path, selection, parent_count):
    manifest = tmp_path / "split.json"
    write_json(manifest, {"appworld_42": {"train": ["one", "two"]}})
    options = {"output": tmp_path / "run", "manifest": manifest, "benchmarks": ["appworld"],
               "mode": "isolated", "seed": 42, "count": None, "past_count": 3,
               "model": "openai/test", "base_url": "http://localhost/v1", "candidate_selection": selection,
               "appworld_candidate_metric": "pct"}
    pipeline = PremisePipeline(**options)
    pipeline._run_agent = lambda **_: {
        "benchmark": "appworld", "task_id": "one", "actions": [{"action": {}}],
        "score": 0.0, "success": False, "partial_score": 0.6, "details": {},
    }
    pipeline.llm = SimpleNamespace(
        reflect=lambda *_args, **_kwargs: {"span": {"start": 0, "end": 1}, "entry_assessments": []},
        mutate=lambda *_: {"operations": [], "reasoning": "already covered"}, judge=lambda *_: {"score": 1},
    )
    artifact = pipeline.step()
    assert artifact["parents"][0]["score"] == 0
    assert len(artifact["parents"]) == parent_count
    assert len(artifact["candidates"]) == 4
    assert len(artifact["selection"]["selected"]) == parent_count
    if selection == "alignment_product":
        assert artifact["selection"]["objectives"]["candidate_1"]["product"] == 0.6
    else:
        assert artifact["selection"]["objectives"]["candidate_1"]["candidate_score"] == (0.6,)
    state = json.loads((pipeline.output / "state.json").read_text())
    assert state["config"]["appworld_candidate_metric"] == "pct"
    assert state["config"]["appworld_reporting_metric"] == "appworld_tgc_v1"
    assert PremisePipeline(**options).cursor == 1
    with pytest.raises(ValueError, match="configuration differs"):
        PremisePipeline(**{**options, "appworld_candidate_metric": "tgc"})


def test_original_pareto_uses_pct_only_for_performance_axis():
    candidates = [{"id": f"candidate_{i}", "valid": True, "current_alignment": alignment,
                   "past_alignment": None, "current_official_score": 0, "past_official_score": None,
                   "current_candidate_score": pct, "past_candidate_score": None}
                  for i, alignment, pct in [(1, 1, 0.2), (2, 1, 0.8), (3, 1, 0.9), (4, 0.5, 1)]]
    selected, detail = select_beam(candidates, score_key="candidate_score")
    assert [item["id"] for item in selected] == ["candidate_3", "candidate_2"]
    assert detail["benchmark_score_key"] == "candidate_score"
    assert detail["alignment_fronts"] == [["candidate_1", "candidate_2", "candidate_3"], ["candidate_4"]]
    assert [item["id"] for item in select_beam(candidates)[0]] == ["candidate_1", "candidate_2"]
