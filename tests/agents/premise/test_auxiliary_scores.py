# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from exgentic.agents.premise.auxiliary_scores import candidate_metric
from exgentic.agents.premise.pipeline import PremisePipeline, write_json


def test_terminal_partial_report_counts_skipped_and_keeps_official_score_independent(tmp_path):
    path = tmp_path / "benchmark/harbor/verifier/ctrf.json"
    write_json(path, {"results": {"summary": {"tests": 3, "passed": 1, "pending": 0, "other": 0},
                                "tests": [{"status": status} for status in ["passed", "failed", "skipped"]]}})
    trace = {"session_dir": str(tmp_path), "score": 0, "success": False}
    result = candidate_metric(trace, "terminalbench2")
    assert result["score"] == 1 / 3 and result["total"] == 3 and trace["score"] == 0
    data = json.loads(path.read_text())
    data["results"]["summary"]["pending"] = 1
    write_json(path, data)
    with pytest.raises(ValueError, match="incomplete"):
        candidate_metric(trace, "terminalbench2")


def test_absent_turn_evidence_never_falls_back_to_official_binary():
    with pytest.raises(ValueError, match="Missing BFCL"):
        candidate_metric({"score": 1, "details": {}}, "bfcl")
    result = candidate_metric({"details": {"session_metadata": {"experiment_budget_exceeded": True}}}, "bfcl")
    assert result["score"] == 0 and result["source"] == "actor_budget_failure"


@pytest.mark.parametrize("selection,parents", [("pareto", 2), ("alignment_product", 1)])
@pytest.mark.parametrize("benchmark", ["bfcl", "terminalbench2"])
def test_auxiliary_reward_drives_selection_and_resumes_without_changing_official_metrics(
    tmp_path, selection, parents, benchmark,
):
    manifest = tmp_path / "split.json"
    write_json(manifest, {f"{benchmark}_42": {"train": ["one", "two"]}})
    args = {"output": tmp_path / "run", "manifest": manifest, "benchmarks": [benchmark], "mode": "isolated",
            "seed": 42, "count": None, "past_count": 3, "model": "test", "base_url": "test",
            "candidate_selection": selection, "benchmark_candidate_metric": "auxiliary"}
    pipeline = PremisePipeline(**args)
    pipeline._run_agent = lambda **_: {
        "benchmark": benchmark, "task_id": "one", "actions": [{"action": {}}], "score": 0, "success": False,
        "auxiliary_metric": {"metric": "test_partial", "score": .75}, "details": {}}
    pipeline.llm = SimpleNamespace(
        reflect=lambda *_a, **_k: {"span": {"start": 0, "end": 1}, "entry_assessments": []},
        mutate=lambda *_: {"operations": [], "reasoning": "unchanged"}, judge=lambda *_: {"score": 1})
    artifact = pipeline.step()
    assert len(artifact["parents"]) == parents and artifact["parents"][0]["score"] == 0
    assert len(artifact["candidates"]) == 4 and len(artifact["selection"]["selected"]) == parents
    if selection == "alignment_product":
        assert artifact["selection"]["objectives"]["candidate_1"]["product"] == .75
    else:
        assert artifact["selection"]["objectives"]["candidate_1"]["candidate_score"] == (.75,)
    assert PremisePipeline(**args).cursor == 1
    with pytest.raises(ValueError, match="configuration differs"):
        PremisePipeline(**{**args, "benchmark_candidate_metric": "official"})
