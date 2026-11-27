# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Regression checks for rubric provenance, split isolation and measured selection."""

import importlib.util
import json
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from exgentic.agents.premise.reward_data import BENCHMARKS, digest, write_json
from exgentic.agents.premise.rubric_audit import (
    AuditConfig,
    RubricAuditor,
    accepts_repair,
    bounded_failure_packet,
    ordinal_alpha,
    pareto_repairs,
    repair_budget,
)
from exgentic.agents.premise.rubric_pipeline import (
    LocalEmbeddings,
    RubricConfig,
    RubricPipeline,
    RubricStageError,
    cosine_clusters,
    mmr_select,
    normalize_partition,
    parse_scores,
    preference_statistics,
    validate_partition,
)

DESCRIPTION = (
    "Evaluate whether the attempt establishes the requested outcome using recorded evidence, "
    "with higher scores for supported results and lower scores for unsupported claims."
)


def test_embedding_cache_root_without_reference_pins_unique_snapshot(tmp_path):
    from pathlib import Path

    repository = tmp_path / "models--BAAI--bge-m3"
    snapshot = repository / "snapshots" / "pinned-revision"
    write_json(snapshot / "config.json", {"hidden_size": 1024})
    config = {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False}
    write_json(snapshot / "1_Pooling/config.json", config)
    worker = Path(__file__)
    embedding = LocalEmbeddings("python", str(repository), worker, tmp_path / "cache")
    assert embedding.model == str(snapshot.resolve())
    assert embedding.identity["pooling"]["config"] == config
    second = repository / "snapshots" / "other-revision"
    write_json(second / "config.json", {"hidden_size": 1024})
    with pytest.raises(ValueError, match="unique snapshot"):
        LocalEmbeddings("python", str(repository), worker, tmp_path / "cache")


def test_bge_worker_selects_cls_without_mean_pooling(tmp_path):
    from pathlib import Path

    script = Path(__file__).resolve().parents[3] / "scripts/premise/embed_rubric_criteria.py"
    spec = importlib.util.spec_from_file_location("embedding_worker", script)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    write_json(tmp_path / "1_Pooling/config.json", {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False})
    assert worker.pooling_mode(tmp_path) == "cls"
    hidden = np.array([[[1, 2], [7, 8]]])
    assert worker.pool_hidden(hidden, np.array([[1, 1]]), "cls").tolist() == [[1, 2]]
    write_json(tmp_path / "1_Pooling/config.json", {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": True})
    with pytest.raises(ValueError, match="exactly one"):
        worker.pooling_mode(tmp_path)


class FakeEmbeddings:
    def __init__(self):
        self.identity = {"model": "test semantic embeddings"}

    def __call__(self, texts):
        return np.tile([1, 0, 0], (len(texts), 1))


class FakeLLM:
    model, base_url = "test-model", "http://test/v1"
    timeout, request_retries, output_retries = 600, 1, 5

    def __init__(self):
        self.calls = []
        self.fail = False

    def ask(self, messages, stage, **_kwargs):
        payload = json.loads(messages[1]["content"])
        self.calls.append((stage, payload))
        if self.fail:
            raise TimeoutError("Injected transport failure")
        if stage == "rubric_extract":
            return {
                "criteria": [
                    {"name": f"quality {i}", "description": DESCRIPTION, "model_a_score": 1, "model_b_score": 9}
                    for i in range(3)
                ]
            }
        if stage == "rubric_subgroup":
            groups = defaultdict(list)
            for criterion in payload["criteria"]:
                groups[criterion["name"]].append(criterion["id"])
            return {
                "subgroups": [group for group in groups.values() if len(group) > 1],
                "unmerged": [group[0] for group in groups.values() if len(group) == 1],
            }
        if stage == "rubric_describe":
            return {"name": payload["criteria"][0]["name"], "description": DESCRIPTION}
        if stage == "rubric_genericity":
            return {"criteria": [{"id": c["id"], "reusable": True} for c in payload["criteria"]]}
        if stage == "rubric_fit":
            return {"scores": [{"id": c["id"], "applicable": True, "a": 1, "b": 9} for c in payload["criteria"]]}
        if stage == "rubric_judge":
            return {"choice": "B"}
        raise AssertionError(stage)


def fixture_pipeline(tmp_path, **kwargs):
    dataset = tmp_path / "dataset"
    records = []
    for benchmark in BENCHMARKS:
        for split in ("train", "val"):
            pair_id = f"{benchmark}-{split}"
            visible = {
                "task": f"public request {pair_id}",
                "context": {},
                "public_tools": [],
                "common_prefix": [{"action": "shared"}],
                "trajectory_a": [{"observation": "unsupported"}],
                "trajectory_b": [{"observation": "supported"}],
            }
            write_json(dataset / "pairs" / f"{pair_id}.json", visible)
            records.append(
                {
                    "id": pair_id,
                    "task_id": pair_id,
                    "benchmark": benchmark,
                    "split": split,
                    "label": "B",
                    "input_sha256": digest(visible),
                }
            )
    write_json(dataset / "manifest.json", {"records": records})
    config = RubricConfig(min_support=1, min_applicable=1, target_size=3, **kwargs)
    return RubricPipeline(dataset, tmp_path / "output", FakeLLM(), FakeEmbeddings(), config)


def test_batch_workflow_keeps_validation_out_of_discovery_and_selection_and_resumes(tmp_path):
    pipeline = fixture_pipeline(tmp_path)
    rubrics = pipeline.run()
    assert len(rubrics["preference_repaired"]) == 3
    assert all(row["support"] == 3 for row in rubrics["discovered"])
    calls = list(pipeline.llm.calls)
    for stage, payload in calls:
        if stage in ("rubric_extract", "rubric_fit"):
            assert "-val" not in payload["comparison"]["task"]
        if stage != "rubric_extract":
            assert "winner" not in payload and "label" not in payload
    validation = json.loads((pipeline.output / "validation.json").read_text())
    for by_benchmark in validation["by_rubric"].values():
        for stats in by_benchmark.values():
            assert stats["accuracy"] == 1
            assert stats["swapped_accuracy"] == 0
            assert stats["order_flip_rate"] == 1
    pipeline.run()
    assert pipeline.llm.calls == calls
    assert sum(stage == "rubric_judge" for stage, _payload in calls) == 12


def test_empty_preference_selection_retains_threshold_and_runs_real_validation(tmp_path):
    pipeline = fixture_pipeline(tmp_path)
    original = pipeline.llm.ask

    def ask(messages, stage, **kwargs):
        value = original(messages, stage, **kwargs)
        if stage == "rubric_fit":
            # Equal scores offer no preference signal; they must not be counted as wins.
            for row in value["scores"]:
                row["a"] = row["b"] = 5
        return value

    pipeline.llm.ask = ask
    rubrics = pipeline.run()
    assert rubrics["preference_repaired"] == []
    assert len(rubrics["discovered"]) == 3
    fit = json.loads((pipeline.output / "preference_fit.json").read_text())
    assert fit["selection_status"] == "no_qualifying_criteria"
    assert fit["threshold"] == 0.6
    assert fit["selected_count"] == 0
    assert fit["max_preference_fit"] == 0
    assert all(row["ties"] == 3 and not row["keep"] for row in fit["statistics"])
    report = json.loads((pipeline.output / "validation.json").read_text())
    assert set(report["by_rubric"]) == {"no_rubric", "discovered"}
    assert report["preference_selection"]["skipped_variants"] == ["preference_repaired"]
    assert all(row["status"] == "completed" for row in report["results"])
    status = json.loads((pipeline.output / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["preference_selection"]["status"] == "no_qualifying_criteria"
    calls = list(pipeline.llm.calls)
    pipeline.run()
    assert pipeline.llm.calls == calls


def test_request_failures_are_resumable_and_do_not_become_zero_preference_scores(tmp_path):
    pipeline = fixture_pipeline(tmp_path)
    pipeline.llm.fail = True
    with pytest.raises(RubricStageError, match="3/3"):
        pipeline.extract()
    pipeline.llm.fail = False
    with pytest.raises(RubricStageError):
        pipeline.extract()
    assert not (pipeline.output / "preference_fit.json").exists()
    pipeline.retry_errors = True
    assert len(pipeline.extract()) == 9


def test_schema_recovery_preserves_constraints_and_reuses_validated_result(tmp_path):
    pipeline = fixture_pipeline(tmp_path)
    requests = []

    def ask(messages, stage, **kwargs):
        requests.append((messages, stage, kwargs))
        return {"valid": len(requests) > 1}

    def parse(value):
        if not value["valid"]:
            raise ValueError("Invalid criterion description length")
        return value

    pipeline.llm.ask = ask
    result = pipeline.call("schema_test", "pair", "original constraints", {"public": "evidence"}, parse)
    assert result == {"valid": True}
    assert len(requests) == 2
    assert requests[1][0][:2] == requests[0][0]
    assert requests[1][0][-2] == {"role": "assistant", "content": '{"valid": false}'}
    assert "Invalid criterion description length" in requests[1][0][-1]["content"]
    assert requests[1][1] == "rubric_schema_test_format_repair"
    cached = json.loads(next((pipeline.output / "jobs/schema_test").glob("*.json")).read_text())
    assert len(cached["validation_recovery"]) == 1
    assert pipeline.call("schema_test", "pair", "original constraints", {"public": "evidence"}, parse) == result
    assert len(requests) == 2


def test_schema_recovery_stops_after_two_corrections_and_preserves_errors(tmp_path):
    pipeline = fixture_pipeline(tmp_path)
    calls = []

    def ask(*args, **kwargs):
        calls.append(args)
        return {"invalid": True}

    def parse(value):
        raise ValueError("Still invalid")

    pipeline.llm.ask = ask
    with pytest.raises(ValueError, match="Still invalid"):
        pipeline.call("schema_test", "pair", "constraints", {}, parse)
    assert len(calls) == 3

    cached = json.loads(next((pipeline.output / "jobs/schema_test").glob("*.json")).read_text())
    assert cached["status"] == "error"
    assert len(cached["validation_recovery"]) == 3
    with pytest.raises(RubricStageError, match="Cached failed job"):
        pipeline.call("schema_test", "pair", "constraints", {}, parse)
    assert len(calls) == 3

    pipeline.retry_errors = True
    resumed = []

    def corrected(*args, **kwargs):
        resumed.append(args)
        return {"valid": True}

    pipeline.llm.ask = corrected
    result = pipeline.call(
        "schema_test", "pair", "constraints", {}, lambda value: value if value.get("valid") else parse(value)
    )
    assert result == {"valid": True}
    assert len(resumed) == 1
    assert resumed[0][1] == "rubric_schema_test_format_repair"


def test_changed_input_hash_is_rejected(tmp_path):
    pipeline = fixture_pipeline(tmp_path)
    record = pipeline.train[0]
    path = pipeline.dataset / "pairs" / f"{record['id']}.json"
    changed = json.loads(path.read_text())
    changed["trajectory_a"].append({"action": "tampered"})
    write_json(path, changed)
    with pytest.raises(ValueError, match="hash mismatch"):
        RubricPipeline(pipeline.dataset, pipeline.output, pipeline.llm, pipeline.embeddings, pipeline.config)


@pytest.mark.parametrize(
    "value",
    [
        {"subgroups": [["a", "b"]], "unmerged": ["b"]},
        {"subgroups": [["a"]], "unmerged": ["b"]},
        {"subgroups": [["a", "missing"]], "unmerged": []},
    ],
)
def test_consolidation_rejects_missing_duplicated_unknown_or_singleton_subgroup_ids(value):
    with pytest.raises(ValueError):
        validate_partition(value, ["a", "b"])


def test_oversized_cosine_clusters_are_recursively_split_without_losing_candidates():
    clusters = cosine_clusters(np.tile([1, 0, 0], (80, 1)), 5)
    assert max(map(len, clusters)) <= 10
    assert sorted(i for cluster in clusters for i in cluster) == list(range(80))


def test_partition_normalization_preserves_singletons_and_caps_equivalent_groups():
    ids = [f"r{i}" for i in range(8)]
    normalized = normalize_partition({"subgroups": [ids[:6], [ids[6]]], "unmerged": [ids[7]]}, ids)
    assert normalized["subgroups"] == [ids[:5]]
    assert set(normalized["unmerged"]) == set(ids[5:])
    with pytest.raises(ValueError, match="exactly once"):
        normalize_partition({"subgroups": [ids[:6], [ids[6]]], "unmerged": [ids[6]]}, ids)


def test_mmr_selects_supported_diverse_criteria():
    criteria = [{"id": "a", "support": 10}, {"id": "b", "support": 10}, {"id": "c", "support": 9}]
    selected = mmr_select(criteria, [[1, 0], [1, 0.001], [0, 1]], 2, 0.5)
    assert [row["id"] for row in selected] == ["a", "c"]


def test_omitted_partition_ids_are_carried_unchanged_without_inferred_merges():
    value = {"subgroups": [["r1", "r2"]], "unmerged": []}
    result = normalize_partition(value, ["r1", "r2", "r3", "r4"])
    assert result["subgroups"] == [["r1", "r2"]]
    assert result["unmerged"] == ["r3", "r4"]
    assert result["carried_missing"] == ["r3", "r4"]
    assert value == {"subgroups": [["r1", "r2"]], "unmerged": []}
    assert normalize_partition({"subgroups": [], "unmerged": []}, ["a"])["unmerged"] == ["a"]
    with pytest.raises(ValueError, match="unknown"):
        normalize_partition({"subgroups": [["r1", "invented"]], "unmerged": []}, ["r1", "r2"])
    with pytest.raises(ValueError, match="duplicated"):
        normalize_partition({"subgroups": [["r1", "r1"]], "unmerged": []}, ["r1", "r2"])


def test_preference_ties_are_not_wins_and_inapplicable_pairs_are_excluded():
    rows = [
        {"pair_id": "win", "applicable": True, "a": 9, "b": 1},
        {"pair_id": "tie", "applicable": True, "a": 5, "b": 5},
        {"pair_id": "na", "applicable": False, "a": 0, "b": 0},
    ]
    assert preference_statistics(rows, {"win": "A", "tie": "A", "na": "B"}) == {
        "applicable": 2,
        "wins": 1,
        "ties": 1,
        "preference_fit": 0.5,
    }
    with pytest.raises(ValueError, match="integers"):
        parse_scores({"scores": [{"id": "x", "applicable": True, "a": True, "b": 8}]}, ["x"])
    with pytest.raises(ValueError, match="exactly"):
        parse_scores({"scores": [{"id": "wrong", "applicable": True, "a": 7, "b": 8}]}, ["x"])


def test_alpha_identical_varying_judges_and_undefined_constant_ratings():
    assert ordinal_alpha([[1, 1, 1], [5, 5, 5], [10, 10, 10]]) == 1
    assert ordinal_alpha([[1, 10, 5], [5, 1, 10], [10, 5, 1]]) < 0
    assert ordinal_alpha([[5, 5, 5], [5, 5, 5]]) is None


def test_repair_requires_measured_gain_and_all_acceptance_gates():
    baseline = {"alpha": 0.6, "vfr": 0.4, "preference_fit": 0.7}
    assert accepts_repair(baseline, {"alpha": 0.6, "vfr": 0.3, "preference_fit": 0.7})
    assert not accepts_repair(baseline, {"alpha": 0.5, "vfr": 0.3, "preference_fit": 0.7})
    assert not accepts_repair(baseline, {"alpha": 0.6, "vfr": 0.3, "preference_fit": 0.6})
    assert not accepts_repair(baseline, {"alpha": None, "vfr": 0.3, "preference_fit": 0.7})
    assert not accepts_repair(baseline, {"alpha": 0.8, "vfr": 0.4, "preference_fit": 0.8})
    frontier = pareto_repairs([{"metrics": {"alpha": 0.6, "vfr": 0.2}}, {"metrics": {"alpha": 0.7, "vfr": 0.1}}])
    assert len(frontier) == 1 and frontier[0]["metrics"]["alpha"] == 0.7


def test_repair_word_budget_and_clause_count():
    assert repair_budget({"name": "evidence", "description": DESCRIPTION})["description"] == DESCRIPTION
    with pytest.raises(ValueError, match="two anti"):
        repair_budget({"name": "evidence", "description": DESCRIPTION, "anti_gaming": ["x", "y", "z"]})


def test_audit_reserves_task_groups_and_keeps_dev_calibration_and_final_val_disjoint(tmp_path):
    original = fixture_pipeline(tmp_path)
    records = list(original.records)
    for benchmark in BENCHMARKS:
        sample = next(r for r in records if r["benchmark"] == benchmark and r["split"] == "train")
        for index in range(3):
            extra = {**sample, "id": f"{benchmark}-extra-{index}", "task_id": f"{benchmark}-extra-{index}"}
            write_json(original.dataset / "pairs" / f"{extra['id']}.json", original.visible[sample["id"]])
            records.append(extra)
    write_json(original.dataset / "manifest.json", {"records": records})
    pipeline = RubricPipeline(
        original.dataset,
        tmp_path / "audit_output",
        original.llm,
        original.embeddings,
        RubricConfig(min_support=1, min_applicable=1, audit_tasks_per_benchmark=2),
    )
    auditor = RubricAuditor(pipeline, [pipeline.llm], AuditConfig())
    pools = [pipeline.train, pipeline.val, auditor.calibration, auditor.dev]
    groups = [{(r["benchmark"], r["task_id"]) for r in pool} for pool in pools]
    assert all(not a & b for index, a in enumerate(groups) for b in groups[index + 1 :])
    assert all(pools)
    with pytest.raises(ValueError, match="distinct"):
        RubricAuditor(pipeline, [pipeline.llm, pipeline.llm], AuditConfig())


def test_validation_transport_failure_does_not_publish_accuracy(tmp_path):
    pipeline = fixture_pipeline(tmp_path)
    pipeline.llm.fail = True
    report = pipeline.evaluate({"empty": []})
    assert all(row["status"] == "error" for row in report["results"])
    assert all(stats["accuracy"] is None and stats["errors"] == 2 for stats in report["by_rubric"]["empty"].values())


def test_failure_packet_token_budget_includes_escaped_excerpt_and_omission_metadata():
    class CharacterTokenizer:
        def encode(self, text, **_kwargs):
            return SimpleNamespace(ids=list(text.encode()))

        def decode(self, tokens):
            return bytes(tokens).decode(errors="ignore")

    tokenizer = CharacterTokenizer()
    packet = {"examples": [{"action": "quoted evidence", "observation": "observed"}] * 100}
    bounded = bounded_failure_packet(packet, tokenizer, 512)
    assert bounded["truncated"] is True and "omission" in bounded
    assert len(tokenizer.encode(json.dumps(bounded, ensure_ascii=False)).ids) <= 512
