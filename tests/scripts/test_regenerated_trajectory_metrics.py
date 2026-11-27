# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Check whole-sequence scoring, independent controls and historical-action exclusion."""

import copy
import json
from pathlib import Path

import pytest

from scripts.premise.analyze_regenerated_trajectories import (
    aggregate,
    build_complete_sequence_payloads,
    build_payloads,
    compare_steps,
    concatenate,
    score_payload,
    score_payload_cached,
)
from scripts.premise.trajectory_text_metrics import LexicalScorer, action_tokens


def action(value):
    return [{"name": "act", "arguments": {"value": value}}]


class DummyLexical:
    def score(self, predicted, reference, mode):
        return {"bleu1": 0, "bleu4": 0, "meteor": 0}


def make_bank():
    cases = [{"id": "one", "benchmark": "bfcl"}, {"id": "two", "benchmark": "bfcl"}]
    trajectories = [{"benchmark": "bfcl", "task": c["id"], "original_actions": action("HISTORICAL_POISON"),
                     "steps": [{"case": c["id"], "action_index": 0,
                                "original_actions": action("HISTORICAL_POISON")}]} for c in cases]
    prepared = {"protocol": "full_trajectory_regenerated_contexts_cp10_t1_v1", "cases": cases,
                "trajectories": trajectories, "checkpoints": {"bfcl": [0, 10, 20]}, "repeats": 2,
                "model": "test", "temperature": 1}
    rows = []
    for case in cases:
        rows.append({"case": case["id"], "context": "cp000", "repeat": -1,
                     "status": "completed", "actions": action("REFERENCE_REGENERATED")})
        rows.extend({"case": case["id"], "context": f"cp{p:03d}", "repeat": k,
                     "status": "completed", "actions": action(f"{p}_{k}")}
                    for p in (0, 10, 20) for k in range(2))
    return prepared, rows


def test_concatenate_scores_once_and_matches_actions_across_prefix_positions():
    first, second = action("a"), action("b")
    result = compare_steps([second, first], [first, second], "strict", DummyLexical())
    assert result["f1"] == 1 and result["prefix_f1"] == 0
    assert result["exact_action_sequence"] == 0
    assert concatenate([second, first]) == [*second, *first]
    with pytest.raises(ValueError, match="every"):
        concatenate([first, None])
    with pytest.raises(ValueError, match="identical original prefixes"):
        compare_steps([first], [first, second], "strict", DummyLexical())


def test_historical_actions_never_enter_comparison_banks_and_fixed_reference_is_independent():
    prepared, rows = make_bank()
    payloads, coverage, counts = build_payloads(prepared, rows)
    assert coverage["bfcl"]["complete_tasks"] == 2 and counts["missing"] == 0
    payload = next(p for p in payloads if p["kind"] == "cp0_reference" and p["later"] == 10)
    assert payload["reference_banks"] == [[action("REFERENCE_REGENERATED")]]
    assert payload["prediction_banks"] == [[action("10_0")], [action("10_1")]]
    altered = copy.deepcopy(prepared)
    for trajectory in altered["trajectories"]:
        trajectory["original_actions"] = action("CHANGED_OLD_ACTION")
        trajectory["steps"][0]["original_actions"] = action("CHANGED_OLD_ACTION")
    assert build_payloads(altered, rows)[0] == payloads
    prepared["protocol"] = "recorded_original"
    with pytest.raises(ValueError, match="exclusively"):
        build_payloads(prepared, rows)


def test_common_complete_trajectory_mask_and_no_zero_fill():
    prepared, rows = make_bank()
    rows[-1].update(status="error", error_type="timeout")
    payloads, coverage, counts = build_payloads(prepared, rows)
    assert counts["errors"] == 1 and coverage["bfcl"]["included_tasks"] == ["one"]
    assert all(p["task"] == "one" for p in payloads)
    with pytest.raises(ValueError, match="missing"):
        build_payloads(prepared, rows[:-1])
    assert build_payloads(prepared, rows[:-1], False)[2]["missing"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        build_payloads(prepared, [*rows, rows[0]])


def test_secondary_view_preserves_only_whole_sequences_and_never_self_compares():
    prepared, rows = make_bank()
    rows[-1].update(status="error", error_type="timeout")
    strict_payloads, strict_coverage, _ = build_payloads(prepared, rows)
    assert strict_coverage["bfcl"]["complete_tasks"] == 1
    payloads, coverage = build_complete_sequence_payloads(prepared, rows)
    assert coverage["bfcl"]["included_tasks"] == ["one", "two"]
    root = next(p for p in payloads if p["task"] == "two" and p["later"] == 0)
    assert root["reference_repeat_ids"] == [-1, 0, 1]
    self_scores = score_payload(root, DummyLexical())[0]
    assert self_scores["sample_pairs"] == 6
    assert all(s["reference_repeat"] != s["prediction_repeat"] for s in self_scores["repeats"])
    later = next(p for p in payloads if p["task"] == "two" and p["later"] == 20)
    assert later["prediction_repeat_ids"] == [0]  # Failed full repeat1 is absent, never zero/partially filled.
    assert score_payload(later, DummyLexical())[0]["sample_pairs"] == 3
    results = [r for p in [*strict_payloads, *payloads] for r in score_payload(p, DummyLexical())]
    result = aggregate(results, strict_coverage, prepared)
    later = next(r for r in result["cp0_complete_sequences"] if r["later"] == 20 and r["mode"] == "strict")
    assert later["tasks"] == 2 and later["min_sample_pairs_per_task"] == 3
    assert later["max_sample_pairs_per_task"] == 6 and later["sample_pairs_per_task"] is None
    # If both repetitions at a later checkpoint fail, exclude the task at all distances.
    rows[-2].update(status="error", error_type="timeout")
    assert build_complete_sequence_payloads(prepared, rows)[1]["bfcl"]["included_tasks"] == ["one"]


def test_same_context_uses_distinct_repeats_and_positive_distance_is_directional():
    payload = {"benchmark": "bfcl", "task": "task", "prefixes": 1, "kind": "pairwise",
               "earlier": 0, "later": 0, "reference_banks": [[action("a")], [action("b")]],
               "prediction_banks": [[action("a")], [action("b")]]}
    strict = score_payload(payload, DummyLexical())[0]
    assert strict["sample_pairs"] == 2 and strict["f1"] == 0
    assert all(s["reference_repeat"] != s["prediction_repeat"] for s in strict["repeats"])
    payload.update(later=10, reference_banks=[[action("a")], [action("a")]],
                   prediction_banks=[[[*action("a"), *action("b")]]] * 2)
    strict = score_payload(payload, DummyLexical())[0]
    assert strict["precision"] == .5 and strict["recall"] == 1 and strict["sample_pairs"] == 4


def test_distance_aggregation_keeps_task_then_checkpoint_pair_weights():
    prepared, rows = make_bank()
    payloads, coverage, _ = build_payloads(prepared, rows)
    results = [r for payload in payloads for r in score_payload(payload, DummyLexical())]
    # Different task scores/lengths must receive equal task weights.
    for row in results:
        row["f1"] = 1 if row["task"] == "one" else 0
        row["prefix_f1"] = 0
        row["prefixes"] = 1 if row["task"] == "one" else 100
    result = aggregate(results, coverage, prepared)
    assert {r["distance"]: r["checkpoint_pairs"] for r in result["distances"] if r["mode"] == "strict"} == {
        0: 3, 10: 2, 20: 1}
    assert all(r["f1"] == .5 and r["concat_gain_f1"] == .5 for r in result["distances"])
    assert not result["historical_actions_used_for_scoring"]


def test_lexical_representation_includes_actual_code_but_loose_only_names():
    first = [{"name": "execute_python", "arguments": {"code": "x = 1\nprint(x)"}}]
    second = [{"name": "execute_python", "arguments": {"code": "x = 2\nprint(x)"}}]
    assert action_tokens(first, "strict") != action_tokens(second, "strict")
    assert action_tokens(first, "loose") == action_tokens(second, "loose") == ("execute_python",)
    assert action_tokens([*first, *first], "loose") == ("execute_python", "execute_python")
    assert "\\n" not in action_tokens(first, "strict") and "print" in action_tokens(first, "strict")
    with pytest.raises(ValueError, match="representation"):
        action_tokens(first, "unknown")


def test_deterministic_cache_is_versioned_and_rejects_changed_results(tmp_path, monkeypatch):
    import scripts.premise.analyze_regenerated_trajectories as analysis

    class CountingScorer(DummyLexical):
        calls = 0

        def score(self, predicted, reference, mode):
            self.calls += 1
            return super().score(predicted, reference, mode)

    prepared, draws = make_bank()
    payload = build_payloads(prepared, draws)[0][0]
    scorer = CountingScorer()
    monkeypatch.setattr(analysis, "SCORER", scorer)
    first = score_payload_cached(payload, tmp_path, "version1")
    count = scorer.calls
    assert score_payload_cached(payload, tmp_path, "version1") == first
    assert scorer.calls == count
    score_payload_cached(payload, tmp_path, "version2")
    assert scorer.calls > count
    path = next(p for p in tmp_path.glob("*.json") if json.loads(p.read_text())["signature"] == "version1")
    changed = json.loads(path.read_text())
    changed["rows"][0]["f1"] = 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="checksum mismatch"):
        score_payload_cached(payload, tmp_path, "version1")


@pytest.fixture(scope="module")
def lexical():
    folder = Path("outputs/group_relative_update_distance10_qwen38_live_unlimited_20261113/bleu_meteor")
    resources = [folder / "dependencies/nltk/__init__.py", folder / "nltk_data/corpora/wordnet.zip"]
    if not all(p.exists() for p in resources):
        pytest.skip("Optional pinned NLTK/WordNet analysis bundle is not installed")
    return LexicalScorer(folder / "dependencies", folder / "nltk_data")


def test_pinned_nltk_short_sentence_scores_and_empty_extension(lexical):
    assert lexical.score_tokens(("a",), ("a",)) == (1, 1, .5)
    assert lexical.score_tokens((), ()) == (1, 1, 1)
    assert lexical.score_tokens(("a",), ()) == (0, 0, 0)
    assert lexical.score_tokens((), ("a",)) == (0, 0, 0)
    assert lexical.score_tokens(("zzzz_unique",), ("yyyy_unique",)) == (0, 0, 0)


def test_meteor_cached_helpers_match_official_default_example(lexical):
    from nltk.translate.meteor_score import single_meteor_score

    prediction = tuple("It is a guide to action which ensures that the military "
                       "always obeys the commands of the party".split())
    reference = tuple("It is a guide to action that ensures that the military will forever heed Party commands".split())
    result = lexical.score_tokens(prediction, reference)[2]
    assert result == pytest.approx(.6944, abs=1e-4)
    assert result == single_meteor_score(reference, prediction)
    assert lexical.score_tokens(("cars",), ("car",))[2] == .5
    # The reference implementation's stemming stage can change WordNet lookup tokens.
    for prediction, reference in ((("automobile",), ("car",)), (("car",), ("auto",))):
        assert lexical.score_tokens(prediction, reference)[2] == single_meteor_score(reference, prediction)
