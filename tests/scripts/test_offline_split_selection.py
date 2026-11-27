# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import pytest

from scripts.utils.plot_offline_split_comparison import select_samples


def fixture():
    from scripts.utils.plot_checkpoint_learning_curves import BENCHMARKS, LABELS

    config = {"seeds": [42], "offline_early": {
        "points": {b: [0, 10] for b in BENCHMARKS},
        "reuse_training_val_best_variants": ["gepa_val", "skillopt_val"]}}
    scores, audits = [], {}
    for variant in LABELS:
        for benchmark in BENCHMARKS:
            audits[variant, benchmark, 42] = {"final_cp": 10, "frozen_policy_matches_final_cp": True,
                                              "training_val_best_score": .75}
            for point in [0, 10]:
                for split in ["train", "val", "test"]:
                    value = (.9 if point == 0 else .5) if split == "val" else (.4 if point == 0 else .6)
                    scores.append({"variant": variant, "benchmark": benchmark, "seed": 42,
                                   "checkpoint": point, "split": split, "score": value, "measured": 5,
                                   "total": 5, "errors": 0, "canonical": f"{variant}_{benchmark}_{point}",
                                   "checkpoint_id": f"{variant}_{benchmark}_{point}"})
    return config, scores, audits


def test_training_val_variants_reuse_final_test_without_independent_val_reselection():
    config, scores, audit = fixture()
    result = select_samples(scores, config, audit)
    for row in result:
        if row["metric"] != "test_early":
            continue
        if row["variant"] in ["gepa_val", "skillopt_val"]:
            final = next(r for r in result if (r["variant"], r["benchmark"], r["metric"]) ==
                         (row["variant"], row["benchmark"], "test"))
            assert row["score"] == final["score"] == .6
            assert row["checkpoint_id"] == final["checkpoint_id"]
            assert row["test_result_reused"] and row["selection_val"] == .75
        else:
            assert row["checkpoint"] == 0 and row["score"] == .4 and not row["test_result_reused"]


def test_missing_final_test_remains_missing_and_unverified_training_policy_is_rejected():
    config, scores, audit = fixture()
    final = next(r for r in scores if (r["variant"], r["benchmark"], r["checkpoint"], r["split"]) ==
                 ("gepa_val", "bfcl", 10, "test"))
    final.update(measured=4, errors=1)
    result = select_samples(scores, config, audit)
    early = next(r for r in result if (r["variant"], r["benchmark"], r["metric"]) ==
                 ("gepa_val", "bfcl", "test_early"))
    assert early["score"] is None and early["checkpoint"] == 10
    audit["gepa_val", "bfcl", 42]["frozen_policy_matches_final_cp"] = False
    with pytest.raises(ValueError, match="verified frozen-policy audit"):
        select_samples(scores, config, audit)
