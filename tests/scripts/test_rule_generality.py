# SPDX-License-Identifier: Apache-2.0
"""Check context-balanced maximum similarity and exclusion of the complete source context."""

import json
import sqlite3

import numpy as np
import pytest

from scripts.ace.analyze_rule_generality import collect_contexts, context_maxima, sha


def test_average_is_over_context_maxima_not_over_all_target_rules_or_global_maximum():
    vectors = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
    maxima, nearest = context_maxima(vectors, [1, 3, 1], block_size=2)
    assert np.isnan(maxima[0, 0])
    assert maxima[0, 1] == 1 and maxima[0, 2] == 0
    assert np.nanmean(maxima[0]) == .5  # Equal target-context weight, despite three rules in context B.
    assert nearest[0, 0] == -1 and nearest[0, 1] == 1


def test_entire_source_context_is_excluded():
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    maxima, nearest = context_maxima(vectors, [2, 1])
    assert np.isnan(maxima[:2, 0]).all()
    assert (nearest[:2, 0] == -1).all()
    assert (maxima[:2, 1] == 0).all()
    assert maxima[2, 0] == 0


def test_blocked_reduction_and_nearest_rule_ids_match_brute_force():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(7, 13)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    sizes = [2, 3, 2]
    context = np.repeat(np.arange(3), sizes)
    for block in (1, 3, 128):
        maxima, nearest = context_maxima(vectors, sizes, block_size=block)
        for i in range(7):
            for target in range(3):
                if target == context[i]:
                    assert np.isnan(maxima[i, target]) and nearest[i, target] == -1
                else:
                    indices = np.flatnonzero(context == target)
                    scores = vectors[indices] @ vectors[i]
                    assert np.isclose(maxima[i, target], scores.max(), atol=1e-6)
                    assert nearest[i, target] == indices[scores.argmax()]
        assert np.isfinite(maxima).sum() == 7 * 2


@pytest.mark.parametrize("sizes", [[3], [0, 3], [1, 1]])
def test_invalid_or_empty_context_partitions_are_not_imputed_zero(sizes):
    with pytest.raises(ValueError):
        context_maxima(np.array([[1, 0]] * 3, dtype=np.float32), sizes)


def test_non_normalized_vectors_are_rejected():
    with pytest.raises(ValueError, match="normalized"):
        context_maxima(np.array([[2, 0], [0, 1]], dtype=np.float32), [1, 1])


def test_collector_deduplicates_reused_context_and_preserves_selection_memberships(tmp_path):
    def write(path, row):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row))

    root = tmp_path / "reflection"
    initial = "## RULES\n\n## OTHERS"
    write(root / "protocol.json", {"initial_playbook": initial, "playbook_edit_mode": "add_only"})
    source = root / "stage2/task_r0"
    capture = source / "capture.json"
    write(capture, {"playbook": initial})
    reflection = source / "reflection.json"
    write(reflection, {"reflection_for_curator": "Reflection text", "playbook_after_tags": initial,
                       "source_capture_sha256": sha(capture.read_bytes())})
    updated = source / "updated_context.txt"
    updated.write_text("## RULES\n\n[r-00001] helpful=0 harmful=0 :: Applied rule.\n## OTHERS")
    result = source / "result.json"
    write(result, {"stage": 2, "task_id": "task", "arm": "without_gt", "source_repeat": 0,
                   "source_capture_path": str(capture), "updated_context_sha256": sha(updated.read_bytes())})
    candidate = {"independent_update_path": str(updated), "independent_result_path": str(result),
                 "source_path": str(reflection), "reflection_sha256": sha("Reflection text"),
                 "independent_update_sha256": sha(updated.read_bytes()), "source_repeat": 0,
                 "rules": [{"id": "r-00001", "content": "Applied rule."}]}
    database = tmp_path / "review.sqlite3"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE pairs (id TEXT,active INTEGER,snapshot TEXT)")
        for seed in (42, 43):
            pair = {"id": str(seed), "seed": seed, "model": "qwen38", "mode": "thinking",
                    "arm": "without_gt", "content_kind": "independent_rules", "task_id": "task",
                    "candidates": {"best_advantage": candidate, "worst_advantage": candidate}}
            conn.execute("INSERT INTO pairs VALUES (?,1,?)", (pair["id"], json.dumps(pair)))
        pair = {**pair, "id": "ignored", "model": "oss20"}
        conn.execute("INSERT INTO pairs VALUES ('ignored',1,?)", (json.dumps(pair),))
    contexts, hashes = collect_contexts(database)
    assert len(contexts) == 1 and len(contexts[0]["selection_usages"]) == 4
    assert contexts[0]["rules"] == candidate["rules"]
    assert hashes[str(updated)] == sha(updated.read_bytes())
    updated.write_text("tampered")
    with pytest.raises(AssertionError):
        collect_contexts(database)
