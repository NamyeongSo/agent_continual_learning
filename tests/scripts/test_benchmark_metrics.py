# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import pytest

from scripts.utils.benchmark_metrics import APPWORLD_METRIC, reported_score


@pytest.mark.parametrize("record,expected", [
    ({"score": 1.0, "success": False}, 0.0),
    ({"score": 0.9, "details": {"session_metrics": {"success": True}}}, 1.0),
    ({"score": 0.9, "status": "unsuccessful"}, 0.0),
    ({"score": 0.9, "status": "completed", "session_status": "unsuccessful"}, 0.0),
    ({"score": 1.0, "status": "error"}, None),
    ({"score": 0, "success": False, "status": "error"}, None),
    ({"score": 1.0}, None),
    ({"score": 0.5, "tgc": 0}, 0.0),
    ({"score": 0, "score_metric": APPWORLD_METRIC}, 0.0),
    ({"score": 1, "success": True, "budget_failure": {"kind": "turns"}}, 0.0),
])
def test_appworld_reports_official_success_without_thresholding_partial_scores(record, expected):
    assert reported_score("appworld", record) == expected


def test_other_benchmarks_keep_their_recorded_scores():
    assert reported_score("bfcl", {"score": 0.5, "success": False}) == 0.5


def test_new_ace_appworld_metrics_write_tgc_and_retain_partial_score(tmp_path, monkeypatch):
    import json
    from collections import defaultdict
    from types import SimpleNamespace

    from scripts.ace import run_experiment as ace

    monkeypatch.setattr(ace, "extract_token_counts", lambda _: (0, 0))
    monkeypatch.setattr(ace, "get_memory_tokens", lambda *_: (0, 0))
    session = SimpleNamespace(score=1.0, success=False, cost_reports={}, steps=1, action_count=1,
                              agent_cost=0, execution_time=1, status="unsuccessful")
    path = tmp_path / "online_metrics.jsonl"
    totals, by_benchmark = [], defaultdict(list)
    row = ace.record_online_metrics(path, 0, "appworld", "one", session, "isolated", 42, "model",
                                    totals, by_benchmark)
    assert row["score"] == row["tgc"] == row["cumulative_avg_score"] == 0
    assert row["partial_score"] == 1 and row["score_metric"] == APPWORLD_METRIC
    assert json.loads(path.read_text())["success"] is False
