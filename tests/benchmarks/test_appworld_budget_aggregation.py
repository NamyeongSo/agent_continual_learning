# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Budget failures must remain in AppWorld's task and scenario denominators."""

import json

import pytest

from exgentic.core.context import run_scope
from exgentic.core.experiment_limits import ExperimentBudgetExceededError, experiment_limits
from exgentic.core.types import SessionIndex
from exgentic.utils.paths import get_run_paths


def test_appworld_metric_counts_budget_failures_without_official_evaluation(tmp_path):
    pytest.importorskip("appworld")
    from exgentic.benchmarks.appworld.appworld_eval import AppWorldEvaluator, AppWorldSession

    with run_scope(run_id="budget-aggregate", output_dir=str(tmp_path)):
        failed = object.__new__(AppWorldSession)
        failed._session_id = "failed"
        failure = ExperimentBudgetExceededError("max_turns", limit=40, actual=41, turn=41)
        failed.fail_experiment_budget(failure.details, experiment_limits("appworld"))

        passed_paths = get_run_paths().session("passed")
        passed_paths.benchmark_dir.mkdir(parents=True)
        (passed_paths.benchmark_dir / AppWorldSession.SCORES_FILE_NAME).write_text(json.dumps({
            "test_tracker": {
                "difficulty": 1, "passes": [{"requirement": "fixture", "label": "no_op_pass"}], "failures": [],
            },
        }))
        evaluator = object.__new__(AppWorldEvaluator)
        evaluator._ensure_appworld_root = lambda: None
        result = evaluator.aggregate_sessions([
            SessionIndex(task_id="7238049_1", session_id="failed"),
            SessionIndex(task_id="7238049_2", session_id="passed"),
        ])
    assert result.total_tasks == 2
    assert result.score == 0.5
