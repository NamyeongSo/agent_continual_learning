# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""AppWorld must persist live mutations before evaluating and resetting the DB."""

from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from exgentic.benchmarks.appworld.appworld_eval import AppWorldSession


@pytest.mark.parametrize("score_first", [True, False])
@pytest.mark.parametrize("budget_failure", [True, False])
def test_scoring_saves_before_evaluation_and_close_preserves_snapshot(
    tmp_path, monkeypatch, score_first, budget_failure,
):
    events = []
    task_output = tmp_path / "experiment" / "tasks" / "one"
    live = {"changes": "real Gmail mutations"}

    def save():
        events.append("save")
        (task_output / "dbs").mkdir(parents=True, exist_ok=True)
        (task_output / "dbs" / "gmail.jsonl").write_text(live["changes"])

    def completed():
        events.append("completed")
        return True

    def evaluate_task(**_kwargs):
        events.append("evaluate")
        assert (task_output / "dbs" / "gmail.jsonl").read_text() == "real Gmail mutations"
        # The real evaluator rebinds the DB and resets its live change tracking.
        live["changes"] = ""
        return SimpleNamespace(
            pass_percentage=100.0, pass_count=10, fail_count=0, num_tests=10,
            difficulty=1, success=True, to_dict=lambda **_: {"success": True},
        )

    modules = {
        name: ModuleType(name)
        for name in ("appworld", "appworld.apps", "appworld.apps.lib", "appworld.apps.lib.models",
                     "appworld.apps.lib.models.db", "appworld.evaluator")
    }
    modules["appworld.apps.lib.models.db"].CachedDBHandler = SimpleNamespace(
        reset=lambda *_: events.append("reset"),
    )
    modules["appworld.evaluator"].evaluate_task = evaluate_task
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    session = object.__new__(AppWorldSession)
    session._session_id = "test-session"
    session._task_id = "one"
    session._experiment_name = "experiment"
    session._task_output_dir = task_output
    session._step_count = 1
    session._done = False
    session._world_closed = False
    session._world_state_saved = False
    session._cached_score = None
    session._logger = logging.getLogger(__name__)
    session._paths = SimpleNamespace(benchmark_dir=tmp_path / "benchmark")
    session._world = SimpleNamespace(save=save, task_completed=completed,
                                     close=lambda: events.append("close"))
    session.save_results = lambda _: None

    if budget_failure:
        from exgentic.core.experiment_limits import ExperimentBudgetExceededError, experiment_limits

        failure = ExperimentBudgetExceededError("input_tokens", limit=100_000, actual=100_001, turn=1)
        session.fail_experiment_budget(failure.details, experiment_limits("appworld"))
    elif score_first:
        assert session.score().is_finished is True
    session.close()
    assert session.score().score == (0.0 if budget_failure else 1.0)
    session.close()
    assert events == (["completed", "save", "close"] if budget_failure
                      else ["completed", "save", "evaluate", "reset", "close"])
    assert (session.paths.benchmark_dir / "task_output" / "dbs" / "gmail.jsonl").read_text() == (
        "real Gmail mutations"
    )


def test_failed_save_does_not_score_or_mark_snapshot_as_saved():
    session = object.__new__(AppWorldSession)
    session._cached_score = None
    session._world_state_saved = False
    session._done = False
    session._world = SimpleNamespace(
        task_completed=lambda: True,
        save=lambda: (_ for _ in ()).throw(OSError("cannot save task DB")),
    )
    with pytest.raises(OSError, match="cannot save task DB"):
        session.score()
    assert session._world_state_saved is False
    assert session._cached_score is None


def test_opt_in_turn_cap_scores_final_state_and_retains_real_tests(tmp_path, monkeypatch):
    import json

    events = []
    modules = {name: ModuleType(name) for name in (
        "appworld", "appworld.apps", "appworld.apps.lib", "appworld.apps.lib.models",
        "appworld.apps.lib.models.db", "appworld.evaluator",
    )}
    tracker = {"success": False, "difficulty": 2, "num_tests": 3,
               "passes": [{"requirement": "one"}, {"requirement": "two"}],
               "failures": [{"requirement": "three"}]}

    def evaluate_task(**_kwargs):
        events.append("evaluate")
        return SimpleNamespace(pass_percentage=66.67, pass_count=2, fail_count=1, num_tests=3,
                               difficulty=2, success=False, to_dict=lambda **_: tracker)

    modules["appworld.evaluator"].evaluate_task = evaluate_task
    modules["appworld.apps.lib.models.db"].CachedDBHandler = SimpleNamespace(reset=lambda *_: None)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    session = object.__new__(AppWorldSession)
    session._session_id, session._task_id, session._experiment_name = "turncap", "task", "test"
    session._world_state_saved, session._done, session._cached_score = False, False, None
    session._world = SimpleNamespace(task_completed=lambda: False, save=lambda: events.append("save"))
    session._logger = logging.getLogger(__name__)
    session._paths = SimpleNamespace(benchmark_dir=tmp_path / "benchmark")
    session.save_results = lambda _: None
    score = session.fail_experiment_budget(
        {"reason": "max_turns", "limit": 40, "actual": 41, "turn": 41},
        {"version": "test", "turn_limit_scoring": "official_pct_at_limit"},
    )
    assert score.score == 2 / 3 and score.is_finished is False
    assert score.session_metadata["turn_limit_reached"] is True
    assert not score.session_metadata.get("experiment_budget_exceeded")
    assert score.session_metadata["test_tracker"] == tracker
    saved = json.loads((tmp_path / "benchmark/scores.json").read_text())
    assert saved["test_tracker"] == tracker
    assert session.score() is score
    assert events == ["save", "evaluate"]
    assert session._world_state_saved is True
    assert session._cached_score is score
