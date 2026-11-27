# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Check separate benchmark histories on the global interleaved iteration axis."""

import json
from itertools import pairwise

import pytest

from scripts.utils.plot_interleaved_benchmark_datapoints import benchmark_datapoint_rows
from scripts.utils.plot_interleaved_online_performance import (
    fit_linear_trends,
    interleaved_cumulative,
    interleaved_windows,
    plot_benchmark_axis,
    plot_interleaved_axis,
)
from scripts.utils.plot_sequential_online_performance import COLORS, aggregate_windows, load_run


def test_datapoints_align_seeds_before_averaging_and_keep_missing_task_positions():
    schedules = {
        42: [("bfcl", 1.0), ("appworld", 0.0), ("bfcl", None), ("terminalbench2", 0.5), ("bfcl", 0.0)],
        43: [("appworld", 1.0), ("bfcl", 0.0), ("bfcl", 1.0), ("terminalbench2", 0.25), ("bfcl", 1.0)],
    }
    tasks = [
        {"method": "ace", "seed": seed, "iteration": i, "benchmark": benchmark, "task_id": str(i), "score": score}
        for seed, schedule in schedules.items()
        for i, (benchmark, score) in enumerate(schedule, 1)
    ]
    rows = benchmark_datapoint_rows(tasks)
    assert len(rows) == len(tasks)
    first = next(row for row in rows if row["seed"] == 43 and row["benchmark"] == "bfcl")
    assert first["iteration"] == 1 and first["global_iteration"] == 2
    missing = next(row for row in rows if row["seed"] == 42 and row["benchmark"] == "bfcl" and row["iteration"] == 2)
    assert missing["global_iteration"] == 3 and missing["score"] == 1
    assert missing["scored_tasks"] == 1 and missing["missing_tasks"] == 1 and missing["window_tasks"] == 2
    bfcl = [row for row in aggregate_windows(rows) if row["benchmark"] == "bfcl"]
    assert [row["iteration"] for row in bfcl] == [1, 2, 3]
    assert [row["mean"] for row in bfcl] == pytest.approx([0.5, 0.75, (0.5 + 2 / 3) / 2])
    assert all(row["seed_count"] == 2 for row in bfcl)


def test_datapoint_panels_end_at_each_benchmarks_own_count():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = [
        {"method": "ace", "benchmark": benchmark, "iteration": k, "mean": 0.2 + k * 0.001}
        for benchmark, count in (("bfcl", 50), ("appworld", 50), ("terminalbench2", 30))
        for k in range(1, count + 1)
    ]
    fig, axes = plt.subplots(1, 3)
    try:
        for ax, benchmark, count in zip(axes, ("bfcl", "appworld", "terminalbench2"), (50, 50, 30), strict=True):
            plot_benchmark_axis(
                ax, rows, benchmark, benchmark_datapoints=True, y_limits=(10, 60), linear_regression=True
            )
            assert ax.get_xlim() == (0.5, count + 0.5)
            assert ax.get_ylim() == (10, 60)
            fit = next(line for line in ax.lines if line.get_gid() == "performance-linear-regression")
            assert list(fit.get_xdata()) == [1, count]
            assert list(fit.get_ydata()) == pytest.approx([20.1, 20 + count * 0.1])
    finally:
        plt.close(fig)


def test_linear_trends_fit_percent_scores_against_global_iteration_and_keep_curves_separate():
    rows = [
        {"method": "ace", "benchmark": "bfcl", "iteration": i, "mean": score}
        for i, score in ((1, 0.12), (2, None), (3, 0.16), (5, 0.20))
    ]
    rows += [
        {"method": "premise", "benchmark": "appworld", "iteration": i, "mean": 0.4}
        for i in (1, 3, 5)
    ]
    fits = fit_linear_trends(rows)
    assert len(fits) == 2
    first, constant = fits
    assert first["method"] == "ace" and first["benchmark"] == "bfcl"
    assert first["point_count"] == 3 and first["start_iteration"] == 1 and first["end_iteration"] == 5
    assert first["slope_pp_per_iteration"] == pytest.approx(2)
    assert first["intercept_percent"] == pytest.approx(10)
    assert first["r_squared"] == pytest.approx(1)
    assert first["fitted_first_percent"] == pytest.approx(12)
    assert first["fitted_last_percent"] == pytest.approx(20)
    assert constant["slope_pp_per_iteration"] == 0 and constant["intercept_percent"] == 40
    assert constant["r_squared"] is None
    later = fit_linear_trends(rows, start_iteration=3)[0]
    assert later["point_count"] == 2 and later["start_iteration"] == 3
    assert later["slope_pp_per_iteration"] == pytest.approx(2)


def test_linear_trends_require_aggregated_unique_points_and_skip_insufficient_history():
    row = {"method": "ace", "benchmark": "bfcl", "iteration": 1, "mean": 0.5}
    assert fit_linear_trends([row]) == []
    assert fit_linear_trends([{**row, "mean": None}]) == []
    with pytest.raises(ValueError, match="one plotted score"):
        fit_linear_trends([row, row])
    with pytest.raises(ValueError, match="positive"):
        fit_linear_trends([row], start_iteration=0)
    with pytest.raises(ValueError, match="finite"):
        fit_linear_trends([{**row, "mean": float("nan")}])


def test_interleaved_histories_do_not_reset_or_mix_tasks_when_benchmarks_switch():
    order = [("bfcl", "a"), ("appworld", "b"), ("bfcl", "c"), ("terminalbench2", "d"), ("appworld", "e")]
    rows = interleaved_cumulative(order, {1: 1.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 1.0}, 5)
    curves = {b: [r["score"] for r in rows if r["benchmark"] == b] for b in ("bfcl", "appworld", "terminalbench2")}
    assert curves["bfcl"] == [1, 1, 0.5, 0.5, 0.5]
    assert curves["appworld"] == [None, 0, 0, 0, 0.5]
    assert curves["terminalbench2"] == [None, None, None, 1, 1]
    assert len(interleaved_cumulative(order, {1: 1}, 1)) == 3


def test_interleaved_missing_scores_retain_iteration_and_benchmark_counts():
    order = [("bfcl", "a"), ("appworld", "b"), ("bfcl", "c")]
    rows = interleaved_cumulative(order, {1: 1, 2: 0}, 3)
    final = next(row for row in rows if row["iteration"] == 3 and row["benchmark"] == "bfcl")
    assert final["score"] == 1
    assert final["scored_tasks"] == 1 and final["window_tasks"] == 2 and final["missing_tasks"] == 1
    with pytest.raises(ValueError, match="outside"):
        interleaved_cumulative(order, {}, 4)


@pytest.mark.parametrize("window", [5, 10, 15, 20, 30, 40])
def test_benchmark_windows_use_exact_recent_task_positions_with_missing_scores(window):
    benchmarks = ("bfcl", "appworld", "terminalbench2")
    order = [(benchmark, str(task)) for task in range(45) for benchmark in benchmarks]
    scores = {i: float(int(task) % 2) for i, (_benchmark, task) in enumerate(order, 1)}
    del scores[130]
    rows = interleaved_windows(order, scores, len(order), window, basis="benchmark")
    for benchmark in benchmarks:
        indices = [i for i, (b, _task) in enumerate(order, 1) if b == benchmark][-window:]
        values = [scores[i] for i in indices if i in scores]
        final = next(row for row in rows if row["iteration"] == len(order) and row["benchmark"] == benchmark)
        assert final["score"] == pytest.approx(sum(values) / len(values))
        assert final["window_tasks"] == window
        assert final["scored_tasks"] == len(values)
        assert final["missing_tasks"] == window - len(values)
        assert final["window_start"] == indices[0]
        curve = [row for row in rows if row["benchmark"] == benchmark]
        for previous, current in pairwise(curve):
            if current["current_task_benchmark"] != benchmark:
                assert current["score"] == previous["score"]


def test_benchmark_window_keeps_missing_tasks_in_range_without_fetching_older_scores():
    order = [("bfcl", "a"), ("appworld", "b"), ("bfcl", "c"), ("terminalbench2", "d"), ("bfcl", "e")]
    rows = interleaved_windows(order, {1: 1, 2: 0, 4: 1, 5: 0}, 5, 2, basis="benchmark")
    final = next(row for row in rows if row["iteration"] == 5 and row["benchmark"] == "bfcl")
    assert final["score"] == 0 and final["window_start"] == 3
    assert final["window_tasks"] == 2 and final["missing_tasks"] == 1
    first = next(row for row in rows if row["iteration"] == 1 and row["benchmark"] == "bfcl")
    assert first["score"] == 1 and first["window_tasks"] == 1
    assert next(row for row in rows if row["iteration"] == 1 and row["benchmark"] == "appworld")["score"] is None


def test_global_window_expires_a_benchmark_when_other_benchmarks_advance():
    order = [("bfcl", "a"), ("appworld", "b"), ("terminalbench2", "c"), ("appworld", "d")]
    rows = interleaved_windows(order, {1: 1, 2: 1, 3: 0, 4: 0}, 4, 2, basis="global")
    bfcl = [row["score"] for row in rows if row["benchmark"] == "bfcl"]
    assert bfcl == [1, 1, None, None]
    final = next(row for row in rows if row["iteration"] == 4 and row["benchmark"] == "appworld")
    assert final["score"] == 0 and final["window_start"] == 3


def test_benchmark_window_covering_all_tasks_equals_cumulative_without_future_leakage():
    order = [("bfcl", "a"), ("appworld", "b"), ("bfcl", "c"), ("terminalbench2", "d")]
    scores = {1: 1, 2: 0, 3: 0, 4: 1}
    windows = interleaved_windows(order, scores, 3, 50, basis="benchmark")
    cumulative = interleaved_cumulative(order, scores, 3)
    assert [row["score"] for row in windows] == [row["score"] for row in cumulative]
    assert all(row["iteration"] <= 3 for row in windows)
    with pytest.raises(ValueError, match="positive"):
        interleaved_windows(order, scores, 4, 0)
    with pytest.raises(ValueError, match="outside"):
        interleaved_windows(order, scores, 5, 5)
    with pytest.raises(ValueError, match="basis"):
        interleaved_windows(order, scores, 4, 5, basis="invalid")


def test_interleaved_loader_checks_mode_and_parent_score_provenance(tmp_path):
    order = [("bfcl", "a"), ("appworld", "b"), ("bfcl", "c"), ("terminalbench2", "d")]
    (tmp_path / "iterations").mkdir()
    (tmp_path / "state.json").write_text(
        json.dumps({"cursor": 4, "config": {"mode": "interleaved", "seed": 42, "order": order}})
    )
    for i, (b, task) in enumerate(order, 1):
        (tmp_path / "iterations" / f"iter_{i:04d}.json").write_text(
            json.dumps(
                {
                    "iteration": i,
                    "benchmark": b,
                    "task_id": task,
                    "parents": [{"score": 0.4}, {"score": 1}],
                }
            )
        )
    job = {"seed": 42, "output_dir": str(tmp_path), "task_order": order}
    run = load_run(tmp_path, job, "premise", "best-parent", "tgc", mode="interleaved")
    assert run["scores"] == {1: 1, 2: 1, 3: 1, 4: 1}
    with pytest.raises(ValueError, match="sequential"):
        load_run(tmp_path, job, "premise", "best-parent", "tgc")


@pytest.mark.parametrize("methods", [("ace", "premise"), ("ace", "premise", "skillopt")])
def test_interleaved_plot_has_three_benchmark_styles_per_method_and_one_axis(methods):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = [
        {"method": method, "benchmark": benchmark, "iteration": i, "mean": i / 10}
        for method in methods
        for benchmark in ("bfcl", "appworld", "terminalbench2")
        for i in (1, 2)
    ]
    fig, ax = plt.subplots()
    try:
        plot_interleaved_axis(ax, rows)
        curves = [line for line in ax.lines if len(line.get_xdata()) == 2]
        assert len(fig.axes) == 1 and len(curves) == 3 * len(methods)
        for method in methods:
            color = COLORS[method]
            selected = [line for line in curves if line.get_color() == color]
            assert len(selected) == 3 and {line.get_linestyle() for line in selected} == {"-", "--", ":"}
            assert all(line.get_drawstyle() == "steps-post" for line in selected)
        assert ax.get_ylim() == (0, 100)
    finally:
        plt.close(fig)
