# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Check benchmark resets, missing observations, and online score provenance."""

import hashlib
import json

import pytest

from scripts.utils.plot_sequential_online_performance import (
    aggregate_windows,
    cumulative_windows,
    load_run,
    load_test_references,
    plot_sequential_axis,
    read_test_scores,
    replace_premise_appworld,
    rolling_windows,
    seed_figure_data,
)


def test_window_resets_at_each_shift_including_a_return_to_a_benchmark():
    order = [("bfcl", "a"), ("bfcl", "b"), ("appworld", "c"), ("appworld", "d"), ("bfcl", "e")]
    rows = rolling_windows(order, {1: 1.0, 2: 1.0, 3: 0.0, 4: 0.5, 5: 0.0}, 5, 3)

    assert [row["window_start"] for row in rows] == [1, 1, 3, 3, 5]
    assert [row["score"] for row in rows] == [1.0, 1.0, 0.0, 0.25, 0.0]
    assert [row["benchmark_iteration"] for row in rows] == [1, 2, 1, 2, 1]


@pytest.mark.parametrize("window", [5, 10, 15, 20])
def test_exact_trailing_range_and_first_point_after_real_boundaries(window):
    order = [
        (benchmark, str(i))
        for benchmark, count in (("bfcl", 50), ("appworld", 50), ("terminalbench2", 30))
        for i in range(count)
    ]
    scores = dict.fromkeys(range(1, 131), 0.0)
    scores[50] = scores[100] = 1.0
    rows = rolling_windows(order, scores, 130, window)

    assert rows[49]["score"] == pytest.approx(1 / window)
    assert rows[50]["score"] == 0.0
    assert rows[50]["window_start"] == 51
    assert rows[99]["score"] == pytest.approx(1 / window)
    assert rows[100]["score"] == 0.0
    assert rows[100]["window_start"] == 101
    assert rows[129]["window_start"] == 131 - window


def test_missing_scores_keep_their_iteration_without_reaching_further_back():
    order = [("bfcl", str(i)) for i in range(6)]
    scores = {1: 1.0, 2: 1.0, 3: 1.0, 5: 0.0, 6: 0.0}
    rows = rolling_windows(order, scores, 6, 3)

    assert rows[-1]["window_start"] == 4
    assert rows[-1]["window_tasks"] == 3
    assert rows[-1]["scored_tasks"] == 2
    assert rows[-1]["missing_tasks"] == 1
    assert rows[-1]["score"] == 0.0
    assert len(rolling_windows(order, scores, 4, 3)) == 4
    assert rolling_windows(order, {}, 1, 3)[0]["score"] is None


def test_cumulative_means_restart_and_exclude_missing_scores_without_future_tasks():
    order = [
        ("bfcl", "a"),
        ("bfcl", "b"),
        ("bfcl", "c"),
        ("appworld", "d"),
        ("appworld", "e"),
        ("appworld", "f"),
        ("terminalbench2", "g"),
    ]
    scores = {1: 1.0, 2: 0.0, 3: 1.0, 5: 0.2, 6: 0.8, 7: 1.0}
    rows = cumulative_windows(order, scores, 7)

    assert [row["window_start"] for row in rows] == [1, 1, 1, 4, 4, 4, 7]
    assert [row["scored_tasks"] for row in rows] == [1, 2, 3, 0, 1, 2, 1]
    assert [row["score"] for row in rows] == [1.0, 0.5, pytest.approx(2 / 3), None, 0.2, 0.5, 1.0]
    assert all(row["window"] == "cumulative" for row in rows)
    assert cumulative_windows(order, scores, 5)[-1]["score"] == 0.2


@pytest.mark.parametrize("window", [20, "cumulative"])
def test_single_axis_has_exact_shift_lines_and_never_connects_benchmarks(window):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    segments = [("bfcl", 1, 50), ("appworld", 51, 100), ("terminalbench2", 101, 130)]
    rows = [
        {"method": method, "window": window, "benchmark": benchmark, "iteration": iteration, "mean": 0.5, "std": 0.1}
        for method in ("ace", "premise")
        for benchmark, start, end in segments
        for iteration in (start, end)
    ]
    fig, ax = plt.subplots()
    try:
        plot_sequential_axis(ax, rows, window, segments)
        assert len(fig.axes) == 1
        assert ax.get_xlim() == (0.5, 130.5)
        lines = [line.get_xdata() for line in ax.lines]
        boundaries = [values[0] for values in lines if len(values) == 2 and values[0] == values[1]]
        assert boundaries == [50.5, 100.5]
        curves = [values for values in lines if len(values) == 2 and values[0] != values[1]]
        assert len(curves) == 6
        assert all(any(start <= min(values) <= max(values) <= end for _, start, end in segments) for values in curves)
        assert ax.get_legend_handles_labels()[1] == ["ACE", "PREMiSE"]
    finally:
        plt.close(fig)


def test_seed_means_are_equally_weighted_with_unequal_observation_counts():
    rows = [
        {
            "method": "ace",
            "window": 5,
            "benchmark": "bfcl",
            "iteration": 5,
            "window_start": 1,
            "window_end": 5,
            "window_tasks": 5,
            "seed": seed,
            "scored_tasks": scored,
            "missing_tasks": 5 - scored,
            "score": score,
        }
        for seed, scored, score in ((42, 5, 0.0), (43, 1, 1.0), (44, 0, None))
    ]
    result = aggregate_windows(rows)[0]

    assert result["mean"] == 0.5
    assert result["std"] == pytest.approx(2**-0.5)
    assert result["seed_count"] == 2
    assert result["seed_ids"] == "42,43"
    assert result["scored_tasks"] == 6
    assert result["missing_tasks"] == 9


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_seed_figures_use_the_same_seed_for_online_and_all_test_references(seed):
    rows = [
        {"method": "ace", "seed": item, **row}
        for item in (42, 43, 44)
        for row in rolling_windows([("bfcl", "task")], {1: (item - 42) / 2}, 1, 50)
    ]
    test_rows = [
        {
            "series": series,
            "benchmark": "bfcl",
            "checkpoint": 0 if series == "initial" else 50,
            "seed": item,
            "score": (item - 42) / 2,
            "test_tasks": 50,
            "execution_errors": int(item == 43),
        }
        for item in (42, 43, 44)
        for series in ("ace", "premise", "initial")
    ]
    online, references = seed_figure_data(rows, test_rows, seed)
    assert online[0]["mean"] == (seed - 42) / 2
    assert online[0]["seed_ids"] == str(seed)
    assert online[0]["std"] is None
    assert {row["series"] for row in references} == {"ace", "premise", "initial"}
    assert all(row["mean"] == (seed - 42) / 2 and row["std"] is None for row in references)
    assert all(row["execution_errors"] == int(seed == 43) for row in references)


def test_premise_uses_pre_update_parents_and_appworld_tgc(tmp_path):
    order = [("bfcl", "a"), ("appworld", "b"), ("terminalbench2", "c")]
    (tmp_path / "iterations").mkdir()
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "cursor": 3,
                "config": {"mode": "sequential", "seed": 42, "order": order},
            }
        )
    )
    for iteration, ((benchmark, task), parents) in enumerate(
        zip(order, ([0.0, 1.0], [0.5, 0.9], [None, None]), strict=True),
        start=1,
    ):
        (tmp_path / "iterations" / f"iter_{iteration:04d}.json").write_text(
            json.dumps(
                {
                    "iteration": iteration,
                    "benchmark": benchmark,
                    "task_id": task,
                    "parents": [{"score": score} if score is not None else None for score in parents],
                    "candidates": [{"current_official_score": 1.0}],
                }
            )
        )
    job = {"seed": 42, "output_dir": str(tmp_path), "task_order": order}

    run = load_run(tmp_path, job, "premise", "best-parent", "tgc")
    assert run["scores"] == {1: 1.0, 2: 0.0}
    assert run["raw_scores"] == {1: 1.0, 2: 0.9}
    assert run["unscored_iterations"] == [3]
    rank1 = load_run(tmp_path, job, "premise", "rank1", "partial")
    assert rank1["scores"] == {1: 0.0, 2: 0.5}

    artifact = tmp_path / "iterations" / "iter_0002.json"
    data = json.loads(artifact.read_text())
    data["task_id"] = "wrong"
    artifact.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="differs from the manifest"):
        load_run(tmp_path, job, "premise", "best-parent", "tgc")


def test_invalid_window_and_cursor_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        rolling_windows([("bfcl", "a")], {}, 1, 0)
    with pytest.raises(ValueError, match="outside"):
        rolling_windows([("bfcl", "a")], {}, 2, 5)


def write_restart(root, order, *, score=1.0, cursor=3):
    source = root / "s43_sequential_t3"
    (source / "iterations").mkdir(parents=True)
    (root / "launch_manifest.json").write_text("{}")
    (source / "state.json").write_text(
        json.dumps(
            {
                "cursor": cursor,
                "config": {"mode": "sequential", "seed": 43, "order": order},
            }
        )
    )
    for i, (benchmark, task) in enumerate(order[:cursor], start=1):
        (source / "iterations" / f"iter_{i:04d}.json").write_text(
            json.dumps(
                {
                    "iteration": i,
                    "benchmark": benchmark,
                    "task_id": task,
                    "parents": [{"score": score}]
                    if benchmark == "appworld" and score is not None
                    else []
                    if benchmark == "appworld"
                    else [{"score": 1.0}],
                }
            )
        )
    return source


@pytest.mark.parametrize("replacement_score", [1.0, None])
def test_appworld_restart_changes_only_matching_premise_tasks(tmp_path, replacement_score):
    order = [("bfcl", "a"), ("appworld", "b"), ("terminalbench2", "c")]
    source = write_restart(tmp_path, order, score=replacement_score)
    runs = [
        {
            "method": method,
            "seed": seed,
            "order": order,
            "cursor": 3,
            "scores": {1: 0.1, 2: 0.0, 3: 0.3},
            "raw_scores": {1: 0.1, 2: 0.2, 3: 0.3},
            "error_iterations": [1, 2, 3],
            "unscored_iterations": [],
            "benchmark_sources": dict.fromkeys(("bfcl", "appworld", "terminalbench2"), "original"),
        }
        for method in ("ace", "premise")
        for seed in (42, 43)
    ]

    result = replace_premise_appworld(runs, tmp_path, "best-parent", "tgc")
    assert result["replaced_seeds"] == [43]
    assert result["retained_original_seeds"] == [42]
    for run in runs:
        if run["method"] == "ace" or run["seed"] == 42:
            assert run["scores"] == {1: 0.1, 2: 0.0, 3: 0.3}
            assert run["raw_scores"] == {1: 0.1, 2: 0.2, 3: 0.3}
            assert run["error_iterations"] == [1, 2, 3]
            assert run["benchmark_sources"]["appworld"] == "original"
        else:
            assert run["scores"] == ({1: 0.1, 2: 1.0, 3: 0.3} if replacement_score else {1: 0.1, 3: 0.3})
            assert run["raw_scores"] == run["scores"]
            assert run["error_iterations"] == [1, 3]
            assert run["unscored_iterations"] == ([] if replacement_score else [2])
            assert run["benchmark_sources"] == {
                "bfcl": "original",
                "appworld": str(source),
                "terminalbench2": "original",
            }


@pytest.mark.parametrize("problem", ["task_order", "incomplete"])
def test_appworld_restart_rejects_unaligned_or_incomplete_training(tmp_path, problem):
    order = [("bfcl", "a"), ("appworld", "b"), ("terminalbench2", "c")]
    source_order = [("bfcl", "a"), ("appworld", "wrong"), ("terminalbench2", "c")] if problem == "task_order" else order
    write_restart(tmp_path, source_order, cursor=1 if problem == "incomplete" else 3)
    runs = [{"method": "premise", "seed": 43, "order": order, "cursor": 3}]
    with pytest.raises(ValueError, match=r"differs from its manifest|completed AppWorld"):
        replace_premise_appworld(runs, tmp_path, "best-parent", "tgc")


def test_appworld_restart_does_not_substitute_interleaved_training(tmp_path):
    (tmp_path / "launch_manifest.json").write_text("{}")
    (tmp_path / "s42_interleaved_t3").mkdir()
    (tmp_path / "s42_interleaved_t3" / "state.json").write_text("{}")
    with pytest.raises(ValueError, match="No matching sequential"):
        replace_premise_appworld([{"method": "premise", "seed": 42}], tmp_path, "best-parent", "tgc")


def test_test_references_use_tgc_and_count_execution_errors_as_zero(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(
        json.dumps(
            {
                "identity": {"benchmark": "appworld", "manifest_sha256": "split", "task_ids": ["a", "b", "c"]},
                "results": [
                    {"task_id": "a", "status": "completed", "score": 1},
                    {"task_id": "b", "status": "unsuccessful", "score": 0.9},
                    {"task_id": "c", "status": "error", "score": 1},
                ],
            }
        )
    )
    _, values = read_test_scores(path, "appworld", ["a", "b", "c"], "split", "tgc")
    assert values["score"] == pytest.approx(1 / 3)
    assert values["execution_errors"] == 1
    _, partial = read_test_scores(path, "appworld", ["a", "b", "c"], "split", "partial")
    assert partial["score"] == pytest.approx(1.9 / 3)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "manifest", "learning", "rank"])
def test_invalid_test_reference_provenance_or_coverage_is_rejected(tmp_path, problem):
    path = tmp_path / "test.json"
    data = {
        "identity": {"benchmark": "bfcl", "manifest_sha256": "split", "task_ids": ["a", "b"]},
        "results": [{"task_id": task, "status": "completed", "score": 1} for task in ["a", "b"]],
    }
    if problem == "missing":
        data["results"].pop()
    elif problem == "duplicate":
        data["results"][1]["task_id"] = "a"
    elif problem == "manifest":
        data["identity"]["manifest_sha256"] = "different"
    elif problem == "learning":
        data["identity"]["learning"] = True
    else:
        data["identity"]["beam_rank"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_test_scores(path, "bfcl", ["a", "b"], "split", "tgc")


@pytest.mark.parametrize("wrong_hash", [False, True])
def test_checkpoint_test_lines_follow_appworld_source_and_only_finishing_checkpoints(tmp_path, wrong_hash):
    order = [
        (benchmark, str(i))
        for benchmark, count in [("bfcl", 50), ("appworld", 50), ("terminalbench2", 30)]
        for i in range(count)
    ]
    manifest = tmp_path / "splits.json"
    manifest.write_text(
        json.dumps({f"{benchmark}_43": {"test": ["test-task"]} for benchmark in ["bfcl", "appworld", "terminalbench2"]})
    )
    sources = {
        benchmark: str(tmp_path / ("restart" if benchmark == "appworld" else "original"))
        for benchmark in ["bfcl", "appworld", "terminalbench2"]
    }
    for benchmark, cp in [("bfcl", 50), ("appworld", 100), ("terminalbench2", 130)]:
        source = tmp_path / ("restart" if benchmark == "appworld" else "original")
        (source / "checkpoints").mkdir(parents=True, exist_ok=True)
        (source / "test_evaluation").mkdir(exist_ok=True)
        checkpoint = source / "checkpoints" / f"checkpoint_{cp:03d}.json"
        checkpoint.write_text(json.dumps({"cursor": cp}))
        data = {
            "identity": {
                "checkpoint": cp,
                "benchmark": benchmark,
                "task_ids": ["test-task"],
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "checkpoint_sha256": "wrong" if wrong_hash else hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            },
            "results": [{"task_id": "test-task", "status": "completed", "score": 1}],
        }
        (source / "test_evaluation" / f"cp{cp:03d}_{benchmark}.json").write_text(json.dumps(data))
    runs = [{"method": "premise", "seed": 43, "order": order, "benchmark_sources": sources}]
    if wrong_hash:
        with pytest.raises(ValueError, match="selected checkpoint"):
            load_test_references(runs, tmp_path, manifest, "tgc", checkpoints=True)
    else:
        seed_rows, references = load_test_references(runs, tmp_path, manifest, "tgc", checkpoints=True)
        assert {(row["benchmark"], row["checkpoint"]) for row in references} == {
            ("bfcl", 50),
            ("appworld", 100),
            ("terminalbench2", 130),
        }
        appworld = next(row for row in seed_rows if row["benchmark"] == "appworld")
        assert "restart/test_evaluation/cp100_appworld.json" in appworld["source_file"]


def test_test_dashes_are_bounded_to_each_benchmark_with_black_initial_context():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    segments = [("bfcl", 1, 50), ("appworld", 51, 100), ("terminalbench2", 101, 130)]
    references = [
        {
            "series": series,
            "benchmark": benchmark,
            "checkpoint": 0 if series == "initial" else end,
            "mean": mean,
            "execution_errors": 0,
        }
        for benchmark, _, end in segments
        for series, mean in [("ace", 0.4), ("premise", 0.5), ("initial", 0.3)]
    ]
    fig, ax = plt.subplots()
    try:
        plot_sequential_axis(ax, [], 50, segments, references=references)
        horizontal = [
            line for line in ax.lines if len(line.get_xdata()) == 2 and line.get_xdata()[0] != line.get_xdata()[-1]
        ]
        assert len(horizontal) == 9
        for _benchmark, start, end in segments:
            lines = [line for line in horizontal if list(line.get_xdata()) == [start - 0.5, end + 0.5]]
            assert len(lines) == 3
            assert {line.get_color() for line in lines} == {"#2474B5", "#E06B35", "black"}
            assert all(line.is_dashed() for line in lines)
        assert sum(line.get_color() == "black" for line in horizontal) == 3
    finally:
        plt.close(fig)
