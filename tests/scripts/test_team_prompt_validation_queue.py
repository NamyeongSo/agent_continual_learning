# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json
import shutil

import pytest

from scripts.utils import evaluate_team_prompt_validation_queue as runner


@pytest.fixture
def exported_winners(tmp_path):
    source = tmp_path / "source"
    splits = tmp_path / "splits.json"
    tasks = [f"val-{i}" for i in range(50)]
    runner.shared.write_json(splits, {"appworld_44": {
        "train": ["train-only"], "val": tasks, "test": ["test-only"],
    }})
    for method, run in runner.HISTORY_RUNS.items():
        history = source / "training_history_20261009/runs" / run / "shared/raw"
        directory = source / f"{method}_interleaved_seed44" / (
            "cp130_shared" if method == "gepa" else "cp130_shared_best"
        )
        directory.mkdir(parents=True)
        if method == "gepa":
            candidate = {"system_prompt": "Selected GEPA prompt"}
            runner.shared.write_json(history / "gepa_result.json", {
                "candidates": [{"system_prompt": "Older prompt"}, candidate],
                "best_idx": 1, "val_aggregate_scores": [0.4, 0.6],
            })
            runner.shared.write_json(history / "experiment_config.json", {
                "validation_task_ids": [{"benchmark": "appworld", "task_id": t} for t in reversed(tasks)],
            })
            path = directory / "candidate.json"
            runner.shared.write_json(path, {"best_candidate": candidate})
        else:
            path = directory / "skill.md"
            path.write_text("Selected SkillOpt skill")
            history.mkdir(parents=True, exist_ok=True)
            (history / "best_skill.md").write_bytes(path.read_bytes())
            for step, hard in [(1, 0), (2, 1)]:
                step_dir = history / "steps" / f"step_{step:04d}"
                rows = [{"benchmark": "appworld", "task_id": t, "hard": hard, "soft": hard}
                        for t in tasks[:20]]
                rows += [{"benchmark": "bfcl", "task_id": f"bfcl-{i}", "hard": hard, "soft": hard}
                         for i in range(32)]
                runner.shared.write_json(step_dir / "selection_eval/results.json", rows)
                (step_dir / "candidate_skill.md").write_bytes(path.read_bytes())
            runner.shared.write_json(history / "summary.json", {"best_step": 2, "best_selection_hard": 1.0})
        runner.shared.write_json(directory / "provenance.json", {
            "method": method, "mode": "interleaved", "seed": 44, "checkpoint": 130,
            "sha256": runner.shared.digest(path.read_bytes()),
        })
    return source, splits, tmp_path / "output", tasks


def test_validation_winners_use_same_full_50_task_set_without_training_or_test_tasks(exported_winners):
    source, splits, output, tasks = exported_winners
    launch = runner.prepare(source, splits, output, 100, "http://localhost/v1")
    assert launch["total_tasks"] == 100
    groups = launch["groups"]
    assert groups[0]["selection"]["candidate_index"] == 1
    assert groups[1]["selection"]["step"] == 2
    records = {g["group"]: [None] * 50 for g in groups}
    jobs = runner.shared.queue_jobs(groups, records)
    assert len(jobs) == 100
    for group in groups:
        assert group["task_ids"] == tasks
        assert group["split"] == "val"
        assert group["subset"] == "test_challenge"
    assert {job["task_id"] for job in jobs} == set(tasks)


def test_checkpoint_export_must_match_optimizer_winner(exported_winners):
    source, splits, output, _ = exported_winners
    path = source / "gepa_interleaved_seed44/cp130_shared/candidate.json"
    path.write_text(json.dumps({"best_candidate": {"system_prompt": "Another checkpoint"}}))
    with pytest.raises(ValueError, match="highest-scoring candidate"):
        runner.prepare(source, splits, output, 100, "http://localhost/v1")


def test_validation_scores_include_errors_as_zero(exported_winners):
    source, splits, output, _ = exported_winners
    launch = runner.prepare(source, splits, output, 100, "http://localhost/v1")
    rows = [{"status": "completed", "score": 1, "partial_score": 1},
            {"status": "error", "score": 0, "partial_score": 0}]
    rows += [{"status": "completed", "score": 0, "partial_score": 0.5}] * 48
    runner.save_scores(output, launch["groups"], {g["group"]: rows for g in launch["groups"]})
    for score in runner.shared.read_json(output / "scores.json")["scores"]:
        assert score["complete"]
        assert score["hard"] == 1 / 50
        assert score["soft"] == 0.5
        assert score["execution_errors"] == 1
        assert score["split"] == "val"


def test_seed42_43_winners_evaluate_200_tasks_with_their_own_validation_splits(exported_winners):
    source, splits, output, _ = exported_winners
    sections = runner.shared.read_json(splits)
    for seed in (42, 43):
        tasks = [f"val-{seed}-{index}" for index in range(50)]
        sections[f"appworld_{seed}"] = {"train": [f"train-{seed}"], "val": tasks, "test": [f"test-{seed}"]}
        for method, run in runner.HISTORY_RUNS_BY_SEED[seed].items():
            old_history = source / "training_history_20261009/runs" / runner.HISTORY_RUNS[method] / "shared/raw"
            history = source / "training_history_20261009/runs" / run / "shared/raw"
            shutil.copytree(old_history, history)
            leaf = "cp130_shared" if method == "gepa" else "cp130_shared_best"
            directory = source / f"{method}_interleaved_seed{seed}" / leaf
            directory.mkdir(parents=True)
            prompt = f"Selected {method} seed {seed}"
            if method == "gepa":
                candidate = {"system_prompt": prompt}
                path = directory / "candidate.json"
                runner.shared.write_json(path, {"best_candidate": candidate})
                result = runner.shared.read_json(history / "gepa_result.json")
                result["candidates"][result["best_idx"]] = candidate
                runner.shared.write_json(history / "gepa_result.json", result)
                runner.shared.write_json(history / "experiment_config.json", {
                    "seed": seed, "mode": "interleaved",
                    "validation_task_ids": [{"benchmark": "appworld", "task_id": task} for task in tasks],
                })
            else:
                path = directory / "skill.md"
                path.write_text(prompt)
                (history / "best_skill.md").write_text(prompt)
                for step in history.glob("steps/step_*"):
                    (step / "candidate_skill.md").write_text(prompt)
                    rows = runner.shared.read_json(step / "selection_eval/results.json")
                    for index, row in enumerate(r for r in rows if r["benchmark"] == "appworld"):
                        row["task_id"] = tasks[index]
                    runner.shared.write_json(step / "selection_eval/results.json", rows)
            runner.shared.write_json(directory / "provenance.json", {
                "method": method, "mode": "interleaved", "seed": seed, "checkpoint": 130,
                "sha256": runner.shared.digest(path.read_bytes()),
            })
    runner.shared.write_json(splits, sections)
    launch = runner.prepare(source, splits, output, 150, "http://localhost/v1", seeds=(42, 43),
                            max_tokens=8192, model_num_retries=5)
    assert launch["checkpoint_count"] == 4
    assert launch["total_tasks"] == 200
    groups = launch["groups"]
    records = {group["group"]: [None] * 50 for group in groups}
    jobs = list(runner.shared.queue_jobs(groups, records))
    assert len({(job["method"], job["seed"], job["task_id"]) for job in jobs}) == 200
    for group in groups:
        seed = group["seed"]
        assert seed in (42, 43)
        assert group["task_ids"] == sections[f"appworld_{seed}"]["val"]
        job = next(item for item in jobs if item["group"] == group["group"])
        config = runner.team.task_config(job, output / "sessions", f"config-{group['method']}-{seed}")
        assert config.agent_kwargs["system_prompt_override"] == f"Selected {group['method']} seed {seed}"
        assert config.agent_kwargs["model_settings"].max_tokens == 2048
        assert config.agent_kwargs["model_settings"].num_retries == 5
    rows = [{"status": "completed", "score": 1, "partial_score": 1}] * 50
    runner.save_scores(output, groups, {group["group"]: rows for group in groups})
    scores = runner.shared.read_json(output / "scores.json")["scores"]
    assert {(row["method"], row["seed"]) for row in scores} == {
        (method, seed) for method in ("gepa", "skillopt") for seed in (42, 43)
    }


@pytest.mark.parametrize("phase,queued,active,processed,expected", [
    ("running", 50, 150, 190, 0),
    ("running", 0, 110, 280, 40),
    ("complete", 0, 0, 390, 150),
])
def test_shared_capacity_keeps_combined_evaluations_at_150(
    tmp_path, phase, queued, active, processed, expected,
):
    dependency = tmp_path / "premise"
    runner.shared.write_json(dependency / "launch_manifest.json", {"total_tasks": 390, "workers": 150})
    runner.shared.write_json(dependency / "status.json", {
        "phase": phase, "queued_tasks": queued, "active_tasks": active, "processed_tasks": processed,
        "total_tasks": 390, "workers": 150,
    })
    launch = {"workers": 150, "capacity_dependency": {
        "output": str(dependency), "total_tasks": 390, "workers": 150, "global_workers": 150,
        "launch_sha256": runner.shared.digest((dependency / "launch_manifest.json").read_bytes()),
    }}
    assert runner.available_workers(launch) == expected
