# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.utils import evaluate_team_prompt_bfcl_queue as runner


@pytest.fixture
def exports(tmp_path):
    source, splits, output = tmp_path / "source", tmp_path / "splits.json", tmp_path / "output"
    sections = {}
    for seed in (42, 43, 44):
        sections[f"bfcl_{seed}"] = {
            partition: [f"{partition}-{seed}-{i}" for i in range(50)] for partition in ("train", "val", "test")
        }
        for method in ("gepa", "skillopt"):
            for mode in ("sequential", "interleaved"):
                folder = "cp130_terminalbench2" if mode == "sequential" else (
                    "cp130_shared" if method == "gepa" else "cp130_shared_best"
                )
                directory = source / f"{method}_{mode}_seed{seed}" / folder
                directory.mkdir(parents=True)
                prompt = f"Exact {method} {mode} seed {seed} final prompt"
                if method == "gepa":
                    path = directory / "candidate.json"
                    path.write_text(json.dumps({"best_candidate": {"system_prompt": prompt},
                                                "best_candidate_index": 3}))
                    (directory / "prompt.md").write_text(prompt)
                else:
                    path = directory / "skill.md"
                    path.write_text(prompt)
                basis = "stage_best" if mode == "sequential" else (
                    "final_best" if method == "gepa" else "final_best_selected_skill"
                )
                runner.shared.write_json(directory / "provenance.json", {
                    "method": method, "mode": mode, "seed": seed, "checkpoint": 130,
                    "basis": basis, "training_policy": "legacy", "sha256": runner.shared.digest(path.read_bytes()),
                })
    runner.shared.write_json(splits, sections)
    return source, splits, output, sections


def test_full_final_prompt_queue_uses_both_settings_and_exact_seed_validation_tasks(exports):
    source, splits, output, sections = exports
    launch = runner.prepare(source, splits, output, 150, "http://localhost/v1")
    assert launch["checkpoint_count"] == 12 and launch["total_tasks"] == 600
    groups = launch["groups"]
    jobs = list(runner.shared.queue_jobs(groups, {g["group"]: [None] * 50 for g in groups}))
    assert len(jobs) == 600
    assert len({(j["method"], j["mode"], j["seed"], j["task_id"]) for j in jobs}) == 600
    for group in groups:
        assert group["task_ids"] == sections[f"bfcl_{group['seed']}"]["val"]
        assert Path(group["checkpoint_path"]).read_bytes() == Path(group["source"]).read_bytes()
        job = next(j for j in jobs if j["group"] == group["group"])
        runner.team._PROMPTS.clear()
        config = runner.team.task_config(job, output / "sessions", "bfcl-config")
        assert config.benchmark == "bfcl"
        assert config.benchmark_kwargs == {"subset": "multi_turn_base", "use_cache": False}
        assert config.agent_kwargs["system_prompt_override"] == (
            f"Exact {group['method']} {group['mode']} seed {group['seed']} final prompt"
        )
        assert config.agent_kwargs["playbook"] == ""
        assert not config.agent_kwargs["enable_tool_shortlisting"]
        assert config.agent_kwargs["model_settings"].max_tokens == 2048
        assert config.agent_kwargs["model_settings"].temperature == 0
        assert config.agent_kwargs["model_settings"].num_retries == 5


def test_validation_and_test_are_distinct_evaluations_of_the_same_frozen_prompts(exports):
    source, splits, output, sections = exports
    launch = runner.prepare(source, splits, output, 150, "http://localhost/v1", partitions=("val", "test"))
    assert launch["checkpoint_count"] == 12
    assert launch["evaluation_group_count"] == 24 and launch["total_tasks"] == 1200
    assert launch["split"] == "val_test"
    for group in launch["groups"]:
        assert group["task_ids"] == sections[f"bfcl_{group['seed']}"][group["split"]]


def test_corrupted_export_is_rejected_before_any_evaluation(exports):
    source, splits, output, _ = exports
    (source / "skillopt_interleaved_seed42/cp130_shared_best/skill.md").write_text("Different skill")
    with pytest.raises(ValueError, match="differs from original"):
        runner.prepare(source, splits, output, 150, "http://localhost/v1")


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_bfcl_worker_uses_official_score_without_appworld_scores_file(exports, monkeypatch, score):
    source, splits, output, _ = exports
    launch = runner.prepare(source, splits, output, 150, "http://localhost/v1", seeds=(42,), modes=("interleaved",))
    group = launch["groups"][0]
    job = {**{k: v for k, v in group.items() if k != "task_ids"}, "index": 0, "task_id": group["task_ids"][0]}
    monkeypatch.setattr(runner.team, "_OUTPUT", output)
    session = SimpleNamespace(session_id="official-bfcl", status=SimpleNamespace(value="success"),
                              score=score, steps=3)
    monkeypatch.setattr("exgentic.interfaces.lib.api.execute",
                        lambda **kwargs: SimpleNamespace(session_results=[session]))
    row = runner.team.evaluate_task(job)
    assert row["status"] == "completed" and row["score"] == score
    assert row["partial_score"] == score and row["benchmark"] == "bfcl"
    assert not (Path(row["session_dir"]) / "benchmark/scores.json").exists()
