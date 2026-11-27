# SPDX-License-Identifier: Apache-2.0
"""Guard selection, cumulative checkpoints, and the seed/stage scheduling barriers."""

import json
from fractions import Fraction

import pytest

from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
from scripts.ace import run_reflection_accumulation as study


def candidates(scores):
    return [{"source_repeat": i, "score_numerator": score.numerator, "score_denominator": score.denominator}
            for i, score in enumerate(scores)]


@pytest.mark.parametrize("mode,expected", [("best", [1, 3]), ("worst", [0, 4])])
def test_exact_ties_are_selected_randomly_and_reproducibly(mode, expected):
    bank = candidates([Fraction(1, 3), Fraction(2, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1, 3)])
    seen = set()
    for seed in range(30):
        chosen = study.select_candidate(bank, mode, seed, "train_task", "without_gt")
        assert chosen == study.select_candidate(bank, mode, seed, "train_task", "without_gt")
        assert chosen["tie_indices"] == expected
        seen.add(chosen["candidate_index"])
    assert seen == set(expected)


def test_best_does_not_turn_nearly_equal_scores_into_ties():
    bank = candidates([Fraction(1, 3), Fraction(333334, 1000000)])
    chosen = study.select_candidate(bank, "best", 42, "task", "with_gt")
    assert chosen["tie_indices"] == [1]


def test_random_pairs_trajectory_indices_between_gt_arms():
    bank = candidates([Fraction(i, 5) for i in range(5)])
    a = study.select_candidate(bank, "random", 42, "task", "without_gt")
    b = study.select_candidate(bank[::-1], "random", 42, "task", "with_gt")
    assert a["candidate_index"] == b["candidate_index"]
    assert a["tie_indices"] == list(range(5))


def test_turn_cap_retains_official_partial_credit():
    assert study.exact_reward({"turn_limit_reached": True, "pct": 2 / 3,
                               "passed_tests": 2, "total_tests": 3}) == Fraction(2, 3)
    assert study.exact_reward({"budget_failure": {"reason": "input_tokens"}, "pct": 0}) == 0
    with pytest.raises(ValueError):
        study.exact_reward({"pct": 0.6667, "passed_tests": 2, "total_tests": 3})


def test_curator_only_training_resumes_from_last_cumulative_checkpoint(tmp_path, monkeypatch):
    condition = study.conditions(42)[0]
    tasks = ["train_a", "train_b", "train_c"]
    study.base.write(tmp_path / "protocol.json", {
        "playbook_edit_mode": "add_only", "initial_playbook": DEFAULT_PLAYBOOK,
        "task_orders": {"42": {"train": tasks}}, "model": "mock-model", "model_settings": {},
    })
    choices = [{"candidate": {"task_id": task, "source_id": task + "_r0", "task": task, "context": {},
                              "reflection": task, "reflection_sha256": study.base.digest(task)},
                "tie_indices": [0], "selection_rng_seed": "42"} for task in tasks]
    directory = tmp_path / "conditions" / condition["id"]
    study.base.write(directory / "selections.json", choices)
    monkeypatch.setattr(study.base, "configure_environment", lambda _: None)

    def forbidden(*args, **kwargs):
        pytest.fail("Actor or reflector was called during curator-only learning")

    monkeypatch.setattr(ACEAgentInstance, "_run_post_session_reflection", forbidden)
    monkeypatch.setattr(ACEAgentInstance, "start", forbidden)
    seen = []
    fail_once = [True]

    def answer(instance, model, prompt, **kwargs):
        seen.append((instance.task, instance._store.playbook))
        if instance.task == "train_b" and fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("Interrupted curator")
        return json.dumps({"operations": [{"type": "ADD", "section": "strategies_and_insights",
                                          "content": instance.task}]})

    monkeypatch.setattr(ACEAgentInstance, "_llm_call_simple", answer)
    with pytest.raises(RuntimeError, match="Interrupted curator"):
        study.train_condition(tmp_path, condition)
    assert (directory / "checkpoints/001.json").exists()
    assert not (directory / "checkpoints/002.json").exists()
    assert study.train_condition(tmp_path, condition)["ok"]
    assert [task for task, _ in seen] == ["train_a", "train_b", "train_b", "train_c"]
    assert "train_a" in seen[-1][1] and "train_b" in seen[-1][1]
    final = study.base.read(directory / "checkpoints/003.json")
    assert all(task in final["playbook"] for task in tasks)
    assert final["session_count"] == 3
    before = len(seen)
    study.train_condition(tmp_path, condition)
    assert len(seen) == before


@pytest.mark.parametrize("phase", ["train_final", "bwt_checkpoint", "val"])
def test_evaluation_has_300_distinct_jobs_with_correct_context_and_split(tmp_path, phase):
    train, val = [f"train_{i}" for i in range(50)], [f"val_{i}" for i in range(50)]
    study.base.write(tmp_path / "protocol.json", {"task_orders": {"42": {"train": train, "val": val}}})
    for condition in study.conditions(42):
        cp_dir = tmp_path / "conditions" / condition["id"] / "checkpoints"
        for step, task in enumerate(train, 1):
            text = f"Accumulated context {step}"
            study.base.write(cp_dir / f"{step:03d}.json", {"task_id": task, "playbook": text})
            (cp_dir / f"{step:03d}.txt").write_text(text)
    jobs = study.eval_jobs(tmp_path, 42, phase)
    assert len(jobs) == len({j["id"] for j in jobs}) == 300
    assert {j["task_id"] for j in jobs} == set(val if phase == "val" else train)
    for job in jobs:
        step = train.index(job["task_id"]) + 1 if phase == "bwt_checkpoint" else 50
        assert job["checkpoint_step"] == step
        parent = study.base.read(tmp_path / "seed42/stage2" / job["id"] / "result.json")
        assert parent["updated_context_path"].endswith(f"/{step:03d}.txt")


def scheduler_fixture(tmp_path):
    source = tmp_path / "source"
    study.base.write(source / "protocol.json", {})
    study.base.write(source / "status.json", {"state": "completed"})
    study.base.write(tmp_path / "protocol.json", {
        "source_reflection_run": str(source), "seeds": [42, 43, 44],
        "source_protocol_sha256": study.base.digest((source / "protocol.json").read_bytes()),
    })
    study.base.write(tmp_path / "source_hashes.json", {})
    study.base.write(tmp_path / "bank_ready.json", {})
    return source


@pytest.mark.parametrize("fail_phase", [None, "bwt_checkpoint"])
def test_all_seed_phases_complete_before_next_seed(tmp_path, monkeypatch, fail_phase):
    scheduler_fixture(tmp_path)
    monkeypatch.setattr(study.base, "configure_environment", lambda _: None)
    monkeypatch.setattr(study.base, "check_endpoint", lambda _: {})
    monkeypatch.setattr(study, "journal", lambda *args: None)
    seen = []
    monkeypatch.setattr(study, "training_phase", lambda output, seed, progress: seen.append((seed, "train")))

    def evaluate(output, seed, phase, progress):
        seen.append((seed, phase))
        if phase == fail_phase:
            raise RuntimeError("Unresolved evaluations")

    monkeypatch.setattr(study, "evaluation_phase", evaluate)
    monkeypatch.setattr(study, "summarize_seed", lambda output, seed: seen.append((seed, "summary")))
    if fail_phase:
        with pytest.raises(RuntimeError, match="Unresolved"):
            study.run(tmp_path)
        assert seen == [(42, "train"), (42, "train_final"), (42, "bwt_checkpoint")]
    else:
        study.run(tmp_path)
        assert seen == [(seed, phase) for seed in (42, 43, 44)
                        for phase in ("train", "train_final", "bwt_checkpoint", "val", "summary")]


def test_incomplete_predecessor_cannot_start_learning(tmp_path, monkeypatch):
    source = scheduler_fixture(tmp_path)
    study.base.write(source / "status.json", {"state": "running"})
    with pytest.raises(ValueError, match="not completed"):
        study.freeze_bank(tmp_path)
    (tmp_path / "STOP").touch()
    monkeypatch.setattr(study, "journal", lambda *args: None)
    monkeypatch.setattr(study.base, "check_endpoint", lambda _: pytest.fail("Premature inference setup"))
    with pytest.raises(InterruptedError, match="STOP"):
        study.run(tmp_path)
