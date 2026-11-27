# SPDX-License-Identifier: Apache-2.0
"""One shared queue must serve all seeds and preserve phase barriers and saved results."""

from collections import Counter
from concurrent.futures import Future
from pathlib import Path

import pytest

from scripts.ace import run_reflection_accumulation as study


def scheduler_fixture(tmp_path):
    source = tmp_path / "source"
    study.base.write(source / "protocol.json", {})
    study.base.write(source / "status.json", {"state": "completed"})
    study.base.write(tmp_path / "protocol.json", {
        "source_reflection_run": str(source), "seeds": [42, 43, 44], "execution_order": "phase_major",
        "global_concurrency": 50, "source_protocol_sha256": study.base.digest((source / "protocol.json").read_bytes()),
    })
    study.base.write(tmp_path / "source_hashes.json", {})
    study.base.write(tmp_path / "bank_ready.json", {})


@pytest.mark.parametrize("fail_phase", [None, "train_final", "bwt_checkpoint"])
def test_all_seeds_finish_each_phase_before_the_next_phase(tmp_path, monkeypatch, fail_phase):
    scheduler_fixture(tmp_path)
    monkeypatch.setattr(study.base, "configure_environment", lambda _: None)
    monkeypatch.setattr(study.base, "check_endpoint", lambda _: {})
    monkeypatch.setattr(study, "journal", lambda *args: None)
    seen = []
    monkeypatch.setattr(study, "training_phase", lambda out, seeds, progress: seen.append(("train", tuple(seeds))))

    def evaluate(out, seeds, phase, progress):
        seen.append((phase, tuple(seeds)))
        if phase == fail_phase:
            raise RuntimeError("Unresolved shared-queue evaluations")

    monkeypatch.setattr(study, "evaluation_phase", evaluate)
    monkeypatch.setattr(study, "summarize_seed", lambda out, seed: seen.append(("summary", seed)))
    if fail_phase:
        with pytest.raises(RuntimeError, match="Unresolved"):
            study.run(tmp_path)
        assert not any(x[0] in ("val", "summary") for x in seen)
        assert study.base.read(tmp_path / "status.json")["state"] == "error"
    else:
        study.run(tmp_path)
        assert seen == [(phase, (42, 43, 44)) for phase in ("train", "train_final", "bwt_checkpoint", "val")] + [
            ("summary", s) for s in (42, 43, 44)
        ]
        assert study.base.read(tmp_path / "status.json")["seed"] == "all"


def immediate_pool(monkeypatch, pools, submitted):
    class Pool:
        def __init__(self, max_workers, **kwargs):
            pools.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args):
            submitted.append(args)
            future = Future()
            try:
                future.set_result(function(*args))
            except BaseException as exc:
                future.set_exception(exc)
            return future

    monkeypatch.setattr(study, "ProcessPoolExecutor", Pool)


@pytest.mark.parametrize("permanent_failure", [False, True])
def test_one_50_worker_queue_interleaves_seeds_resumes_and_keeps_retries(tmp_path, monkeypatch, permanent_failure):
    study.base.write(tmp_path / "protocol.json", {"global_concurrency": 50})
    monkeypatch.setattr(study, "journal", lambda *args: None)
    pools, submitted = [], []
    immediate_pool(monkeypatch, pools, submitted)

    def jobs(out, seed, phase):
        return [{"id": f"s{seed}_{phase}_{i}", "seed": seed, "stage": 3, "task_id": f"t{i}"}
                for i in range(300)]

    monkeypatch.setattr(study, "eval_jobs", jobs)
    saved_job = jobs(tmp_path, 42, "train_final")[0]
    saved = {**saved_job, "pct": 0.0, "success": False, "budget_failure": {"reason": "input_tokens"}}
    path = study.base.job_dir(tmp_path / "seed42", saved_job) / "result.json"
    study.base.write(path, saved)
    saved_bytes = path.read_bytes()

    def validate(seed_dir, job, protocol):
        return study.base.read(study.base.job_dir(seed_dir, job) / "result.json")

    monkeypatch.setattr(study, "validate_evaluation", validate)

    def worker(seed_dir, job, attempt):
        assert Path(seed_dir).name == f"seed{job['seed']}"
        if (job["id"] == "s43_train_final_0" and attempt == 1) or (
                permanent_failure and job["id"] == "s44_train_final_299"):
            return {"ok": False, "error": "infrastructure"}
        study.base.write(study.base.job_dir(seed_dir, job) / "result.json", {**job, "pct": 1.0, "success": True})
        return {"ok": True}

    monkeypatch.setattr(study.base, "worker", worker)
    real_wait, batch_sizes = study.wait, []

    def observed_wait(futures, **kwargs):
        batch_sizes.append(len(futures))
        assert len(futures) <= 50
        return real_wait(futures, **kwargs)

    monkeypatch.setattr(study, "wait", observed_wait)
    if permanent_failure:
        with pytest.raises(RuntimeError, match="next phase/seed will not start"):
            study.evaluation_phase(tmp_path, [42, 43, 44], "train_final", study.Progress(tmp_path))
        assert not list(tmp_path.glob("seed*/train_final_results.json"))
        assert not (tmp_path / "phase_results/train_final_completed.json").exists()
        assert [a[2] for a in submitted if a[1]["id"] == "s44_train_final_299"] == [1, 2, 3]
    else:
        study.evaluation_phase(tmp_path, [42, 43, 44], "train_final", study.Progress(tmp_path))
        for seed in (42, 43, 44):
            assert len(study.base.read(tmp_path / f"seed{seed}/train_final_results.json")) == 300
        marker = study.base.read(tmp_path / "phase_results/train_final_completed.json")
        assert marker["completed"] == marker["total"] == 900
        status = study.base.read(tmp_path / "status.json")
        assert status["seed"] == "all" and status["completed"] == 900
        assert status["global_concurrency"] == status["pool_workers"] == 50
        assert all(v["completed"] == 300 for v in status["by_seed"].values())
    assert pools == [50] and max(batch_sizes) == 50
    assert set(Counter(a[1]["seed"] for a in submitted[:50])) == {42, 43, 44}
    assert max(Counter(a[1]["seed"] for a in submitted[:50]).values()) <= 17
    assert not any(a[1]["id"] == saved_job["id"] for a in submitted)
    assert path.read_bytes() == saved_bytes
    assert [a[2] for a in submitted if a[1]["id"] == "s43_train_final_0"] == [1, 2]
    assert len(study.base.read(tmp_path / "phase_queues/train_final_queue.json")) == 900


def test_training_runs_18_independent_context_chains_under_global_limit_50(tmp_path, monkeypatch):
    study.base.write(tmp_path / "protocol.json", {"global_concurrency": 50, "playbook_edit_mode": "add_update_delete"})
    monkeypatch.setattr(study, "journal", lambda *args: None)
    pools, submitted = [], []
    immediate_pool(monkeypatch, pools, submitted)

    def train(out, condition):
        d = Path(out) / "conditions" / condition["id"]
        for i in range(51):
            study.base.write(d / "checkpoints" / f"{i:03d}.json", {"step": i})
        study.base.write(d / "training_completed.json", {"completed": 50})
        return {"ok": True}

    monkeypatch.setattr(study, "train_condition", train)
    study.training_phase(tmp_path, [42, 43, 44], study.Progress(tmp_path))
    assert pools == [18] and len(submitted) == 18
    assert len({a[1]["id"] for a in submitted}) == 18
    status = study.base.read(tmp_path / "status.json")
    assert status["seed"] == "all" and status["completed"] == status["total"] == 900
    assert status["global_concurrency"] == 50 and status["pool_workers"] == 18
    assert all(v["completed"] == 300 for v in status["by_seed"].values())


def test_shared_bwt_keeps_each_seeds_own_task_order_and_post_learning_context(tmp_path, monkeypatch):
    tasks = [f"task{i}" for i in range(50)]
    orders = {42: tasks, 43: tasks[::-1], 44: tasks[17:] + tasks[:17]}
    protocol = {"global_concurrency": 50, "appworld_policy_sha256": "pinned_policy",
                "task_orders": {str(s): {"train": order, "val": ["heldout"]} for s, order in orders.items()}}
    study.base.write(tmp_path / "protocol.json", protocol)
    for seed, order in orders.items():
        for condition in study.conditions(seed):
            cp = tmp_path / "conditions" / condition["id"] / "checkpoints"
            for step, task in enumerate(order, 1):
                text = f"{condition['id']} after task {task} at step {step}"
                study.base.write(cp / f"{step:03d}.json", {"task_id": task, "playbook": text})
                (cp / f"{step:03d}.txt").write_text(text)
    pools, submitted = [], []
    immediate_pool(monkeypatch, pools, submitted)
    monkeypatch.setattr(study, "journal", lambda *args: None)

    def worker(seed_dir, job, attempt):
        step = orders[job["seed"]].index(job["task_id"]) + 1
        assert job["checkpoint_step"] == step
        parent = study.base.read(Path(seed_dir) / "stage2" / job["id"] / "result.json")
        expected = tmp_path / "conditions" / job["condition_id"] / "checkpoints" / f"{step:03d}.txt"
        assert Path(parent["updated_context_path"]) == expected
        assert parent["updated_context_sha256"] == study.base.digest(expected.read_bytes())
        study.base.write(study.base.job_dir(seed_dir, job) / "result.json", {
            **job, "initial_context_sha256": parent["updated_context_sha256"],
            "appworld_policy_sha256": "pinned_policy", "pct": 1.0, "passed_tests": 1,
            "total_tests": 1, "success": True,
        })
        return {"ok": True}

    monkeypatch.setattr(study.base, "worker", worker)
    study.evaluation_phase(tmp_path, [42, 43, 44], "bwt_checkpoint", study.Progress(tmp_path))
    assert pools == [50] and len(submitted) == 900
    for seed in orders:
        rows = study.base.read(tmp_path / f"seed{seed}/bwt_checkpoint_results.json")
        assert len(rows) == 300 and {r["task_id"] for r in rows} == set(tasks)


@pytest.mark.parametrize("seeds", [[], [42, 42, 43], [42, "43"]])
def test_seed_group_rejects_duplicate_contexts_or_invalid_groups(seeds):
    with pytest.raises(ValueError, match="Invalid seed group"):
        study.seed_group(seeds)
