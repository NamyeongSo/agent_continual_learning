# SPDX-License-Identifier: Apache-2.0
"""A final Test dispatch must not consume a training slot or collide on ports."""

from pathlib import Path
from types import SimpleNamespace

from scripts.utils import run_peace_alignment_product as queue


def test_training_completion_starts_test_and_next_training_in_same_tick(tmp_path, monkeypatch):
    jobs = [{"id": f"job{i}", "benchmark": "bfcl" if i < 3 else "appworld",
             "output": str(tmp_path / f"job{i}")} for i in range(4)]
    jobs[3]["snapshot"] = str(tmp_path / "appworld_pct_revision")
    plan = {"output": str(tmp_path), "snapshot": str(tmp_path), "jobs": jobs,
            "max_parallel": 3, "max_evaluation_parallel": 3, "port_blocks": list(range(24000, 30000, 1000))}
    progress = {"jobs": {
        "job0": {"status": "running", "stage": "train", "pid": 10, "port_block_start": 24000},
        "job1": {"status": "running", "stage": "train", "pid": 11, "port_block_start": 25000},
        "job2": {"status": "running", "stage": "train", "pid": 12, "port_block_start": 26000},
    }}
    monkeypatch.setattr(queue, "alive", lambda state: state.get("pid") != 10)
    monkeypatch.setattr(queue, "training_complete", lambda job, _: job["id"] == "job0")
    monkeypatch.setattr(queue.shared, "complete", lambda *_: False)
    monkeypatch.setattr(queue.shared, "environment", lambda *_: {})
    monkeypatch.setattr(queue.psutil, "Process", lambda *_: SimpleNamespace(create_time=lambda: 100))
    starts = []
    runtime_paths = []

    def start(command, **kwargs):
        starts.append(command)
        runtime_paths.append(kwargs["cwd"])
        return SimpleNamespace(pid=100 + len(starts))

    monkeypatch.setattr(queue.subprocess, "Popen", start)
    assert queue.tick(plan, progress) == {"train": 3, "test": 1}
    assert [(cmd[cmd.index("--job") + 1], cmd[cmd.index("--stage") + 1]) for cmd in starts] == [
        ("job3", "train"), ("job0", "test")]
    assert runtime_paths == [jobs[3]["snapshot"], plan["snapshot"]]
    assert starts[0][0] == str(Path(jobs[3]["snapshot"]) / ".venv/bin/python")
    assert len({state["port_block_start"] for state in progress["jobs"].values()}) == 4
    assert progress["jobs"]["job0"]["status"] == "evaluating"
    assert progress["jobs"]["job3"]["status"] == "running"
    # A second poll does not launch duplicates while either phase is running.
    assert queue.tick(plan, progress) == {"train": 3, "test": 1}
    assert len(starts) == 2
    assert (Path(plan["output"]) / "progress.json").exists()


def test_pct_comparison_creates_six_matched_jobs_without_changing_source_schedule(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for name in ["src/exgentic/benchmarks", "scripts", ".venv", ".exgentic"]:
        (root / name).mkdir(parents=True)
    for name in ["pyproject.toml", "uv.lock"]:
        (root / name).write_text("fixture\n")
    reflector = root / "src/exgentic/reflection.py"
    reflector.write_text("# shared reflector\n")
    source = tmp_path / "existing_cohort"
    source.mkdir()
    for name in ["task_splits.json", "private_environment.json", "appworld_original_policy.txt"]:
        (source / name).write_text("{}")
    profile = {"reflection_template_sha256": queue.shared.sha(reflector)}
    queue.shared.write(source / "inference_profile.json", profile)
    jobs = []
    for benchmark in queue.shared.BENCHMARKS:
        for seed in [42, 43, 44]:
            folder = source / f"runs/peace_no_alignment_aud/{benchmark}/s{seed}"
            jobs.append({"id": f"peace_no_alignment_aud_{benchmark}_s{seed}", "method": "premise",
                         "variant": "peace_no_alignment_aud", "benchmark": benchmark, "seed": seed,
                         "output": str(folder), "train_dir": str(folder / "train"),
                         "train_ids": ["train"], "test_ids": ["test"], "train_count": 1,
                         "train_command": [str(source / "snapshot/.venv/bin/python"), "--disable-alignment",
                                           "--seed", str(seed), "--actor-temperature", "0.0"]})
    queue.shared.write(source / "launch_manifest.json", {
        "output": str(source), "snapshot": str(source / "snapshot"), "manifest": str(source / "task_splits.json"),
        "inference_profile": str(source / "inference_profile.json"),
        "inference_profile_sha256": queue.shared.sha(source / "inference_profile.json"),
        "appworld_policy": str(source / "appworld_original_policy.txt"), "profile": profile, "jobs": jobs,
    })
    before = (source / "launch_manifest.json").read_bytes()
    monkeypatch.setattr(queue, "ROOT", root)
    plan = queue.prepare(source, tmp_path / "new_cohort", appworld_pct_comparison=True)
    assert len(plan["jobs"]) == 6
    assert plan["max_parallel"] == 3
    assert plan["appworld_candidate_metric"] == "pct"
    assert plan["appworld_reporting_metric"] == "appworld_tgc_v1"
    assert plan["port_blocks"] == list(range(30000, 36000, 1000))
    for index in range(0, 6, 2):
        old, new = plan["jobs"][index:index + 2]
        assert old["candidate_selection"] == "pareto"
        assert new["candidate_selection"] == "alignment_product"
        assert old["train_ids"] == new["train_ids"]
        assert old["test_ids"] == new["test_ids"]
        assert old["seed"] == new["seed"] == 42 + index // 2
        for job in [old, new]:
            assert job["benchmark"] == "appworld"
            command = job["train_command"]
            assert "--disable-alignment" not in command
            assert command[command.index("--appworld-candidate-metric") + 1] == "pct"
    assert (source / "launch_manifest.json").read_bytes() == before
