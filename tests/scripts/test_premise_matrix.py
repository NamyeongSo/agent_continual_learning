# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""The matrix launcher refills its two running experiment slots."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar

import pytest


def test_finished_experiment_is_replaced_while_other_one_runs(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/premise/run_matrix.py"
    spec = importlib.util.spec_from_file_location("premise_matrix_test", script)
    assert spec is not None and spec.loader is not None
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)

    rows = [
        {"name": name, "seed": 42, "mode": "sequential", "past": 3}
        for name in ("first", "second", "third")
    ]
    monkeypatch.setattr(matrix, "experiments", lambda: rows)
    real_sleep = time.sleep
    monkeypatch.setattr(matrix.time, "sleep", lambda _seconds: real_sleep(0.01))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        f"{slug}_{seed}": {part: [f"{slug}-{part}-{i}" for i in range(count)]
                           for part, count in zip(
                               ("train", "val", "test"),
                               (30, 30, 29) if slug == "terminalbench2" else (50, 50, 50),
                               strict=True,
                           )}
        for slug in ("bfcl", "appworld", "terminalbench2") for seed in (42, 43, 44)
    }))
    monkeypatch.setattr(sys, "argv", ["run_matrix.py", "--manifest", str(manifest),
                                      "--output-root", str(tmp_path / "out"), "--max-parallel", "2",
                                      "--port-block-base", "26000", "--train-only"])

    class FakeProcess:
        active = 0
        peak = 0
        started: ClassVar[dict[str, float]] = {}
        ended: ClassVar[dict[str, float]] = {}
        ports: ClassVar[dict[str, str]] = {}

        def __init__(self, cmd, **_kwargs):
            self.output = Path(cmd[cmd.index("--output-dir") + 1])
            self.name = self.output.name
            self.ports[self.name] = _kwargs["env"]["EXGENTIC_PORT_BLOCK_START"]
            self.pid = 1000 + len(self.started)
            self.started[self.name] = time.monotonic()
            self.deadline = self.started[self.name] + (0.2 if self.name == "second" else 0.03)
            self.returncode = None
            FakeProcess.active += 1
            FakeProcess.peak = max(FakeProcess.peak, FakeProcess.active)

        def poll(self):
            if self.returncode is None and time.monotonic() >= self.deadline:
                self.output.mkdir(parents=True, exist_ok=True)
                (self.output / "state.json").write_text('{"cursor":130}')
                self.returncode = 0
                FakeProcess.active -= 1
                self.ended[self.name] = time.monotonic()
            return self.returncode

    monkeypatch.setattr(matrix.subprocess, "Popen", FakeProcess)
    assert matrix.main() == 0
    assert FakeProcess.peak == 2
    assert FakeProcess.ports == {"first": "26000", "second": "27000", "third": "28000"}
    assert FakeProcess.started["third"] < FakeProcess.ended["second"]
    status = json.loads((tmp_path / "out/matrix_status.json").read_text())
    assert [row["status"] for row in status["experiments"]] == ["done"] * 3


def test_default_matrix_freezes_six_runs_and_fifty_four_test_groups():
    script = Path(__file__).resolve().parents[2] / "scripts/premise/run_matrix.py"
    spec = importlib.util.spec_from_file_location("premise_matrix_plan_test", script)
    assert spec is not None and spec.loader is not None
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    plan = matrix.build_plan(matrix.DEFAULT_MANIFEST, matrix.experiments())
    assert plan["training_runs"] == 6
    assert plan["evaluation_groups"] == 54
    assert plan["evaluation_tasks"] == 2322
    assert len({(item["seed"], item["mode"]) for item in plan["jobs"]}) == 6
    assert {len(item["task_order"]) for item in plan["jobs"]} == {130}
    for job in plan["jobs"]:
        if job["mode"] == "sequential":
            assert [slug for slug, _ in job["task_order"]] == [
                *["bfcl"] * 50, *["appworld"] * 50, *["terminalbench2"] * 30,
            ]


def test_ps_st_pareto_matrix_passes_method_and_keeps_checkpoint_test_plan():
    script = Path(__file__).resolve().parents[2] / "scripts/premise/run_matrix.py"
    spec = importlib.util.spec_from_file_location("ps_st_pareto_matrix_test", script)
    assert spec is not None and spec.loader is not None
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    plan = matrix.build_plan(matrix.DEFAULT_MANIFEST, matrix.experiments(), "ps_st_pareto")
    assert plan["method"] == "ps_st_pareto"
    assert plan["evaluation_groups"] == 54
    row = matrix.experiments()[0]
    command = matrix.command(row, matrix.DEFAULT_MANIFEST, matrix.PARETO_OUTPUT / str(row["name"]), "ps_st_pareto")
    assert command[command.index("--method") + 1] == "ps_st_pareto"
    assert "--evaluate-test" in command


def test_early_bfcl_controller_plans_one_partial_test_per_sequential_seed(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/premise/evaluate_early_bfcl.py"
    plan = {
        "method": "ps_st_pareto", "source_manifest": str(tmp_path / "split.json"),
        "jobs": [
            {"name": f"s{seed}_sequential_t3", "seed": seed, "mode": "sequential", "past": 3}
            for seed in (42, 43, 44)
        ],
    }
    manifest = tmp_path / "launch_manifest.json"
    manifest.write_text(json.dumps(plan))
    result = subprocess.run(
        [sys.executable, str(script), "--launch-manifest", str(manifest),
         "--port-block-base", "54000", "--dry-run"],
        check=True, capture_output=True, text=True,
    )
    lines = result.stdout.splitlines()
    assert len(lines) == 3
    assert [line.split(" ", 1)[0] for line in lines] == ["port=54000", "port=55000", "port=56000"]
    assert all("--evaluation-only --evaluate-test-checkpoint 50 --evaluate-test-benchmark bfcl" in line
               for line in lines)
    assert all("--evaluate-test --method" not in line for line in lines)


@pytest.mark.parametrize("complete", [False, True])
def test_adopted_worker_survives_launcher_replacement_and_detects_pid_reuse(tmp_path, monkeypatch, complete):
    script = Path(__file__).resolve().parents[2] / "scripts/premise/run_matrix.py"
    spec = importlib.util.spec_from_file_location("premise_matrix_adoption_test", script)
    assert spec is not None and spec.loader is not None
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    identity = (["python", "run_experiment.py"], "100")
    monkeypatch.setattr(matrix, "process_identity", lambda _pid: identity)
    process = matrix.ExistingExperimentProcess(123, identity, tmp_path, train_only=True)
    assert process.poll() is None
    if complete:
        (tmp_path / "state.json").write_text('{"cursor":130}')
    # A new process using the same PID must never be adopted or signalled.
    monkeypatch.setattr(matrix, "process_identity", lambda _pid: (identity[0], "101"))
    assert process.poll() == (0 if complete else 1)
    assert process.wait(timeout=0) == (0 if complete else 1)


def test_process_identity_reads_live_process_and_rejects_missing_pid():
    import os

    script = Path(__file__).resolve().parents[2] / "scripts/premise/run_matrix.py"
    spec = importlib.util.spec_from_file_location("premise_matrix_identity_test", script)
    assert spec is not None and spec.loader is not None
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    identity = matrix.process_identity(os.getpid())
    assert identity is not None
    assert identity[0][0]
    assert identity[1].isdigit()
    assert matrix.process_identity(-1) is None
