# SPDX-License-Identifier: Apache-2.0
"""Verify rolling slots, exclusive ports and 15-minute optimizer seed releases."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.utils import run_shared_reflection_matrix as matrix


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for name in ("src", "scripts", ".venv", ".exgentic"):
        (root / name).mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock"):
        (root / name).write_text("# fixture\n")
    (root / "src/exgentic/benchmarks").mkdir(parents=True)
    template = root / "src/exgentic/reflection.py"
    template.write_text("# validated fixture\n")
    manifest = {"seed_scope": "within_split_order_only"}
    for benchmark, count in (("bfcl", 50), ("appworld", 50), ("terminalbench2", 30)):
        for seed in (42, 43, 44):
            manifest[f"{benchmark}_{seed}"] = {
                "train": [f"{benchmark}-train-{i}" for i in range(count)],
                "val": [f"{benchmark}-val-{i}" for i in range(count)],
                "test": [f"{benchmark}-test-{i}" for i in range(count - (benchmark == "terminalbench2"))],
            }
    matrix.write(
        root / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_fixed_balanced_order_seeds42_44_v1.json",
        manifest,
    )
    policy = root / "original_policy.txt"
    policy.write_text("Original policy bytes")
    matrix.write(
        root / "outputs/qwen_epoch2_continuation_four_baselines_20261109/launch_manifest.json",
        {"jobs": [{"appworld_policy": str(policy)}], "environment": {"DOCKER_HOST": "private"}},
    )
    profile = tmp_path / "profile.json"
    matrix.write(
        profile,
        {
            "reflection_variant": "first_decision",
            "reasoning_tokens": 8192,
            "reflection_template_sha256": matrix.sha(template),
        },
    )
    validation = tmp_path / "validation"
    calibration = tmp_path / "calibration"
    matrix.write(validation / "summary.json", {"selected_on_development": "first_decision"})
    matrix.write(calibration / "pilot_summary.json", {"all_completed": True})
    monkeypatch.setattr(matrix, "ROOT", root)
    output = tmp_path / "matrix"
    plan = matrix.prepare(output, profile, validation, calibration)
    return output, plan


def test_all_36_fresh_jobs_pin_splits_roles_policy_and_seed_offsets(prepared):
    output, plan = prepared
    assert len(plan["jobs"]) == len({job["id"] for job in plan["jobs"]}) == 36
    assert plan["epochs"] == 1 and plan["max_parallel"] == 12
    for job in plan["jobs"]:
        command = job["train_command"]
        assert "--resume-sessions" not in command and "--continue-from" not in command
        assert command[command.index("--actor-temperature") + 1] == "0.0"
        assert command[command.index("--reasoning-effort") + 1] == "medium"
        if job["method"] == "premise":
            assert command[command.index("--reflection-temperature") + 1] == "0.7"
            assert command[command.index("--mutation-temperature") + 1] == "0.7"
            assert "--disable-alignment" in command and command[command.index("--past") + 1] == "3"
        else:
            assert command[command.index("--analysis-temperature") + 1] == "0.7"
        if job["benchmark"] == "appworld":
            assert Path(command[command.index("--appworld-policy") + 1]).read_text() == "Original policy bytes"
        assert set(job["train_ids"]).isdisjoint(job["test_ids"])
    assert "private" not in json.dumps(plan)
    assert (output / "private_environment.json").stat().st_mode & 0o777 == 0o600


def test_controller_preserves_optimizer_seed_intervals_without_a_benchmark_barrier(prepared, monkeypatch):
    import psutil

    output, plan = prepared
    clock = {"now": 1000}
    monkeypatch.setattr(
        matrix,
        "time",
        SimpleNamespace(time=lambda: clock["now"], sleep=lambda seconds: clock.update(now=clock["now"] + seconds)),
    )
    monkeypatch.setattr(psutil, "Process", lambda pid=None: SimpleNamespace(create_time=lambda: 10))
    starts = []

    def start(command, **kwargs):
        identity = command[command.index("--job") + 1]
        job = next(job for job in plan["jobs"] if job["id"] == identity)
        starts.append((job["benchmark"], job["method"], job["seed"], clock["now"]))
        matrix.write(
            Path(job["output"]) / "job_complete.json",
            {
                "job_id": identity,
                "complete": True,
                "tasks": len(job["test_ids"]),
                "test_ids": job["test_ids"],
                "infrastructure_errors": 0,
                "inference_profile_sha256": plan["inference_profile_sha256"],
                "score": 0.5,
            },
        )
        return SimpleNamespace(pid=100 + len(starts))

    monkeypatch.setattr(matrix.subprocess, "Popen", start)
    matrix.run(output)
    assert len(starts) == 36
    assert matrix.read(output / "status.json")["phase"] == "completed"
    for benchmark in matrix.BENCHMARKS:
        rows = [row for row in starts if row[0] == benchmark]
        for method in ("gepa", "skillopt"):
            method_rows = [r for r in rows if r[1] == method]
            anchor = next(r[3] for r in method_rows if r[2] == 42)
            for _, _, seed, instant in method_rows:
                assert instant - anchor == matrix.release_delay(method, seed)
    assert min(r[3] for r in starts if r[0] == "appworld") < max(r[3] for r in starts if r[0] == "bfcl")


def test_rolling_queue_fills_free_slots_and_never_reuses_a_live_port(prepared, monkeypatch):
    import psutil

    output, plan = prepared
    clock = {"now": 1000}
    monkeypatch.setattr(
        matrix,
        "time",
        SimpleNamespace(time=lambda: clock["now"], sleep=lambda seconds: clock.update(now=clock["now"] + seconds)),
    )
    jobs = {j["id"]: j for j in plan["jobs"]}
    running = {}
    starts = []
    original_complete = matrix.complete

    def complete(job, plan):
        entry = running.get(job["id"])
        return bool(entry and clock["now"] >= entry["end"] and original_complete(job, plan))

    monkeypatch.setattr(matrix, "complete", complete)
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid=None: SimpleNamespace(create_time=lambda: 10, status=lambda: psutil.STATUS_RUNNING),
    )

    def start(command, **kwargs):
        identity = command[command.index("--job") + 1]
        job = jobs[identity]
        port = int(command[command.index("--port-block-start") + 1])
        live = [v for v in running.values() if v["end"] > clock["now"]]
        assert len(live) < 12
        assert port not in {v["port"] for v in live}
        assert kwargs["env"]["EXGENTIC_PORT_BLOCK_START"] == str(port)
        duration = 2400 if job["benchmark"] == "bfcl" else 30
        running[identity] = {"end": clock["now"] + duration, "port": port}
        starts.append((identity, clock["now"]))
        matrix.write(
            Path(job["output"]) / "job_complete.json",
            {
                "job_id": identity,
                "complete": True,
                "tasks": len(job["test_ids"]),
                "test_ids": job["test_ids"],
                "infrastructure_errors": 0,
                "inference_profile_sha256": plan["inference_profile_sha256"],
                "score": 0.5,
            },
        )
        return SimpleNamespace(pid=len(starts) + 100)

    monkeypatch.setattr(matrix.subprocess, "Popen", start)
    matrix.run(output)
    assert len(starts) == 36
    first_appworld = min(t for key, t in starts if jobs[key]["benchmark"] == "appworld")
    assert first_appworld < min(v["end"] for key, v in running.items() if jobs[key]["benchmark"] == "bfcl")
    assert matrix.read(output / "status.json")["completed"] == 36


def test_matching_task_ids_cannot_reuse_a_different_profile_completion(prepared):
    _output, plan = prepared
    job = plan["jobs"][0]
    matrix.write(
        Path(job["output"]) / "job_complete.json",
        {
            "job_id": job["id"],
            "complete": True,
            "tasks": len(job["test_ids"]),
            "test_ids": job["test_ids"],
            "infrastructure_errors": 0,
            "inference_profile_sha256": "different",
        },
    )
    with pytest.raises(ValueError, match="completion evidence"):
        matrix.complete(job, plan)


@pytest.mark.parametrize("partial,success", [(0.999, False), (1.0, True)])
def test_fixed_evaluation_recovers_official_appworld_flag_after_protocol_error(tmp_path, monkeypatch, partial, success):
    import exgentic.interfaces.lib.api as api

    session = SimpleNamespace(status=SimpleNamespace(value="error"), score=0.0, success=False,
                              session_id="session", task_id="task", cost_reports={},
                              details={"session_metadata": {"error_source": "agent",
                                      "error": "LiteLLM completion missing assistant content."}})

    def execute(*, config):
        matrix.write(tmp_path / "evaluation/sessions" / config.run_id / "sessions/session/benchmark/results.json",
                     {"score": partial, "success": success, "session_id": "session", "task_id": "task"})
        return SimpleNamespace(session_results=[session])

    monkeypatch.setattr(api, "execute", execute)
    job = {"id": "test", "benchmark": "appworld", "method": "ace", "output": str(tmp_path)}
    result = matrix.evaluate_task((job, {"benchmark_configs": {"appworld": {}}}, "", "task", 0, 1))
    assert result["score"] == int(success) and result["success"] is success
    assert result["official_reward"] == partial
    assert result["agent_protocol_error"] == session.details["session_metadata"]["error"]
