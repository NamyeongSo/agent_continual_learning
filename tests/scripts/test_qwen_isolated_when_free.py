# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Check deferred GPU launch safety and the all-role Qwen experiment contract."""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest

from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.agents.premise.llm import PremiseLLM
from exgentic.core.context import run_scope
from exgentic.core.types import ModelSettings
from exgentic.utils.cost import LiteLLMCostReport
from scripts.utils import run_qwen_isolated_when_free as runner


def test_resumed_environment_uses_local_qwen_and_docker_while_retaining_private_transport(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    original = {"PATH": "/usr/bin", "DOCKER_HOST": "ssh://fixture-host",
                "OPENAI_API_KEY": "fixture-key",  # pragma: allowlist secret
                "OPENAI_API_BASE": "https://api.openai.com/v1", "OPENAI_BASE_URL": "https://api.openai.com/v1"}
    env = runner.experiment_environment(original)
    assert env["OPENAI_API_BASE"] == env["OPENAI_BASE_URL"] == "http://127.0.0.1:18006/v1"
    assert env["PATH"].split(os.pathsep) == [str(tmp_path / ".venv/bin"), "/usr/bin"]
    assert env["DOCKER_HOST"] == original["DOCKER_HOST"] and env["OPENAI_API_KEY"] == original["OPENAI_API_KEY"]
    assert original["PATH"] == "/usr/bin"
    assert runner.experiment_environment(env) == env


def test_missing_docker_blocks_preflight_before_any_actor_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.os, "environ", dict(os.environ))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.shutil, "which", lambda _: None)
    monkeypatch.setattr(runner, "request", lambda *_: pytest.fail("A model must not run before runtime validation"))
    with pytest.raises(RuntimeError, match="Docker CLI"):
        runner.preflight_model(tmp_path)


def test_explicit_endpoint_covers_actor_and_analysis_environment_without_changing_defaults():
    env = runner.experiment_environment({"PATH": "/usr/bin"}, base_url="http://127.0.0.1:19007/v1")
    assert env["OPENAI_API_BASE"] == env["OPENAI_BASE_URL"] == "http://127.0.0.1:19007/v1"
    assert runner.BASE_URL == "http://127.0.0.1:18006/v1"


def test_training_only_freezes_the_result_without_creating_evaluation_workers(tmp_path, monkeypatch):
    job = {"id": "one", "method": "ace", "status": "pending", "attempts": 0,
           "output": str(tmp_path / "run"), "command": ["frozen-training"], "test_ids": ["test"]}
    plan = {"jobs": [job], "train_workers": 1, "base_url": "http://127.0.0.1:19007/v1"}
    calls = []
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda **_: pytest.fail("Extra inference must stay stopped"))
    monkeypatch.setattr(runner, "frozen_evaluation_jobs", lambda job: [{"id": "unused-frozen-test"}])
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    def launch(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    service = SimpleNamespace(poll=lambda: None)
    result = runner.run_training_and_evaluation(plan, tmp_path, service, service, defer_evaluation=True)
    assert result[0]["status"] == "test_pending" and result[0]["evaluation_deferred"]
    assert calls[0][0] == ["frozen-training"]
    assert calls[0][1]["OPENAI_API_BASE"] == calls[0][1]["OPENAI_BASE_URL"] == plan["base_url"]
    assert not (tmp_path / "run/evaluation/records").exists()


def test_deferred_final_test_can_resume_through_the_normal_evaluation_path(tmp_path, monkeypatch):
    from concurrent.futures import Future

    job = {"id": "one", "method": "ace", "status": "test_pending", "attempts": 1,
           "output": str(tmp_path / "run"), "test_ids": ["test"]}
    task = {"output": job["output"], "index": 0, "frozen_sha256": "frozen"}
    calls = []

    class Pool:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def submit(self, function, item):
            calls.append(item)
            future = Future()
            future.set_result({"status": "completed", "score": 0, "frozen_sha256": "frozen"})
            return future

    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda **_: Pool())
    monkeypatch.setattr(runner, "frozen_evaluation_jobs", lambda job: [task])
    service = SimpleNamespace(poll=lambda: None)
    result = runner.run_training_and_evaluation({"jobs": [job], "train_workers": 1}, tmp_path, service, service)
    assert calls == [task] and result[0]["status"] == "completed"
    assert runner.read(tmp_path / "run/evaluation/summary.json")["score"] == 0


@pytest.mark.parametrize("stage", ["reflection", "mutation", "judge"])
def test_premise_analysis_override_covers_every_stage(tmp_path, stage):
    captured = []
    llm = PremiseLLM("openai/test", "http://localhost/v1", temperature_override=0.7)

    def create(**kwargs):
        captured.append(kwargs)
        return litellm.ModelResponse(choices=[{
            "message": {"role": "assistant", "content": '{"valid": true}'}, "finish_reason": "stop",
        }])

    llm._client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert llm.ask("request", temperature=0, stage=stage) == {"valid": True}
    assert captured[0]["temperature"] == 0.7
    assert captured[0]["max_tokens"] == 8192


def test_ace_generator_and_learning_share_override_without_changing_analysis_budget(tmp_path, monkeypatch):
    captured = []
    actor = object.__new__(ACEAgentInstance)
    actor._model_settings = ModelSettings(max_tokens=8192)
    actor.temperature_override = 0.7
    actor.enable_thinking = False
    actor._cost = LiteLLMCostReport.initialize_empty(model_name="openai/test")
    actor._logger = logging.getLogger(__name__)

    def completion(**kwargs):
        captured.append(kwargs)
        return litellm.ModelResponse(choices=[{
            "message": {"role": "assistant", "content": "response"}, "finish_reason": "stop",
        }], usage={"prompt_tokens": 10, "completion_tokens": 10})

    monkeypatch.setattr(litellm, "completion", completion)
    with run_scope(run_id="temperature-test", output_dir=str(tmp_path)):
        actor._completion(model="openai/test", messages=[])
        actor._llm_call_simple("openai/test", "reflection")
        actor._llm_call_simple("openai/test", "curation")
    assert [request["temperature"] for request in captured] == [0.7, 0.7, 0.7]
    assert captured[1]["max_tokens"] == captured[2]["max_tokens"] == 8192


def test_plan_has_exactly_eighteen_jobs_and_frozen_split_counts(tmp_path):
    root = Path(__file__).resolve().parents[2]
    manifest = root / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
    plan = runner.build_plan(tmp_path, root, manifest)
    assert len(plan["jobs"]) == 18
    assert plan["training_iterations"] == 780
    assert plan["test_tasks"] == 774
    assert plan["train_workers"] == 6
    assert plan["evaluation_workers"] == 100
    assert plan["premise_workers"] == {"parent": 2, "reflection": 4, "mutation": 4, "replay": 28, "llm_slots": 32}
    for job in plan["jobs"]:
        command = job["command"]
        assert command[command.index("--temperature") + 1] == "0.7"
        assert command[command.index("--mode") + 1] == "isolated"
        assert "--evaluate-test" not in command  # Evaluation uses the shared 100-worker queue.
    assert plan["experiment_limits"]["bfcl"]["max_turns"] == 30
    assert plan["experiment_limits"]["terminalbench2"]["output_tokens"] == 2048


@pytest.mark.parametrize("memory,utilization,processes,free", [
    (0, 0, [], True), (300, 0, [], True), (1000, 0, [], False),
    (0, 100, [], False), (0, 0, [{"pid": 123}], False),
])
def test_gpu_must_have_no_compute_owners_and_be_idle(memory, utilization, processes, free):
    assert runner.gpu_is_free({"memory_mib": memory, "utilization": utilization, "processes": processes}) is free


@pytest.mark.parametrize("actor_context", ["legacy", "shared-empty-v1"])
def test_freeze_rejects_wrong_temperature_and_preserves_rank_one(tmp_path, actor_context):
    train = tmp_path / "train"
    train.mkdir()
    runner.write(train / "training_complete.json", {"completed_task_count": 50})
    state = {"cursor": 50, "config": {"temperature_all_roles": 0.7, "seed": 42,
                                       "mode": "isolated", "model": runner.MODEL},
             "states": {"bfcl": {"beam": [{"playbook": "rank one"}, {"playbook": "rank two"}]}}}
    if actor_context != "legacy":
        from exgentic.actor_context import actor_context_metadata

        state["config"].update(actor_context=actor_context,
                               actor_context_metadata=actor_context_metadata(actor_context, ["bfcl"]),
                               benchmark_configs={"bfcl": {"subset": "multi_turn_base"}})
    runner.write(train / "state.json", state)
    job = {"train_dir": str(train), "output": str(tmp_path), "method": "premise", "benchmark": "bfcl",
           "seed": 42, "train_count": 50, "test_ids": ["task"]}
    tasks = runner.frozen_evaluation_jobs(job)
    assert runner.read(tasks[0]["frozen"])["playbook"] == "rank one"
    assert tasks[0]["frozen_sha256"] == runner.digest(tasks[0]["frozen"])
    if actor_context != "legacy":
        assert tasks[0]["actor_context"] == actor_context
        assert tasks[0]["benchmark_kwargs"]["subset"] == "multi_turn_base"
        assert runner.read(tasks[0]["frozen"])["actor_context_metadata"] == state["config"]["actor_context_metadata"]
    state["config"]["temperature_all_roles"] = 0.0
    runner.write(train / "state.json", state)
    with pytest.raises(ValueError, match="temperature/seed"):
        runner.frozen_evaluation_jobs(job)


def test_ace_freeze_uses_completed_task_evidence_when_curator_counter_lags(tmp_path):
    train = tmp_path / "train"
    order = [["bfcl", f"t{i}"] for i in range(5)]
    runner.write(train / "training_complete.json", {"completed_task_count": 5})
    runner.write(train / "experiment_config.json", {"task_order": order, "temperature_all_roles": 0.7,
                                                    "seed": 42, "mode": "isolated", "model": runner.MODEL})
    metrics = []
    for index, (slug, task) in enumerate(order):
        session = train / "run/sessions" / str(index)
        runner.write(session / "config.json", {"benchmark": slug, "task_id": task})
        runner.write(session / "results.json", {"status": "unsuccessful", "score": 0})
        if index != 2:
            runner.write(session / "agent/playbook_checkpoint.json", {"playbook": f"policy{index}"})
        metrics.append({"benchmark_slug": slug, "task_id": task, "session_index": index,
                        "status": "unsuccessful"})
    (train / "online_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    path = train / "playbook_ace_isolated_bfcl.json"
    runner.write(path, {"playbook": "policy4", "session_count": 4})
    job = {"train_dir": str(train), "output": str(tmp_path), "method": "ace", "benchmark": "bfcl",
           "seed": 42, "train_count": 5, "test_ids": ["test"]}
    assert runner.read(runner.frozen_evaluation_jobs(job)[0]["frozen"])["playbook"] == "policy4"
    metrics[2]["status"] = "error"
    (train / "online_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    with pytest.raises(ValueError, match="without complete task/policy evidence"):
        runner.frozen_evaluation_jobs(job)


def test_gpu_query_ignores_other_gpus_but_preserves_owner_pid(monkeypatch):
    def query(command, **kwargs):
        if "--query-gpu=uuid,memory.used,utilization.gpu" in command:
            return "GPU-one, 97000, 100\n"
        return "GPU-zero, 1, 500\nGPU-one, 23, 97000\n"

    monkeypatch.setattr(runner.subprocess, "check_output", query)
    state = runner.gpu_state(1)
    assert state["processes"] == [{"pid": 23, "memory_mib": "97000"}]
    assert not runner.gpu_is_free(state)


def test_supplied_script_change_stops_launch_before_any_process(tmp_path, monkeypatch):
    script = tmp_path / "serve.sh"
    script.write_text("exit 0\n")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Must not launch"))
    with pytest.raises(ValueError, match="script changed"):
        runner.launch_server({"serve_script": str(script), "serve_script_sha256": "wrong"}, tmp_path)


def test_budget_failed_appworld_evaluation_is_a_measured_zero(tmp_path, monkeypatch):
    from exgentic.interfaces.lib import api

    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"playbook": "frozen"}))
    calls = []

    def execute(*, config):
        calls.append(config)
        result = SimpleNamespace(
            score=0.0, session_id="session", status=SimpleNamespace(value="unsuccessful"),
            details={"session_metadata": {"experiment_budget_exceeded": True,
                                          "experiment_budget_failure": {"reason": "input_tokens"}}},
        )
        return SimpleNamespace(session_results=[result])

    monkeypatch.setattr(api, "execute", execute)
    row = runner.evaluate_one({"frozen": str(frozen), "frozen_sha256": runner.digest(frozen),
                               "method": "premise", "benchmark": "appworld", "seed": 42,
                               "index": 0, "task_id": "task", "output": str(tmp_path)})
    assert row["status"] == "completed"
    assert row["score"] == 0
    assert calls[0].agent_kwargs["temperature_override"] == 0.7
    assert calls[0].max_steps == 40


def test_training_capacity_can_change_without_restarting_workers(tmp_path):
    assert runner.training_workers({"train_workers": 3}, tmp_path) == 3
    runner.write(tmp_path / "runtime_schedule.json", {"train_workers": 6})
    assert runner.training_workers({"train_workers": 3}, tmp_path) == 6
    runner.write(tmp_path / "runtime_schedule.json", {"train_workers": 0})
    with pytest.raises(ValueError, match="train_workers"):
        runner.training_workers({"train_workers": 3}, tmp_path)


def test_resume_validates_commands_before_adopting_jobs(tmp_path):
    job = {"id": "ace_bfcl_s42", "method": "ace", "benchmark": "bfcl", "seed": 42,
           "command": ["python", "original.py"], "train_dir": "train", "output": "output",
           "test_ids": ["one"], "train_count": 1, "status": "pending"}
    previous = {**job, "status": "training", "pid": 123}
    runner.write(tmp_path / "progress.json", {"jobs": [previous]})
    assert runner.resume_jobs({"jobs": [job]}, tmp_path) == [previous]
    previous["command"] = ["python", "different.py"]
    runner.write(tmp_path / "progress.json", {"jobs": [previous]})
    with pytest.raises(ValueError, match="command differs"):
        runner.resume_jobs({"jobs": [job]}, tmp_path)


def test_adopted_process_checks_identity_and_explicit_training_completion(tmp_path, monkeypatch):
    import psutil

    state = {"start": 100}
    process = SimpleNamespace(status=lambda: "sleeping", cmdline=lambda: ["training"],
                              create_time=lambda: state["start"])
    monkeypatch.setattr(psutil, "Process", lambda _pid: process)
    monkeypatch.setattr(runner, "process_start_ticks", lambda _pid: None)
    marker = tmp_path / "training_complete.json"
    adopted = runner.ExistingProcess(123, command=["training"], completion=marker)
    assert adopted.poll() is None
    with pytest.raises(ValueError, match="different command"):
        runner.ExistingProcess(123, command=["other"])
    state["start"] = 200  # A reused PID must never be mistaken for our worker.
    assert adopted.poll() == 1
    completed = runner.ExistingProcess(123, command=["training"], completion=marker)
    marker.write_text("{}")
    state["start"] = 300
    assert completed.poll() == 0


def test_six_stream_scheduler_preserves_three_live_workers_and_fills_other_benchmark(tmp_path, monkeypatch):
    jobs = []
    for index in range(9):
        folder = tmp_path / str(index)
        folder.mkdir()
        jobs.append({"id": str(index), "status": "training" if index < 3 else "pending",
                     "method": "ace", "benchmark": "appworld" if index < 3 else "terminalbench2",
                     "output": str(folder), "command": ["train", str(index)], "attempts": int(index < 3),
                     "pid": 100 + index if index < 3 else None})
    spawned = []
    captured = []
    live = SimpleNamespace(poll=lambda: None)

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 100

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    class EndCycleError(Exception):
        pass

    def spawn(command, **kwargs):
        spawned.append((command, kwargs["env"]["EXGENTIC_PORT_BLOCK_START"]))
        return SimpleNamespace(pid=200 + len(spawned), poll=lambda: None)

    def capture(path, value):
        assert path.name == "progress.json"
        captured.append(value)
        raise EndCycleError

    monkeypatch.setattr(runner, "resume_jobs", lambda *_args: jobs)
    monkeypatch.setattr(runner, "adopt_training", lambda job: (live, job, int(job["id"]), io.BytesIO()))
    monkeypatch.setattr(runner, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    monkeypatch.setattr(runner, "write", capture)
    with pytest.raises(EndCycleError):
        runner.run_training_and_evaluation({"train_workers": 6}, tmp_path, live, live, resume=True)
    assert spawned == [(["train", str(index)], str(27000 + index * 1000)) for index in (3, 4, 5)]
    assert [job["pid"] for job in jobs[:3]] == [100, 101, 102]
    assert [job["attempts"] for job in jobs[:3]] == [1, 1, 1]
    assert captured[0]["running_training"] == captured[0]["train_workers"] == 6
    assert [job["status"] for job in jobs[6:]] == ["pending"] * 3


def test_editable_plan_passes_policy_to_every_method_and_benchmark(tmp_path):
    root = Path(__file__).resolve().parents[2]
    manifest = root / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
    plan = runner.build_plan(tmp_path, root, manifest, playbook_edit_mode="add_update_delete")
    assert len(plan["jobs"]) == 18 and plan["training_iterations"] == 780 and plan["test_tasks"] == 774
    for job in plan["jobs"]:
        assert job["playbook_edit_mode"] == "add_update_delete"
        assert job["command"][job["command"].index("--playbook-edit-mode") + 1] == "add_update_delete"


def test_per_method_capacity_fills_six_ace_and_six_premise_slots(tmp_path):
    jobs = [{"id": f"{method}{index}", "method": method, "status": "pending"}
            for method in ("ace", "premise") for index in range(9)]
    plan = {"train_workers_per_method": {"ace": 6, "premise": 6}}
    running = {}
    for index in range(12):
        job = runner.next_training_job(plan, jobs, running)
        assert job is not None
        job["status"] = "training"
        running[job["id"]] = (None, job, index, None)
    assert sum(item[1]["method"] == "ace" for item in running.values()) == 6
    assert sum(item[1]["method"] == "premise" for item in running.values()) == 6
    assert runner.next_training_job(plan, jobs, running) is None
    finished = running.pop("premise0")[1]
    finished["status"] = "completed"
    assert runner.next_training_job(plan, jobs, running)["id"] == "premise6"


def test_curator_schedule_launches_nine_ace_and_six_premise_then_refills_with_terminalbench():
    jobs = [{"id": f"{method}_{benchmark}_{seed}", "method": method, "benchmark": benchmark, "status": "pending"}
            for method in ("ace", "premise") for benchmark in ("bfcl", "appworld", "terminalbench2")
            for seed in (42, 43, 44)]
    plan = {"train_workers_per_method": {"ace": 9, "premise": 6}, "port_block_base": 50000}
    running = {}
    for slot in range(15):
        job = runner.next_training_job(plan, jobs, running)
        job["status"] = "training"
        running[job["id"]] = (None, job, slot, None)
    assert sum(v[1]["method"] == "ace" for v in running.values()) == 9
    assert {v[1]["benchmark"] for v in running.values() if v[1]["method"] == "premise"} == {"bfcl", "appworld"}
    assert runner.next_training_job(plan, jobs, running) is None
    for seed in (42, 43, 44):
        completed = running.pop(f"premise_bfcl_{seed}")[1]
        completed["status"] = "evaluating"
        next_job = runner.next_training_job(plan, jobs, running)
        assert next_job["id"] == f"premise_terminalbench2_{seed}"
        next_job["status"] = "training"
        running[next_job["id"]] = (None, next_job, 9, None)
        assert sum(v[1]["method"] == "premise" for v in running.values()) == 6
    assert runner.training_port_base(plan, 15) == 50000
    with pytest.raises(ValueError, match="port blocks"):
        runner.training_port_base(plan, 16)


def test_freeze_rejects_add_only_checkpoint_for_editable_job(tmp_path):
    train = tmp_path / "train"
    runner.write(train / "training_complete.json", {"completed_task_count": 1})
    state = {"cursor": 1, "config": {"temperature_all_roles": 0.7, "seed": 42,
                                     "mode": "isolated", "model": runner.MODEL},
             "states": {"bfcl": {"beam": [{"playbook": "learned"}]}}}
    runner.write(train / "state.json", state)
    job = {"train_dir": str(train), "output": str(tmp_path), "method": "premise", "benchmark": "bfcl",
           "seed": 42, "train_count": 1, "test_ids": ["task"], "playbook_edit_mode": "add_update_delete"}
    with pytest.raises(ValueError, match="edit mode differs"):
        runner.frozen_evaluation_jobs(job)
    state["config"]["playbook_edit_mode"] = "add_update_delete"
    runner.write(train / "state.json", state)
    tasks = runner.frozen_evaluation_jobs(job)
    assert tasks[0]["playbook_edit_mode"] == "add_update_delete"
    assert runner.read(tasks[0]["frozen"])["playbook_edit_mode"] == "add_update_delete"


def test_adopted_process_survives_clock_correction_but_rejects_pid_reuse(monkeypatch):
    import psutil

    state = {"created": 100, "ticks": "1234", "boot": "same-boot"}
    process = SimpleNamespace(status=lambda: "sleeping", cmdline=lambda: ["training"],
                              create_time=lambda: state["created"])
    monkeypatch.setattr(psutil, "Process", lambda _pid: process)
    monkeypatch.setattr(runner, "process_start_ticks", lambda _pid: state["ticks"])
    monkeypatch.setattr(runner, "current_boot_id", lambda: state["boot"])
    adopted = runner.ExistingProcess(123, command=["training"])
    state["created"] += 3  # psutil boot-time changes after a wall-clock correction.
    assert adopted.poll() is None
    state["ticks"] = "5678"
    assert adopted.poll() == 1
    next_boot = runner.ExistingProcess(123, command=["training"])
    state["boot"] = "different-boot"
    assert next_boot.poll() == 1
