# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Development-stream ordering, submission boundaries, replay, and aggregation."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from exgentic import SWEMilestoneBenchmark
from exgentic.benchmarks.swemilestone import swemilestone_eval, worker
from exgentic.benchmarks.swemilestone.swemilestone_benchmark import (
    BashAction,
    BashArgs,
    RunTerminus2Action,
    RunTerminus2Args,
    SubmitArgs,
    SubmitMilestoneAction,
)
from exgentic.benchmarks.swemilestone.swemilestone_eval import SWEMilestoneEvaluator, SWEMilestoneSession
from exgentic.benchmarks.swemilestone.transport import MilestoneBackend, ssh_command
from exgentic.core.context import run_scope
from exgentic.core.orchestrator.session import _send_training_feedback
from exgentic.core.types import RunConfig, SessionIndex, SessionScore
from exgentic.interfaces.registry import load_benchmark

ROWS = [
    {
        "task_id": "repo-a/first",
        "repository": "repo-a",
        "repo_alias": "a",
        "milestone_id": "first",
        "graded": True,
        "srs": "First specification",
        "prerequisites": [],
    },
    {
        "task_id": "repo-a/second",
        "repository": "repo-a",
        "repo_alias": "a",
        "milestone_id": "second",
        "graded": True,
        "srs": "Second specification",
        "prerequisites": ["first"],
    },
    {
        "task_id": "repo-a/style",
        "repository": "repo-a",
        "repo_alias": "a",
        "milestone_id": "style",
        "graded": False,
        "srs": "Style specification",
        "prerequisites": [],
    },
    {
        "task_id": "repo-b/first",
        "repository": "repo-b",
        "repo_alias": "b",
        "milestone_id": "first",
        "graded": True,
        "srs": "Other specification",
        "prerequisites": [],
    },
]


@pytest.fixture
def fake_backend(monkeypatch):
    calls = []
    snapshot = {"image_id": "sha256:" + "a" * 64, "submitted": {}}

    def call(self, operation, **kwargs):
        calls.append((operation, kwargs))
        if operation == "catalog":
            return ROWS
        if operation == "start":
            return {"submitted": False, "start_snapshot": snapshot}
        if operation == "execute":
            return {"stdout": "previous milestone file\n", "stderr": "", "exit_code": 0}
        if operation == "submit":
            return {"submitted_commit": "c" * 40}
        if operation == "evaluate":
            submitted = any(op == "submit" and kw["session_id"] == kwargs["session_id"] for op, kw in calls)
            return {
                "submitted": submitted,
                "score": 0.5 if submitted else 0.0,
                "resolved": False,
                "precision": 1.0,
                "recall": 1 / 3,
                "evaluation": {"resolved": False},
            }
        if operation == "close":
            return {}
        raise AssertionError(operation)

    monkeypatch.setattr(MilestoneBackend, "call", call)
    return calls


def test_registry_is_lightweight_and_ordered_prefix_is_not_shuffled(fake_backend, tmp_path):
    assert load_benchmark("swemilestone") is SWEMilestoneBenchmark
    config = RunConfig(benchmark="swemilestone", agent="ace", num_tasks=2, benchmark_kwargs={"runner": "direct"})
    with run_scope(run_id="ordered", output_dir=str(tmp_path)):
        assert [s.task_id for s in config.get_session_configs()] == [row["task_id"] for row in ROWS[:2]]
    assert "start" not in [op for op, _ in fake_backend]


@pytest.mark.parametrize("tasks", [None, ["repo-a/first"]])
def test_parallel_development_stream_is_rejected_before_start(tasks):
    config = RunConfig(benchmark="swemilestone", agent="ace", task_ids=tasks, max_workers=2)
    with pytest.raises(ValueError, match="requires max_workers=1"):
        config.get_session_configs()


def test_sessions_share_namespace_and_grade_only_after_submission(fake_backend, tmp_path):
    with run_scope(run_id="stream", output_dir=str(tmp_path)):
        evaluator = SWEMilestoneEvaluator()
        first = SWEMilestoneSession(
            **evaluator.get_session_kwargs(SessionIndex(task_id="repo-a/first", session_id="a"))
        )
        first.start()
        assert "First specification" in first.task
        observation = first.step(BashAction(arguments=BashArgs(command="cat carried-file")))
        assert "previous milestone file" in observation.result
        assert "evaluate" not in [op for op, _ in fake_backend]
        assert first.step(SubmitMilestoneAction(arguments=SubmitArgs())) is None
        score = first.score()
        assert score.score == 0.5 and score.success and score.is_finished
        assert score.session_metadata["start_snapshot"]["image_id"].startswith("sha256:")
        assert first.paths.benchmark_results.exists()
        assert first.paths.benchmark_dir.joinpath("start_snapshot.json").exists()
        assert first.score() is score
        first.close()
        first.close()
        second = SWEMilestoneSession(
            **evaluator.get_session_kwargs(SessionIndex(task_id="repo-a/second", session_id="b"))
        )
        second.start()
        second.close()
    starts = [kw for op, kw in fake_backend if op == "start"]
    assert starts[0]["namespace"] == starts[1]["namespace"]
    assert len([op for op, _ in fake_backend if op == "evaluate"]) == 1
    assert len([op for op, _ in fake_backend if op == "close"]) == 2


def test_interaction_limit_does_not_submit_partial_code(fake_backend, tmp_path):
    with run_scope(run_id="limit", output_dir=str(tmp_path)):
        evaluator = SWEMilestoneEvaluator(max_interactions=1)
        session = SWEMilestoneSession(
            **evaluator.get_session_kwargs(SessionIndex(task_id="repo-a/first", session_id="a"))
        )
        session.start()
        session.step(BashAction(arguments=BashArgs(command="edit source")))
        assert session.done()
        score = session.score()
        assert not score.is_finished and score.score == 0
        session.close()
    assert "submit" not in [op for op, _ in fake_backend]


def test_ungraded_sessions_do_not_send_false_ground_truth_feedback(fake_backend, tmp_path):
    received = []
    agent = SimpleNamespace(training_time=True)
    instance = SimpleNamespace(receive_training_feedback=received.append)
    with run_scope(run_id="ungraded", output_dir=str(tmp_path)):
        evaluator = SWEMilestoneEvaluator()
        session = SWEMilestoneSession(
            **evaluator.get_session_kwargs(SessionIndex(task_id="repo-a/style", session_id="s"))
        )
        session.start()
        session.step(SubmitMilestoneAction(arguments=SubmitArgs()))
        score = session.score()
        _send_training_feedback(agent, instance, score)
        assert not received
        _send_training_feedback(agent, instance, SessionScore(score=1.0, success=True))
        assert len(received) == 1  # Other benchmarks retain the existing default.
        session.close()


def test_macro_average_excludes_ungraded_and_counts_missing_as_zero(fake_backend, tmp_path):
    with run_scope(run_id="aggregate", output_dir=str(tmp_path)):
        evaluator = SWEMilestoneEvaluator()
        indexes = [SessionIndex(task_id=row["task_id"], session_id=str(i)) for i, row in enumerate(ROWS)]
        # Only the first graded milestone is evaluated; an ungraded task's score must not inflate the result.
        for index, value in [(indexes[0], 1.0), (indexes[2], 1.0)]:
            path = evaluator.get_sessions_paths([index])[0].benchmark_results
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"score": value, "session_metadata": {"submitted": True}, "session_metrics": {"resolved": True}}
                )
            )
        result = evaluator.aggregate_sessions(indexes)
        assert result.score == 0.25  # mean(repo-a: 1/2, repo-b: 0/1)
        assert result.total_tasks == 4
        assert result.metrics["missing_graded_milestones"] == 2
        assert result.metrics["ungraded_milestones"] == 1
        assert result.metrics["partial"] is True
        with pytest.raises(ValueError, match="Duplicate milestone"):
            evaluator.aggregate_sessions([indexes[0], indexes[0]])


def test_ssh_transport_quotes_remote_command_and_preserves_user_port():
    command = ["python3", "-c", "print('literal $HOME; `whoami`')"]
    result = ssh_command("ssh://person@example.test:2222", command)
    assert result[-2] == "person@example.test"
    assert result[result.index("-p") + 1] == "2222"
    assert shlex.split(result[-1]) == command


@pytest.mark.parametrize(
    "endpoint", ["tcp://host:2375", "ssh://-host", "ssh://-user@host", "ssh://u:p@host", "ssh://host/path"]
)
def test_ssh_transport_rejects_unsupported_endpoints(endpoint):
    with pytest.raises(ValueError):
        ssh_command(endpoint, ["python3"])


def test_backend_propagates_errors_and_pinned_provenance(monkeypatch):
    requests = []

    def invoke(command, **kwargs):
        requests.append(json.loads(kwargs["input"]))
        return subprocess.CompletedProcess(command, 1, '{"error": "blocked prerequisite"}', "")

    monkeypatch.setattr(subprocess, "run", invoke)
    backend = MilestoneBackend("ssh://docker-alias")
    with pytest.raises(RuntimeError, match="blocked prerequisite"):
        backend.call("start", task_id="repo/milestone")
    assert requests[0]["benchmark_version"] == "v1.0.2"
    assert len(requests[0]["harness_commit"]) == 40
    assert backend.namespace("/run/one", "branch-a") != backend.namespace("/run/one", "branch-b")


def test_backend_retries_only_disconnects_before_remote_command_execution(monkeypatch):
    attempts = []
    replies = iter([
        subprocess.CompletedProcess([], 255, "", "kex_exchange_identification: Connection reset by peer"),
        subprocess.CompletedProcess([], 0, '{"result": {"exit_code": 0}}', ""),
    ])

    def invoke(command, **kwargs):
        attempts.append((json.loads(kwargs["input"]), kwargs["timeout"]))
        return next(replies)

    monkeypatch.setattr(subprocess, "run", invoke)
    monkeypatch.setattr("exgentic.benchmarks.swemilestone.transport.time.sleep", lambda _seconds: None)
    backend = MilestoneBackend("ssh://docker-alias")
    assert backend.call("execute", timeout=10, command="change-file") == {"exit_code": 0}
    assert len(attempts) == 2
    assert attempts[0][0] == attempts[1][0]
    assert 0 < attempts[1][1] <= attempts[0][1] <= 10


def test_backend_does_not_repeat_commands_after_a_connection_loss(monkeypatch):
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 255, "", "Connection to host closed by remote host.")

    monkeypatch.setattr(subprocess, "run", invoke)
    with pytest.raises(RuntimeError, match="backend failed"):
        MilestoneBackend("ssh://docker-alias").call("execute", command="change-file")
    assert len(calls) == 1


def test_backend_handshake_retry_is_bounded(monkeypatch):
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 255, "", "kex_exchange_identification: Connection reset")

    monkeypatch.setattr(subprocess, "run", invoke)
    monkeypatch.setattr("exgentic.benchmarks.swemilestone.transport.time.sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="backend failed"):
        MilestoneBackend("ssh://docker-alias").call("execute", command="change-file")
    assert len(calls) == 3


def test_worker_keeps_subprocess_progress_out_of_json_stdout():
    program = (
        "import os; from exgentic.benchmarks.swemilestone import worker; "
        "worker.catalog = lambda request: (os.write(1, b'docker progress\\n'), print('harness progress'), [1])[-1]; "
        "worker.main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], input='{"operation":"catalog"}', capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == {"result": [1]}
    assert "docker progress" in result.stderr and "harness progress" in result.stderr


def test_worker_rejects_concurrent_and_blocked_sessions():
    request = {"milestone_id": "next", "session_id": "second"}
    with pytest.raises(RuntimeError, match="active session"):
        worker.start(request, {"active_session": "first"}, SimpleNamespace(), [])
    engine = SimpleNamespace(dag=SimpleNamespace(get_next_runnable=lambda: ["prerequisite"]))
    with pytest.raises(RuntimeError, match="blocked"):
        worker.start(request, {}, engine, [])


def test_worker_capture_uses_pinned_partition_when_metadata_omits_production_directories(monkeypatch):
    evaluator = ModuleType("harness.e2e.evaluator")
    canonical = SimpleNamespace(
        src_dirs=["dubbo-common/", "dubbo-compiler/"], test_dirs=["**/src/test/**"],
        exclude_patterns=["**/target/**"], generated_patterns=[], modifiable_test_patterns=[],
    )
    binding = SimpleNamespace(config={"residue_prune": True}, sha256="a" * 64)
    metadata = {"repo_src_dirs": ["dubbo-common/"]}

    def resolve(config, supplied_metadata, **kwargs):
        assert config == binding.config and supplied_metadata is metadata
        assert kwargs == {"repo_config_binding_mode": "trial-pinned", "repo_config_sha256": binding.sha256}
        return SimpleNamespace(src_filter=canonical)

    evaluator.resolve_residue_prune_config = resolve
    monkeypatch.setitem(sys.modules, evaluator.__name__, evaluator)
    actual = worker.capture_filter_kwargs(binding, metadata)
    assert "dubbo-compiler/" in actual["repo_src_dirs"]
    assert actual["test_dirs"] == canonical.test_dirs
    assert actual["exclude_patterns"] == canonical.exclude_patterns


def test_worker_unsubmitted_close_restores_full_start_image():
    request = {"milestone_id": "next", "session_id": "session"}
    state = {"active_session": "session", "starts": {"next": {"image_id": "checkpoint"}}}
    engine = SimpleNamespace(container_name="persistent-container")
    result = worker.close_session(request, state, engine)
    assert state["restore_image"] == "checkpoint" and state["active_session"] is None
    assert result["retained_container"] == "persistent-container"
    assert result["submitted"] is False


def test_worker_failed_grade_does_not_roll_back_valid_submission():
    request = {"milestone_id": "first", "session_id": "session"}
    state = {"active_session": "session", "submitted": {"first": "commit"}, "current_image": "submitted-image"}
    worker.close_session(request, state, SimpleNamespace(container_name="persistent-container"))
    assert "restore_image" not in state
    assert state["submitted"] == {"first": "commit"}


@pytest.mark.parametrize("completed", [False, True])
def test_terminus_observation_reaches_ace_before_confirmed_submission(fake_backend, monkeypatch, tmp_path, completed):
    from exgentic.agents.ace.ace_instance import ACEAgentInstance

    requests = []

    def run_actor(request, **kwargs):
        requests.append(request)
        return {
            "completed": completed,
            "n_episodes": 2,
            "trace_path": "trajectory.json",
            "agent_trace": "full terminal trajectory",
            "input_tokens": 10,
            "output_tokens": 2,
            "summarization_count": 1,
            "harbor_version": "0.23.0",
        }

    def unexpected_model_call(**kwargs):
        raise AssertionError("ACE must delegate acting to Terminus-2")

    monkeypatch.setattr(swemilestone_eval, "run_terminus", run_actor)
    monkeypatch.setattr("exgentic.agents.ace.ace_instance.litellm.completion", unexpected_model_call)
    with run_scope(run_id=f"terminus-{completed}", output_dir=str(tmp_path)):
        evaluator = SWEMilestoneEvaluator(actor="terminus2", max_interactions=2)
        session = SWEMilestoneSession(
            **evaluator.get_session_kwargs(SessionIndex(task_id="repo-a/first", session_id="t"))
        )
        observation = session.start()
        agent = ACEAgentInstance("t", benchmark_id="swemilestone", evaluation_mode=True)
        agent.start(session.task, session.context, session.actions)
        action = agent.react(observation)
        assert isinstance(action, RunTerminus2Action)
        observation = session.step(action)
        assert not session.done() and "submit" not in [op for op, _ in fake_backend]
        finish = agent.react(observation)
        assert isinstance(finish, SubmitMilestoneAction)
        assert "full terminal trajectory" in agent._build_session_trace()
        session.step(finish)
        score = session.score()
        assert score.is_finished is completed
        assert score.session_metadata["context_recovery"] == "on_overflow"
        assert score.session_metadata["proactive_summarization_threshold"] == 0
        assert score.session_metadata["summarization_count"] == 1
        assert ("submit" in [op for op, _ in fake_backend]) is completed
        assert requests[0]["max_turns"] == 2
        assert "First specification" in requests[0]["instruction"]
        session.close()
        agent.close()


def test_terminus_infrastructure_error_is_not_swallowed_as_a_tool_observation(fake_backend, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("SSH transport failed")

    monkeypatch.setattr(swemilestone_eval, "run_terminus", fail)
    with run_scope(run_id="terminus-error", output_dir=str(tmp_path)):
        evaluator = SWEMilestoneEvaluator(actor="terminus2")
        session = SWEMilestoneSession(
            **evaluator.get_session_kwargs(SessionIndex(task_id="repo-a/first", session_id="t"))
        )
        session.start()
        with pytest.raises(RuntimeError, match="SSH transport failed"):
            session.step(RunTerminus2Action(arguments=RunTerminus2Args(model="openai/test")))
        session.close()
    assert "submit" not in [op for op, _ in fake_backend]


def test_worker_exec_preserves_literal_environment_and_working_directory(tmp_path):
    working = tmp_path / "path with spaces"
    working.mkdir()
    marker = tmp_path / "must-not-exist"
    value = f"literal $(touch {shlex.quote(str(marker))}); 'quoted'"

    def docker_exec(command, **kwargs):
        assert kwargs["user"] == "fakeroot"
        return subprocess.run(command, capture_output=True, text=True)

    engine = SimpleNamespace(container_setup=SimpleNamespace(docker_exec=docker_exec))
    request = {
        "milestone_id": "first",
        "session_id": "t",
        "command_timeout_sec": 5,
        "command": 'printf "%s\\n" "$TASK_VALUE"; pwd',
        "cwd": str(working),
        "env": {"TASK_VALUE": value},
    }
    result = worker.execute(request, {"active_session": "t"}, engine)
    assert result["stdout"] == f"{value}\n{working}\n" and result["exit_code"] == 0
    assert not marker.exists()
    result = worker.execute(
        {**request, "cwd": str(tmp_path / "missing"), "command": f"pwd; touch {shlex.quote(str(marker))}"},
        {"active_session": "t"},
        engine,
    )
    assert result["exit_code"] != 0 and not marker.exists()
    with pytest.raises(ValueError, match="Invalid command environment"):
        worker.execute({**request, "env": {"BAD;KEY": "value"}}, {"active_session": "t"}, engine)


def test_worker_source_capture_failure_does_not_unlock_dependents(monkeypatch):
    request = {"milestone_id": "first", "session_id": "session", "repository": "repo"}
    state = {"active_session": "session"}
    monkeypatch.setattr(worker, "pull_image", lambda *args: "pinned-image")
    engine = SimpleNamespace(
        dag=SimpleNamespace(get_next_runnable=lambda: ["first"]),
        _docker_exec_git=lambda *args: subprocess.CompletedProcess(args, 0, "c" * 40, ""),
        _handle_submission=lambda *args, **kwargs: False,
    )
    with pytest.raises(RuntimeError, match="capture was rejected"):
        worker.submit(request, state, engine, [{"milestone_id": "first", "graded": True}])
    assert not state.get("submitted")


@pytest.mark.parametrize(
    "error, expected_zero",
    [
        (
            "RuntimeError: No valid test report files generated under /trial/artifacts\n"
            "First fatal error: compiler error",
            True,
        ),
        ("InfrastructureFailureError: outer test runner returned 137", False),
        ("RuntimeError: Patch application failed", False),
    ],
)
def test_worker_compilation_failure_is_zero_but_infrastructure_failure_is_an_error(
    monkeypatch, tmp_path, error, expected_zero
):
    calls = []
    collector = ModuleType("harness.e2e.collect_results")
    collector.calculate_score_reliable = lambda result: 0.0
    collector.calculate_precision_recall = lambda result: (0.0, 0.0)
    collector.check_log_for_failure = lambda path: "compilation_failure"
    collector.is_infra_invalid = lambda result: False
    collector.is_resolved = lambda result: False
    evaluator = ModuleType("harness.e2e.evaluator")
    evaluator.generate_filtered_evaluation = lambda *args: None
    orchestrator = ModuleType("harness.e2e.orchestrator")

    def grade(**kwargs):
        calls.append(kwargs)
        return "first", False, False, None, error

    orchestrator.run_evaluation_task = grade
    for module in (collector, evaluator, orchestrator):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    summary = tmp_path / "summary.json"

    def record(mid, dag_status, eval_status, **kwargs):
        summary.write_text(json.dumps({"results": {mid: {"eval_status": eval_status, "error": kwargs["error_msg"]}}}))

    engine = SimpleNamespace(
        trial_root=tmp_path,
        workspace_root=tmp_path,
        _update_evaluation_summary=record,
        _summary_file_path=lambda: summary,
    )
    request = {"milestone_id": "first", "session_id": "session"}
    state = {"active_session": "session", "submitted": {"first": "commit"}, "evaluations": {"first": {}}}
    rows = [{"milestone_id": "first", "graded": True}]
    if expected_zero:
        outcome = worker.evaluate(request, state, engine, rows)
        assert outcome["score"] == 0 and outcome["build_failure"] and outcome["submitted"]
        assert worker.evaluate(request, state, engine, rows) == outcome
        assert len(calls) == 1
    else:
        with pytest.raises(RuntimeError, match="Official milestone evaluation failed"):
            worker.evaluate(request, state, engine, rows)
        assert not state.get("build_failures")


@pytest.mark.parametrize("has_filter", [False, True])
def test_worker_scores_official_filtered_report_and_reuses_immutable_evaluation(monkeypatch, tmp_path, has_filter):
    collector = ModuleType("harness.e2e.collect_results")
    collector.calculate_score_reliable = lambda result: result["official_score"]
    collector.calculate_precision_recall = lambda result: (result["official_score"], 1.0)
    collector.check_log_for_failure = lambda path: None
    collector.is_infra_invalid = lambda result: False
    collector.is_resolved = lambda result: result["resolved"]
    evaluator = ModuleType("harness.e2e.evaluator")
    orchestrator = ModuleType("harness.e2e.orchestrator")
    result_dir = tmp_path / "evaluation" / "first"
    calls = []

    def grade(**kwargs):
        assert all(
            isinstance(kwargs[key], type(tmp_path))
            for key in ("snapshot_path", "repo_config_path", "runtime_policy_path")
        )
        assert kwargs["repo_config_sha256"] == "frozen-config-hash"
        calls.append(kwargs)
        result_dir.mkdir(parents=True)
        (result_dir / "evaluation_result.json").write_text(json.dumps({"official_score": 0.1, "resolved": False}))
        return "first", False, False, object(), None

    def filter_report(original, workspace, mid):
        assert workspace == tmp_path and mid == "first"
        if not has_filter:
            return None
        filtered = original.with_name("evaluation_result_filtered.json")
        filtered.write_text(json.dumps({"official_score": 0.75, "resolved": True}))
        return filtered

    orchestrator.run_evaluation_task = grade
    evaluator.generate_filtered_evaluation = filter_report
    for module in (collector, evaluator, orchestrator):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    engine = SimpleNamespace(trial_root=tmp_path, workspace_root=tmp_path)
    request = {"milestone_id": "first", "session_id": "session"}
    state = {
        "active_session": "session",
        "submitted": {"first": "commit"},
        "evaluations": {
            "first": {
                "snapshot_path": str(tmp_path / "source_snapshot.tar"),
                "repo_config_path": str(tmp_path / "config.yaml"),
                "runtime_policy_path": str(tmp_path / "policy.json"),
                "repo_config_sha256": "frozen-config-hash",
            }
        },
    }
    rows = [{"milestone_id": "first", "graded": True}]
    result = worker.evaluate(request, state, engine, rows)
    assert result["score"] == (0.75 if has_filter else 0.1)
    assert result["resolved"] is has_filter
    assert worker.evaluate(request, state, engine, rows) == result
    assert len(calls) == 1
