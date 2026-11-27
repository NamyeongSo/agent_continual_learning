# SPDX-License-Identifier: Apache-2.0
"""Generated contexts may contain CR/CRLF; validate exact bytes without weakening integrity checks."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ace import evaluate_advantage_accumulation as evaluator
from scripts.ace import run_context_proposal_bottleneck as base
from scripts.ace import run_gemma4_train_only as gemma
from scripts.ace import run_reflection_accumulation as accumulation
from scripts.ace import run_training_first as training


@pytest.mark.parametrize("context", ["First\rSecond\r", "First\r\nSecond\r\n"])
def test_actor_receives_exact_context_and_still_rejects_actual_file_changes(tmp_path, monkeypatch, context):
    context_path = tmp_path / "context.txt"
    context_path.write_bytes(context.encode("utf-8"))
    job = {"id": "evaluation", "stage": 3, "task_id": "task", "source_id": "source", "context_id": "candidate"}
    base.write(tmp_path / "stage2/candidate/result.json", {
        "task_id": "task", "source_id": "source", "updated_context_path": str(context_path),
        "updated_context_sha256": base.digest(context),
    })
    protocol = {"initial_playbook": "empty", "model": "openai/test", "model_settings": {"temperature": 1},
                "actor_context": "shared-empty-v1", "benchmark_kwargs": {"subset": "test_challenge"}}
    received = []

    def execute(*, config):
        received.append(config.agent_kwargs["initial_playbook"])
        directory = Path(config.output_dir) / config.run_id / "sessions/saved/agent"
        base.write(directory / "ace_capture.json", {"playbook": config.agent_kwargs["initial_playbook"]})
        return SimpleNamespace(session_results=[SimpleNamespace(
            session_id="saved", score=1.0, status="completed", cost_reports={},
        )])

    monkeypatch.setattr("exgentic.interfaces.lib.api.execute", execute)
    monkeypatch.setattr(base, "pct_outcome", lambda *args: {"pct": 1.0, "success": True})
    result = base.actor_job(tmp_path, protocol, job, tmp_path / "attempt")
    assert received == [context]
    assert result["initial_context_sha256"] == base.digest(context_path.read_bytes())
    context_path.write_bytes(context.encode("utf-8") + b"changed")
    with pytest.raises(ValueError, match="Updated context changed"):
        base.actor_job(tmp_path, protocol, job, tmp_path / "another_attempt")
    assert received == [context]


@pytest.mark.parametrize("context", ["First\rSecond\r", "First\r\nSecond\r\n"])
def test_checkpoint_validation_and_final_evaluation_preserve_generated_newlines(tmp_path, context):
    base.write(tmp_path / "protocol.json", {
        "arms": ["without_gt"], "task_orders": {"42": {"train": ["task"], "val": []}},
    })
    for condition in accumulation.conditions(42, ("without_gt",)):
        directory = tmp_path / "conditions" / condition["id"]
        base.write(directory / "checkpoints/001.json", {
            "task_id": "task", "playbook": context, "session_count": 1, "condition": condition,
        })
        (directory / "checkpoints/001.txt").write_bytes(context.encode("utf-8"))
        base.write(directory / "training_completed.json", {})
        assert training.completed({"path": str(tmp_path), "condition": condition, "steps": 1})
    jobs = accumulation.eval_jobs(tmp_path, 42, "train_final")
    assert len(jobs) == 3
    for job in jobs:
        parent = base.read(tmp_path / "seed42/stage2" / job["id"] / "result.json")
        assert parent["updated_context_sha256"] == base.digest(context)


def test_saved_gemma_result_validation_does_not_normalize_context(tmp_path):
    context = "literal\rreturn\rcharacters"
    path = tmp_path / "context.txt"
    path.write_bytes(context.encode("utf-8"))
    job = {"id": "saved", "stage": 3, "context_id": "candidate"}
    base.write(tmp_path / "protocol.json", {"initial_playbook": "empty", "appworld_policy_sha256": "policy"})
    base.write(tmp_path / "stage2/candidate/result.json", {"updated_context_path": str(path)})
    row = {**job, "pct": 1.0, "success": True, "passed_tests": 1, "total_tests": 1,
           "initial_context_sha256": base.digest(context), "appworld_policy_sha256": "policy"}
    base.write(tmp_path / "stage3/saved/result.json", row)
    assert gemma.validate_reflection({"root": str(tmp_path), "job": job}) == row


def test_advantage_final_evaluation_accepts_exact_cr_context(tmp_path):
    source = tmp_path / "thinking/accumulation"
    tasks = [f"task_{i}" for i in range(50)]
    split = {"train": tasks, "val": ["val_only"]}
    base.write(source / "protocol.json", {"task_orders": {"42": split}})
    output = tmp_path / "advantage"
    directory = output / "thinking/without_gt"
    condition = {"id": "s42_without_gt_best_advantage", "seed": 42, "arm": "without_gt"}
    base.write(directory / "protocol.json", {"task_orders": {"42": {"train": tasks[:1], "val": ["val_only"]}}})
    context = "first\rsecond"
    checkpoint = directory / "conditions" / condition["id"] / "checkpoints"
    base.write(checkpoint / "001.json", {"playbook": context})
    (checkpoint / "001.txt").write_bytes(context.encode("utf-8"))
    base.write(checkpoint.parent / "training_completed.json", {})
    jobs = evaluator.build_queue(output, [{"path": str(directory), "condition": condition, "steps": 1}],
                                 "train_final", {"thinking": source})
    assert len(jobs) == 50
    for root, job in jobs:
        parent = base.read(Path(root) / "stage2" / job["context_id"] / "result.json")
        assert parent["updated_context_sha256"] == base.digest(context)
