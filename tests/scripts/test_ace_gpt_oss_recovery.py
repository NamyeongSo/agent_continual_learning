# SPDX-License-Identifier: Apache-2.0
"""Exercise GPT-OSS recovery without changing AppWorld results or using its server."""

from scripts.ace import gpt_oss_recovery as recovery


def test_rescue_only_retries_the_unresolved_transient_evaluation(tmp_path, monkeypatch):
    controller = tmp_path / "high/accumulation"
    root = controller / "seed44"
    base = recovery.base
    jobs = [{"id": name, "stage": 3} for name in ("saved", "timeout", "bad_meta")]
    base.write(controller / "status.json", {"state": "error", "phase": "train_final", "seed": 44})
    base.write(root / "train_final_queue.json", jobs)
    base.write(root / "protocol.json", {"model": "test"})
    saved = base.job_dir(root, jobs[0]) / "result.json"
    base.write(saved, {"pct": 0.0, "success": False})
    saved_bytes = saved.read_bytes()
    transient = base.job_dir(root, jobs[1]) / "attempt_3/error.json"
    base.write(transient, {"error": "litellm.Timeout"})
    base.write(base.job_dir(root, jobs[2]) / "attempt_3/error.json", {"error": "Context/task pairing mismatch"})
    monkeypatch.setattr(recovery.study, "verify_sources", lambda path: None)
    monkeypatch.setattr(base, "check_endpoint", lambda settings: None)
    attempts = []

    def worker(source, directory, job, attempt):
        attempts.append((source, directory, job["id"], attempt))
        result = base.job_dir(directory, job)
        if attempt == 4:
            base.write(result / "attempt_4/error.json", {"error": "unexpected tokens remaining in message header"})
            return False
        base.write(result / "result.json", {"pct": 0.5, "success": False})
        return True

    monkeypatch.setattr(recovery, "run_worker", worker)
    assert recovery.recover_evaluations(tmp_path, "high/accumulation")
    source = controller / "source_snapshot/scripts/ace/run_context_proposal_bottleneck.py"
    assert attempts == [(source, root, "timeout", 4), (source, root, "timeout", 5)]
    assert saved.read_bytes() == saved_bytes
    assert transient.exists()
    assert not recovery.recover_evaluations(tmp_path, "high/accumulation")


def test_exhausted_curator_recovery_is_persisted_per_checkpoint(tmp_path):
    training = tmp_path / "high/accumulation"
    condition = training / "conditions/s42_with_gt_worst"
    base = recovery.base
    base.write(training / "status.json", {"state": "error", "phase": "curator_training", "seed": 42})
    base.write(condition / "checkpoints/014.json", {"step": 14})
    base.write(condition / "training_error.json", {"step": 15, "traceback": "litellm.Timeout"})

    for _ in range(recovery.RECOVERY_ATTEMPTS):
        assert recovery.recover_training(tmp_path, "high/accumulation")
    assert not recovery.recover_training(tmp_path, "high/accumulation")
    assert base.read(condition / "checkpoints/014.json") == {"step": 14}
