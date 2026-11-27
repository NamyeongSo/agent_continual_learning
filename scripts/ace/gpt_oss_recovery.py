# SPDX-License-Identifier: Apache-2.0
"""Retry failed AppWorld evaluation jobs through their original runner."""

import json
import subprocess
import sys
from pathlib import Path

from scripts.ace import schedule_qwen36_add_only as study

base = study.base
EVALUATION_PHASES = ("train_final", "bwt_checkpoint", "val")
RECOVERY_ATTEMPTS = 3
TRANSIENT_ERRORS = (
    "unexpected tokens remaining in message header",
    "litellm.timeout",
    "apitimeouterror",
    "apiconnectionerror",
    "connection reset by peer",
    "serviceunavailableerror",
)


def retryable(error):
    return any(message in error.lower() for message in TRANSIENT_ERRORS)


def phase_queue(output, phase):
    """Return the saved queue, with the same worker source as its controller."""
    if phase in {"high/reflection", "low/reflection"}:
        path = output / phase
        status = base.read(path / "status.json")
        if status["state"] != "error":
            return None
        queue = base.read(path / f"stage{status['stage']}_queue.json")
        return path, [(path, job) for job in queue], base.read(path / "protocol.json")["infrastructure_attempts"]
    if phase in {"high/accumulation", "low/accumulation"}:
        path = output / phase
        status = base.read(path / "status.json")
        if status["state"] != "error" or status["phase"] not in EVALUATION_PHASES:
            return None
        root = path / f"seed{status['seed']}"
        jobs = base.read(root / f"{status['phase']}_queue.json")
        return path, [(root, job) for job in jobs], 3
    if phase == "advantage_evaluation":
        path = output / "advantage/evaluations"
        status = path / "status.json"
        if not status.exists():
            return None
        phase = base.read(status).get("phase")
        queue = path / f"{phase}_queue.json"
        if phase in EVALUATION_PHASES and queue.exists():
            return None, [(Path(root), job) for root, job in base.read(queue)], 3
    return None


def last_attempt(directory):
    attempts = sorted(directory.glob("attempt_*"), key=lambda path: int(path.name.removeprefix("attempt_")))
    if not attempts or not (attempts[-1] / "error.json").is_file():
        return None
    latest = attempts[-1]
    return int(latest.name.removeprefix("attempt_")), base.read(latest / "error.json")["error"]


def run_worker(source, root, job, attempt):
    # Run in a fresh interpreter so the frozen runner installs its own import paths.
    worker = (
        "import json, runpy, sys; "
        "source, root, job, attempt = sys.argv[1:]; "
        "result = runpy.run_path(source)['worker'](root, json.loads(job), int(attempt)); "
        "sys.exit(not result['ok'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", worker, str(source), str(root), json.dumps(job), str(attempt)],
        cwd=study.ROOT, stdin=subprocess.DEVNULL,
    )
    return result.returncode == 0


def recover_evaluations(output, phase):
    selection = phase_queue(output, phase)
    if selection is None or (output / "STOP").exists():
        return False
    controller, queue, initial_attempts = selection
    if controller is not None:
        if (controller / "STOP").exists():
            return False
        study.verify_sources(controller)
        source = controller / "source_snapshot/scripts/ace/run_context_proposal_bottleneck.py"
    else:
        source = Path(base.__file__)

    recovered = False
    checked_endpoint = False
    for root, job in queue:
        directory = base.job_dir(root, job)
        result = directory / "result.json"
        if result.exists():
            continue
        last = last_attempt(directory)
        if last is None or last[0] < initial_attempts or not retryable(last[1]):
            continue
        for attempt in range(last[0] + 1, initial_attempts + RECOVERY_ATTEMPTS + 1):
            if not checked_endpoint:
                base.check_endpoint(base.read(root / "protocol.json"))
                checked_endpoint = True
            print(f"{base.now()} RECOVERY {phase} job={job['id']} attempt={attempt}", flush=True)
            if run_worker(source, root, job, attempt):
                if not result.is_file():
                    raise RuntimeError(f"Recovered job did not write a result: {job['id']}")
                recovered = True
                break
            latest = last_attempt(directory)
            if latest is None or latest[0] != attempt or not retryable(latest[1]):
                break
    return recovered


def recover_training(output, phase):
    if phase == "advantage_training":
        training = output / "advantage"
        markers = training.glob("*/*/conditions/*/training_error.json")
    elif phase in {"high/accumulation", "low/accumulation"}:
        training = output / phase
        status = base.read(training / "status.json")
        if status.get("phase") != "curator_training":
            return False
        markers = training.glob(f"conditions/s{status['seed']}_*/training_error.json")
    else:
        return False
    status_path = training / "status.json"
    if not status_path.exists() or base.read(status_path)["state"] != "error":
        return False
    changed = False
    counts_path = training / "curator_recovery_attempts.json"
    counts = base.read(counts_path) if counts_path.exists() else {}
    for marker in sorted(markers):
        if (marker.parents[2] / "STOP").exists() or (output / "STOP").exists():
            continue
        error = base.read(marker)
        step = error["step"]
        checkpoint = marker.parent / "checkpoints"
        if ((checkpoint / f"{step:03d}.json").exists()
                or not (checkpoint / f"{step - 1:03d}.json").exists()
                or not retryable(error["traceback"])):
            continue
        key = f"{marker.parent.relative_to(training)}/{step}"
        if counts.get(key, 0) >= RECOVERY_ATTEMPTS:
            continue
        counts[key] = counts.get(key, 0) + 1
        print(f"{base.now()} RECOVERY {phase} curator={key} attempt={counts[key]}", flush=True)
        changed = True
    if changed:
        base.write(counts_path, counts)
    return changed


def recover_phase(output, phase):
    return recover_evaluations(output, phase) or recover_training(output, phase)


def run_study(output, *, model, handoff_path, modes, previous_failure=None):
    if previous_failure:
        recover_phase(output, previous_failure["phase"])
    while True:
        try:
            study.run(output, model=model, handoff_path=handoff_path, modes=modes)
            return
        except (RuntimeError, FileNotFoundError):
            phase = base.read(output / "status.json")["phase"]
            if not recover_phase(output, phase):
                raise
