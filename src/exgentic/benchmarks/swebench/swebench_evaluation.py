# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json
import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any

from ...utils.logging import capture_stdio_to_session
from ...utils.paths import SessionPaths


@dataclass
class HarnessResult:
    harness_report: dict[str, Any]
    patch: str
    patch_valid: bool
    harness_ran: bool
    error: str | None = None


def is_patch_valid(patch: str) -> bool:
    if not patch or not patch.strip():
        return False
    return any(m in patch for m in ["---", "+++", "@@", "diff --git", "*** Begin Patch"])


@contextmanager
def _pushd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@contextmanager
def _docker_ssh_transport():
    """Use separate SSH client connections for concurrent SWE harness jobs."""
    if not os.environ.get("DOCKER_HOST", "").startswith("ssh://"):
        yield
        return
    import docker

    original_from_env = docker.from_env

    def from_env(*args, **kwargs):
        kwargs.setdefault("use_ssh_client", True)
        return original_from_env(*args, **kwargs)

    docker.from_env = from_env
    try:
        yield
    finally:
        docker.from_env = original_from_env


def run_harness(
    patch: str,
    instance_id: str,
    subset: str,
    paths: SessionPaths,
    eval_config: dict[str, Any],
    logger: Logger,
) -> HarnessResult:
    patch = patch or ""
    valid = is_patch_valid(patch)

    pred_path = paths.benchmark_dir / "predictions.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "model_patch": patch,
                "patch_valid": valid,
                "model_name_or_path": "exgentic",
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    logger.info(f"EVAL | Writing patch | valid: {valid} | size: {len(patch)}")

    if not valid:
        logger.warning("EVAL | Skipping harness - invalid patch structure")
        return HarnessResult({}, patch, False, False)

    logger.info("EVAL | Running SWE-bench harness evaluation")
    from swebench.harness import run_evaluation

    base_run_id = "exgentic-" + hashlib.sha256(
        f"{paths.output_dir.resolve()}:{paths.run_id}:{paths.session_id}".encode()
    ).hexdigest()[:16]
    report: dict[str, Any] = {}
    error = "SWE-bench harness did not complete the requested instance"
    for attempt in range(1, 3):
        try:
            with _pushd(paths.benchmark_dir), capture_stdio_to_session(logger), _docker_ssh_transport():
                report_path = run_evaluation.main(
                    dataset_name=subset,
                    split="test",
                    instance_ids=[instance_id],
                    predictions_path=pred_path.name,
                    max_workers=eval_config["max_workers"],
                    run_id=f"{base_run_id}-try{attempt}",
                    namespace="swebench",
                    force_rebuild=False,
                    cache_level=eval_config["cache_level"],
                    clean=False,
                    open_file_limit=eval_config["open_file_limit"],
                    timeout=eval_config["harness_timeout"],
                    rewrite_reports=False,
                    modal=False,
                )
                report = json.loads(Path(report_path).read_text()) if report_path and Path(report_path).is_file() else {}
            if instance_id in report.get("completed_ids", []):
                logger.info("EVAL | Harness evaluation completed")
                return HarnessResult(report, patch, True, True)
            error = f"SWE-bench harness did not complete {instance_id}; error_ids={report.get('error_ids', [])}"
            logger.warning("EVAL | Harness attempt %d/2 incomplete: %s", attempt, error)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("EVAL | Harness attempt %d/2 failed: %s", attempt, error)
    return HarnessResult(report, patch, True, True, error)
