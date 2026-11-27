# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Run a frozen Qwen comparison after its predecessor finishes training and evaluation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from filelock import FileLock, Timeout

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils import run_qwen_isolated_when_free as runner  # noqa: E402

TERMINAL = {"completed", "training_error", "evaluation_error"}


def predecessor_finished(output):
    """Require the complete planned queue to be terminal, including final test work."""
    import psutil

    try:
        status = runner.read(output / "status.json")
        progress = runner.read(output / "progress.json")
        plan = runner.read(output / "launch_manifest.json")
    except FileNotFoundError:
        return False
    if status.get("phase") not in {"completed", "blocked"}:
        return False
    jobs = progress.get("jobs", [])
    if len(jobs) != len(plan["jobs"]) or {j["id"] for j in jobs} != {j["id"] for j in plan["jobs"]}:
        return False
    if progress.get("running_training") != 0 or progress.get("pending_evaluations") != 0:
        return False
    if any(job["status"] not in TERMINAL for job in jobs):
        return False
    for job in jobs:
        if job.get("pid"):
            try:
                process = psutil.Process(job["pid"])
                if process.status() != psutil.STATUS_ZOMBIE and process.cmdline() == job["command"]:
                    return False
            except psutil.NoSuchProcess:
                pass
        if job["status"] != "training_error":
            folder = Path(job["output"]) / "evaluation"
            if not (folder / "summary.json").exists():
                return False
            summary = runner.read(folder / "summary.json")
            if not summary.get("complete") or summary.get("tasks") != len(job["test_ids"]):
                return False
            if len(list((folder / "records").glob("*.json"))) != len(job["test_ids"]):
                return False
    return True


def wait_for_predecessor(predecessor, output, poll_seconds):
    while not predecessor_finished(predecessor):
        marker = predecessor / "status.json"
        runner.write(output / "status.json", {
            "phase": "waiting_predecessor", "at": runner.now(), "after_output": str(predecessor),
            "predecessor_phase": runner.read(marker).get("phase") if marker.exists() else "missing",
        })
        time.sleep(poll_seconds)


def acquire_gpu_lease(output, poll_seconds):
    path = Path.home() / ".cache/exgentic/qwen_gpu1_experiment.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    lease = FileLock(str(path), timeout=0)
    while True:
        try:
            lease.acquire()
            return lease
        except Timeout:
            runner.write(output / "status.json", {"phase": "waiting_gpu_lease", "at": runner.now()})
            time.sleep(poll_seconds)


def reuse_services(predecessor, output):
    server, bridge = runner.existing_services(predecessor)
    for name in ("server_process.json", "bridge_process.json"):
        runner.write(output / name, runner.read(predecessor / name))
    runner.write(output / "services_reused.json", {
        "at": runner.now(), "source_output": str(predecessor), "server_pid": server.pid, "bridge_pid": bridge.pid,
        "vllm_log": str(predecessor / "output_nohup_vllm_server_qwen_gpu.log"),
        "bridge_log": str(predecessor / "parser_bridge.log"),
    })
    return server, bridge


def run_followup(output, predecessor, poll_seconds=30):
    os.environ.update(runner.experiment_environment())
    if output == predecessor:
        raise ValueError("Follow-up must use a different output directory")
    plan = runner.read(output / "launch_manifest.json")
    if Path(plan["after_output"]).resolve() != predecessor:
        raise ValueError("Predecessor differs from the frozen launch manifest")
    with FileLock(str(output / "controller.lock"), timeout=0):
        runner.write(output / "controller_process.json", {"pid": os.getpid(), "started_at": runner.now(),
                                                         "after_output": str(predecessor)})
        try:
            wait_for_predecessor(predecessor, output, poll_seconds)
            lease = acquire_gpu_lease(output, poll_seconds)
            try:
                if not predecessor_finished(predecessor):
                    raise RuntimeError("Predecessor state changed before handoff")
                progress = runner.read(predecessor / "progress.json")
                failed = [j["id"] for j in progress["jobs"] if j["status"] != "completed"]
                runner.write(output / "predecessor_completion.json", {
                    "at": runner.now(), "output": str(predecessor), "failed_jobs": failed,
                    "status_sha256": runner.digest(predecessor / "status.json"),
                    "progress_sha256": runner.digest(predecessor / "progress.json"),
                })
                server, bridge = reuse_services(predecessor, output)
                runner.preflight_model(output)
                runner.journal(plan, output, "running",
                               f"선행 ADD-only 학습·최종 평가 종료 후 후속 실험 시작. "
                               f"편집 모드 {plan['playbook_edit_mode']}, 학습 {plan['train_workers']}개, "
                               f"선행 미해결 작업: {failed}. 기존 Qwen 서버 재사용.")
                runner.write(output / "status.json", {"phase": "running", "at": runner.now(),
                                                       "server_pid": server.pid, "bridge_pid": bridge.pid})
                jobs = runner.run_training_and_evaluation(plan, output, server, bridge)
                errors = [job["id"] for job in jobs if job["status"] != "completed"]
                phase = "blocked" if errors else "completed"
                runner.write(output / "status.json", {"phase": phase, "at": runner.now(), "failed_jobs": errors})
                runner.journal(plan, output, phase,
                               f"후속 실험 종료: {len(jobs) - len(errors)}/{len(jobs)} 완료. 미해결 작업: {errors}")
            finally:
                lease.release()
        except Exception as exc:
            runner.write(output / "status.json", {"phase": "blocked", "at": runner.now(),
                                                   "error": f"{type(exc).__name__}: {exc}"})
            runner.journal(plan, output, "blocked", f"후속 실험 실행기 중단: {type(exc).__name__}: {exc}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--after-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    run_followup(args.output_root.resolve(), args.after_output.resolve(), args.poll_seconds)


if __name__ == "__main__":
    main()
