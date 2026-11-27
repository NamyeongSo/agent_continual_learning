# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Keep a long-running pairwise policy-shift queue alive and publish progress."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.policy_shift_metrics import digest, load_json, write_json  # noqa: E402

STOP_REQUESTED = False


def now():
    return datetime.now(UTC).astimezone(timezone(timedelta(hours=9))).isoformat()


def request_stop(_signal, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def counts_for(output: Path):
    db_path = output / "policy_shift.sqlite3"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30) as db:
        db.execute("BEGIN")
        return dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status").fetchall())


def journal_patch(plan, patch):
    if not plan.get("journal_id"):
        return
    from scripts.utils.schedule_checkpoint_policy_shift import api

    experiment = "/api/experiments/" + plan["journal_id"]
    current = api(experiment)
    api(experiment, {**patch, "revision": current["revision"]}, "PUT")


def append_event(output: Path, event: dict):
    path = output / "analysis/operational_events.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    events = load_json(path) if path.exists() else []
    events.append({"at": now(), **event})
    write_json(path, events)


def progress(output: Path, plan: dict, counts: dict, phase: str, pid: int | None = None):
    total = sum(value for key, value in counts.items() if key != "deduplicated")
    finished = counts.get("completed", 0) + counts.get("error", 0)
    return {
        "phase": phase,
        "at": now(),
        "supervisor_pid": os.getpid(),
        "runner_pid": pid,
        "workers": plan["workers"],
        "temperature": plan["temperature"],
        "counts": counts,
        "effective_requests": total,
        "processed_requests": finished,
        "progress": finished / total if total else 0,
        "pending_plus_running": counts.get("pending", 0) + counts.get("running", 0),
    }


def publish_periodic_progress(plan, state):
    total = state["effective_requests"]
    done = state["processed_requests"]
    counts = state["counts"]
    journal_patch(
        plan,
        {
            "status": "running",
            "summary": f"temperature0 pairwise ΔCP · global concurrency {plan['workers']}. "
            f"처리 {done:,}/{total:,} ({state['progress']:.1%}); "
            f"잔여 {state['pending_plus_running']:,}; 기록된 요청 오류 {counts.get('error', 0)}건.",
        },
    )


def import_final_artifacts(plan, output: Path):
    from scripts.utils.schedule_checkpoint_policy_shift import api

    artifacts = [
        (output / "analysis/summary.json", "ΔCP policy-shift 최종 집계"),
        (output / "analysis/distance_metrics.csv", "ΔCP별 benchmark/method/metric 점수"),
        (output / "analysis/pairwise_metrics.csv", "체크포인트 쌍별 비교"),
        (output / "analysis/pairwise_task_metrics.csv", "task별 체크포인트 쌍 비교"),
        (output / "analysis/coverage.json", "history 및 seed coverage"),
        (output / "analysis/comparison_summary.json", "ADD/AUD/GEPA/SkillOpt 비교 집계"),
        (output / "analysis/comparison_distance_metrics.csv", "비교 곡선의 ΔCP별 평균 shift"),
        (output / "analysis/correlations.csv", "각 곡선의 ΔCP 대 평균 shift Pearson r"),
        *[
            (path, "최종 ΔCP policy-shift 그래프")
            for path in sorted((output / "analysis").glob("*_delta_cp_policy_shift.png"))
        ],
        *[(path, "ACE/PEACE ADD-only ΔCP 그래프") for path in sorted((output / "analysis/add_only").glob("*.png"))],
    ]
    for path, caption in artifacts:
        if path.exists():
            api(
                "/api/experiments/" + plan["journal_id"] + "/import",
                {"path": str(path.resolve()), "caption": caption},
            )


def run(output: Path, journal_interval: int = 1800, max_restarts: int = 20):
    plan = load_json(output / "plan.json")
    if plan.get("temperature") != 0.0 or plan.get("workers") != 128:
        raise ValueError("Unexpected pairwise policy-shift protocol")
    for relative, sha in load_json(output / "code_hashes.json").items():
        path = output / "code_snapshot" / relative
        if digest(path.read_bytes()) != sha:
            raise ValueError(f"Frozen policy-shift runtime changed: {relative}")
    if not output.joinpath("analysis/deduplication.json").exists():
        raise ValueError("Deduplication audit is missing")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    with (output / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner_log_path = output / "runner.log"
        restarts = 0
        last_journal_update = 0.0
        last_status_write = 0.0

        try:
            journal_patch(
                plan,
                {
                    "status": "running",
                    "summary": "추론 worker를 detached supervisor가 감시 중. "
                    "종료 시 미완료 요청을 보존하고 재개합니다.",
                },
            )
        except Exception as exc:  # A journal outage must not stop inference.
            append_event(output, {"event": "journal_update_failed", "error": str(exc)})

        while True:
            if STOP_REQUESTED:
                write_json(output / "schedule_status.json", {"phase": "stopping", "at": now(), "restarts": restarts})
                return

            log = runner_log_path.open("a", buffering=1)
            child = subprocess.Popen(
                plan["command"],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write_json(
                output / "runner_process.json",
                {"pid": child.pid, "at": now(), "command": plan["command"], "supervisor_pid": os.getpid()},
            )
            append_event(output, {"event": "runner_started", "pid": child.pid, "restart_count": restarts})

            while child.poll() is None:
                if STOP_REQUESTED:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    write_json(
                        output / "schedule_status.json", {"phase": "stopped", "at": now(), "runner_pid": child.pid}
                    )
                    log.close()
                    return

                current = time.monotonic()
                if current - last_status_write >= 10:
                    counts = counts_for(output)
                    state = progress(output, plan, counts, "running_policy_shift", child.pid)
                    write_json(output / "schedule_status.json", state)
                    last_status_write = current
                    if current - last_journal_update >= journal_interval:
                        try:
                            publish_periodic_progress(plan, state)
                        except Exception as exc:  # Keep the queue going if the journal is temporarily unavailable.
                            append_event(output, {"event": "journal_update_failed", "error": str(exc)})
                        plot_command = plan.get("progress_plot_command")
                        if plot_command and counts.get("completed", 0) >= 1000:
                            with (output / "progress_plot.log").open("a", buffering=1) as plot_log:
                                try:
                                    result = subprocess.run(
                                        plot_command,
                                        cwd=ROOT,
                                        stdin=subprocess.DEVNULL,
                                        stdout=plot_log,
                                        stderr=subprocess.STDOUT,
                                        timeout=180,
                                    )
                                    if result.returncode:
                                        append_event(
                                            output, {"event": "progress_plot_failed", "exit_code": result.returncode}
                                        )
                                except Exception as exc:
                                    append_event(output, {"event": "progress_plot_failed", "error": str(exc)})
                        last_journal_update = current
                time.sleep(2)

            return_code = child.returncode
            log.close()
            summary_path = output / "analysis/summary.json"
            if return_code == 0 and summary_path.exists():
                summary = load_json(summary_path)
                phase = summary.get("status", "completed")
                final_state = progress(output, plan, counts_for(output), phase, child.pid)
                write_json(output / "schedule_status.json", final_state)
                try:
                    journal_patch(
                        plan,
                        {
                            "status": "completed",
                            "finished_on": now()[:10],
                            "summary": f"ΔCP 비교 완료: {summary.get('counts', {})}; "
                            f"common histories={summary.get('common_histories', 0)}.",
                        },
                    )
                    import_final_artifacts(plan, output)
                except Exception as exc:
                    append_event(output, {"event": "journal_final_import_failed", "error": str(exc)})
                append_event(output, {"event": "study_completed", "runner_exit_code": return_code})
                return

            restarts += 1
            append_event(
                output,
                {"event": "runner_exited_without_summary", "exit_code": return_code, "restart_count": restarts},
            )
            if restarts > max_restarts:
                state = progress(output, plan, counts_for(output), "restart_limit_reached", child.pid)
                write_json(output / "schedule_status.json", state)
                try:
                    journal_patch(
                        plan,
                        {
                            "status": "blocked",
                            "summary": "Worker restart limit reached; see runner.log and operational_events.json.",
                        },
                    )
                except Exception:
                    pass
                return
            write_json(
                output / "schedule_status.json",
                {
                    **progress(output, plan, counts_for(output), "restarting_worker"),
                    "exit_code": return_code,
                    "restart_count": restarts,
                },
            )
            time.sleep(min(60, 5 * restarts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal-interval", type=int, default=1800)
    parser.add_argument("--max-restarts", type=int, default=20)
    args = parser.parse_args()
    run(args.output.resolve(), args.journal_interval, args.max_restarts)


if __name__ == "__main__":
    main()
