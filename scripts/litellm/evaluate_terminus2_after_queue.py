# SPDX-License-Identifier: Apache-2.0
"""Run frozen native Terminus-2 TB2.1 evaluation after the BFCL/AppWorld queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.litellm.evaluate_terminalbench21 import journal, now, summary, write  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def predecessor_finished(root):
    """A score can be missing, but every planned task must have finished its attempts."""
    root = Path(root)
    try:
        config, state = read(root / "config.json"), read(root / "status.json")
        if state.get("phase") not in {"completed", "infrastructure_errors"} or len(config["tasks"]) != 300:
            return False
        return all(read(root / "records" / task["benchmark"] / f"{task['task_id']}.json").get("status")
                   in {"completed", "infrastructure_error"} for task in config["tasks"])
    except (OSError, ValueError, KeyError):
        return False


def validate_tasks(config):
    tasks = config["tasks"]
    selected = {task["task_id"] for task in tasks}
    if len(selected) != len(tasks):
        raise ValueError("Duplicate task IDs in native evaluation")
    if config.get("retry_timeout_source"):
        source = Path(config["retry_timeout_source"])
        original = read(source / "config.json")
        expected = set()
        for task in original["tasks"]:
            record = read(source / "records" / f"{task['task_id']}.json")
            metadata = record.get("session", {}).get("session_metadata", {})
            if record.get("status") == "completed" and record.get("success") is False \
                    and metadata.get("agent_timed_out") is True:
                expected.add(task["task_id"])
        if selected != expected or not selected:
            raise ValueError("Retry selection must contain only and all original task-timeout failures")
    elif len(tasks) != 89:
        raise ValueError("Full native evaluation must cover all 89 tasks")


def worker(config, task):
    """Execute in Harbor's isolated environment, with its default prompt and parser."""
    from exgentic.benchmarks.terminalbench2.terminalbench2_benchmark import RunTerminus2Action, RunTerminus2Args
    from exgentic.benchmarks.terminalbench2.terminalbench2_eval import TerminalBench2Session, resolve_task_root
    from exgentic.core.context import Context, set_context

    output = Path(config["output"])
    task_root = resolve_task_root(config["benchmark_kwargs"].get("task_root"), "2.1")
    last_error = None
    for attempt in range(1, config["infrastructure_attempts"] + 1):
        error_path = output / "errors" / f"{task['task_id']}_a{attempt}.json"
        if error_path.exists():
            last_error = read(error_path)
            continue
        run_id = f"terminus2_tb21_{task['split']}_{task['index']:02d}_a{attempt}"
        started = now()
        session = None
        try:
            set_context(Context(run_id=run_id, output_dir=str(output / "sessions"), cache_dir=str(output / "cache")))
            session = TerminalBench2Session(
                task_dir=str(task_root / task["task_id"]), task_id=task["task_id"],
                subset="2.1", actor="terminus2",
                max_interactions=config["benchmark_kwargs"].get("max_interactions", 40),
                exclude_model_call_time=config["benchmark_kwargs"].get("exclude_model_call_time", False),
                native_terminus2=config["benchmark_kwargs"].get("native_terminus2", False),
                native_context_recovery=config["benchmark_kwargs"].get("native_context_recovery", False),
            )
            session.start()
            session.step(RunTerminus2Action(arguments=RunTerminus2Args(
                model=config["model"], model_settings=config["model_settings"],
                model_info=config.get("native_model_info"),
            )))
            score = session.score()
            row = {**task, "status": "completed", "success": score.score == 1.0, "score": score.score,
                   "attempt": attempt, "started_at": started, "ended_at": now(), "run_id": run_id,
                   "session_id": session.session_id, "session": score.model_dump(mode="json")}
            write(output / "records" / f"{task['task_id']}.json", row)
            return row
        except Exception as exc:
            last_error = {**task, "status": "infrastructure_error", "attempt": attempt,
                          "started_at": started, "ended_at": now(), "error": str(exc),
                          "traceback": traceback.format_exc(), "run_id": run_id}
            write(error_path, last_error)
        finally:
            if session is not None:
                session.close()
    write(output / "records" / f"{task['task_id']}.json", last_error)
    return last_error


def launch_worker(config, task):
    output = Path(config["output"])
    log = output / "logs" / f"{task['task_id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        completed = subprocess.run(
            [config["harbor_python"], "-u", str(Path(__file__).resolve()), "--output", str(output),
             "--task", task["task_id"]], stdout=stream, stderr=subprocess.STDOUT,
        )
    record = output / "records" / f"{task['task_id']}.json"
    if record.exists():
        return read(record)
    error = {**task, "status": "infrastructure_error", "error": f"Worker exited {completed.returncode}; see {log}",
             "ended_at": now()}
    write(record, error)
    return error


def adopted_worker_alive(identity):
    """Validate process start time and command before adopting an orphan worker."""
    try:
        process = Path("/proc") / str(identity["pid"])
        fields = (process / "stat").read_text().rsplit(") ", 1)[1].split()
        command = (process / "cmdline").read_bytes().decode().split("\0")
        return (fields[0] != "Z" and fields[19] == str(identity["start_ticks"])
                and command == identity["command"])
    except (OSError, ValueError, IndexError):
        return False


def wait_adopted_worker(config, task, identity):
    """Preserve the existing environment and trajectory across controller resizing."""
    record = Path(config["output"]) / "records" / f"{task['task_id']}.json"
    while True:
        # Workers commit the official outcome before container cleanup finishes.
        if record.exists():
            return read(record)
        if not adopted_worker_alive(identity):
            # A record may have been atomically committed between the first
            # read and process exit. Never overwrite a measured outcome.
            if record.exists():
                return read(record)
            error = {**task, "status": "infrastructure_error", "ended_at": now(),
                     "error": "Adopted worker exited without an official outcome", "adopted_pid": identity["pid"]}
            write(record, error)
            return error
        time.sleep(1)


def adoption_manifest(config):
    if not config.get("adopt_workers_file"):
        return {}
    identities = read(config["adopt_workers_file"])["workers"]
    selected = {task["task_id"] for task in config["tasks"]}
    adopted = {row["task_id"]: row for row in identities}
    if len(adopted) != len(identities) or not set(adopted) <= selected:
        raise ValueError("Adoption manifest has duplicate or unknown tasks")
    if len(adopted) > config["workers"]:
        raise ValueError("Adopted workers exceed requested concurrency")
    return adopted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    config = read(output / "config.json")
    validate_tasks(config)
    planned = len(config["tasks"])
    if args.prepare_only:
        print(json.dumps({"planned": planned, "workers": config["workers"], "after_output": config["after_output"]}))
        return
    os.environ.update(OPENAI_API_BASE=config["base_url"], OPENAI_BASE_URL=config["base_url"], OPENAI_API_KEY="EMPTY",
                      EXGENTIC_ACTOR_OUTPUT_UNLIMITED="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    for name in ("EXGENTIC_INFERENCE_PROFILE", "EXGENTIC_INFERENCE_PROFILE_SHA256"):
        os.environ.pop(name, None)
    if args.task:
        worker(config, next(task for task in config["tasks"] if task["task_id"] == args.task))
        return
    write(output / "controller_process.json", {"pid": os.getpid(), "started_at": now(), "command": sys.argv})
    while not predecessor_finished(config["after_output"]):
        write(output / "status.json", {"phase": "waiting_predecessor", "at": now(), "planned": planned})
        time.sleep(10)
    records = {p.stem: read(p) for p in (output / "records").glob("*.json")}
    # Never rerun a measured outcome or an already exhausted infrastructure failure on resume.
    pending = [task for task in config["tasks"] if task["task_id"] not in records]
    adopted = adoption_manifest(config)
    write(output / "status.json", {"phase": "evaluating", "at": now(), "workers": config["workers"]})
    journal(output, summary(config, records), "running")
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        # Submit adopted jobs first: they occupy slots while new work fills the
        # remaining slots, so resizing cannot duplicate or exceed concurrency.
        futures = [pool.submit(wait_adopted_worker, config, task, adopted[task["task_id"]])
                   for task in pending if task["task_id"] in adopted]
        futures.extend(pool.submit(launch_worker, config, task) for task in pending if task["task_id"] not in adopted)
        for future in as_completed(futures):
            row = future.result()
            records[row["task_id"]] = row
            report = {**summary(config, records), "agent": "terminus2"}
            write(output / "summary.json", report)
            print(json.dumps({"at": now(), "task": row["task_id"], "status": row["status"],
                              "measured": report["reports"]["all"]["measured"]}), flush=True)
            if len(records) % 5 == 0:
                journal(output, report, "running")
    report = {**summary(config, records), "agent": "terminus2"}
    write(output / "summary.json", report)
    write(output / "status.json", {
        "phase": "completed" if report["complete"] else "infrastructure_errors", "at": now(),
    })
    journal(output, report, "completed" if report["complete"] else "blocked")


if __name__ == "__main__":
    main()
