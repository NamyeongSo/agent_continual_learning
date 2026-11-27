# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Queue 150 BFCL and 150 AppWorld plain ReAct tasks after Terminal-Bench drains."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def predecessor_finished(folder):
    """Require the final controller marker and a terminal record for every task."""
    folder = Path(folder)
    try:
        state = read(folder / "status.json")
        config = read(folder / "config.json")
        if state.get("phase") not in {"completed", "infrastructure_errors"}:
            return False
        expected = {task["task_id"] for task in config["tasks"]}
        if len(expected) != 89:
            return False
        records = {p.stem: read(p) for p in (folder / "records").glob("*.json")}
        return set(records) == expected and all(
            row.get("status") in {"completed", "infrastructure_error"} for row in records.values()
        )
    except (OSError, ValueError, KeyError):
        return False


def make_tasks(manifest, seed=42):
    from exgentic.utils.task_splits import select_task_split_sections, validate_task_split

    sections = select_task_split_sections(manifest, ["bfcl", "appworld"], seed)
    splits = {benchmark: validate_task_split(section, benchmark) for benchmark, section in sections.items()}
    for benchmark, split in splits.items():
        if any(len(split[phase]) != 50 for phase in ("train", "val", "test")):
            raise ValueError(f"{benchmark} must contain 50 train, 50 val and 50 test IDs")
    return [
        {"id": f"{benchmark}:{splits[benchmark][phase][index]}", "benchmark": benchmark,
         "task_id": splits[benchmark][phase][index], "split": phase, "index": index}
        for phase in ("test", "val", "train") for index in range(50) for benchmark in ("bfcl", "appworld")
    ]


def record_path(output, task):
    return Path(output) / "records" / task["benchmark"] / f"{task['task_id']}.json"


def load_records(config):
    return {task["id"]: read(record_path(config["output"], task)) for task in config["tasks"]
            if record_path(config["output"], task).exists()}


def summarize(config, records):
    reports = {}
    for benchmark in ("bfcl", "appworld"):
        reports[benchmark] = {}
        for split in ("test", "val", "train", "all"):
            tasks = [task for task in config["tasks"] if task["benchmark"] == benchmark
                     and (split == "all" or task["split"] == split)]
            measured = [records[task["id"]] for task in tasks
                        if records.get(task["id"], {}).get("status") == "completed"]
            successes = sum(row["success"] for row in measured)
            reports[benchmark][split] = {
                "planned": len(tasks), "measured": len(measured), "successes": successes,
                "score_percent": 100 * successes / len(measured) if measured else None,
                "complete": len(measured) == len(tasks),
                "unmeasured_task_ids": [task["task_id"] for task in tasks
                                        if records.get(task["id"], {}).get("status") != "completed"],
            }
    return {"at": now(), "benchmarks": reports, "planned": len(config["tasks"]),
            "measured": sum(reports[b]["all"]["measured"] for b in reports),
            "complete": all(reports[b]["all"]["complete"] for b in reports)}


def evaluate_task(payload):
    from exgentic.core.types import RunConfig
    from exgentic.interfaces.lib.api import execute
    from scripts.baselines.actor import official_protocol_failure, restore_protocol_outcome

    config, task = payload
    output = Path(config["output"])
    benchmark = task["benchmark"]
    steps = 30 if benchmark == "bfcl" else 40
    last_error = None
    for attempt in range(1, 4):
        error_path = output / "errors" / benchmark / f"{task['task_id']}_a{attempt}.json"
        if error_path.exists():
            last_error = read(error_path)
            continue
        namespace = hashlib.sha256(str(output).encode()).hexdigest()[:8]
        run_id = f"react_{benchmark}_{namespace}_{task['split']}_{task['index']:03d}_a{attempt}"
        kwargs = dict(config["benchmark_kwargs"][benchmark])
        if benchmark == "appworld":
            kwargs["env_kwargs"] = {**kwargs.get("env_kwargs", {}), "experiment_name": run_id}
        log = output / "logs" / f"{run_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = now()
        try:
            with log.open("a", buffering=1) as stream, contextlib.redirect_stdout(stream), \
                    contextlib.redirect_stderr(stream):
                result = execute(config=RunConfig(
                    benchmark=benchmark, agent="tool_calling", model=config["model"],
                    task_ids=[task["task_id"]], output_dir=str(output / "sessions"), run_id=run_id,
                    benchmark_kwargs=kwargs,
                    agent_kwargs={"runner": "direct", "model": config["model"], "max_steps": steps,
                                  "enable_tool_shortlisting": False, "allow_truncated_messages": False,
                                  "model_settings": config["model_settings"]},
                    max_steps=steps, max_actions=100, max_workers=1,
                ))
                if len(result.session_results) != 1:
                    raise RuntimeError("Expected exactly one task result")
                session = result.session_results[0]
                session_dir = output / "sessions" / run_id / "sessions" / session.session_id
                recovered = official_protocol_failure(session, session_dir, benchmark)
                if recovered:
                    session = restore_protocol_outcome(session, recovered)
                if ((session.status.value in {"error", "cancelled", "unknown"} and not recovered)
                        or session.score is None):
                    raise RuntimeError(f"No official outcome: {session.status}: {session.details}")
                record = {**task, "status": "completed", "success": bool(session.success),
                          "score": int(session.success), "partial_score": session.score,
                          "metric": "appworld_tgc_v1" if benchmark == "appworld" else "official_task_success",
                          "attempt": attempt, "started_at": started, "ended_at": now(),
                          "run_id": run_id, "session_id": session.session_id,
                          "session": session.model_dump(mode="json"), "log": str(log)}
                write(record_path(output, task), record)
                return record
        except Exception as exc:
            last_error = {**task, "status": "infrastructure_error", "attempt": attempt,
                          "started_at": started, "ended_at": now(), "error": str(exc),
                          "traceback": traceback.format_exc(), "run_id": run_id, "log": str(log)}
            write(error_path, last_error)
    write(record_path(output, task), last_error)
    return last_error


def update_journals(output, report, status):
    import urllib.request

    if not (output / "journal.json").exists():
        return
    journals = read(output / "journal.json")
    base = journals["base_url"]

    def get(path):
        with urllib.request.urlopen(base + path, timeout=10) as response:
            return json.load(response)

    for benchmark, identity in journals["experiments"].items():
        try:
            token = get("/api/info")["csrf_token"]
            current = get("/api/experiments/" + identity)
            payload = {"revision": current["revision"], "status": status,
                       "summary": json.dumps(report["benchmarks"][benchmark], ensure_ascii=False), "metrics": []}
            for split, row in report["benchmarks"][benchmark].items():
                if row["measured"]:
                    payload["metrics"].extend([
                        {"name": "TGC" if benchmark == "appworld" else "Task success",
                         "value": row["score_percent"], "unit": "%", "split": split,
                         "source": f"{output}/summary.json; {row['successes']}/{row['measured']} measured"},
                        {"name": "Measured tasks", "value": row["measured"], "unit": "tasks", "split": split,
                         "source": str(output / "summary.json")},
                    ])
            request = urllib.request.Request(
                base + "/api/experiments/" + identity, method="PUT", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", "X-Journal-Token": token},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                response.read()
        except Exception:
            write(output / f"journal_error_{benchmark}.json", {"at": now(), "error": traceback.format_exc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    config = read(output / "config.json")
    if args.prepare_only:
        assert len(config["tasks"]) == 300 and config["workers"] == 100
        print(json.dumps({"planned": 300, "workers": 100, "after_output": config["after_output"]}))
        return
    os.environ.update(OPENAI_API_BASE=config["base_url"], OPENAI_BASE_URL=config["base_url"], OPENAI_API_KEY="EMPTY",
                      EXGENTIC_ACTOR_OUTPUT_UNLIMITED="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    for name in ("EXGENTIC_INFERENCE_PROFILE", "EXGENTIC_INFERENCE_PROFILE_SHA256"):
        os.environ.pop(name, None)
    write(output / "controller_process.json", {"pid": os.getpid(), "started_at": now(), "command": sys.argv})
    while not predecessor_finished(config["after_output"]):
        write(output / "status.json", {"phase": "waiting_terminalbench", "at": now(),
                                       "after_output": config["after_output"], "planned": 300, "workers": 100})
        time.sleep(10)
    write(output / "dependency_finished.json", {"at": now(), "after_output": config["after_output"],
          "terminal_summary": read(Path(config["after_output"]) / "summary.json")})
    records = load_records(config)
    pending = [task for task in config["tasks"] if records.get(task["id"], {}).get("status") != "completed"]
    report = summarize(config, records)
    update_journals(output, report, "running")
    write(output / "status.json", {"phase": "evaluating", "at": now(), "pending": len(pending), "workers": 100})
    with ProcessPoolExecutor(max_workers=100, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(evaluate_task, (config, task)): task for task in pending}
        for future in as_completed(futures):
            row = future.result()
            records[row["id"]] = row
            report = summarize(config, records)
            write(output / "summary.json", report)
            print(json.dumps({"at": now(), "task": row["id"], "status": row["status"],
                              "success": row.get("success"), "measured": report["measured"]}), flush=True)
            if len(records) % 10 == 0:
                update_journals(output, report, "running")
    report = summarize(config, records)
    write(output / "summary.json", report)
    write(output / "status.json", {"phase": "completed" if report["complete"] else "infrastructure_errors",
                                   "at": now(), "measured": report["measured"], "planned": 300})
    update_journals(output, report, "completed" if report["complete"] else "blocked")


if __name__ == "__main__":
    main()
