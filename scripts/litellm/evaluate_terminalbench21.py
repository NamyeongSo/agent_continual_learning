# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Evaluate plain ReAct on all TB2.1 tasks, completing the current test split first."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def now():
    return datetime.now(UTC).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def summary(config, records):
    reports = {}
    for split in ("test", "val", "train", "all"):
        tasks = [row for row in config["tasks"] if split == "all" or row["split"] == split]
        measured = [records[row["task_id"]] for row in tasks
                    if records.get(row["task_id"], {}).get("status") == "completed"]
        successes = sum(row["success"] for row in measured)
        reports[split] = {
            "planned": len(tasks), "measured": len(measured), "successes": successes,
            "complete": len(measured) == len(tasks),
            "score": successes / len(measured) if measured else None,
            "score_percent": 100 * successes / len(measured) if measured else None,
            "unmeasured_task_ids": [row["task_id"] for row in tasks
                                    if records.get(row["task_id"], {}).get("status") != "completed"],
        }
    return {"at": now(), "dataset_version": "2.1", "dataset_commit": config["dataset_commit"],
            "model": config["model"], "agent": "tool_calling", "reports": reports,
            "complete": reports["all"]["complete"]}


def evaluate_task(payload):
    from exgentic.core.types import RunConfig
    from exgentic.interfaces.lib.api import execute

    config, task = payload
    output = Path(config["output"])
    last_error = None
    for attempt in range(1, config["infrastructure_attempts"] + 1):
        run_id = f"react_tb21_{task['split']}_{task['index']:02d}_a{attempt}"
        log = output / "logs" / f"{run_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = now()
        try:
            with log.open("a", buffering=1) as stream, contextlib.redirect_stdout(stream), \
                    contextlib.redirect_stderr(stream):
                result = execute(config=RunConfig(
                    benchmark="terminalbench2", agent="tool_calling", model=config["model"],
                    task_ids=[task["task_id"]], output_dir=str(output / "sessions"), run_id=run_id,
                    benchmark_kwargs=config["benchmark_kwargs"],
                    agent_kwargs={"runner": "direct", "model": config["model"],
                                  "max_steps": 40, "enable_tool_shortlisting": False,
                                  "allow_truncated_messages": False,
                                  "model_settings": config["model_settings"]},
                    max_steps=40, max_actions=100, max_workers=1, overwrite_sessions=False,
                ))
                if len(result.session_results) != 1:
                    raise RuntimeError("Expected exactly one official task outcome")
                session = result.session_results[0]
                if session.status.value in {"error", "cancelled", "unknown"} or session.score is None:
                    raise RuntimeError(f"No measured official outcome: {session.status}: {session.details}")
                record = {
                    **task, "status": "completed", "score": float(session.score),
                    "success": float(session.score) == 1.0, "attempt": attempt,
                    "started_at": started, "ended_at": now(), "run_id": run_id,
                    "session_id": session.session_id, "session": session.model_dump(mode="json"),
                    "log": str(log),
                }
                write(output / "records" / f"{task['task_id']}.json", record)
                return record
        except Exception as exc:
            last_error = {**task, "status": "infrastructure_error", "attempt": attempt,
                          "started_at": started, "ended_at": now(), "error": str(exc),
                          "traceback": traceback.format_exc(), "run_id": run_id, "log": str(log)}
            write(output / "errors" / f"{task['task_id']}_a{attempt}.json", last_error)
    write(output / "records" / f"{task['task_id']}.json", last_error)
    return last_error


def journal(output, report, phase):
    """Update the existing local record without touching the user's notebook."""
    import urllib.request

    path = output / "journal.json"
    if not path.exists():
        return
    experiment = json.loads(path.read_text())
    base, identity = experiment["base_url"], experiment["id"]

    def get(endpoint):
        with urllib.request.urlopen(base + endpoint, timeout=10) as response:
            return json.load(response)

    try:
        token = get("/api/info")["csrf_token"]
        current = get("/api/experiments/" + identity)
        metrics = []
        for split, row in report["reports"].items():
            if row["measured"]:
                metrics.extend([
                    {"name": "Task success", "value": row["score_percent"], "unit": "%", "split": split,
                     "source": f"{output}/summary.json; {row['successes']}/{row['measured']} measured; "
                               f"{row['planned']} planned"},
                    {"name": "Measured tasks", "value": row["measured"], "unit": "tasks", "split": split,
                     "source": str(output / "summary.json")},
                ])
        payload = {"revision": current["revision"], "status": phase, "metrics": metrics,
                   "summary": json.dumps(report["reports"], ensure_ascii=False),
                   "next_steps": "" if phase == "completed" else
                       f"예정 {report['reports']['all']['planned']}개 과제 평가와 오류 복구를 완료한다."}
        request = urllib.request.Request(
            base + "/api/experiments/" + identity, method="PUT", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "X-Journal-Token": token},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except Exception:
        write(output / "journal_update_error.json", {"at": now(), "traceback": traceback.format_exc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unlimited-output", action="store_true",
                        help="User-authorized removal of all reasoning and visible output token caps")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.unlimited_output:
        os.environ["EXGENTIC_ACTOR_OUTPUT_UNLIMITED"] = "1"
    from exgentic.benchmarks.terminalbench2.terminalbench2_benchmark import TERMINALBENCH21_COMMIT
    from exgentic.benchmarks.terminalbench2.terminalbench2_eval import list_task_ids, resolve_task_root
    from exgentic.core.experiment_limits import experiment_limits
    from exgentic.utils.task_splits import select_task_split_sections, validate_task_split

    output = args.output.resolve()
    manifest = json.loads(args.manifest.read_text())
    split = validate_task_split(select_task_split_sections(manifest, ["terminalbench2"], args.seed)["terminalbench2"],
                                "terminalbench2")
    tasks = [{"task_id": task_id, "split": phase, "index": index}
             for phase in ("test", "val", "train") for index, task_id in enumerate(split[phase])]
    available = list_task_ids(resolve_task_root(None, "2.1"))
    if len(tasks) != 89 or set(available) != {row["task_id"] for row in tasks}:
        raise ValueError("The frozen split must cover all 89 official 2.1 tasks exactly once")
    config = {
        "output": str(output), "model": "openai/Qwen/Qwen3.8-27B-FP8",
        "base_url": "http://127.0.0.1:8007/v1", "agent": "tool_calling", "learning": False,
        "dataset_version": "2.1", "dataset_commit": TERMINALBENCH21_COMMIT,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "seed": args.seed,
        "tasks": tasks, "phase_order": ["test", "val", "train"], "workers": args.workers,
        "infrastructure_attempts": 3, "experiment_limits": experiment_limits("terminalbench2"),
        "output_token_scope": "unlimited" if args.unlimited_output else "total_completion",
        "benchmark_kwargs": {"subset": "2.1", "actor": "interactive", "max_interactions": 40,
                             "enforce_session_timeouts": True, "use_cache": False},
        "model_settings": {
            "reasoning_effort": "xhigh", "temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0,
            "max_tokens": None if args.unlimited_output else 2048, "timeout": 600, "num_retries": 1,
            "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
                           "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}},
        },
    }
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output configuration differs; use a fresh directory")
    write(config_path, config)
    if args.prepare_only:
        print(json.dumps({"output": str(output), "test": len(split["test"]), "total": len(tasks)}))
        return
    os.environ.update(OPENAI_API_BASE=config["base_url"], OPENAI_BASE_URL=config["base_url"],
                      OPENAI_API_KEY="EMPTY", LITELLM_LOCAL_MODEL_COST_MAP="True")
    records = {p.stem: json.loads(p.read_text()) for p in (output / "records").glob("*.json")}
    write(output / "controller_process.json", {"pid": os.getpid(), "started_at": now(), "command": sys.argv})
    report = summary(config, records)
    journal(output, report, "running")
    for phase in config["phase_order"]:
        pending = [task for task in tasks if task["split"] == phase
                   and records.get(task["task_id"], {}).get("status") != "completed"]
        write(output / "status.json", {"phase": phase, "at": now(), "pending": len(pending)})
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(evaluate_task, (config, task)): task for task in pending}
            for future in as_completed(futures):
                record = future.result()
                records[record["task_id"]] = record
                report = summary(config, records)
                write(output / "summary.json", report)
                print(json.dumps({"at": now(), "task_id": record["task_id"], "split": phase,
                                  "status": record["status"], "success": record.get("success"),
                                  "test": report["reports"]["test"], "all": report["reports"]["all"]}), flush=True)
        report = summary(config, records)
        write(output / f"{phase}_summary.json", report["reports"][phase])
        journal(output, report, "running")
    report = summary(config, records)
    write(output / "summary.json", report)
    phase = "completed" if report["complete"] else "infrastructure_errors"
    write(output / "status.json", {"phase": phase, "at": now()})
    journal(output, report, "completed" if report["complete"] else "blocked")


if __name__ == "__main__":
    main()
