# SPDX-License-Identifier: Apache-2.0
"""Reevaluate a frozen subset of missing ReAct outcomes after a Terminal-Bench queue."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.litellm.evaluate_terminalbench21 import evaluate_task, journal, now, summary, write  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def predecessor_finished(folder):
    folder = Path(folder)
    try:
        state, config = read(folder / "status.json"), read(folder / "config.json")
        tasks = config["tasks"]
        ids = {task["task_id"] for task in tasks}
        if state.get("phase") not in {"completed", "infrastructure_errors"} or len(ids) != len(tasks) or not ids:
            return False
        return all(read(folder / "records" / f"{task_id}.json").get("status")
                   in {"completed", "infrastructure_error"} for task_id in ids)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_retry(config):
    """Never select measured failures or silently change task membership on resume."""
    original = Path(config["retry_source"])
    source = read(original / "config.json")
    selected = {task["task_id"] for task in config["tasks"]}
    expected = {task["task_id"] for task in source["tasks"]
                if read(original / "records" / f"{task['task_id']}.json").get("status") == "infrastructure_error"}
    if selected != expected or len(selected) != 5 or len(config["tasks"]) != 5:
        raise ValueError("Retry queue must contain exactly the five original unmeasured tasks")
    if config["model_settings"].get("disable_request_timeout") is not True:
        raise ValueError("Retry requires explicit disabling of the provider request timeout")
    if config["workers"] != 10:
        raise ValueError("Terminal-Bench queue requires 10 worker slots")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    config = read(output / "config.json")
    validate_retry(config)
    if args.prepare_only:
        print(json.dumps({"tasks": len(config["tasks"]), "workers": config["workers"],
                          "after_output": config["after_output"]}))
        return
    os.environ.update(OPENAI_API_BASE=config["base_url"], OPENAI_BASE_URL=config["base_url"], OPENAI_API_KEY="EMPTY",
                      EXGENTIC_ACTOR_OUTPUT_UNLIMITED="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    for key in ("EXGENTIC_INFERENCE_PROFILE", "EXGENTIC_INFERENCE_PROFILE_SHA256"):
        os.environ.pop(key, None)
    write(output / "controller_process.json", {"pid": os.getpid(), "started_at": now(), "command": sys.argv})
    while not predecessor_finished(config["after_output"]):
        write(output / "status.json", {"phase": "waiting_predecessor", "at": now(), "planned": 5})
        time.sleep(10)
    records = {path.stem: read(path) for path in (output / "records").glob("*.json")}
    pending = [task for task in config["tasks"] if task["task_id"] not in records]
    write(output / "status.json", {"phase": "evaluating", "at": now(), "planned": 5, "workers": config["workers"]})
    journal(output, summary(config, records), "running")
    with ProcessPoolExecutor(max_workers=config["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(evaluate_task, (config, task)) for task in pending]
        for future in as_completed(futures):
            record = future.result()
            records[record["task_id"]] = record
            report = summary(config, records)
            write(output / "summary.json", report)
            journal(output, report, "running")
            print(json.dumps({"at": now(), "task": record["task_id"], "status": record["status"]}), flush=True)
    report = summary(config, records)
    write(output / "summary.json", report)
    write(output / "status.json", {"phase": "completed" if report["complete"] else "infrastructure_errors",
                                   "at": now(), "planned": 5})
    journal(output, report, "completed" if report["complete"] else "blocked")


if __name__ == "__main__":
    main()
