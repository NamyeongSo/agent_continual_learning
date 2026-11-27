# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Wait for complete BFCL alignment figures, then launch the CP5 policy-shift study."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.utils.evaluate_isolated_checkpoint_queue import read_json, timestamp, write_json  # noqa: E402


def dependency_state(dependency: dict) -> dict:
    directory = Path(dependency["output"])
    state = {"name": dependency["name"], "output": str(directory), "ready": False}
    try:
        pipeline = read_json(directory / "analysis_pipeline_status.json")
        progress = read_json(directory / "status.json")
        state.update(pipeline_phase=pipeline.get("phase"), phase=progress.get("phase"),
                     processed=progress.get("processed_tasks"), total=progress.get("total_evaluations"),
                     errors=progress.get("execution_errors"))
        # Some wrappers use subprocess.run(check=True) and only write phase=complete
        # after that analyzer returns; they do not record a separate analyzer exit code.
        if (pipeline.get("phase") != "complete" or pipeline.get("runner_exit_code") != 0
                or pipeline.get("analysis_exit_code", 0) != 0 or not pipeline.get("finished_at")):
            state["waiting_for"] = "Successful runner and final analysis completion"
            return state
        if (progress.get("phase") not in ("complete", "complete_with_errors")
                or progress.get("total_evaluations", 0) <= 0
                or progress.get("processed_tasks") != progress["total_evaluations"]
                or progress.get("active_tasks") != 0):
            state["waiting_for"] = "All evaluation tasks and active workers to finish"
            return state
        analysis = directory / dependency["analysis"]
        summary = read_json(analysis / "correlation_summary.json")
        if "partial_progress" in summary:
            state["waiting_for"] = "Final full-data summary replacing interim analysis"
            return state
        missing = [name for name in dependency["required_figures"] if not (analysis / name).is_file()
                   or (analysis / name).stat().st_size == 0]
        if missing:
            state["waiting_for"] = missing
            return state
        state.update(ready=True, finished_at=pipeline.get("finished_at"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        state["waiting_for"] = str(exc)
    return state


def verify_endpoint(model: dict) -> None:
    import httpx

    response = httpx.get(model["base_url"].rstrip("/") + "/models", timeout=10)
    response.raise_for_status()
    expected = model["model"].removeprefix("openai/")
    if expected not in {item["id"] for item in response.json()["data"]}:
        raise ValueError(f"Endpoint {model['base_url']} does not serve {expected}")


def process_fingerprint(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] not in ("Z", "X") else None
    except (OSError, IndexError):
        return None


def completed_model(output: Path) -> bool:
    path = output / "status.json"
    return bool(path.exists() and read_json(path).get("phase") in ("complete", "complete_with_errors")
                and (output / "analysis/summary.json").exists())


class ExistingController:
    """Observe a separately launched model controller without issuing duplicate work."""

    def __init__(self, output: Path, pid: int, start_ticks: str):
        self.output, self.pid, self.start_ticks = output, pid, start_ticks

    def poll(self) -> int | None:
        if process_fingerprint(self.pid) == self.start_ticks:
            return None
        return 0 if completed_model(self.output) else 1


def existing_controller(output: Path, key: str) -> ExistingController | None:
    path = output / "runtime_manifest.json"
    if not path.exists():
        return None
    runtime = read_json(path)
    pid, fingerprint = runtime["controller_pid"], runtime["controller_start_ticks"]
    if (runtime["model"]["key"] != key or runtime["output"] != str(output.resolve())
            or process_fingerprint(pid) != fingerprint):
        return None
    try:
        args = [arg.decode() for arg in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if arg]
    except (OSError, UnicodeDecodeError):
        return None
    if (str(ROOT / "scripts/premise/run_bfcl_policy_shift.py") not in args or "--model-key" not in args
            or args.index("--model-key") + 1 == len(args) or args[args.index("--model-key") + 1] != key):
        return None
    return ExistingController(output, pid, fingerprint)


def run(plan_path: Path, check_only: bool = False) -> None:
    plan = read_json(plan_path)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scheduler.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commands = {model["key"]: [sys.executable, "-u", str(ROOT / "scripts/premise/run_bfcl_policy_shift.py"),
                                    "--plan", str(plan_path), "--model-key", model["key"]]
                    for model in plan["models"]}
        while True:
            dependencies = [dependency_state(dependency) for dependency in plan["dependencies"]]
            ready = all(item["ready"] for item in dependencies)
            status = {"phase": "ready" if ready else "waiting_for_alignment_analysis", "scheduler_pid": os.getpid(),
                      "updated_at": timestamp(), "dependencies": dependencies, "commands": commands,
                      "checkpoint_interval": 5, "checkpoints": plan["checkpoints"],
                      "workers_per_model": plan["workers"]}
            status["independently_started_models"] = {
                model["key"]: controller.pid for model in plan["models"]
                if (controller := existing_controller(output / model["key"], model["key"])) is not None
            }
            write_json(output / "schedule_status.json", status)
            if check_only:
                print(json.dumps(status, indent=2), flush=True)
                return
            if ready:
                try:
                    for model in plan["models"]:
                        if completed_model(output / model["key"]) or existing_controller(
                            output / model["key"], model["key"]
                        ):
                            continue
                        verify_endpoint(model)
                    break
                except Exception as exc:
                    status.update(phase="waiting_for_model_endpoint", endpoint_error=str(exc))
                    write_json(output / "schedule_status.json", status)
            time.sleep(30)
        processes, logs, finished = {}, [], {}
        try:
            for model in plan["models"]:
                key = model["key"]
                prior_path = output / key / "status.json"
                prior = read_json(prior_path) if prior_path.exists() else {}
                if completed_model(output / key):
                    finished[key] = {"exit_code": 0, "phase": prior["phase"], "adopted_completed": True}
                    continue
                controller = existing_controller(output / key, key)
                if controller:
                    processes[key] = controller
                    continue
                stream = (output / f"{key}.log").open("a", buffering=1)
                logs.append(stream)
                processes[key] = subprocess.Popen(commands[key], cwd=ROOT, stdin=subprocess.DEVNULL,
                                                  stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while processes:
                status.update(phase="running_policy_shift", updated_at=timestamp(),
                              processes={key: process.pid for key, process in processes.items()}, finished=finished)
                write_json(output / "schedule_status.json", status)
                for key, process in list(processes.items()):
                    result = process.poll()
                    if result is not None:
                        model_path = output / key / "status.json"
                        current = read_json(model_path) if model_path.exists() else {}
                        finished[key] = {"exit_code": result, "phase": current.get("phase"),
                                         "error": current.get("error")}
                        del processes[key]
                if processes:
                    time.sleep(30)
            status.update(phase="complete" if all(item["exit_code"] == 0 for item in finished.values())
                          else "failed", updated_at=timestamp(), processes={}, finished=finished)
            write_json(output / "schedule_status.json", status)
        finally:
            for stream in logs:
                stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run(args.plan.resolve(), args.check_only)


if __name__ == "__main__":
    main()
