# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Train isolated AppWorld ACE contexts, then probe saved inputs at CP0/5/.../50."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.ace import run_appworld_policy_shift as study  # noqa: E402
from scripts.ace.checkpoint_recovery import completed_prefix  # noqa: E402
from scripts.utils.evaluate_isolated_checkpoint_queue import digest, read_json, timestamp, write_json  # noqa: E402

OSS_MODEL = "openai/openai/gpt-oss-20b"


def identity(pid: int) -> dict | None:
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "argv": [arg.decode() for arg in (root / "cmdline").read_bytes().split(b"\0") if arg]}
    except (OSError, IndexError, UnicodeDecodeError):
        return None


def training_complete(job: dict) -> bool:
    path = Path(job["run"]) / "training_complete.json"
    return path.is_file() and read_json(path).get("completed_task_count") == 50


def plan_study(output: Path, manifest: Path, base_url: str, workers: int, model: str,
               reasoning_effort: str | None, max_steps: int) -> dict:
    from exgentic.utils.task_splits import select_task_split_sections, validate_task_split
    from scripts.ace.run_experiment import task_order_from_train_manifest

    splits = read_json(manifest)
    jobs = []
    for seed in study.SEEDS:
        tasks = validate_task_split(select_task_split_sections(splits, ["appworld"], seed)["appworld"],
                                    "appworld")["train"]
        if len(tasks) != 50 or len(set(tasks)) != 50:
            raise ValueError("Expected fifty distinct training tasks per seed")
        order = task_order_from_train_manifest(manifest, ["appworld"], seed, None, "isolated")
        run = output / "training" / f"s{seed}"
        command = [sys.executable, "-u", str(ROOT / "scripts/ace/run_experiment.py"),
                   "--mode", "isolated", "--seed", str(seed), "--benchmarks", "appworld",
                   "--task-split-manifest", str(manifest), "--all-train-tasks", "--resume-sessions",
                   "--model", model, "--max-tokens", "8192", "--disable-thinking", "--training-time",
                   "--appworld-max-turns", str(max_steps), "--output-dir", str(run)]
        if reasoning_effort:
            command.extend(("--reasoning-effort", reasoning_effort))
        jobs.append({"seed": seed, "run": str(run), "task_order": [list(item) for item in order],
                     "command": command})
    return {"model": model, "base_url": base_url, "workers": workers, "reasoning_effort": reasoning_effort,
            "max_steps": max_steps, "max_tokens": 8192, "method": "ace", "mode": "isolated",
            "checkpoints": list(study.POINTS), "history_limit_per_task": study.HISTORY_LIMIT,
            "manifest": str(manifest), "manifest_sha256": digest(manifest.read_bytes()), "jobs": jobs,
            "learning": "Own-model AppWorld ACE, all three seeds", "evaluation_environment_execution": False}


class ExistingTraining:
    def __init__(self, saved: dict, job: dict):
        self.saved, self.job, self.pid = saved, job, saved["pid"]

    def poll(self):
        current = identity(self.pid)
        if current and current["start_ticks"] == self.saved["start_ticks"]:
            return None
        return 0 if training_complete(self.job) else 1


def find_training(output: Path, job: dict) -> ExistingTraining | None:
    path = output / "training_workers" / f"s{job['seed']}.json"
    if not path.exists():
        return None
    saved = read_json(path)["identity"]
    current = identity(saved["pid"])
    if current is None or current["start_ticks"] != saved["start_ticks"]:
        return None
    argv = current["argv"]
    if (str(ROOT / "scripts/ace/run_experiment.py") not in argv or "--output-dir" not in argv
            or argv[argv.index("--output-dir") + 1] != job["run"]):
        raise ValueError("Saved training process does not own the planned output")
    if argv != job["command"]:
        raise ValueError("Saved training process differs from the planned command")
    return ExistingTraining(current, job)


def validate_training(job: dict, plan: dict) -> None:
    path = Path(job["run"]) / "experiment_config.json"
    if not path.exists():
        return
    config = read_json(path)
    expected = {"model": plan["model"], "agent": "ace", "mode": "isolated", "seed": job["seed"],
                "benchmarks": ["appworld"], "task_order": job["task_order"], "disable_thinking": True,
                "training_time": True, "task_split_manifest_sha256": plan["manifest_sha256"], "max_tokens": 8192}
    if plan["reasoning_effort"]:
        expected["reasoning_effort"] = plan["reasoning_effort"]
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Own-model learning configuration changed: seed{job['seed']}")
    if config["benchmark_max_turns"]["appworld"] != plan["max_steps"]:
        raise ValueError("Learning step budget changed")


def validate_training_infrastructure(prefix: list[dict], seed: int) -> None:
    for index, item in enumerate(prefix):
        metadata = (item["result"].get("details") or {}).get("session_metadata") or {}
        error = str(metadata.get("error", ""))
        if any(text in error for text in ("Connection error", "Connection refused", "APIConnectionError",
                                         "Request timed out", "Venv service did not become healthy")):
            raise RuntimeError(f"Infrastructure failure at seed{seed} task{index + 1}; retry from the prior checkpoint")


def signal_identity(saved: dict, value: int) -> None:
    if identity(saved["pid"]) == saved:
        try:
            os.kill(saved["pid"], value)
        except ProcessLookupError:
            pass


def retire_training(output: Path, processes: dict, plan: dict) -> None:
    """Stop only fingerprinted experiment descendants when the learning supervisor fails."""
    owned = {}
    for job in plan["jobs"]:
        process = processes.get(job["seed"])
        if process is None:
            continue
        path = output / "training_workers" / f"s{job['seed']}.json"
        saved = read_json(path)["identity"]
        current = identity(process.pid)
        if current and current["start_ticks"] == saved["start_ticks"] and current["argv"] == job["command"]:
            owned[process.pid] = current
    children = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            parent = int((path / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children[int(path.name)] = parent
    while True:
        additions = {pid: current for pid, parent in children.items() if parent in owned and pid not in owned
                     if (current := identity(pid)) is not None}
        if not additions:
            break
        owned.update(additions)
    write_json(output / "infrastructure_stop.json", {"at": timestamp(), "owned_processes": list(owned.values())})
    for current in owned.values():
        signal_identity(current, signal.SIGSTOP)
    for current in reversed(list(owned.values())):
        signal_identity(current, signal.SIGTERM)
        signal_identity(current, signal.SIGCONT)
    time.sleep(0.5)
    for current in owned.values():
        signal_identity(current, signal.SIGKILL)


def finish_training(output: Path, plan: dict) -> None:
    processes, streams = {}, []
    finished = False
    environment = {**os.environ, "OPENAI_API_BASE": plan["base_url"],
                   "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "EMPTY"),
                   "LITELLM_LOCAL_MODEL_COST_MAP": "True", "EXGENTIC_LLM_TRACE_FORMAT": "dedup_v1",
                   "PYTHONUNBUFFERED": "1"}
    environment.pop("EXGENTIC_PORT_BLOCK_START", None)
    try:
        for job in plan["jobs"]:
            validate_training(job, plan)
            if training_complete(job):
                continue
            process = find_training(output, job)
            if process is None:
                stream = (output / f"training_s{job['seed']}.log").open("ab", buffering=0)
                streams.append(stream)
                process = subprocess.Popen(job["command"], cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                saved = identity(process.pid)
                if saved is None:
                    raise RuntimeError("Training runner exited before its identity could be saved")
                write_json(output / "training_workers" / f"s{job['seed']}.json",
                           {"identity": saved, "run": job["run"], "launched_at": timestamp()})
            processes[job["seed"]] = process
        while True:
            progress = {}
            for job in plan["jobs"]:
                validate_training(job, plan)
                prefix = completed_prefix(Path(job["run"]), job["task_order"])
                validate_training_infrastructure(prefix, job["seed"])
                progress[str(job["seed"])] = len(prefix)
            write_json(output / "status.json", {"phase": "own_model_training", "updated_at": timestamp(),
                       "controller_pid": os.getpid(), "model": plan["model"], "base_url": plan["base_url"],
                       "seed_cursors": progress, "target_cursor": 50, "new_training_tasks": 150,
                       "new_training_attempted": sum(progress.values()), "max_steps": plan["max_steps"],
                       "training_parallel_seeds": len(processes), "probe_concurrency": plan["workers"],
                       "training_processes": {str(s): p.pid for s, p in processes.items()},
                       "evaluation_environment_execution": False})
            for seed, process in list(processes.items()):
                result = process.poll()
                if result is not None:
                    job = next(j for j in plan["jobs"] if j["seed"] == seed)
                    if result != 0 or not training_complete(job):
                        raise RuntimeError(f"AppWorld ACE training failed: seed{seed}, exit{result}")
                    del processes[seed]
            if not processes:
                finished = True
                break
            time.sleep(10)
    finally:
        if not finished:
            retire_training(output, processes, plan)
        for stream in streams:
            stream.close()


def export_bundle(output: Path, plan: dict) -> Path:
    """Freeze each checkpoint from the training prefix, carrying state across skipped updates."""
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK

    bundle = output / "checkpoint_bundle"
    manifest_path = Path(plan["manifest"])
    if digest(manifest_path.read_bytes()) != plan["manifest_sha256"]:
        raise ValueError("Frozen task split changed")
    study.save_immutable(bundle / "task_split_manifest.json", read_json(manifest_path))
    entries, runs = [], []
    for job in plan["jobs"]:
        validate_training(job, plan)
        if not training_complete(job):
            raise ValueError("Cannot export checkpoints from incomplete learning")
        source = Path(job["run"])
        prefix = completed_prefix(source, job["task_order"])
        if len(prefix) != 50:
            raise ValueError("Incomplete recorded training prefix")
        validate_training_infrastructure(prefix, job["seed"])
        study.save_immutable(bundle / "source_configs" / f"ace_s{job['seed']}.json",
                             read_json(source / "experiment_config.json"))
        initial = bundle / "initial" / f"s{job['seed']}.json"
        study.save_immutable(initial, {"store_id": "ace_isolated_appworld", "playbook": DEFAULT_PLAYBOOK,
                                     "next_global_id": 1, "session_count": 0, "history_len": 0,
                                     "benchmark_counts": {}})
        for point in study.POINTS[1:]:
            state = next((item for item in reversed(prefix[:point]) if item["checkpoint_path"]), None)
            original = Path(state["checkpoint_path"]) if state else initial
            if point == 50:
                original = source / "playbook_ace_isolated_appworld.json"
            snapshot = read_json(original)
            if snapshot["store_id"] != "ace_isolated_appworld":
                raise ValueError("Wrong isolated AppWorld playbook store")
            path = bundle / "ace" / f"s{job['seed']}" / f"checkpoint_{point:03d}.json"
            study.save_immutable(path, snapshot)
            entries.append({"method": "ace", "seed": job["seed"], "checkpoint": point,
                            "path": str(path.relative_to(bundle)), "sha256": digest(path.read_bytes()),
                            "source": str(original), "source_sha256": digest(original.read_bytes()),
                            "training_task_id": job["task_order"][point - 1][1],
                            "checkpoint_unit": "Attempted training tasks; skipped updates retain prior state"})
        runs.append({"method": "ace", "seed": job["seed"], "source_run": str(source), "study_training_tasks": 50})
    study.save_immutable(bundle / "manifest.json", {"benchmark": "appworld", "mode": "isolated",
                         "manifest_sha256": plan["manifest_sha256"], "checkpoints": entries, "runs": runs,
                         "learning_model": plan["model"], "max_steps": plan["max_steps"]})
    return bundle


def save_runtime_plan(output: Path, plan: dict) -> dict:
    path = output / "launch_manifest.json"
    if path.exists():
        original = read_json(path)
        if {k: v for k, v in original.items() if k != "base_url"} != {k: v for k, v in plan.items() if k != "base_url"}:
            raise ValueError("Frozen AppWorld experiment changed beyond endpoint routing")
    else:
        study.save_immutable(path, plan)
        original = plan
    write_json(output / "runtime_route.json", {"base_url": plan["base_url"],
               "declared_base_url": original["base_url"], "model": plan["model"], "updated_at": timestamp()})
    return original


def run(output: Path, plan: dict, inference_upstream: str | None = None) -> None:
    import httpx

    output.mkdir(parents=True, exist_ok=True)
    with (output / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = save_runtime_plan(output, plan)
        if digest(Path(plan["manifest"]).read_bytes()) != plan["manifest_sha256"]:
            raise ValueError("Frozen task split changed")
        response = httpx.get(plan["base_url"].rstrip("/") + "/models", timeout=15)
        response.raise_for_status()
        if plan["model"].removeprefix("openai/") not in {r["id"] for r in response.json()["data"]}:
            raise ValueError("Endpoint does not serve the planned learning model")
        write_json(output / "pipeline_runtime_manifest.json", {"controller": identity(os.getpid()),
                   "started_at": timestamp(), "base_url": plan["base_url"], "workers": plan["workers"],
                   "training_max_steps": plan["max_steps"], "declared_base_url": original["base_url"],
                   "inference_upstream": inference_upstream or plan["base_url"],
                   "evaluation_environment_execution": False})
        try:
            finish_training(output, plan)
            write_json(output / "status.json", {"phase": "exporting_checkpoints", "updated_at": timestamp(),
                       "controller_pid": os.getpid()})
            bundle = export_bundle(output, plan)
            study.run(output, bundle, plan["base_url"], plan["workers"], model_name=plan["model"],
                      reasoning_effort=plan["reasoning_effort"])
        except Exception as exc:
            write_json(output / "status.json", {"phase": "error", "error": str(exc),
                       "updated_at": timestamp(), "controller_pid": os.getpid()})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "scripts/utils/task_splits"
                        / "bfcl_appworld_terminalbench2_seed42_44.json")
    parser.add_argument("--model", default=OSS_MODEL)
    parser.add_argument("--base-url", default="http://127.0.0.1:18010/v1")
    parser.add_argument("--inference-upstream", help="Record the model server behind an optional parsing bridge")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--workers", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()
    if args.workers < 1 or args.max_steps < 1:
        parser.error("Concurrency and step budget must be positive")
    output = args.output.resolve()
    plan = plan_study(output, args.manifest.resolve(), args.base_url, args.workers,
                      args.model, args.reasoning_effort, args.max_steps)
    run(output, plan, args.inference_upstream)


if __name__ == "__main__":
    main()
