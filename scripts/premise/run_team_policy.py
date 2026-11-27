# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Fork ACE/PREMiSE BFCL checkpoints; relearn AppWorld and evaluate CP100 in one task queue."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/ace"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from checkpoint_recovery import completed_prefix  # noqa: E402
from restart_sequential import configure_workspace_source, fork_bfcl_checkpoint, training_command  # noqa: E402
from run_parallel_bfcl_appworld_terminalbench2 import prepare_sequential_bfcl_prefix  # noqa: E402

from exgentic.agents.premise.pipeline import playbook_text, write_json  # noqa: E402
from exgentic.utils.task_splits import select_task_split_sections, validate_task_split  # noqa: E402

BENCHMARKS = ("appworld", "bfcl", "terminalbench2")
SEEDS = (42, 43, 44)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output: Path, premise_source: Path, ace_source: Path, policy: Path, base_url: str) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    policy_copy = output / "appworld_team_policy.txt"
    shutil.copy2(policy, policy_copy)
    old_ace = json.loads((ace_source / "launch_manifest.json").read_text())
    ace_jobs = []
    jobs = []
    for seed in SEEDS:
        destination = output / "premise" / f"s{seed}_sequential_t3"
        config = fork_bfcl_checkpoint(premise_source / f"s{seed}_sequential_t3", destination, seed)
        command = training_command(destination, config, base_url, "team_policy")
        command += ["--appworld-policy", str(policy_copy), "--stop-after", "100"]
        jobs.append({
            "method": "premise", "mode": "sequential", "seed": seed,
            "output_dir": str(destination), "command": command,
            "task_order": config["order"], "manifest": config["manifest"], "model": config["model"],
            "benchmark_configs": config["benchmark_configs"], "max_steps": config["max_steps"],
            "checkpoint_050": str(destination / "checkpoints/checkpoint_050.json"),
            "checkpoint_050_sha256": digest(destination / "checkpoints/checkpoint_050.json"),
        })
        original = next(j for j in old_ace["jobs"] if j["mode"] == "sequential" and j["seed"] == seed)
        destination = output / "ace" / f"sequential_s{seed}"
        command = original["command"][:]
        command[command.index("--output-dir") + 1] = str(destination)
        command[command.index("--max-tokens") + 1] = "2048"
        if "--continue-from" in command:
            index = command.index("--continue-from")
            del command[index:index + 2]
        command += ["--appworld-policy", str(policy_copy), "--stop-after", "100"]
        ace_jobs.append({
            "method": "ace", "mode": "sequential", "seed": seed,
            "output_dir": str(destination), "command": command, "task_order": original["task_order"],
            "manifest": old_ace["source_manifest"], "model": old_ace["model"],
            "benchmark_configs": copy.deepcopy(config["benchmark_configs"]), "max_steps": config["max_steps"],
        })
    provenance = prepare_sequential_bfcl_prefix(
        ace_source, output / "ace", ace_jobs, old_ace["source_manifest_sha256"],
    )
    for job in ace_jobs:
        prefix = provenance[f"sequential_s{job['seed']}"]
        job["checkpoint_050"] = str(Path(prefix["continued_from"]) / "playbook_ace_sequential_global.json")
        job["checkpoint_050_sha256"] = digest(Path(job["checkpoint_050"]))
        job["provenance"] = prefix
    jobs.extend(ace_jobs)
    evaluation_plan = []
    for index, job in enumerate(jobs):
        job["id"] = f"{job['method']}_sequential_s{job['seed']}"
        job["port_block_start"] = 42000 + index * 1000
        job["log"] = str(output / "logs" / f"{job['id']}.log")
        sections = select_task_split_sections(
            json.loads(Path(job["manifest"]).read_text()), list(BENCHMARKS), job["seed"],
        )
        job["test_tasks"] = {b: validate_task_split(sections[b], b)["test"] for b in BENCHMARKS}
        if [b for b, _ in job["task_order"][:100]] != ["bfcl"] * 50 + ["appworld"] * 50:
            raise ValueError("Expected BFCL 50 followed by AppWorld 50")
        job["manifest_sha256"] = digest(Path(job["manifest"]))
        for bench in BENCHMARKS:
            evaluation_plan.append({"run": job["id"], "checkpoint": 100, "benchmark": bench,
                                    "task_ids": job["test_tasks"][bench]})
    manifest = {
        "created_at": datetime.now(UTC).isoformat(), "policy": str(policy_copy), "policy_sha256": digest(policy_copy),
        "base_url": base_url, "evaluation_workers": 50, "training_stop": 100, "resume_cursor": 50,
        "premise_source": str(premise_source.resolve()), "ace_source": str(ace_source.resolve()),
        "jobs": jobs, "evaluation_plan": evaluation_plan,
        "evaluation_tasks": sum(len(plan["task_ids"]) for plan in evaluation_plan),
        "queue": "One (method, seed, checkpoint, benchmark, datapoint) queue; enqueue immediately at CP100",
        "evaluation_process_isolation": "Each task has its own process, isolating ACE PlaybookStore",
        "acting_max_tokens": {"bfcl": 2048, "appworld": 2048}, "native_terminus_sampling": "unchanged",
        "ace_bfcl_prefix_provenance": provenance,
    }
    manifest["source_sha256"] = {str(path.relative_to(ROOT)): digest(path) for path in (
        Path(__file__).resolve(), ROOT / "scripts/ace/run_experiment.py",
        ROOT / "scripts/premise/run_experiment.py", ROOT / "src/exgentic/agents/premise/pipeline.py",
        ROOT / "src/exgentic/benchmarks/appworld/appworld_benchmark.py",
        ROOT / "src/exgentic/benchmarks/appworld/appworld_eval.py",
    )}
    write_json(output / "launch_manifest.json", manifest)
    return manifest


def checkpoint_at_100(job: dict) -> Path | None:
    directory = Path(job["output_dir"])
    if job["method"] == "premise":
        path = directory / "checkpoints/checkpoint_100.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if data["cursor"] != 100 or data["config"]["seed"] != job["seed"]:
            raise ValueError("Unexpected PREMiSE checkpoint")
        return path
    prefix = completed_prefix(directory, job["task_order"][:100])
    if len(prefix) != 100:
        return None
    item = next((record for record in reversed(prefix) if record["checkpoint"] is not None), None)
    if item is None or item["checkpoint"]["store_id"] != "ace_sequential_global":
        raise ValueError("Missing ACE checkpoint after AppWorld")
    path = directory / "checkpoints/checkpoint_100.json"
    payload = {**item["checkpoint"], "training_task_ordinal": 100, "source_checkpoint": item["checkpoint_path"],
               "source_checkpoint_sha256": digest(Path(item["checkpoint_path"]))}
    if not path.exists():
        write_json(path, payload)
    elif json.loads(path.read_text()) != payload:
        raise ValueError("Frozen ACE checkpoint changed")
    return path


def evaluate_task(spec: dict) -> dict:
    """Runs in a separate process, so frozen ACE playbooks cannot mix between seeds."""
    from exgentic.agents.ace.playbook_store import PlaybookStore
    from exgentic.core.types import RunConfig
    from exgentic.interfaces.lib.api import execute

    checkpoint = Path(spec["checkpoint"])
    if digest(checkpoint) != spec["checkpoint_sha256"] or digest(Path(spec["policy"])) != spec["policy_sha256"]:
        raise ValueError("Frozen experiment inputs changed")
    snapshot = json.loads(checkpoint.read_text())
    benchmark = spec["benchmark"]
    kwargs = copy.deepcopy(spec["benchmark_configs"][benchmark])
    kwargs["use_cache"] = False
    if benchmark == "appworld":
        kwargs["policy_override"] = Path(spec["policy"]).read_text()
    settings = {"timeout": 600, "num_retries": 1}
    if benchmark in ("bfcl", "appworld"):
        settings.update(temperature=0.0, max_tokens=2048)
    agent_kwargs = {"model": spec["model"], "runner": "direct", "enable_thinking": False,
                    "model_settings": settings, "enable_tool_shortlisting": benchmark == "appworld",
                    "max_selected_tools": 30}
    if spec["method"] == "premise":
        agent_kwargs["playbook"] = playbook_text(snapshot["states"]["global"]["beam"][0], benchmark)
    else:
        PlaybookStore.reset_all()
        agent_kwargs.update(curator_model=spec["model"], initial_playbook=snapshot["playbook"],
                            shuffle_mode="sequential", benchmark_id=benchmark,
                            evaluation_mode=True, training_time=False)
    run_id = f"team_policy_{spec['id']}_c100_{benchmark}_{spec['index']:03d}"
    config = RunConfig(
        benchmark=benchmark, agent=spec["method"], task_ids=[str(spec["task_id"])],
        output_dir=spec["evaluation_dir"], run_id=run_id, model=spec["model"],
        benchmark_kwargs=kwargs, agent_kwargs=agent_kwargs,
        max_steps=spec["max_steps"][benchmark], max_actions=max(100, spec["max_steps"][benchmark]),
        overwrite_sessions=False,
    )
    result = execute(config=config)
    if len(result.session_results) != 1:
        raise RuntimeError("Expected one evaluated task")
    session = result.session_results[0]
    session_dir = Path(spec["evaluation_dir"]) / run_id / "sessions" / session.session_id
    row = {"task_id": str(spec["task_id"]), "run_id": run_id, "session_dir": str(session_dir),
           "status": session.status.value, "score": session.score, "details": session.details}
    if benchmark == "appworld":
        scores = session_dir / "benchmark/scores.json"
        payload = json.loads(scores.read_text()) if scores.exists() else {}
        row["tgc"] = int(payload.get("success") is True)
    if spec["method"] == "ace":
        for store in PlaybookStore.list_stores().values():
            if store.playbook != snapshot["playbook"] or store.session_count != 0:
                raise RuntimeError("Evaluation changed frozen ACE playbook")
    return row


def controller(output: Path) -> int:
    manifest = json.loads((output / "launch_manifest.json").read_text())
    if digest(Path(manifest["policy"])) != manifest["policy_sha256"]:
        raise ValueError("Policy changed after preparing the experiment")
    lock = (output / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "status.json").exists():
        raise ValueError("Controller already started; refusing duplicate workers")
    configure_workspace_source()
    os.environ["OPENAI_API_BASE"] = manifest["base_url"]
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("DOCKER_HOST", "ssh://exgentic-daysi-docker:14233")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    children = {}
    child_lock = threading.Lock()
    training = {}
    futures = {}
    queued = set()
    statuses = {}
    for job in manifest["jobs"]:
        env = os.environ.copy()
        env["EXGENTIC_PORT_BLOCK_START"] = str(job["port_block_start"])
        log = Path(job["log"]).open("a")
        process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        log.close()
        training[job["id"]] = process
        statuses[job["id"]] = {"pid": process.pid, "phase": "training", "cursor": 50}
    write_json(output / "controller_launch.json", {"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()})

    def one(spec: dict) -> dict:
        result_path = Path(spec["result_path"])
        if result_path.exists():
            return json.loads(result_path.read_text())
        if stopped.is_set():
            return {"task_id": spec["task_id"], "status": "cancelled"}
        request = result_path.with_suffix(".job.json")
        write_json(request, spec)
        env = os.environ.copy()
        env.pop("EXGENTIC_PORT_BLOCK_START", None)
        with result_path.with_suffix(".log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", __file__, "--evaluation-job", str(request)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with child_lock:
                children[process.pid] = process
            process.wait()
            with child_lock:
                children.pop(process.pid, None)
        if result_path.exists():
            return json.loads(result_path.read_text())
        row = {"task_id": spec["task_id"], "status": "error", "score": 0, "tgc": 0,
               "error": f"Evaluation process exited {process.returncode}; inspect {result_path.with_suffix('.log')}"}
        write_json(result_path, row)
        return row

    with ThreadPoolExecutor(max_workers=manifest["evaluation_workers"]) as pool:
        while True:
            for job in manifest["jobs"]:
                identifier = job["id"]
                process = training[identifier]
                status = statuses[identifier]
                if job["method"] == "premise":
                    status["cursor"] = json.loads((Path(job["output_dir"]) / "state.json").read_text())["cursor"]
                else:
                    status["cursor"] = len(completed_prefix(Path(job["output_dir"]), job["task_order"][:100]))
                status["exit_code"] = process.poll()
                if identifier not in queued:
                    checkpoint = checkpoint_at_100(job)
                    if checkpoint is not None:
                        status["phase"] = "evaluation"
                        queued.add(identifier)
                        checkpoint_hash = digest(checkpoint)
                        task_count = max(len(job["test_tasks"][b]) for b in BENCHMARKS)
                        for index in range(task_count):
                            for benchmark in BENCHMARKS:
                                if index >= len(job["test_tasks"][benchmark]):
                                    continue
                                task_id = job["test_tasks"][benchmark][index]
                                directory = output / "evaluation" / identifier / "checkpoint_100" / benchmark
                                spec = {**job, "benchmark": benchmark, "index": index, "task_id": task_id,
                                        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                                        "policy": manifest["policy"], "policy_sha256": manifest["policy_sha256"],
                                        "evaluation_dir": str(directory / "sessions"),
                                        "result_path": str(directory / "tasks" / f"{index:03d}.json")}
                                future = pool.submit(one, spec)
                                futures[future] = (identifier, benchmark, index)
                        print(f"ENQUEUED {identifier} CP100 across all three benchmarks", flush=True)
                    elif process.poll() is not None:
                        status["phase"] = "training_failed"
                        status["log"] = job["log"]
            results = {}
            for future, (identifier, benchmark, _index) in futures.items():
                if future.done():
                    key = (identifier, benchmark)
                    results.setdefault(key, []).append(future.result())
            for (identifier, benchmark), rows in results.items():
                job = next(j for j in manifest["jobs"] if j["id"] == identifier)
                summary = {"run": identifier, "checkpoint": 100, "benchmark": benchmark, "sessions": rows,
                           "completed": len(rows), "expected": len(job["test_tasks"][benchmark]),
                           "error_count": sum(row["status"] in ("error", "cancelled") for row in rows),
                           "mean_task_score": sum(float(row.get("score") or 0) for row in rows) / len(rows),
                           "policy_sha256": manifest["policy_sha256"]}
                if benchmark == "appworld":
                    summary["tgc"] = sum(row.get("tgc", 0) for row in rows) / len(rows)
                write_json(output / "evaluation" / identifier / "checkpoint_100" / benchmark / "summary.json", summary)
            all_training_ended = all(p.poll() is not None for p in training.values())
            all_evaluation_ended = all(f.done() for f in futures)
            phase = "stopping" if stopped.is_set() else "running"
            if all_training_ended and all_evaluation_ended:
                phase = "complete" if len(queued) == len(training) else "failed"
            write_json(output / "status.json", {
                "phase": phase, "updated_at": datetime.now(UTC).isoformat(), "controller_pid": os.getpid(),
                "training": statuses, "evaluation_workers": 50, "evaluation_queued": len(futures),
                "evaluation_completed": sum(f.done() for f in futures),
                "evaluation_expected": manifest["evaluation_tasks"],
            })
            if stopped.is_set():
                with child_lock:
                    active = list(training.values()) + list(children.values())
                for process in active:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                break
            if all_training_ended and all_evaluation_ended:
                return 0 if phase == "complete" else 1
            stopped.wait(5)
    return 130


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--premise-source", type=Path,
                        default=ROOT / "outputs/premise_bfcl_appworld_terminalbench2_seed42_44_scorefix_20260928")
    parser.add_argument("--ace-source", type=Path,
                        default=ROOT / "scripts/ace/outputs/ace_appworld_scorefix_resume_bfcl50_20260928")
    parser.add_argument("--policy", type=Path,
                        default=ROOT / "scripts/premise/policies/appworld_teammate_completion.txt")
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    parser.add_argument("--evaluation-job", type=Path)
    args = parser.parse_args()
    if args.evaluation_job:
        spec = json.loads(args.evaluation_job.read_text())
        try:
            row = evaluate_task(spec)
        except Exception as exc:
            row = {"task_id": spec["task_id"], "status": "error", "score": 0, "tgc": 0,
                   "error_type": type(exc).__name__, "error": str(exc)}
        write_json(Path(spec["result_path"]), row)
        return 0
    if args.output_root is None:
        parser.error("--output-root is required")
    if not args.run_prepared:
        prepare(args.output_root, args.premise_source, args.ace_source, args.policy, args.base_url)
    if args.prepare_only:
        return 0
    return controller(args.output_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
