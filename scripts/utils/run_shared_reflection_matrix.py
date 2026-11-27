# SPDX-License-Identifier: Apache-2.0
"""Fresh four-method matrix with a rolling queue and staggered optimizer seeds."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("bfcl", "appworld", "terminalbench2")
METHODS = ("ace", "premise", "gepa", "skillopt")
VARIANTS = {"ace": "ace_aud", "premise": "peace_no_alignment_aud", "gepa": "gepa_val", "skillopt": "skillopt_val"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(UTC).isoformat()


def release_delay(method, seed):
    return (seed - 42) * 900 if method in {"gepa", "skillopt"} else 0


def complete(job, plan):
    path = Path(job["output"]) / "job_complete.json"
    if not path.exists():
        return False
    result = read(path)
    if (
        result.get("complete") is not True
        or result.get("job_id") != job["id"]
        or result.get("inference_profile_sha256") != plan["inference_profile_sha256"]
        or result.get("test_ids") != job["test_ids"]
        or result.get("tasks") != len(job["test_ids"])
        or result.get("infrastructure_errors") != 0
    ):
        raise ValueError("Existing completion evidence differs from the matrix protocol")
    return True


def environment(plan, job=None):
    private = read(Path(plan["output"]) / "private_environment.json")
    snapshot = Path(plan["snapshot"])
    env = {
        **os.environ,
        **private,
        "OPENAI_API_BASE": plan["base_url"],
        "OPENAI_BASE_URL": plan["base_url"],
        "OPENAI_API_KEY": "EMPTY",
        "PYTHONUNBUFFERED": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "EXGENTIC_INFERENCE_PROFILE": plan["inference_profile"],
        "EXGENTIC_INFERENCE_PROFILE_SHA256": plan["inference_profile_sha256"],
        "PYTHONPATH": str(snapshot / "src") + os.pathsep + str(snapshot),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    env["PATH"] = str(snapshot / ".venv/bin") + os.pathsep + env.get("PATH", "")
    if job:
        env["EXGENTIC_PORT_BLOCK_START"] = str(job["port_block_start"])
    return env


def prepare(output, profile_path, validation, calibration):
    if (output / "launch_manifest.json").exists():
        raise ValueError("Output already contains a matrix; resume its controller explicitly")
    profile = read(profile_path)
    if profile.get("reflection_template_sha256") != sha(ROOT / "src/exgentic/reflection.py"):
        raise ValueError("Validated reflection template differs from the code being frozen")
    selected = read(validation / "summary.json")
    if profile["reflection_variant"] != selected["selected_on_development"]:
        raise ValueError("Selected reflection prompt differs from the validation result")
    if not read(calibration / "pilot_summary.json")["all_completed"]:
        raise ValueError("Real inference/training calibration has not passed")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=False)
    for name in ("src", "scripts"):
        shutil.copytree(
            ROOT / name,
            snapshot / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "outputs", ".pytest_cache", "installation"),
        )
    for installation in (ROOT / "src/exgentic/benchmarks").glob("*/installation"):
        (snapshot / installation.relative_to(ROOT)).symlink_to(installation.resolve(), target_is_directory=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, snapshot / name)
    for name in (".venv", ".exgentic"):
        (snapshot / name).symlink_to((ROOT / name).resolve(), target_is_directory=True)
    manifest = output / "task_splits.json"
    shutil.copy2(
        ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_fixed_balanced_order_seeds42_44_v1.json",
        manifest,
    )
    shutil.copy2(profile_path, output / "inference_profile.json")
    previous = read(ROOT / "outputs/qwen_epoch2_continuation_four_baselines_20261109/launch_manifest.json")
    source = next(j["appworld_policy"] for j in previous["jobs"] if j.get("appworld_policy"))
    policy = output / "appworld_original_policy.txt"
    shutil.copy2(source, policy)
    private = {
        key: value
        for key, value in previous.get("environment", {}).items()
        if key in {"DOCKER_HOST", "DOCKER_SSH_COMMAND", "SSH_AUTH_SOCK", "ACE_DOCKER_HOST"}
    }
    write(output / "private_environment.json", private)
    os.chmod(output / "private_environment.json", 0o600)
    splits = read(manifest)
    jobs = []
    python = str(snapshot / ".venv/bin/python")
    for benchmark in BENCHMARKS:
        for method in METHODS:
            for seed in (42, 43, 44):
                folder = output / "runs" / VARIANTS[method] / benchmark / f"s{seed}"
                train = folder / "train"
                section = splits[f"{benchmark}_{seed}"]
                common = [
                    "--model",
                    "openai/Qwen/Qwen3.6-35B-A3B",
                    "--seed",
                    str(seed),
                    "--actor-context",
                    "shared-empty-v1",
                    "--reasoning-effort",
                    "medium",
                ]
                if method in {"ace", "premise"}:
                    command = [
                        python,
                        "-u",
                        str(snapshot / f"scripts/{method}/run_experiment.py"),
                        "--mode",
                        "isolated",
                        "--benchmarks",
                        benchmark,
                        "--all-train-tasks",
                        "--task-split-manifest",
                        str(manifest),
                        "--output-dir",
                        str(train),
                        "--playbook-edit-mode",
                        "add_update_delete",
                        "--actor-temperature",
                        "0.0",
                    ]
                    if method == "ace":
                        command += ["--training-time", "--analysis-temperature", "0.7", "--max-tokens", "8192"]
                    else:
                        command += [
                            "--base-url",
                            "http://127.0.0.1:18007/v1",
                            "--disable-alignment",
                            "--past",
                            "3",
                            "--reflection-temperature",
                            "0.7",
                            "--mutation-temperature",
                            "0.7",
                            "--llm-timeout",
                            "1800",
                            "--replay-workers",
                            "28",
                            "--llm-calls",
                            "32",
                        ]
                else:
                    command = [
                        python,
                        "-u",
                        str(snapshot / "scripts/baselines/run_experiment.py"),
                        "--method",
                        method,
                        "--benchmark",
                        benchmark,
                        "--manifest",
                        str(manifest),
                        "--output",
                        str(train),
                        "--workers",
                        "50",
                        "--optimizer-seed",
                        "42",
                        "--actor-temperature",
                        "0.0",
                        "--analysis-temperature",
                        "0.7",
                        "--base-url",
                        "http://127.0.0.1:18007/v1",
                    ]
                command += common
                if benchmark == "appworld":
                    command += ["--appworld-policy", str(policy)]
                job = {
                    "id": f"{VARIANTS[method]}_{benchmark}_s{seed}",
                    "method": method,
                    "variant": VARIANTS[method],
                    "benchmark": benchmark,
                    "seed": seed,
                    "data_order_seed": seed,
                    "optimizer_seed": 42 if method in {"gepa", "skillopt"} else seed,
                    "output": str(folder),
                    "train_dir": str(train),
                    "train_count": len(section["train"]),
                    "train_ids": section["train"],
                    "val_ids": section["val"],
                    "test_ids": section["test"],
                    "train_command": command,
                    "command": command,
                    "status": "pending",
                    "release_delay_seconds": release_delay(method, seed),
                    "epochs": 1,
                    "test_workers": 50,
                    "port_block_start": 42000 + 1000 * (len(jobs) % 12),
                    "playbook_edit_mode": "add_update_delete" if method in {"ace", "premise"} else None,
                    "actor_context": "shared-empty-v1",
                    "candidate_evaluation": "environment_replay",
                }
                jobs.append(job)
    plan = {
        "version": "shared-reflection-matrix-v1",
        "created_at": now(),
        "output": str(output),
        "snapshot": str(snapshot),
        "model": "Qwen/Qwen3.6-35B-A3B",
        "base_url": "http://127.0.0.1:18007/v1",
        "epochs": 1,
        "max_parallel": 12,
        "benchmark_order": list(BENCHMARKS),
        "benchmark_barrier": False,
        "scheduling": "rolling; start the next eligible job as soon as a slot becomes free",
        "seed_stagger_anchor": "each optimizer's seed42 actual start within its benchmark",
        "port_blocks": list(range(42000, 54000, 1000)),
        "seed_stagger_seconds": 900,
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
        "inference_profile": str(output / "inference_profile.json"),
        "inference_profile_sha256": sha(output / "inference_profile.json"),
        "profile": profile,
        "appworld_policy": str(policy),
        "appworld_policy_sha256": sha(policy),
        "prompt_validation": str(validation),
        "prompt_validation_sha256": sha(validation / "summary.json"),
        "token_calibration": str(calibration),
        "jobs": jobs,
    }
    write(output / "launch_manifest.json", plan)
    files = {
        str(path.relative_to(snapshot)): sha(path)
        for path in snapshot.rglob("*")
        if path.is_file()
        and not any(part in {".venv", ".exgentic", "__pycache__", "installation"} for part in path.parts)
    }
    write(output / "source_sha256.json", files)
    return plan


def evaluate_task(payload):
    from exgentic.actor_context import shared_actor_prompt
    from exgentic.core.types import RunConfig
    from exgentic.interfaces.lib.api import execute

    job, config, playbook, task_id, index, attempt = payload
    benchmark = job["benchmark"]
    kwargs = dict(config["benchmark_configs"][benchmark])
    kwargs["use_cache"] = False
    run_id = f"final_{job['id']}_{index}_{attempt}_{uuid.uuid4().hex[:8]}"
    if benchmark == "appworld":
        kwargs["env_kwargs"] = {**kwargs.get("env_kwargs", {}), "experiment_name": run_id}
    agent = {
        "runner": "direct",
        "enable_thinking": False,
        "actor_context": "shared-empty-v1",
        "model_settings": {
            "temperature": 0.0,
            "max_tokens": 2048,
            "timeout": 1800,
            "reasoning_effort": "medium",
            "num_retries": 1,
        },
        "temperature_override": 0.0,
        "enable_tool_shortlisting": False,
    }
    if job["method"] == "ace":
        agent.update(
            initial_playbook=playbook,
            evaluation_mode=True,
            training_time=False,
            shuffle_mode="isolated",
            benchmark_id=benchmark,
            model="openai/Qwen/Qwen3.6-35B-A3B",
        )
    else:
        agent.update(playbook=playbook, actor_benchmark=benchmark, model="openai/Qwen/Qwen3.6-35B-A3B")
    result = execute(
        config=RunConfig(
            benchmark=benchmark,
            agent=job["method"],
            model="openai/Qwen/Qwen3.6-35B-A3B",
            benchmark_kwargs=kwargs,
            agent_kwargs=agent,
            task_ids=[task_id],
            max_steps=30 if benchmark == "bfcl" else 40,
            max_actions=100,
            output_dir=str(Path(job["output"]) / "evaluation/sessions"),
            run_id=run_id,
            overwrite_sessions=True,
        )
    )
    if len(result.session_results) != 1:
        raise RuntimeError("Missing official final outcome")
    session = result.session_results[0]
    from scripts.baselines.actor import official_protocol_failure, restore_protocol_outcome

    session_dir = Path(job["output"]) / "evaluation/sessions" / run_id / "sessions" / session.session_id
    protocol_failure = official_protocol_failure(session, session_dir, benchmark)
    if protocol_failure:
        session = restore_protocol_outcome(session, protocol_failure)
    if (
        (session.status.value in {"error", "cancelled"} and not protocol_failure)
        or session.score is None
        or session.success is None
    ):
        raise RuntimeError(f"Infrastructure failure: {session.status}: {session.details}")
    success = bool(session.success) if benchmark != "terminalbench2" else float(session.score) == 1.0
    auxiliary = {}
    if job.get("benchmark_candidate_metric") == "auxiliary":
        from exgentic.agents.premise.auxiliary_scores import candidate_metric

        auxiliary["auxiliary_metric"] = candidate_metric(
            {"details": session.details, "session_dir": str(session_dir)}, benchmark)
    return {
        "task_id": task_id,
        "index": index,
        "success": success,
        "score": int(success),
        "official_reward": session.score,
        "session_id": session.session_id,
        "run_id": run_id,
        "cost_reports": {
            key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
            for key, value in (session.cost_reports or {}).items()
        },
        "actor_prompt_sha256": hashlib.sha256(shared_actor_prompt(benchmark, playbook).encode()).hexdigest(),
        "attempt": attempt,
        **auxiliary,
        **(
            {
                "agent_protocol_error": protocol_failure["error"],
                "official_outcome_source": protocol_failure["official_outcome_source"],
            }
            if protocol_failure
            else {}
        ),
    }


def freeze_native(job):
    folder = Path(job["train_dir"])
    if job["method"] == "ace":
        config = read(folder / "experiment_config.json")
        if read(folder / "training_complete.json")["completed_task_count"] != job["train_count"]:
            raise ValueError("Incomplete ACE stream")
        playbook = read(folder / f"playbook_ace_isolated_{job['benchmark']}.json")["playbook"]
    else:
        state = read(folder / "state.json")
        if state["cursor"] != job["train_count"]:
            raise ValueError("Incomplete PEACE stream")
        config = state["config"]
        playbook = state["states"][job["benchmark"]]["beam"][0]["playbook"]
    from exgentic.actor_context import require_matching_actor_context

    require_matching_actor_context(config, "shared-empty-v1", [job["benchmark"]])
    evaluation = Path(job["output"]) / "evaluation"
    frozen = {
        "playbook": playbook,
        "sha256": hashlib.sha256(playbook.encode()).hexdigest(),
        "actor_context_metadata": config["actor_context_metadata"],
        "train_dir": str(folder),
        "benchmark_configs": config["benchmark_configs"],
        "test_ids": job["test_ids"],
    }
    frozen_path = evaluation / "frozen_checkpoint.json"
    if frozen_path.exists() and read(frozen_path) != frozen:
        raise ValueError("Frozen evaluation policy changed")
    write(frozen_path, frozen)
    return config, playbook, frozen


def finish_native(job):
    config, playbook, frozen = freeze_native(job)
    evaluation = Path(job["output"]) / "evaluation"
    results = {}
    for index, task in enumerate(job["test_ids"]):
        path = evaluation / "records" / f"{index:03d}.json"
        if path.exists():
            result = read(path)
            if result["task_id"] != task:
                raise ValueError("Existing final Test membership differs")
            results[index] = result
    for attempt in range(1, 4):
        pending = [
            (job, config, playbook, task, index, attempt)
            for index, task in enumerate(job["test_ids"])
            if index not in results
        ]
        if not pending:
            break
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(job["test_workers"], len(pending)), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {pool.submit(evaluate_task, item): item[4] for item in pending}
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    write(evaluation / "attempts" / f"{index:03d}_{attempt}.json", {"error": str(error)})
                    continue
                results[index] = result
                write(evaluation / "records" / f"{index:03d}.json", result)
                write(
                    Path(job["output"]) / "job_status.json",
                    {"phase": "test", "at": now(), "completed": len(results), "total": len(job["test_ids"])},
                )
    if len(results) != len(job["test_ids"]):
        raise RuntimeError("Final Test infrastructure retries exhausted; benchmark remains blocked")
    rows = [results[i] for i in range(len(results))]
    summary = {
        "complete": True,
        "tasks": len(rows),
        "score": sum(row["score"] for row in rows) / len(rows),
        "test_ids": job["test_ids"],
        "infrastructure_errors": 0,
        "policy_sha256": frozen["sha256"],
    }
    write(evaluation / "summary.json", summary)
    return summary


def work(output, identity, port_block_start=None):
    plan = read(output / "launch_manifest.json")
    job = next(job for job in plan["jobs"] if job["id"] == identity)
    if port_block_start is not None:
        if port_block_start not in plan.get("port_blocks", range(42000, 54000, 1000)):
            raise ValueError("Worker port block is outside the allocated pool")
        job = {**job, "port_block_start": port_block_start}
    root = Path(job["output"])
    root.mkdir(parents=True, exist_ok=True)
    if complete(job, plan):
        return
    env = environment(plan, job)
    os.environ.update(env)
    write(root / "job_status.json", {"phase": "training", "at": now(), "pid": os.getpid()})
    with (root / "training.log").open("a") as log:
        completed = subprocess.run(
            job["train_command"], cwd=plan["snapshot"], env=env, stdout=log, stderr=subprocess.STDOUT
        )
    if completed.returncode:
        raise RuntimeError(f"Training exited {completed.returncode}; inspect {root / 'training.log'}")
    if job["method"] in {"ace", "premise"}:
        write(root / "job_status.json", {"phase": "test", "at": now(), "completed": 0, "total": len(job["test_ids"])})
        summary = finish_native(job)
    else:
        config = read(Path(job["train_dir"]) / "config.json")
        rows = read(Path(job["train_dir"]) / "test_results.json")
        if [row["id"] for row in rows] != job["test_ids"]:
            raise ValueError("Optimizer final Test membership differs")
        if len(list((Path(job["train_dir"]) / "online").glob("*.json"))) != len(config["train"]):
            raise ValueError("Optimizer training stream incomplete")
        summary = read(Path(job["train_dir"]) / "summary.json")
        if summary.get("tasks") != len(job["test_ids"]) or summary.get("complete") is not True:
            raise ValueError("Optimizer final Test summary incomplete")
    write(
        root / "job_complete.json",
        {
            **summary,
            "job_id": identity,
            "at": now(),
            "inference_profile_sha256": plan["inference_profile_sha256"],
            "test_ids": job["test_ids"],
            "infrastructure_errors": 0,
        },
    )
    write(root / "job_status.json", {"phase": "completed", "at": now(), "score": summary["score"]})


def run(output):
    import psutil

    with (output / "controller.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(output / "launch_manifest.json")
        if sha(plan["inference_profile"]) != plan["inference_profile_sha256"]:
            raise ValueError("Inference profile changed")
        path = output / "progress.json"
        progress = (
            read(path)
            if path.exists()
            else {"jobs": {j["id"]: {"status": "pending"} for j in plan["jobs"]}, "benchmark_started_at": {}}
        )
        anchors = progress.setdefault("optimizer_started_at", {})
        # Existing controllers used a common benchmark anchor. Preserve already
        # released seeds when migrating a live cohort to the rolling queue.
        for benchmark in BENCHMARKS:
            for method in ("gepa", "skillopt"):
                key = f"{benchmark}:{method}"
                released = [
                    progress["jobs"][j["id"]]["release_at"] - j["release_delay_seconds"]
                    for j in plan["jobs"]
                    if j["benchmark"] == benchmark
                    and j["method"] == method
                    and progress["jobs"][j["id"]].get("release_at") is not None
                ]
                if released:
                    anchors.setdefault(key, min(released))
        write(output / "controller_process.json", {"pid": os.getpid(), "created_at": psutil.Process().create_time()})
        limit = plan["max_parallel"]
        port_pool = plan.get("port_blocks", list(range(42000, 54000, 1000)))
        while True:
            active = []
            for job in plan["jobs"]:
                state = progress["jobs"][job["id"]]
                if complete(job, plan):
                    state.update(status="completed", result=read(Path(job["output"]) / "job_complete.json"))
                elif state["status"] == "running":
                    try:
                        process = psutil.Process(state["pid"])
                        alive = (
                            process.create_time() == state["created_at"] and process.status() != psutil.STATUS_ZOMBIE
                        )
                    except psutil.Error:
                        alive = False
                    if alive:
                        active.append((job, state))
                    else:
                        state.update(status="error", ended_at=now())
            used = {state.get("port_block_start", job["port_block_start"]) for job, state in active}
            if len(used) != len(active) or len(active) > limit:
                raise ValueError("Live workers exceed the concurrency or exclusive port allocation")
            for job in plan["jobs"]:
                if len(active) >= limit:
                    break
                state = progress["jobs"][job["id"]]
                if state["status"] != "pending":
                    continue
                instant = time.time()
                key = f"{job['benchmark']}:{job['method']}"
                if job["release_delay_seconds"]:
                    anchor = anchors.get(key)
                    if anchor is None or instant < anchor + job["release_delay_seconds"]:
                        continue
                else:
                    anchor = instant
                free = next((port for port in port_pool if port not in used), None)
                if free is None:
                    raise ValueError("No exclusive port block available for the free execution slot")
                root = Path(job["output"])
                root.mkdir(parents=True, exist_ok=True)
                command = [
                    str(Path(plan["snapshot"]) / ".venv/bin/python"),
                    "-u",
                    str(Path(plan["snapshot"]) / "scripts/utils/run_shared_reflection_matrix.py"),
                    "work",
                    "--output",
                    str(output),
                    "--job",
                    job["id"],
                    "--port-block-start",
                    str(free),
                ]
                with (root / "worker.log").open("a") as log:
                    child = subprocess.Popen(
                        command,
                        cwd=plan["snapshot"],
                        env=environment(plan, {**job, "port_block_start": free}),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                progress["benchmark_started_at"].setdefault(job["benchmark"], instant)
                if job["method"] in {"gepa", "skillopt"} and job["seed"] == 42:
                    anchors.setdefault(key, instant)
                state.pop("ended_at", None)
                state.update(
                    status="running",
                    pid=child.pid,
                    created_at=psutil.Process(child.pid).create_time(),
                    started_at=now(),
                    release_at=anchor + job["release_delay_seconds"],
                    port_block_start=free,
                    command=command,
                )
                used.add(free)
                active.append((job, state))
            write(path, progress)
            states = [progress["jobs"][j["id"]]["status"] for j in plan["jobs"]]
            phase = (
                "completed"
                if all(s == "completed" for s in states)
                else "running"
                if active
                else "waiting_seed_release"
                if "pending" in states
                else "blocked"
            )
            write(
                output / "status.json",
                {
                    "phase": phase,
                    "at": now(),
                    "scheduling": "rolling",
                    "max_parallel": limit,
                    "benchmark": next(
                        (
                            b
                            for b in BENCHMARKS
                            if any(
                                j["benchmark"] == b and progress["jobs"][j["id"]]["status"] != "completed"
                                for j in plan["jobs"]
                            )
                        ),
                        None,
                    ),
                    "active_benchmarks": sorted({job["benchmark"] for job, _state in active}),
                    "running": states.count("running"),
                    "completed": states.count("completed"),
                    "pending": states.count("pending"),
                    "errors": states.count("error"),
                    "runs": len(states),
                },
            )
            if phase in {"completed", "blocked"}:
                return
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "work"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--port-block-start", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        prepare(output, args.profile, args.validation, args.calibration)
    elif args.action == "run":
        run(output)
    else:
        try:
            work(output, args.job, args.port_block_start)
        except Exception as error:
            plan = read(output / "launch_manifest.json")
            job = next(job for job in plan["jobs"] if job["id"] == args.job)
            write(Path(job["output"]) / "job_status.json", {"phase": "error", "at": now(), "error": str(error)})
            raise


if __name__ == "__main__":
    main()
