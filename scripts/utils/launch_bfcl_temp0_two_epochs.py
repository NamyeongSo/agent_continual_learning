# SPDX-License-Identifier: Apache-2.0
"""Fresh BFCL two-epoch matrix, temperature zero, seed-pair launch staggering."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path("/home/thskadud/Sico/labs/AgentStream/exgentic")
PREVIOUS = REPO / "outputs/qwen_epoch2_continuation_four_baselines_20261109"
VARIANTS = ("ace_aud", "peace_no_alignment_aud", "gepa_val", "skillopt_val")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def release_delay(method, seed):
    return (seed - 42) * 600 if method in ("gepa", "skillopt") else 0


def copy_snapshot(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("src", "scripts"):
        shutil.copytree(
            source / name,
            destination / name,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "outputs", ".pytest_cache"),
        )
    for name in (".venv", ".exgentic"):
        (destination / name).symlink_to((REPO / name).resolve(), target_is_directory=True)
    shutil.copy2(REPO / "scripts/utils/continue_epoch2.py", destination / "scripts/utils/continue_epoch2.py")
    evaluation = destination / "scripts/utils/run_qwen_isolated_when_free.py"
    code = evaluation.read_text()
    import re

    code, changed = re.subn(r"^TEMPERATURE = [0-9.]+$", "TEMPERATURE = 0.0", code, flags=re.MULTILINE)
    assert changed == 1
    evaluation.write_text(code)
    return destination


def patch_baselines(snapshot):
    runner = snapshot / "scripts/baselines/run_experiment.py"
    code = runner.read_text()
    assert 'parser.add_argument("--epochs"' in code and 'parser.add_argument("--temperature"' in code
    anchor = '        item["encounter_id"] = '
    line = next(line for line in code.splitlines() if line.startswith(anchor))
    code = code.replace(line, line + '\n        item["epoch"] = index // len(base_train) + 1')
    code = code.replace(
        "    runtime = Runtime(config, scripted=scripted)",
        '    config["epoch1_train"] = config["train"][:config["epoch_size"]]\n'
        "    runtime = Runtime(config, scripted=scripted)\n"
        "    runtime.activate = lambda: None",
    )
    code = code.replace(
        "                from scripts.baselines.skillopt_runtime import train",
        "                from scripts.utils.continue_epoch2 import skillopt_train\n"
        "                def train(runtime, upstream_dir):\n"
        '                    return skillopt_train(runtime, {"source_snapshot": str(ROOT), '
        '"train_dir": str(runtime.output)})',
    )
    code = code.replace('"termination": "single_pass_stream_exhausted"', '"termination": "two_epochs_stream_exhausted"')
    runner.write_text(code)
    # The transport passes temperature through to actors and all optimizer calls.
    runtime = (snapshot / "scripts/baselines/runtime.py").read_text()
    actor = (snapshot / "scripts/baselines/actor.py").read_text()
    assert 'temperature=self.config["temperature"]' in runtime
    assert '"temperature": request["temperature"]' in actor
    assert 'item.get("encounter_id", item["task_id"])' in runtime


def prepare(output):
    if (output / "launch_manifest.json").exists():
        return read(output / "launch_manifest.json")
    output.mkdir(parents=True, exist_ok=True)
    previous = read(PREVIOUS / "launch_manifest.json")
    originals = [job for job in previous["jobs"] if job["benchmark"] == "bfcl"]
    assert len(originals) == 12
    snapshots = {}
    for source in sorted({job["source_snapshot"] for job in originals}):
        name = "peace" if "peace_no_alignment_" in source else "base"
        snapshots[source] = copy_snapshot(Path(source), output / "code_snapshots" / name)
        if name == "base":
            patch_baselines(snapshots[source])
    manifest = output / "task_splits.json"
    shutil.copy2(Path(originals[0]["source_spec"]["root"]) / "task_splits.json", manifest)
    assert sha(manifest) == previous["manifest_sha256"]
    frozen = output / "fresh_matrix.py"
    shutil.copy2(__file__, frozen)
    jobs = []
    for original in sorted(originals, key=lambda j: (VARIANTS.index(j["variant"]), j["seed"])):
        job = copy.deepcopy(original)
        directory = output / "runs" / job["variant"] / "bfcl" / f"s{job['seed']}"
        train = directory / "train"
        train.mkdir(parents=True, exist_ok=False)
        snapshot = snapshots[job["source_snapshot"]]
        job.update(
            output=str(directory),
            train_dir=str(train),
            source_snapshot=str(snapshot),
            base_url="http://127.0.0.1:18006/v1",
            train_count=100,
            restored_tasks=0,
            temperature_all_roles=0.0,
            from_scratch=True,
            epochs=2,
            release_delay_seconds=release_delay(job["method"], job["seed"]),
            status="pending",
            port_block_start=47000 + len(jobs) * 500,
        )
        # Metadata only: no old state, learned playbook, metrics, or caches are copied.
        job["source_spec"].update(temperature=0.0, snapshot=str(snapshot))
        job["epoch1_count"] = 50  # Epoch size; not restored task count.
        job["train_ids"] = job["epoch1_order"] + job["epoch2_order"]
        assert len(job["train_ids"]) == 100 and len(set(job["train_ids"])) == 50
        python = str(snapshot / ".venv/bin/python")
        if job["method"] in ("gepa", "skillopt"):
            job["train_command"] = [
                python,
                "-u",
                str(snapshot / "scripts/baselines/run_experiment.py"),
                "--method",
                job["method"],
                "--benchmark",
                "bfcl",
                "--seed",
                str(job["seed"]),
                "--optimizer-seed",
                str(job.get("optimizer_seed", 42)),
                "--validation-source",
                "val",
                "--model",
                "openai/Qwen/Qwen3.6-35B-A3B",
                "--manifest",
                str(manifest),
                "--base-url",
                job["base_url"],
                "--workers",
                "50",
                "--epochs",
                "2",
                "--temperature",
                "0.0",
                "--output",
                str(train),
            ]
        else:
            job["train_command"] = [python, "-u", str(frozen), "train", "--output", str(output), "--job", job["id"]]
        job["command"] = [
            str(REPO / ".venv/bin/python"),
            "-u",
            str(frozen),
            "worker",
            "--output",
            str(output),
            "--job",
            job["id"],
        ]
        write(
            directory / "fresh_start_provenance.json",
            {
                "from_scratch": True,
                "restored_tasks": 0,
                "temperature_all_roles": 0.0,
                "epoch1_order": job["epoch1_order"],
                "epoch2_order": job["epoch2_order"],
                "epoch2_shuffle": "Random(seed).shuffle(epoch1_actual_order.copy())",
                "copied_training_artifacts": [],
                "source_code_template": original["source_snapshot"],
                "frozen_code_snapshot": str(snapshot),
            },
        )
        jobs.append(job)
    hashes = {
        str(path.relative_to(output)): sha(path)
        for snapshot in snapshots.values()
        for directory in ("src", "scripts")
        for path in (snapshot / directory).rglob("*.py")
    }
    hashes["fresh_matrix.py"] = sha(frozen)
    plan = {
        "created_at": now(),
        "model": "Qwen/Qwen3.6-35B-A3B",
        "mode": "isolated",
        "epochs": 2,
        "from_scratch": True,
        "temperature_all_roles": 0.0,
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
        "base_url": "http://127.0.0.1:18006/v1",
        "max_parallel": 12,
        "initial_jobs": 8,
        "seed_stagger_seconds": 600,
        "environment": previous["environment"],
        "jobs": jobs,
        "code_hashes": hashes,
        "final_evaluation_splits": ["train", "val", "test"],
        "initial_validation_reuse": False,
    }
    write(output / "launch_manifest.json", plan)
    write(output / "progress.json", {"jobs": {j["id"]: {"status": "pending"} for j in jobs}})
    write(output / "status.json", {"phase": "planned", "at": now()})
    return plan


def environment(plan, job):
    env = dict(os.environ, **plan["environment"])
    env.update(
        OPENAI_BASE_URL=job["base_url"],
        OPENAI_API_BASE=job["base_url"],
        OPENAI_API_KEY="EMPTY",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        EXGENTIC_PORT_BLOCK_START=str(job["port_block_start"]),
    )
    env["PATH"] = str(Path(job["source_snapshot"]) / ".venv/bin") + os.pathsep + env.get("PATH", "")
    return env


def train(plan, job):
    snapshot = Path(job["source_snapshot"])
    sys.path[:0] = [str(snapshot / "src"), str(snapshot)]
    old_plan = read(job["source_plan"])
    old = next(j for j in old_plan["jobs"] if j["id"] == job["id"])
    argv = old["train_command"][3:]
    argv = [arg for arg in argv if arg != "--resume-sessions"]
    for flag, value in (
        ("--output-dir", job["train_dir"]),
        ("--temperature", "0.0"),
        ("--task-split-manifest", plan["manifest"]),
    ):
        argv[argv.index(flag) + 1] = value
    if "--base-url" in argv:
        argv[argv.index("--base-url") + 1] = job["base_url"]
    if job["method"] == "ace":
        file = snapshot / "scripts/ace/run_experiment.py"
        module = load_module("fresh_ace_runner", file)
        module.task_order_from_train_manifest = lambda *args, **kwargs: [("bfcl", task) for task in job["train_ids"]]
    else:
        from exgentic.agents.premise.pipeline import PremisePipeline

        original = PremisePipeline._restore_or_initialize

        def restore(self):
            self.order = [("bfcl", task) for task in job["train_ids"]]
            return original(self)

        PremisePipeline._restore_or_initialize = restore
        file = snapshot / "scripts/premise/run_experiment.py"
        module = load_module("fresh_peace_runner", file)
    sys.argv = [str(file), *argv]
    module.main()


def worker(plan, job):
    snapshot = Path(job["source_snapshot"])
    sys.path[:0] = [str(snapshot / "src"), str(snapshot)]
    os.environ.update(environment(plan, job))
    fixed = load_module("fresh_fixed_worker", snapshot / "scripts/utils/fixed_matrix_worker.py")
    try:
        fixed.run({**plan, "snapshot": str(snapshot)}, job)
    except BaseException as exc:
        write(Path(job["output"]) / "job_status.json", {"phase": "error", "at": now(), "error": repr(exc)})
        raise


def run(output, plan):
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = read(output / "progress.json")
        helper = load_module("fresh_process_helper", REPO / "scripts/utils/continue_epoch2.py")
        active = {}
        anchor_file = output / "release_anchor.json"
        anchor = read(anchor_file)["epoch"] if anchor_file.exists() else time.time()
        write(anchor_file, {"epoch": anchor, "at": now()})
        write(output / "controller_process.json", {"pid": os.getpid(), "output": str(output), "at": now()})
        for job in plan["jobs"]:
            prior = progress["jobs"][job["id"]]
            if helper.completed_job(job):
                prior["status"] = "completed"
            elif prior.get("status") == "running":
                adopted = helper.existing_worker(job, prior, output)
                if adopted:
                    active[job["id"]] = (job, adopted)
        while True:
            for job in plan["jobs"]:
                entry = progress["jobs"][job["id"]]
                if entry["status"] != "pending" or time.time() < anchor + job["release_delay_seconds"]:
                    continue
                logfile = (Path(job["output"]) / "worker.log").open("a")
                process = subprocess.Popen(
                    job["command"],
                    cwd=REPO,
                    env=environment(plan, job),
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                logfile.close()
                active[job["id"]] = (job, process)
                entry.update(
                    status="running",
                    pid=process.pid,
                    started_at=now(),
                    scheduled_at=datetime.fromtimestamp(
                        anchor + job["release_delay_seconds"], timezone(timedelta(hours=9))
                    ).isoformat(),
                )
            for identity, (job, process) in list(active.items()):
                result = process.poll()
                if result is None:
                    continue
                good = result == 0 and helper.completed_job(job)
                progress["jobs"][identity].update(
                    status="completed" if good else "error", returncode=result, finished_at=now()
                )
                del active[identity]
            write(output / "progress.json", progress)
            pending = sum(j["status"] == "pending" for j in progress["jobs"].values())
            errors = sum(j["status"] == "error" for j in progress["jobs"].values())
            phase = "running" if active or pending else "blocked" if errors else "completed"
            write(
                output / "status.json",
                {
                    "phase": phase,
                    "at": now(),
                    "active": len(active),
                    "pending": pending,
                    "errors": errors,
                    "benchmark": "bfcl",
                    "epochs": 2,
                    "temperature_all_roles": 0.0,
                },
            )
            if not active and not pending:
                break
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        plan = prepare(output)
        print(json.dumps({"jobs": len(plan["jobs"]), "output": str(output)}))
        return
    plan = read(output / "launch_manifest.json")
    if args.action == "run":
        run(output, plan)
    else:
        job = next(j for j in plan["jobs"] if j["id"] == args.job)
        (worker if args.action == "worker" else train)(plan, job)


if __name__ == "__main__":
    main()
