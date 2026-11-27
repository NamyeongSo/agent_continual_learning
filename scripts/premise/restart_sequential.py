# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Restart sequential runs after BFCL and optional fresh interleaved runs with one evaluation queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from exgentic.agents.premise.evaluation_queue import CheckpointEvaluationQueue  # noqa: E402
from exgentic.agents.premise.pipeline import PremisePipeline, write_json  # noqa: E402
from exgentic.utils.task_splits import select_task_split_sections, validate_task_split  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_workspace_source() -> None:
    """Use the validated workspace code inside benchmark venvs, retaining their dependencies."""
    source = str(ROOT / "src")
    inherited = [path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path and path != source]
    os.environ["PYTHONPATH"] = os.pathsep.join([source, *inherited])


def fork_bfcl_checkpoint(source: Path, destination: Path, seed: int) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Restart output must be separate from the source experiment")
    checkpoint = source / "checkpoints/checkpoint_050.json"
    snapshot_bytes = checkpoint.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    config = snapshot["config"]
    expected_order = ["bfcl"] * 50 + ["appworld"] * 50 + ["terminalbench2"] * 30
    if (
        snapshot["cursor"] != 50 or config["seed"] != seed or config["mode"] != "sequential"
        or config.get("method", "premise") != "premise"
        or [slug for slug, _ in config["order"]] != expected_order
    ):
        raise ValueError("Source must be this seed's PREMiSE sequential BFCL checkpoint 50")
    if sha256(Path(config["manifest"])) != config["manifest_sha256"]:
        raise ValueError("Source task manifest has changed")
    artifacts = [Path("iterations") / f"iter_{index:04d}.json" for index in range(1, 51)]
    for relative in artifacts:
        if not (source / relative).is_file():
            raise FileNotFoundError(source / relative)
    for memory in snapshot["states"]["global"]["memory"]:
        if memory["benchmark"] != "bfcl":
            raise ValueError("BFCL checkpoint contains another benchmark's memory")
        if "trace" not in memory and Path(memory["trace_artifact"]) not in artifacts:
            raise ValueError("Memory references an artifact outside the BFCL prefix")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "checkpoints").mkdir()
    (destination / "iterations").mkdir()
    (destination / "state.json").write_bytes(snapshot_bytes)
    (destination / "checkpoints/checkpoint_050.json").write_bytes(snapshot_bytes)
    hashes = {}
    for relative in artifacts:
        shutil.copy2(source / relative, destination / relative)
        hashes[str(relative)] = sha256(destination / relative)
        if hashes[str(relative)] != sha256(source / relative):
            raise RuntimeError(f"BFCL source changed during copy: {relative}")
    provenance = {
        "source": str(source), "source_checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "seed": seed, "resume_cursor": 50, "next_task": config["order"][50],
        "copied_artifacts": hashes, "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(destination / "restart_provenance.json", provenance)
    return config


def pipeline_for(output: Path, base_url: str, workers: int, execution_tag: str = "") -> PremisePipeline:
    config = json.loads((output / "state.json").read_text())["config"]
    return pipeline_from_config(output, config, base_url, workers, execution_tag)


def pipeline_from_config(output: Path, config: dict, base_url: str, workers: int,
                         execution_tag: str = "") -> PremisePipeline:
    return PremisePipeline(
        output=output, manifest=Path(config["manifest"]), benchmarks=config["benchmarks"],
        mode=config["mode"], seed=config["seed"], count=config["count"], past_count=config["past_count"],
        model=config["model"], base_url=base_url, replay_retries=config["replay_retries"],
        max_steps=config["max_steps"], llm_calls=workers,
        tool_calling_max_tokens=2048,
        execution_tag=execution_tag,
        benchmark_limits=dict.fromkeys(config["benchmarks"]),
    )


def fork_interleaved_initial(source: Path, destination: Path, seed: int, base_url: str,
                             execution_tag: str = "") -> dict:
    """Reuse frozen configuration/order, initializing a new empty beam and RNG."""
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Restart output must be separate from the source experiment")
    source_state = source / "state.json"
    config = json.loads(source_state.read_text())["config"]
    if config["seed"] != seed or config["mode"] != "interleaved" or config.get("method", "premise") != "premise":
        raise ValueError("Source must be this seed's PREMiSE interleaved experiment")
    if sha256(Path(config["manifest"])) != config["manifest_sha256"]:
        raise ValueError("Source task manifest has changed")
    destination.mkdir(parents=True, exist_ok=False)
    pipeline = pipeline_from_config(destination, config, base_url, 32, execution_tag)
    if json.loads(json.dumps(pipeline.order)) != config["order"]:
        raise ValueError("Fresh interleaved task order differs from the original experiment")
    if pipeline.cursor != 0 or pipeline.states != {"global": pipeline._empty_state()}:
        raise ValueError("Fresh interleaved initialization unexpectedly contains learned state")
    write_json(destination / "restart_provenance.json", {
        "source": str(source), "source_state_sha256": sha256(source_state), "seed": seed,
        "resume_cursor": 0, "initialization": "Empty playbook beam and memory; RNG initialized from seed",
        "reused": "Frozen configuration and task order only", "created_at": datetime.now(UTC).isoformat(),
    })
    return config


def training_command(output: Path, config: dict, base_url: str, execution_tag: str = "") -> list[str]:
    command = [
        sys.executable, "-u", str(ROOT / "scripts/premise/run_experiment.py"),
        "--task-split-manifest", config["manifest"], "--output-dir", str(output),
        "--benchmarks", ",".join(config["benchmarks"]), "--mode", config["mode"],
        "--seed", str(config["seed"]), "--all-train-tasks", "--past", str(config["past_count"]),
        "--model", config["model"], "--base-url", base_url,
        "--replay-workers", "28", "--replay-retries", str(config["replay_retries"]),
        "--llm-timeout", "600", "--llm-retries", "1", "--llm-calls", "32",
        "--tool-calling-max-tokens", "2048",
        "--execution-tag", execution_tag,
        "--unlimited-benchmark-sessions",
    ]
    for slug, steps in config["max_steps"].items():
        command.extend((f"--{slug}-max-steps", str(steps)))
    return command


class AdoptedTrainingProcess:
    """Monitor a detached training worker without restarting its active task."""

    def __init__(self, pid: int, output: Path, proc_root: Path = Path("/proc")):
        self.pid, self.output, self.proc_root = pid, output, proc_root
        directory = proc_root / str(pid)
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        argv = [os.fsdecode(item) for item in (directory / "cmdline").read_bytes().split(b"\0") if item]
        script = str(ROOT / "scripts/premise/run_experiment.py")
        if (fields[0] in ("Z", "X") or script not in argv or "--output-dir" not in argv
                or argv[argv.index("--output-dir") + 1] != str(output)):
            raise ValueError(f"PID {pid} is not a live training worker for {output}")
        self.start_ticks = fields[19]

    def poll(self):
        try:
            fields = (self.proc_root / str(self.pid) / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] not in ("Z", "X") and fields[19] == self.start_ticks:
                return None
        except (OSError, IndexError):
            pass
        complete = self.output / "training_complete.json"
        if complete.exists():
            result = json.loads(complete.read_text())
            count = result.get("completed_task_count")
            if isinstance(count, int) and count > 0 and count == result.get("expected_task_count"):
                return 0
        return 1


def stop_training_worker_at_boundary(worker: AdoptedTrainingProcess, archive: Path) -> None:
    """Preserve completed state and stop a worker tree before loading revised runtime code."""
    def scan():
        table = {}
        for directory in Path("/proc").glob("[0-9]*"):
            try:
                fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
                if fields[0] not in ("Z", "X"):
                    table[int(directory.name)] = {"ppid": int(fields[1]), "start_ticks": fields[19]}
            except (OSError, IndexError, ValueError):
                continue
        return table

    targets = {}

    def send(pid, sig):
        try:
            fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[19] == targets[pid]["start_ticks"] and fields[0] not in ("Z", "X"):
                os.kill(pid, sig)
        except (OSError, IndexError):
            pass

    if worker.poll() is not None:
        raise RuntimeError(f"Worker {worker.pid} exited before its planned runtime refresh")
    archive.mkdir(parents=True, exist_ok=False)
    for _ in range(3):
        table = scan()
        if worker.pid in table and table[worker.pid]["start_ticks"] == worker.start_ticks:
            targets[worker.pid] = table[worker.pid]
        while True:
            additions = {pid: item for pid, item in table.items() if item["ppid"] in targets and pid not in targets}
            if not additions:
                break
            targets.update(additions)
        for pid in targets:
            send(pid, signal.SIGSTOP)
    contents = (worker.output / "state.json").read_bytes()
    (archive / "state.json").write_bytes(contents)
    cursor = json.loads(contents)["cursor"]
    for partial in (worker.output / "iterations").glob("iter_*.json"):
        if int(partial.stem.split("_")[1]) > cursor:
            shutil.copy2(partial, archive / partial.name)
    write_json(archive / "stopped_worker.json", {"pid": worker.pid, "cursor": cursor, "processes": targets,
                                                "stopped_at": datetime.now(UTC).isoformat()})
    for pid in targets:
        send(pid, signal.SIGTERM)
        send(pid, signal.SIGCONT)
    time.sleep(0.2)
    for pid in targets:
        send(pid, signal.SIGKILL)


def run_controller(output: Path, seeds: list[int], base_url: str, workers: int, port_base: int,
                   execution_tag: str = "", interleaved_seeds: list[int] | None = None,
                   adopt_training_pids: dict[int, int] | None = None,
                   refresh_adopted_at_boundary: bool | list[int] = False,
                   evaluation_temperature_overrides: list[dict] | None = None) -> int:
    interleaved_seeds = interleaved_seeds or []
    all_seeds = seeds + interleaved_seeds
    checkpoints = {seed: (50, 100, 130) if seed in interleaved_seeds else (100, 130) for seed in all_seeds}
    queue = CheckpointEvaluationQueue(workers)
    pipelines, processes, rows = {}, {}, {}
    refresh_cursors = {}
    # Validate every adopted worker before starting any replacement workers.
    for seed, pid in (adopt_training_pids or {}).items():
        mode = "interleaved" if seed in interleaved_seeds else "sequential"
        processes[seed] = AdoptedTrainingProcess(pid, output / f"s{seed}_{mode}_t3")
        if refresh_adopted_at_boundary is True or (
            isinstance(refresh_adopted_at_boundary, list) and seed in refresh_adopted_at_boundary
        ):
            refresh_cursors[seed] = json.loads((processes[seed].output / "state.json").read_text())["cursor"]
    # Build frozen evaluator configuration before training starts. Evaluation
    # loads each checkpoint's beam, never the mutable training beam or memory.
    for seed in all_seeds:
        mode = "interleaved" if seed in interleaved_seeds else "sequential"
        run_output = output / f"s{seed}_{mode}_t3"
        pipelines[seed] = pipeline_for(run_output, base_url, workers, execution_tag)
        pipelines[seed].evaluation_temperature_overrides = {
            (item["checkpoint"], item["benchmark"], item["task_id"]): item["temperature"]
            for item in (evaluation_temperature_overrides or []) if item["seed"] == seed
        }
        rows[seed] = {"seed": seed, "mode": mode, "output": str(run_output),
                      "evaluations": {}, "status": "pending"}
    for index, seed in enumerate(all_seeds):
        run_output = pipelines[seed].output
        state = json.loads((run_output / "state.json").read_text())
        config = state["config"]
        env = os.environ.copy()
        env["EXGENTIC_PORT_BLOCK_START"] = str(port_base + 1000 * index)
        process = processes.get(seed)
        if process is None:
            with (output / "logs" / f"s{seed}_training.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    training_command(run_output, config, base_url, execution_tag), cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
        processes[seed] = process
        rows[seed].update(status="training", pid=process.pid, port_block=env["EXGENTIC_PORT_BLOCK_START"])
        print(f"START seed={seed} training pid={process.pid} cursor={state['cursor']}", flush=True)
    evaluation_keys = {}
    try:
        while True:
            queue.poll()
            for seed in all_seeds:
                row, pipeline = rows[seed], pipelines[seed]
                state = json.loads((pipeline.output / "state.json").read_text())
                row["cursor"] = state["cursor"]
                code = processes[seed].poll()
                if seed in refresh_cursors and state["cursor"] > refresh_cursors[seed]:
                    if code is None:
                        archive = output / "runtime_revisions" / execution_tag / f"s{seed}_runtime_refresh"
                        stop_training_worker_at_boundary(processes[seed], archive)
                        env = os.environ.copy()
                        env["EXGENTIC_PORT_BLOCK_START"] = row["port_block"]
                        with (output / "logs" / f"s{seed}_training.log").open("a", encoding="utf-8") as log:
                            process = subprocess.Popen(
                                training_command(pipeline.output, state["config"], base_url, execution_tag),
                                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True,
                            )
                        processes[seed] = process
                        row.update(pid=process.pid, runtime_refreshed_at=datetime.now(UTC).isoformat(),
                                   runtime_refresh_cursor=state["cursor"])
                        print(f"REFRESH seed={seed} pid={process.pid} cursor={state['cursor']}", flush=True)
                        code = process.poll()
                    del refresh_cursors[seed]
                row["returncode"] = code
                if code is not None:
                    row["status"] = "trained" if code == 0 and state["cursor"] == 130 else "training_failed"
                for checkpoint in checkpoints[seed]:
                    key = (seed, checkpoint)
                    if key not in evaluation_keys and (
                        pipeline.output / "checkpoints" / f"checkpoint_{checkpoint:03d}.json"
                    ).is_file():
                        evaluation_keys[key] = queue.enqueue(pipeline, checkpoint)
                        print(f"ENQUEUE seed={seed} checkpoint={checkpoint} all benchmarks", flush=True)
                    if key in evaluation_keys:
                        row["evaluations"][str(checkpoint)] = (
                            "complete" if queue.complete(evaluation_keys[key]) else "running"
                        )
                if all(row["evaluations"].get(str(point)) == "complete" for point in checkpoints[seed]):
                    write_json(pipeline.output / "test_evaluation/complete.json", {
                        "evaluation_groups": 3 * len(checkpoints[seed]), "checkpoints": list(checkpoints[seed]),
                        "total_tasks": sum(len(queue.groups[key]["results"])
                                           for point in checkpoints[seed] for key in evaluation_keys[(seed, point)]),
                    })
            status = {
                "updated_at": datetime.now(UTC).isoformat(), "controller_pid": os.getpid(),
                "queue": queue.status(), "experiments": list(rows.values()),
            }
            write_json(output / "restart_status.json", status)
            if all(process.poll() is not None for process in processes.values()) and not queue.futures:
                success = all(row["status"] == "trained" and all(
                    row["evaluations"].get(str(point)) == "complete" for point in checkpoints[row["seed"]]
                ) for row in rows.values())
                write_json(output / "restart_complete.json", {**status, "success": success})
                return 0 if success else 1
            time.sleep(2)
    finally:
        queue.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    parser.add_argument("--interleaved-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--evaluation-workers", type=int, default=50)
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--port-block-base", type=int, default=30000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    parser.add_argument("--execution-tag", default="")
    parser.add_argument("--adopt-training-pids", nargs="*", default=[], metavar="SEED:PID",
                        help="Keep specified live training workers when replacing their controller")
    parser.add_argument("--refresh-adopted-at-boundary", action="store_true",
                        help="Reload revised runtime code after each adopted worker completes its current task")
    parser.add_argument("--refresh-adopted-seeds", type=int, nargs="*", default=[],
                        help="Refresh only these adopted workers after their current task")
    parser.add_argument("--evaluation-temperature-overrides", type=Path,
                        help="JSON list of temperature overrides for selected seed/checkpoint/benchmark/task jobs")
    args = parser.parse_args()
    all_seeds = args.seeds + args.interleaved_seeds
    try:
        adopt_training_pids = dict(tuple(map(int, item.split(":"))) for item in args.adopt_training_pids)
    except (ValueError, TypeError):
        parser.error("--adopt-training-pids requires SEED:PID pairs")
    if ((args.adopt_training_pids and not args.run_prepared)
            or len(adopt_training_pids) != len(args.adopt_training_pids)
            or any(seed not in all_seeds or pid < 1 for seed, pid in adopt_training_pids.items())):
        parser.error("Adopted workers must have distinct prepared seeds and positive PIDs")
    if args.refresh_adopted_at_boundary and not adopt_training_pids:
        parser.error("Runtime refresh requires adopted training workers")
    if any(seed not in adopt_training_pids for seed in args.refresh_adopted_seeds):
        parser.error("Runtime refresh seeds must be adopted workers")
    overrides = []
    if args.evaluation_temperature_overrides:
        try:
            overrides = json.loads(args.evaluation_temperature_overrides.read_text())
            if not isinstance(overrides, list) or any(
                item["seed"] not in all_seeds or item["checkpoint"] not in (50, 100, 130)
                or item["benchmark"] not in ("appworld", "bfcl") or not isinstance(item["task_id"], str)
                or not isinstance(item["temperature"], (int, float)) or not 0 <= item["temperature"] <= 2
                for item in overrides
            ):
                parser.error("Invalid evaluation temperature override")
        except (OSError, ValueError, KeyError, TypeError):
            parser.error("Unable to read evaluation temperature overrides")
    if len(set(all_seeds)) != len(all_seeds) or args.evaluation_workers < 1:
        parser.error("Seeds must be distinct and evaluation workers must be positive")
    if not 1024 <= args.port_block_base <= 64535 - 1000 * len(all_seeds):
        parser.error("Port blocks exceed the valid range")
    if args.prepare_only and args.run_prepared:
        parser.error("Choose either prepare-only or run-prepared")
    output = args.output_root.resolve()
    if not args.run_prepared:
        output.mkdir(parents=True, exist_ok=False)
        (output / "logs").mkdir()
        plans = []
        for seed in all_seeds:
            mode = "interleaved" if seed in args.interleaved_seeds else "sequential"
            source = args.source_root / f"s{seed}_{mode}_t3"
            destination = output / f"s{seed}_{mode}_t3"
            if mode == "interleaved":
                config = fork_interleaved_initial(source, destination, seed, args.base_url, args.execution_tag)
            else:
                config = fork_bfcl_checkpoint(source, destination, seed)
            sections = select_task_split_sections(
                json.loads(Path(config["manifest"]).read_text()), config["benchmarks"], seed,
            )
            for checkpoint in ((50, 100, 130) if mode == "interleaved" else (100, 130)):
                for slug in ("appworld", "terminalbench2", "bfcl"):
                    plans.append({"seed": seed, "checkpoint": checkpoint, "benchmark": slug,
                                  "task_ids": validate_task_split(sections[slug], slug)["test"]})
        write_json(output / "launch_manifest.json", {
            "source_root": str(args.source_root.resolve()), "seeds": args.seeds,
            "interleaved_seeds": args.interleaved_seeds,
            "base_url": args.base_url, "evaluation_workers": args.evaluation_workers,
            "port_block_base": args.port_block_base,
            "training": (
                "BFCL checkpoint 50 -> AppWorld 50 tasks -> TerminalBench 30 tasks, without evaluation barriers"
                "; optional interleaved seeds start from an empty playbook and train all 130 tasks"
            ),
            "evaluation": (
                "One shared (seed, checkpoint, benchmark, datapoint) queue; "
                "sequential 100/130, interleaved 50/100/130"
            ),
            "llm_timeout": 600, "llm_retries": 1, "native_terminus_sampling": "unchanged",
            "acting_max_tokens": {"bfcl": 2048, "appworld": 2048},
            "execution_tag": args.execution_tag,
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
                ROOT / "src/exgentic/agents/premise/llm.py",
                ROOT / "src/exgentic/agents/premise/pipeline.py",
                ROOT / "scripts/premise/restart_sequential.py",
            )},
            "evaluation_plan": plans, "evaluation_tasks": sum(len(item["task_ids"]) for item in plans),
        })
        print(f"PREPARED {output}: {sum(len(item['task_ids']) for item in plans)} evaluation tasks", flush=True)
    if args.prepare_only:
        return 0
    expected = json.loads((output / "launch_manifest.json").read_text())
    for name in ("seeds", "base_url", "evaluation_workers", "port_block_base"):
        if expected[name] != getattr(args, name):
            parser.error(f"Prepared configuration differs: {name}")
    if expected.get("interleaved_seeds", []) != args.interleaved_seeds:
        parser.error("Prepared configuration differs: interleaved_seeds")
    lock = (output / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "restart_status.json").exists():
        parser.error("This prepared run has already started; refusing to duplicate its training workers")
    os.environ["OPENAI_API_BASE"] = args.base_url
    configure_workspace_source()
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("DOCKER_HOST", os.environ.get("ACE_DOCKER_HOST", "ssh://exgentic-daysi-docker:14233"))
    os.environ["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["EXGENTIC_PORT_BLOCK_START"] = str(args.port_block_base + 1000 * len(all_seeds))
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
    return run_controller(
        output, args.seeds, args.base_url, args.evaluation_workers, args.port_block_base, args.execution_tag,
        args.interleaved_seeds, adopt_training_pids,
        args.refresh_adopted_at_boundary or args.refresh_adopted_seeds, overrides,
    )


if __name__ == "__main__":
    raise SystemExit(main())
