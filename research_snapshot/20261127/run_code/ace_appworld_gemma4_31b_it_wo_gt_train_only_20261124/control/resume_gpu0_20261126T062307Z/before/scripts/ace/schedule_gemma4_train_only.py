# SPDX-License-Identifier: Apache-2.0
"""Hand GPU1 to the next Gemma Train-only experiment after durable predecessor completion."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ace import run_gemma4_train_only as study  # noqa: E402
from scripts.ace.schedule_qwen35_9b_add_only import pid_alive  # noqa: E402

base = study.base
CHECK_INTERVAL_SECONDS = 7200


def checkpoint_ready(model, revision):
    checkpoint = Path.home() / ".cache/huggingface/hub" / ("models--" + model.replace("/", "--"))
    checkpoint = checkpoint / "snapshots" / revision
    index = checkpoint / "model.safetensors.index.json"
    if not index.exists():
        return False
    shards = set(base.read(index)["weight_map"].values())
    return bool(shards) and all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0
                                for name in shards | {"config.json", "tokenizer.json", "chat_template.jinja"})


def predecessor_complete(output):
    status_path, receipt_path = output / "status.json", output / "completed.json"
    if not status_path.exists() or not receipt_path.exists():
        return False
    status, receipt = base.read(status_path), base.read(receipt_path)
    if status.get("state") != "completed" or status.get("phase") != "train_only_complete":
        return False
    if receipt.get("contexts") != 30 or receipt.get("train_evaluations") != 1500:
        raise ValueError("Invalid Train-only completion receipt")
    queue = base.read(output / "control/evaluation_queue.json")
    if len(queue) != 1500 or any(r["job"]["phase"] != "train_final" for r in queue):
        raise ValueError("Invalid predecessor evaluation coverage")
    for item in queue:
        row = study.training.validate_result(item)
        if type(row["success"]) is not bool:
            raise ValueError("Missing official success in predecessor")
    scores = base.read(output / "train_final_by_seed.json")
    if len(scores) != 30 or any(r["tasks"] != 50 for r in scores):
        raise ValueError("Predecessor Train results are incomplete")
    return True


def emit(output, phase, **fields):
    state = {"phase": phase, "updated_at": base.now(), "pid": os.getpid(), **fields}
    base.write(output / "chain_schedule/status.json", state)
    with (output / "chain_schedule/progress.log").open("a") as stream:
        stream.write(f"{base.now()} {phase} {fields}\n")


def controller_command(output):
    plan = base.read(output / "plan.json")
    return [sys.executable, "-u", str(Path(study.__file__)), "--output", str(output),
            "--model", plan["model"], "--revision", plan["model_revision"]]


def launch_controller(output):
    command = controller_command(output)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    with (output / "control/controller.log").open("a") as log:
        child = subprocess.Popen(command, cwd=study.ROOT, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    base.write(output / "control/launcher.json", {"pid": child.pid, "command": command, "started_at": base.now()})
    return child.pid


def monitor(output, schedule):
    """Bounded recovery of dead controllers; STOP always prevents a restart or a handoff."""
    recoveries, last_check = 0, 0.0
    while True:
        if (output / "STOP").exists() or (schedule / "STOP").exists():
            raise InterruptedError("User STOP marker prevents automatic continuation")
        if predecessor_complete(output):
            return
        process = output / "control/process.json"
        state = base.read(output / "status.json") if (output / "status.json").exists() else {}
        if process.exists() and not pid_alive(base.read(process)["pid"]):
            message = state.get("error", "")
            if recoveries < 3 and (state.get("state") != "error" or study.training.retryable(message)):
                recoveries += 1
                emit(schedule, "recovering_controller", experiment=str(output), attempt=recoveries,
                     previous_state=state.get("state"), error=message)
                launch_controller(output)
                time.sleep(5)
            else:
                emit(schedule, "predecessor_needs_debugging", experiment=str(output), error=message,
                     completed=state.get("completed"), total=state.get("total"))
                raise RuntimeError(f"Controller requires debugging: {output}: {message}")
        checked_at = time.monotonic()
        if checked_at - last_check >= CHECK_INTERVAL_SECONDS:
            protocol = base.read(output / "thinking/reflection/protocol.json")
            endpoint = base.check_endpoint(protocol)
            emit(schedule, "two_hour_check", experiment=str(output), state=state,
                 server_pid=endpoint["server"]["pid"], recoveries=recoveries)
            last_check = checked_at
        time.sleep(30)


def start_server(output, predecessor):
    plan = base.read(output / "plan.json")
    protocol = base.read(output / "thinking/reflection/protocol.json")
    previous = base.read(predecessor / "serving/status.json")
    pid = previous["server_pid"]
    if pid_alive(pid):
        process = Path(f"/proc/{pid}")
        command = process.joinpath("cmdline").read_bytes().split(b"\0")
        environment = process.joinpath("environ").read_bytes().split(b"\0")
        if (previous["model"].encode() not in command or b"8006" not in command
                or b"CUDA_VISIBLE_DEVICES=1" not in environment):
            raise RuntimeError("Predecessor server identity mismatch")
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 120
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(2)
        if pid_alive(pid):
            raise RuntimeError("Predecessor server did not stop")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8006)) == 0:
            raise RuntimeError("Port8006 is occupied after predecessor shutdown")
    with (output / "serving/vllm.log").open("a") as log:
        child = subprocess.Popen(["bash", str(output / "serving/serve.sh")], cwd=output,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    base.write(output / "serving/status.json", {
        "state": "starting", "model": plan["model"], "model_revision": plan["model_revision"],
        "server_pid": child.pid, "physical_gpu": 1, "port": 8006, "started_at": base.now(),
    })
    deadline = time.monotonic() + 900
    while pid_alive(child.pid) and time.monotonic() < deadline:
        try:
            endpoint = base.check_endpoint(protocol)
        except (OSError, ValueError):
            time.sleep(5)
            continue
        if endpoint["server"]["pid"] != child.pid:
            raise RuntimeError("New server PID mismatch")
        base.write(output / "serving/endpoint_verification.json", endpoint)
        base.write(output / "serving/status.json", {**base.read(output / "serving/status.json"), "state": "ready"})
        return
    raise RuntimeError("Gemma server startup failed; see serving/vllm.log")


def run(output, predecessor):
    plan = base.read(output / "plan.json")
    emit(output, "waiting_for_gemma26_train_completion", predecessor=str(predecessor),
         next_model=plan["model"], revision=plan["model_revision"])
    monitor(predecessor, output)
    for path, expected in base.read(output / "input_hashes.json").items():
        if study.training.digest(Path(path)) != expected:
            raise ValueError(f"Scheduled Gemma input changed: {path}")
    attempts = 0
    while not checkpoint_ready(plan["model"], plan["model_revision"]):
        if (output / "STOP").exists():
            raise InterruptedError("STOP requested")
        process = output / "serving/download_process.json"
        if not process.exists() or not pid_alive(base.read(process)["pid"]):
            if attempts >= 3:
                raise RuntimeError("Gemma31 checkpoint download incomplete after bounded retries")
            command = [str(Path.home() / ".vllm/bin/hf"), "download", plan["model"],
                       "--revision", plan["model_revision"], "--exclude", "original/*", "--exclude", "*.gguf"]
            with (output / "serving/download.log").open("a") as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            base.write(process, {"pid": child.pid, "command": command, "started_at": base.now()})
            attempts += 1
        time.sleep(30)
    emit(output, "starting_gemma31_server")
    start_server(output, predecessor)
    pid = launch_controller(output)
    emit(output, "gemma31_started", controller_pid=pid)
    monitor(output, output)
    emit(output, "chain_completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    (output / "chain_schedule").mkdir(exist_ok=True)
    with FileLock(str(output / "chain_schedule/scheduler.lock"), timeout=0):
        base.write(output / "chain_schedule/process.json", {"pid": os.getpid(), "command": sys.argv,
                                                          "started_at": base.now()})
        try:
            run(output, args.predecessor.resolve())
        except Exception as exc:
            emit(output, "error", error=repr(exc))
            study.accumulation.journal(output, "blocked", f"Gemma 순차 스케줄러 중단: {exc}")
            raise


if __name__ == "__main__":
    main()
