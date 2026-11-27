# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Start an explicitly owned GPU0 serving variant for a controlled comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.schedule_qwen_pause import identify, read, write  # noqa: E402


def command_with_batch(command, tokens):
    if tokens not in (8192, 16384, 65536):
        raise ValueError("Choose a measured candidate token batch")
    result = list(command)
    result[result.index("--max-num-batched-tokens") + 1] = str(tokens)
    return result


def install(output, tokens):
    """Persist validated BF16 tiles separately from the installed vLLM package."""
    plan = read(output / "maintenance_plan.json")
    if not (output / "kernel_tuning_complete.json").exists():
        raise ValueError("Refuse to deploy incomplete kernel tuning")
    script = Path("/home/thskadud/vllm_serve_qwen.sh")
    original = script.read_text()
    if hashlib.sha256(script.read_bytes()).hexdigest() != plan["original_serve_script_sha256"]:
        raise ValueError("Serving script changed outside this maintenance; reconcile instead of overwriting")
    destination = Path.home() / ".config/vllm/moe" / (
        "qwen36_35b_bf16_h200_" + output.name.removeprefix("qwen_h200_tuning_")
    )
    destination.mkdir(parents=True, exist_ok=False)
    source = output / "configs/E=256,N=512,device_name=NVIDIA_H200.json"
    shutil.copy2(source, destination / source.name)
    changed = original.replace("--max-num-batched-tokens 65536", f"--max-num-batched-tokens {tokens}")
    anchor = "export CUDA_VISIBLE_DEVICES=$GPU_NOS\n"
    if original.count(anchor) != 1:
        raise ValueError("Unexpected GPU selection syntax; do not rewrite an unknown serving script")
    changed = changed.replace(anchor, anchor + (
        "# Measured BF16 MoE tiles for this Qwen/H200; weights and KV precision are unchanged.\n"
        f"export VLLM_TUNED_CONFIG_FOLDER={shlex.quote(str(destination))}\n"
        'export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"\n'
    ))
    temporary = script.with_name(script.name + ".h200_tuning.tmp")
    temporary.write_text(changed)
    os.chmod(temporary, script.stat().st_mode)
    subprocess.run(["bash", "-n", str(temporary)], check=True)
    temporary.replace(script)
    write(output / "installed.json", {"tokens": tokens, "config_dir": str(destination),
                                      "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                                      "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    return destination


def start(output, name, tokens, tuned, final_log=False, config_dir=None):
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", 8006)) == 0:
            raise ValueError("Port8006 is occupied; refuse to start another model")
    plan = read(output / "maintenance_plan.json")
    environment = read(output / "private_server_environment.json")
    if environment.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("The preserved service must be restricted to physical GPU0")
    if tuned:
        if not (output / "kernel_tuning_complete.json").exists():
            raise ValueError("MoE numerical validation and tuning must complete before serving")
        environment["VLLM_TUNED_CONFIG_FOLDER"] = str(config_dir or output / "configs")
    else:
        environment.pop("VLLM_TUNED_CONFIG_FOLDER", None)
    command = command_with_batch(plan["server"]["command"], tokens)
    logfile = Path("/home/thskadud/output_nohup_vllm_server_qwen_gpu.log") if final_log else (
        output / f"server_{name}.log")
    with logfile.open("ab") as log:
        process = subprocess.Popen(command, cwd=output, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": process.pid, "command": command, "tokens": tokens, "tuned": tuned,
              "vllm_log": str(logfile), "gpu": 0}
    write(output / f"server_process_{name}.json", record)
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Serving variant exited: {logfile}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8006/v1/models", timeout=3) as response:
                models = json.load(response)
            if any(m["id"] == "Qwen/Qwen3.6-35B-A3B" for m in models["data"]):
                identity = identify(process.pid)
                write(output / f"server_process_{name}.json", {**record, **identity})
                print(f"READY {name}: batch={tokens}, tuned={tuned}, pid={process.pid}", flush=True)
                return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Serving variant startup exceeded deadline: {logfile}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--tuned", action="store_true")
    p.add_argument("--final-log", action="store_true")
    args = p.parse_args()
    start(args.output.resolve(), args.name, args.tokens, args.tuned, args.final_log)
