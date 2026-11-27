# SPDX-License-Identifier: Apache-2.0
"""Prepare Qwen3.5-9B and hand GPU 1 over after the Qwen3.6 ACE study."""

import argparse
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from filelock import FileLock

from scripts.ace import schedule_qwen36_add_only as study

base = study.base
MODEL = "Qwen/Qwen3.5-9B"
PRIOR = study.ROOT / "outputs/ace_appworld_qwen36_add_only_20261118"
HF = Path.home() / ".vllm/bin/hf"
SERVING_FILES = ("serve.sh", "chat_template.jinja", "upstream_chat_template.jinja", "model_revision.txt")


def publish(path, state, **details):
    base.write(path, {"state": state, "pid": os.getpid(), "updated_at": base.now(), **details})


def pid_alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def prior_complete():
    status = base.read(PRIOR / "status.json")
    if status["state"] != "completed":
        return False
    evaluations = PRIOR / "advantage/evaluations"
    summary = base.read(evaluations / "summary.json")
    assert len(summary) == 24
    n_bwt = sum(row["bwt_tasks"] for row in summary.values())
    for name, expected in (("train_final", 1200), ("bwt_checkpoint", n_bwt), ("val", 1200)):
        assert len(base.read(evaluations / f"{name}_results.json")) == expected
    return True


def prefetch(serving, revision):
    command = [str(HF), "download", MODEL, "--revision", revision]
    with (serving / "download.log").open("a") as log:
        for _ in range(3):
            if subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode == 0:
                return
            time.sleep(15)
    raise RuntimeError("Qwen3.5 checkpoint download failed. See serving/download.log")


def ensure_server(output, serving, handoff):
    protocol = base.read(output / "thinking/reflection/protocol.json")
    if handoff.exists():
        saved = base.read(handoff)
        if saved["state"] in {"starting", "ready"} and pid_alive(saved["server_pid"]):
            return ready(protocol, saved["server_pid"])
    old = base.read(study.HANDOFF)
    if old["model"] != study.MODEL:
        raise RuntimeError("Unexpected predecessor endpoint identity")
    pid = old["server_pid"]
    if pid_alive(pid):
        proc = Path(f"/proc/{pid}")
        args = proc.joinpath("cmdline").read_bytes().split(b"\0")
        env = proc.joinpath("environ").read_bytes().split(b"\0")
        if study.MODEL.encode() not in args or b"8006" not in args or b"CUDA_VISIBLE_DEVICES=1" not in env:
            raise RuntimeError("Qwen3.6 server identity could not be verified")
        os.kill(pid, signal.SIGTERM)
        for _ in range(90):
            if not pid_alive(pid):
                break
            time.sleep(2)
        else:
            raise RuntimeError("Qwen3.6 server did not shut down")
    with (serving / "vllm.log").open("a") as log:
        child = subprocess.Popen(
            ["bash", str(serving / "serve.sh")],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    publish(handoff, "starting", model=MODEL, server_pid=child.pid)
    return ready(protocol, child.pid)


def ready(protocol, pid):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline and pid_alive(pid):
        try:
            endpoint = base.check_endpoint(protocol)
        except (OSError, ValueError):
            time.sleep(5)
            continue
        if endpoint["server"]["pid"] != pid:
            raise RuntimeError("Unexpected server PID on 8006")
        return endpoint
    raise RuntimeError("Qwen3.5 endpoint startup failed. See serving/vllm.log")


def canary(output):
    import json

    checks = {}
    for mode in ("thinking", "nonthinking"):
        protocol = base.read(output / mode / "reflection/protocol.json")
        settings = protocol["model_settings"]
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Multiply 47 by 83. Give the final number."}],
            "max_tokens": 4096,
            **{key: settings[key] for key in ("temperature", "top_p", "presence_penalty")},
            **settings["extra_body"],
        }
        request = urllib.request.Request(
            protocol["base_url"] + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.load(response)
        message = result["choices"][0]["message"]
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if bool(reasoning) != (mode == "thinking") or not re.search(r"\b3,?901\b", message.get("content") or ""):
            raise RuntimeError(f"{mode} generation canary did not meet the expected mode or answer")
        checks[mode] = {"request": payload, "response": result}
    return checks


def run(output, serving, handoff, *, prepare_only):
    hashes = {name: base.digest((serving / name).read_bytes()) for name in SERVING_FILES}
    revision = (serving / "model_revision.txt").read_text().strip()
    assert re.fullmatch(r"[a-f0-9]{40}", revision)
    plan_path = output / "plan.json"
    if not plan_path.exists():
        study.prepare(output, model=MODEL, handoff=handoff)
        base.write(
            plan_path,
            {
                **base.read(plan_path),
                "predecessor": str(PRIOR),
                "model_revision": revision,
                "serving_sha256": hashes,
                "gpu_physical_id": 1,
                "port": 8006,
            },
        )
    plan = base.read(plan_path)
    if plan["serving_sha256"] != hashes:
        raise RuntimeError("Pinned serving files changed")
    if prepare_only:
        return
    status = output / "status.json"
    publish(status, "scheduled", phase="prefetch_checkpoint")
    if not (serving / "download_completed.json").exists():
        prefetch(serving, revision)
        publish(serving / "download_completed.json", "completed", model=MODEL, revision=revision)
    while not prior_complete():
        publish(
            status,
            "scheduled",
            phase="waiting_for_qwen36_completion",
            predecessor=base.read(PRIOR / "status.json")["state"],
        )
        time.sleep(30)
    publish(status, "running", phase="starting_qwen35_server")
    endpoint = ensure_server(output, serving, handoff)
    base.write(serving / "endpoint_verification.json", endpoint)
    base.write(serving / "generation_canaries.json", canary(output))
    publish(handoff, "ready", model=MODEL, server_pid=endpoint["server"]["pid"], port=8006, physical_gpu=1)
    study.run(output, model=MODEL, handoff_path=handoff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    serving = output / "serving"
    with FileLock(str(output / "scheduler.lock"), timeout=0):
        try:
            run(output, serving, serving / "status.json", prepare_only=args.prepare_only)
        except Exception as exc:
            previous = output / "status.json"
            phase = base.read(previous).get("phase", "prepare") if previous.exists() else "prepare"
            publish(previous, "error", phase=phase, error=str(exc))
            if not args.prepare_only:
                study.accumulation.journal(output, "blocked", f"Qwen3.5 실험 중단: {phase}: {exc}")
            raise
