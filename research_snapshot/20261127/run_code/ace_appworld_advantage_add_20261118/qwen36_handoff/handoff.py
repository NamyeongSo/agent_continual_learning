import fcntl
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL = ROOT.parent / "evaluations"
OLD_PID = 5631
OLD = "Qwen/Qwen3.8-27B-FP8"
NEW = "Qwen/Qwen3.6-35B-A3B-FP8"


def status(state, **fields):
    row = {"state": state, "pid": os.getpid(), "at": datetime.now(timezone.utc).isoformat(), **fields}
    tmp = ROOT / "status.tmp"
    tmp.write_text(json.dumps(row, indent=2) + "\n")
    tmp.replace(ROOT / "status.json")
    print(json.dumps(row), flush=True)


def served_models():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8006/v1/models", timeout=5) as response:
            return {row["id"] for row in json.load(response)["data"]}
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return set()


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def run():
    while True:
        evaluation = json.loads((EVAL / "status.json").read_text())
        if evaluation["state"] == "completed":
            break
        status("waiting_for_evaluation", phase=evaluation["phase"], completed=evaluation["completed"],
               total=evaluation["total"])
        time.sleep(30)
    assert evaluation["completed"] == evaluation["total"] == 2982
    for phase, count in (("train_final", 1200), ("bwt_checkpoint", 582), ("val", 1200)):
        if len(json.loads((EVAL / f"{phase}_results.json").read_text())) != count:
            raise RuntimeError(f"Evaluation count mismatch: {phase}")
    if len(json.loads((EVAL / "summary.json").read_text())) != 24:
        raise RuntimeError("Missing Context summaries")
    models = served_models()
    if NEW in models:
        status("ready", model=NEW, port=8006)
        return
    if alive(OLD_PID):
        command = Path(f"/proc/{OLD_PID}/cmdline").read_bytes().split(b"\0")
        environment = Path(f"/proc/{OLD_PID}/environ").read_bytes().split(b"\0")
        if OLD.encode() not in command or b"8006" not in command or b"CUDA_VISIBLE_DEVICES=1" not in environment:
            raise RuntimeError("Original server identity could not be verified")
        status("stopping_qwen38", server_pid=OLD_PID)
        os.kill(OLD_PID, signal.SIGTERM)
        for _ in range(90):
            if not alive(OLD_PID):
                break
            time.sleep(2)
        else:
            raise RuntimeError("Original server did not exit after SIGTERM")
    if served_models():
        raise RuntimeError("Unexpected server on port 8006")
    result = subprocess.run(["bash", str(ROOT / "serve.sh")], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    match = re.search(r"vLLM started \(PID: (\d+)\)", result.stdout)
    if match is None:
        raise RuntimeError("Missing vLLM PID from launcher")
    pid = int(match[1])
    status("starting_qwen36", server_pid=pid, model=NEW)
    for _ in range(180):
        if NEW in served_models():
            environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            if b"CUDA_VISIBLE_DEVICES=1" not in environment:
                raise RuntimeError("New server GPU pin does not match")
            status("ready", server_pid=pid, model=NEW, port=8006, physical_gpu=1)
            return
        if not alive(pid):
            raise RuntimeError("Qwen3.6 exited before becoming ready. See vLLM log.")
        time.sleep(5)
    raise RuntimeError("Qwen3.6 readiness timed out. See vLLM log.")


with (ROOT / "handoff.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        run()
    except Exception as exc:
        status("error", error=str(exc))
        raise
