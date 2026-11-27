# SPDX-License-Identifier: Apache-2.0
"""Resume the preserved Gemma cohort on the verified 100-sequence server and record a short probe."""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from scripts.ace import run_gemma4_train_only as study  # noqa: E402

OUT = ROOT / "outputs/ace_appworld_gemma4_26b_a4b_it_wo_gt_train_only_20261124"
NEXT = ROOT / "outputs/ace_appworld_gemma4_31b_it_wo_gt_train_only_20261124"
AUDIT = OUT / "control/concurrency100_20261124"


def metrics():
    with urllib.request.urlopen("http://127.0.0.1:8006/metrics", timeout=10) as response:
        raw = response.read().decode()
    names = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "num_preemptions_total",
             "generation_tokens_total", "e2e_request_latency_seconds_count", "e2e_request_latency_seconds_sum",
             "request_queue_time_seconds_count", "request_queue_time_seconds_sum")
    values = {"monotonic": time.monotonic(), "at": study.base.now()}
    for name in names:
        values[name] = sum(float(line.rsplit(" ", 1)[-1]) for line in raw.splitlines()
                           if line.startswith("vllm:" + name + "{"))
    return values


def main():
    protocol = study.base.read(OUT / "thinking/reflection/protocol.json")
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        serving = study.base.read(OUT / "serving/status.json")
        if serving.get("max_num_seqs") != 100:
            time.sleep(3)
            continue
        try:
            endpoint = study.base.check_endpoint(protocol)
        except (OSError, ValueError):
            time.sleep(5)
            continue
        assert endpoint["server"]["pid"] == serving["server_pid"]
        process = psutil.Process(serving["server_pid"])
        args = process.cmdline()
        assert args[args.index("--max-num-seqs") + 1] == "100"
        assert process.environ()["CUDA_VISIBLE_DEVICES"] == "1"
        break
    else:
        raise RuntimeError("Server did not become ready with max_num_seqs100")
    study.base.write(OUT / "serving/endpoint_verification_concurrency100.json", endpoint)
    study.base.write(OUT / "serving/status.json", {**serving, "state": "ready", "ready_at": study.base.now()})
    hashes = study.base.read(AUDIT / "preserved_after_drain_hashes.json")
    for path, sha in hashes.items():
        assert study.training.digest(Path(path)) == sha
    for path, sha in study.base.read(OUT / "input_hashes.json").items():
        assert study.training.digest(Path(path)) == sha
    marker = OUT / "STOP"
    if not marker.read_text().startswith("Temporary graceful drain for user-approved concurrency"):
        raise InterruptedError("STOP marker was changed; user cancellation takes precedence")
    marker.unlink()
    command = study.base.read(OUT / "control/launcher.json")["command"]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    with (OUT / "control/controller.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    study.base.write(OUT / "control/launcher.json", {
        "pid": child.pid, "command": command, "started_at": study.base.now(),
    })
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        status = study.base.read(OUT / "status.json")
        if status.get("pid") == child.pid and status.get("worker_limit") == 100 and status.get("running") == 100:
            break
        if child.poll() is not None:
            raise RuntimeError("Controller exited during concurrency100 startup")
        time.sleep(2)
    else:
        raise RuntimeError("Controller failed to fill the 100-worker queue")
    migration = study.base.read(AUDIT / "migration.json")
    scheduler = psutil.Process(migration["paused_scheduler_pid"])
    assert "schedule_gemma4_train_only.py" in " ".join(scheduler.cmdline())
    scheduler.send_signal(signal.SIGCONT)
    migration.update(state="running", controller_pid=child.pid, worker_limit=100,
                     verified_running_workers=status["running"], scheduler_resumed_at=study.base.now())
    study.base.write(AUDIT / "migration.json", migration)
    study.base.write(OUT / "runtime_settings.json", {
        "state": "running", "workers": 100, "server_max_num_seqs": 100,
        "server_pid": serving["server_pid"], "controller_pid": child.pid, "next_model_workers": 50,
        "updated_at": study.base.now(),
    })
    print(json.dumps({"resumed": True, "controller_pid": child.pid, "server_pid": serving["server_pid"],
                      "preserved_results": len(hashes), "queue_running": status["running"]}), flush=True)
    warmup_deadline = time.monotonic() + 150
    while time.monotonic() < warmup_deadline:
        sample = metrics()
        if sample["num_requests_running"] >= 90 and sample["generation_tokens_total"] >= 5000:
            break
        time.sleep(5)
    samples = [metrics()]
    for _ in range(6):
        time.sleep(5)
        samples.append(metrics())
    first, last = samples[0], samples[-1]
    seconds = last["monotonic"] - first["monotonic"]
    completed = last["e2e_request_latency_seconds_count"] - first["e2e_request_latency_seconds_count"]
    result = {
        "samples": samples, "duration_seconds": seconds,
        "kv_cache_min_percent": 100 * min(s["kv_cache_usage_perc"] for s in samples),
        "kv_cache_max_percent": 100 * max(s["kv_cache_usage_perc"] for s in samples),
        "generation_tokens_per_second": (last["generation_tokens_total"] - first["generation_tokens_total"]) / seconds,
        "completed_requests": completed,
        "mean_completed_request_latency_seconds": (
            (last["e2e_request_latency_seconds_sum"] - first["e2e_request_latency_seconds_sum"]) / completed
            if completed else None),
        "preemptions_during_probe": last["num_preemptions_total"] - first["num_preemptions_total"],
        "comparison_limit": "A short observational window; different request lengths and reset prefix cache",
    }
    study.base.write(AUDIT / "metrics_after_30s.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        study.base.write(AUDIT / "migration_error.json", {"at": study.base.now(), "error": repr(exc)})
        raise
