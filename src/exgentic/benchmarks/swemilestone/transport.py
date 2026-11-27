# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Run JSON requests on the Docker host, avoiding remote bind-mount mismatches."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from .swemilestone_benchmark import BENCHMARK_VERSION, DATA_COMMIT, HARNESS_COMMIT, REPOSITORIES

# Only code and JSON are sent to SSH; credentials remain with the user's SSH client.
_LAUNCHER = """
import hashlib, json, os, pathlib, subprocess, sys
request = json.load(sys.stdin)
source = request.pop("worker_source")
root = pathlib.Path(request.get("backend_root") or "~/.exgentic/benchmarks/swemilestone").expanduser().resolve()
request["backend_root"] = str(root)
workers = root / "workers"
workers.mkdir(parents=True, exist_ok=True)
path = workers / (hashlib.sha256(source.encode()).hexdigest() + ".py")
if not path.exists():
    temporary = path.with_suffix(".tmp-" + str(os.getpid()))
    temporary.write_text(source)
    temporary.replace(path)
python = root / "runtime" / "bin" / "python"
result = subprocess.run([str(python) if python.exists() else sys.executable, str(path)],
                        input=json.dumps(request), text=True)
sys.exit(result.returncode)
"""


def ssh_command(docker_host: str, command: list[str]) -> list[str]:
    """Use OpenSSH config aliases, explicit users, ports, and IPv6 safely."""
    url = urlsplit(docker_host)
    if url.scheme != "ssh" or not url.hostname or url.password or url.path not in ("", "/"):
        raise ValueError("Remote SWE-Milestone requires DOCKER_HOST=ssh://[user@]host[:port]")
    if url.query or url.fragment or url.hostname.startswith("-") or (url.username and url.username.startswith("-")):
        raise ValueError("Invalid SSH Docker endpoint")
    destination = f"{url.username}@{url.hostname}" if url.username else url.hostname
    options = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    if url.port is not None:
        options.extend(["-p", str(url.port)])
    return [*options, destination, shlex.join(command)]


class MilestoneBackend:
    def __init__(self, docker_host: str | None = None, backend_root: str | None = None):
        self.docker_host = docker_host or os.environ.get("DOCKER_HOST")
        self.backend_root = backend_root
        if self.docker_host and not self.docker_host.startswith("ssh://"):
            raise ValueError("Use an SSH Docker endpoint, or unset DOCKER_HOST to run on the local Docker host")

    def call(self, operation: str, *, timeout: int = 3600, **kwargs):
        worker = Path(__file__).with_name("worker.py").read_text(encoding="utf-8")
        request = {
            **kwargs,
            "operation": operation,
            "backend_root": self.backend_root,
            "harness_commit": HARNESS_COMMIT,
            "data_commit": DATA_COMMIT,
            "benchmark_version": BENCHMARK_VERSION,
            "repositories": REPOSITORIES,
            "worker_source": worker,
        }
        command = ["python3" if self.docker_host else sys.executable, "-c", _LAUNCHER]
        if self.docker_host:
            command = ssh_command(self.docker_host, command)
        deadline = time.monotonic() + timeout
        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            result = subprocess.run(
                command, input=json.dumps(request), capture_output=True, text=True, timeout=remaining
            )
            # Identification/key-exchange failures occur before the remote command starts.
            # Retrying other disconnects could execute a mutating command twice.
            handshake_failure = (
                self.docker_host
                and result.returncode == 255
                and not result.stdout.strip()
                and any(marker in result.stderr for marker in (
                    "kex_exchange_identification:", "ssh_exchange_identification:",
                ))
            )
            if not handshake_failure or attempt == 2:
                break
            time.sleep(min(0.5 * 2**attempt, max(0, deadline - time.monotonic())))
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            detail = (result.stderr or result.stdout)[-4000:]
            raise RuntimeError(f"SWE-Milestone backend failed ({result.returncode}): {detail}") from exc
        if result.returncode or payload.get("error"):
            raise RuntimeError(f"SWE-Milestone backend: {payload.get('error') or result.stderr[-4000:]}")
        return payload["result"]

    def namespace(self, run_root: str, stream_id: str | None) -> str:
        return hashlib.sha256(f"{run_root}:{stream_id or 'default'}".encode()).hexdigest()[:20]
