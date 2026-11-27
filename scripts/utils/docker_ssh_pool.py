#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Forward Docker over a private Unix socket without consuming SSH exec sessions.

Install a symlink named ``ssh`` in the experiment's existing PATH. Other SSH
commands pass straight through to the system client. Docker dial-stdio streams
use SSH's direct-streamlocal channels, which do not consume MaxSessions slots.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import random
import re
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

SYSTEM_SSH = "/usr/bin/ssh"
MASTERS = 64
CHANNELS = 8


def docker_destination(arguments):
    if arguments[-3:] == ["docker", "system", "dial-stdio"]:
        return arguments[:-3]
    if arguments and arguments[-1] == "docker system dial-stdio":
        return arguments[:-1]
    return None


@contextmanager
def channel(root, *, masters=MASTERS, channels=CHANNELS):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    start = random.randrange(masters * channels)
    pause = 0.05
    while True:
        for offset in range(masters * channels):
            index = (start + offset) % (masters * channels)
            stream = (root / f"channel{index:03d}.lock").open("a")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            try:
                yield index // channels
            finally:
                stream.close()
            return
        time.sleep(pause + random.uniform(0, pause / 4))
        pause = min(0.5, pause * 1.5)


def master_options(socket):
    return ["-o", f"ControlPath={socket}", "-o", "ControlMaster=auto", "-o", "ControlPersist=600",
            "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"]


def ensure_master(root, destination, socket):
    options = master_options(socket)
    check = [SYSTEM_SSH, *options, "-O", "check", *destination]
    if subprocess.run(check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=35).returncode == 0:
        return options
    with (root.parent / "connect.lock").open("a") as lock:
        # Serialize authentication, also avoiding sshd's unauthenticated-connection limit.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if subprocess.run(check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=35).returncode:
            subprocess.run([SYSTEM_SSH, *options, "-M", "-N", "-f", *destination],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=True, timeout=40)
    return options


def forward_root(destination):
    key = hashlib.sha256("\0".join(destination).encode()).hexdigest()[:12]
    return Path.home() / ".cache/exgentic/ssh_pool_http_close" / key


def socket_available(path):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(str(path))
        return True
    except TimeoutError:
        # A full listener backlog is busy, not stale. Unlinking it would leave
        # an established SSH forward registered with no reachable socket path.
        return path.exists()
    except OSError as exc:
        if exc.errno in {errno.EAGAIN, errno.EINPROGRESS, errno.ETIMEDOUT}:
            return path.exists()
        return False


def ensure_forward(destination):
    root = forward_root(destination)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    endpoint = root / "docker.sock"
    if socket_available(endpoint):
        return endpoint
    with (root / "forward.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if socket_available(endpoint):
            return endpoint
        # Reuse an authenticated master. Forwarding opens direct-streamlocal
        # channels, not shell sessions, even for Docker's hijacked exec streams.
        options = ensure_master(root, destination, root / "m63.sock")
        # OpenSSH deduplicates an existing forwarding specification even if its
        # Unix path disappeared. Explicitly cancel that listener before rebinding.
        # Established forwarded connections are independent of the listener.
        subprocess.run([SYSTEM_SSH, *options, "-O", "cancel", "-L",
                        f"{endpoint}:/var/run/docker.sock", *destination],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False, timeout=35)
        endpoint.unlink(missing_ok=True)
        subprocess.run([SYSTEM_SSH, *options, "-O", "forward", "-L",
                        f"{endpoint}:/var/run/docker.sock", *destination],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True, timeout=35)
        deadline = time.monotonic() + 2
        while not endpoint.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os.chmod(endpoint, 0o600)
        if not socket_available(endpoint):
            raise RuntimeError("Docker SSH Unix socket forwarding is unavailable")
    return endpoint


@contextmanager
def network_operation(root, header):
    """Apply backpressure before Docker's global bridge lock, not to actor requests."""
    method, path, _protocol = header.split(b"\r\n", 1)[0].split(b" ", 2)
    path = re.sub(rb"^/v[0-9.]+/", b"/", path)
    changes_bridge = ((method == b"POST" and path == b"/networks/create")
                      or (method == b"DELETE" and path.startswith(b"/networks/")))
    if not changes_bridge:
        yield
        return
    # Cancelled clients waiting here never enqueue an orphaned network create
    # inside dockerd. All model calls and container exec streams remain parallel.
    with (root / "network_operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    destination = docker_destination(arguments)
    if destination is None:
        os.execv(SYSTEM_SSH, [SYSTEM_SSH, *arguments])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from docker_stdio import relay

    endpoint = ensure_forward(destination)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(endpoint))
        relay(sys.stdin.fileno(), sys.stdout.fileno(), connection,
              gate=lambda header: network_operation(endpoint.parent, header))
    return 0


if __name__ == "__main__":
    sys.exit(main())
