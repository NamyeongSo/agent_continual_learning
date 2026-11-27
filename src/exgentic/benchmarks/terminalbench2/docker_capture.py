# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Recover a read-only tmux observation through a finite Docker HTTP response."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shlex
import socket
import struct
from pathlib import Path
from urllib.parse import quote, urlsplit


def capture_arguments(command: str) -> list[str] | None:
    """Accept Harbor's exact read-only capture form, never a shell command batch."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if (len(parts) not in (5, 7) or parts[:3] != ["tmux", "capture-pane", "-p"]
            or parts[-2] != "-t" or not re.fullmatch(r"[A-Za-z0-9_.:-]+", parts[-1])):
        return None
    if len(parts) == 7 and parts[3:5] != ["-S", "-"]:
        return None
    return parts


def docker_socket() -> str:
    host = urlsplit(os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock"))
    if host.scheme == "unix":
        return host.path
    if host.scheme != "ssh" or not host.hostname:
        raise ValueError("Terminal capture recovery requires the existing Unix or SSH Docker endpoint")
    destination = (host.username + "@" if host.username else "") + host.hostname
    args = (["-p", str(host.port)] if host.port else []) + ["-o ConnectTimeout=30", "-T", "--", destination]
    # The normal Compose request has already established this authenticated
    # forward. Resolve its existing socket without importing checkout scripts;
    # frozen benchmark snapshots need only this self-contained source module.
    key = hashlib.sha256("\0".join(args).encode()).hexdigest()[:12]
    endpoint = Path.home() / ".cache/exgentic/ssh_pool_http_close" / key / "docker.sock"
    if not endpoint.is_socket():
        raise RuntimeError("The experiment's existing Docker SSH forward is unavailable")
    return str(endpoint)



def capture_pane(project: str, command: str, *, user=None, cwd=None, env=None) -> dict:
    arguments = capture_arguments(command)
    if arguments is None:
        raise ValueError("Only a read-only tmux capture can use transport recovery")
    endpoint = docker_socket()

    class DockerConnection(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect(endpoint)

    def request(method, path, data=None):
        connection = DockerConnection("localhost", timeout=10)
        try:
            body = json.dumps(data).encode() if data is not None else None
            connection.request(method, path, body=body, headers={"Content-Type": "application/json",
                                                               "Connection": "close"})
            response = connection.getresponse()
            raw = response.read()
            if response.status not in (200, 201):
                raise RuntimeError(f"Docker terminal observation HTTP {response.status}")
            return raw
        finally:
            connection.close()

    filters = quote(json.dumps({"label": [f"com.docker.compose.project={project}",
                                          "com.docker.compose.service=main"]}))
    containers = json.loads(request("GET", "/containers/json?filters=" + filters))
    # Verify labels again so a mismatching daemon response can never select another task.
    containers = [row for row in containers if row.get("Labels", {}).get("com.docker.compose.project") == project
                  and row.get("Labels", {}).get("com.docker.compose.service") == "main"
                  and row.get("State") == "running"]
    if len(containers) != 1:
        raise RuntimeError("Cannot identify exactly one owned task container for terminal observation")
    payload = {"AttachStdin": False, "AttachStdout": True, "AttachStderr": True,
               "Tty": False, "Cmd": arguments}
    if user is not None:
        payload["User"] = str(user)
    if cwd:
        payload["WorkingDir"] = cwd
    if env:
        payload["Env"] = [f"{key}={value}" for key, value in env.items()]
    execution = json.loads(request("POST", f"/containers/{containers[0]['Id']}/exec", payload))["Id"]
    raw = request("POST", f"/exec/{execution}/start", {"Detach": False, "Tty": False})
    output = {1: bytearray(), 2: bytearray()}
    while raw:
        if len(raw) < 8:
            raise RuntimeError("Truncated Docker terminal observation header")
        stream, size = raw[0], struct.unpack(">I", raw[4:8])[0]
        if stream not in output or len(raw) < 8 + size:
            raise RuntimeError("Truncated Docker terminal observation output")
        output[stream].extend(raw[8:8 + size])
        raw = raw[8 + size:]
    status = json.loads(request("GET", f"/exec/{execution}/json"))
    if status.get("Running") or not isinstance(status.get("ExitCode"), int):
        raise RuntimeError("Docker terminal observation did not finish")
    return {"stdout": output[1].decode(errors="replace"), "stderr": output[2].decode(errors="replace"),
            "return_code": status["ExitCode"]}
