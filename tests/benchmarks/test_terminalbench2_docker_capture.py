# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Verify finite Docker observation transport, stream decoding and task ownership."""
from __future__ import annotations

import json
import socketserver
import struct
import threading
from http.server import BaseHTTPRequestHandler

import pytest

from exgentic.benchmarks.terminalbench2 import docker_capture


@pytest.mark.parametrize("command", [
    "tmux send-keys -t agent rm", "tmux capture-pane -p -t agent; touch /changed",
    "tmux capture-pane -p -S - -t agent && echo changed", "tmux capture-pane -p -t $(command)",
])
def test_mutating_or_unrecognized_commands_never_open_a_connection(command, monkeypatch):
    monkeypatch.setattr(docker_capture, "docker_socket", lambda: pytest.fail("not a read-only capture"))
    assert docker_capture.capture_arguments(command) is None
    with pytest.raises(ValueError, match="read-only"):
        docker_capture.capture_pane("owned", command)


@pytest.mark.parametrize("wrong_owner", [False, True])
def test_capture_uses_finite_noninteractive_http_and_checks_exact_owner(tmp_path, monkeypatch, wrong_owner):
    path = tmp_path / "docker.sock"
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            calls.append(("GET", self.path, None))
            if self.path.startswith("/containers/json?"):
                data = [{"Id": "container", "State": "running", "Labels": {
                    "com.docker.compose.project": "other" if wrong_owner else "owned",
                    "com.docker.compose.service": "main",
                }}]
            else:
                assert self.path == "/exec/command/json"
                data = {"Running": False, "ExitCode": 0}
            self.reply(json.dumps(data).encode())

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(("POST", self.path, data))
            assert self.headers.get("Upgrade") is None
            if self.path == "/containers/container/exec":
                assert data["AttachStdin"] is False and data["User"] == "1000"
                assert data["Cmd"] == ["tmux", "capture-pane", "-p", "-S", "-", "-t", "agent"]
                self.reply(b'{"Id":"command"}', status=201)
            else:
                assert self.path == "/exec/command/start"
                assert data == {"Detach": False, "Tty": False}
                frames = b"".join(bytes([stream, 0, 0, 0]) + struct.pack(">I", len(text)) + text
                                  for stream, text in [(1, b"fresh screen"), (2, b"diagnostic")])
                self.reply(frames)

        def reply(self, raw, status=200):
            self.send_response(status)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

    monkeypatch.setattr(docker_capture, "docker_socket", lambda: str(path))
    with socketserver.UnixStreamServer(str(path), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            if wrong_owner:
                with pytest.raises(RuntimeError, match="exactly one owned"):
                    docker_capture.capture_pane("owned", "tmux capture-pane -p -S - -t agent", user=1000)
                assert len(calls) == 1
            else:
                result = docker_capture.capture_pane("owned", "tmux capture-pane -p -S - -t agent", user=1000)
                assert result == {"stdout": "fresh screen", "stderr": "diagnostic", "return_code": 0}
                assert len(calls) == 4
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_frozen_snapshot_resolves_existing_forward_without_checkout_scripts(monkeypatch):
    import hashlib
    import socket
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="capture-") as temporary:
        home = Path(temporary)
        args = ["-p", "2222", "-o ConnectTimeout=30", "-T", "--", "user@fixture"]
        key = hashlib.sha256("\0".join(args).encode()).hexdigest()[:12]
        endpoint = home / ".cache/exgentic/ssh_pool_http_close" / key / "docker.sock"
        endpoint.parent.mkdir(parents=True)
        monkeypatch.setenv("DOCKER_HOST", "ssh://user@fixture:2222")
        monkeypatch.setattr(docker_capture.Path, "home", lambda: home)
        monkeypatch.setattr(docker_capture, "__file__", str(home / "frozen/src/docker_capture.py"))
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(endpoint))
            assert docker_capture.docker_socket() == str(endpoint)
        endpoint.unlink()
        with pytest.raises(RuntimeError, match="existing Docker SSH forward"):
            docker_capture.docker_socket()
