# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

from scripts.utils.docker_stdio import close_idle_connection, relay_stream


def test_only_regular_http_releases_idle_channels_and_upgrade_headers_are_preserved():
    request = b"POST /exec/id/start HTTP/1.1\r\nConnection: Upgrade\r\nUpgrade: tcp"
    assert close_idle_connection(request) == request
    regular = b"GET /version HTTP/1.1\r\nConnection: keep-alive\r\nHost: docker"
    assert close_idle_connection(regular) == b"GET /version HTTP/1.1\r\nHost: docker\r\nConnection: close"


def test_relay_exits_after_response_even_while_docker_client_keeps_stdin_open(tmp_path):
    path = tmp_path / "daemon.sock"
    received = []
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(path))
        server.listen(1)

        def respond():
            with server.accept()[0] as connection:
                data = b""
                while b"\r\n\r\n" not in data:
                    data += connection.recv(4096)
                received.append(data)
                connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK")

        thread = threading.Thread(target=respond, daemon=True)
        thread.start()
        script = Path(__file__).resolve().parents[2] / "scripts/utils/docker_stdio.py"
        process = subprocess.Popen([sys.executable, str(script), "--socket", str(path)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            process.stdin.write(b"GET /version HTTP/1.1\r\nHost: docker\r\n\r\n")
            process.stdin.flush()
            process.wait(timeout=5)
            assert process.returncode == 0
            assert process.stdout.read().endswith(b"OK")
            assert b"Connection: close" in received[0]
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()
            thread.join(timeout=5)


def test_exec_stdin_half_close_keeps_stdout_alive_until_command_finishes():
    input_fd, writer = os.pipe()
    reader, output_fd = os.pipe()
    local, daemon = socket.socketpair()
    thread = threading.Thread(target=relay_stream, args=(input_fd, output_fd, local), daemon=True)
    try:
        thread.start()
        os.write(writer, b"POST /exec/id/start HTTP/1.1\r\nUpgrade: tcp\r\n\r\n")
        daemon.settimeout(2)
        assert daemon.recv(4096).endswith(b"Upgrade: tcp\r\n\r\n")
        os.close(writer)
        writer = None
        assert daemon.recv(4096) == b""
        assert thread.is_alive()
        daemon.sendall(b"HTTP/1.1 101 UPGRADED\r\n\r\nhello")
        assert os.read(reader, 4096).endswith(b"hello")
        daemon.shutdown(socket.SHUT_WR)
        thread.join(timeout=3)
        assert not thread.is_alive()
    finally:
        if writer is not None:
            os.close(writer)
        local.close()
        daemon.close()
        for fd in (input_fd, reader, output_fd):
            os.close(fd)
