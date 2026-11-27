#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Relay one Docker HTTP request over SSH without retaining an idle pool channel."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
from contextlib import nullcontext


def close_idle_connection(header):
    lines = header.split(b"\r\n")
    fields = [line.split(b":", 1) for line in lines[1:] if b":" in line]
    if any(key.strip().lower() == b"upgrade" for key, _ in fields):
        return header  # Exec/attach hijacking must retain its duplex stream.
    lines = [lines[0], *(line for line in lines[1:] if not line.lower().startswith(b"connection:"))]
    return b"\r\n".join([*lines, b"Connection: close"])


def relay_stream(input_fd, output_fd, daemon_socket):
    """Relay a local forwarded stream, including exec's intentional stdin half-close."""
    def upload():
        try:
            while chunk := os.read(input_fd, 65536):
                daemon_socket.sendall(chunk)
        except OSError:
            pass
        finally:
            # Noninteractive Docker exec closes stdin before reading stdout.
            # Preserve the daemon-to-client direction until its response ends.
            try:
                daemon_socket.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    threading.Thread(target=upload, daemon=True).start()
    try:
        while chunk := daemon_socket.recv(65536):
            view = memoryview(chunk)
            while view:
                view = view[os.write(output_fd, view):]
    except (BrokenPipeError, ConnectionResetError):
        pass


def relay(input_fd, output_fd, daemon_socket, gate=None):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = os.read(input_fd, 65536)
        if not chunk:
            return
        data += chunk
        if len(data.split(b"\r\n\r\n", 1)[0]) > 131072:
            raise ValueError("Docker HTTP headers exceed 128 KiB")
    header, body = data.split(b"\r\n\r\n", 1)
    with gate(header) if gate is not None else nullcontext():
        daemon_socket.sendall(close_idle_connection(header) + b"\r\n\r\n" + body)
        relay_stream(input_fd, output_fd, daemon_socket)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default="/var/run/docker.sock")
    args = parser.parse_args()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as daemon:
        daemon.connect(args.socket)
        relay(sys.stdin.fileno(), sys.stdout.fileno(), daemon)


if __name__ == "__main__":
    main()
