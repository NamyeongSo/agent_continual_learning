# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import fcntl

from scripts.utils.docker_ssh_pool import channel, docker_destination, ensure_master


def test_only_docker_stdio_uses_pool():
    assert docker_destination(["-T", "--", "fixture", "docker", "system", "dial-stdio"]) == ["-T", "--", "fixture"]
    assert docker_destination(["fixture", "docker system dial-stdio"]) == ["fixture"]
    assert docker_destination(["-G", "fixture"]) is None
    assert docker_destination(["fixture", "echo", "hello"]) is None


def test_channel_skips_saturated_master_and_releases_its_lock(tmp_path):
    with (tmp_path / "channel000.lock").open("a") as occupied:
        fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with channel(tmp_path, masters=2, channels=1) as index:
            assert index == 1
        with (tmp_path / "channel001.lock").open("a") as released:
            fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_master_is_established_only_when_missing(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1 if "check" in args else 0)

    root = tmp_path / "target"
    root.mkdir()
    monkeypatch.setattr("scripts.utils.docker_ssh_pool.subprocess.run", run)
    options = ensure_master(root, ["-T", "--", "fixture"], root / "m00.sock")
    assert len(calls) == 3 and "-N" in calls[2] and "-f" in calls[2]
    assert "ControlMaster=auto" in options
    calls.clear()
    monkeypatch.setattr("scripts.utils.docker_ssh_pool.subprocess.run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=0))
    ensure_master(root, ["-T", "--", "fixture"], root / "m00.sock")
    assert not calls


def test_forward_reuses_existing_socket_without_acquiring_an_exec_channel(tmp_path, monkeypatch):
    from scripts.utils import docker_ssh_pool as transport

    monkeypatch.setattr(transport, "forward_root", lambda destination: tmp_path)
    monkeypatch.setattr(transport, "socket_available", lambda path: True)
    monkeypatch.setattr(transport, "ensure_master", lambda *args: (_ for _ in ()).throw(AssertionError()))
    assert transport.ensure_forward(["fixture"]) == tmp_path / "docker.sock"


def test_forward_uses_streamlocal_control_request_not_remote_shell(tmp_path, monkeypatch):
    from scripts.utils import docker_ssh_pool as transport

    probes = iter([False, False, True])
    calls = []
    monkeypatch.setattr(transport, "forward_root", lambda destination: tmp_path)
    monkeypatch.setattr(transport, "socket_available", lambda path: next(probes))
    monkeypatch.setattr(transport, "ensure_master", lambda *args: ["-S", "master.sock"])
    monkeypatch.setattr(transport.os, "chmod", lambda *args: None)
    monkeypatch.setattr(transport.Path, "exists", lambda path: True)
    monkeypatch.setattr(transport.subprocess, "run", lambda args, **kwargs: calls.append(args))
    transport.ensure_forward(["fixture"])
    assert calls == [[transport.SYSTEM_SSH, "-S", "master.sock", "-O", operation, "-L",
                      f"{tmp_path / 'docker.sock'}:/var/run/docker.sock", "fixture"]
                     for operation in ("cancel", "forward")]


def test_saturated_unix_listener_is_not_mistaken_for_a_stale_forward(tmp_path, monkeypatch):
    from scripts.utils import docker_ssh_pool as transport

    endpoint = tmp_path / "docker.sock"
    endpoint.touch()

    class BusySocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            raise TimeoutError("listener backlog is full")

    monkeypatch.setattr(transport.socket, "socket", lambda *args: BusySocket())
    assert transport.socket_available(endpoint)


def test_only_bridge_mutations_acquire_backpressure_and_actor_exec_remains_parallel(tmp_path):
    import pytest

    from scripts.utils.docker_ssh_pool import network_operation

    for request in (b"POST /v1.53/networks/create HTTP/1.1", b"DELETE /v1.53/networks/owned HTTP/1.1"):
        with network_operation(tmp_path, request):
            with (tmp_path / "network_operation.lock").open("a") as second:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with network_operation(tmp_path, b"POST /v1.53/exec/actor/start HTTP/1.1"):
                    pass
                with network_operation(tmp_path, b"GET /v1.53/networks HTTP/1.1"):
                    pass
        with (tmp_path / "network_operation.lock").open("a") as released:
            fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
