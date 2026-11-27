# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
from types import SimpleNamespace

from exgentic.benchmarks.terminalbench2.container_lifecycle import (
    process_identity,
    stop_process_group,
    unused_containers,
    unused_networks,
    write_lease,
)


def test_cleanup_preserves_active_foreign_recent_and_unrecognized_containers(monkeypatch, tmp_path):
    root = tmp_path / "tasks"
    def container(session, **extra):
        project = f"exgentic-{session}-012345abcdef"
        return {"id": session, "created": 0, "labels": {
            "com.docker.compose.project": project,
            "com.docker.compose.project.working_dir": str(root / "task/environment")}, **extra}

    rows = [container(f"{i:08x}") for i in range(1, 8)]
    rows[3]["labels"]["com.docker.compose.project.working_dir"] = "/other-user/tasks/task/environment"
    rows[4]["created"] = 1990
    rows[5]["labels"]["com.docker.compose.project"] = "unrecognized-container"
    lease = {"hostname": socket.gethostname(), "owner_pid": 123, "owner_start": "same", "phase": "active"}
    monkeypatch.setattr("exgentic.benchmarks.terminalbench2.container_lifecycle.process_identity", lambda pid: "same")
    leases = {rows[6]["labels"]["com.docker.compose.project"]: lease}
    result = unused_containers(rows, leases=leases, active_sessions={"00000002"},
                               active_projects={rows[2]["labels"]["com.docker.compose.project"]},
                               task_root=root, now=2000)
    assert [row["id"] for row in result] == ["00000001"]
    # A recycled PID does not make an abandoned lease appear to be owned by a live actor.
    lease.update(owner_start="different", created_at=0)
    result = unused_containers(rows[-1:], leases=leases, active_sessions=set(), active_projects=set(),
                               task_root=root, now=2000)
    assert result[0]["id"] == "00000007"


def test_cancelling_compose_also_terminates_its_ssh_descendants():
    async def exercise():
        script = ("import subprocess,sys,signal; "
                  "p=subprocess.Popen([sys.executable,'-c','import signal;signal.pause()']); "
                  "print(p.pid,flush=True);signal.pause()")
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script, stdout=asyncio.subprocess.PIPE, start_new_session=True,
        )
        child = int(await asyncio.wait_for(process.stdout.readline(), timeout=5))
        try:
            await asyncio.wait_for(stop_process_group(process), timeout=5)
            assert process_identity(child) is None
        finally:
            if process.returncode is None:
                os.killpg(process.pid, 9)
                await process.wait()
    asyncio.run(exercise())


def test_orphan_networks_require_finished_local_task_lease_and_no_active_work(tmp_path):
    root = tmp_path / "tasks"
    (root / "demo").mkdir(parents=True)
    (root / "demo/task.toml").write_text("")
    project = "exgentic-01234567-012345abcdef"
    rows = [{"Id": "network", "Labels": {"com.docker.compose.project": project}}]
    lease = {"project": project, "hostname": socket.gethostname(), "task_id": "demo", "session_id": "01234567",
             "owner_pid": os.getpid(), "owner_start": process_identity(os.getpid()),
             "phase": "closed", "updated_at": 0}
    kwargs = {"leases": {project: lease}, "active_sessions": set(), "active_projects": set(),
              "container_projects": set(), "task_root": root, "now": 2000}
    assert unused_networks(rows, **kwargs)[0]["id"] == "network"
    for change in ({"leases": {}}, {"active_sessions": {"01234567"}}, {"active_projects": {project}},
                   {"container_projects": {project}}):
        assert unused_networks(rows, **(kwargs | change)) == []
    lease["hostname"] = "another-host"
    assert unused_networks(rows, **kwargs) == []


def test_cleanup_errors_are_deferred_without_discarding_a_final_score(tmp_path, monkeypatch):
    from exgentic.benchmarks.terminalbench2.terminalbench2_eval import TerminalBench2Session

    monkeypatch.setenv("EXGENTIC_TERMINALBENCH2_LEASE_DIR", str(tmp_path))
    project = "exgentic-01234567-012345abcdef"
    write_lease(project, phase="active")
    class BrokenEnvironment:
        async def stop(self, delete):
            raise RuntimeError("transport unavailable")
    session = SimpleNamespace(_loop_runner=asyncio.Runner(), _environment=BrokenEnvironment(),
                              _docker_project=project, logger=logging.getLogger(__name__))
    TerminalBench2Session.close(session)
    assert session._loop_runner is None and session._environment is None
    assert json.loads((tmp_path / f"{project}.json").read_text())["phase"] == "cleanup_pending"
