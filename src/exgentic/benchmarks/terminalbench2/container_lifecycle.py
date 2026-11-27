# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Persist container ownership and bound teardown independently of task scores."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import socket
import time
from pathlib import Path


def lease_directory():
    return Path(os.environ.get("EXGENTIC_TERMINALBENCH2_LEASE_DIR",
                               Path.home() / ".cache/exgentic/terminalbench2/leases"))


def process_identity(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rpartition(")")[2].split()
        return fields[19] if fields[0] not in {"Z", "X"} else None
    except (OSError, IndexError):
        return None


def write_lease(project, **fields):
    directory = lease_directory()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{project}.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {"project": project, "owner_pid": os.getpid(), "owner_start": process_identity(os.getpid()),
                "hostname": socket.gethostname(), "created_at": time.time()}
    data.update(fields, updated_at=time.time())
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


async def stop_process_group(process):
    """A cancelled Compose command must release its SSH children and pool leases."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=1)
        except TimeoutError:
            pass
        # Even an exited CLI can leave descendants holding subprocess pipes.


def unused_containers(rows, *, leases, active_sessions, active_projects, task_root, now=None, grace=600):
    """Fail closed for unknown ownership, active actors and recent creations."""
    now = time.time() if now is None else now
    result = []
    for row in rows:
        labels = row.get("labels") or {}
        project = labels.get("com.docker.compose.project", "")
        directory = Path(labels.get("com.docker.compose.project.working_dir", "/"))
        if not project.startswith("exgentic-") or directory.parent.parent != Path(task_root):
            continue
        lease = leases.get(project)
        if lease:
            if lease.get("hostname") != socket.gethostname():
                continue
            live = process_identity(lease.get("owner_pid"))
            owner_alive = live is not None and live == lease.get("owner_start")
            finished = lease.get("phase") in {"closed", "cleanup_pending"}
            if owner_alive and not finished:
                continue
            if finished and now - lease.get("updated_at", now) < 30:
                continue
            if not finished and now - lease.get("created_at", now) < grace:
                continue
        else:
            match = re.fullmatch(r"exgentic-([0-9a-f]{8})-[0-9a-f]{12}", project)
            if (not match or match[1] in active_sessions or project in active_projects
                    or now - row.get("created", now) < grace):
                continue
        result.append({"id": row["id"], "project": project, "working_dir": str(directory),
                       "reason": "finished_lease" if lease else "orphan_without_live_session"})
    return result


def unused_networks(rows, *, leases, active_sessions, active_projects, container_projects, task_root, now=None):
    """Require a local task lease even when an aborted start never created a container."""
    result = []
    for row in rows:
        project = (row.get("Labels") or {}).get("com.docker.compose.project", "")
        lease = leases.get(project)
        if (not re.fullmatch(r"exgentic-[0-9a-f]{8}-[0-9a-f]{12}", project) or not lease
                or lease.get("project") != project or project in container_projects
                or project in active_projects or lease.get("session_id") in active_sessions):
            continue
        task_id = lease.get("task_id", "")
        if not task_id or Path(task_id).name != task_id or not (Path(task_root) / task_id / "task.toml").is_file():
            continue
        # Reuse the same hostname, process-start identity and lifecycle grace checks.
        candidate = {"id": row["Id"], "labels": {"com.docker.compose.project": project,
                     "com.docker.compose.project.working_dir": str(Path(task_root) / task_id / "environment")}}
        if unused_containers([candidate], leases=leases, active_sessions=active_sessions,
                             active_projects=active_projects, task_root=task_root, now=now):
            result.append({"id": row["Id"], "project": project, "reason": "finished_task_without_container"})
    return result
