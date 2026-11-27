# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Remove only unused Exgentic Terminal-Bench containers, keeping active work and other users' resources."""

from __future__ import annotations

import argparse
import fcntl
import http.client
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()


class DockerHTTP(http.client.HTTPConnection):
    socket_path = "/var/run/docker.sock"

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def daemon_request(method, path):
    with closing(DockerHTTP("localhost", timeout=10)) as connection:
        connection.request(method, path)
        response = connection.getresponse()
        body = response.read()
        if response.status not in {200, 204, 404}:
            raise RuntimeError(f"Docker API returned HTTP {response.status}")
        return response.status, json.loads(body) if body else None


def remote(request):
    if request["operation"] == "list":
        _, rows = daemon_request("GET", "/containers/json?all=1")
        return [{"id": row["Id"], "names": row["Names"], "state": row["State"],
                 "created": row["Created"], "labels": row.get("Labels", {})} for row in rows]
    if request["operation"] == "list_networks":
        return daemon_request("GET", "/networks")[1]
    if request["operation"] == "remove_networks":
        results = []
        for candidate in request["candidates"]:
            identity = candidate["id"]
            row = {"id": identity, "resource": "network"}
            try:
                status, current = daemon_request("GET", f"/networks/{identity}")
                if status == 404:
                    row["state"] = "already_removed"
                elif (current.get("Labels") or {}).get("com.docker.compose.project") != candidate["project"]:
                    row["state"] = "ownership_changed"
                elif current.get("Containers"):
                    row["state"] = "in_use"
                else:
                    daemon_request("DELETE", f"/networks/{identity}")
                    row["state"] = "removed"
            except (OSError, ValueError, RuntimeError, http.client.HTTPException) as exc:
                row.update(state="retry_next_pass", error_type=type(exc).__name__)
            results.append(row)
        return results
    if request["operation"] != "remove":
        raise ValueError("Unknown container cleanup operation")
    results = []
    for candidate in request["candidates"]:
        identity = candidate["id"]
        try:
            status, current = daemon_request("GET", f"/containers/{identity}/json")
            if status == 404:
                results.append({"id": identity, "state": "already_removed"})
                continue
            labels = current["Config"].get("Labels") or {}
            if (labels.get("com.docker.compose.project") != candidate["project"]
                    or labels.get("com.docker.compose.project.working_dir") != candidate["working_dir"]):
                results.append({"id": identity, "state": "ownership_changed"})
                continue
            daemon_request("DELETE", f"/containers/{identity}?force=1&v=0")
            results.append({"id": identity, "project": candidate["project"], "state": "removed"})
        except (OSError, ValueError, RuntimeError, http.client.HTTPException) as exc:
            results.append({"id": identity, "state": "retry_next_pass", "error_type": type(exc).__name__})
    return results


def call_remote(request, host):
    target = urlsplit(host)
    if target.scheme != "ssh" or not target.hostname:
        raise ValueError("The container janitor requires this experiment's explicit SSH Docker endpoint")
    destination = (target.username + "@" if target.username else "") + target.hostname
    spec = importlib.util.spec_from_file_location("docker_ssh_forward", ROOT / "scripts/utils/docker_ssh_pool.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    # Match Docker's SSH argv (its -o option is one argument) so all callers
    # share the authenticated master and the same private local Unix socket.
    destination_args = (["-p", str(target.port)] if target.port else []) + [
        "-o ConnectTimeout=30", "-T", "--", destination]
    previous = DockerHTTP.socket_path
    try:
        DockerHTTP.socket_path = str(transport.ensure_forward(destination_args))
        return remote(request)
    finally:
        DockerHTTP.socket_path = previous


def active_work():
    result = subprocess.run(["ps", "-eo", "uid=,args="], capture_output=True, text=True, check=True, timeout=20)
    sessions, projects = set(), set()
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or parts[0] != str(os.getuid()):
            continue
        args = parts[1]
        if "terminalbench2_eval:TerminalBench2Session" in args:
            match = re.search(r'"session_id"\s*:\s*"([^"\\]+)"', args)
            if not match:
                raise RuntimeError("Cannot identify an active Terminal-Bench session; cleanup skipped")
            sessions.add(match[1])
        if "docker" in args:
            match = re.search(r"--project-name\s+(exgentic-[a-z0-9_-]+)", args)
            if match:
                projects.add(match[1])
    return sessions, projects


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def schedule_cleanup(candidates, retries, *, now, limit):
    """Defer locked containers so a slow first batch cannot starve every other one."""
    eligible = [row for row in candidates if retries.get(row["id"], {}).get("after", 0) <= now]
    eligible.sort(key=lambda row: retries.get(row["id"], {}).get("last_attempt", 0))
    return eligible[:limit]


def update_retries(retries, results, *, now):
    for row in results:
        identity = row["id"]
        if row["state"] in {"removed", "already_removed"}:
            retries.pop(identity, None)
        else:
            failures = retries.get(identity, {}).get("failures", 0) + 1
            retries[identity] = {"failures": failures, "last_attempt": now,
                                 "after": now + min(600, 30 * 2 ** min(failures - 1, 5))}


def within_experiment_scope(candidates, leases, experiment_roots):
    """A shared benchmark directory or project prefix alone does not prove experiment ownership."""
    roots = [Path(root).resolve() for root in experiment_roots]
    selected = []
    for candidate in candidates:
        artifacts = leases.get(candidate["project"], {}).get("artifacts")
        if artifacts and any(Path(artifacts).resolve().is_relative_to(root) for root in roots):
            selected.append(candidate)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-root", type=Path,
                        default=Path.home() / ".exgentic/benchmarks/terminalbench2/terminal-bench-2")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--max-removals", type=int, default=20)
    parser.add_argument("--max-network-removals", type=int, default=0,
                        help="Explicit opt-in to remove empty task-owned networks; disabled by default")
    parser.add_argument("--experiment-root", action="append", type=Path, default=[],
                        help="Only clean resources whose lifecycle artifacts are inside these explicit run roots")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.remote:
        print(json.dumps(remote(json.load(sys.stdin))))
        return
    if (not args.output or args.interval < 10 or not 1 <= args.max_removals <= 100
            or not 0 <= args.max_network_removals <= 20):
        parser.error("An output directory, interval >=10 and max-removals in [1,100] are required")
    if args.apply and not args.experiment_root:
        parser.error("Deletion requires at least one explicit --experiment-root")
    # This operational worker needs only the stdlib ownership helper, without
    # loading model/benchmark dependencies through the framework's package init.
    spec = importlib.util.spec_from_file_location(
        "terminalbench_container_lifecycle", ROOT / "src/exgentic/benchmarks/terminalbench2/container_lifecycle.py")
    lifecycle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lifecycle)

    args.output.mkdir(parents=True, exist_ok=True)
    retry_path = args.output / "cleanup_retries.json"
    retries = json.loads(retry_path.read_text()) if retry_path.exists() else {}
    host = os.environ["DOCKER_HOST"]
    with (args.output / "janitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(args.output / "janitor_process.json", {"pid": os.getpid(), "command": [sys.executable, *sys.argv]})
        while True:
            try:
                write(args.output / "janitor_status.json", {"at": time.time(), "phase": "scanning"})
                rows = call_remote({"operation": "list"}, host)
                sessions, projects = active_work()
                leases = {path.stem: json.loads(path.read_text())
                          for path in lifecycle.lease_directory().glob("*.json")}
                candidates = lifecycle.unused_containers(rows, leases=leases, active_sessions=sessions,
                                               active_projects=projects, task_root=args.task_root)
                candidates = within_experiment_scope(candidates, leases, args.experiment_root)
                write(args.output / "cleanup_plan.json", {"at": time.time(), "containers": len(rows),
                      "active_sessions": len(sessions), "active_projects": len(projects), "candidates": candidates})
                # Refresh protection immediately before mutations, not just before slow inventory collection.
                sessions, projects = active_work()
                candidates = lifecycle.unused_containers(rows, leases=leases, active_sessions=sessions,
                                               active_projects=projects, task_root=args.task_root)
                candidates = within_experiment_scope(candidates, leases, args.experiment_root)
                candidates = schedule_cleanup(candidates, retries, now=time.time(), limit=args.max_removals)
                results = []
                if args.apply and candidates:
                    write(args.output / "janitor_status.json", {"at": time.time(), "phase": "cleaning",
                                                               "eligible": len(candidates)})
                    results = call_remote({"operation": "remove", "candidates": candidates}, host)
                network_candidates = []
                if args.max_network_removals:
                    networks = call_remote({"operation": "list_networks"}, host)
                    sessions, projects = active_work()
                    container_projects = {r.get("labels", {}).get("com.docker.compose.project") for r in rows}
                    network_candidates = lifecycle.unused_networks(
                        networks, leases=leases, active_sessions=sessions, active_projects=projects,
                        container_projects=container_projects, task_root=args.task_root)
                    network_candidates = within_experiment_scope(network_candidates, leases, args.experiment_root)
                    network_candidates = schedule_cleanup(network_candidates, retries, now=time.time(),
                                                          limit=args.max_network_removals)
                    if args.apply and network_candidates:
                        results += call_remote({"operation": "remove_networks", "candidates": network_candidates}, host)
                state = {"phase": "watching" if args.apply else "dry_run", "eligible": len(candidates),
                         "eligible_networks": len(network_candidates),
                         "removed": sum(r["state"] in {"removed", "already_removed"} for r in results),
                         "removed_networks": sum(r.get("resource") == "network"
                                                 and r["state"] in {"removed", "already_removed"} for r in results)}
                if results:
                    update_retries(retries, results, now=time.time())
                    write(retry_path, retries)
                    with (args.output / "cleanup_events.jsonl").open("a") as events:
                        events.write(json.dumps({"at": time.time(), "results": results}) + "\n")
            except (OSError, ValueError, RuntimeError, http.client.HTTPException, subprocess.SubprocessError) as exc:
                state = {"phase": "retry_next_pass", "error_type": type(exc).__name__}
            write(args.output / "janitor_status.json", {"at": time.time(), **state})
            if args.once:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
