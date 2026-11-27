# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from scripts.utils import cleanup_terminalbench_containers as janitor


def test_remote_cleanup_rechecks_ownership_and_never_deletes_a_changed_container(monkeypatch):
    calls = []
    def request(method, path):
        calls.append((method, path))
        return 200, {"Config": {"Labels": {"com.docker.compose.project": "another-user"}}}
    monkeypatch.setattr(janitor, "daemon_request", request)
    result = janitor.remote({"operation": "remove", "candidates": [
        {"id": "owned", "project": "exgentic-owned", "working_dir": "/owned"}]})
    assert result[0]["state"] == "ownership_changed"
    assert calls == [("GET", "/containers/owned/json")]


def test_remote_cleanup_keeps_volumes_and_networks_with_attached_containers(monkeypatch):
    calls = []
    def request(method, path):
        calls.append((method, path))
        if path == "/containers/owned/json":
            return 200, {"Config": {"Labels": {"com.docker.compose.project": "exgentic-owned",
                                               "com.docker.compose.project.working_dir": "/owned"}}}
        if path.startswith("/networks?"):
            return 200, [{"Id": "in-use"}, {"Id": "empty-foreign"}]
        if path == "/networks/in-use":
            return 200, {"Containers": {"another-container": {}}}
        if path == "/networks/empty-foreign":
            return 200, {"Containers": {}, "Labels": {"com.docker.compose.project": "other-user"}}
        return 204, None
    monkeypatch.setattr(janitor, "daemon_request", request)
    result = janitor.remote({"operation": "remove", "candidates": [
        {"id": "owned", "project": "exgentic-owned", "working_dir": "/owned"}]})
    assert result[0]["state"] == "removed"
    assert ("DELETE", "/containers/owned?force=1&v=0") in calls
    assert ("DELETE", "/networks/in-use") not in calls
    assert ("DELETE", "/networks/empty-foreign") not in calls


def test_slow_containers_back_off_without_starving_other_finished_containers():
    retries = {}
    candidates = [{"id": "locked"}, {"id": "ready"}]
    assert janitor.schedule_cleanup(candidates, retries, now=100, limit=1) == [{"id": "locked"}]
    janitor.update_retries(retries, [{"id": "locked", "state": "retry_next_pass"}], now=100)
    assert janitor.schedule_cleanup(candidates, retries, now=101, limit=1) == [{"id": "ready"}]
    janitor.update_retries(retries, [{"id": "ready", "state": "removed"}], now=101)
    assert janitor.schedule_cleanup([{"id": "locked"}], retries, now=131, limit=1) == [{"id": "locked"}]


def test_orphan_network_cleanup_rechecks_ownership_and_attached_containers(monkeypatch):
    calls = []

    def request(method, path):
        calls.append((method, path))
        if path == "/networks/foreign":
            return 200, {"Labels": {"com.docker.compose.project": "other"}}
        if path == "/networks/attached":
            return 200, {"Labels": {"com.docker.compose.project": "exgentic-owned"}, "Containers": {"live": {}}}
        return 200, {"Labels": {"com.docker.compose.project": "exgentic-owned"}, "Containers": {}}

    monkeypatch.setattr(janitor, "daemon_request", request)
    results = janitor.remote({"operation": "remove_networks", "candidates": [
        {"id": identity, "project": "exgentic-owned"} for identity in ("foreign", "attached", "empty")]})
    assert [r["state"] for r in results] == ["ownership_changed", "in_use", "removed"]
    assert [path for method, path in calls if method == "DELETE"] == ["/networks/empty"]


def test_cleanup_requires_artifacts_inside_an_explicit_experiment_root(tmp_path):
    root = tmp_path / "my-experiment"
    candidates = [{"id": key, "project": key} for key in ("mine", "other", "similar-prefix", "unknown")]
    leases = {"mine": {"artifacts": str(root / "sessions/one")},
              "other": {"artifacts": str(tmp_path / "someone-else/sessions")},
              "similar-prefix": {"artifacts": str(tmp_path / "my-experiment-other/sessions")}}
    assert janitor.within_experiment_scope(candidates, leases, [root]) == [candidates[0]]
    assert janitor.within_experiment_scope(candidates, leases, []) == []
