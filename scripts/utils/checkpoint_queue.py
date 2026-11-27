# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Watch completed five-task checkpoints and evaluate them through one bounded durable queue."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pickle
import sqlite3
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.utils import checkpoint_sources as sources  # noqa: E402
from scripts.utils.checkpoint_eval_worker import evaluate, initialize  # noqa: E402
from scripts.utils.run_qwen_isolated_when_free import configure_experiment_environment, write  # noqa: E402


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


class Queue:
    def __init__(self, output, *, interval=5, share_initial=False):
        self.interval = interval
        self.share_initial = share_initial
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.output / "queue.sqlite3", timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS streams(id TEXT PRIMARY KEY, spec TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS stream_invalidations(stream TEXT PRIMARY KEY, reason TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS checkpoints(
                id TEXT PRIMARY KEY, stream TEXT NOT NULL, point INTEGER NOT NULL, sha TEXT NOT NULL,
                canonical TEXT NOT NULL, payload TEXT NOT NULL, file TEXT NOT NULL,
                UNIQUE(stream,point));
            CREATE TABLE IF NOT EXISTS tasks(
                id INTEGER PRIMARY KEY, cp TEXT NOT NULL, split TEXT NOT NULL, task_id TEXT NOT NULL,
                priority INTEGER NOT NULL, ordinal INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0, result TEXT, started TEXT, ended TEXT,
                UNIQUE(cp,split,task_id));
            CREATE INDEX IF NOT EXISTS task_dispatch ON tasks(status,priority,ordinal,id);
            CREATE INDEX IF NOT EXISTS cp_stream ON checkpoints(stream,point);
            CREATE TRIGGER IF NOT EXISTS prioritize_final_checkpoint_tests AFTER INSERT ON checkpoints
            WHEN NEW.point=json_extract((SELECT spec FROM streams WHERE id=NEW.stream),'$.train_total')
            BEGIN
                UPDATE tasks SET priority=0 WHERE cp=NEW.canonical AND split='test';
            END;
        """)

    def register(self, spec, snapshot):
        point = snapshot["point"]
        if (point % self.interval or not 0 <= point <= spec["train_total"]
                or (point == 0 and not self.share_initial)):
            raise ValueError("Checkpoint does not match the configured training-task interval")
        identity = f"{spec['id']}_cp{point:03d}"
        sha = sources.digest(snapshot["content"])
        prior_stream = self.db.execute("SELECT spec FROM streams WHERE id=?", (spec["id"],)).fetchone()
        if prior_stream:
            previous_spec = json.loads(prior_stream["spec"])
            for field in ("method", "benchmark", "seed", "model", "temperature", "splits", "playbook_edit_mode",
                          "appworld_policy_sha256", "actor_context", "actor_context_metadata",
                          "inference_profile_sha256"):
                if previous_spec.get(field) != spec.get(field):
                    raise ValueError(f"Checkpoint stream configuration changed: {field}")
        existing = self.db.execute("SELECT * FROM checkpoints WHERE id=?", (identity,)).fetchone()
        if existing:
            if existing["sha"] != sha:
                raise ValueError(f"An immutable checkpoint changed: {identity}")
            return False
        previous = self.db.execute("SELECT * FROM checkpoints WHERE stream=? AND point=?",
                                   (spec["id"], point - self.interval)).fetchone()
        canonical = previous["canonical"] if previous and previous["sha"] == sha else identity
        matching = self.db.execute("SELECT canonical FROM checkpoints WHERE stream=? AND sha=? LIMIT 1",
                                   (spec["id"], sha)).fetchone()
        if matching:
            canonical = matching["canonical"]
        if point == 0 and self.share_initial:
            if not snapshot.get("initial_group"):
                raise ValueError("Shared CP0 needs a verified evaluation-context fingerprint")
            shared = self.db.execute("""SELECT canonical FROM checkpoints WHERE point=0 AND sha=?
                AND json_extract(payload,'$.initial_group')=? LIMIT 1""",
                (sha, snapshot["initial_group"])).fetchone()
            if shared:
                canonical = shared["canonical"]
        path = self.output / "frozen" / spec["id"] / f"cp{point:03d}.json"
        payload = {**snapshot, "stream": spec, "playbook": snapshot["content"], "created_at": now(),
                   "content_sha256": sha, "canonical": canonical}
        if path.exists():
            prior = json.loads(path.read_text())
            if prior["content_sha256"] != sha:
                raise ValueError("An existing frozen file has a different policy")
            payload = prior
        else:
            write(path, payload)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO streams VALUES(?,?)", (spec["id"], json.dumps(spec)))
            self.db.execute("INSERT INTO checkpoints VALUES(?,?,?,?,?,?,?)",
                            (identity, spec["id"], point, sha, canonical, json.dumps(payload), str(path)))
            if canonical == identity:
                for split_index, (split, task_ids) in enumerate(spec["splits"].items()):
                    priority = 0 if point == spec["train_total"] and split == "test" else 1
                    self.db.executemany("""INSERT INTO tasks(cp,split,task_id,priority,ordinal)
                        VALUES(?,?,?,?,?)""", [(identity, split, task, priority, i * 3 + split_index)
                                              for i, task in enumerate(task_ids)])
        return True

    def pending(self, limit, stream_ids=None):
        scope, parameters = stream_scope(stream_ids)
        return self.db.execute("""SELECT t.*,c.file AS checkpoint_file,c.sha AS content_sha256
            FROM tasks t JOIN checkpoints c ON c.id=t.cp WHERE t.status='pending'
            AND NOT EXISTS(SELECT 1 FROM stream_invalidations i WHERE i.stream=c.stream)
            """ + scope + " ORDER BY priority,ordinal,t.id LIMIT ?", (*parameters, limit)).fetchall()

    def recover_interrupted(self, stream_ids=None):
        scope, parameters = stream_scope(stream_ids)
        with self.db:
            return self.db.execute("""UPDATE tasks SET status='pending',attempts=max(0,attempts-1)
                WHERE status='running' AND cp IN (SELECT c.id FROM checkpoints c WHERE
                NOT EXISTS(SELECT 1 FROM stream_invalidations i WHERE i.stream=c.stream))"""
                + " AND cp IN (SELECT c.id FROM checkpoints c WHERE 1=1" + scope + ")", parameters).rowcount

    def claim(self, row):
        with self.db:
            self.db.execute("UPDATE tasks SET status='running',attempts=attempts+1,started=? WHERE id=?",
                            (now(), row["id"]))
        return {**dict(row), "attempt": row["attempts"] + 1}

    def finish(self, task, result):
        write(self.output / "attempts" / f"{task['id']:07d}_{task['attempt']}.json", result)
        good = result.get("status") == "completed" and isinstance(result.get("score"), (int, float))
        status = "completed" if good else "pending" if task["attempt"] < 3 else "error"
        with self.db:
            self.db.execute("UPDATE tasks SET status=?,result=?,ended=? WHERE id=?",
                            (status, json.dumps(result), now(), task["id"]))
        if status in {"completed", "error"}:
            write(self.output / "records" / f"{task['id']:07d}.json", {**task, "status": status, "result": result})

    def import_final(self, spec):
        cp = self.db.execute("SELECT * FROM checkpoints WHERE stream=? AND point=?",
                             (spec["id"], spec["train_total"])).fetchone()
        if cp is None:
            return
        output = Path(spec["output"])
        rows = []
        if spec["method"] in {"ace", "premise"}:
            frozen = output / "evaluation/frozen_checkpoint.json"
            if not frozen.exists():
                return
            metadata = json.loads(frozen.read_text())
            if (sources.digest(metadata["playbook"]) != cp["sha"]
                    or metadata.get("model") != spec["model"]
                    or metadata.get("temperature_all_roles") != spec["temperature"]):
                return
            for path in (output / "evaluation/records").glob("*.json"):
                row = json.loads(path.read_text())
                if row.get("status") == "completed":
                    rows.append((str(row["task_id"]), row, str(path)))
        else:
            frozen, path = output / "frozen_instruction.json", output / "test_results.json"
            if not frozen.exists() or not path.exists():
                return
            if sources.digest(json.loads(frozen.read_text())["content"]) != cp["sha"]:
                return
            for row in json.loads(path.read_text()):
                if row.get("hard") in (0, 1):
                    rows.append((str(row["id"]), {"status": "completed", "score": row["hard"],
                                 "partial_score": row.get("soft")}, str(path)))
        with self.db:
            for task_id, row, source in rows:
                result = json.dumps({**row, "reused_from": source})
                self.db.execute("""UPDATE tasks SET status='completed',result=?,ended=?
                    WHERE cp=? AND split='test' AND task_id=? AND status='pending'""",
                                (result, now(), cp["canonical"], task_id))

    def dashboard(self, issues=None, stream_ids=None, workers=100):
        scope, parameters = stream_scope(stream_ids)
        counts = {r[0]: r[1] for r in self.db.execute("""SELECT t.status,count(*) FROM tasks t
            JOIN checkpoints c ON c.id=t.cp
            WHERE NOT EXISTS(SELECT 1 FROM stream_invalidations i WHERE i.stream=c.stream)"""
            + scope + " GROUP BY t.status", parameters)}
        totals = {}
        for group in self.db.execute("""SELECT cp,split,count(*) AS total,
            sum(status='completed') AS done,sum(status='running') AS running,sum(status='error') AS errors,
            avg(CASE WHEN status='completed' THEN json_extract(result,'$.score') END) AS score
            FROM tasks GROUP BY cp,split"""):
            totals[(group["cp"], group["split"])] = {
                key: group[key] for key in ("total", "done", "running", "errors", "score")}
        rows = []
        for cp in self.db.execute("""SELECT * FROM checkpoints c
            WHERE NOT EXISTS(SELECT 1 FROM stream_invalidations i WHERE i.stream=c.stream) ORDER BY stream,point"""):
            payload = json.loads(cp["payload"])
            spec = payload["stream"]
            row = {"id": cp["id"], "stream": spec["id"], "method": spec["method"], "benchmark": spec["benchmark"],
                   "seed": spec["seed"], "variant": spec["variant"], "point": cp["point"],
                   "run_variant": spec.get("run_variant"),
                   "selection": payload["selection"], "selected_val_score": payload.get("selected_val_score"),
                   "policy_hash": cp["sha"][:12], "alias": cp["canonical"] if cp["canonical"] != cp["id"] else None,
                   "splits": {}}
            row["selected"] = stream_ids is None or spec["id"] in stream_ids
            for split in ("train", "val", "test"):
                row["splits"][split] = dict(totals.get((cp["canonical"], split), {
                    "done": 0, "total": 0, "running": 0, "errors": 0, "score": None}))
                if not row["selected"]:
                    row["splits"][split]["running"] = 0
            groups = list(row["splits"].values())
            row["state"] = "skipped" if row["alias"] else "error" if any(g["errors"] for g in groups) else (
                "completed" if all(g["done"] == g["total"] for g in groups) else
                "running" if any(g["running"] for g in groups) else "pending")
            if not row["selected"] and row["state"] in {"running", "pending"}:
                row["state"] = "paused"
            rows.append(row)
        return {"kind": "five_task_checkpoint_queue", "id": self.output.name, "updated_at": now(),
                "checkpoint_interval": self.interval,
                "controller_pid": os.getpid(), "workers": workers, "task_counts": counts,
                "checkpoints": len(rows), "skipped": sum(bool(r["alias"]) for r in rows),
                "invalidated_checkpoints": self.db.execute("""SELECT count(*) FROM checkpoints c
                    WHERE EXISTS(SELECT 1 FROM stream_invalidations i WHERE i.stream=c.stream)""").fetchone()[0],
                "rows": rows, "source_issues": issues or [],
                "selected_stream_ids": list(stream_ids) if stream_ids is not None else None,
                "phase": "running" if counts.get("running") else "watching"}


def stream_scope(stream_ids):
    if stream_ids is None:
        return "", ()
    values = tuple(stream_ids)
    return (" AND c.stream IN (" + ",".join("?" for _ in values) + ")", values) if values else (" AND 0", ())


def discover_roots(plan):
    roots = set(plan["roots"])
    if not plan.get("auto_discover", True):
        return sorted(roots)
    directory = Path(plan["watch_directory"])
    for path in directory.glob("*/launch_manifest.json"):
        try:
            data = json.loads(path.read_text())
            if (data.get("created_at", "") >= plan["earliest_source"] and data.get("jobs")
                    and "Qwen/Qwen3.6-35B-A3B" in data.get("model", "")
                    and not (path.parent / "superseded_by.json").exists()):
                roots.add(str(path.parent))
        except (OSError, ValueError):
            continue
    return sorted(roots)


def collect(queue, plan, reader):
    issues = []
    specs = sources.streams(discover_roots(plan), reader)
    selected = plan.get("selected_stream_ids")
    if selected is not None:
        specs = [spec for spec in specs if spec["id"] in selected]
        if set(selected) != {spec["id"] for spec in specs}:
            raise ValueError("Selected checkpoint sources are missing or changed")
    for spec in specs:
        try:
            for snapshot in sources.checkpoints(spec, reader):
                queue.register(spec, snapshot)
            queue.import_final(spec)
            marker = Path(spec["output"]) / "checkpoint_queue.json"
            target = {"output": str(queue.output), "stream": spec["id"], "point": spec["train_total"],
                      "proxy_module": str(ROOT / "scripts/utils/checkpoint_eval_proxy.py")}
            if not marker.exists() or json.loads(marker.read_text()) != target:
                write(marker, target)
        except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as exc:
            issues.append({"stream": spec["id"], "method": spec["method"], "benchmark": spec["benchmark"],
                           "seed": spec["seed"], "error": f"{type(exc).__name__}: {exc}"[:600]})
    return issues


def stop_orphan_workers(output):
    import psutil

    from scripts.utils.process_identity import matches_identity

    for path in (output / "workers").glob("*.json"):
        record = json.loads(path.read_text())
        try:
            p = psutil.Process(record["pid"])
            if matches_identity(record, p.create_time()) and p.status() != psutil.STATUS_ZOMBIE:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except psutil.TimeoutExpired:
                    p.kill()
        except psutil.NoSuchProcess:
            pass


def dispatch(queue, pool, active, stream_ids=None, workers=100):
    for row in queue.pending(max(0, workers - len(active)), stream_ids):
        task = queue.claim(row)
        active[pool.submit(evaluate, task)] = task


def run(output, once=False):
    plan = json.loads((output / "checkpoint_queue_manifest.json").read_text())
    workers = worker_limit(plan)
    queue, reader = Queue(output, interval=plan.get("checkpoint_interval", 5),
                          share_initial=plan.get("share_initial", False)), sources.Reader()
    selected = plan.get("selected_stream_ids")
    if once:
        issues = [] if plan.get("prepared") else collect(queue, plan, reader)
        write(output / "dashboard.json", queue.dashboard(issues, selected, workers))
        return
    if plan.get("environment_source_manifest"):
        environment_path = Path(plan["environment_source_manifest"])
        if sources.digest(environment_path.read_bytes()) != plan["environment_source_sha256"]:
            raise ValueError("Checkpoint environment source changed")
        original_environment = json.loads(environment_path.read_text())["environment"]
        os.environ.update({key: original_environment[key] for key in plan["environment_keys"]})
    configure_experiment_environment()
    os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"] = plan["base_url"]
    from scripts.utils.process_identity import stable_identity

    write(output / "controller_process.json", {"pid": os.getpid(), "started_at": now(),
                                              **stable_identity(os.getpid())})
    stop_orphan_workers(output)
    queue.recover_interrupted(selected)
    active = {}
    last_scan = 0
    issues = []
    model_ready = False
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(str(output), plan["base_url"])) as pool:
        while True:
            if time.monotonic() - last_scan >= 10:
                issues = [] if plan.get("prepared") else collect(queue, plan, reader)
                try:
                    with urllib.request.urlopen(plan["base_url"].rstrip("/") + "/models", timeout=5) as response:
                        model_ready = any("Qwen/Qwen3.6-35B-A3B" in row["id"] for row in json.load(response)["data"])
                except (OSError, ValueError, KeyError):
                    model_ready = False
                last_scan = time.monotonic()
            for future, task in list(active.items()):
                if future.done():
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"status": "error", "score": None, "error": f"{type(exc).__name__}: {exc}"}
                    queue.finish(task, result)
                    del active[future]
            if model_ready and not (output / "pause_dispatch.json").exists():
                dispatch(queue, pool, active, selected, workers)
            dashboard = queue.dashboard(issues, selected, workers)
            if plan.get("query_router_url"):
                try:
                    with urllib.request.urlopen(plan["query_router_url"], timeout=1) as response:
                        dashboard["gpu_queries"] = json.load(response)
                except (OSError, ValueError):
                    dashboard["gpu_queries"] = {"unavailable": True}
            if not model_ready:
                dashboard["phase"] = "waiting_model"
            elif (output / "pause_dispatch.json").exists():
                dashboard["phase"] = "paused"
            drained = (plan.get("stop_when_drained", False) and not active and not issues
                       and not any(dashboard["task_counts"].get(status) for status in ("pending", "running")))
            if drained:
                dashboard["phase"] = "error" if dashboard["task_counts"].get("error") else "completed"
            write(output / "dashboard.json", dashboard)
            write(output / "heartbeat.json", {"at": now(), "active": len(active), "workers": workers,
                                               "selected_stream_ids": selected, "controller_pid": os.getpid(),
                                               "base_url": plan["base_url"]})
            if drained:
                break
            time.sleep(2)


def worker_limit(plan):
    workers = plan.get("workers", 100)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("Checkpoint workers must be a positive integer")
    return workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Retain the historical global lock path so queues using different limits cannot overlap.
    with FileLock(str(args.output / "controller.lock"), timeout=0), FileLock(
        str(Path.home() / ".cache/exgentic/checkpoint_evaluation_100.lock"), timeout=0
    ):
        run(args.output.resolve(), once=args.collect_only)


if __name__ == "__main__":
    main()
