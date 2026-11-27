# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Read live queue artifacts without importing agents or touching experiment state."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path

from scripts.utils.benchmark_metrics import APPWORLD_METRIC, reported_score

from .ace_study_progress import ace_studies
from .evaluation_progress import evaluations
from .parent_outcomes import first_parent_outcome
from .proposal_monitor import evidence, retrieval_summary
from .result_tables import build_table
from .store import JournalError

METHODS = {"ace", "premise", "ps_st_pareto", "gepa", "skillopt"}
VARIANT_LABELS = {"peace_no_alignment_aud": "PEACE · w/o alignment · AUD",
                  "peace_aux_pareto_aud": "PEACE · 보조지표 · Pareto 2→4→2 · AUD",
                  "peace_aux_alignment_product_aud": "PEACE · 보조지표 · Product 1→4→1 · AUD",
                  "peace_proposal_a_aud": "PEACE · A only",
                  "peace_proposal_b_aud": "PEACE · B only",
                  "peace_proposal_ab_aud": "PEACE · A+B",
                  "peace_proposal_random_aud": "PEACE · Random",
                  "ace_aud": "ACE · AUD", "peace_aud": "PEACE · AUD",
                  "gepa_val": "GEPA · 기본 val", "skillopt_val": "SkillOpt · 기본 val",
                  "gepa_train_as_val": "GEPA · train-as-val",
                  "skillopt_train_as_val": "SkillOpt · train-as-val"}


def measured_score(values, total, label, *, errors=0, metric=None):
    values = [float(value) for value in values if isinstance(value, (int, float))
              and math.isfinite(value) and 0 <= value <= 1]
    return {"value": sum(values) / len(values) if values else None, "count": len(values),
            "total": total, "errors": errors, "label": label,
            "complete": bool(total and len(values) == total and not errors),
            **({"metric": metric, "successes": sum(values)} if metric == APPWORLD_METRIC else {})}


def progress(done, total, *, errors=0, complete=None):
    total = max(0, int(total or 0))
    done = min(total, max(0, int(done or 0)))
    return {"done": done, "total": total, "errors": errors,
            "ratio": done / total if total else None,
            "complete": bool(total and done == total and not errors) if complete is None else complete}


def live_pid(pid, target):
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        root = Path("/proc") / str(pid)
        state = (root / "stat").read_text().rpartition(")")[2].split()[0]
        args = (root / "cmdline").read_bytes().decode().split("\0")
        return state not in {"Z", "X"} and any(arg == str(target) or arg.startswith(str(target) + "/") for arg in args)
    except (OSError, UnicodeError, IndexError):
        return False


class FileCache:
    """Cache projected fields, never whole trajectories, credentials or prompts."""

    def __init__(self):
        self.entries = {}
        self.issues = 0

    def read(self, path, keys=None, default=None, *, immutable=False):
        key = (path, keys)
        previous = self.entries.get(key)
        if immutable and previous:
            return previous[1]
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if previous and previous[0] == stamp:
                return previous[1]
            data = json.loads(path.read_text(encoding="utf-8"))
            if keys is not None:
                data = {name: data[name] for name in keys if name in data}
            if len(self.entries) > 30000:
                self.entries.clear()
            self.entries[key] = (stamp, data)
            return data
        except FileNotFoundError:
            return default
        except (OSError, ValueError, TypeError, UnicodeError):
            self.issues += 1
            return previous[1] if previous else default

    def modified(self, path, keys):
        entry = self.entries.get((path, keys))
        return entry[0][0] / 1e9 if entry else 0

    def parent_score(self, path, benchmark=None):
        key = (path, ("_first_parent_score", benchmark, APPWORLD_METRIC))
        previous = self.entries.get(key)
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if previous and previous[0] == stamp and previous[1].get("score") is not None:
                return previous[1]
            data = json.loads(path.read_text(encoding="utf-8"))
            parents = data.get("parents") or []
            parent = parents[0] if parents and parents[0] else {}
            result = {"task_id": data.get("task_id"),
                      "score": reported_score(benchmark or data.get("benchmark"), parent)}
            if result["score"] is None and path.stem.startswith("iter_"):
                iteration = int(path.stem.removeprefix("iter_"))
                if data.get("iteration", iteration) == iteration:
                    restored = first_parent_outcome(path.parent.parent, iteration, result["task_id"],
                                                    benchmark or data.get("benchmark"))
                    if restored:
                        result.update(restored)
            self.entries[key] = (stamp, result)
            return result
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError, UnicodeError):
            self.issues += 1
            return {}

    def policy_hash(self, path):
        key = (path, ("_policy_sha256",))
        try:
            stat = path.stat()
            previous = self.entries.get(key)
            stamp = (stat.st_mtime_ns, stat.st_size)
            if previous and previous[0] == stamp:
                return previous[1]
            data = json.loads(path.read_text(encoding="utf-8"))
            content = data.get("content", data.get("playbook"))
            value = hashlib.sha256(content.encode()).hexdigest() if isinstance(content, str) else None
            self.entries[key] = (stamp, value)
            return value
        except (OSError, ValueError, TypeError, UnicodeError):
            return None

    def retrieval(self, path):
        key = (path, ("_proposal_retrieval",))
        previous = self.entries.get(key)
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if previous and previous[0] == stamp:
                return previous[1]
            value = retrieval_summary(json.loads(path.read_text(encoding="utf-8")))
            self.entries[key] = (stamp, value)
            return value
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError, UnicodeError):
            self.issues += 1
            return previous[1] if previous else {}


class LiveProgress:
    def __init__(self, store, *, base_dir=None, ttl=3):
        self.store = store
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2]
        self.ttl = ttl
        self.files = FileCache()
        self.lock = threading.Lock()
        self.cached = None
        self.cached_at = 0
        self.completed_rows = {}
        self.latest_checkpoints = {}
        self.policy_shift_samples = {}
        self.policy_shift_router_cache = {}
        self.refresh_lock = threading.Lock()
        self.refresh_thread = None
        self.initial = None
        self.checkpoint_files = FileCache()
        self.checkpoint_lock = threading.Lock()
        self.checkpoint_cached = None
        self.checkpoint_cached_at = 0
        self.evaluation_cached = []
        self.evaluation_cached_at = 0
        self.study_files = FileCache()

    def snapshot(self):
        """Serve the last complete snapshot while the filesystem scan runs in the background."""
        with self.refresh_lock:
            if self.cached is None and self.initial is None:
                self.initial = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                                "refresh_seconds": 5, "summary": {"runs": 0, "running": 0}, "batches": [],
                                "checkpoint_queues": [], "policy_shifts": [],
                                "result_tables": self.result_tables([], []),
                                "read_issues": 0}
            stale = self.cached is None or time.monotonic() - self.cached_at >= self.ttl
            if stale and not (self.refresh_thread and self.refresh_thread.is_alive()):
                self.refresh_thread = threading.Thread(target=self.refresh, daemon=True, name="live-progress")
                self.refresh_thread.start()
            value = self.cached or self.initial
            if time.monotonic() - self.evaluation_cached_at >= 2:
                self.evaluation_cached = evaluations(self.base_dir, self.checkpoint_files, self.allowed, live_pid)
                self.evaluation_cached_at = time.monotonic()
            return {**value, "evaluations": self.evaluation_cached,
                    "ace_studies": ace_studies(self.base_dir, self.study_files, self.allowed, live_pid),
                    "checkpoint_queues": self.checkpoint_snapshot(),
                    "refreshing": bool(self.refresh_thread and self.refresh_thread.is_alive()),
                    "initializing": self.cached is None}

    def checkpoint_snapshot(self):
        """Refresh small queue dashboards independently of historical trajectory scans."""
        with self.checkpoint_lock:
            if self.checkpoint_cached is None or time.monotonic() - self.checkpoint_cached_at >= 2:
                self.checkpoint_cached = self.checkpoint_queues(files=self.checkpoint_files)
                self.checkpoint_cached_at = time.monotonic()
            return self.checkpoint_cached

    def refresh(self):
        try:
            self.get()
        except (OSError, ValueError, TypeError, KeyError):
            # A concurrent artifact rewrite must not discard the last complete snapshot.
            self.files.issues += 1

    def allowed(self, path):
        path = path.resolve()
        for root in self.store.allowed_roots:
            if path.is_relative_to(root) and not path.is_relative_to(self.store.data_dir):
                parts = path.relative_to(root).parts
                return not any(p.startswith(".") or p in {"venv", "credentials", "secrets"} for p in parts)
        return False

    def discover(self, documents):
        roots = {p.parent for p in (self.base_dir / "outputs").glob("*/launch_manifest.json")}
        for document in documents:
            if document.get("run_path"):
                path = Path(document["run_path"])
                roots.add(path if path.is_absolute() else self.base_dir / path)
        return sorted(path for path in roots if self.allowed(path) and (path / "launch_manifest.json").is_file())

    def proposal_summary(self, folder, mode, done):
        latest_path = folder / "iterations" / f"iter_{done:04d}.json"
        latest = self.files.retrieval(latest_path) if done and self.allowed(latest_path) else {}
        saved = {}
        for index in range(done, 0, -1):
            path = folder / "iterations" / f"iter_{index:04d}.json"
            if self.allowed(path):
                item = self.files.retrieval(path)
                if item.get("candidates"):
                    saved = item
                    break
        return {**saved, "mode": mode, "processed": done,
                "latest_status": latest.get("status"), "latest_reason": latest.get("reason"),
                "latest_error": latest.get("error_message")}

    def proposal_details(self, batch_id, run_id, iteration):
        root = next((root for root in self.discover(self.store.list()) if root.name == batch_id), None)
        if root is None:
            raise JournalError("실험 큐를 찾을 수 없습니다.", 404)
        plan = self.files.read(root / "launch_manifest.json", default={})
        job = next((job for job in plan.get("jobs", []) if job.get("id") == run_id), None)
        if not job or (job.get("proposal_retrieval") not in {"a", "b", "ab", "random"}
                       and not job.get("grounded_proposals")):
            raise JournalError("Proposal 검색 run을 찾을 수 없습니다.", 404)
        try:
            index = int(iteration)
        except (TypeError, ValueError) as exc:
            raise JournalError("올바른 iteration이 필요합니다.") from exc
        if not 1 <= index <= 100000:
            raise JournalError("올바른 iteration이 필요합니다.")
        folder = Path(job.get("train_dir") or Path(job["output"]) / "train")
        folder = folder if folder.is_absolute() else root / folder

        def read(position):
            path = folder / "iterations" / f"iter_{position:04d}.json"
            if not self.allowed(path):
                raise JournalError("허용되지 않은 기록 경로입니다.", 403)
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeError) as exc:
                raise JournalError("저장된 iteration 기록을 읽을 수 없습니다.", 404) from exc

        artifact = read(index)
        result = retrieval_summary(artifact)
        result.update(batch=batch_id, run=run_id, sources={}, queries={})
        for candidate in result["candidates"]:
            ref_id = candidate["reflection_id"]
            result["queries"][ref_id] = evidence(artifact, ref_id)
            for example in candidate["examples"]:
                identity = example["id"]
                if identity in result["sources"]:
                    continue
                position = example["iteration"]
                if not isinstance(position, int) or not 1 <= position < index:
                    raise JournalError("과거 검색 기록의 iteration이 올바르지 않습니다.")
                prefix, _, reflection_id = str(identity).partition(":")
                if prefix != f"{position:04d}":
                    raise JournalError("과거 검색 기록의 ID가 올바르지 않습니다.")
                source = evidence(read(position), reflection_id)
                if source.get("available") and (source["task_id"], source["benchmark"]) != (
                    example["task_id"], example["benchmark"]
                ):
                    raise JournalError("과거 검색 기록의 task가 일치하지 않습니다.")
                result["sources"][identity] = source
        return result

    def get(self):
        with self.lock:
            if self.cached is not None and time.monotonic() - self.cached_at < self.ttl:
                return self.cached
            self.files.issues = 0
            documents = self.store.list()
            checkpoint_queues = self.checkpoint_queues(include_hidden=True)
            self.latest_checkpoints = {}
            for queue in checkpoint_queues:
                for row in queue["rows"]:
                    key = (row["variant"], row["method"], row["benchmark"], row["seed"])
                    if row.get("run_variant") is not None:
                        key = (*key, row["run_variant"])
                    if row["point"] > self.latest_checkpoints.get(key, {}).get("point", -1):
                        self.latest_checkpoints[key] = row
            by_path = {str((self.base_dir / d["run_path"]).resolve()): d
                       for d in documents if d.get("run_path")}
            batches = []
            seen = set()
            for root in self.discover(documents):
                try:
                    batch = self.batch(root, by_path.get(str(root.resolve())), seen)
                    if batch:
                        batches.append(batch)
                except (OSError, ValueError, TypeError, KeyError):
                    self.files.issues += 1
            rows = [row for batch in batches for row in batch["runs"]]
            priority = {"running": 0, "error": 1, "stopped": 1, "paused": 2, "pending": 4, "completed": 3}
            for batch in batches:
                batch["runs"].sort(key=lambda r: (priority[r["state"]], r["method"], r["benchmark"], r["seed"]))
            batches.sort(key=lambda b: b["created_at"], reverse=True)
            batches.sort(key=lambda b: min(priority[r["state"]] for r in b["runs"]))
            states = Counter(row["state"] for row in rows)
            summary = {"runs": len(rows), "running": states["running"], "pending": states["pending"],
                       "paused": states["paused"],
                       "completed": states["completed"], "attention": states["error"] + states["stopped"],
                       "phases": dict(Counter(r["phase"] for r in rows if r["state"] == "running"))}
            for name in ("train", "test"):
                summary[name] = progress(sum(r[name]["done"] for r in rows), sum(r[name]["total"] for r in rows),
                                         errors=sum(r[name].get("errors", 0) for r in rows))
            policy_shifts = {}
            for queue in checkpoint_queues:
                followup = queue.get("policy_shift_followup")
                if queue["live_monitor_visible"] and followup:
                    policy_shifts[followup["id"]] = {
                        **followup, "source_journal_ids": [queue["journal_id"], *queue["source_journal_ids"]],
                    }
            self.cached = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "refresh_seconds": 5,
                           "summary": summary, "batches": batches,
                           "policy_shifts": list(policy_shifts.values()),
                           "checkpoint_queues": [queue for queue in checkpoint_queues
                                                 if queue["live_monitor_visible"]],
                           "result_tables": self.result_tables(batches, checkpoint_queues),
                           "read_issues": self.files.issues}
            self.cached_at = time.monotonic()
            return self.cached

    def result_tables(self, batches, queues):
        result = []
        for marker in sorted((self.base_dir / "outputs").glob("*/results_table_manifest.json")):
            if not self.allowed(marker):
                continue
            config = self.files.read(marker, default={})
            try:
                result.append(build_table(config, batches, queues))
            except (KeyError, TypeError, ValueError):
                self.files.issues += 1
        return sorted(result, key=lambda table: table.get("priority", 0), reverse=True)

    def policy_shift_rate(self, target, owner, done, remaining, phase):
        """Estimate inference throughput from observed progress, including idle time."""
        empty = {"per_minute": None, "eta_seconds": None, "window_seconds": 0}
        if phase != "probing":
            self.policy_shift_samples.pop(target, None)
            return empty
        identity = (owner.get("pid"), owner.get("start_ticks"))
        previous = self.policy_shift_samples.get(target)
        if previous is None or previous[0] != identity or previous[1][-1][1] > done:
            previous = (identity, deque())
            self.policy_shift_samples[target] = previous
        samples = previous[1]
        now = time.monotonic()
        samples.append((now, done))
        while len(samples) > 1 and samples[1][0] <= now - 600:
            samples.popleft()
        elapsed = now - samples[0][0]
        if elapsed < 10:
            return empty
        rate = (done - samples[0][1]) / elapsed
        return {"per_minute": rate * 60, "eta_seconds": remaining / rate if rate > 0 else None,
                "window_seconds": elapsed}

    def policy_shift_gpu_queries(self, url):
        if not url or urllib.parse.urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            return {}
        cached = self.policy_shift_router_cache.get(url)
        if cached and time.monotonic() - cached[0] < 3:
            return cached[1]
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                data = json.load(response)
            data = {key: data[key] for key in ("per_gpu_query_limit", "active") if key in data}
        except (OSError, ValueError, TypeError):
            data = {}
        self.policy_shift_router_cache[url] = (time.monotonic(), data)
        return data

    def policy_shift_followup(self, root):
        reservation = self.files.read(root / "policy_shift_followup.json", ("output", "journal_id"), {})
        if not reservation.get("output"):
            return None
        target = Path(reservation["output"])
        if not target.is_absolute() or not self.allowed(target):
            return None
        state = self.files.read(target / "status.json", ("phase", "workers", "counts", "queued"), {})
        owner = self.files.read(target / "controller_process.json", ("pid", "start_ticks"), {})
        prepared = self.files.read(target / "prepared.json", ("histories", "effective_requests"), {})
        plan = self.files.read(target / "plan.json", (
            "temperature", "workers", "per_gpu_query_limit", "query_router_url"), {})
        phase = state.get("phase", "planned")
        if not live_pid(owner.get("pid"), target) and phase not in {
            "completed", "completed_with_errors", "failed",
        }:
            phase = "stopped" if owner else "planned"
        counts = {key: value for key, value in state.get("counts", {}).items()
                  if key in {"pending", "running", "completed", "error"}}
        total = prepared.get("effective_requests", sum(counts.values()))
        done = counts.get("completed", 0) + counts.get("error", 0)
        modified = self.files.modified(target / "status.json", ("phase", "workers", "counts", "queued"))
        return {"id": target.name, "phase": phase, "journal_id": reservation.get("journal_id"), "counts": counts,
                "workers": state.get("workers", plan.get("workers")), "temperature": plan.get("temperature"),
                "per_gpu_query_limit": plan.get("per_gpu_query_limit"),
                "gpu_queries": self.policy_shift_gpu_queries(plan.get("query_router_url"))
                if phase == "probing" else {},
                "updated_at": datetime.fromtimestamp(modified, UTC).isoformat() if modified else None,
                "histories": prepared.get("histories"),
                "throughput": self.policy_shift_rate(target, owner, done, max(0, total - done), phase),
                "progress": progress(done, total, errors=counts.get("error", 0))}

    def checkpoint_queues(self, *, include_hidden=False, files=None):
        files = files or self.files
        queues = []
        for marker in (self.base_dir / "outputs").glob("*/checkpoint_queue_manifest.json"):
            root = marker.parent
            if not self.allowed(root):
                continue
            visibility = files.read(root / "live_monitor_visibility.json", default={})
            visible = visibility.get("visible", True) is not False
            if not visible and not include_hidden:
                continue
            data = files.read(root / "dashboard.json", default={})
            if data.get("kind") != "five_task_checkpoint_queue":
                continue
            plan = files.read(marker, ("journal_id", "source_journal_ids", "scope_label", "live_overview",
                                       "expected_checkpoints", "initial_reused_outcomes", "serving_gpu_indices"), {})
            rows = []
            latest = {}
            for row in data.get("rows", []):
                if isinstance(row.get("stream"), str) and isinstance(row.get("point"), int):
                    latest[row["stream"]] = max(latest.get(row["stream"], 0), row["point"])
            for row in data.get("rows", []):
                projected = {key: row.get(key) for key in (
                    "id", "stream", "method", "benchmark", "seed", "variant", "point", "selection",
                    "selected_val_score", "policy_hash", "alias", "state", "selected", "run_variant",
                )}
                projected["splits"] = {name: {key: values.get(key) for key in (
                    "done", "total", "running", "errors", "score",
                )} for name, values in row.get("splits", {}).items() if name in {"train", "val", "test"}}
                if projected["benchmark"] == "appworld":
                    # Shared CP actors already return official binary success.
                    for group in projected["splits"].values():
                        group["metric"] = APPWORLD_METRIC
                        if group.get("score") is not None:
                            group["successes"] = round(group["score"] * group["done"])
                if row.get("stream") in latest and row.get("point") == latest[row["stream"]]:
                    frozen = root / "frozen" / row["stream"] / f"cp{row['point']:03d}.json"
                    if self.allowed(frozen):
                        projected["policy_sha256"] = files.read(
                            frozen, ("content_sha256",), {}, immutable=True).get("content_sha256")
                rows.append(projected)
            controller_alive = live_pid(data.get("controller_pid"), root)
            phase = data.get("phase") if controller_alive else (
                data.get("phase") if data.get("phase") in {"completed", "error"} else "stopped")
            reservation = files.read(root / "resume_schedule.json", ("output",), {})
            schedule = {}
            if reservation.get("output") and self.allowed(Path(reservation["output"])):
                schedule_root = Path(reservation["output"])
                schedule = files.read(schedule_root / "state.json", ("phase", "at", "contexts", "error"), {})
                owner = files.read(schedule_root / "scheduler_process.json", ("pid",), {})
                if not controller_alive and live_pid(owner.get("pid"), schedule_root):
                    phase = schedule.get("phase", phase)
            queues.append({"id": root.name, "updated_at": data.get("updated_at"),
                           "live_monitor_visible": visible,
                           "journal_id": plan.get("journal_id"),
                           "source_journal_ids": plan.get("source_journal_ids", []),
                           "phase": phase, "resume_schedule": schedule,
                           "policy_shift_followup": self.policy_shift_followup(root),
                           "scope_label": plan.get("scope_label") or data.get("scope_label"),
                           "live_overview": bool(plan.get("live_overview")),
                           "expected_checkpoints": data.get("expected_checkpoints", plan.get("expected_checkpoints")),
                           "initial_reused_outcomes": plan.get("initial_reused_outcomes", 0),
                           "serving_gpu_indices": plan.get("serving_gpu_indices", []),
                           "workers": data.get("workers", 100), "task_counts": data.get("task_counts", {}),
                           "checkpoint_interval": data.get("checkpoint_interval", 5),
                           "gpu_queries": {key: value for key, value in data.get("gpu_queries", {}).items()
                                           if key in {"per_gpu_query_limit", "active", "maximum",
                                                      "completed", "errors", "per_endpoint_query_limits",
                                                      "waiting", "queue"}},
                           "checkpoints": len(rows), "skipped": data.get("skipped", 0), "rows": rows,
                           "invalidated_checkpoints": data.get("invalidated_checkpoints", 0),
                           "source_issues": len(data.get("source_issues", []))})
        return queues

    def batch(self, root, document, seen):
        plan = self.files.read(root / "launch_manifest.json", default={})
        status = self.files.read(root / "status.json", default={})
        schedule = self.files.read(root / "control_schedule.json", ("phase", "pause_at", "resume_at"), {})
        if status.get("phase") == "superseded" or (root / "superseded_by.json").exists():
            return None
        jobs = plan.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            return None
        saved = self.files.read(root / "progress.json", default={})
        saved_jobs = saved.get("jobs", [])
        if isinstance(saved_jobs, dict):
            current = {key: {**value, "id": key, "status": value.get("status", value.get("phase", "pending"))}
                       for key, value in saved_jobs.items() if isinstance(value, dict)}
        else:
            current = {j["id"]: j for j in saved_jobs if isinstance(j, dict) and "id" in j}
        owner = self.files.read(root / "controller_process.json", default={})
        owner_root = Path(owner.get("output") or root)
        owner_alive = self.allowed(owner_root) and live_pid(owner.get("identity", owner).get("pid"), owner_root)
        rows = []
        for planned in jobs:
            job = {**planned, **current.get(planned.get("id"), {})}
            if job.get("method") not in METHODS or not job.get("output"):
                continue
            output = Path(job["output"])
            output = (output if output.is_absolute() else root / output).resolve()
            if not self.allowed(output) or str(output) in seen:
                continue
            seen.add(str(output))
            rows.append(self.run(root, output, job, owner_alive, plan))
        if not rows:
            return None
        edit_modes = {job["playbook_edit_mode"] for job in jobs if job.get("playbook_edit_mode")}
        edit_mode = plan.get("playbook_edit_mode") or (
            next(iter(edit_modes)) if len(edit_modes) == 1 else "mixed" if edit_modes else "add_only")
        counts = Counter(row["state"] for row in rows)
        return {"id": root.name, "title": document["title"] if document else root.name,
                "journal_id": document["id"] if document else None,
                "model": plan.get("model", ""), "created_at": plan.get("created_at", ""),
                "setting": plan.get("mode", "isolated"),
                "playbook_edit_mode": edit_mode,
                "appworld_policy_sha256": plan.get("appworld_policy_sha256"),
                "temperature": plan.get("temperature_all_roles", plan.get("temperature", plan.get(
                    "profile", {}).get("actor_temperature"))),
                "inference_profile_sha256": plan.get("inference_profile_sha256"),
                "max_parallel": plan.get("max_parallel"),
                "server_max_num_seqs": plan.get("server_max_num_seqs"),
                "waiting_for": Path(plan.get("after_output") or plan["after_cka"]).name
                if (plan.get("after_output") or plan.get("after_cka"))
                and not plan.get("join_output") and status.get("phase", "").startswith("waiting") else None,
                "counts": dict(counts), "gpu_counts": status.get("gpu_counts", {}), "schedule": schedule, "runs": rows}

    def ace_train(self, folder):
        cfg = self.files.read(folder / "experiment_config.json", ("task_order",), {})
        order = [(x[0], str(x[1])) for x in cfg.get("task_order", [])]
        completed = set()
        metrics = folder / "online_metrics.jsonl"
        occurrences = Counter()
        if metrics.exists():
            for line in metrics.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    key = (row["benchmark_slug"], str(row["task_id"]))
                    completed.add(key)
                    occurrences[key] += 1
                except (ValueError, KeyError, TypeError):
                    break  # The last record can still be in flight.
        if len(order) != len(set(order)):
            # Repeated tasks in a later epoch are separate completed exposures.
            # Metrics are appended after curation; a session result alone may
            # still be waiting for its playbook update.
            count = 0
            for key in order:
                if not occurrences[key]:
                    break
                occurrences[key] -= 1
                count += 1
            return count, len(order)
        if order and set(order) <= completed:
            return len(order), len(order)
        for path in folder.glob("*/sessions/*/results.json"):
            result = self.files.read(path, ("status",), {})
            if not (path.parent / "agent/playbook_checkpoint.json").is_file() and result.get("status") not in {
                "error", "cancelled"
            }:
                continue  # Scoring is saved before curator completion.
            config = self.files.read(path.parent / "config.json", ("benchmark", "task_id"), {})
            if "benchmark" in config and "task_id" in config:
                completed.add((config["benchmark"], str(config["task_id"])))
        count = 0
        for key in order:
            if key not in completed:
                break
            count += 1
        return count, len(order)

    def legacy(self, root, output, job):
        benchmark = job.get("benchmark")
        folder = Path(job.get("train_dir", output / "train"))
        if not folder.is_absolute():
            folder = root / folder
        if not self.allowed(folder):
            return progress(0, job.get("train_count")), progress(0, 0), None, {}
        if job["method"] == "ace":
            done, total = self.ace_train(folder)
        else:
            state = self.files.read(folder / "state.json", ("cursor", "config"), {})
            done = state.get("cursor", 0)
            total = len(state.get("config", {}).get("order", []))
        train = progress(done, total or job.get("train_count", 0))
        online = {}
        if job["method"] == "ace":
            path = folder / "online_metrics.jsonl"
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines() if path.exists() else []):
                try:
                    item = json.loads(line)
                except ValueError:
                    break
                if item.get("status") not in {"error", "cancelled"}:
                    key = index if job.get("epoch1_count") else str(item.get("task_id"))
                    online[key] = reported_score(benchmark, item)
        else:
            for index in range(1, int(done) + 1):
                item = self.files.parent_score(folder / "iterations" / f"iter_{index:04d}.json", benchmark)
                if item.get("task_id") is not None:
                    key = index if job.get("epoch1_count") else str(item["task_id"])
                    online[key] = item.get("score")
        expected_ids = list(map(str, job.get("test_ids", [])))
        expected = set(expected_ids)
        records = {}
        for path in (output / "evaluation/records").glob("*.json"):
            row = self.files.read(path, ("task_id", "status", "score", "success", "session_status", "tgc",
                                         "score_metric", "budget_failure", "experiment_budget_failure",
                                         "index", "official_reward", "session_id", "run_id",
                                         "actor_prompt_sha256", "infrastructure_error"), {})
            task = str(row.get("task_id", ""))
            if "status" not in row:
                # The shared-reflector worker commits an official outcome atomically,
                # without a status field. A numeric score alone is not completion evidence.
                index = row.get("index")
                reward = row.get("official_reward")
                prompt_hash = row.get("actor_prompt_sha256")
                if not (type(index) is int and 0 <= index < len(expected_ids) and expected_ids[index] == task
                        and isinstance(row.get("success"), bool) and type(row.get("score")) in {int, float}
                        and row["score"] == int(row["success"])
                        and type(reward) in {int, float} and math.isfinite(reward) and 0 <= reward <= 1
                        and isinstance(prompt_hash, str) and len(prompt_hash) == 64
                        and all(char in "0123456789abcdef" for char in prompt_hash)
                        and all(isinstance(row.get(key), str) and row[key] for key in ("session_id", "run_id"))
                        and not row.get("infrastructure_error")):
                    continue
                row = {**row, "status": "completed"}
            if task in expected and (row.get("status") == "error" or (
                row.get("status") == "completed" and isinstance(row.get("score"), (int, float))
            )):
                records[task] = {**row, "score": reported_score(benchmark, row)}
        errors = sum(row["status"] == "error" for row in records.values())
        test = progress(len(records), len(expected), errors=errors)
        metric = APPWORLD_METRIC if benchmark == "appworld" else None
        scores = {"train": measured_score(list(online.values()), train["total"], "Online · 학습 평균", metric=metric),
                  "test": measured_score([r.get("score") for r in records.values()
                                          if r["status"] == "completed"], len(expected), "Offline · Test 성공률",
                                         errors=errors, metric=metric)}
        return train, test, None, scores

    def optimized(self, output, job, plan):
        cfg = self.files.read(output / "config.json", ("train", "validation", "test"), {})
        if not cfg and "train_ids" in job:
            validation_ids = (job.get("train_ids", []) if job.get("validation_source") == "train"
                              else job.get("val_ids", []))
            cfg = {name: [{"task_id": str(task)} for task in ids]
                   for name, ids in (("train", job["train_ids"]), ("validation", validation_ids),
                                     ("test", job.get("test_ids", [])))}
        if not cfg:
            snapshot = Path(plan.get("snapshot", self.base_dir))
            manifest_path = snapshot / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
            manifest = self.files.read(manifest_path, default={}) if self.allowed(manifest_path) else {}
            section = manifest.get(f"{job.get('benchmark')}_{job.get('seed')}", {})
            cfg = {name: [{"task_id": str(task)} for task in section.get(key, [])]
                   for name, key in (("train", "train"), ("validation", "val"), ("test", "test"))}
        def encounter(item):
            return item.get("task_id"), item.get("epoch", 1)

        expected_train = {encounter(item) for item in cfg.get("train", [])}
        expected_test = {item["task_id"] for item in cfg.get("test", [])}
        encountered = set()
        online_scores = {}
        for path in (output / "online").glob("*.json"):
            row = self.files.read(path, ("task_id", "epoch", "hard"), {}, immutable=True)
            if encounter(row) in expected_train:
                encountered.add(encounter(row))
                online_scores[encounter(row)] = row.get("hard")
        training_complete = (output / "training_result.json").is_file()
        train = progress(len(encountered), len(expected_train), complete=(
            bool(expected_train) and training_complete and len(encountered) == len(expected_train)))
        summary = self.files.read(output / "summary.json", ("complete", "tasks", "score", "prompt_sha256"), {})
        policy = self.files.policy_hash(output / "frozen_instruction.json") if summary else None
        if (summary.get("complete") is True and summary.get("tasks") == len(expected_test)
                and policy and summary.get("prompt_sha256") == policy
                and isinstance(summary.get("score"), (int, float)) and 0 <= summary["score"] <= 1):
            # The final writer commits this summary only after every official test result.
            # Avoid reopening thousands of historical candidate rollouts on each server start.
            score = {"value": summary["score"], "count": len(expected_test), "total": len(expected_test),
                     "errors": 0, "complete": bool(expected_test), "label": "Offline · Test 성공률"}
            return train, progress(len(expected_test), len(expected_test)), None, {
                "train": measured_score(list(online_scores.values()), len(expected_train), "Online · 학습 성공률"),
                "test": score,
            }
        groups = {}
        folder = output / "rollouts"
        for directory in list(folder.iterdir()) if folder.is_dir() else []:
            path = directory / "request.json"
            request_keys = ("label", "item", "prompt_sha256")
            request = self.files.read(path, request_keys, {}, immutable=True)
            if not request.get("item") or not request.get("label"):
                continue
            label, item = request["label"], request["item"]
            key = (label, request.get("prompt_sha256"))
            group = groups.setdefault(key, {"label": label, "split": item["split"], "time": 0,
                                            "ids": set(), "done": set(), "scores": {}, "encounters": set()})
            group["time"] = max(group["time"], self.files.modified(path, request_keys))
            group["ids"].add(item["task_id"])
            group["encounters"].add(encounter(item))
            result = self.files.read(path.parent / "result.json", ("id", "hard"), {}, immutable=True)
            if result.get("id") == item["task_id"] and result.get("hard") in (0, 1):
                group["done"].add(item["task_id"])
                group["scores"][item["task_id"]] = result["hard"]
        test_group = next((g for g in groups.values() if g["label"] in {"final_test", "epoch2_final_test"}), {})
        test = progress(len(set(test_group.get("done", set())) & expected_test), len(expected_test))
        scores = {"train": measured_score(list(online_scores.values()), len(expected_train), "Online · 학습 성공률"),
                  "test": measured_score([value for task, value in test_group.get("scores", {}).items()
                                          if task in expected_test], len(expected_test), "Offline · Test 성공률")}
        validation = [group for group in groups.values() if group["split"] == "val"]
        if validation:
            latest_validation = max(validation, key=lambda group: group["time"])
            expected_validation = {item["task_id"] for item in cfg.get("validation", [])}
            scores["val"] = measured_score([value for task, value in latest_validation["scores"].items()
                                             if task in expected_validation], len(expected_validation),
                                            "선택 평가 · Train" if job.get("validation_source") == "train"
                                            else "후보 Offline-val 성공률")
        if not groups:
            return train, test, None, scores
        latest = max(groups.values(), key=lambda g: g["time"])
        split = latest["split"]
        if split == "val":
            total = len(cfg.get("validation", []))
            name = ("초기 선택 평가 · Train" if not encountered else "후보 선택 평가 · Train") if (
                job.get("validation_source") == "train") else (
                "초기 Offline-val" if not encountered else "후보 Offline-val")
        elif split == "test":
            total, name = len(expected_test), "Test"
        else:
            positions = {encounter(item): index for index, item in enumerate(cfg.get("train", []))}
            indices = [positions[item] for item in latest["encounters"] if item in positions]
            start = (min(indices) // 5) * 5 if indices else 0
            total, name = min(5, len(expected_train) - start), "학습 rollout"
        current = {**progress(len(latest["done"]), total), "label": name,
                   "phase": {"val": "validation", "test": "test", "train": "training"}[split]}
        if split == "train" and current["done"] == total:
            current.update(phase="updating", label="지침 분석·업데이트", indeterminate=True)
        return train, test, current, scores

    def checkpoint_validation(self, row, root):
        key = (root.name, row["method"], row["benchmark"], row["seed"])
        checkpoint = self.latest_checkpoints.get((*key, row.get("variant")), self.latest_checkpoints.get(key))
        if checkpoint and (row["method"] in {"ace", "premise", "ps_st_pareto"} or row["state"] == "completed"):
            value = checkpoint["splits"].get("val", {})
            row["scores"]["val"] = {
                "value": value.get("score"), "count": value.get("done", 0), "total": value.get("total", 0),
                "errors": value.get("errors", 0), "label": f"CP{checkpoint['point']} Offline-val",
                "complete": bool(value.get("total") and value.get("done") == value["total"]
                                 and not value.get("errors")),
            }
        if row["benchmark"] == "appworld":
            for split, score in row["scores"].items():
                score["metric"] = APPWORLD_METRIC
                phase = {"train": "Online", "test": "Offline", "val": "Offline-val"}[split]
                if split == "val" and row.get("validation_source") == "train" and not checkpoint:
                    phase = "선택 평가 · Train"
                score["label"] = f"{phase} · TGC" + (
                    f" · CP{checkpoint['point']}" if split == "val" and checkpoint else ""
                )
                if score.get("value") is not None:
                    score["successes"] = round(score["value"] * score["count"])
        return row

    def run(self, root, output, job, owner_alive, plan):
        raw = job.get("status", "pending")
        if raw == "completed" and output in self.completed_rows:
            return self.checkpoint_validation(self.completed_rows[output], root)
        optimized = job["method"] in {"gepa", "skillopt"}
        training_output = Path(job.get("train_dir", output))
        if not training_output.is_absolute():
            training_output = root / training_output
        if not self.allowed(training_output):
            training_output = output
        if optimized:
            train, test, current, scores = self.optimized(training_output, job, plan)
        else:
            train, test, current, scores = self.legacy(root, output, job)
        worker = self.files.read(output / "job_status.json", ("phase",), {}) or self.files.read(
            output / "status.json", ("phase",), {})
        alive = live_pid(job.get("pid"), output) or bool(job.get("command") and live_pid(job.get("pid"), root))
        external = self.files.read(output / "evaluation/live_evaluation.json", (
            "phase", "controller_pid", "queue_output", "policy_sha256",
        ), {})
        external_root = Path(external.get("queue_output") or output)
        external_valid = bool(external.get("policy_sha256") and self.allowed(external_root)
                              and external["policy_sha256"] == self.files.policy_hash(
                                  output / "evaluation/frozen_checkpoint.json"))
        state = "pending"
        phase = "pending"
        if raw == "paused":
            state, phase = "paused", "paused"
        elif raw in {"error", "training_error", "evaluation_error"} or worker.get("phase") == "error":
            state, phase = "error", "error"
        elif raw == "completed" and train["complete"] and test["complete"]:
            state, phase = "completed", "completed"
        elif external_valid and train["complete"] and test["complete"]:
            state, phase = "completed", "completed"
        elif external_valid and external.get("phase") == "running" and train["complete"]:
            state, phase = ("running", "test") if live_pid(external.get("controller_pid"), external_root) else (
                "stopped", "stopped")
        elif raw in {"training", "running", "evaluating"}:
            if alive or (raw == "evaluating" and owner_alive):
                state = "running"
                phase = "test" if raw == "evaluating" or worker.get("phase") == "test" or (
                    optimized and (training_output / "frozen_instruction.json").exists()) else (
                    current["phase"] if current else "training")
            else:
                state, phase = "stopped", "stopped"
        elif test["errors"]:
            state, phase = "error", "error"
        elif train["complete"]:
            state, phase = "pending", "test_pending"
        if optimized:
            protocol = self.files.read(
                training_output / "config.json", ("split_manifest_sha256", "experiment_limits"), {})
        else:
            if job["method"] == "ace":
                configuration = self.files.read(training_output / "experiment_config.json", ("experiment_limits",), {})
            else:
                configuration = self.files.read(training_output / "state.json", ("config",), {}).get("config", {})
            limits = configuration.get("experiment_limits", plan.get("experiment_limits", {}))
            protocol = {"split_manifest_sha256": plan.get("manifest_sha256"),
                        "experiment_limits": limits.get(job.get("benchmark"), {})}
        row = {"id": job["id"], "method": job["method"], "benchmark": job.get("benchmark", ""),
               "seed": job.get("seed"), "variant": job.get("variant"),
               "variant_label": VARIANT_LABELS.get(job.get("variant")),
               "validation_source": job.get("validation_source"), "gpu": job.get("gpu"),
               "state": state, "phase": phase, "train": train, "test": test,
               "current": current if state == "running" else None, "pid": job.get("pid") if alive else None,
               "scores": scores,
               "frozen_policy_sha256": self.files.policy_hash(
                   training_output / "frozen_instruction.json" if optimized
                   else output / "evaluation/frozen_checkpoint.json"),
               "split_manifest_sha256": protocol.get("split_manifest_sha256"),
               "actor_limits": protocol.get("experiment_limits", {}),
               "invalidated": (output / "invalidated.json").exists(),
               "started_at": job.get("started_at"), "train_unit": "태스크 노출" if optimized else "태스크 완료"}
        if job.get("proposal_retrieval") in {"a", "b", "ab", "random"}:
            row["proposal_retrieval"] = self.proposal_summary(
                training_output, job["proposal_retrieval"], train["done"])
            if job.get("candidate_evaluation") == "span_policy_shift":
                row["variant_label"] = (f"PEACE · {job['proposal_retrieval'].upper()} · policy shift"
                                        f" · past {job['past_count']}")
            if raw == "pending" and job.get("external_predecessor"):
                row["proposal_retrieval"]["waiting_for"] = job["external_predecessor"]["id"]
        if job.get("role_temperatures"):
            row["role_temperatures"] = job["role_temperatures"]
            row["variant_label"] = (
                f"PEACE · policy shift · past {job['past_count']}"
                if job.get("candidate_evaluation") == "span_policy_shift" else "PEACE · AUD w/o alignment")
        if job.get("grounded_proposals"):
            row["variant_label"] = "PEACE · 검증된 대조 proposal · A+B"
            row["proposal_retrieval"] = {**self.proposal_summary(
                training_output, job.get("grounded_retrieval", "ab"), train["done"]), "grounded": True}
            if raw == "pending" and job.get("external_predecessors"):
                row["proposal_retrieval"]["waiting_for"] = "기본 PEACE seeds42-44의 학습·최종 Test 완료"
        if state == "completed":
            self.completed_rows[output] = row
        return self.checkpoint_validation(row, root)
