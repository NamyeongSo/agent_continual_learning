# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Run repeated, sequential SWE-Milestone experiments with a fresh ACE per repository."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOS = ("ripgrep", "navidrome", "dubbo", "go-zero", "nushell", "element-web", "scikit-learn")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_plan(rows: list[dict], repos: list[str], repeats: int, settings: dict) -> dict:
    jobs = []
    for repeat in range(1, repeats + 1):
        for repo in repos:
            selected = [row for row in rows if row["repo_alias"] == repo]
            if not selected:
                raise ValueError(f"No milestones found for {repo}")
            jobs.append({
                "name": f"repeat{repeat}_{repo}",
                "repeat": repeat,
                "repo": repo,
                "task_ids": [row["task_id"] for row in selected],
                "graded_milestones": sum(row["graded"] for row in selected),
                "ungraded_milestones": sum(not row["graded"] for row in selected),
            })
    return {
        "benchmark": "swemilestone",
        "method": "ace",
        "mode": "isolated",
        "actor": "terminus2",
        "repeats": repeats,
        "repos": repos,
        "max_parallel_repositories": 1,
        "context_recovery": "on_overflow",
        "proactive_summarization_threshold": 0,
        "settings": settings,
        "jobs": jobs,
        "total_sessions": sum(len(job["task_ids"]) for job in jobs),
        "standard_deviation_ddof": 1,
    }


def describe_samples(samples: list[dict], repeats: int) -> dict:
    scores = [sample["score"] for sample in samples]
    return {
        "samples": samples,
        "n": len(scores),
        "mean": statistics.mean(scores) if scores else None,
        "std": statistics.stdev(scores) if len(scores) > 1 else None,
        "complete": len(scores) == repeats,
    }


def legacy_context_recovery_errors(job_output: Path) -> list[dict]:
    """Identify failed Harbor recovery loops in retained, older trajectories."""
    errors = []
    for path in (job_output / "sessions").glob("*/benchmark/terminus/trajectory.json"):
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        consecutive = peak = total = 0
        for step in trajectory.get("steps", []):
            if step.get("source") != "agent":
                continue
            synthetic = (
                step.get("message") == "Technical difficulties. Please continue with the task."
                and not step.get("metrics", {}).get("prompt_tokens", 0)
                and not step.get("metrics", {}).get("completion_tokens", 0)
            )
            consecutive = consecutive + 1 if synthetic else 0
            total += int(synthetic)
            peak = max(peak, consecutive)
        if peak < 3:
            continue
        session = json.loads(path.parents[2].joinpath("session.json").read_text(encoding="utf-8"))
        errors.append({
            "task_id": session["task_id"],
            "error": "Terminus-2 context recovery failed: repeated synthetic replies in legacy trajectory",
            "synthetic_turns": total,
            "maximum_consecutive_synthetic_turns": peak,
        })
    return errors


def read_job_result(plan: dict, output: Path, job: dict) -> dict | None:
    """Use a provenance-checked recovery result while retaining the original run."""
    path = output / job["name"] / "experiment_result.json"
    if not path.exists():
        return None
    index_path = output / "recovery_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entry = index.get("jobs", {}).get(job["name"])
        if entry:
            manifest = output / "launch_manifest.json"
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != index["source_manifest_sha256"]:
                raise ValueError("SWE-Milestone recovery source manifest changed")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["source_result_sha256"]:
                raise ValueError("SWE-Milestone original result changed during recovery")
            repaired = Path(entry["result_path"])
            repair_plan = json.loads(repaired.parents[1].joinpath("launch_manifest.json").read_text())
            repair_job = next(row for row in repair_plan["jobs"] if row["name"] == job["name"])
            if repair_job != job or repair_plan["settings"] != plan["settings"]:
                raise ValueError("SWE-Milestone recovery task order or model settings differ")
            if hashlib.sha256(repaired.read_bytes()).hexdigest() != entry["result_sha256"]:
                raise ValueError("SWE-Milestone recovery result changed")
            path = repaired
    result = json.loads(path.read_text(encoding="utf-8"))
    recovery_errors = legacy_context_recovery_errors(path.parent)
    if recovery_errors:
        result["valid_for_statistics"] = False
        result["infrastructure_errors"] = [*result.get("infrastructure_errors", []), *recovery_errors]
    return result


def summarize(plan: dict, output: Path) -> dict:
    available = {}
    excluded = []
    for job in plan["jobs"]:
        result = read_job_result(plan, output, job)
        if result is None:
            continue
        if result["valid_for_statistics"]:
            available[(job["repeat"], job["repo"])] = result
        else:
            excluded.append({"job": job["name"], "errors": result["infrastructure_errors"]})
    repositories = {}
    for repo in plan["repos"]:
        samples = [
            {"repeat": repeat, "score": available[(repeat, repo)]["score"],
             "missing_graded_milestones": available[(repeat, repo)]["missing_graded_milestones"]}
            for repeat in range(1, plan["repeats"] + 1)
            if (repeat, repo) in available
        ]
        repositories[repo] = describe_samples(samples, plan["repeats"])
    overall_samples = [
        {"repeat": repeat,
         "score": statistics.mean(available[(repeat, repo)]["score"] for repo in plan["repos"])}
        for repeat in range(1, plan["repeats"] + 1)
        if all((repeat, repo) in available for repo in plan["repos"])
    ]
    overall = describe_samples(overall_samples, plan["repeats"])
    return {
        "updated_at": utc_now(),
        "expected_repeats": plan["repeats"],
        "standard_deviation_ddof": 1,
        "overall_score_definition": "Unweighted mean of official repository scores within each repeat",
        "missing_score_policy": "Unsubmitted graded milestones have score zero in the official denominator",
        "infrastructure_error_policy": "Exclude affected jobs and incomplete overall repeats; never fill with zero",
        "repositories": repositories,
        "overall": overall,
        "excluded_jobs": excluded,
        "complete": overall["complete"],
    }


def write_summary(plan: dict, output: Path) -> None:
    summary = summarize(plan, output)
    write_json(output / "summary.json", summary)
    repeats = plan["repeats"]
    headers = ["Repository", *[f"Repeat {i}" for i in range(1, repeats + 1)], "Mean (%)", "Sample SD", "n"]
    lines = ["# SWE-Milestone isolated ACE results", "", "Scores are percentages; sample SD uses ddof=1.", "",
             "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for label, result in [*summary["repositories"].items(), ("Overall", summary["overall"])]:
        by_repeat = {sample["repeat"]: sample["score"] for sample in result["samples"]}
        values = [f"{100 * by_repeat[i]:.3f}" if i in by_repeat else "pending" for i in range(1, repeats + 1)]
        mean = f"{100 * result['mean']:.3f}" if result["mean"] is not None else "pending"
        std = f"{100 * result['std']:.3f}" if result["std"] is not None else "pending"
        lines.append("| " + " | ".join([label, *values, mean, std, str(result["n"])]) + " |")
    lines.extend(["", "Unsubmitted graded milestones retain zero in the official repository denominator.",
                  "Jobs with infrastructure errors are excluded and listed in summary.json.",
                  "A final three-repeat estimate requires n=3; pending entries are never treated as zero.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_worker(output: Path, name: str) -> int:
    from exgentic import ACEAgent, SWEMilestoneBenchmark, evaluate
    from exgentic.agents.ace.playbook_store import PlaybookStore
    from exgentic.core.orchestrator.observer import Observer
    from exgentic.core.types import ModelSettings

    plan = json.loads((output / "launch_manifest.json").read_text(encoding="utf-8"))
    job = next(job for job in plan["jobs"] if job["name"] == name)
    settings = plan["settings"]
    job_output = output / name
    job_output.mkdir(parents=True, exist_ok=True)
    if (job_output / "experiment_result.json").exists() or (job_output / "sessions").exists():
        raise RuntimeError("This repository run already has artifacts; use a fresh experiment output directory")
    os.environ["OPENAI_API_BASE"] = settings["api_base"]
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["DOCKER_HOST"] = settings["docker_host"]
    os.environ["EXGENTIC_LLM_TRACE_FORMAT"] = "dedup_v1"
    PlaybookStore.reset_all()

    class ProgressObserver(Observer):
        def __init__(self):
            super().__init__()
            self.state = {"job": name, "completed_sessions": 0, "planned_sessions": len(job["task_ids"])}

        def update(self, phase, task_id=None):
            self.state.update(phase=phase, updated_at=utc_now())
            if task_id is not None:
                self.state["current_task"] = str(task_id)
            write_json(job_output / "progress.json", self.state)

        def on_session_creation(self, session):
            self.update("preparing", session.task_id)

        def on_session_start(self, session, agent, observation):
            self.update("acting", session.task_id)

        def on_session_scoring(self, session):
            self.update("grading", session.task_id)

        def on_session_success(self, session, score, agent):
            self.state["completed_sessions"] += 1
            self.state["last_score"] = score.score
            self.update("learning", session.task_id)

        def on_session_error(self, session, error):
            self.state["last_error"] = str(error)
            self.update("session_error", session.task_id)

        def on_session_execution_error(self, session_config, error):
            self.state["last_error"] = str(error)
            self.update("execution_error", session_config.task_id)

    observer = ProgressObserver()
    observer.update("starting")
    benchmark = SWEMilestoneBenchmark(
        subset=job["repo"], actor="terminus2", seed=settings["seed"],
        max_interactions=settings["max_turns"], agent_timeout_sec=settings["agent_timeout_sec"],
        model_info=settings["model_info"], docker_host=settings["docker_host"],
    )
    agent = ACEAgent(
        model=settings["model"], curator_model=settings["model"],
        benchmark_id=f"swemilestone_{job['repo']}", shuffle_mode="isolated", runner="direct",
        curator_frequency=1, training_time=settings["training_feedback"], enable_thinking=False,
        model_settings=ModelSettings(max_tokens=settings["max_tokens"]),
    )
    results = evaluate(
        benchmark=benchmark, agent=agent, task_ids=job["task_ids"], max_workers=1,
        output_dir=str(output), run_id=name, max_steps=200, max_actions=200, observers=[observer],
    )
    metadata = results.benchmark_results or {}
    metrics = metadata.get("metrics", {})
    errors = []
    for result in results.session_results:
        if result.status in ("error", "cancelled"):
            error = result.details.get("session_metadata", {}).get("error", result.status)
            if "is blocked; submit prerequisites first" not in str(error):
                errors.append({"task_id": result.task_id, "error": str(error)})
    if len(results.session_results) != len(job["task_ids"]):
        errors.append({"error": "The run did not process every planned milestone"})
    payload = {
        "job": name, "repo": job["repo"], "repeat": job["repeat"],
        "score": results.benchmark_score,
        "valid_for_statistics": not errors and results.benchmark_score is not None,
        "infrastructure_errors": errors,
        "partial": metrics.get("partial", True),
        "missing_graded_milestones": metrics.get("missing_graded_milestones"),
        "finished_at": utc_now(),
    }
    for store in PlaybookStore.list_stores().values():
        store.save_checkpoint(str(job_output / "playbook_final.json"))
        store.save_playbook_text(str(job_output / "playbook_final.txt"))
    write_json(job_output / "experiment_result.json", payload)
    observer.update("done" if payload["valid_for_statistics"] else "failed")
    return 0 if payload["valid_for_statistics"] else 1


def run_queue(plan: dict, output: Path) -> int:
    rows = [{**job, "status": "pending"} for job in plan["jobs"]]
    state = {"launcher_pid": os.getpid(), "started_at": utc_now(), "status": "running", "jobs": rows}
    stopped = False

    def request_stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    output.joinpath("logs").mkdir(exist_ok=True)
    write_summary(plan, output)
    for row in rows:
        if stopped:
            break
        log_path = output / "logs" / f"{row['name']}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--output-root", str(output),
                   "--worker-job", row["name"]]
        print(f"START {row['name']} ({len(row['task_ids'])} milestones)", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row.update(status="running", pid=process.pid, started_at=utc_now(), log=str(log_path))
            state["current_job"] = row["name"]
            write_json(output / "status.json", state)
            while process.poll() is None:
                if stopped:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                    break
                progress_path = output / row["name"] / "progress.json"
                if progress_path.exists():
                    row["progress"] = json.loads(progress_path.read_text(encoding="utf-8"))
                state["updated_at"] = utc_now()
                write_json(output / "status.json", state)
                time.sleep(5)
        row.update(status="cancelled" if stopped else ("done" if process.returncode == 0 else "failed"),
                   returncode=process.returncode, finished_at=utc_now())
        write_json(output / "status.json", state)
        write_summary(plan, output)
        print(f"{row['status'].upper()} {row['name']} exit={process.returncode}", flush=True)
    state.update(status="cancelled" if stopped else ("done" if all(r["status"] == "done" for r in rows)
                                                     else "finished_with_errors"), finished_at=utc_now())
    state.pop("current_job", None)
    write_json(output / "status.json", state)
    write_summary(plan, output)
    return 0 if state["status"] == "done" else 1


def retryable_repository_result(result: dict | None) -> bool:
    if not result or result.get("valid_for_statistics") is not False:
        return False
    errors = result.get("infrastructure_errors", [])
    return bool(errors) and all(
        (
            "SWE-Milestone backend failed (255)" in str(error.get("error", ""))
            and any(marker in str(error.get("error", "")) for marker in (
                "kex_exchange_identification:", "ssh_exchange_identification:",
            ))
        ) or any(marker in str(error.get("error", "")) for marker in (
            "residue prune: capture_filter does not match pinned repo config",
            "Terminus-2 context recovery failed",
        ))
        for error in errors
    )


def recover_infrastructure(output: Path, max_attempts: int = 3) -> int:
    """Wait for the sequential queue, then rerun affected repository streams from scratch."""
    manifest = output / "launch_manifest.json"
    source_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    state_path = output / "recovery_status.json"
    with (output / "recovery.lock").open("a") as recovery_lock:
        fcntl.flock(recovery_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(state_path, {"phase": "waiting_for_original_queue", "pid": os.getpid(), "updated_at": utc_now()})
        with (output / "launcher.lock").open("a") as launcher_lock:
            fcntl.flock(launcher_lock, fcntl.LOCK_EX)
            # A terminated launcher can leave its last child alive. Preserve one-repository concurrency.
            while original_worker_running(output):
                time.sleep(30)
            index_path = output / "recovery_index.json"
            index = (json.loads(index_path.read_text()) if index_path.exists()
                     else {"source_manifest_sha256": source_hash, "jobs": {}})
            if index["source_manifest_sha256"] != source_hash:
                raise ValueError("SWE-Milestone recovery source manifest changed")
            for attempt in range(1, max_attempts + 1):
                if hashlib.sha256(manifest.read_bytes()).hexdigest() != source_hash:
                    raise ValueError("SWE-Milestone source manifest changed")
                jobs = [job for job in plan["jobs"] if retryable_repository_result(read_job_result(plan, output, job))]
                if not jobs:
                    break
                stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
                repaired = output / "repairs" / f"{stamp}_attempt{attempt}"
                repaired.mkdir(parents=True)
                repair_plan = {**plan, "jobs": jobs, "total_sessions": sum(len(job["task_ids"]) for job in jobs)}
                write_json(repaired / "launch_manifest.json", repair_plan)
                write_json(state_path, {"phase": "running", "pid": os.getpid(), "attempt": attempt,
                                        "output_root": str(repaired), "jobs": [job["name"] for job in jobs],
                                        "updated_at": utc_now()})
                run_queue(repair_plan, repaired)
                for job in jobs:
                    result_path = repaired / job["name"] / "experiment_result.json"
                    if not result_path.exists():
                        continue
                    original = output / job["name"] / "experiment_result.json"
                    index["jobs"][job["name"]] = {
                        "result_path": str(result_path),
                        "source_result_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                    }
                write_json(index_path, index)
                write_summary(plan, output)
            remaining = [job["name"] for job in plan["jobs"]
                         if (read_job_result(plan, output, job) or {}).get("valid_for_statistics") is False]
            write_json(state_path, {"phase": "failed" if remaining else "complete", "pid": os.getpid(),
                                    "remaining": remaining, "updated_at": utc_now()})
            return 1 if remaining else 0


def original_worker_running(output: Path) -> bool:
    for directory in Path("/proc").glob("[0-9]*"):
        try:
            if directory.joinpath("stat").read_text().rpartition(")")[2].split()[0] in ("Z", "X"):
                continue
            argv = directory.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if (str(Path(__file__).resolve()) in argv and "--worker-job" in argv
                    and "--output-root" in argv and argv[argv.index("--output-root") + 1] == str(output)):
                return True
        except (OSError, UnicodeDecodeError, IndexError):
            continue
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repos", default=",".join(REPOS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="openai/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8006/v1"))
    parser.add_argument("--docker-host", default=os.environ.get("DOCKER_HOST"))
    parser.add_argument("--seed", type=int, default=42, help="Task-selection seed; the dependency order stays fixed")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--context-tokens", type=int, default=131072)
    parser.add_argument("--agent-timeout", type=int, default=10800)
    parser.add_argument("--no-training-feedback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-job", help=argparse.SUPPRESS)
    parser.add_argument("--retry-infrastructure-errors", action="store_true",
                        help="After the current queue, rerun streams affected by known SSH or capture config errors")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.retry_infrastructure_errors:
        if args.worker_job:
            parser.error("Recovery and --worker-job cannot be combined")
        return recover_infrastructure(output)
    if args.worker_job:
        return run_worker(output, args.worker_job)
    repos = args.repos.split(",")
    if len(set(repos)) != len(repos) or any(repo not in REPOS for repo in repos):
        parser.error("--repos must contain distinct supported repository names")
    if min(args.repeats, args.max_turns, args.max_tokens, args.context_tokens, args.agent_timeout) < 1:
        parser.error("Repeats and runtime limits must be positive")
    if not args.docker_host:
        parser.error("Set DOCKER_HOST or --docker-host to the SSH Docker endpoint")
    from exgentic.benchmarks.swemilestone.transport import MilestoneBackend

    rows = MilestoneBackend(args.docker_host).call("catalog", subset="all")
    settings = {
        "model": args.model, "api_base": args.api_base, "docker_host": args.docker_host,
        "seed": args.seed, "max_turns": args.max_turns, "max_tokens": args.max_tokens,
        "agent_timeout_sec": args.agent_timeout, "training_feedback": not args.no_training_feedback,
        "model_info": {"max_input_tokens": args.context_tokens, "max_output_tokens": args.max_tokens,
                       "input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
        "sampling": "Terminus backend defaults and ACE temperature=1; fresh calls for each repeat",
    }
    plan = build_plan(rows, repos, args.repeats, settings)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with (output / "launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another launcher already holds this output directory")
        if (output / "launch_manifest.json").exists():
            parser.error("This experiment output already has a manifest; choose a fresh output directory")
        write_json(output / "launch_manifest.json", plan)
        return run_queue(plan, output)


if __name__ == "__main__":
    raise SystemExit(main())
