# SPDX-License-Identifier: Apache-2.0
"""Run the independent, three-stage ACE AppWorld reflection experiment."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
ARMS = ("without_gt", "with_gt")
SPLITS = "scripts/utils/task_splits/bfcl_appworld_terminalbench2_fixed_balanced_order_seeds42_44_v1.json"
REFERENCES = "outputs/appworld_train50_reference_solutions_20261112/solutions.jsonl"


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def jobs_for_stage(task_ids, stage, stage3_repeats=5):
    jobs = []
    for repeat in range(5):
        for task_id in task_ids:
            source_id = f"{task_id}_r{repeat}"
            base = {"task_id": task_id, "source_repeat": repeat, "source_id": source_id, "stage": stage}
            if stage == 1:
                jobs.append({**base, "id": source_id})
            elif stage == 2:
                jobs.extend({**base, "id": f"{arm}_{source_id}", "arm": arm} for arm in ARMS)
            elif stage == 3:
                for evaluation_repeat in range(stage3_repeats):
                    for arm in ARMS:
                        context_id = f"{arm}_{source_id}"
                        jobs.append({**base, "id": f"{context_id}_e{evaluation_repeat}", "arm": arm,
                                     "context_id": context_id, "evaluation_repeat": evaluation_repeat})
            else:
                raise ValueError(f"Unknown stage {stage}")
    return jobs


def job_dir(output, job):
    return Path(output) / f"stage{job['stage']}" / job["id"]


def configure_environment(protocol):
    os.environ.update(
        CUDA_VISIBLE_DEVICES="1", OPENAI_API_BASE=protocol["base_url"], OPENAI_BASE_URL=protocol["base_url"],
        OPENAI_API_KEY="EMPTY", EXGENTIC_ACTOR_OUTPUT_UNLIMITED="1", LITELLM_LOCAL_MODEL_COST_MAP="True",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
    )
    for name in ("EXGENTIC_INFERENCE_PROFILE", "EXGENTIC_INFERENCE_PROFILE_SHA256"):
        os.environ.pop(name, None)
    if protocol.get("turn_limit_scoring") == "official_pct_at_limit":
        os.environ["EXGENTIC_APPWORLD_TURN_CAP_PCT"] = "1"
    else:
        os.environ.pop("EXGENTIC_APPWORLD_TURN_CAP_PCT", None)
    if protocol.get("benchmark_dependency_overlay"):
        overlay = protocol["benchmark_dependency_overlay"]
        paths = [overlay, *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
        os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(p for p in paths if p))


def role_call_overrides(protocol):
    """Keep ACE's explicit overrides consistent with the frozen request settings."""
    settings = protocol["model_settings"]
    thinking = settings.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking", True)
    if not isinstance(thinking, bool):
        raise ValueError("enable_thinking must be a boolean")
    temperature = settings.get("temperature", 1.0)
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError("Invalid role temperature")
    return {"enable_thinking": thinking, "temperature_override": temperature,
            "analysis_temperature_override": temperature}


def check_endpoint(protocol):
    """Read serving metadata and verify the local server's device before inference."""
    with urllib.request.urlopen(protocol["base_url"] + "/models", timeout=15) as response:
        models = json.load(response)
    model_id = protocol["model"].removeprefix("openai/")
    if model_id not in {item["id"] for item in models["data"]}:
        raise ValueError(f"Endpoint does not serve {model_id}")
    port = protocol["base_url"].split(":")[-1].split("/")[0]
    processes = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if b"serve" not in args or b"--port" not in args or port.encode() not in args:
                continue
            if not any(b"vllm" in arg for arg in args):
                continue
            env = dict(item.split(b"=", 1) for item in (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
            processes.append({"pid": int(proc.name), "command": b" ".join(args).decode(),
                              "cuda_visible_devices": env.get(b"CUDA_VISIBLE_DEVICES", b"").decode()})
        except (OSError, ValueError):
            continue
    if len(processes) != 1 or processes[0]["cuda_visible_devices"] != "1":
        raise ValueError(f"Cannot verify port {port} is exclusively on physical GPU 1: {processes}")
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True,
    )
    return {"checked_at": now(), "models": models, "server": processes[0], "gpus": gpus}


def prepare(output, config_path):
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK

    output = Path(output).resolve()
    if (output / "protocol.json").exists():
        raise FileExistsError("Prepared cohort already exists; use --run to resume it")
    protocol = read(config_path)
    policy = protocol["benchmark_kwargs"]["policy_override"]
    if digest(policy) != protocol["appworld_policy_sha256"]:
        raise ValueError("Policy snapshot hash mismatch")
    if (ROOT / protocol["appworld_policy_source"]).read_bytes() != policy.encode():
        raise ValueError("Teammate policy source differs from the frozen policy")
    split_path = ROOT / SPLITS
    split = read(split_path)["appworld_42"]
    task_ids = split["train"]
    if len(task_ids) != 50 or len(set(task_ids)) != 50:
        raise ValueError("Expected exactly the original 50 training tasks")
    if set(task_ids) & (set(split["val"]) | set(split["test"])):
        raise ValueError("Training split overlaps a held-out split")
    references = [json.loads(line) for line in (ROOT / REFERENCES).read_text().splitlines() if line.strip()]
    if len(references) != 50 or {row["task_id"] for row in references} != set(task_ids):
        raise ValueError("Reference task membership differs from train50")
    for row in references:
        grading = row["verification"]
        tracker = grading["evaluation"]
        if (not grading["success"] or len(tracker["passes"]) != tracker["num_tests"]
                or digest(row["code"]) != row["code_sha256"]):
            raise ValueError(f"Invalid reference {row['task_id']}")
    configure_environment(protocol)
    endpoint = check_endpoint(protocol)
    protocol.update(
        status="prepared", prepared_at=now(), task_ids=task_ids, seed=42, actor_context="shared-empty-v1",
        playbook_edit_mode="add_only", training_time=True, initial_playbook=DEFAULT_PLAYBOOK,
        split_manifest_source=str(split_path), split_manifest_sha256=digest(split_path.read_bytes()),
        gt_reference_source=REFERENCES, gt_reference_sha256=digest((ROOT / REFERENCES).read_bytes()),
        infrastructure_attempts=3, actor_trajectories_planned=2750, proposal_chains_planned=500,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        worktree_dirty=True, source_worktree=str(ROOT),
    )
    protocol["benchmark_kwargs"].update(interaction_mode="code_generation")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    # Match the environment in which the successful references were verified.
    # The installed SQLModel 0.0.47 has a documented AppWorld datetime regression;
    # reuse the pinned 0.0.24 overlay only inside this cohort's subprocesses.
    reference_root = ROOT / Path(REFERENCES).parent
    dependency_environment = read(reference_root / "source/verification_environment.json")
    if dependency_environment["packages"]["sqlmodel"] != "0.0.24":
        raise ValueError("Unexpected reference verification dependency version")
    overlay = output / "benchmark_dependencies"
    shutil.copytree(reference_root / "verification_deps", overlay,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    protocol.update(benchmark_dependency_overlay=str(overlay), sqlmodel_version="0.0.24",
                    benchmark_dependency_reason="Match verified GT environment; isolate datetime compatibility fix")
    write(output / "benchmark_dependency_hashes.json", {
        str(path.relative_to(overlay)): digest(path.read_bytes()) for path in overlay.rglob("*") if path.is_file()
    })
    write(output / "protocol.json", protocol)
    write(output / "endpoint_verification.json", endpoint)
    write(output / "references.json", {row["task_id"]: row for row in references})
    (output / "initial_playbook.txt").write_text(DEFAULT_PLAYBOOK)
    (output / "appworld_policy.txt").write_text(policy)
    shutil.copy2(split_path, output / "task_splits.json")
    snapshot = output / "source_snapshot"
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "installation"))
    (snapshot / "scripts/ace").mkdir(parents=True)
    shutil.copy2(__file__, snapshot / "scripts/ace" / Path(__file__).name)
    for name in ("rescore_saved_turn_cap.py", "migrate_turn_cap_pct.py",
                 "plot_initial_rollouts.py", "plot_initial_scatter_pass5_candles.py"):
        path = ROOT / "scripts/ace" / name
        if path.exists():
            shutil.copy2(path, snapshot / "scripts/ace" / name)
    shutil.copy2(ROOT / "pyproject.toml", snapshot / "pyproject.toml")
    write(output / "source_hashes.json", {
        str(path.relative_to(snapshot)): digest(path.read_bytes()) for path in snapshot.rglob("*") if path.is_file()
    })
    for stage in (1, 2, 3):
        jobs = jobs_for_stage(task_ids, stage, protocol.get("stage3_repeats_per_context", 5))
        if len(jobs) != protocol["stages"][stage - 1]["total_jobs"]:
            raise ValueError("Prepared queue size differs from the agreed protocol")
        write(output / f"stage{stage}_queue.json", jobs)
    write(output / "status.json", {"state": "prepared", "updated_at": now(), "stage": 0})
    return protocol


def pct_outcome(session, session_dir):
    """Use exact official test counts, retaining budget failures as measured zeroes."""
    details = session.details or {}
    metadata = details.get("session_metadata") or {}
    if metadata.get("experiment_budget_exceeded"):
        return {"pct": 0.0, "success": False, "budget_failure": metadata.get("experiment_budget_failure")}
    status = getattr(session.status, "value", session.status)
    if status in {"error", "cancelled"} or session.score is None or session.success is None:
        raise RuntimeError(f"No valid official outcome: {status}: {str(details)[:600]}")
    path = Path(session_dir) / "benchmark/scores.json"
    tracker = read(path).get("test_tracker") if path.exists() else metadata.get("test_tracker")
    if not isinstance(tracker, dict):
        raise RuntimeError("Missing official test tracker")
    total = tracker.get("num_tests", 0)
    passed, failed = len(tracker.get("passes", [])), len(tracker.get("failures", []))
    if total <= 0 or passed + failed != total:
        raise RuntimeError("Incomplete official test outcomes")
    return {"pct": passed / total, "success": bool(tracker["success"]), "passed_tests": passed,
            "total_tests": total, "difficulty": tracker.get("difficulty"),
            "turn_limit_reached": bool(metadata.get("turn_limit_reached")),
            "turn_cap_details": metadata.get("turn_cap_details")}


def actor_job(output, protocol, job, attempt_dir):
    from exgentic.agents.ace.ace_instance import ACEAgentInstance
    from exgentic.agents.ace.playbook_store import PlaybookStore
    from exgentic.core.types import RunConfig
    from exgentic.interfaces.lib.api import execute

    playbook = protocol["initial_playbook"]
    if job["stage"] == 3:
        parent = read(Path(output) / "stage2" / job["context_id"] / "result.json")
        if parent["task_id"] != job["task_id"] or parent["source_id"] != job["source_id"]:
            raise ValueError("Context/task pairing mismatch")
        playbook = Path(parent["updated_context_path"]).read_text()
        if digest(playbook) != parent["updated_context_sha256"]:
            raise ValueError("Updated context changed after proposal")
    PlaybookStore.reset_all()
    original_close = ACEAgentInstance.close

    def capture_close(instance):
        write(instance.paths.agent_dir / "ace_capture.json", {
            "task": instance.task, "context": instance.context, "actions": instance._action_log,
            "observations": instance._observation_log, "session_trace": instance._build_session_trace(),
            "training_feedback": instance._training_feedback, "playbook": instance._store.playbook,
            "messages": [m if isinstance(m, dict) else m.model_dump(mode="json") for m in instance.messages],
        })
        original_close(instance)

    ACEAgentInstance.close = capture_close
    try:
        run_id = "actor_" + uuid.uuid4().hex[:12]
        benchmark_kwargs = {**protocol["benchmark_kwargs"], "env_kwargs": {"experiment_name": run_id}}
        results = execute(config=RunConfig(
            benchmark="appworld", agent="ace", model=protocol["model"],
            task_ids=[job["task_id"]], output_dir=str(attempt_dir / "sessions"), run_id=run_id,
            max_workers=1, max_steps=40, max_actions=100, overwrite_sessions=True,
            benchmark_kwargs=benchmark_kwargs,
            agent_kwargs={
                "runner": "direct", "model_settings": protocol["model_settings"],
                "actor_context": protocol["actor_context"], **role_call_overrides(protocol),
                "allow_shared_thinking": True,
                "evaluation_mode": True, "training_time": True, "initial_playbook": playbook,
                "benchmark_id": "appworld", "shuffle_mode": "isolated", "enable_tool_shortlisting": False,
            },
        ))
    finally:
        ACEAgentInstance.close = original_close
    if len(results.session_results) != 1:
        raise RuntimeError("Actor job did not produce exactly one session result")
    session = results.session_results[0]
    session_dir = attempt_dir / "sessions" / run_id / "sessions" / session.session_id
    outcome = pct_outcome(session, session_dir)
    capture = session_dir / "agent/ace_capture.json"
    if not capture.is_file() or read(capture)["playbook"] != playbook:
        raise RuntimeError("Missing or mismatched captured ACE state")
    return {**outcome, "session_dir": str(session_dir), "capture_path": str(capture),
            "cost_reports": session.cost_reports, "initial_context_sha256": digest(playbook),
            "official_score": session.score, "session_status": str(session.status)}


def proposal_job(output, protocol, job, attempt_dir, role):
    from exgentic.agents.ace.ace_instance import ACEAgentInstance
    from exgentic.agents.ace.playbook_store import PlaybookStore
    from exgentic.agents.ace.playbook_utils import extract_json_from_text
    from exgentic.core.types import ModelSettings
    from exgentic.utils.paths import SessionPaths

    source = read(Path(output) / "stage1" / job["source_id"] / "result.json")
    if source["task_id"] != job["task_id"] or source["source_repeat"] != job["source_repeat"]:
        raise ValueError("Reflection/source trajectory pairing mismatch")
    capture = read(source["capture_path"])
    if capture["playbook"] != protocol["initial_playbook"]:
        raise ValueError("Initial trajectory contains learned context")
    instance = ACEAgentInstance(
        session_id="proposal", model=protocol["model"], curator_model=protocol["model"],
        benchmark_id="appworld", training_time=True, model_settings=ModelSettings(**protocol["model_settings"]),
        **role_call_overrides(protocol),
    )
    instance._paths = SessionPaths("proposal", "analysis", attempt_dir)
    instance.task, instance.context = capture["task"], capture["context"]
    instance._store = PlaybookStore(job["id"], protocol["initial_playbook"])
    instance._store.increment_session()
    instance._action_log, instance._observation_log = capture["actions"], capture["observations"]
    instance._training_feedback = capture["training_feedback"]
    if instance._training_feedback is None:
        raise ValueError("Initial trajectory has no captured official grading feedback")
    gt_trajectory = None
    if job["arm"] == "with_gt":
        reference = read(Path(output) / "references.json")[job["task_id"]]
        gt_trajectory = json.dumps(reference["trajectory"], ensure_ascii=False, indent=2)
    directory = job_dir(output, job)
    reflection_cache = directory / "reflection.json"
    if reflection_cache.exists():
        cached = read(reflection_cache)
        expected_gt = digest(gt_trajectory) if gt_trajectory is not None else None
        if (cached["source_capture_sha256"] != digest(Path(source["capture_path"]).read_bytes())
                or cached["gt_trajectory_sha256"] != expected_gt):
            raise ValueError("Cached reflection provenance mismatch")
        reflection = cached["reflection_for_curator"]
        instance._store.playbook = cached["playbook_after_tags"]
    else:
        role["name"] = instance._active_llm_stage = "reflector"
        reflection = instance._run_post_session_reflection(gt_trajectory=gt_trajectory)
        write(reflection_cache, {
            "reflection_for_curator": reflection, "playbook_after_tags": instance._store.playbook,
            "source_capture_sha256": digest(Path(source["capture_path"]).read_bytes()),
            "gt_trajectory_sha256": digest(gt_trajectory) if gt_trajectory is not None else None,
        })
    role["name"] = instance._active_llm_stage = "curator"
    instance._run_curator(reflection)
    updated = instance._store.playbook
    context_path = directory / "updated_context.txt"
    context_path.write_text(updated)
    # Native ACE can return the original context for empty/invalid edits; retain
    # those candidates rather than selecting only well-formed improvements.
    calls = [read(p) for p in sorted((attempt_dir / "llm_calls").glob("*.json"))]
    curator = next((c for c in reversed(calls) if c["role"] == "curator" and "response" in c), None)
    raw = curator["response"]["choices"][0]["message"].get("content", "") if curator else ""
    return {"updated_context_path": str(context_path), "updated_context_sha256": digest(updated),
            "context_changed": updated != protocol["initial_playbook"],
            "curator_parsed": extract_json_from_text(raw), "cost_reports": {"ace": instance.get_cost().model_dump()},
            "source_capture_path": source["capture_path"], "reflection_path": str(reflection_cache)}


def worker(output, job, attempt):
    """Run one job in a process isolated from all other tasks and contexts."""
    protocol = read(Path(output) / "protocol.json")
    configure_environment(protocol)
    directory = job_dir(output, job)
    attempt_dir = directory / f"attempt_{attempt}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    role = {"name": "actor"}
    with (attempt_dir / "worker.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            import litellm

            original_completion = litellm.completion
            call_number = 0

            def recorded_completion(*args, **kwargs):
                nonlocal call_number
                call_number += 1
                started = time.monotonic()
                entry = {"at": now(), "role": role["name"], "request": kwargs}
                path = attempt_dir / "llm_calls" / f"{call_number:04d}.json"
                try:
                    response = original_completion(*args, **kwargs)
                    entry["response"] = response.model_dump(mode="json")
                    return response
                except Exception as exc:
                    entry["error"] = str(exc)
                    raise
                finally:
                    entry["seconds"] = time.monotonic() - started
                    write(path, entry)

            litellm.completion = recorded_completion
            try:
                if job["stage"] in {1, 3}:
                    result = actor_job(output, protocol, job, attempt_dir)
                else:
                    result = proposal_job(output, protocol, job, attempt_dir, role)
                result = {**job, **result, "attempt": attempt, "completed_at": now(),
                          "appworld_policy_sha256": protocol["appworld_policy_sha256"]}
                write(directory / "result.json", result)
                return {"ok": True, "job_id": job["id"]}
            except Exception as exc:
                write(attempt_dir / "error.json", {"at": now(), "error": str(exc), "traceback": traceback.format_exc()})
                return {"ok": False, "job_id": job["id"], "error": str(exc)}
            finally:
                litellm.completion = original_completion


class Progress:
    def __init__(self, output, protocol):
        self.output, self.protocol = Path(output), protocol
        self.started = time.monotonic()
        self.totals = {}

    def emit(self, stage, completed, active, pending, errors, event="PROGRESS", state="running"):
        spec = self.protocol["stages"][stage - 1]
        by_arm = Counter(job.get("arm") for job in completed)
        elapsed = int(time.monotonic() - self.started)
        duration = f"{elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d}"
        snapshot = {"updated_at": now(), "state": state, "stage": stage, "stage_name": spec["name"],
                    "completed": len(completed), "total": spec["total_jobs"],
                    "percent": 100 * len(completed) / spec["total_jobs"], "running": len(active),
                    "pending": len(pending), "errors": len(errors), "by_arm": {
                        arm: {"completed": by_arm[arm], "total": spec.get("jobs_per_arm", {}).get(arm)}
                        for arm in ARMS if "jobs_per_arm" in spec
                    }, "stage_elapsed": duration, "controller_pid": os.getpid(), "stages": dict(self.totals)}
        self.totals[str(stage)] = {k: snapshot[k] for k in ("completed", "total", "running", "errors")}
        snapshot["stages"] = dict(self.totals)
        write(self.output / "status.json", snapshot)
        line = (f"[{snapshot['updated_at'][11:19]}] {event} stage={stage}/3 {spec['name']} "
                f"done={len(completed)}/{spec['total_jobs']} ({snapshot['percent']:.1f}%) "
                f"running={len(active)} pending={len(pending)} errors={len(errors)} elapsed={duration}")
        if snapshot["by_arm"]:
            line += " | " + " ".join(
                f"{arm}={counts['completed']}/{counts['total']}" for arm, counts in snapshot["by_arm"].items()
            )
        print(line, flush=True)
        with (self.output / "progress.log").open("a") as stream:
            stream.write(line + "\n")


def run_stage(output, protocol, stage, pool, progress):
    jobs = read(Path(output) / f"stage{stage}_queue.json")
    completed = [job for job in jobs if (job_dir(output, job) / "result.json").exists()]
    done_ids = {job["id"] for job in completed}
    pending = [job for job in jobs if job["id"] not in done_ids]
    active, errors = {}, []
    attempts = {job["id"]: len(list(job_dir(output, job).glob("attempt_*"))) for job in pending}
    progress.started = time.monotonic()
    progress.emit(stage, completed, active, pending, errors, "START")
    heartbeat = time.monotonic()
    while pending or active:
        if (Path(output) / "STOP").exists():
            progress.emit(stage, completed, active, pending, errors, "STOPPING", "stopping")
            raise InterruptedError("STOP requested; active attempts retained for resume")
        while pending and len(active) < 50:
            job = pending.pop(0)
            attempts[job["id"]] += 1
            future = pool.submit(worker, str(output), job, attempts[job["id"]])
            active[future] = job
        finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
        for future in finished:
            job = active.pop(future)
            try:
                result = future.result()
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            if result["ok"]:
                completed.append(job)
            else:
                message = {"at": now(), "stage": stage, "job": job, "attempt": attempts[job["id"]],
                           "error": result["error"]}
                with (Path(output) / "errors.jsonl").open("a") as stream:
                    stream.write(json.dumps(message, ensure_ascii=False) + "\n")
                line = f"ERROR stage={stage} job={job['id']} attempt={attempts[job['id']]} {result['error'][:300]}"
                print(line, flush=True)
                with (Path(output) / "progress.log").open("a") as stream:
                    stream.write(line + "\n")
                if attempts[job["id"]] < protocol["infrastructure_attempts"]:
                    pending.append(job)
                else:
                    errors.append(job)
                progress.emit(stage, completed, active, pending, errors, "ERROR")
        if time.monotonic() - heartbeat >= protocol["progress_logging"]["heartbeat_interval_seconds"]:
            progress.emit(stage, completed, active, pending, errors)
            heartbeat = time.monotonic()
    if errors or len(completed) != len(jobs):
        progress.emit(stage, completed, active, pending, errors, "BLOCKED", "error")
        raise RuntimeError(f"Stage {stage} has {len(errors)} unresolved jobs; next stage will not start")
    progress.emit(stage, completed, active, pending, errors, "DONE")
    summarize(output, stage)


def summarize(output, stage):
    output = Path(output)
    rows = [read(p) for p in (output / f"stage{stage}").glob("*/result.json")]
    write(output / f"stage{stage}_results.json", rows)
    if stage == 2:
        write(output / "stage2_summary.json", {
            arm: {"contexts": sum(r["arm"] == arm for r in rows),
                  "changed": sum(r["arm"] == arm and r["context_changed"] for r in rows)} for arm in ARMS
        })
        return
    grouped = {}
    for row in rows:
        key = row["task_id"] if stage == 1 else row["context_id"]
        grouped.setdefault(key, []).append(row)
    report = {key: {"task_id": items[0]["task_id"], "rollouts": len(items),
                    "mean_pct": sum(r["pct"] for r in items) / len(items),
                    "official_success_rate": sum(r["success"] for r in items) / len(items)}
              for key, items in grouped.items()}
    if stage == 3:
        baseline = read(output / "stage1_summary.json")
        for value in report.values():
            value["delta_pct"] = value["mean_pct"] - baseline[value["task_id"]]["mean_pct"]
    write(output / f"stage{stage}_summary.json", report)


def run(output):
    from filelock import FileLock

    output = Path(output).resolve()
    with FileLock(str(output / "controller.lock"), timeout=0):
        protocol = read(output / "protocol.json")
        configure_environment(protocol)
        write(output / "endpoint_verification_at_start.json", check_endpoint(protocol))
        for name, expected in read(output / "source_hashes.json").items():
            if digest((output / "source_snapshot" / name).read_bytes()) != expected:
                raise ValueError(f"Frozen source changed: {name}")
        write(output / "controller.json", {"pid": os.getpid(), "started_at": now(), "command": sys.argv})
        progress = Progress(output, protocol)
        try:
            with ProcessPoolExecutor(max_workers=50, mp_context=multiprocessing.get_context("spawn")) as pool:
                for stage in (1, 2, 3):
                    run_stage(output, protocol, stage, pool, progress)
            status = read(output / "status.json")
            write(output / "status.json", {**status, "state": "completed", "completed_at": now()})
            print(f"[{now()}] ALL DONE: {protocol['actor_trajectories_planned']} actor trajectories "
                  "and 500 independent proposal chains", flush=True)
        except BaseException as exc:
            status = read(output / "status.json")
            state = "stopped" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "error"
            write(output / "status.json", {**status, "state": state, "updated_at": now(), "error": str(exc)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "scripts/ace/configs/appworld_context_proposal_bottleneck.json")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if args.prepare:
        prepare(args.output, args.config)
        print(f"Prepared {args.output}")
    else:
        run(args.output)


if __name__ == "__main__":
    main()
