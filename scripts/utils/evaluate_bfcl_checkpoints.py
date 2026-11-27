# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Evaluate BFCL-only checkpoint 50 across ACE, PREMiSE and ps_st_pareto."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.ace import evaluate_test_checkpoints as ace  # noqa: E402

BENCHMARKS = ace.BENCHMARKS
PORTS = (26000, 27000, 30000, 31000, 51000, 52000, 53000, 54000, 56000)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_path(job: dict, slug: str, *, native: bool = False) -> Path:
    root = Path(job["source_run"] if native else job["evaluation_output"])
    if job["method"] == "ace":
        return Path(job["evaluation_output"]) / f"sequential_s{job['seed']}" / "checkpoint_050" / slug / "summary.json"
    return root / "test_evaluation" / f"cp050_{slug}.json"


def premise_identity(job: dict, slug: str) -> dict:
    return {
        "checkpoint": 50, "benchmark": slug, "task_ids": job["task_ids"][slug],
        "checkpoint_sha256": job["checkpoint_sha256"], "manifest_sha256": job["manifest_sha256"],
        "beam_rank": 1, "learning": False,
    }


def validate_result(job: dict, slug: str, summary: dict) -> list:
    ids = job["task_ids"][slug]
    if job["method"] == "ace":
        if (summary.get("checkpoint_sha256") != job["checkpoint_sha256"]
                or summary.get("manifest_sha256") != job["manifest_sha256"]
                or summary.get("task_ids") != ids
                or summary.get("stage") != 50 or summary.get("seed") != job["seed"]
                or summary.get("benchmark") != slug or summary.get("mode") != "sequential"
                or summary.get("evaluation_mode") is not True or summary.get("training_time") is not False):
            raise ValueError(f"Wrong frozen ACE evaluation: {job['name']} {slug}")
        records = summary.get("sessions", [])
        if len(records) != len(ids) or {row["task_id"] for row in records} != set(ids):
            raise ValueError("ACE evaluation returned incomplete or duplicate test IDs")
        return records
    if summary.get("identity") != premise_identity(job, slug):
        raise ValueError(f"Wrong frozen PREMiSE evaluation: {job['name']} {slug}")
    records = summary.get("results", [])
    if len(records) != len(ids) or any(row is not None and row.get("task_id") != ids[i]
                                      for i, row in enumerate(records)):
        raise ValueError("PREMiSE evaluation returned mismatching test IDs")
    if summary.get("complete") and any(row is None for row in records):
        raise ValueError("PREMiSE evaluation marked incomplete records as complete")
    return records


def copy_frozen(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"Frozen checkpoint differs: {path}")
    if not path.exists():
        path.write_bytes(raw)


def prepare_plan(manifests: list[Path], output: Path) -> dict:
    jobs = []
    for launch_path in manifests:
        launch_path = launch_path.resolve()
        launch = read_json(launch_path)
        method = launch.get("method", launch.get("agent"))
        if method not in ("ace", "premise", "ps_st_pareto"):
            raise ValueError(f"Unsupported method: {method}")
        split_path = Path(launch["source_manifest"]).resolve()
        if ace.digest(split_path) != launch["source_manifest_sha256"]:
            raise ValueError("Task split manifest changed after training launch")
        split = read_json(split_path)
        if set(split) != {f"{slug}_{seed}" for slug in BENCHMARKS for seed in (42, 43, 44)}:
            raise ValueError("Expected the nine benchmark/seed split keys")
        ace_jobs = ace.build_plan(launch) if method == "ace" else []
        rows = sorted((row for row in launch["jobs"] if row["mode"] == "sequential"), key=lambda row: row["seed"])
        if [row["seed"] for row in rows] != [42, 43, 44]:
            raise ValueError("Expected three sequential seed-42-44 runs per method")
        for row in rows:
            if len(row["task_order"]) != 130 or any(slug != "bfcl" for slug, _ in row["task_order"][:50]):
                raise ValueError("Checkpoint 50 must contain only the 50 BFCL training tasks")
            source_run = Path(row["output_dir"]) if method == "ace" else launch_path.parent / row["name"]
            name = f"{method}_sequential_s{row['seed']}" + ("" if method == "ace" else "_t3")
            evaluation_output = launch_path.parent / "test_evaluation" if method == "ace" else output / name
            if method == "ace":
                checkpoint = ace.archive_snapshots(row, evaluation_output, stages=(50,))[50]
            else:
                checkpoint = source_run / "checkpoints/checkpoint_050.json"
                saved = read_json(checkpoint)
                config = saved["config"]
                if (saved["cursor"] != 50 or config["order"] != row["task_order"]
                        or config["seed"] != row["seed"] or config["mode"] != "sequential"
                        or config["past_count"] != 3 or config.get("method", "premise") != method
                        or config["manifest_sha256"] != launch["source_manifest_sha256"]):
                    raise ValueError(f"Wrong BFCL checkpoint configuration: {checkpoint}")
                raw = checkpoint.read_bytes()
                copy_frozen(evaluation_output / "state.json", raw)
                copy_frozen(evaluation_output / "checkpoints/checkpoint_050.json", raw)
            ids = {}
            for slug in BENCHMARKS:
                section = split[f"{slug}_{row['seed']}"]
                expected = 29 if slug == "terminalbench2" else 50
                ids[slug] = section["test"]
                train = [task for name_slug, task in row["task_order"] if name_slug == slug]
                if (len(ids[slug]) != expected or len(set(ids[slug])) != expected
                        or set(train) != set(section["train"]) or set(train) & set(ids[slug])):
                    raise ValueError(f"Invalid training/test partition: {name} {slug}")
            job = {
                "name": name, "method": method, "seed": row["seed"], "checkpoint": 50,
                "source_run": str(source_run), "source_launch_manifest": str(launch_path),
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": ace.digest(checkpoint),
                "manifest_path": str(split_path), "manifest_sha256": launch["source_manifest_sha256"],
                "evaluation_output": str(evaluation_output), "task_ids": ids,
                "ace_jobs": [item for item in ace_jobs if item["seed"] == row["seed"]
                             and item["mode"] == "sequential" and item["stage"] == 50],
            }
            if method != "ace":
                for slug in BENCHMARKS:
                    native = result_path(job, slug, native=True)
                    if native.exists() and not result_path(job, slug).exists():
                        summary = read_json(native)
                        validate_result(job, slug, summary)
                        ace.save_json(result_path(job, slug), summary)
            jobs.append(job)
    if len(jobs) != 9 or len({job["name"] for job in jobs}) != 9:
        raise ValueError("Expected nine distinct ACE/PREMiSE/ps_st_pareto checkpoints")
    if len({job["manifest_sha256"] for job in jobs}) != 1:
        raise ValueError("All methods must use the same task split manifest")
    for job, port in zip(jobs, PORTS, strict=True):
        job["port_block_start"] = port
    return {"created_at": datetime.now(UTC).isoformat(), "jobs": jobs,
            "planned_evaluations": 27, "planned_test_tasks": 1161}


def publish_result(job: dict, slug: str, summary: dict) -> str:
    """Publish only complete results without replacing a live evaluator's file."""
    validate_result(job, slug, summary)
    if not summary.get("complete"):
        raise ValueError("Cannot publish a partial evaluation")
    native = result_path(job, slug, native=True)
    native.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=native.parent, prefix=".cp050_publish_", delete=False) as temporary:
        json.dump(summary, temporary, indent=2, ensure_ascii=False)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    try:
        try:
            os.link(temporary_path, native)
            return "published"
        except FileExistsError:
            existing = read_json(native)
            validate_result(job, slug, existing)
            return "native_complete" if existing.get("complete") else "native_evaluation_in_progress"
    finally:
        temporary_path.unlink(missing_ok=True)


def assert_checkpoint(job: dict) -> None:
    if ace.digest(Path(job["checkpoint_path"])) != job["checkpoint_sha256"]:
        raise ValueError("The source checkpoint changed")
    if ace.digest(Path(job["manifest_path"])) != job["manifest_sha256"]:
        raise ValueError("The task split manifest changed")
    if job["method"] != "ace":
        for relative in ("state.json", "checkpoints/checkpoint_050.json"):
            if ace.digest(Path(job["evaluation_output"]) / relative) != job["checkpoint_sha256"]:
                raise ValueError("Test evaluation modified the frozen training state")


def retryable_infrastructure_record(record: dict | None) -> bool:
    if not record or record.get("status") != "error" or record.get("context_error"):
        return False
    message = record.get("error", "")
    return any(marker in message for marker in (
        "all predefined address pools", "Pool overlaps with other one", "litellm.Timeout", "APITimeoutError",
        "is not accessible: TimeoutError(",
    )) or ("'error_source': 'benchmark'" in message and "'error': 'timed out'" in message)


def prepare_infrastructure_retry(job: dict, output: Path) -> dict:
    """Archive invalid cached results and reopen only known infrastructure failures."""
    if job["method"] == "ace":
        raise ValueError("ACE recovery uses retry_terminalbench2_test.py")
    assert_checkpoint(job)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    archive = output / "repairs" / stamp / job["name"]
    reopened = {}
    for slug in BENCHMARKS:
        path = result_path(job, slug)
        if not path.exists():
            continue
        summary = read_json(path)
        records = validate_result(job, slug, summary)
        indexes = [index for index, record in enumerate(records) if retryable_infrastructure_record(record)]
        if not indexes:
            continue
        native = result_path(job, slug, native=True)
        if native.exists():
            existing = read_json(native)
            validate_result(job, slug, existing)
            if existing.get("results") != summary["results"]:
                raise ValueError(f"Native results changed during recovery: {native}")
            source_state = read_json(Path(job["source_run"]) / "state.json")
            if source_state["cursor"] >= 130:
                raise ValueError("Live post-training evaluation may own the native results; defer recovery")
            copy_frozen(archive / f"native_cp050_{slug}.json", native.read_bytes())
            # The archive preserves the original. Removing this invalid cache
            # allows the completed corrected result to be published atomically.
            native.unlink()
        copy_frozen(archive / f"cp050_{slug}.json", path.read_bytes())
        task_ids = [records[index]["task_id"] for index in indexes]
        for index in indexes:
            records[index] = None
        ace.save_json(path, {"identity": summary["identity"], "results": records, "complete": False,
                             "infrastructure_retry": {"task_ids": task_ids, "original_archive": str(archive)}})
        reopened[slug] = task_ids
    return reopened


def worker(job: dict, output: Path, retry_infrastructure: bool = False) -> int:
    from exgentic.agents.premise.pipeline import PremisePipeline

    directory = output / "workers" / job["name"]
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".worker.lock"), timeout=0):
        assert_checkpoint(job)
        if retry_infrastructure:
            reopened = prepare_infrastructure_retry(job, output)
            print(f"REOPEN known infrastructure errors: {json.dumps(reopened)}", flush=True)
        pipeline = None
        if job["method"] != "ace":
            config = read_json(Path(job["checkpoint_path"]))["config"]
            pipeline = PremisePipeline(
                output=Path(job["evaluation_output"]), manifest=Path(job["manifest_path"]),
                benchmarks=config["benchmarks"], mode=config["mode"], seed=config["seed"],
                count=config["count"], past_count=config["past_count"], model=config["model"],
                base_url=os.environ["OPENAI_API_BASE"], replay_workers=4,
                replay_retries=config["replay_retries"], max_steps=config["max_steps"], method=job["method"],
            )
        failed = []
        published = {}
        for slug in BENCHMARKS:
            try:
                assert_checkpoint(job)
                print(f"START {job['name']} checkpoint=50 benchmark={slug}", flush=True)
                if job["method"] == "ace":
                    ace_job = next(item for item in job["ace_jobs"] if item["benchmark"] == slug)
                    summary = ace.run_job(ace_job, Path(job["evaluation_output"]), max_workers=4)
                else:
                    summary = pipeline.evaluate_test_checkpoints(
                        checkpoints=(50,), benchmarks=(slug,), allow_incomplete=True,
                    )[0]
                    published[slug] = publish_result(job, slug, summary)
                validate_result(job, slug, summary)
                assert_checkpoint(job)
                print(f"DONE {job['name']} checkpoint=50 benchmark={slug}", flush=True)
            except Exception as exc:
                failed.append({"benchmark": slug, "error": f"{type(exc).__name__}: {exc}"})
                traceback.print_exc()
            ace.save_json(directory / "status.json", {"complete": False, "failed": failed, "published": published})
        ace.save_json(directory / "status.json", {"complete": not failed, "failed": failed, "published": published})
        return int(bool(failed))


def progress(job: dict, slug: str) -> dict:
    path = result_path(job, slug)
    complete, score, errors = False, None, 0
    records = []
    if path.exists():
        summary = read_json(path)
        records = [row for row in validate_result(job, slug, summary) if row is not None]
        complete = job["method"] == "ace" or summary.get("complete", False)
        if complete:
            score = summary.get("mean_task_score", summary.get("mean_official_score"))
        errors = sum(row["status"] == "error" for row in records)
    elif job["method"] == "ace":
        sessions = path.parent / f"ace_test_sequential_s{job['seed']}_c50_{slug}" / "sessions"
        records = [read_json(item) for item in sessions.glob("*/results.json")]
        records = [item for item in records if str(item.get("task_id")) in job["task_ids"][slug]]
    return {"benchmark": slug, "complete": complete, "completed_tasks": len(records),
            "total_tasks": len(job["task_ids"][slug]), "score": score, "errors": errors, "result": str(path)}


def controller(plan: dict, output: Path, poll_seconds: float) -> int:
    with FileLock(str(output / ".controller.lock"), timeout=0):
        processes = {}
        for job in plan["jobs"]:
            groups = [progress(job, slug) for slug in BENCHMARKS]
            if all(group["complete"] for group in groups):
                continue
            log_path = output / "logs" / f"{job['name']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--output", str(output),
                       "--worker", job["name"]]
            env = os.environ.copy() | {"EXGENTIC_PORT_BLOCK_START": str(job["port_block_start"])}
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes[job["name"]] = process
            print(f"START {job['name']} pid={process.pid} port={job['port_block_start']} log={log_path}", flush=True)
        while True:
            rows = [{"name": job["name"], "method": job["method"], "seed": job["seed"],
                     "pid": processes[job["name"]].pid if job["name"] in processes else None,
                     "exit_code": processes[job["name"]].poll() if job["name"] in processes else 0,
                     "benchmarks": [progress(job, slug) for slug in BENCHMARKS]} for job in plan["jobs"]]
            groups = [group for row in rows for group in row["benchmarks"]]
            running = [name for name, process in processes.items() if process.poll() is None]
            done = sum(group["complete"] for group in groups)
            success = done == 27 and all(row["exit_code"] == 0 for row in rows)
            phase = "evaluating" if running else ("complete" if success else "evaluation_failed")
            ace.save_json(output / "status.json", {
                "updated_at": datetime.now(UTC).isoformat(), "phase": phase,
                "planned_evaluations": 27, "completed_evaluations": done,
                "planned_test_tasks": 1161, "completed_test_tasks": sum(group["completed_tasks"] for group in groups),
                "running_workers": running, "jobs": rows,
            })
            if not running:
                print(f"FINISHED {done}/27 evaluations", flush=True)
                return int(not success)
            time.sleep(min(poll_seconds, 60))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-manifest", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker")
    parser.add_argument("--retry-infrastructure-errors", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("Poll interval must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "evaluation_plan.json"
    if not plan_path.exists():
        if not args.launch_manifest or args.worker:
            parser.error("Provide three launch manifests to prepare the evaluation batch")
        plan = prepare_plan(args.launch_manifest, output)
        ace.save_json(plan_path, plan)
    else:
        plan = read_json(plan_path)
    if args.prepare_only:
        groups = [progress(job, slug) for job in plan["jobs"] for slug in BENCHMARKS]
        print(json.dumps({"checkpoints": len(plan["jobs"]), "evaluation_groups": len(groups),
                          "reused_groups": sum(group["complete"] for group in groups), "test_tasks": 1161}))
        return 0
    if args.worker:
        matches = [job for job in plan["jobs"] if job["name"] == args.worker]
        if len(matches) != 1:
            parser.error("Unknown checkpoint worker")
        return worker(matches[0], output, args.retry_infrastructure_errors)
    if args.retry_infrastructure_errors:
        parser.error("--retry-infrastructure-errors requires --worker")
    return controller(plan, output, args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
