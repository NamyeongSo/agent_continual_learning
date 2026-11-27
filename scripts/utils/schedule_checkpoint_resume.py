# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Resume an explicitly scoped CP queue after teammate PEACE training and final testing commit."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.utils import checkpoint_sources as sources  # noqa: E402
from scripts.utils.checkpoint_queue import now  # noqa: E402
from scripts.utils.process_identity import stable_identity  # noqa: E402
from scripts.utils.run_qwen_isolated_when_free import experiment_environment, write  # noqa: E402


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        if default is None:
            raise
        return default


def file_sha(path):
    return sources.digest(Path(path).read_bytes())


@contextmanager
def readonly_db(output):
    db = sqlite3.connect((Path(output) / "queue.sqlite3").as_uri() + "?mode=ro", uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def validate_sources(plan):
    specs = sources.streams(plan["queue_plan"]["roots"], sources.Reader())
    expected = {(method, benchmark, seed) for method in ("ace", "premise", "gepa", "skillopt")
                for benchmark in ("bfcl", "appworld", "terminalbench2") for seed in (42, 43, 44)}
    if len(specs) != 36 or {(s["method"], s["benchmark"], s["seed"]) for s in specs} != expected:
        raise ValueError("Expected exactly 36 original-policy streams")
    if {s["id"] for s in specs} != set(plan["queue_plan"]["selected_stream_ids"]):
        raise ValueError("Original-policy output paths changed")
    for spec in specs:
        if spec["method"] in {"ace", "premise"} and spec["playbook_edit_mode"] != "add_update_delete":
            raise ValueError("Only AUD ACE/PEACE streams may resume")
        if spec.get("appworld_policy_override") is not None or spec.get("appworld_policy_sha256") is not None:
            raise ValueError("A teammate-policy stream entered the original-policy schedule")
    if sources.digest(json.dumps(specs, sort_keys=True)) != plan["source_specs_sha256"]:
        raise ValueError("Scheduled source configuration changed")
    with readonly_db(plan["queue_output"]) as db:
        invalid = {r[0] for r in db.execute("SELECT stream FROM stream_invalidations")}
    if invalid.intersection(plan["queue_plan"]["selected_stream_ids"]):
        raise ValueError("A selected learning stream was invalidated")
    return specs


def prerequisites(plan):
    jobs = plan["prerequisites"]
    expected = {(mode, seed) for mode in ("add_only", "add_update_delete") for seed in (42, 43, 44)}
    if len(jobs) != 6 or {(j["variant"], j["seed"]) for j in jobs} != expected:
        raise ValueError("Expected all six teammate PEACE contexts")
    output = Path(plan["final_queue_output"])
    final = read(output / "final_test_queue_manifest.json")
    contexts = {c["source_output"]: c for c in final["contexts"]}
    rows = []
    with readonly_db(output) as db:
        for job in jobs:
            train = Path(job["train_dir"])
            marker = read(train / "training_complete.json", {})
            state = read(train / "state.json", {})
            training = (marker.get("completed_task_count") == marker.get("expected_task_count") == 50
                        and state.get("cursor") == 50)
            context = contexts.get(job["output"])
            done, errors, test_complete = 0, 0, False
            if context:
                frozen = read(context["frozen"])
                policy = frozen.get("appworld_policy_override")
                if (not isinstance(policy, str) or sources.digest(policy) != plan["teammate_policy_sha256"]
                        or frozen.get("appworld_policy_sha256") != plan["teammate_policy_sha256"]
                        or file_sha(context["frozen"]) != context["frozen_sha256"]
                        or frozen.get("temperature_all_roles") != 0.7):
                    raise ValueError("Teammate final context policy/configuration changed")
                tasks = db.execute("""SELECT t.*,c.sha FROM tasks t JOIN checkpoints c ON c.id=t.cp
                    WHERE c.stream=?""", (context["stream"],)).fetchall()
                if (len(tasks) != 50 or {r["task_id"] for r in tasks} != set(job["test_ids"])
                        or any(r["split"] != "test" or r["sha"] != sources.digest(frozen["playbook"]) for r in tasks)):
                    raise ValueError("Teammate final test tuples differ from the frozen context")
                for task in tasks:
                    result = json.loads(task["result"]) if task["result"] else {}
                    done += int(task["status"] == "completed" and result.get("score") in (0, 1)
                                and result.get("status") == "completed"
                                and result.get("score_metric") == "appworld_tgc_v1")
                    errors += int(task["status"] == "error")
                summary = read(Path(job["output"]) / "evaluation/summary.json", {})
                test_complete = (done == 50 and not errors and summary.get("complete") is True
                                 and summary.get("tasks") == 50 and summary.get("execution_errors") == 0)
            rows.append({"variant": job["variant"], "seed": job["seed"], "train_done": state.get("cursor", 0),
                         "training_complete": training, "test_done": done, "test_errors": errors,
                         "test_complete": test_complete})
        counts = dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status"))
    expected_contexts = {(method, mode, seed) for method in ("ace", "premise")
                         for mode in ("add_only", "add_update_delete") for seed in (42, 43, 44)}
    registered = {(c.get("method", "ace"), c["variant"], c["seed"]) for c in final["contexts"]}
    drained = len(final["contexts"]) == 12 and registered == expected_contexts and counts == {"completed": 600}
    return {"ready": all(r["training_complete"] and r["test_complete"] for r in rows) and drained,
            "contexts": rows, "final_queue_drained": drained, "final_task_counts": counts}


def model_ready(plan):
    for url in (plan["upstream_base_url"], plan["queue_plan"]["base_url"]):
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as response:
                if not any(row["id"] == "Qwen/Qwen3.6-35B-A3B" for row in json.load(response)["data"]):
                    return False
        except (OSError, ValueError, KeyError):
            return False
    return True


def alive(record):
    return (record.get("start_ticks") is not None
            and stable_identity(record["pid"]) == {key: record.get(key) for key in ("start_ticks", "boot_id")})


def publish_start(output, plan, started):
    if not plan.get("journal_id"):
        return
    base = plan.get("journal_url", "http://127.0.0.1:8878")
    path = base + "/api/experiments/" + plan["journal_id"]
    text = (f"CP 재개 예약 실행: {output}\n팀원 PEACE 6개 학습·최종 test와 공용 최종 test 600개 완료 확인. "
            f"기존 policy ACE/PEACE AUD 및 GEPA/SkillOpt 36개 스트림만 동시성100으로 재개. "
            f"완료 결과 재사용, 추가 학습 없음. PID {started['pid']}; endpoint {started['base_url']}.")

    def request(url, value=None, method=None, token=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Journal-Token"] = token
        data = json.dumps(value, ensure_ascii=False).encode() if value is not None else None
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method=method),
                                    timeout=10) as response:
            return json.load(response)

    try:
        record = request(path)
        if any(entry.get("text") == text for entry in record.get("log", [])):
            return
        token = request(base + "/api/info")["csrf_token"]
        request(path, {"revision": record["revision"], "status": "running",
                       "next_steps": "기존 policy 36개 스트림의 남은 CP train/val/test 평가를 공용100 큐에서 수행."},
                "PUT", token)
        request(path + "/notes", {"text": text}, "POST", token)
        write(output / "journal_started.json", {"at": now(), "journal_id": plan["journal_id"]})
    except (OSError, ValueError, KeyError) as exc:
        write(output / "journal_update_error.json", {"at": now(), "error": f"{type(exc).__name__}: {exc}"[:400]})


def start_queue(output, plan, evidence):
    queue = Path(plan["queue_output"])
    if not evidence["ready"]:
        raise ValueError("Training and final-test prerequisites are incomplete")
    validate_sources(plan)
    marker = read(queue / "pause_dispatch.json", {})
    if marker != plan["expected_pause_marker"]:
        raise ValueError("Queue pause owner changed; retaining the newer pause")
    current_plan = queue / "checkpoint_queue_manifest.json"
    launch = read(output / "launch.json", {})
    if not launch:
        if file_sha(current_plan) != plan["previous_queue_manifest_sha256"]:
            raise ValueError("Queue configuration changed after scheduling")
        # Check both existing locks without terminating any other controller.
        with FileLock(str(queue / "controller.lock"), timeout=0), FileLock(
            str(Path.home() / ".cache/exgentic/checkpoint_evaluation_100.lock"), timeout=0
        ):
            environment = experiment_environment(read(plan["private_environment"]))
            environment["OPENAI_API_BASE"] = environment["OPENAI_BASE_URL"] = plan["queue_plan"]["base_url"]
            environment.pop("EXGENTIC_PORT_BLOCK_START", None)
            check = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], env=environment,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20, check=False)
            if check.returncode:
                raise RuntimeError("The existing experiment Docker connection is unavailable")
            write(output / "release_evidence.json", evidence)
            write(current_plan, plan["queue_plan"])
        with (output / "checkpoint_controller.log").open("ab") as log:
            child = subprocess.Popen(plan["command"], cwd=plan["runtime"], env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        launch = {"pid": child.pid, "command": plan["command"], "started_at": now(), **stable_identity(child.pid)}
        write(output / "launch.json", launch)
    if not alive(launch):
        raise RuntimeError("Scheduled CP controller exited; dispatch remains paused")
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        controller = read(queue / "controller_process.json", {})
        heartbeat = read(queue / "heartbeat.json", {})
        if (controller.get("pid") == launch["pid"] and heartbeat.get("selected_stream_ids")
                == plan["queue_plan"]["selected_stream_ids"] and alive(launch)):
            break
        if not alive(launch):
            raise RuntimeError("Scheduled CP controller exited before its first heartbeat")
        time.sleep(1)
    else:
        raise RuntimeError("Scoped CP controller heartbeat was not confirmed")
    if read(queue / "pause_dispatch.json", {}) != plan["expected_pause_marker"]:
        raise ValueError("A new pause was requested during the handoff")
    if not prerequisites(plan)["ready"]:
        raise ValueError("Completion evidence changed during the handoff")
    (queue / "pause_dispatch.json").unlink()
    record = {**launch, "released_at": now(), "queue_output": str(queue), "workers": 100,
              "selected_streams": 36, "base_url": plan["queue_plan"]["base_url"]}
    write(output / "started.json", record)
    return record


def run(output, check_only=False):
    plan = read(output / "schedule.json")
    validate_sources(plan)
    if check_only:
        return prerequisites(plan)
    with FileLock(str(output / "scheduler.lock"), timeout=0):
        write(output / "scheduler_process.json", {"pid": os.getpid(), "started_at": now(),
                                                  **stable_identity(os.getpid())})
        while True:
            if (output / "started.json").exists():
                write(output / "state.json", {"phase": "started", "at": now(), **read(output / "started.json")})
                publish_start(output, plan, read(output / "started.json"))
                return read(output / "started.json")
            try:
                evidence = prerequisites(plan)
                state = {"phase": "waiting_prerequisites", "at": now(), **evidence}
                if evidence["ready"]:
                    if model_ready(plan):
                        started = start_queue(output, plan, evidence)
                        write(output / "state.json", {"phase": "started", "at": now(), **started})
                        publish_start(output, plan, started)
                        return started
                    state["phase"] = "waiting_model"
            except (OSError, ValueError, KeyError, RuntimeError, Timeout, subprocess.TimeoutExpired) as exc:
                state = {"phase": "waiting_recovery", "at": now(), "error": f"{type(exc).__name__}: {exc}"[:600]}
            write(output / "state.json", state)
            time.sleep(plan.get("poll_seconds", 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve(), args.check_only), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
