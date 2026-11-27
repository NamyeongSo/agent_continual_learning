# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Wait for the selected checkpoint queue, then run a frozen 128-request policy study."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.utils.policy_shift_metrics import digest, load_json, write_json  # noqa: E402
from scripts.utils.run_checkpoint_policy_shift import checkpoint_gate, readonly_queue  # noqa: E402


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def api(path, payload=None, method=None):
    base = "http://127.0.0.1:8878"
    if payload is None:
        with urllib.request.urlopen(base + path, timeout=30) as response:
            return json.load(response)
    token = api("/api/info")["csrf_token"]
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        method=method or "POST",
        headers={"Content-Type": "application/json", "X-Journal-Token": token},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def journal_update(plan, patch):
    eid = plan.get("journal_id")
    if eid:
        current = api("/api/experiments/" + eid)
        api("/api/experiments/" + eid, {**patch, "revision": current["revision"]}, "PUT")


def prepare(output, queue_output):
    output.mkdir(parents=True, exist_ok=False)
    queue_plan = load_json(queue_output / "checkpoint_queue_manifest.json")
    selected = queue_plan["selected_stream_ids"]
    with readonly_queue(queue_output) as db:
        specs = [
            json.loads(db.execute("SELECT spec FROM streams WHERE id=?", (sid,)).fetchone()[0]) for sid in selected
        ]
        points = {
            sid: [row[0] for row in db.execute("SELECT point FROM checkpoints WHERE stream=? ORDER BY point", (sid,))]
            for sid in selected
        }
    if len(specs) != 36 or {(s["benchmark"], s["method"], s["seed"]) for s in specs} != {
        (b, m, s)
        for b in ("bfcl", "appworld", "terminalbench2")
        for m in ("ace", "gepa", "skillopt", "premise")
        for s in (42, 43, 44)
    }:
        raise ValueError("Expected the current 36 original-policy streams")
    if any(s.get("appworld_policy_sha256") for s in specs):
        raise ValueError("The current study expects the original environment policy cohort")
    snapshot = output / "code_snapshot"
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("policy_shift_metrics.py", "run_checkpoint_policy_shift.py", "schedule_checkpoint_policy_shift.py"):
        target = snapshot / "scripts/utils" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts/utils" / name, target)
    parser_candidates = sorted(
        (Path.home() / ".exgentic/benchmarks/terminalbench2/venv/lib").glob(
            "python*/site-packages/harbor/agents/terminus_2/terminus_json_plain_parser.py"
        )
    )
    if len(parser_candidates) != 1:
        raise ValueError("Cannot identify the installed official Terminal-Bench action parser")
    parser = output / "frozen/terminus_json_plain_parser.py"
    parser.parent.mkdir(parents=True)
    shutil.copy2(parser_candidates[0], parser)
    write_json(
        output / "code_hashes.json",
        {str(p.relative_to(snapshot)): digest(p.read_bytes()) for p in snapshot.rglob("*") if p.is_file()},
    )
    plan = {
        "created_at": now(),
        "output": str(output),
        "queue_output": str(queue_output),
        "selected_stream_ids": selected,
        "minimum_checkpoint_points": points,
        "checkpoint_mode": "all",
        "workers": 128,
        "model": "openai/Qwen/Qwen3.6-35B-A3B",
        "base_url": queue_plan["base_url"],
        "temperature": 0.7,
        "enable_thinking": False,
        "input_tokens": 100000,
        "output_tokens": 2048,
        "actor_policy_version": "actor-turn-v1",
        "environment_execution": False,
        "terminal_parser_path": str(parser),
        "terminal_parser_sha256": digest(parser.read_bytes()),
        "history_bank": "Shared ACE earliest available CP train-evaluation acting inputs per benchmark/seed",
        "history_selection": "All valid recorded acting turns; tool selection and duplicate callbacks excluded",
        "reference": "Method-specific CP0 predictions on identical histories; separate CP0 repeat control",
        "metrics": {
            "bfcl": ["loose_accuracy", "strict_accuracy", "loose_f1", "strict_f1"],
            "appworld": ["loose_accuracy", "strict_accuracy", "loose_f1", "strict_f1"],
            "terminalbench2": ["accuracy"],
        },
        "aggregation": "Common successful-history intersection across all methods/CPs within benchmark/seed; "
        "mean histories within task, equal tasks within seed, equal seeds; sample seed SD",
        "budget_errors": "Recorded as missing agreement observations, never retried or filled with zero agreement",
        "scope": "Current original-policy isolated ACE/PEACE AUD and GEPA/SkillOpt; seeds42/43/44",
        "sampling_limitations": "temperature0.7 and no provider sampling seed; CP0 repeats estimate sampling variance",
        "missing_checkpoints": "Use observed checkpoints only; never substitute a missing final checkpoint",
        "command": [
            sys.executable,
            "-u",
            str(snapshot / "scripts/utils/run_checkpoint_policy_shift.py"),
            "--output",
            str(output),
        ],
    }
    record = api(
        "/api/experiments",
        {
            "title": "Qwen3.6 · ACE/GEPA/SkillOpt/PEACE · 3 benchmarks checkpoint policy shift · queue128",
            "purpose": "현재 checkpoint 평가 완료 후 동일 history에서 CP0와 확보된 모든 CP의 다음 행동 변화를 측정",
            "status": "planned",
            "method": "ACE AUD / GEPA / SkillOpt / PEACE AUD · conditional policy agreement",
            "benchmark": "BFCL, AppWorld, Terminal-Bench",
            "model": plan["model"],
            "seeds": "42,43,44",
            "run_path": str(output),
            "split": plan["history_bank"],
            "budget": "Global128 concurrent requests; temperature0.7; thinkingoff; input100k/output2048; "
            "CP0 repeat control; no environment execution or learning",
            "command": " ".join(plan["command"]),
            "summary": "Checkpoint queue의 선택된36개 stream 전부 공식 평가 완료 후 자동 시작 예약.",
            "limitations": plan["sampling_limitations"] + "; " + plan["missing_checkpoints"],
            "tags": ["policy-shift", "checkpoint", "qwen36", "isolated", "queue128"],
        },
    )
    plan["journal_id"] = record["id"]
    write_json(output / "plan.json", plan)
    write_json(output / "source_scope.json", specs)
    api(
        "/api/experiments/" + record["id"] + "/import",
        {"path": str(output / "plan.json"), "caption": "대기 gate,36개stream,동시성128,history/지표/샘플링 프로토콜"},
    )
    return plan


def run(output, check_only=False):
    plan = load_json(output / "plan.json")
    with (output / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                gate = checkpoint_gate(plan)
            except (OSError, ValueError, KeyError) as exc:
                gate = {"ready": False, "reason": str(exc)}
            status = {
                "phase": "ready" if gate["ready"] else "waiting_for_checkpoint_evaluation",
                "at": now(),
                "scheduler_pid": os.getpid(),
                "gate": gate,
                "workers": plan["workers"],
            }
            write_json(output / "schedule_status.json", status)
            if check_only:
                print(json.dumps(status, ensure_ascii=False, indent=2))
                return
            if gate["ready"]:
                break
            time.sleep(30)
        # Any later restart reuses the immutable data and completed probes.
        for relative, sha in load_json(output / "code_hashes.json").items():
            if digest((output / "code_snapshot" / relative).read_bytes()) != sha:
                raise ValueError("Frozen policy-shift runtime changed")
        with (output / "runner.log").open("a", buffering=1) as log:
            process = subprocess.Popen(
                plan["command"],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write_json(output / "runner_process.json", {"pid": process.pid, "at": now(), "command": plan["command"]})
            journal_update(
                plan,
                {
                    "status": "running",
                    "started_on": now()[:10],
                    "summary": "Checkpoint 평가 완료 gate 통과. Policy shift 실행 시작, global concurrency128.",
                },
            )
            while process.poll() is None:
                active = load_json(output / "status.json") if (output / "status.json").exists() else {}
                write_json(
                    output / "schedule_status.json",
                    {
                        **status,
                        "phase": "running_policy_shift",
                        "at": now(),
                        "runner_pid": process.pid,
                        "progress": active,
                    },
                )
                time.sleep(10)
        analysis = output / "analysis/summary.json"
        if process.returncode != 0 or not analysis.exists():
            journal_update(
                plan,
                {
                    "status": "blocked",
                    "summary": f"Policy shift runner failed (exit={process.returncode}); runner.log 참조.",
                },
            )
            write_json(
                output / "schedule_status.json",
                {**status, "phase": "failed", "at": now(), "exit_code": process.returncode},
            )
            return
        summary = load_json(analysis)
        journal_update(
            plan,
            {
                "status": "completed",
                "summary": f"Policy shift 완료: {summary['counts']}; "
                f"공통 history {summary['common_histories']}개. task/seed 균등 집계 및 seed SD 기록.",
            },
        )
        for path in [
            analysis,
            output / "analysis/checkpoint_metrics.csv",
            output / "analysis/coverage.json",
            *sorted((output / "analysis").glob("*.png")),
        ]:
            api(
                "/api/experiments/" + plan["journal_id"] + "/import",
                {
                    "path": str(path),
                    "caption": "동일history CP0대비 policy agreement/shift; BFCL·AppWorld F1/accuracy, TB accuracy",
                },
            )
        write_json(output / "schedule_status.json", {**status, "phase": summary["status"], "at": now(), "exit_code": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.queue_output is None:
            parser.error("--prepare requires --queue-output")
        prepare(args.output.resolve(), args.queue_output.resolve())
    run(args.output.resolve(), args.check_only)


if __name__ == "__main__":
    main()
