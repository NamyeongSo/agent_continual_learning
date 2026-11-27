# SPDX-License-Identifier: Apache-2.0
"""Replay only the 15 shutdown/answer failures with two conflicting rules removed."""

import csv
import json
import multiprocessing
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from filelock import FileLock

OUTPUT = Path(__file__).resolve().parent
SOURCE = OUTPUT.parents[1] / "thinking" / "accumulation"
SNAPSHOT = SOURCE / "source_snapshot"
sys.path.insert(0, str(SNAPSHOT))
sys.path.insert(0, str(SNAPSHOT / "src"))

from scripts.ace import run_reflection_accumulation as accumulation  # noqa: E402

base = accumulation.base
RULES = ("misc-00056", "misc-00086")
CONDITION = "s42_without_gt_best"
WORKERS = 15


def target_failure(row):
    if row["success"] or row.get("budget_failure"):
        return None
    path = Path(row["session_dir"]) / "benchmark" / "scores.json"
    failures = base.read(path)["test_tracker"]["failures"]
    if len(failures) != 1 or "answers match" not in failures[0]["requirement"].lower():
        return None
    trace = failures[0]["trace"]
    if not re.search(r"==\s*(['\"])null\1", trace):
        return None
    return "missing_completion" if "<<not_given>>" in trace else "unrequested_answer"


def prepare():
    for path, digest in base.read(SOURCE / "source_hashes.json").items():
        if base.digest((SNAPSHOT / path).read_bytes()) != digest:
            raise ValueError(f"Frozen accumulation source changed: {path}")
    protocol = base.read(SOURCE / "seed42" / "protocol.json")
    policy = protocol["benchmark_kwargs"]["policy_override"]
    assert base.digest(policy) == protocol["appworld_policy_sha256"]
    checkpoint = SOURCE / "conditions" / CONDITION / "checkpoints" / "050.txt"
    context = checkpoint.read_text()
    original_digest = base.digest(context)
    assert original_digest == base.read(checkpoint.with_suffix(".json"))["context_sha256"]
    lines = context.splitlines(keepends=True)
    for rule in RULES:
        assert sum(line.startswith(f"[{rule}]") for line in lines) == 1
    edited = "".join(line for line in lines if not line.startswith(tuple(f"[{rule}]" for rule in RULES)))
    edited_path = OUTPUT / "c50_without_conflicting_completion_rules.txt"
    edited_path.write_text(edited)
    edited_digest = base.digest(edited)
    original = [r for r in base.read(SOURCE / "seed42" / "train_final_results.json")
                if r["condition_id"] == CONDITION]
    train = protocol["task_orders"]["42"]["train"]
    assert len(original) == 50 and {r["task_id"] for r in original} == set(train)
    assert all(r["initial_context_sha256"] == original_digest and type(r["success"]) is bool for r in original)
    jobs, cohort = [], []
    for row in original:
        category = target_failure(row)
        if category is None:
            continue
        job = {**{key: row[key] for key in ("id", "stage", "task_id", "source_id", "context_id", "source_repeat",
                                           "evaluation_repeat", "arm", "seed", "checkpoint_step")},
               "condition_id": f"{CONDITION}_completion_rule_ablation", "phase": "completion_rule_ablation"}
        jobs.append(job)
        cohort.append({"task_id": row["task_id"], "category": category, "original_pct": row["pct"],
                       "original_tgc": row["success"], "original_turn_cap": bool(row.get("turn_limit_reached"))})
        base.write(OUTPUT / "stage2" / job["id"] / "result.json", {
            "task_id": job["task_id"], "source_id": job["source_id"],
            "updated_context_path": str(edited_path), "updated_context_sha256": edited_digest,
        })
    assert len(jobs) == 15
    assert sum(r["category"] == "unrequested_answer" for r in cohort) == 11
    assert sum(r["category"] == "missing_completion" for r in cohort) == 4
    protocol["stages"] = [{}, {}, {"name": "completion_rule_ablation", "total_jobs": len(jobs), "max_workers": WORKERS}]
    base.write(OUTPUT / "protocol.json", protocol)
    base.write(OUTPUT / "stage3_queue.json", jobs)
    shutil.copy2(Path(protocol["source_reflection_run"]) / "stage1_summary.json", OUTPUT / "stage1_summary.json")
    base.write(OUTPUT / "cohort.json", {
        "tasks": cohort, "original_successes_full_train": sum(r["success"] for r in original),
        "checkpoint": str(checkpoint), "original_context_sha256": original_digest,
        "edited_context_sha256": edited_digest, "removed_rules": RULES,
    })


def report(protocol, jobs):
    cohort = base.read(OUTPUT / "cohort.json")
    outcomes = {job["task_id"]: accumulation.validate_evaluation(OUTPUT, job, protocol) for job in jobs}
    rows = []
    for old in cohort["tasks"]:
        row = outcomes[old["task_id"]]
        score_path = Path(row["session_dir"]) / "benchmark" / "scores.json"
        failures = base.read(score_path)["test_tracker"]["failures"] if score_path.is_file() else []
        assert type(row["success"]) is bool
        rows.append({**old, "rerun_pct": row["pct"], "rerun_tgc": row["success"],
                     "rerun_turn_cap": bool(row.get("turn_limit_reached")),
                     "rerun_answer_test_failed": any("answers match" in x["requirement"].lower() for x in failures),
                     "rerun_num_failed_tests": len(failures) if score_path.is_file() else None})
    summary = {}
    for category in ("all", "unrequested_answer", "missing_completion"):
        group = rows if category == "all" else [r for r in rows if r["category"] == category]
        summary[category] = {
            "tasks": len(group), "original_pct": mean(r["original_pct"] for r in group),
            "rerun_pct": mean(r["rerun_pct"] for r in group),
            "rerun_successes": sum(r["rerun_tgc"] for r in group),
            "rerun_answer_test_failures": sum(r["rerun_answer_test_failed"] for r in group),
            "rerun_turn_caps": sum(r["rerun_turn_cap"] for r in group),
        }
    summary["mixed_train_rate"] = (cohort["original_successes_full_train"] + summary["all"]["rerun_successes"]) / 50
    base.write(OUTPUT / "summary.json", summary)
    with (OUTPUT / "task_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


if __name__ == "__main__":
    import exgentic

    assert Path(exgentic.__file__).resolve().is_relative_to(SNAPSHOT / "src")
    with FileLock(str(OUTPUT / "controller.lock"), timeout=0):
        if not (OUTPUT / "cohort.json").exists():
            prepare()
        protocol = base.read(OUTPUT / "protocol.json")
        jobs = base.read(OUTPUT / "stage3_queue.json")
        base.write(OUTPUT / "endpoint_verification.json", base.check_endpoint(protocol))
        accumulation.journal(OUTPUT, "running", "종료 규칙 두 개를 뺀 C50으로 문제 Task 15개를 재평가합니다.")
        try:
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
                base.run_stage(OUTPUT, protocol, 3, pool, base.Progress(OUTPUT, protocol))
            result = report(protocol, jobs)
        except BaseException as exc:
            accumulation.journal(OUTPUT, "blocked", f"종료 규칙 재평가 중단. {exc}")
            raise
        status = base.read(OUTPUT / "status.json")
        base.write(OUTPUT / "status.json", {**status, "state": "completed", "completed_at": base.now()})
        accumulation.journal(OUTPUT, "completed", f"15개 중 {result['all']['rerun_successes']}개 공식 성공.")
        print(json.dumps(result, indent=2), flush=True)
